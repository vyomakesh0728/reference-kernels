#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA with bf16 KV — tiled V accumulator for tl.dot.

Key insight: Use SEPARATE accumulators for each V tile.
acc0 for V[0:64], acc1 for V[64:128], ..., acc7 for V[448:512]

Then each V tile uses: acc_i += tl.dot(P, V_tile_i)
No dynamic indexing needed!

Pack 16 heads into M dimension for proper MFMA usage.
"""

import torch
from task import input_t, output_t

try:
    import triton
    import triton.language as tl
    TRITON_OK = True
except ImportError:
    TRITON_OK = False

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
LOG2E = 1.4426950408889634

_cache = {}

if TRITON_OK:

    @triton.jit
    def _mla_bf16_tiled_kernel(
        Q,       # (bs, 16, 576) bf16
        KV,      # (total_kv, 576) bf16
        Out,     # (bs, 16, 512) bf16
        kv_indptr,
        stride_qb, stride_qh, stride_qd,
        stride_kt, stride_kd,
        stride_ob, stride_oh, stride_od,
        SM_SCALE_LOG2E: tl.constexpr,
        QK_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEADS: tl.constexpr,
    ):
        """MLA decode with tiled V accumulators. One block per batch."""
        bid = tl.program_id(0)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        offs_h = tl.arange(0, HEADS)

        # Softmax state
        m_i = tl.full((HEADS,), value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros((HEADS,), dtype=tl.float32)

        # 8 separate V accumulators of (HEADS, BLOCK_D) each
        # 512 / 64 = 8 tiles
        acc0 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc1 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc2 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc3 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc4 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc5 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc6 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)
        acc7 = tl.zeros((HEADS, BLOCK_D), dtype=tl.float32)

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # QK: tiled dot product
            qk = tl.zeros((HEADS, BLOCK_N), dtype=tl.float32)

            # 576 / 64 = 9 tiles
            for d_tile in tl.static_range(9):
                d_start = d_tile * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                q_ptrs = Q + bid * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
                q_tile = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)

                k_ptrs = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + offs_d[None, :] * stride_kd
                k_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

                qk += tl.dot(q_tile, tl.trans(k_tile))

            # Softmax
            qk = tl.where(mask_n[None, :], qk, -float("inf"))
            qk = qk * SM_SCALE_LOG2E

            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])
            p = tl.where(mask_n[None, :], p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)

            # Rescale all V accumulators
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc2 = acc2 * alpha[:, None]
            acc3 = acc3 * alpha[:, None]
            acc4 = acc4 * alpha[:, None]
            acc5 = acc5 * alpha[:, None]
            acc6 = acc6 * alpha[:, None]
            acc7 = acc7 * alpha[:, None]

            # Cast P for dot
            p_bf16 = p.to(tl.bfloat16)

            # V tile 0: dims [0:64]
            v_ptrs0 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + tl.arange(0, BLOCK_D)[None, :] * stride_kd
            v0 = tl.load(v_ptrs0, mask=mask_n[:, None], other=0.0)
            acc0 += tl.dot(p_bf16, v0)

            # V tile 1: dims [64:128]
            v_ptrs1 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (64 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v1 = tl.load(v_ptrs1, mask=mask_n[:, None], other=0.0)
            acc1 += tl.dot(p_bf16, v1)

            # V tile 2: dims [128:192]
            v_ptrs2 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (128 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v2 = tl.load(v_ptrs2, mask=mask_n[:, None], other=0.0)
            acc2 += tl.dot(p_bf16, v2)

            # V tile 3: dims [192:256]
            v_ptrs3 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (192 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v3 = tl.load(v_ptrs3, mask=mask_n[:, None], other=0.0)
            acc3 += tl.dot(p_bf16, v3)

            # V tile 4: dims [256:320]
            v_ptrs4 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (256 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v4 = tl.load(v_ptrs4, mask=mask_n[:, None], other=0.0)
            acc4 += tl.dot(p_bf16, v4)

            # V tile 5: dims [320:384]
            v_ptrs5 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (320 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v5 = tl.load(v_ptrs5, mask=mask_n[:, None], other=0.0)
            acc5 += tl.dot(p_bf16, v5)

            # V tile 6: dims [384:448]
            v_ptrs6 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (384 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v6 = tl.load(v_ptrs6, mask=mask_n[:, None], other=0.0)
            acc6 += tl.dot(p_bf16, v6)

            # V tile 7: dims [448:512]
            v_ptrs7 = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + (448 + tl.arange(0, BLOCK_D))[None, :] * stride_kd
            v7 = tl.load(v_ptrs7, mask=mask_n[:, None], other=0.0)
            acc7 += tl.dot(p_bf16, v7)

            m_i = m_new

        # Normalize
        inv_l = 1.0 / l_i
        acc0 = acc0 * inv_l[:, None]
        acc1 = acc1 * inv_l[:, None]
        acc2 = acc2 * inv_l[:, None]
        acc3 = acc3 * inv_l[:, None]
        acc4 = acc4 * inv_l[:, None]
        acc5 = acc5 * inv_l[:, None]
        acc6 = acc6 * inv_l[:, None]
        acc7 = acc7 * inv_l[:, None]

        # Store: (HEADS, BLOCK_D) -> need to write each head's row
        offs_d = tl.arange(0, BLOCK_D)
        out_base = Out + bid * stride_ob

        # Store all heads x all V tiles
        # Use 2D store: (HEADS, BLOCK_D) with proper strides
        out_ptrs0 = out_base + offs_h[:, None] * stride_oh + (0 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs0, acc0.to(tl.bfloat16))

        out_ptrs1 = out_base + offs_h[:, None] * stride_oh + (64 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs1, acc1.to(tl.bfloat16))

        out_ptrs2 = out_base + offs_h[:, None] * stride_oh + (128 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs2, acc2.to(tl.bfloat16))

        out_ptrs3 = out_base + offs_h[:, None] * stride_oh + (192 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs3, acc3.to(tl.bfloat16))

        out_ptrs4 = out_base + offs_h[:, None] * stride_oh + (256 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs4, acc4.to(tl.bfloat16))

        out_ptrs5 = out_base + offs_h[:, None] * stride_oh + (320 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs5, acc5.to(tl.bfloat16))

        out_ptrs6 = out_base + offs_h[:, None] * stride_oh + (384 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs6, acc6.to(tl.bfloat16))

        out_ptrs7 = out_base + offs_h[:, None] * stride_oh + (448 + offs_d)[None, :] * stride_od
        tl.store(out_ptrs7, acc7.to(tl.bfloat16))


BLOCK_D = 64


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])

    kv_bf16 = kv_data["bf16"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
    kv_flat = kv_bf16.view(kv_bf16.shape[0], kv_bf16.shape[-1])

    out_key = ("bf16_out", bs)
    if out_key not in _cache:
        _cache[out_key] = torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
    out = _cache[out_key]

    grid = (bs,)
    _mla_bf16_tiled_kernel[grid](
        q_reshaped, kv_flat, out, kv_indptr,
        q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),
        kv_flat.stride(0), kv_flat.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        SM_SCALE_LOG2E=SM_SCALE * LOG2E,
        QK_DIM=QK_HEAD_DIM,
        V_DIM=V_HEAD_DIM,
        BLOCK_N=16,
        BLOCK_D=BLOCK_D,
        HEADS=NUM_HEADS,
    )
    return out
