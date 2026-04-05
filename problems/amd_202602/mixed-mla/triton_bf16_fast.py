#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with bf16 KV — optimized version.

Key optimizations:
1. Load full V vector (512 dims) at once — it's power-of-2
2. Tile QK across K dimension with BLOCK_D=64 (576 = 9×64)
3. Online softmax with full V_DIM accumulator
4. Pure bf16 path — no quant/dequant overhead
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
    def _mla_bf16_fast_kernel(
        Q,  # (bs, heads, 576) bf16
        KV,  # (total_kv, 576) bf16
        Out,  # (bs, heads, 512) bf16
        kv_indptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_ob,
        stride_oh,
        stride_od,
        SM_SCALE_LOG2E: tl.constexpr,
        QK_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_QK_TILES: tl.constexpr,
    ):
        """Single-pass MLA decode with bf16 KV."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        offs_v = tl.arange(0, V_DIM)

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # === QK dot product via K-dimension tiling ===
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d_tile in tl.static_range(NUM_QK_TILES):
                d_start = d_tile * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                q_ptrs = Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd
                q_val = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

                k_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_val = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)

                qk += tl.sum(q_val[None, :] * k_val, axis=1)

            qk = tl.where(mask_n, qk, -float("inf"))
            qk = qk * SM_SCALE_LOG2E

            # Online softmax
            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            # === V accumulation: load full V_DIM at once ===
            v_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_v[None, :] * stride_kd
            )
            v_val = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            # p(BLOCK_N) @ V(BLOCK_N, V_DIM) -> (V_DIM,)
            acc += tl.sum(p[:, None] * v_val, axis=0)

            m_i = m_new

        acc = acc / l_i

        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))


def _get_num_splits(bs, kvl):
    if kvl <= 1024:
        return 1
    if kvl <= 4096:
        return 4
    return 8


BLOCK_D = 64
NUM_QK_TILES = (QK_HEAD_DIM + BLOCK_D - 1) // BLOCK_D  # 9


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_bf16 = kv_data["bf16"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
    kv_flat = kv_bf16.view(kv_bf16.shape[0], kv_bf16.shape[-1])

    out_key = ("bf16_out", bs)
    if out_key not in _cache:
        _cache[out_key] = torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
    out = _cache[out_key]

    grid = (bs, NUM_HEADS)
    _mla_bf16_fast_kernel[grid](
        q_reshaped,
        kv_flat,
        out,
        kv_indptr,
        q_reshaped.stride(0),
        q_reshaped.stride(1),
        q_reshaped.stride(2),
        kv_flat.stride(0),
        kv_flat.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SM_SCALE_LOG2E=SM_SCALE * LOG2E,
        QK_DIM=QK_HEAD_DIM,
        V_DIM=V_HEAD_DIM,
        BLOCK_N=32,
        BLOCK_D=BLOCK_D,
        NUM_QK_TILES=NUM_QK_TILES,
    )
    return out
