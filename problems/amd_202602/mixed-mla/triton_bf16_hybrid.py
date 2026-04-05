#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Hybrid: bf16 Triton for small shapes, aiter fp8 for large.

Insight: small shapes (bs=4, kv=1k) are dispatch-limited, not compute-limited.
bf16 Triton avoids quant overhead and aiter's Python dispatch cost.
Large shapes (bs>=32, kv=8k) need MFMA — use aiter's assembly kernels.

This kernel packs all 16 heads into M dimension for tl.dot:
Q: (16, 576) @ K: (576, BLOCK_N) -> QK: (16, BLOCK_N)
P: (16, BLOCK_N) @ V: (BLOCK_N, 512) -> out: (16, 512)
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
    def _mla_bf16_heads_kernel(
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
        NUM_QK_TILES: tl.constexpr,
        NUM_V_TILES: tl.constexpr,
        HEADS: tl.constexpr,
    ):
        """MLA with all heads packed in M dim for tl.dot."""
        bid = tl.program_id(0)  # batch index

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        offs_h = tl.arange(0, HEADS)  # (16,)

        # Online softmax state per head
        m_i = tl.full((HEADS,), value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros((HEADS,), dtype=tl.float32)

        # V accumulator: (HEADS, V_DIM) — too large for registers if V_DIM=512
        # Instead, accumulate V in tiles and store incrementally
        # Actually 16×512 = 8192 floats = 32KB — feasible in registers on MI355X (512 VGPRs)
        acc = tl.zeros((HEADS, V_DIM), dtype=tl.float32)

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # QK: (HEADS, QK_DIM) @ (QK_DIM, BLOCK_N) via K-dim tiling
            qk = tl.zeros((HEADS, BLOCK_N), dtype=tl.float32)

            for d_tile in tl.static_range(NUM_QK_TILES):
                d_start = d_tile * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM

                # Q: (HEADS, BLOCK_D)
                q_ptrs = Q + bid * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
                q_tile = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)

                # K: (BLOCK_N, BLOCK_D)
                k_ptrs = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + offs_d[None, :] * stride_kd
                k_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

                # (HEADS, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (HEADS, BLOCK_N)
                qk += tl.dot(q_tile, tl.trans(k_tile))

            # Softmax
            qk = tl.where(mask_n[None, :], qk, -float("inf"))
            qk = qk * SM_SCALE_LOG2E

            m_ij = tl.max(qk, axis=1)  # (HEADS,)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])
            p = tl.where(mask_n[None, :], p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # V: (HEADS, BLOCK_N) @ (BLOCK_N, V_DIM) via V-dim tiling
            for v_tile in tl.static_range(NUM_V_TILES):
                v_start = v_tile * BLOCK_D
                v_offs = v_start + tl.arange(0, BLOCK_D)
                v_mask = v_offs < V_DIM

                v_ptrs = KV + (kv_start + n_start + offs_n[:, None]) * stride_kt + v_offs[None, :] * stride_kd
                v_tile_data = tl.load(v_ptrs, mask=mask_n[:, None] & v_mask[None, :], other=0.0)

                # (HEADS, BLOCK_N) @ (BLOCK_N, BLOCK_D) -> (HEADS, BLOCK_D)
                p_bf16 = p.to(tl.bfloat16)
                pv = tl.dot(p_bf16, v_tile_data)  # (HEADS, BLOCK_D) f32

                # Write into acc[:, v_start:v_start+BLOCK_D]
                # Can't do dynamic slicing — need a different approach
                # Use tl.where with broadcast mask
                for j in tl.static_range(BLOCK_D):
                    if v_start + j < V_DIM:
                        acc_col = tl.sum(acc * tl.where(tl.arange(0, V_DIM) == v_start + j, 1.0, 0.0)[None, :], axis=1)
                        pv_col = tl.sum(pv * tl.where(tl.arange(0, BLOCK_D) == j, 1.0, 0.0)[None, :], axis=1)
                        acc = tl.where(
                            (tl.arange(0, V_DIM) == v_start + j)[None, :],
                            (acc_col + pv_col)[:, None],
                            acc,
                        )

            m_i = m_new

        # Normalize
        acc = acc / l_i[:, None]

        # Store
        offs_v = tl.arange(0, V_DIM)
        for h in range(HEADS):
            out_ptrs = Out + bid * stride_ob + h * stride_oh + offs_v * stride_od
            tl.store(out_ptrs, acc[h, :].to(tl.bfloat16))


# aiter fallback
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8


def _get_config(bs, kvl):
    if kvl <= 1024:
        if bs <= 32:
            return (8, False, 2, True)
        if bs <= 64:
            return (4, False, 2, True)
        return (4, False, 2, True)
    return (32, True, 1, False)


def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = (bs, kvl, ns, qd, ps, fm)
    if key in _cache:
        return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(
        bs, 1, NUM_HEADS, qd, kvd,
        is_sparse=False, fast_mode=fm, num_kv_splits=ns, intra_batch_mode=True,
    )
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(
        qo, kvi, kl, NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        wm, ws, wi, ri, rf, rp,
        page_size=ps, kv_granularity=max(ps, 16), max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=fm, max_split_per_batch=ns, intra_batch_mode=True,
        dtype_q=qd, dtype_kv=kvd,
    )
    e = {
        "meta": {
            "work_meta_data": wm, "work_indptr": wi, "work_info_set": ws,
            "reduce_indptr": ri, "reduce_final_map": rf, "reduce_partial_map": rp,
        },
        "kl": kl, "ki": ki,
        "out": torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev),
    }
    _cache[key] = e
    return e


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    # Use aiter for everything — Triton bf16 is too slow without proper MFMA
    ns, use_a8w8, ps, fm = _get_config(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if use_a8w8:
        bkey = ("dq", q.numel())
        if bkey not in _cache:
            _cache[bkey] = (
                torch.empty_like(q, dtype=FP8_DTYPE),
                torch.empty(1, dtype=torch.float32, device=q.device),
            )
        qi, qs = _cache[bkey]
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
        qs = None

    c = _get_or_build(
        bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm
    )
    mla_decode_fwd(
        qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"], 1,
        page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0,
        num_kv_splits=ns, q_scale=qs, kv_scale=kv_scale, intra_batch_mode=True,
        **c["meta"],
    )
    return c["out"]
