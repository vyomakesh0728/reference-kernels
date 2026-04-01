#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with proper power-of-2 tiling and constexpr parameters.

Key fixes:
- QK_DIM=576 handled via BLOCK_D=64 tiling
- SM_SCALE/LOG2E passed as constexpr parameters
- V_DIM=512 is already power of 2

Strategy:
- Small/medium shapes: use aiter (highly optimized)
- Large shapes (bs>=128, kv>=8192): try Triton single-pass
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
    def _mla_decode_kernel(
        Q,
        KV,
        Out,
        kv_indptr,
        kv_scale,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_ob,
        stride_oh,
        stride_od,
        SM_SCALE_C: tl.constexpr,
        LOG2E_C: tl.constexpr,
        QK_DIM_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_D_BLOCKS: tl.constexpr,
        V_DIM: tl.constexpr,
    ):
        """Single-pass MLA decode. Handles non-power-of-2 QK_DIM via tiling."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start
        scale = tl.load(kv_scale)

        m_i = -float("inf")
        l_i = 0.0

        offs_v = tl.arange(0, V_DIM)
        acc = tl.zeros((V_DIM,), dtype=tl.float32)

        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for d_block in tl.static_range(NUM_D_BLOCKS):
                d_start = d_block * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                mask_d = offs_d < QK_DIM_C

                q_ptr = Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd
                q_val = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

                k_ptrs = (
                    KV
                    + (kv_start + n_start + offs_n[:, None]) * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_val = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                )
                k_f32 = k_val.to(tl.float32) * scale

                qk += tl.sum(q_val[None, :] * k_f32, axis=1)

            qk = tl.where(mask_n, qk, -float("inf"))
            qk = qk * SM_SCALE_C * LOG2E_C

            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            v_ptrs = (
                KV
                + (kv_start + n_start + offs_n[:, None]) * stride_kt
                + offs_v[None, :] * stride_kd
            )
            v_val = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            v_f32 = v_val.to(tl.float32) * scale

            acc += tl.sum(p[:, None] * v_f32, axis=0)
            m_i = m_new

        acc = acc / l_i

        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))


def _triton_mla(q, kv, kv_scale, kv_indptr, out):
    bs = q.shape[0]
    grid = (bs, NUM_HEADS)

    BLOCK_D = 64
    NUM_D_BLOCKS = (QK_HEAD_DIM + BLOCK_D - 1) // BLOCK_D

    _mla_decode_kernel[grid](
        q,
        kv,
        out,
        kv_indptr,
        kv_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SM_SCALE_C=SM_SCALE,
        LOG2E_C=LOG2E,
        QK_DIM_C=QK_HEAD_DIM,
        BLOCK_N=64,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
        V_DIM=V_HEAD_DIM,
    )
    return out


from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8


def _get_aiter_config(bs, kvl):
    if kvl <= 1024:
        if bs <= 32:
            return (8, False, 2, True)
        if bs <= 64:
            return (4, False, 2, True)
        return (4, False, 2, True)
    else:
        if bs <= 4:
            return (32, False, 2, True)
        if bs <= 32:
            return (8, True, 1, False)
        if bs <= 64:
            return (8, True, 1, False)
        return (16, True, 1, False)


def _get_or_build_aiter(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = ("aiter", bs, kvl, ns, qd, ps, fm)
    if key in _cache:
        return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(
        bs,
        1,
        NUM_HEADS,
        qd,
        kvd,
        is_sparse=False,
        fast_mode=fm,
        num_kv_splits=ns,
        intra_batch_mode=True,
    )
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(
        qo,
        kvi,
        kl,
        NUM_HEADS // NUM_KV_HEADS,
        NUM_KV_HEADS,
        True,
        wm,
        ws,
        wi,
        ri,
        rf,
        rp,
        page_size=ps,
        kv_granularity=max(ps, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=fm,
        max_split_per_batch=ns,
        intra_batch_mode=True,
        dtype_q=qd,
        dtype_kv=kvd,
    )
    e = {
        "meta": {
            "work_meta_data": wm,
            "work_indptr": wi,
            "work_info_set": ws,
            "reduce_indptr": ri,
            "reduce_final_map": rf,
            "reduce_partial_map": rp,
        },
        "kl": kl,
        "ki": ki,
        "out": torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
        ),
    }
    _cache[key] = e
    return e


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]

    use_triton = TRITON_OK and bs >= 128 and kvl >= 8192

    if use_triton:
        q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
        kv_flat = kv_fp8.view(kv_fp8.shape[0], kv_fp8.shape[-1])

        out_key = ("triton_out", bs)
        if out_key not in _cache:
            _cache[out_key] = torch.empty(
                (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
            )
        out = _cache[out_key]

        return _triton_mla(q_reshaped, kv_flat, kv_scale, kv_indptr, out)

    ns, use_a8w8, ps, fm = _get_aiter_config(bs, kvl)
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

    c = _get_or_build_aiter(
        bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm
    )
    mla_decode_fwd(
        qv,
        kv_4d,
        c["out"],
        qo_indptr,
        kv_indptr,
        c["ki"],
        c["kl"],
        1,
        page_size=ps,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **c["meta"],
    )
    return c["out"]
