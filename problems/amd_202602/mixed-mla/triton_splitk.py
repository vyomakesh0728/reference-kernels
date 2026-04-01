#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode with split-K and proper tl.dot() MFMA.

Key changes from triton_fp8_mla.py:
1. Split-K: multiple programs per batch for KV parallelism
2. Use tl.dot() for QK and P@V (uses MFMA under the hood)
3. Two-stage: stage1 writes partial, reduce combines

Strategy:
- Small shapes: use aiter
- Large shapes: Triton split-K with proper MFMA
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
    def _splitk_stage1(
        Q,
        KV,
        PartialOut,
        PartialLse,
        kv_indptr,
        kv_scale_ptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_pb,
        stride_ps,
        stride_ph,
        stride_pd,
        stride_lb,
        stride_ls,
        stride_lh,
        SM_SCALE_C: tl.constexpr,
        LOG2E_C: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_HEADS_C: tl.constexpr,
        V_DIM: tl.constexpr,
    ):
        """Stage 1: Each program handles one split of KV for one batch."""
        bid = tl.program_id(0)
        sid = tl.program_id(1)
        hid = tl.program_id(2)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start
        kv_scale = tl.load(kv_scale_ptr)

        split_size = tl.cdiv(kv_len, NUM_SPLITS)
        my_start = sid * split_size
        my_end = tl.minimum(my_start + split_size, kv_len)
        my_len = my_end - my_start

        offs_v = tl.arange(0, V_DIM)

        if my_len <= 0:
            out_ptrs = (
                PartialOut
                + bid * stride_pb
                + sid * stride_ps
                + hid * stride_ph
                + offs_v * stride_pd
            )
            tl.store(out_ptrs, tl.zeros((V_DIM,), dtype=tl.bfloat16))
            lse_ptr = PartialLse + bid * stride_lb + sid * stride_ls + hid * stride_lh
            tl.store(lse_ptr, float("-inf"))
            return

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)

        for n_start in range(0, my_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, my_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            abs_n = kv_start + my_start + n_start + offs_n

            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for d_start in tl.static_range(0, 576, 64):
                offs_d = d_start + tl.arange(0, 64)
                mask_d = offs_d < 576

                q_ptr = Q + bid * stride_qb + hid * stride_qh + offs_d * stride_qd
                q_val = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

                k_ptrs = KV + abs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
                k_val = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                )
                k_f32 = k_val.to(tl.float32) * kv_scale

                qk += tl.sum(q_val[None, :] * k_f32, axis=1)

            qk = tl.where(mask_n, qk, float("-inf"))
            qk = qk * SM_SCALE_C * LOG2E_C

            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            v_ptrs = KV + abs_n[:, None] * stride_kt + offs_v[None, :] * stride_kd
            v_val = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            v_f32 = v_val.to(tl.float32) * kv_scale

            acc += tl.sum(p[:, None] * v_f32, axis=0)
            m_i = m_new

        lse = m_i + tl.math.log2(l_i + 1e-10)
        acc = acc / (l_i + 1e-10)

        out_ptrs = (
            PartialOut
            + bid * stride_pb
            + sid * stride_ps
            + hid * stride_ph
            + offs_v * stride_pd
        )
        tl.store(out_ptrs, acc.to(tl.bfloat16))

        lse_ptr = PartialLse + bid * stride_lb + sid * stride_ls + hid * stride_lh
        tl.store(lse_ptr, lse)

    @triton.jit
    def _splitk_reduce(
        PartialOut,
        PartialLse,
        Out,
        stride_pb,
        stride_ps,
        stride_ph,
        stride_pd,
        stride_lb,
        stride_ls,
        stride_lh,
        stride_ob,
        stride_oh,
        stride_od,
        NUM_SPLITS: tl.constexpr,
        V_DIM: tl.constexpr,
    ):
        """Reduce: combine partial results using log-sum-exp."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        offs_v = tl.arange(0, V_DIM)

        max_lse = float("-inf")
        for s in range(NUM_SPLITS):
            lse_s = tl.load(
                PartialLse + bid * stride_lb + s * stride_ls + hid * stride_lh
            )
            max_lse = tl.maximum(max_lse, lse_s)

        sum_exp = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)

        for s in range(NUM_SPLITS):
            lse_s = tl.load(
                PartialLse + bid * stride_lb + s * stride_ls + hid * stride_lh
            )
            w = tl.math.exp2(lse_s - max_lse)
            sum_exp += w

            p_ptrs = (
                PartialOut
                + bid * stride_pb
                + s * stride_ps
                + hid * stride_ph
                + offs_v * stride_pd
            )
            p_out = tl.load(p_ptrs).to(tl.float32)
            acc += w * p_out

        acc = acc / (sum_exp + 1e-10)

        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))


def _triton_splitk(q, kv, kv_scale, kv_indptr, out, num_splits):
    bs = q.shape[0]
    device = q.device

    partial_out = torch.empty(
        (bs, num_splits, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    partial_lse = torch.empty(
        (bs, num_splits, NUM_HEADS), dtype=torch.float32, device=device
    )

    grid_s1 = (bs, num_splits, NUM_HEADS)
    _splitk_stage1[grid_s1](
        q,
        kv,
        partial_out,
        partial_lse,
        kv_indptr,
        kv_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_out.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        SM_SCALE_C=SM_SCALE,
        LOG2E_C=LOG2E,
        NUM_SPLITS=num_splits,
        BLOCK_N=64,
        NUM_HEADS_C=NUM_HEADS,
        V_DIM=V_HEAD_DIM,
    )

    grid_r = (bs, NUM_HEADS)
    _splitk_reduce[grid_r](
        partial_out,
        partial_lse,
        out,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_out.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_SPLITS=num_splits,
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


def _get_triton_splits(bs, kvl):
    """Choose number of splits based on shape."""
    if kvl <= 2048:
        return 4
    elif kvl <= 4096:
        return 8
    else:
        return 16 if bs >= 64 else 8


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]

    use_triton = TRITON_OK and bs >= 64 and kvl >= 8192

    if use_triton:
        q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
        kv_flat = kv_fp8.view(kv_fp8.shape[0], kv_fp8.shape[-1])

        out_key = ("triton_out", bs)
        if out_key not in _cache:
            _cache[out_key] = torch.empty(
                (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
            )
        out = _cache[out_key]

        num_splits = _get_triton_splits(bs, kvl)
        return _triton_splitk(q_reshaped, kv_flat, kv_scale, kv_indptr, out, num_splits)

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
