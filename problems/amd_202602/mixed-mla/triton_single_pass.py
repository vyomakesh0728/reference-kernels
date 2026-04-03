#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton single-pass MQA decode kernel. Eliminates split-K materialization.

Based on Agent 3 analysis. One program per batch, all 16 heads, no reduce pass.
Target: bs=256,kv=8k from 317us → ~180-220us.
"""

import torch
from task import input_t, output_t

# Try importing Triton
try:
    import triton
    import triton.language as tl
    from triton import Config

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
LOG2E = 1.4426950408889634

# Constants
BLOCK_K = 64
BLOCK_V = 128

# Fallback to aiter if Triton not available
if not TRITON_AVAILABLE:
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd
    from aiter.ops.quant import dynamic_per_tensor_quant

    FP8_DTYPE = aiter_dtypes.fp8
    _cache = {}

    def _get_config(bs, kvl):
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

    def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
        key = (bs, kvl, ns, qd, ps, fm)
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

    def _aiter_fallback(data: input_t) -> output_t:
        q, kv_data, qo_indptr, kv_indptr, config = data
        bs = int(config["batch_size"])
        kvl = int(config["kv_seq_len"])
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

else:
    # Triton is available - define the kernel
    @triton.autotune(
        configs=[
            Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
            Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
            Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
        ],
        key=["B"],
    )
    @triton.jit
    def mla_decode_mqa_fp8_kernel(
        q_ptr,
        kv_ptr,
        kv_indptr_ptr,
        kv_scale_ptr,
        out_ptr,
        B,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_ob,
        stride_oh,
        stride_od,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        NUM_HEADS_C: tl.constexpr,
        QK_DIM: tl.constexpr,
        SM_SCALE_C: tl.constexpr,
        LOG2E_C: tl.constexpr,
    ):
        bid = tl.program_id(0)
        offs_h = tl.arange(0, NUM_HEADS_C)
        kv_begin = tl.load(kv_indptr_ptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr_ptr + bid + 1).to(tl.int32)
        kv_scale = tl.load(kv_scale_ptr)

        m_i = tl.full((NUM_HEADS_C,), -float("inf"), tl.float32)
        l_i = tl.zeros((NUM_HEADS_C,), tl.float32)

        acc0 = tl.zeros((NUM_HEADS_C, BLOCK_V), tl.float32)
        acc1 = tl.zeros((NUM_HEADS_C, BLOCK_V), tl.float32)
        acc2 = tl.zeros((NUM_HEADS_C, BLOCK_V), tl.float32)
        acc3 = tl.zeros((NUM_HEADS_C, BLOCK_V), tl.float32)

        for rel_n in tl.range(0, kv_end - kv_begin, BLOCK_N):
            start_n = kv_begin + rel_n
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < kv_end

            qk = tl.zeros((NUM_HEADS_C, BLOCK_N), tl.float32)

            for k0 in range(0, QK_DIM, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < QK_DIM

                q_ptrs = (
                    q_ptr
                    + bid * stride_qb
                    + offs_h[:, None] * stride_qh
                    + offs_k[None, :] * stride_qd
                )
                q = tl.load(q_ptrs, mask=mask_k[None, :], other=0.0).to(tl.bfloat16)

                k_ptrs = (
                    kv_ptr + offs_n[:, None] * stride_kt + offs_k[None, :] * stride_kd
                )
                k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                k = k.to(tl.bfloat16) * kv_scale

                qk += tl.dot(q, tl.trans(k))

            qk *= SM_SCALE_C * LOG2E_C

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk - m_ij[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc2 = acc2 * alpha[:, None]
            acc3 = acc3 * alpha[:, None]

            p_bf16 = p.to(tl.bfloat16)

            v0_ptrs = (
                kv_ptr
                + offs_n[:, None] * stride_kt
                + (0 * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] * stride_kd
            )
            v1_ptrs = (
                kv_ptr
                + offs_n[:, None] * stride_kt
                + (1 * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] * stride_kd
            )
            v2_ptrs = (
                kv_ptr
                + offs_n[:, None] * stride_kt
                + (2 * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] * stride_kd
            )
            v3_ptrs = (
                kv_ptr
                + offs_n[:, None] * stride_kt
                + (3 * BLOCK_V + tl.arange(0, BLOCK_V))[None, :] * stride_kd
            )

            v0 = (
                tl.load(v0_ptrs, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
                * kv_scale
            )
            v1 = (
                tl.load(v1_ptrs, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
                * kv_scale
            )
            v2 = (
                tl.load(v2_ptrs, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
                * kv_scale
            )
            v3 = (
                tl.load(v3_ptrs, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
                * kv_scale
            )

            acc0 += tl.dot(p_bf16, v0)
            acc1 += tl.dot(p_bf16, v1)
            acc2 += tl.dot(p_bf16, v2)
            acc3 += tl.dot(p_bf16, v3)

            m_i = m_ij

        inv_l = 1.0 / l_i
        acc0 = acc0 * inv_l[:, None]
        acc1 = acc1 * inv_l[:, None]
        acc2 = acc2 * inv_l[:, None]
        acc3 = acc3 * inv_l[:, None]

        offs_v = tl.arange(0, BLOCK_V)

        o0_ptrs = (
            out_ptr
            + bid * stride_ob
            + offs_h[:, None] * stride_oh
            + (0 * BLOCK_V + offs_v)[None, :] * stride_od
        )
        o1_ptrs = (
            out_ptr
            + bid * stride_ob
            + offs_h[:, None] * stride_oh
            + (1 * BLOCK_V + offs_v)[None, :] * stride_od
        )
        o2_ptrs = (
            out_ptr
            + bid * stride_ob
            + offs_h[:, None] * stride_oh
            + (2 * BLOCK_V + offs_v)[None, :] * stride_od
        )
        o3_ptrs = (
            out_ptr
            + bid * stride_ob
            + offs_h[:, None] * stride_oh
            + (3 * BLOCK_V + offs_v)[None, :] * stride_od
        )

        tl.store(o0_ptrs, acc0.to(tl.bfloat16))
        tl.store(o1_ptrs, acc1.to(tl.bfloat16))
        tl.store(o2_ptrs, acc2.to(tl.bfloat16))
        tl.store(o3_ptrs, acc3.to(tl.bfloat16))

    def mla_decode_triton(q, kv_fp8, kv_scale, kv_indptr):
        B = q.shape[0]
        out = torch.empty(
            (B, NUM_HEADS, V_HEAD_DIM), device=q.device, dtype=torch.bfloat16
        )

        grid = (B,)
        mla_decode_mqa_fp8_kernel[grid](
            q,
            kv_fp8,
            kv_indptr,
            kv_scale,
            out,
            B,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_fp8.stride(0),
            kv_fp8.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_K=BLOCK_K,
            BLOCK_V=BLOCK_V,
            NUM_HEADS_C=NUM_HEADS,
            QK_DIM=QK_HEAD_DIM,
            SM_SCALE_C=SM_SCALE,
            LOG2E_C=LOG2E,
        )
        return out


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    # Use Triton for large shapes, fallback for small
    if TRITON_AVAILABLE and bs >= 64 and kvl >= 8192:
        kv_fp8, kv_scale = kv_data["fp8"]
        # Flatten KV: (total_kv, 576)
        kv_flat = kv_fp8.view(kv_fp8.shape[0], kv_fp8.shape[-1])
        # Reshape Q: (B, 16, 576)
        q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
        return mla_decode_triton(q_reshaped, kv_flat, kv_scale, kv_indptr)

    # Fallback to aiter
    if not TRITON_AVAILABLE:
        return _aiter_fallback(data)

    # For small shapes or when Triton isn't optimal, use aiter
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd
    from aiter.ops.quant import dynamic_per_tensor_quant

    FP8_DTYPE = aiter_dtypes.fp8
    if not hasattr(custom_kernel, "_cache"):
        custom_kernel._cache = {}
    _cache = custom_kernel._cache

    def _get_config(bs, kvl):
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

    def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
        key = (bs, kvl, ns, qd, ps, fm)
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
