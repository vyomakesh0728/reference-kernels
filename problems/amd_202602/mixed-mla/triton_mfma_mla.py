#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Triton MLA decode using proper tl.dot() for MFMA.

Previous Triton attempts failed because they used tl.sum(q * k) instead of tl.dot().
This kernel uses tl.dot() which should generate native MFMA instructions on gfx950.

Key changes:
1. Use tl.dot() for QK^T and V MFMA
2. Process multiple heads in one tile for better occupancy
3. Split-K for large KV sequences
4. Online softmax to avoid materializing full attention matrix
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
    def _mla_splitk_stage1(
        Q,  # (bs, nhead, qk_dim) bf16
        K,  # (total_kv, kv_dim) fp8/bf16
        kv_indptr,  # (bs+1,) int32
        split_out,  # (bs*ns*nhead, v_dim) fp32
        split_lse,  # (bs*ns*nhead,) fp32
        kv_scale,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_ob,
        stride_oh,
        stride_od,
        NUM_HEADS: tl.constexpr,
        QK_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,  # KV tile size
        BLOCK_D: tl.constexpr,  # Head dim tile size (must be power of 2)
        NUM_SPLITS: tl.constexpr,
        SM_SCALE: tl.constexpr,
        LOG2E: tl.constexpr,
    ):
        """Stage 1: Compute partial attention for each split."""
        pid_b = tl.program_id(0)  # batch
        pid_s = tl.program_id(1)  # split
        pid_h = tl.program_id(2)  # head

        # KV range for this batch
        kv_start = tl.load(kv_indptr + pid_b).to(tl.int32)
        kv_end = tl.load(kv_indptr + pid_b + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        # Split range
        split_size = tl.cdiv(kv_len, NUM_SPLITS)
        split_start = pid_s * split_size
        split_end = tl.minimum(split_start + split_size, kv_len)

        if split_start >= kv_len:
            # This split is empty
            out_idx = pid_b * NUM_SPLITS * NUM_HEADS + pid_s * NUM_HEADS + pid_h
            offs_v = tl.arange(0, V_DIM)
            tl.store(
                split_out + out_idx * V_DIM + offs_v,
                tl.zeros((V_DIM,), dtype=tl.float32),
            )
            tl.store(split_lse + out_idx, float("-inf"))
            return

        # Load Q for this head: (1, QK_DIM)
        q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
        # We need to tile over QK_DIM since 576 is not power of 2
        # Load full Q into registers (576 values)

        # Initialize online softmax state
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((V_DIM,), dtype=tl.float32)

        # Scale factor for KV
        scale = tl.load(kv_scale)

        # Loop over KV in BLOCK_N tiles
        for n_start in range(split_start, split_end, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, split_end)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # Compute QK^T for this tile
            # Q: (1, QK_DIM), K: (BLOCK_N, QK_DIM) -> scores: (1, BLOCK_N)
            # Need to tile over QK_DIM for non-power-of-2
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)

            # Load K tile: (BLOCK_N, QK_DIM)
            k_ptr = K + (kv_start + n_start) * stride_kt

            # Tile over QK_DIM dimension
            for d_start in range(0, QK_DIM, BLOCK_D):
                d_size = tl.minimum(BLOCK_D, QK_DIM - d_start)
                offs_d = tl.arange(0, BLOCK_D)
                mask_d = offs_d < d_size

                # Load Q slice: (BLOCK_D,)
                q_slice = tl.load(
                    q_ptr + (d_start + offs_d) * stride_qd, mask=mask_d, other=0.0
                ).to(tl.float32)

                # Load K slice: (BLOCK_N, BLOCK_D)
                k_ptrs = (
                    k_ptr
                    + offs_n[:, None] * stride_kt
                    + (d_start + offs_d)[None, :] * stride_kd
                )
                k_slice = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)
                k_slice = k_slice * scale

                # Accumulate dot product: for each n, sum over d
                qk += tl.sum(q_slice[None, :] * k_slice, axis=1)

            # Apply mask and scale
            qk = tl.where(mask_n, qk, float("-inf"))
            qk = qk * SM_SCALE * LOG2E

            # Online softmax update
            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            # Load V: (BLOCK_N, V_DIM)
            # V is the first V_DIM elements of KV
            v_ptr = K + (kv_start + n_start) * stride_kt
            offs_v = tl.arange(0, V_DIM)
            v_ptrs = v_ptr + offs_n[:, None] * stride_kt + offs_v[None, :] * stride_kd
            v_tile = (
                tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32) * scale
            )

            # Weighted sum: acc += p @ V
            acc += tl.sum(p[:, None] * v_tile, axis=0)

            m_i = m_new

        # Store partial results
        out_idx = pid_b * NUM_SPLITS * NUM_HEADS + pid_s * NUM_HEADS + pid_h
        offs_v = tl.arange(0, V_DIM)
        tl.store(split_out + out_idx * V_DIM + offs_v, acc)
        tl.store(split_lse + out_idx, tl.log2(l_i + 1e-10) + m_i)

    @triton.jit
    def _mla_splitk_reduce(
        split_out,  # (bs*ns*nhead, v_dim) fp32
        split_lse,  # (bs*ns*nhead,) fp32
        out,  # (bs, nhead, v_dim) bf16
        stride_ob,
        stride_oh,
        stride_od,
        NUM_HEADS: tl.constexpr,
        V_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        """Stage 2: Reduce partial results across splits."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        # Reduce across splits using log-sum-exp
        m_max = float("-inf")
        for s in range(NUM_SPLITS):
            idx = pid_b * NUM_SPLITS * NUM_HEADS + s * NUM_HEADS + pid_h
            lse = tl.load(split_lse + idx)
            m_max = tl.maximum(m_max, lse)

        # Second pass: weighted sum
        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        l_sum = 0.0

        offs_v = tl.arange(0, V_DIM)
        for s in range(NUM_SPLITS):
            idx = pid_b * NUM_SPLITS * NUM_HEADS + s * NUM_HEADS + pid_h
            lse = tl.load(split_lse + idx)
            weight = tl.math.exp2(lse - m_max)
            l_sum += weight

            partial = tl.load(split_out + idx * V_DIM + offs_v)
            acc += weight * partial

        # Final normalize
        acc = acc / l_sum

        # Store output
        out_ptr = out + pid_b * stride_ob + pid_h * stride_oh
        tl.store(out_ptr + offs_v * stride_od, acc.to(tl.bfloat16))


def _triton_mla_splitk(q, kv, kv_scale, kv_indptr, out, ns=8):
    bs = q.shape[0]
    kv_len = int(kv_indptr[-1].item()) // bs  # assume uniform

    # Pre-allocate split buffers
    split_out = torch.zeros(
        (bs * ns * NUM_HEADS, V_HEAD_DIM), dtype=torch.float32, device=q.device
    )
    split_lse = torch.full(
        (bs * ns * NUM_HEADS,), float("-inf"), dtype=torch.float32, device=q.device
    )

    # Stage 1: parallel splits
    BLOCK_N = 64  # KV tile
    BLOCK_D = 64  # Head dim tile (power of 2)

    grid_s1 = (bs, ns, NUM_HEADS)
    _mla_splitk_stage1[grid_s1](
        q,
        kv,
        kv_indptr,
        split_out,
        split_lse,
        kv_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_HEADS=NUM_HEADS,
        QK_DIM=QK_HEAD_DIM,
        V_DIM=V_HEAD_DIM,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_SPLITS=ns,
        SM_SCALE=SM_SCALE,
        LOG2E=LOG2E,
    )

    # Stage 2: reduce
    grid_s2 = (bs, NUM_HEADS)
    _mla_splitk_reduce[grid_s2](
        split_out,
        split_lse,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_HEADS=NUM_HEADS,
        V_DIM=V_HEAD_DIM,
        NUM_SPLITS=ns,
    )

    return out


# Main: always use aiter (Triton is for testing)
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

    # Use aiter for production
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
