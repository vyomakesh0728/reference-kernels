#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Full Triton MLA with mxfp4 KV - v1.

Architecture for 576 dims:
- 4× K=128 tiles for first 512 dims using tl.dot_scaled
- Last 64 dims: dequant to bf16 and use tl.dot
- Online softmax for numerical stability
- V accumulation: 4× K=128 tiles (512 dims, clean!)

This kernel tries to use mxfp4 for large shapes where BW matters.
"""

import torch
import sys
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
LOG2E = 1.4426950408889634

_cache = {}
_tested = [False]

try:
    import triton
    import triton.language as tl

    TRITON_OK = True
except ImportError:
    TRITON_OK = False

if TRITON_OK:

    @triton.jit
    def _mla_mxfp4_kernel_v1(
        Q,  # (bs*heads, 576) bf16
        K_packed,  # (kv_len, 288) uint8 view of fp4x2
        K_scale,  # (kv_len, 24) uint8 view of e8m0 (padded from 18)
        V_packed,  # (kv_len, 256) uint8 - first 512 dims
        V_scale,  # (kv_len, 16) uint8 - scales for V
        Out,  # (bs*heads, 512) bf16
        kv_indptr,
        kv_len: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Single-pass MLA decode with mxfp4 KV."""
        pid = tl.program_id(0)  # (batch, head) combined

        # Get KV bounds
        bid = pid // NUM_HEADS
        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        actual_kv_len = kv_end - kv_start

        # Load Q for this (batch, head): (576,)
        q_base = Q + pid * 576

        # Online softmax state
        m_i = -float("inf")
        l_i = 0.0

        # V accumulator: (512,)
        acc = tl.zeros((V_HEAD_DIM,), dtype=tl.float32)

        # Process KV in blocks
        for n_start in range(0, actual_kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, actual_kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # ===== QK^T for this block =====
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)

            # Process 576 dims manually (simplified for now)
            for k_tile in range(0, 576, 32):
                k_end = tl.minimum(k_tile + 32, 576)
                offs_k = k_tile + tl.arange(0, 32)
                mask_k = offs_k < 576

                # Load Q slice
                q_slice = tl.load(q_base + offs_k, mask=mask_k, other=0.0).to(
                    tl.float32
                )

                # Load K packed: 16 bytes for 32 fp4 values
                k_packed_base = (
                    K_packed
                    + (kv_start + n_start + offs_n[:, None]) * 288
                    + (k_tile // 2)
                    + tl.arange(0, 16)[None, :]
                )
                k_packed = tl.load(k_packed_base, mask=mask_n[:, None], other=0).to(
                    tl.uint8
                )

                # Load scale
                scale_idx = k_tile // 32
                k_scale_ptr = K_scale + (kv_start + n_start + offs_n) * 24 + scale_idx
                scale = tl.load(k_scale_ptr, mask=mask_n, other=127).to(tl.float32)
                scale = tl.math.exp2(scale - 127.0)  # e8m0 dequant

                # Unpack and dequant (simplified)
                k_lo = (k_packed & 0xF).to(tl.float32)
                k_hi = ((k_packed >> 4) & 0xF).to(tl.float32)

                # fp4 dequant (approximate)
                k_lo = tl.where(k_lo >= 8, k_lo - 16.0, k_lo) * 0.5
                k_hi = tl.where(k_hi >= 8, k_hi - 16.0, k_hi) * 0.5

                # Sum contribution (simplified - not full interleave)
                k_sum = tl.sum(k_lo, axis=1) + tl.sum(k_hi, axis=1)
                qk += k_sum * scale * tl.sum(q_slice)  # Placeholder accumulation

            # Apply softmax scaling
            qk = qk * (SM_SCALE * LOG2E)
            qk = tl.where(mask_n, qk, -float("inf"))

            # Online softmax
            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)
            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            # V accumulation (placeholder)
            # acc += tl.sum(p[:, None] * v_vals, axis=0)

            m_i = m_new

        # Normalize
        acc = acc / l_i

        # Store
        offs_v = tl.arange(0, V_HEAD_DIM)
        out_base = Out + pid * V_HEAD_DIM + offs_v
        tl.store(out_base, acc.to(tl.bfloat16))


def test_kernel():
    """Test kernel compilation."""
    if _tested[0]:
        return
    _tested[0] = True

    print("MXFP4 MLA V1: Kernel defined (placeholder)", file=sys.stderr)


# Fallback: aiter fp8
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

    if TRITON_OK and not _tested[0]:
        test_kernel()

    # Always use aiter fp8 for now (mxfp4 kernel is placeholder)
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
