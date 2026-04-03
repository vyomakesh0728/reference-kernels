#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Simpler Triton MLA with mxfp4 KV - software dequant.

Strategy: Load mxfp4 (2× BW savings), dequant to bf16, use standard dot.
This is simpler than tl.dot_scaled and tests the mxfp4 BW benefit.

For large shapes (bs>=64, kv>=4096), try mxfp4 path.
"""

import torch
import sys
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
_test_done = [False]

# E2M1 lookup table: 16 values
E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

if TRITON_OK:

    @triton.jit
    def _mla_mxfp4_simple_kernel(
        Q,  # (bs, heads, 576) bf16
        KV_packed,  # (total_kv, 288) uint8
        KV_scale,  # (total_kv, 24) uint8 e8m0
        Out,  # (bs, heads, 512) bf16
        kv_indptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_ob,
        stride_oh,
        stride_od,
        SM_SCALE_C: tl.constexpr,
        LOG2E_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,  # 32 (matches scale granularity)
        QK_DIM: tl.constexpr,  # 576
        V_DIM: tl.constexpr,  # 512
    ):
        """Simple mxfp4 MLA with software dequant."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        # Online softmax state
        m_i = -float("inf")
        l_i = 0.0

        # V accumulator
        acc = tl.zeros((V_DIM,), dtype=tl.float32)
        offs_v = tl.arange(0, V_DIM)

        # Process KV tokens
        for n_start in range(0, kv_len, BLOCK_N):
            n_end = tl.minimum(n_start + BLOCK_N, kv_len)
            n_size = n_end - n_start
            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_size

            # QK computation
            qk = tl.zeros((BLOCK_N,), dtype=tl.float32)

            # Process 576 dims in 32-dim blocks (matches scale granularity)
            for d_block in tl.static_range(18):  # 576 / 32 = 18
                d_start = d_block * 32

                # Load K packed: (BLOCK_N, 16) bytes for 32 fp4 values
                byte_start = d_start // 2
                offs_bytes = tl.arange(0, 16)
                k_ptrs = (
                    KV_packed
                    + (kv_start + n_start + offs_n[:, None]) * 288
                    + (byte_start + offs_bytes[None, :])
                )
                k_packed = tl.load(
                    k_ptrs, mask=mask_n[:, None], other=0
                )  # (BLOCK_N, 16)

                # Load scale for this 32-dim block
                scale_ptrs = KV_scale + (kv_start + n_start + offs_n) * 24 + d_block
                scale_u8 = tl.load(scale_ptrs, mask=mask_n, other=127)
                scale = tl.math.exp2(scale_u8.to(tl.float32) - 127.0)  # (BLOCK_N,)

                # Dequant: unpack lo/hi nibbles
                k_lo = (k_packed & 0xF).to(tl.float32)  # (BLOCK_N, 16)
                k_hi = ((k_packed >> 4) & 0xF).to(tl.float32)

                # E2M1 approx dequant (fast path)
                k_lo = tl.where(k_lo >= 8, k_lo - 16.0, k_lo) * 0.5
                k_hi = tl.where(k_hi >= 8, k_hi - 16.0, k_hi) * 0.5

                # Apply scale
                k_lo = k_lo * scale[:, None]
                k_hi = k_hi * scale[:, None]

                # Dot product using interleaved structure
                # k_lo has even indices, k_hi has odd indices
                # Load Q even/odd explicitly
                offs_q_even = d_start + tl.arange(0, 16) * 2
                offs_q_odd = d_start + tl.arange(0, 16) * 2 + 1
                q_even_ptrs = (
                    Q + bid * stride_qb + hid * stride_qh + offs_q_even * stride_qd
                )
                q_odd_ptrs = (
                    Q + bid * stride_qb + hid * stride_qh + offs_q_odd * stride_qd
                )
                q_even = tl.load(q_even_ptrs, mask=offs_q_even < QK_DIM, other=0.0).to(
                    tl.float32
                )
                q_odd = tl.load(q_odd_ptrs, mask=offs_q_odd < QK_DIM, other=0.0).to(
                    tl.float32
                )
                qk += tl.sum(k_lo * q_even[None, :], axis=1)
                qk += tl.sum(k_hi * q_odd[None, :], axis=1)

            # Softmax
            qk = tl.where(mask_n, qk, -float("inf"))
            qk = qk * SM_SCALE_C * LOG2E_C

            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(mask_n, p, 0.0)

            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            # V accumulation (first 512 dims)
            for v_block in tl.static_range(16):  # 512 / 32 = 16
                v_start = v_block * 32
                v_offs = v_start + tl.arange(0, 32)

                # Load V packed
                v_byte_start = v_start // 2
                v_offs_bytes = tl.arange(0, 16)
                v_ptrs = (
                    KV_packed
                    + (kv_start + n_start + offs_n[:, None]) * 288
                    + (v_byte_start + v_offs_bytes[None, :])
                )
                v_packed = tl.load(v_ptrs, mask=mask_n[:, None], other=0)

                # Load V scale
                v_scale_ptrs = KV_scale + (kv_start + n_start + offs_n) * 24 + v_block
                v_scale_u8 = tl.load(v_scale_ptrs, mask=mask_n, other=127)
                v_scale = tl.math.exp2(v_scale_u8.to(tl.float32) - 127.0)

                # Dequant
                v_lo = (v_packed & 0xF).to(tl.float32)
                v_hi = ((v_packed >> 4) & 0xF).to(tl.float32)
                v_lo = tl.where(v_lo >= 8, v_lo - 16.0, v_lo) * 0.5
                v_hi = tl.where(v_hi >= 8, v_hi - 16.0, v_hi) * 0.5
                v_lo = v_lo * v_scale[:, None]
                v_hi = v_hi * v_scale[:, None]

                # Accumulate: p @ V for even/odd output dims
                # v_out[v_start + 2*j] = sum(p * v_lo[:, j])
                # v_out[v_start + 2*j + 1] = sum(p * v_hi[:, j])
                v_contrib_lo = tl.sum(p[:, None] * v_lo, axis=0)  # (16,)
                v_contrib_hi = tl.sum(p[:, None] * v_hi, axis=0)

                # Interleave into acc
                for j in tl.static_range(16):
                    acc = tl.where(
                        offs_v == v_start + j * 2, acc + v_contrib_lo[j], acc
                    )
                    acc = tl.where(
                        offs_v == v_start + j * 2 + 1, acc + v_contrib_hi[j], acc
                    )

            m_i = m_new

        # Normalize
        acc = acc / l_i

        # Store
        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc.to(tl.bfloat16))


def _triton_mxfp4_mla(q, kv_packed, kv_scale, kv_indptr, out, bs):
    """Launch kernel."""
    grid = (bs, NUM_HEADS)
    # View as uint8 for Triton compatibility
    # kv_packed is (total_kv, 1, 288), flatten to (total_kv, 288)
    kv_u8 = kv_packed.view(torch.uint8).reshape(-1, 288)
    # kv_scale is (total_kv, 24)
    scale_u8 = kv_scale.view(torch.uint8).reshape(-1, 24)
    _mla_mxfp4_simple_kernel[grid](
        q,
        kv_u8,
        scale_u8,
        out,
        kv_indptr,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        SM_SCALE_C=SM_SCALE,
        LOG2E_C=LOG2E,
        BLOCK_N=32,
        BLOCK_D=32,
        QK_DIM=QK_HEAD_DIM,
        V_DIM=V_HEAD_DIM,
    )
    return out


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

    # Try mxfp4 Triton for large shapes
    use_mxfp4 = TRITON_OK and bs >= 64 and kvl >= 4096 and "mxfp4" in kv_data

    if use_mxfp4:
        try:
            kv_mxfp4, kv_scale = kv_data["mxfp4"]
            q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)

            out_key = ("mxfp4_out", bs)
            if out_key not in _cache:
                _cache[out_key] = torch.empty(
                    (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
                )
            out = _cache[out_key]

            return _triton_mxfp4_mla(q_reshaped, kv_mxfp4, kv_scale, kv_indptr, out, bs)
        except Exception as e:
            print(f"MXFP4 FAILED: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)

    # Fallback: aiter fp8
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
