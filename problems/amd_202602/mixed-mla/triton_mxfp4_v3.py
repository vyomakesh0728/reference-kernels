#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA with mxfp4 KV - v3 simplified.

Focus on getting a working kernel first.
Uses simple vectorized V accumulation.
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

if TRITON_OK:

    @triton.jit
    def _mla_mxfp4_v3_kernel(
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
        QK_DIM: tl.constexpr,
        V_DIM: tl.constexpr,
    ):
        """Simplified mxfp4 MLA kernel."""
        bid = tl.program_id(0)
        hid = tl.program_id(1)

        kv_start = tl.load(kv_indptr + bid).to(tl.int32)
        kv_end = tl.load(kv_indptr + bid + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        # Online softmax state
        m_i = -float("inf")
        l_i = 0.0

        # V accumulator - use fixed size 512
        acc = tl.zeros((512,), dtype=tl.float32)

        # Process KV tokens one at a time (simple but slow)
        for n in range(kv_len):
            kv_idx = kv_start + n

            # ===== QK computation =====
            qk = 0.0

            # Process QK in 32-dim blocks (matches scale granularity)
            for d_block in tl.static_range(18):  # 576/32 = 18
                d_start = d_block * 32
                byte_start = d_start // 2  # 16 bytes per 32 dims

                # Load K packed: 16 bytes
                offs_bytes = tl.arange(0, 16)
                k_ptrs = KV_packed + kv_idx * 288 + byte_start + offs_bytes
                k_packed = tl.load(k_ptrs)  # (16,) uint8

                # Load scale
                scale_u8 = tl.load(KV_scale + kv_idx * 24 + d_block)
                scale = tl.math.exp2(scale_u8.to(tl.float32) - 127.0)

                # Dequant
                k_lo = (k_packed & 0xF).to(tl.float32)
                k_hi = ((k_packed >> 4) & 0xF).to(tl.float32)
                k_lo = tl.where(k_lo >= 8, k_lo - 16.0, k_lo) * 0.5 * scale
                k_hi = tl.where(k_hi >= 8, k_hi - 16.0, k_hi) * 0.5 * scale

                # Load Q even/odd
                offs_q_even = d_start + tl.arange(0, 16) * 2
                offs_q_odd = d_start + tl.arange(0, 16) * 2 + 1
                q_even = tl.load(
                    Q + bid * stride_qb + hid * stride_qh + offs_q_even * stride_qd,
                    mask=offs_q_even < QK_DIM,
                    other=0.0,
                ).to(tl.float32)
                q_odd = tl.load(
                    Q + bid * stride_qb + hid * stride_qh + offs_q_odd * stride_qd,
                    mask=offs_q_odd < QK_DIM,
                    other=0.0,
                ).to(tl.float32)

                qk += tl.sum(k_lo * q_even) + tl.sum(k_hi * q_odd)

            # Softmax update
            qk_scaled = qk * SM_SCALE_C * LOG2E_C
            m_new = tl.maximum(m_i, qk_scaled)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk_scaled - m_new)

            l_i = l_i * alpha + p
            acc = acc * alpha

            # ===== V accumulation =====
            # V is first 512 dims, process in 32-dim blocks
            for v_block in tl.static_range(16):  # 512/32 = 16
                v_start = v_block * 32
                v_byte_start = v_start // 2

                # Load V packed
                v_offs_bytes = tl.arange(0, 16)
                v_ptrs = KV_packed + kv_idx * 288 + v_byte_start + v_offs_bytes
                v_packed = tl.load(v_ptrs)

                # Load V scale
                v_scale_u8 = tl.load(KV_scale + kv_idx * 24 + v_block)
                v_scale = tl.math.exp2(v_scale_u8.to(tl.float32) - 127.0)

                # Dequant
                v_lo = (v_packed & 0xF).to(tl.float32)
                v_hi = ((v_packed >> 4) & 0xF).to(tl.float32)
                v_lo = tl.where(v_lo >= 8, v_lo - 16.0, v_lo) * 0.5 * v_scale
                v_hi = tl.where(v_hi >= 8, v_hi - 16.0, v_hi) * 0.5 * v_scale

                # Accumulate V
                # Output indices: v_start + 2*j (even), v_start + 2*j + 1 (odd)
                offs_v_even = v_start + tl.arange(0, 16) * 2
                offs_v_odd = v_start + tl.arange(0, 16) * 2 + 1

                # Use atomic-like update via masks
                acc_mask_even = (
                    (tl.arange(0, 512) >= v_start)
                    & (tl.arange(0, 512) < v_start + 32)
                    & ((tl.arange(0, 512) - v_start) % 2 == 0)
                )
                acc_mask_odd = (
                    (tl.arange(0, 512) >= v_start)
                    & (tl.arange(0, 512) < v_start + 32)
                    & ((tl.arange(0, 512) - v_start) % 2 == 1)
                )

                # Get the v_lo/v_hi values expanded to 512
                # This is inefficient but should work
                for j in range(16):
                    acc = tl.where(
                        tl.arange(0, 512) == v_start + j * 2,
                        acc + p * v_lo[j : j + 1],
                        acc,
                    )
                    acc = tl.where(
                        tl.arange(0, 512) == v_start + j * 2 + 1,
                        acc + p * v_hi[j : j + 1],
                        acc,
                    )

            m_i = m_new

        # Normalize
        acc = acc / l_i

        # Store
        offs_v = tl.arange(0, V_DIM)
        out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_v * stride_od
        tl.store(out_ptrs, acc[:V_DIM].to(tl.bfloat16))


def _triton_mxfp4_mla(q, kv_packed, kv_scale, kv_indptr, out, bs):
    """Launch kernel."""
    grid = (bs, NUM_HEADS)
    kv_u8 = kv_packed.view(torch.uint8).reshape(-1, 288)
    scale_u8 = kv_scale.view(torch.uint8).reshape(-1, 24)
    _mla_mxfp4_v3_kernel[grid](
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

    # Disable mxfp4 for now - always use aiter
    use_mxfp4 = False

    if use_mxfp4 and TRITON_OK and "mxfp4" in kv_data:
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
