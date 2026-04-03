#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Test tl.dot_scaled for MLA QK^T computation with mxfp4 K.

Strategy: Compute K @ Q.T instead of Q @ K.T (avoids transpose).
- K: [kv_len, 576] mxfp4 stored as [kv_len, 288] uint8 with [kv_len, 18] e8m0 scale
- Q.T: [576, num_heads] bf16 = Q transposed from [num_heads, 576]
- Output: [kv_len, num_heads] = scores.T

Then transpose to get scores [num_heads, kv_len].

This leverages the native storage layout of the KV cache.
"""

import torch
import sys
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

_cache = {}
_mxfp4_qk_tested = [False]
_mxfp4_qk_works = [False]


def test_mxfp4_qk():
    """Test mxfp4 K @ bf16 Q.T using tl.dot_scaled."""
    if _mxfp4_qk_tested[0]:
        return _mxfp4_qk_works[0]

    _mxfp4_qk_tested[0] = True

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _mxfp4_qk_kernel(
            K,  # (kv_len, K_DIM//2) uint8 packed mxfp4
            K_scale,  # (kv_len, K_DIM//32) uint8 e8m0
            Q,  # (num_heads, K_DIM) bf16
            Out,  # (kv_len, num_heads) float32
            kv_len,
            K_DIM: tl.constexpr,
            NUM_HEADS: tl.constexpr,
            BLOCK_KV: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            """Compute K @ Q.T where K is mxfp4 and Q.T is bf16."""
            pid = tl.program_id(0)

            # This block handles BLOCK_KV rows of K
            kv_start = pid * BLOCK_KV
            offs_kv = kv_start + tl.arange(0, BLOCK_KV)
            mask_kv = offs_kv < kv_len

            # Accumulator: (BLOCK_KV, NUM_HEADS)
            acc = tl.zeros((BLOCK_KV, NUM_HEADS), dtype=tl.float32)

            # Loop over K dimension in tiles
            BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
            NUM_K_TILES: tl.constexpr = K_DIM // BLOCK_K

            for k_tile in tl.static_range(NUM_K_TILES):
                k_start = k_tile * BLOCK_K
                offs_k_packed = tl.arange(0, BLOCK_K_PACKED)

                # Load K tile: (BLOCK_KV, BLOCK_K) from (kv_len, K_DIM//2)
                # K is packed along dim=1 (K dimension)
                k_ptrs = (
                    K
                    + offs_kv[:, None] * (K_DIM // 2)
                    + (k_start // 2 + offs_k_packed)[None, :]
                )
                k_tile_data = tl.load(k_ptrs, mask=mask_kv[:, None], other=0)

                # Load K_scale: (BLOCK_KV, BLOCK_K//32) from (kv_len, K_DIM//32)
                offs_scale_k = tl.arange(0, BLOCK_K // 32)
                k_scale_ptrs = (
                    K_scale
                    + offs_kv[:, None] * (K_DIM // 32)
                    + (k_start // 32 + offs_scale_k)[None, :]
                )
                k_scale_data = tl.load(k_scale_ptrs, mask=mask_kv[:, None], other=127)

                # Load Q.T tile: (BLOCK_K, NUM_HEADS) from Q (NUM_HEADS, K_DIM)
                # Need to transpose Q on load
                offs_k = k_start + tl.arange(0, BLOCK_K)
                offs_h = tl.arange(0, NUM_HEADS)
                q_ptrs = (
                    Q + offs_h[None, :] * K_DIM + offs_k[:, None]
                )  # Note: transposed access
                q_tile = tl.load(q_ptrs)  # (BLOCK_K, NUM_HEADS) bf16

                # dot_scaled: K_tile (BLOCK_KV, BLOCK_K) @ Q.T (BLOCK_K, NUM_HEADS)
                # lhs: mxfp4 (BLOCK_KV, BLOCK_K) with scale (BLOCK_KV, BLOCK_K//32)
                # rhs: bf16 (BLOCK_K, NUM_HEADS) with scale None
                acc = tl.dot_scaled(
                    k_tile_data,  # (BLOCK_KV, BLOCK_K//2) packed
                    k_scale_data,  # (BLOCK_KV, BLOCK_K//32)
                    "e2m1",
                    q_tile,  # (BLOCK_K, NUM_HEADS)
                    None,  # bf16 doesn't need scale
                    "bf16",
                    acc,
                    lhs_k_pack=True,  # K packed along K dimension
                )

            # Store output: (BLOCK_KV, NUM_HEADS)
            out_ptrs = (
                Out + offs_kv[:, None] * NUM_HEADS + tl.arange(0, NUM_HEADS)[None, :]
            )
            tl.store(out_ptrs, acc, mask=mask_kv[:, None])

        # Test with small shapes
        kv_len = 128
        K_DIM = 576  # Must be divisible by 32 for scales
        BLOCK_K = 128  # CDNA4 native MFMA tile

        # Problem: K_DIM=576 is not divisible by 128!
        # We need K_DIM divisible by BLOCK_K for this simple test
        # Let's use K_DIM=512 for testing (will need padding for real 576)
        K_DIM = 512  # Simplified for test

        # K: (kv_len, K_DIM//2) packed mxfp4
        K = torch.randint(
            0, 256, (kv_len, K_DIM // 2), dtype=torch.uint8, device="cuda"
        )
        # K_scale: (kv_len, K_DIM//32) e8m0
        K_scale = torch.full(
            (kv_len, K_DIM // 32), 127, dtype=torch.uint8, device="cuda"
        )
        # Q: (NUM_HEADS, K_DIM) bf16
        Q = torch.randn((NUM_HEADS, K_DIM), dtype=torch.bfloat16, device="cuda")
        # Output: (kv_len, NUM_HEADS) float32
        Out = torch.zeros((kv_len, NUM_HEADS), dtype=torch.float32, device="cuda")

        BLOCK_KV = 64
        grid = (triton.cdiv(kv_len, BLOCK_KV),)

        _mxfp4_qk_kernel[grid](
            K,
            K_scale,
            Q,
            Out,
            kv_len,
            K_DIM=K_DIM,
            NUM_HEADS=NUM_HEADS,
            BLOCK_KV=BLOCK_KV,
            BLOCK_K=BLOCK_K,
        )
        torch.cuda.synchronize()

        print(f"MXFP4_QK: SUCCESS! Out mean: {Out.mean().item():.4f}", file=sys.stderr)
        _mxfp4_qk_works[0] = True
        return True

    except Exception as e:
        import traceback

        print(f"MXFP4_QK: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _mxfp4_qk_works[0] = False
        return False


# Main kernel: aiter fallback
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

    # Run mxfp4 QK test on first call
    if not _mxfp4_qk_tested[0]:
        test_mxfp4_qk()

    # Always use aiter for correctness
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
