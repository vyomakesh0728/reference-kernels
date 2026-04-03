#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA decode using tl.dot_scaled for mxfp4 KV.

Harness mxfp4 format (from test run):
- kv_data: (total_kv, 1, 288) fp4x2 packed
- kv_scale: (total_kv, 24) e8m0 (padded from 18 to 24)

For tl.dot_scaled with bf16 @ mxfp4:
- A (lhs): (M, K) bf16 - Q reshaped
- B (rhs): (K//2, N) uint8 packed - K transposed
- B_scale: (N, K//32) e8m0 - NOT transposed

Challenge: QK_HEAD_DIM=576, need K=64 or 128 for MFMA
- 576 = 9 * 64 = 4.5 * 128
- Use BLOCK_K=64 and loop 9 times
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
_triton_tested = [False]
_triton_works = [False]

try:
    import triton
    import triton.language as tl

    TRITON_OK = True
except ImportError:
    TRITON_OK = False


def test_dotscaled_mla():
    """Test tl.dot_scaled for MLA QK computation."""
    if _triton_tested[0]:
        return _triton_works[0]
    _triton_tested[0] = True

    if not TRITON_OK:
        print("DOTSCALED MLA: Triton not available", file=sys.stderr)
        return False

    try:

        @triton.jit
        def _test_qk_dotscaled(
            Q,  # (M, K) bf16 where M=1 (single query), K=64 (tile)
            K_packed,  # (K//2, N) uint8 where K=64, N=kv_len tile
            K_scale,  # (N, K//32) e8m0
            Out,  # (M, N) float32
            M: tl.constexpr,
            N: tl.constexpr,
            K: tl.constexpr,
        ):
            """Single tile QK dot product using dot_scaled."""
            pid = tl.program_id(0)

            # Load Q tile: (M, K)
            offs_m = tl.arange(0, M)
            offs_k = tl.arange(0, K)
            q_ptrs = Q + offs_m[:, None] * K + offs_k[None, :]
            q = tl.load(q_ptrs)  # (M, K) bf16

            # Load K tile: (K//2, N) packed
            offs_k_packed = tl.arange(0, K // 2)
            offs_n = tl.arange(0, N)
            k_ptrs = K_packed + offs_k_packed[:, None] * N + offs_n[None, :]
            k = tl.load(k_ptrs)  # (K//2, N) uint8

            # Load K scale: (N, K//32)
            offs_scale = tl.arange(0, K // 32)
            k_scale_ptrs = K_scale + offs_n[:, None] * (K // 32) + offs_scale[None, :]
            k_scale = tl.load(k_scale_ptrs)  # (N, K//32) uint8

            # dot_scaled: Q @ K^T
            acc = tl.zeros((M, N), dtype=tl.float32)
            acc = tl.dot_scaled(
                q,
                None,
                "bf16",  # lhs
                k,
                k_scale,
                "e2m1",  # rhs
                acc,
                rhs_k_pack=True,
            )

            # Store
            out_ptrs = Out + offs_m[:, None] * N + offs_n[None, :]
            tl.store(out_ptrs, acc)

        # Test with K=128 tile (what test_dotscaled_v3.py used successfully)
        # 576 = 4*128 + 64, so we need to handle 4 full tiles + partial
        M, N, K = 16, 16, 128  # Match test_dotscaled_v3.py exactly

        Q = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        # B: (K//2, N) uint8 packed - exactly like test_dotscaled_v3.py
        K_packed = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device="cuda")
        # B_scale: (N, K//32) - exactly like test_dotscaled_v3.py
        K_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
        Out = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        _test_qk_dotscaled[(1,)](Q, K_packed, K_scale, Out, M, N, K)
        torch.cuda.synchronize()

        print(
            f"DOTSCALED MLA: SUCCESS! Out sum: {Out.sum().item():.4f}", file=sys.stderr
        )
        _triton_works[0] = True
        return True

    except Exception as e:
        import traceback

        print(f"DOTSCALED MLA: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _triton_works[0] = False
        return False


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

    # Test dot_scaled on first call
    if TRITON_OK and not _triton_tested[0]:
        test_dotscaled_mla()
        # Print shapes
        if "mxfp4" in kv_data:
            kv_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
            print(
                f"MXFP4: data={kv_mxfp4.shape} dtype={kv_mxfp4.dtype}", file=sys.stderr
            )
            print(
                f"MXFP4: scale={kv_scale_mxfp4.shape} dtype={kv_scale_mxfp4.dtype}",
                file=sys.stderr,
            )

    # Use aiter fp8 for correctness
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
