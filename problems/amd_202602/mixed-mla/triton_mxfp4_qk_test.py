#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Test tl.dot_scaled with actual harness mxfp4 data layout.

Goal: Verify we can compute Q @ K^T using tl.dot_scaled with the harness format.

Harness provides:
- kv_data: (total_kv, 1, 288) torch.float4_e2m1fn_x2 (row-major)
- kv_scale: (total_kv, 24) torch.float8_e8m0fnu

For tl.dot_scaled:
- A (lhs): Q as (M, K) bf16 where M=bs*heads, K=padded_dim
- B (rhs): K^T as (K//2, N) uint8 where N=kv_len
- B_scale: (N, K//32) e8m0

Challenge: Need to transpose K from (kv_len, 288) to (288, kv_len)
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
_test_done = [False]

try:
    import triton
    import triton.language as tl

    TRITON_OK = True
except ImportError:
    TRITON_OK = False


def test_harness_layout():
    """Test tl.dot_scaled with transposed harness data."""
    if _test_done[0]:
        return
    _test_done[0] = True

    if not TRITON_OK:
        print("HARNESS TEST: Triton not available", file=sys.stderr)
        return

    try:

        @triton.jit
        def _qk_tile_kernel(
            Q,  # (M, K) bf16 - single query expanded
            K_T,  # (K//2, N) uint8 - K transposed and packed
            K_scale,  # (N, K//32) uint8 - per block scales
            Out,  # (M, N) float32 - QK scores
            M: tl.constexpr,
            N: tl.constexpr,
            K: tl.constexpr,
        ):
            """Compute single tile of QK^T."""
            pid = tl.program_id(0)

            # For now just test with small tile
            offs_m = tl.arange(0, M)
            offs_n = tl.arange(0, N)
            offs_k = tl.arange(0, K)

            # Load Q: (M, K) bf16
            q_ptrs = Q + offs_m[:, None] * K + offs_k[None, :]
            q = tl.load(q_ptrs)

            # Load K_T: (K//2, N) uint8
            offs_k_packed = tl.arange(0, K // 2)
            k_ptrs = K_T + offs_k_packed[:, None] * N + offs_n[None, :]
            k = tl.load(k_ptrs)

            # Load K_scale: (N, K//32)
            offs_scale = tl.arange(0, K // 32)
            scale_ptrs = K_scale + offs_n[:, None] * (K // 32) + offs_scale[None, :]
            k_scale = tl.load(scale_ptrs)

            # dot_scaled
            acc = tl.zeros((M, N), dtype=tl.float32)
            acc = tl.dot_scaled(
                q,
                None,
                "bf16",
                k,
                k_scale,
                "e2m1",
                acc,
                rhs_k_pack=True,
            )

            # Store
            out_ptrs = Out + offs_m[:, None] * N + offs_n[None, :]
            tl.store(out_ptrs, acc)

        # Test with K=128 (known working)
        M, N, K = 16, 32, 128  # M queries, N KV tokens, K dims per tile

        Q = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

        # Simulate transposed K: create (N, K) then pack and transpose
        K_raw = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
        # Pack to (N, K//2) uint8 (simulate fp4x2)
        K_packed_row = torch.randint(
            0, 256, (N, K // 2), dtype=torch.uint8, device="cuda"
        )
        # Transpose to (K//2, N)
        K_T = K_packed_row.T.contiguous()

        # Scale: (N, K//32)
        K_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")

        Out = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        _qk_tile_kernel[(1,)](Q, K_T, K_scale, Out, M, N, K)
        torch.cuda.synchronize()

        print(f"HARNESS TEST: SUCCESS! Out sum={Out.sum().item():.4f}", file=sys.stderr)
        print(
            f"HARNESS TEST: Out shape={Out.shape}, range=[{Out.min().item():.2f}, {Out.max().item():.2f}]",
            file=sys.stderr,
        )

    except Exception as e:
        import traceback

        print(f"HARNESS TEST: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


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

    # Run test on first call
    if TRITON_OK and not _test_done[0]:
        test_harness_layout()
        # Also print actual mxfp4 tensor info
        if "mxfp4" in kv_data:
            kv_mxfp4, kv_scale = kv_data["mxfp4"]
            print(
                f"ACTUAL MXFP4: data={kv_mxfp4.shape} {kv_mxfp4.dtype} stride={kv_mxfp4.stride()}",
                file=sys.stderr,
            )
            print(
                f"ACTUAL MXFP4: scale={kv_scale.shape} {kv_scale.dtype} stride={kv_scale.stride()}",
                file=sys.stderr,
            )
            # Check if we can view as uint8
            try:
                kv_u8 = kv_mxfp4.view(torch.uint8)
                print(f"ACTUAL MXFP4: uint8 view shape={kv_u8.shape}", file=sys.stderr)
            except Exception as e:
                print(f"ACTUAL MXFP4: uint8 view failed: {e}", file=sys.stderr)

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
