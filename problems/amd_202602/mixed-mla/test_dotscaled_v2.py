#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Test tl.dot_scaled with CORRECT tensor layouts.

From Triton docs:
- lhs: [M, K], rhs: [K, N] for C = lhs @ rhs
- For fp4: elements packed into uint8, first element in lower bits
- lhs_scale: [M, K//32], rhs_scale: [N, K//32] (N dimension, not K!)
- "Important: Do NOT transpose rhs_scale"

For fp4 with K=64:
- lhs (bf16): [M, K] = [16, 64]
- rhs (e2m1): [K, N] but packed along K -> [K//2, N] = [32, 16]
- rhs_scale: [N, K//32] = [16, 2]
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
_dotscaled_tested = [False]
_dotscaled_works = [False]


def test_dot_scaled_v2():
    """Test tl.dot_scaled with corrected tensor layouts."""
    if _dotscaled_tested[0]:
        return _dotscaled_works[0]

    _dotscaled_tested[0] = True

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _test_kernel_v2(
            A,  # (M, K) bf16
            B,  # (K//2, N) uint8 packed fp4 - K packed along first dim
            B_scale,  # (N, K//32) uint8 e8m0 - shape matches Triton spec
            C,  # (M, N) float32
            M: tl.constexpr,
            N: tl.constexpr,
            K: tl.constexpr,
        ):
            pid = tl.program_id(0)

            # Load tiles
            offs_m = tl.arange(0, M)
            offs_n = tl.arange(0, N)
            offs_k = tl.arange(0, K)

            # A: (M, K) bf16
            a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])  # (M, K)

            # B: (K//2, N) packed fp4 - each byte has 2 K elements
            # This gives us K elements across 2 dimensions
            b_ptrs = B + (offs_k // 2)[:, None] * N + offs_n[None, :]
            b = tl.load(b_ptrs)  # (K, N) but loaded from (K//2, N)

            # Scale: (N, K//32) per Triton docs
            # Each row of scale is for one N element, covering K//32 blocks
            b_scale = tl.load(
                B_scale + offs_n[:, None] * (K // 32) + (offs_k // 32)[None, :]
            )  # (N, K)

            acc = tl.zeros((M, N), dtype=tl.float32)

            # Try dot_scaled: C = A @ B
            # A is (M, K) bf16
            # B is (K, N) e2m1 with scale (N, K//32)
            result = tl.dot_scaled(
                a,  # lhs: (M, K) bf16
                None,  # lhs_scale: None for bf16
                "bf16",  # lhs_format
                b,  # rhs: (K, N) - but b is actually packed...
                b_scale,  # rhs_scale: (N, K//32)
                "e2m1",  # rhs_format
                acc=acc,
            )

            tl.store(C + offs_m[:, None] * N + offs_n[None, :], result)

        # Small test shapes - K must be divisible by 32 for scale blocks
        M, N, K = 16, 16, 64

        A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        # B packed: (K//2, N) - 2 fp4 values per byte along K dimension
        B = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device="cuda")
        # Scale: (N, K//32) per Triton spec
        B_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
        C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        _test_kernel_v2[(1,)](A, B, B_scale, C, M, N, K)
        torch.cuda.synchronize()

        print(f"DOT_SCALED V2: SUCCESS! C sum: {C.sum().item()}", file=sys.stderr)
        _dotscaled_works[0] = True
        return True

    except Exception as e:
        print(f"DOT_SCALED V2: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        _dotscaled_works[0] = False
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

    # Run dot_scaled test on first call
    if not _dotscaled_tested[0]:
        test_dot_scaled_v2()

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
