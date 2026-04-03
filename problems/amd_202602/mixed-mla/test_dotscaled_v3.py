#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Test tl.dot_scaled based on Triton's test_matmul.py patterns.

From Triton test_matmul.py (test_scaled_fp4_matmul):
- For bf16 @ mxfp4: lhs_scale=None, rhs_scale=[N, K//32]
- rhs tensor: [K, N] conceptually, stored as [K//2, N] uint8 with rhs_k_pack=True
- B generated as MXFP4Tensor(size=(N, K)).to_packed_tensor(dim=1).T -> (K//2, N)

Key: The packed tensor is created from (N, K), packed along K, then transposed!
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


def test_dot_scaled_v3():
    """Test tl.dot_scaled with patterns from Triton's test_matmul.py."""
    if _dotscaled_tested[0]:
        return _dotscaled_works[0]

    _dotscaled_tested[0] = True

    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _test_kernel_v3(
            A,  # (M, K) bf16
            B,  # (K//2, N) uint8 packed fp4 - from (N, K) packed along K=1, then .T
            B_scale,  # (N, K//32) uint8 e8m0 - NOT transposed!
            C,  # (M, N) float32
            M: tl.constexpr,
            N: tl.constexpr,
            K: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            """bf16 @ mxfp4 GEMM using dot_scaled."""
            pid = tl.program_id(0)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            # Offsets
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            # Pointers for A (bf16): shape (M, K), stride (K, 1)
            a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]

            # Pointers for B (packed fp4): shape (K//2, N), stride (N, 1)
            # For k_pack=True, we read K//2 bytes per column
            BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
            offs_k_packed = tl.arange(0, BLOCK_K_PACKED)
            b_ptrs = B + offs_k_packed[:, None] * N + offs_n[None, :]

            # Pointers for B_scale: shape (N, K//32), stride (K//32, 1)
            offs_scale_k = tl.arange(0, BLOCK_K // 32)
            b_scale_ptrs = B_scale + offs_n[:, None] * (K // 32) + offs_scale_k[None, :]

            # Accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Load tiles
            a = tl.load(a_ptrs)  # (BLOCK_M, BLOCK_K) bf16
            b = tl.load(b_ptrs)  # (BLOCK_K//2, BLOCK_N) uint8
            b_scale = tl.load(b_scale_ptrs)  # (BLOCK_N, BLOCK_K//32) uint8

            # dot_scaled: C = A @ B where A is bf16, B is e2m1
            # A: (M, K), A_scale: None, A_format: "bf16"
            # B: (K, N) conceptually, (K//2, N) physically with k_pack=True
            # B_scale: (N, K//32)
            acc = tl.dot_scaled(
                a,  # lhs: (BLOCK_M, BLOCK_K) bf16
                None,  # lhs_scale: None for bf16
                "bf16",  # lhs_format
                b,  # rhs: (BLOCK_K//2, BLOCK_N) packed e2m1
                b_scale,  # rhs_scale: (BLOCK_N, BLOCK_K//32)
                "e2m1",  # rhs_format
                acc,  # accumulator
                rhs_k_pack=True,  # B is packed along K dimension
            )

            # Store
            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = C + offs_cm[:, None] * N + offs_cn[None, :]
            tl.store(c_ptrs, acc)

        # Test shapes - must satisfy:
        # - K divisible by 32 (for e8m0 scale blocks)
        # - K divisible by 128 for CDNA4 MFMA (or 64 for 32x32 variant)
        M, N, K = 16, 16, 128  # Use K=128 for native MFMA
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 128

        # A: bf16 (M, K)
        A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

        # B: Create like Triton test - (N, K) then pack along K, then transpose
        # This simulates: MXFP4Tensor(size=(N, K)).to_packed_tensor(dim=1).T
        # Result shape: (K//2, N) uint8
        B = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device="cuda")

        # B_scale: (N, K//32) - NOT transposed
        B_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")

        # Output
        C = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        # Launch
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _test_kernel_v3[grid](A, B, B_scale, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
        torch.cuda.synchronize()

        print(f"DOT_SCALED V3: SUCCESS! C sum: {C.sum().item()}", file=sys.stderr)
        _dotscaled_works[0] = True
        return True

    except Exception as e:
        import traceback

        print(f"DOT_SCALED V3: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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
        test_dot_scaled_v3()

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
