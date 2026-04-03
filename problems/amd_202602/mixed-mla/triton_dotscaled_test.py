#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Test if tl.dot_scaled works on gfx950 for mxfp4.

This is a minimal test to check:
1. Does tl.dot_scaled compile on gfx950?
2. Does it use native V_MFMA_SCALE_F32_16X16X128_F8F6F4?
3. What's the performance vs fp8?

If this works, we can build a full FlashAttention-style kernel.
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

_cache = {}

if TRITON_OK:

    @triton.jit
    def _test_dot_scaled_kernel(
        A,  # (M, K) bf16
        B,  # (N, K//2) uint8 packed fp4
        B_scale,  # (N, K//32) uint8 e8m0
        C,  # (M, N) bf16
        M,
        N,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Test dot_scaled: bf16 A @ mxfp4 B^T."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Process K in blocks
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)

            # Load A tile (bf16)
            a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
            a = tl.load(
                a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
            )

            # Load B tile (fp4 packed)
            # B is (N, K//2) with 2 fp4 per byte
            b_k_packed = k // 2
            offs_k_packed = b_k_packed + tl.arange(0, BLOCK_K // 2)
            b_ptrs = B + offs_n[:, None] * (K // 2) + offs_k_packed[None, :]
            b_packed = tl.load(
                b_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k_packed[None, :] < K // 2),
                other=0,
            )

            # Load B scale (e8m0)
            # Scale is per 32 elements: (N, K//32)
            scale_k = k // 32
            scale_ptrs = B_scale + offs_n * (K // 32) + scale_k
            b_scale = tl.load(scale_ptrs, mask=offs_n < N, other=127)

            # Try dot_scaled
            # Note: This may require specific input formats
            # For now, try with the available types
            try:
                # Attempt native dot_scaled
                # lhs = A (bf16), rhs = B (e2m1 packed), rhs_scale = e8m0
                result = tl.dot_scaled(
                    a.to(tl.bfloat16),  # lhs
                    None,  # lhs_scale (None for bf16)
                    "bf16",  # lhs_format
                    b_packed,  # rhs (packed fp4)
                    b_scale[:, None],  # rhs_scale (broadcast to match)
                    "e2m1",  # rhs_format
                    acc=acc,
                )
                acc = result
            except:
                # Fallback: manual computation
                # This won't be fast, just for testing
                pass

        # Store result
        c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
        tl.store(
            c_ptrs,
            acc.to(tl.bfloat16),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def test_dot_scaled():
    """Test if dot_scaled compiles and runs on gfx950."""
    M, N, K = 16, 64, 576

    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    B_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    B_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
    C = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    try:
        _test_dot_scaled_kernel[grid](
            A,
            B_packed,
            B_scale,
            C,
            M,
            N,
            K,
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=128,
        )
        print(f"dot_scaled test completed. C sum: {C.sum()}")
        return True
    except Exception as e:
        print(f"dot_scaled test failed: {e}")
        return False


# Main kernel: fallback to aiter for now
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

    # Use aiter fp8 (proven path)
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
