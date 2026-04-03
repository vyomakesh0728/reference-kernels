#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Triton MLA with tiled mxfp4 QK computation.

Strategy: Accumulate QK scores across K-dimension tiles.
For Q(1, 576) @ K(kv_len, 576)^T:
- Tile K dim: 576 = 4×128 + 64
- Use K=128 tiles for first 4 (native MFMA)
- Handle last 64 dims separately (or pad)

Key insight: Don't need full transpose. Process K tiles and accumulate.
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
_kernel_compiled = [False]

try:
    import triton
    import triton.language as tl

    TRITON_OK = True
except ImportError:
    TRITON_OK = False

if TRITON_OK:

    @triton.jit
    def _qk_tile_128(
        Q,  # (BLOCK_M, 128) bf16 - Q slice for this K tile
        K_packed,  # (64, BLOCK_N) uint8 - K^T slice (128/2=64 packed)
        K_scale,  # (BLOCK_N, 4) uint8 - scales for 128/32=4 blocks
        Out,  # (BLOCK_M, BLOCK_N) float32 - accumulator
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Single 128-dim K tile contribution to QK scores."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load Q: (BLOCK_M, 128)
        offs_k = tl.arange(0, 128)
        q_ptrs = Q + offs_m[:, None] * 128 + offs_k[None, :]
        q = tl.load(q_ptrs)

        # Load K_packed: (64, BLOCK_N) - already transposed
        offs_k_packed = tl.arange(0, 64)
        k_ptrs = K_packed + offs_k_packed[:, None] * BLOCK_N + offs_n[None, :]
        k = tl.load(k_ptrs)

        # Load K_scale: (BLOCK_N, 4)
        offs_scale = tl.arange(0, 4)
        scale_ptrs = K_scale + offs_n[:, None] * 4 + offs_scale[None, :]
        k_scale = tl.load(scale_ptrs)

        # dot_scaled
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
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
        out_ptrs = Out + offs_m[:, None] * BLOCK_N + offs_n[None, :]
        tl.store(out_ptrs, acc)


def test_tiled_qk():
    """Test tiled QK computation with actual tile sizes."""
    if _kernel_compiled[0]:
        return True
    _kernel_compiled[0] = True

    if not TRITON_OK:
        print("TILED QK: Triton not available", file=sys.stderr)
        return False

    try:
        # Test single K=128 tile
        BLOCK_M, BLOCK_N = 16, 32
        K_TILE = 128

        Q = torch.randn((BLOCK_M, K_TILE), dtype=torch.bfloat16, device="cuda")
        # K_packed: (K_TILE//2, BLOCK_N) = (64, 32) - transposed
        K_packed = torch.randint(
            0, 256, (K_TILE // 2, BLOCK_N), dtype=torch.uint8, device="cuda"
        )
        # K_scale: (BLOCK_N, K_TILE//32) = (32, 4)
        K_scale = torch.full(
            (BLOCK_N, K_TILE // 32), 127, dtype=torch.uint8, device="cuda"
        )
        Out = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32, device="cuda")

        grid = (1, 1)
        _qk_tile_128[grid](Q, K_packed, K_scale, Out, BLOCK_M, BLOCK_N)
        torch.cuda.synchronize()

        print(f"TILED QK: SUCCESS! Out sum={Out.sum().item():.4f}", file=sys.stderr)
        return True

    except Exception as e:
        import traceback

        print(f"TILED QK: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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

    # Test tiled kernel on first call
    if TRITON_OK and not _kernel_compiled[0]:
        test_tiled_qk()

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
