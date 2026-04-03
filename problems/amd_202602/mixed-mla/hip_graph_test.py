#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""HIP Graph capture test for MLA decode.

Goal: Understand what operations can be captured in a HIP graph and measure
the potential dispatch overhead reduction.

Findings from research:
- aiter's mla_decode_fwd crashes on graph replay: "memory access fault"
- Root cause: aiter assembly kernels do internal allocations for split-K buffers
- HIP graphs require fixed memory addresses between capture and replay

Test strategy:
1. Identify which operations can/cannot be captured
2. Try different isolation strategies (private pool, pre-allocation)
3. Measure overhead difference

Expected outcome:
- If we can capture: ~270µs overhead → ~5-10µs overhead
- Wall time: 293µs → ~30-40µs (matching GPU time)
"""

import torch
import sys
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8
_cache = {}
_graph_cache = {}
_tested_graph = [False]


def _get_config(bs, kvl):
    """Aggressive split policy: keep 1k path, force 8k path to a8w8+ns32."""
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

    # Also pre-allocate split buffers that aiter might use internally
    # aiter uses num_kv_splits * batch_size * num_heads elements
    split_size = ns * bs * NUM_HEADS * V_HEAD_DIM
    lse_size = ns * bs * NUM_HEADS

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
        # Pre-allocated split buffers (may not be used by aiter, but for our custom path)
        "split_out": torch.empty(
            (ns, bs * NUM_HEADS, V_HEAD_DIM), dtype=torch.float32, device=dev
        ),
        "split_lse": torch.empty((ns, bs * NUM_HEADS), dtype=torch.float32, device=dev),
    }
    _cache[key] = e
    return e


def _test_graph_capture():
    """Test what operations can be captured in a HIP graph."""
    if _tested_graph[0]:
        return
    _tested_graph[0] = True

    print("=" * 60, file=sys.stderr)
    print("HIP GRAPH CAPTURE TEST", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    device = torch.device("cuda")

    # Test 1: Basic tensor ops (should work)
    print("\nTest 1: Basic tensor ops...", file=sys.stderr)
    try:
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        c = torch.empty(100, 100, device=device)

        # Warmup
        c.copy_(torch.matmul(a, b))
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            c.copy_(torch.matmul(a, b))

        g.replay()
        torch.cuda.synchronize()
        print("  PASS: Basic tensor ops can be captured", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)

    # Test 2: Quantization ops
    print("\nTest 2: dynamic_per_tensor_quant...", file=sys.stderr)
    try:
        q = torch.randn(
            16, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        qi = torch.empty_like(q, dtype=FP8_DTYPE)
        qs = torch.empty(1, dtype=torch.float32, device=device)

        # Warmup
        dynamic_per_tensor_quant(qi, q, qs)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            dynamic_per_tensor_quant(qi, q, qs)

        g.replay()
        torch.cuda.synchronize()
        print("  PASS: Quantization ops can be captured", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)

    # Test 3: aiter mla_decode_fwd (expected to fail)
    print("\nTest 3: aiter mla_decode_fwd...", file=sys.stderr)
    try:
        bs, kvl = 4, 1024
        ns, use_a8w8, ps, fm = _get_config(bs, kvl)

        q = torch.randn(
            bs, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        kv = torch.randn(
            bs * kvl,
            1,
            NUM_KV_HEADS,
            QK_HEAD_DIM + V_HEAD_DIM,
            dtype=FP8_DTYPE,
            device=device,
        )
        kv_scale = torch.ones(1, dtype=torch.float32, device=device)
        qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=device)
        kv_indptr = torch.arange(
            0, (bs + 1) * kvl, kvl, dtype=torch.int32, device=device
        )

        c = _get_or_build(
            bs, kvl, torch.bfloat16, FP8_DTYPE, qo_indptr, kv_indptr, ns, device, ps, fm
        )
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
        kv_4d = kv.view(bs * kvl, 1, NUM_KV_HEADS, -1)

        # Warmup
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
            q_scale=None,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **c["meta"],
        )
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
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
                q_scale=None,
                kv_scale=kv_scale,
                intra_batch_mode=True,
                **c["meta"],
            )

        g.replay()
        torch.cuda.synchronize()
        print("  PASS: aiter mla_decode_fwd can be captured!", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL (expected): {e}", file=sys.stderr)

    # Test 4: Try with private memory pool
    print("\nTest 4: aiter with private memory pool...", file=sys.stderr)
    try:
        # Create a private pool for graph capture
        pool = torch.cuda.graph_pool_handle()

        # Warmup in pool context
        with torch.cuda.device(device):
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
                q_scale=None,
                kv_scale=kv_scale,
                intra_batch_mode=True,
                **c["meta"],
            )
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
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
                q_scale=None,
                kv_scale=kv_scale,
                intra_batch_mode=True,
                **c["meta"],
            )

        g.replay()
        torch.cuda.synchronize()
        print("  PASS: aiter with private pool works!", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)

    # Test 5: Check if graph capture mode info is available
    print("\nTest 5: Graph capture API check...", file=sys.stderr)
    try:
        import torch._C

        print(
            f"  torch.cuda.is_current_stream_capturing exists: {hasattr(torch.cuda, 'is_current_stream_capturing')}",
            file=sys.stderr,
        )
        print(
            f"  torch.cuda.graph_pool_handle exists: {hasattr(torch.cuda, 'graph_pool_handle')}",
            file=sys.stderr,
        )
        print(f"  ROCm backend: {torch.version.hip is not None}", file=sys.stderr)
        if torch.version.hip:
            print(f"  HIP version: {torch.version.hip}", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("END HIP GRAPH CAPTURE TEST", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    # Run graph capture test once
    if not _tested_graph[0]:
        _test_graph_capture()

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
