#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""HIP Graph-based MLA decode with aggressive pre-allocation.

Strategy: Capture the ENTIRE pipeline (quant + mla_decode_fwd) in a graph
for each unique (bs, kvl) combination. On replay, just copy input data
and replay the graph.

Key insight: The harness uses fixed shapes for benchmarking. If we can
capture a graph per shape during warmup, subsequent calls are pure replay.

Expected overhead reduction:
- Current: 293µs wall time (24µs GPU + 269µs dispatch)
- With graph: ~30-40µs wall time (24µs GPU + ~10µs graph replay overhead)

Fallback: If graph capture fails, fall back to standard aiter path.
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

# Caches for buffers and graphs
_buffer_cache = {}
_graph_cache = {}
_graph_inputs = {}
_graph_outputs = {}
_graph_failed = set()
_first_call = [True]


def _get_config(bs, kvl):
    if kvl <= 1024:
        if bs <= 32:
            return (8, False, 2, True)
        if bs <= 64:
            return (4, False, 2, True)
        return (4, False, 2, True)
    return (32, True, 1, False)


def _build_buffers(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    """Build all pre-allocated buffers for a given shape."""
    key = (bs, kvl, ns, str(qd), ps, fm)
    if key in _buffer_cache:
        return _buffer_cache[key]

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

    buffers = {
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
    _buffer_cache[key] = buffers
    return buffers


def _try_capture_graph(
    bs, kvl, q, kv_fp8, kv_scale, qo_indptr, kv_indptr, ns, use_a8w8, ps, fm, device
):
    """Try to capture the MLA pipeline as a HIP graph."""
    graph_key = (bs, kvl)

    if graph_key in _graph_failed:
        return None

    if graph_key in _graph_cache:
        return _graph_cache[graph_key]

    print(f"[HIP-GRAPH] Attempting capture for bs={bs}, kvl={kvl}...", file=sys.stderr)

    try:
        # Pre-allocate input buffers that will be reused
        q_buf = torch.empty_like(q)
        kv_buf = torch.empty_like(kv_fp8)
        kv_scale_buf = torch.empty_like(kv_scale)

        if use_a8w8:
            qi_buf = torch.empty_like(q, dtype=FP8_DTYPE)
            qs_buf = torch.empty(1, dtype=torch.float32, device=device)
        else:
            qi_buf = None
            qs_buf = None

        # Copy inputs to buffers
        q_buf.copy_(q)
        kv_buf.copy_(kv_fp8)
        kv_scale_buf.copy_(kv_scale)

        kv_4d = kv_buf.view(kv_buf.shape[0], 1, NUM_KV_HEADS, kv_buf.shape[-1])

        if use_a8w8:
            qd = FP8_DTYPE
        else:
            qd = torch.bfloat16

        buffers = _build_buffers(
            bs, kvl, qd, kv_buf.dtype, qo_indptr, kv_indptr, ns, device, ps, fm
        )

        # Warmup call to ensure JIT compilation is done
        if use_a8w8:
            dynamic_per_tensor_quant(qi_buf, q_buf, qs_buf)
            qv = qi_buf.view(-1, NUM_HEADS, QK_HEAD_DIM)
        else:
            qv = q_buf.view(-1, NUM_HEADS, QK_HEAD_DIM)

        mla_decode_fwd(
            qv,
            kv_4d,
            buffers["out"],
            qo_indptr,
            kv_indptr,
            buffers["ki"],
            buffers["kl"],
            1,
            page_size=ps,
            nhead_kv=NUM_KV_HEADS,
            sm_scale=SM_SCALE,
            logit_cap=0.0,
            num_kv_splits=ns,
            q_scale=qs_buf if use_a8w8 else None,
            kv_scale=kv_scale_buf,
            intra_batch_mode=True,
            **buffers["meta"],
        )
        torch.cuda.synchronize()

        # Try to capture
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            with torch.cuda.graph(g, stream=stream):
                if use_a8w8:
                    dynamic_per_tensor_quant(qi_buf, q_buf, qs_buf)
                    qv = qi_buf.view(-1, NUM_HEADS, QK_HEAD_DIM)
                else:
                    qv = q_buf.view(-1, NUM_HEADS, QK_HEAD_DIM)

                mla_decode_fwd(
                    qv,
                    kv_4d,
                    buffers["out"],
                    qo_indptr,
                    kv_indptr,
                    buffers["ki"],
                    buffers["kl"],
                    1,
                    page_size=ps,
                    nhead_kv=NUM_KV_HEADS,
                    sm_scale=SM_SCALE,
                    logit_cap=0.0,
                    num_kv_splits=ns,
                    q_scale=qs_buf if use_a8w8 else None,
                    kv_scale=kv_scale_buf,
                    intra_batch_mode=True,
                    **buffers["meta"],
                )

        torch.cuda.synchronize()

        # Test replay
        g.replay()
        torch.cuda.synchronize()

        print(
            f"[HIP-GRAPH] SUCCESS! Graph captured for bs={bs}, kvl={kvl}",
            file=sys.stderr,
        )

        # Store graph and associated buffers
        _graph_cache[graph_key] = g
        _graph_inputs[graph_key] = {
            "q": q_buf,
            "kv": kv_buf,
            "kv_scale": kv_scale_buf,
            "qi": qi_buf,
            "qs": qs_buf,
            "use_a8w8": use_a8w8,
        }
        _graph_outputs[graph_key] = buffers["out"]

        return g

    except Exception as e:
        print(f"[HIP-GRAPH] FAILED for bs={bs}, kvl={kvl}: {e}", file=sys.stderr)
        _graph_failed.add(graph_key)
        return None


def _run_with_graph(graph_key, q, kv_fp8, kv_scale):
    """Run MLA using a captured graph."""
    inputs = _graph_inputs[graph_key]
    graph = _graph_cache[graph_key]

    # Copy inputs to graph buffers
    inputs["q"].copy_(q)
    inputs["kv"].copy_(kv_fp8)
    inputs["kv_scale"].copy_(kv_scale)

    # Replay graph
    graph.replay()

    return _graph_outputs[graph_key]


def _run_standard(
    bs, kvl, q, kv_fp8, kv_scale, qo_indptr, kv_indptr, ns, use_a8w8, ps, fm, device
):
    """Run MLA using standard aiter path."""
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if use_a8w8:
        bkey = ("dq", q.numel())
        if bkey not in _buffer_cache:
            _buffer_cache[bkey] = (
                torch.empty_like(q, dtype=FP8_DTYPE),
                torch.empty(1, dtype=torch.float32, device=device),
            )
        qi, qs = _buffer_cache[bkey]
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
        qd = FP8_DTYPE
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
        qs = None
        qd = torch.bfloat16

    buffers = _build_buffers(
        bs, kvl, qd, kv_fp8.dtype, qo_indptr, kv_indptr, ns, device, ps, fm
    )

    mla_decode_fwd(
        qv,
        kv_4d,
        buffers["out"],
        qo_indptr,
        kv_indptr,
        buffers["ki"],
        buffers["kl"],
        1,
        page_size=ps,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **buffers["meta"],
    )
    return buffers["out"]


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    device = q.device

    ns, use_a8w8, ps, fm = _get_config(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]

    graph_key = (bs, kvl)

    # First call: log status
    if _first_call[0]:
        _first_call[0] = False
        print("[HIP-GRAPH] MLA decode with graph capture enabled", file=sys.stderr)
        print(
            f"[HIP-GRAPH] ROCm backend: {torch.version.hip is not None}",
            file=sys.stderr,
        )
        if torch.version.hip:
            print(f"[HIP-GRAPH] HIP version: {torch.version.hip}", file=sys.stderr)

    # Try graph path
    if graph_key in _graph_cache:
        return _run_with_graph(graph_key, q, kv_fp8, kv_scale)

    # Try to capture graph (only if not already failed)
    if graph_key not in _graph_failed:
        graph = _try_capture_graph(
            bs,
            kvl,
            q,
            kv_fp8,
            kv_scale,
            qo_indptr,
            kv_indptr,
            ns,
            use_a8w8,
            ps,
            fm,
            device,
        )
        if graph is not None:
            return _run_with_graph(graph_key, q, kv_fp8, kv_scale)

    # Fallback to standard path
    return _run_standard(
        bs, kvl, q, kv_fp8, kv_scale, qo_indptr, kv_indptr, ns, use_a8w8, ps, fm, device
    )
