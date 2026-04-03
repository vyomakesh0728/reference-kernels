#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Deep probe of torch.ops.aiter namespace to find .default accessors."""

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
_probed = [False]


def _deep_probe():
    """Deep probe of torch.ops.aiter namespace."""
    if _probed[0]:
        return
    _probed[0] = True

    print("=== DEEP PROBE START ===", file=sys.stderr)

    # List all ops in torch.ops.aiter
    if hasattr(torch.ops, "aiter"):
        aiter_ns = torch.ops.aiter
        ops = [n for n in dir(aiter_ns) if not n.startswith("_")]
        print(f"torch.ops.aiter has {len(ops)} ops:", file=sys.stderr)
        for op_name in sorted(ops):
            op = getattr(aiter_ns, op_name)
            has_default = hasattr(op, "default")
            print(
                f"  {op_name}: type={type(op).__name__}, .default={has_default}",
                file=sys.stderr,
            )
            if has_default:
                print(
                    f"    -> torch.ops.aiter.{op_name}.default AVAILABLE!",
                    file=sys.stderr,
                )

    # Check if the low-level ops are registered
    print("\n=== Checking for specific MLA ops ===", file=sys.stderr)

    # Try to import and check the actual C++ bindings
    try:
        from aiter.ops import mla as mla_ops

        print(f"aiter.ops.mla module: {dir(mla_ops)}", file=sys.stderr)
    except Exception as e:
        print(f"aiter.ops.mla import failed: {e}", file=sys.stderr)

    # Check the jit modules
    try:
        import aiter.jit.module_mla_asm as mla_asm

        print(f"module_mla_asm: {dir(mla_asm)}", file=sys.stderr)
        if hasattr(mla_asm, "mla_decode_stage1_asm_fwd"):
            func = mla_asm.mla_decode_stage1_asm_fwd
            print(f"mla_decode_stage1_asm_fwd type: {type(func)}", file=sys.stderr)
    except Exception as e:
        print(f"module_mla_asm check failed: {e}", file=sys.stderr)

    # Check quant ops
    print("\n=== Checking quant ops ===", file=sys.stderr)
    try:
        from aiter.ops import quant as quant_ops

        print(
            f"aiter.ops.quant module: {[n for n in dir(quant_ops) if not n.startswith('_')]}",
            file=sys.stderr,
        )

        # Check if dynamic_per_tensor_quant is in torch.ops
        if hasattr(torch.ops, "aiter"):
            quant_related = [n for n in dir(torch.ops.aiter) if "quant" in n.lower()]
            print(f"torch.ops.aiter quant ops: {quant_related}", file=sys.stderr)
    except Exception as e:
        print(f"quant ops check failed: {e}", file=sys.stderr)

    print("=== DEEP PROBE END ===", file=sys.stderr)


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

    if not _probed[0]:
        _deep_probe()

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
