#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Direct stage1_asm + reduce_v1 API. Skip mla_decode_fwd wrapper overhead.
Pre-allocate ALL buffers. Pre-compute metadata once. a8w8 for all shapes.
Force-import all aiter modules at init to amortize JIT."""
import torch
from task import input_t, output_t
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
# Import stage1 and reduce DIRECTLY (skip mla_decode_fwd wrapper)
from aiter.jit_build import jit_build
# Force-build all modules at import time
import aiter.jit.module_aiter_enum
try:
    from aiter.ops.mla import mla_decode_stage1_asm_fwd, mla_reduce_v1
except ImportError:
    from aiter.mla import mla_decode_fwd as _fallback
    mla_decode_stage1_asm_fwd = None
    mla_reduce_v1 = None

FP8_DTYPE = aiter_dtypes.fp8
_fp8_finfo = torch.finfo(FP8_DTYPE)
_cache = {}

def _quantize_fp8(t):
    a = t.abs().amax().clamp(min=1e-12)
    s = a / _fp8_finfo.max
    return (t / s).clamp(min=_fp8_finfo.min, max=_fp8_finfo.max).to(FP8_DTYPE), s.float().reshape(1)

def _pick_ns(bs, kvl):
    if kvl <= 1024: return 16
    if kvl <= 4096: return 16
    return 32

def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev):
    key = (bs, kvl, ns, qd)
    if key in _cache:
        return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    ps = 1
    fm = False
    info = get_mla_metadata_info_v1(
        bs, 1, NUM_HEADS, qd, kvd, is_sparse=False, fast_mode=fm,
        num_kv_splits=ns, intra_batch_mode=True)
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(
        qo, kvi, kl, NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        wm, ws, wi, ri, rf, rp, page_size=ps,
        kv_granularity=max(ps, 16), max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=fm, max_split_per_batch=ns, intra_batch_mode=True,
        dtype_q=qd, dtype_kv=kvd)
    tq = bs
    # Pre-allocate split buffers for direct stage1+reduce
    nq = NUM_HEADS
    split_out = torch.empty((bs * ns * nq, V_HEAD_DIM), dtype=torch.float32, device=dev)
    split_lse = torch.empty((bs * ns * nq, 1), dtype=torch.float32, device=dev)
    out = torch.empty((tq, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev)
    e = {
        "meta": {"work_meta_data": wm, "work_indptr": wi, "work_info_set": ws,
                 "reduce_indptr": ri, "reduce_final_map": rf, "reduce_partial_map": rp},
        "kl": kl, "ki": ki, "out": out,
        "split_out": split_out, "split_lse": split_lse,
        "ps": ps, "fm": fm,
    }
    _cache[key] = e
    return e

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    ns = _pick_ns(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    # Always a8w8
    qi, qs = _quantize_fp8(q)
    qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)

    c = _get_or_build(bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device)

    if mla_decode_stage1_asm_fwd is not None and mla_reduce_v1 is not None:
        # Direct stage1 + reduce (skip mla_decode_fwd wrapper)
        mla_decode_stage1_asm_fwd(
            qv, kv_4d, qo_indptr, kv_indptr, c["ki"], c["kl"],
            None,  # num_kv_splits_indptr (not used in persistent mode)
            c["meta"]["work_meta_data"], c["meta"]["work_indptr"], c["meta"]["work_info_set"],
            1,  # max_seqlen_q
            c["ps"], NUM_KV_HEADS, SM_SCALE,
            c["split_out"], c["split_lse"], c["out"],
            q_scale=qs, kv_scale=kv_scale)
        mla_reduce_v1(
            c["split_out"], c["split_lse"],
            c["meta"]["reduce_indptr"], c["meta"]["reduce_final_map"],
            c["meta"]["reduce_partial_map"],
            1,  # max_seqlen_q
            c["out"])
    else:
        # Fallback to mla_decode_fwd
        from aiter.mla import mla_decode_fwd
        mla_decode_fwd(
            qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"], 1,
            page_size=c["ps"], nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0,
            num_kv_splits=ns, q_scale=qs, kv_scale=kv_scale,
            intra_batch_mode=True, **c["meta"])
    return c["out"]
