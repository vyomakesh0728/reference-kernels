#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Dispatch overhead analysis using only wall time."""

import torch
import sys
import time
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
_analysis_done = [False]


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


def _run_analysis(q, kv_fp8, kv_scale, qo_indptr, kv_indptr, bs, kvl):
    if _analysis_done[0]:
        return
    _analysis_done[0] = True

    device = q.device
    ns, use_a8w8, ps, fm = _get_config(bs, kvl)

    print("=" * 60, file=sys.stderr)
    print("DISPATCH OVERHEAD ANALYSIS", file=sys.stderr)
    print(f"Shape: bs={bs}, kvl={kvl}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])
    qi = torch.empty_like(q, dtype=FP8_DTYPE)
    qs = torch.empty(1, dtype=torch.float32, device=device)

    if use_a8w8:
        qd = FP8_DTYPE
    else:
        qd = torch.bfloat16
    c = _get_or_build(
        bs, kvl, qd, kv_fp8.dtype, qo_indptr, kv_indptr, ns, device, ps, fm
    )

    # Warmup
    for _ in range(5):
        if use_a8w8:
            dynamic_per_tensor_quant(qi, q, qs)
            qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
        else:
            qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
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
            q_scale=qs if use_a8w8 else None,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **c["meta"],
        )
    torch.cuda.synchronize()

    n_iters = 20

    if use_a8w8:
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)

    # Wall time with sync
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
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
            q_scale=qs if use_a8w8 else None,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **c["meta"],
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wall_sync = (t1 - t0) * 1e6 / n_iters

    # Wall time no sync (dispatch only)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
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
            q_scale=qs if use_a8w8 else None,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            **c["meta"],
        )
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    dispatch = (t1 - t0) * 1e6 / n_iters

    print(f"Wall time (sync): {wall_sync:.1f} us", file=sys.stderr)
    print(f"Dispatch overhead: {dispatch:.1f} us", file=sys.stderr)
    print(f"GPU time (est): {wall_sync - dispatch:.1f} us", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]

    if not _analysis_done[0]:
        _run_analysis(q, kv_fp8, kv_scale, qo_indptr, kv_indptr, bs, kvl)

    ns, use_a8w8, ps, fm = _get_config(bs, kvl)
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
