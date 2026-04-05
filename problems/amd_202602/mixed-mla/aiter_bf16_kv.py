#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""aiter MLA with bf16 KV (no quantization overhead).

The leaderboard hint: "noobmaster69_og: submission_bf16kv_prejit.py"
suggests using bf16 KV + pre-JIT.

Key: bf16 Q + bf16 KV = a16w16, no quant overhead at all.
"""

import torch
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

_cache = {}


def _get_config(bs, kvl):
    """Tuned config for bf16 KV."""
    if kvl <= 1024:
        if bs <= 32:
            return (8, 2, True)
        if bs <= 64:
            return (4, 2, True)
        return (4, 2, True)
    else:
        if bs <= 4:
            return (32, 1, False)
        if bs <= 32:
            return (16, 1, False)
        if bs <= 64:
            return (16, 1, False)
        return (32, 1, False)


def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = ("bf16", bs, kvl, ns, qd, kvd, ps, fm)
    if key in _cache:
        return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(
        bs, 1, NUM_HEADS, qd, kvd,
        is_sparse=False, fast_mode=fm, num_kv_splits=ns, intra_batch_mode=True,
    )
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(
        qo, kvi, kl, NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        wm, ws, wi, ri, rf, rp,
        page_size=ps, kv_granularity=max(ps, 16), max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=fm, max_split_per_batch=ns, intra_batch_mode=True,
        dtype_q=qd, dtype_kv=kvd,
    )
    e = {
        "meta": {
            "work_meta_data": wm, "work_indptr": wi, "work_info_set": ws,
            "reduce_indptr": ri, "reduce_final_map": rf, "reduce_partial_map": rp,
        },
        "kl": kl, "ki": ki,
        "out": torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev),
    }
    _cache[key] = e
    return e


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    # Use bf16 KV directly - no quant/dequant
    kv_bf16 = kv_data["bf16"]
    kv_4d = kv_bf16.view(kv_bf16.shape[0], 1, NUM_KV_HEADS, kv_bf16.shape[-1])

    # bf16 Q (no quantization)
    qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)

    ns, ps, fm = _get_config(bs, kvl)

    c = _get_or_build(
        bs, kvl, qv.dtype, kv_bf16.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm
    )
    mla_decode_fwd(
        qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"], 1,
        page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0,
        num_kv_splits=ns, q_scale=None, kv_scale=None, intra_batch_mode=True,
        **c["meta"],
    )
    return c["out"]
