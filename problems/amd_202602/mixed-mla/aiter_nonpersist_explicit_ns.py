#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Experimental: non-persistent a8w8 with explicit split tuning.

Goal: reduce persistent metadata overhead while avoiding auto-split bugs.
"""

import torch
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

from aiter import dtypes as aiter_dtypes
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8
_cache = {}


def _pick_splits(bs: int, kvl: int) -> int:
    if kvl <= 1024:
        if bs <= 32:
            return 8
        return 4
    if bs <= 4:
        return 32
    if bs <= 64:
        return 8
    return 16


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    ns = _pick_splits(bs, kvl)

    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    ckey = ("shape", bs, kvl, q.device.index if q.device.index is not None else -1)
    cached = _cache.get(ckey)
    if cached is None:
        tkv = bs * kvl
        cached = {
            "ki": torch.arange(tkv, dtype=torch.int32, device=q.device),
            "kl": (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32),
            "out": torch.empty(
                (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
            ),
        }
        _cache[ckey] = cached

    qkey = ("dq", q.numel(), q.device.index if q.device.index is not None else -1)
    qbuf = _cache.get(qkey)
    if qbuf is None:
        qbuf = (
            torch.empty_like(q, dtype=FP8_DTYPE),
            torch.empty(1, dtype=torch.float32, device=q.device),
        )
        _cache[qkey] = qbuf

    qi, qs = qbuf
    dynamic_per_tensor_quant(qi, q, qs)
    qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)

    mla_decode_fwd(
        qv,
        kv_4d,
        cached["out"],
        qo_indptr,
        kv_indptr,
        cached["ki"],
        cached["kl"],
        1,
        page_size=1,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=False,
    )
    return cached["out"]
