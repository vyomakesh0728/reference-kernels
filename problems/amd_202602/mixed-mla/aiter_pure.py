#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Pure aiter - no HIP compilation overhead."""

from __future__ import annotations
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
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8
_cache = {}


def _get_ns(bs: int, kvl: int) -> int:
    if kvl <= 1024:
        return 8 if bs <= 32 else 4
    return 16 if bs <= 64 else 32


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    ns = _get_ns(bs, kvl)

    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])
    dev = q.device

    bkey = ("dq", q.numel())
    if bkey not in _cache:
        _cache[bkey] = (
            torch.empty_like(q, dtype=FP8_DTYPE),
            torch.empty(1, dtype=torch.float32, device=dev),
        )
    qi, qs = _cache[bkey]
    dynamic_per_tensor_quant(qi, q, qs)
    qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)

    key = (bs, kvl, ns, qv.dtype, 1, False)
    if key not in _cache:
        tkv = bs * kvl
        kl = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        ki = torch.arange(tkv, dtype=torch.int32, device=dev)
        info = get_mla_metadata_info_v1(
            bs,
            1,
            NUM_HEADS,
            qv.dtype,
            kv_fp8.dtype,
            is_sparse=False,
            fast_mode=False,
            num_kv_splits=ns,
            intra_batch_mode=True,
        )
        w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
        wm, wi, ws, ri, rf, rp = w
        get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
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
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=ns,
            intra_batch_mode=True,
            dtype_q=qv.dtype,
            dtype_kv=kv_fp8.dtype,
        )
        _cache[key] = {
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

    c = _cache[key]
    mla_decode_fwd(
        qv,
        kv_4d,
        c["out"],
        qo_indptr,
        kv_indptr,
        c["ki"],
        c["kl"],
        1,
        page_size=1,
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
