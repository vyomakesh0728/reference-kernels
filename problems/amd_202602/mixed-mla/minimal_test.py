#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Minimal working test - just call aiter directly."""

import torch
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8 = aiter_dtypes.fp8
_c = {}


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kv = int(config["kv_seq_len"])
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, 1, kv_fp8.shape[-1])
    dev = q.device

    k = ("q", q.numel())
    if k not in _c:
        _c[k] = (
            torch.empty_like(q, dtype=FP8),
            torch.empty(1, dtype=torch.float32, device=dev),
        )
    qi, qs = _c[k]
    dynamic_per_tensor_quant(qi, q, qs)
    qv = qi.view(-1, 16, 576)

    ns = 8
    ps = 1
    key = (bs, kv)
    if key not in _c:
        kl = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        ki = torch.arange(bs * kv, dtype=torch.int32, device=dev)
        info = get_mla_metadata_info_v1(
            bs, 1, 16, qv.dtype, kv_fp8.dtype, False, False, ns, True
        )
        w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
        get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kl,
            16,
            1,
            True,
            w[0],
            w[2],
            w[1],
            w[3],
            w[4],
            w[5],
            page_size=ps,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=ns,
            intra_batch_mode=True,
            dtype_q=qv.dtype,
            dtype_kv=kv_fp8.dtype,
        )
        _c[key] = {
            "m": {
                "work_meta_data": w[0],
                "work_indptr": w[1],
                "work_info_set": w[2],
                "reduce_indptr": w[3],
                "reduce_final_map": w[4],
                "reduce_partial_map": w[5],
            },
            "kl": kl,
            "ki": ki,
            "o": torch.empty((bs, 16, 512), dtype=torch.bfloat16, device=dev),
        }
    c = _c[key]
    mla_decode_fwd(
        qv,
        kv_4d,
        c["o"],
        qo_indptr,
        kv_indptr,
        c["ki"],
        c["kl"],
        1,
        page_size=ps,
        nhead_kv=1,
        sm_scale=0.04167,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **c["m"],
    )
    return c["o"]
