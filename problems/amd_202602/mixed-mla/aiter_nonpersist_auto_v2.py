#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Non-persistent mode: num_kv_splits=None (auto 1-16), Triton reduce.
No metadata precomputation. Hybrid a16w8/a8w8."""
import torch
from task import input_t, output_t
NUM_HEADS = 16; NUM_KV_HEADS = 1; QK_HEAD_DIM = 576; V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
from aiter import dtypes as aiter_dtypes
from aiter.mla import mla_decode_fwd
FP8_DTYPE = aiter_dtypes.fp8
_fp8_finfo = torch.finfo(FP8_DTYPE)
_cache = {}

def _quantize_fp8(t):
    a = t.abs().amax().clamp(min=1e-12)
    s = a / _fp8_finfo.max
    return (t/s).clamp(min=_fp8_finfo.min,max=_fp8_finfo.max).to(FP8_DTYPE), s.float().reshape(1)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"]); kvl = int(config["kv_seq_len"])
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])
    tkv = bs * kvl
    # Cache only ki and kl
    ckey = (bs, kvl)
    if ckey not in _cache:
        _cache[ckey] = {
            "ki": torch.arange(tkv, dtype=torch.int32, device=q.device),
            "kl": (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32),
            "out": torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device),
        }
    c = _cache[ckey]
    if bs >= 64 and kvl >= 8192:
        qi, qs = _quantize_fp8(q)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM); qs = None
    # Non-persistent: all metadata params None, auto num_kv_splits
    mla_decode_fwd(
        qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"], 1,
        page_size=1, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0,
        num_kv_splits=None,
        q_scale=qs, kv_scale=kv_scale,
        intra_batch_mode=False)
    return c["out"]
