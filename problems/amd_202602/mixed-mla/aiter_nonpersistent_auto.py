#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Non-persistent mode: let aiter auto-pick num_kv_splits + use Triton reduce.
Skip all metadata precomputation. Minimal Python overhead.
a8w8 for all shapes."""
import torch
from task import input_t, output_t
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
from aiter import dtypes as aiter_dtypes
from aiter.mla import mla_decode_fwd
FP8_DTYPE = aiter_dtypes.fp8
_fp8_finfo = torch.finfo(FP8_DTYPE)
_cache = {}

def _quantize_fp8(t):
    a = t.abs().amax().clamp(min=1e-12)
    s = a / _fp8_finfo.max
    return (t / s).clamp(min=_fp8_finfo.min, max=_fp8_finfo.max).to(FP8_DTYPE), s.float().reshape(1)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])
    tkv = bs * kvl
    ki = _cache.get(("ki", tkv))
    if ki is None:
        ki = torch.arange(tkv, dtype=torch.int32, device=q.device)
        _cache[("ki", tkv)] = ki
    kl = _cache.get(("kl", bs, kvl))
    if kl is None:
        kl = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        _cache[("kl", bs, kvl)] = kl
    out = _cache.get(("out", bs))
    if out is None:
        out = torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)
        _cache[("out", bs)] = out
    qi, qs = _quantize_fp8(q)
    qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    # Non-persistent mode: NO metadata, NO num_kv_splits, aiter picks automatically
    mla_decode_fwd(
        qv, kv_4d, out, qo_indptr, kv_indptr, ki, kl, 1,
        page_size=1, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0,
        num_kv_splits=0,  # 0 = auto
        q_scale=qs, kv_scale=kv_scale,
        intra_batch_mode=False)  # non-persistent = no intra_batch_mode
    return out
