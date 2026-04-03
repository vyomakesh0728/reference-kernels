#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""a16w8 for ALL shapes (zero fp8 quant overhead), low ns, ps=2, fast_mode=True."""
import torch
from task import input_t, output_t
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
_cache = {}
def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = (bs, kvl, ns, qd, ps, fm)
    if key in _cache: return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(bs, 1, NUM_HEADS, qd, kvd, is_sparse=False, fast_mode=fm, num_kv_splits=ns, intra_batch_mode=True)
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(qo, kvi, kl, NUM_HEADS//NUM_KV_HEADS, NUM_KV_HEADS, True, wm, ws, wi, ri, rf, rp, page_size=ps, kv_granularity=max(ps,16), max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=fm, max_split_per_batch=ns, intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)
    e = {"meta": {"work_meta_data":wm,"work_indptr":wi,"work_info_set":ws,"reduce_indptr":ri,"reduce_final_map":rf,"reduce_partial_map":rp}, "kl":kl, "ki":ki, "out":torch.empty((bs,NUM_HEADS,V_HEAD_DIM),dtype=torch.bfloat16,device=dev)}
    _cache[key] = e
    return e
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"]); kvl = int(config["kv_seq_len"])
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])
    qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
    # a16w8 always, minimal splits
    ps, fm = 2, True
    if kvl <= 1024:
        ns = 4
    else:
        ns = 8  # only 8 splits for 8k tokens = 1024 tokens per split
    c = _get_or_build(bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm)
    mla_decode_fwd(qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"], 1, page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0, num_kv_splits=ns, q_scale=None, kv_scale=kv_scale, intra_batch_mode=True, **c["meta"])
    return c["out"]
