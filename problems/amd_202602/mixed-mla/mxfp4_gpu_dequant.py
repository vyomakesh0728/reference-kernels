#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Use aiter's mxfp4_to_f32 for GPU dequant, then quant to fp8 and use aiter."""
import torch
from task import input_t, output_t
NUM_HEADS = 16; NUM_KV_HEADS = 1; QK_HEAD_DIM = 576; V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

FP8_DTYPE = aiter_dtypes.fp8
_cache = {}

def _get_config(bs, kvl):
    """Best config: use mxfp4 for large shapes only."""
    if kvl <= 1024:
        if bs <= 32: return (8, False, 2, True, False)
        if bs <= 64: return (4, False, 2, True, False)
        return (4, False, 2, True, False)
    else:  # kv=8192
        if bs <= 4:  return (32, False, 2, True, False)
        # Try mxfp4 for bs>=32, kv=8k (GPU dequant, save bandwidth)
        if bs <= 32: return (8, True, 1, False, True)   # use mxfp4
        if bs <= 64: return (8, True, 1, False, True)   # use mxfp4
        return (16, True, 1, False, True)                # use mxfp4

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

def _mxfp4_to_fp8(mxfp4_data, mxfp4_scale):
    """GPU dequant mxfp4 -> bf16 -> quant to fp8."""
    total_kv = mxfp4_data.shape[0]
    device = mxfp4_data.device

    # Unpack fp4x2 to f32
    fp4_data_2d = mxfp4_data[:, 0, :]  # (total_kv, 288)
    float_vals = mxfp4_to_f32(fp4_data_2d)  # (total_kv, 576)

    # Convert E8M0 scales to f32 and apply (may be 2D)
    if mxfp4_scale.dim() == 3:
        scale_e8m0 = mxfp4_scale[:, 0, :]  # (total_kv, 144)
    else:
        scale_e8m0 = mxfp4_scale  # (total_kv, 144)
    scale_f32 = e8m0_to_f32(scale_e8m0)  # (total_kv, 144)

    # Apply block scales (block_size=32, 576/32=18 blocks)
    num_blocks = QK_HEAD_DIM // 32  # 18
    float_vals_blocked = float_vals.view(total_kv, num_blocks, 32)
    scale_f32 = scale_f32[:, :num_blocks]  # trim if padded
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)
    bf16_kv = scaled.view(total_kv, QK_HEAD_DIM).to(torch.bfloat16)

    # Quant to fp8
    finfo = torch.finfo(FP8_DTYPE)
    amax = bf16_kv.abs().amax().clamp(min=1e-12)
    fp8_scale = amax / finfo.max
    fp8_kv = (bf16_kv / fp8_scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)

    return fp8_kv, fp8_scale.to(torch.float32).reshape(1)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"]); kvl = int(config["kv_seq_len"])
    ns, use_a8w8, ps, fm, use_mxfp4 = _get_config(bs, kvl)

    if use_mxfp4 and "mxfp4" in kv_data:
        # GPU dequant mxfp4 -> fp8
        mxfp4_data, mxfp4_scale = kv_data["mxfp4"]
        kv_fp8, kv_scale = _mxfp4_to_fp8(mxfp4_data, mxfp4_scale)
    else:
        kv_fp8, kv_scale = kv_data["fp8"]

    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if use_a8w8:
        bkey = ("dq", q.numel())
        if bkey not in _cache:
            _cache[bkey] = (torch.empty_like(q, dtype=FP8_DTYPE), torch.empty(1, dtype=torch.float32, device=q.device))
        qi, qs = _cache[bkey]
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM); qs = None

    c = _get_or_build(bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm)
    mla_decode_fwd(qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"],
                   1, page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
                   logit_cap=0.0, num_kv_splits=ns, q_scale=qs, kv_scale=kv_scale,
                   intra_batch_mode=True, **c["meta"])
    return c["out"]
