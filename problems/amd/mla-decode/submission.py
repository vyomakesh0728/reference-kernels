#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"attempt": 2, "gpu": "MI355X", "leaderboard": "amd-mixed-mla", "problem": "mixed_mla", "variant": {"family": "anchor", "strategy": "contract_anchor", "variant_name": "aiter_fp8_anchor"}, "variant_index": 0}
import torch
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
PAGE_SIZE = 1
NUM_KV_SPLITS = 32
FP8_DTYPE = aiter_dtypes.fp8


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> dict[str, torch.Tensor]:
    info = get_mla_metadata_info_v1(
        batch_size,
        max_q_len,
        nhead,
        q_dtype,
        kv_dtype,
        is_sparse=False,
        fast_mode=False,
        num_kv_splits=NUM_KV_SPLITS,
        intra_batch_mode=True,
    )
    work = [torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in info]
    (
        work_metadata,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = work
    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=NUM_KV_SPLITS,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )
    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


def _aiter_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
    q_scale: torch.Tensor | None,
    kv_scale: torch.Tensor | None,
) -> torch.Tensor:
    batch_size = int(config["batch_size"])
    nq = int(config["num_heads"])
    nkv = int(config["num_kv_heads"])
    dq = int(config["qk_head_dim"])
    dv = int(config["v_head_dim"])
    q_seq_len = int(config["q_seq_len"])
    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    meta = _make_mla_decode_metadata(
        batch_size,
        q_seq_len,
        nq,
        nkv,
        q.dtype,
        kv_buffer.dtype,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
    )
    out = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q.view(-1, nq, dq),
        kv_buffer_4d,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        q_seq_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return out


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    q_input, q_scale = quantize_fp8(q)
    kv_input, kv_scale = kv_data["fp8"]
    return _aiter_mla_decode(
        q_input,
        kv_input,
        qo_indptr,
        kv_indptr,
        config,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
