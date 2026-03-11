#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"attempt": 34, "generator": {"kind": "llm", "model": "(codex default)", "parallel_agents": 3, "provider": "codex_cli", "use_plan": true}, "gpu": "MI355X", "leaderboard": "amd-mixed-mla", "policy_profile": {"family": "kernel_explore", "focus": "reduce small-batch overhead and keep q_seq_len=1 path lean", "name": "latency_small_block", "trigger_signals": ["throughput_shift", "latency_repair"]}, "problem": "mixed_mla", "variant": {"BLOCK_DQ": 64, "BLOCK_DV": 64, "BLOCK_N": 64, "NUM_STAGES": 2, "NUM_WARPS": 4, "USE_FP8_INPUTS": true, "family": "kernel_explore", "strategy": "fp8_decode", "variant_name": "fp8_decode_b64_v64"}, "variant_index": 2}

import torch

try:
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd
    AITER_AVAILABLE = True
except Exception:
    aiter_dtypes = None
    get_mla_metadata_info_v1 = None
    get_mla_metadata_v1 = None
    mla_decode_fwd = None
    AITER_AVAILABLE = False

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

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

TRITON_CONFIG = {
    "BLOCK_N": 128,
    "BLOCK_DQ": 64,
    "BLOCK_DV": 128,
    "NUM_WARPS": 8,
    "NUM_STAGES": 2,
}


def _resolve_fp8_dtype():
    if AITER_AVAILABLE:
        return aiter_dtypes.fp8
    for name in ("float8_e4m3fnuz", "float8_e4m3fn"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            return dtype
    return None


FP8_DTYPE = _resolve_fp8_dtype()


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _normalize_scalar_scale(scale: torch.Tensor | None) -> torch.Tensor | None:
    if scale is None or scale.numel() != 1:
        return None
    return scale.to(torch.float32).reshape(1).contiguous()


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
    work = [torch.empty(shape, dtype=dtype, device=qo_indptr.device) for shape, dtype in info]
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
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device=kv_buffer.device)
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
    out = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device=q.device)
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
        sm_scale=float(config.get("sm_scale", SM_SCALE)),
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return out


def _torch_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    num_heads = int(config.get("num_heads", NUM_HEADS))
    v_head_dim = int(config.get("v_head_dim", V_HEAD_DIM))
    sm_scale = float(config.get("sm_scale", SM_SCALE))
    out = torch.empty((q.shape[0], num_heads, v_head_dim), dtype=torch.bfloat16, device=q.device)

    batch_size = qo_indptr.numel() - 1
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        q_slice = q[q_start:q_end].to(torch.float32).transpose(0, 1)
        kv_slice = kv_buffer[kv_start:kv_end, 0].to(torch.float32)
        scores = torch.matmul(q_slice, kv_slice.transpose(0, 1)) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out_slice = torch.matmul(probs, kv_slice[:, :v_head_dim]).transpose(0, 1)
        out[q_start:q_end] = out_slice.to(torch.bfloat16)

    return out


if TRITON_AVAILABLE:

    @triton.jit
    def _mla_decode_fp8_kernel(
        q_ptr,
        kv_ptr,
        kv_indptr_ptr,
        q_scale_ptr,
        kv_scale_ptr,
        out_ptr,
        total_q,
        sm_scale,
        NUM_HEADS_CONST: tl.constexpr,
        QK_HEAD_DIM_CONST: tl.constexpr,
        V_HEAD_DIM_CONST: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DQ: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_v = tl.program_id(1)

        q_idx = pid_row // NUM_HEADS_CONST
        if q_idx >= total_q:
            return

        kv_start = tl.load(kv_indptr_ptr + q_idx)
        kv_end = tl.load(kv_indptr_ptr + q_idx + 1)

        q_scale = tl.load(q_scale_ptr).to(tl.float32)
        kv_scale = tl.load(kv_scale_ptr).to(tl.float32)
        score_scale = q_scale * kv_scale * sm_scale

        v_offsets = pid_v * BLOCK_DV + tl.arange(0, BLOCK_DV)
        q_base = q_ptr + pid_row * QK_HEAD_DIM_CONST

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

        for block_start in tl.range(0, kv_end - kv_start, BLOCK_N):
            n_offsets = kv_start + block_start + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < kv_end
            scores = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for d_start in tl.range(0, QK_HEAD_DIM_CONST, BLOCK_DQ):
                d_offsets = d_start + tl.arange(0, BLOCK_DQ)
                mask_d = d_offsets < QK_HEAD_DIM_CONST

                q_vals = tl.load(
                    q_base + d_offsets,
                    mask=mask_d,
                    other=0.0,
                ).to(tl.float32)
                k_ptrs = kv_ptr + n_offsets[:, None] * QK_HEAD_DIM_CONST + d_offsets[None, :]
                k_vals = tl.load(
                    k_ptrs,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                scores += tl.sum(k_vals * q_vals[None, :], axis=1)

            scores = tl.where(mask_n, scores * score_scale, -float("inf"))
            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = alpha * l_i + tl.sum(p, axis=0)

            v_ptrs = kv_ptr + n_offsets[:, None] * QK_HEAD_DIM_CONST + v_offsets[None, :]
            v_vals = tl.load(
                v_ptrs,
                mask=mask_n[:, None] & (v_offsets[None, :] < V_HEAD_DIM_CONST),
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
            m_i = m_new

        out_ptrs = out_ptr + pid_row * V_HEAD_DIM_CONST + v_offsets
        out_vals = acc * kv_scale / l_i
        tl.store(out_ptrs, out_vals.to(tl.bfloat16), mask=v_offsets < V_HEAD_DIM_CONST)


def _triton_decode_fp8(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    q_rows = q.contiguous().reshape(-1, QK_HEAD_DIM)
    kv_rows = kv_buffer.contiguous().reshape(-1, QK_HEAD_DIM)
    total_q = q.shape[0]
    out = torch.empty((total_q, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    out_rows = out.view(-1, V_HEAD_DIM)

    grid = (q_rows.shape[0], triton.cdiv(V_HEAD_DIM, TRITON_CONFIG["BLOCK_DV"]))
    _mla_decode_fp8_kernel[grid](
        q_rows,
        kv_rows,
        kv_indptr,
        q_scale,
        kv_scale,
        out_rows,
        total_q,
        float(config.get("sm_scale", SM_SCALE)),
        NUM_HEADS_CONST=NUM_HEADS,
        QK_HEAD_DIM_CONST=QK_HEAD_DIM,
        V_HEAD_DIM_CONST=V_HEAD_DIM,
        BLOCK_N=TRITON_CONFIG["BLOCK_N"],
        BLOCK_DQ=TRITON_CONFIG["BLOCK_DQ"],
        BLOCK_DV=TRITON_CONFIG["BLOCK_DV"],
        num_warps=TRITON_CONFIG["NUM_WARPS"],
        num_stages=TRITON_CONFIG["NUM_STAGES"],
    )
    return out


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    if FP8_DTYPE is not None:
        q_fp8, q_scale = quantize_fp8(q.contiguous())
        kv_fp8, kv_scale = kv_data["fp8"]

        if AITER_AVAILABLE:
            try:
                return _aiter_mla_decode(
                    q_fp8,
                    kv_fp8,
                    qo_indptr,
                    kv_indptr,
                    config,
                    q_scale=q_scale,
                    kv_scale=kv_scale,
                )
            except Exception:
                pass

        can_use_triton = (
            TRITON_AVAILABLE
            and int(config.get("q_seq_len", 1)) == 1
            and int(config.get("batch_size", q.shape[0])) == q.shape[0]
            and int(config.get("num_heads", NUM_HEADS)) == NUM_HEADS
            and int(config.get("num_kv_heads", NUM_KV_HEADS)) == NUM_KV_HEADS
            and int(config.get("qk_head_dim", QK_HEAD_DIM)) == QK_HEAD_DIM
            and int(config.get("v_head_dim", V_HEAD_DIM)) == V_HEAD_DIM
            and qo_indptr.numel() == q.shape[0] + 1
            and kv_indptr.numel() == q.shape[0] + 1
            and int(qo_indptr[0].item()) == 0
            and int(qo_indptr[-1].item()) == q.shape[0]
            and q.shape[1] == NUM_HEADS
            and q.shape[2] == QK_HEAD_DIM
            and kv_fp8.shape[1] == NUM_KV_HEADS
            and kv_fp8.shape[2] == QK_HEAD_DIM
        )
        if can_use_triton:
            q_scale_scalar = _normalize_scalar_scale(q_scale)
            kv_scale_scalar = _normalize_scalar_scale(kv_scale)
            if q_scale_scalar is not None and kv_scale_scalar is not None:
                try:
                    return _triton_decode_fp8(
                        q_fp8,
                        kv_fp8,
                        kv_indptr,
                        q_scale_scalar,
                        kv_scale_scalar,
                        config,
                    )
                except Exception:
                    pass

    return _torch_decode(q, kv_data["bf16"], qo_indptr, kv_indptr, config)
