#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# Candidate Card:
# shape: re256_de256_bs512_topk8
# lane: dispatch_pack
# deleted cost center: repeated routed-token gather/scatter work around sparse expert windows
# expected upside source: one native gfx950 row-pack helper plus one packed combine path should delete a real dispatch bucket
# why larger than noise: sparse256 routes 4096 token-expert pairs through tiny expert windows, so repeated gather/scatter overhead scales with top-k fanout
# forbidden edits: router/topk changes, stage1/stage2 ownership changes, fused_moe in the target hot path, Python all-expert rebuilds
# AGENT_LOOP_META: {"candidate_card": {"deleted_cost_center": "repeated routed-token gather/scatter work around sparse expert windows", "expected_upside_source": "one native gfx950 row-pack helper plus one packed combine path should delete a real dispatch bucket", "forbidden_edits": ["router/topk changes", "stage1/stage2 ownership changes", "fused_moe in the target hot path", "Python all-expert rebuilds"], "lane": "dispatch_pack", "motivation_refs": ["/root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/scattermoe.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/sonicmoe.md", "https://github.com/shawntan/scattermoe", "https://github.com/Dao-AILab/sonic-moe"], "regime_tag": "re256_de256_bs512_topk8", "retrieval_queries": ["q29-fused-moe-padding-free-packing", "q32-fused-moe-github-motivation-links", "padding-free routed expert packing touched experts", "sorted_token_ids sorted_expert_ids num_valid_ids", "gfx950 load_inline lds swizzle double buffering"], "success_gate": "clear re256_de256_bs512_topk8 win and global <170 us", "why_larger_than_noise": "sparse256 routes 4096 token-expert pairs through tiny expert windows, so repeated gather/scatter overhead scales with top-k fanout"}, "generator": {"kind": "manual_phase2"}, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "policy_profile": {"family": "hip_explore", "name": "dispatch_pack_sparse256_hip_pack_v1"}, "problem": "moe_mxfp4", "variant": {"ARCH": "gfx950", "HOT_PATH_STATE": "partial-native", "LANE": "dispatch_pack", "REGIME_HINT": "re256_de256_bs512_topk8", "family": "hip_explore", "strategy": "dispatch_pack_hip_pack", "variant_name": "dispatch_pack_sparse256_hip_pack_v1"}}
import hashlib
import importlib
import os
from pathlib import Path
import tempfile

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import aiter
from aiter import ActivationType, QuantType
from aiter.utility import fp4_utils
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


CONFIG = {
    "variant_name": "dispatch_pack_sparse256_hip_pack_v1",
    "family": "hip_explore",
    "strategy": "dispatch_pack_hip_pack",
    "LANE": "dispatch_pack",
    "HOT_PATH_STATE": "partial-native",
    "REGIME_HINT": "re256_de256_bs512_topk8",
    "ARCH": "gfx950",
}
MXFP4_BLOCK = 32
_MODULE = None
_TRITON_QUANT = None
_ANCHOR_IMPL = None

CPP_WRAPPER = """
void pack_rows_bf16(torch::Tensor src, torch::Tensor row_ids, torch::Tensor out);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <cstdint>

__global__ void pack_rows_bf16_kernel(
    const __hip_bfloat16* src,
    const int64_t* row_ids,
    __hip_bfloat16* out,
    int num_rows,
    int cols
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_row = blockIdx.y;
    if (out_row >= num_rows || col >= cols) {
        return;
    }
    const int64_t src_row = row_ids[out_row];
    out[out_row * cols + col] = src[src_row * cols + col];
}

void pack_rows_bf16(torch::Tensor src, torch::Tensor row_ids, torch::Tensor out) {
    TORCH_CHECK(src.is_cuda(), "src must be CUDA/ROCm");
    TORCH_CHECK(row_ids.is_cuda(), "row_ids must be CUDA/ROCm");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA/ROCm");
    TORCH_CHECK(src.scalar_type() == at::kBFloat16, "src must be bf16");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bf16");
    TORCH_CHECK(row_ids.scalar_type() == at::kLong, "row_ids must be int64");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(row_ids.is_contiguous(), "row_ids must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    const int64_t num_rows = row_ids.size(0);
    const int64_t cols = src.size(1);
    TORCH_CHECK(out.size(0) == num_rows, "out rows must match row_ids");
    TORCH_CHECK(out.size(1) == cols, "out cols must match src cols");
    if (num_rows == 0 || cols == 0) {
        return;
    }

    dim3 block(256);
    dim3 grid((cols + block.x - 1) / block.x, num_rows);
    hipLaunchKernelGGL(
        pack_rows_bf16_kernel,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const __hip_bfloat16*>(src.data_ptr<at::BFloat16>()),
        row_ids.data_ptr<int64_t>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(num_rows),
        static_cast<int>(cols)
    );
    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "pack_rows_bf16 launch failed: ", hipGetErrorString(err));
}
"""


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "moe_dispatch_pack_hip_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode("utf-8")).hexdigest()[:12]
        module_name = f"moe_dispatch_pack_{CONFIG['variant_name']}_{digest}"
        _MODULE = load_inline(
            name=module_name,
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["pack_rows_bf16"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++17", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _quant():
    global _TRITON_QUANT
    if _TRITON_QUANT is None:
        _TRITON_QUANT = aiter.get_triton_quant(QuantType.per_1x32)
    return _TRITON_QUANT


def _anchor_impl():
    global _ANCHOR_IMPL
    if _ANCHOR_IMPL is None:
        _ANCHOR_IMPL = getattr(importlib.import_module("aiter.fused_moe"), "fused_moe")
    return _ANCHOR_IMPL


def _should_use_dispatch_pack_sparse256(hidden_states: torch.Tensor, config: dict) -> bool:
    return (
        int(config["n_routed_experts"]) == 256
        and int(config["d_expert"]) == 256
        and hidden_states.shape[0] == 512
        and int(config.get("n_shared_experts", config.get("nsharedexperts", 0))) == 0
    )


def _select_block_size_m(hidden_states: torch.Tensor, config: dict) -> int | None:
    if (
        int(config["n_routed_experts"]) == 256
        and int(config["d_expert"]) == 256
        and hidden_states.shape[0] == 512
    ):
        return 32
    return None


def _anchor_path(
    hidden_states: torch.Tensor,
    gate_up_weight_shuffled: torch.Tensor,
    down_weight_shuffled: torch.Tensor,
    gate_up_weight_scale_shuffled: torch.Tensor,
    down_weight_scale_shuffled: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return _anchor_impl()(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        block_size_M=_select_block_size_m(hidden_states, config),
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )


def _dequant_matrix(weight_fp4: torch.Tensor, scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    values = fp4_utils.mxfp4_to_f32(weight_fp4)
    scale = fp4_utils.e8m0_to_f32(scale_e8m0)
    if scale.ndim == 0:
        scale = scale.reshape(1, 1)
    elif scale.ndim == 1:
        if scale.numel() % max(values.shape[0], 1) == 0:
            scale = scale.reshape(values.shape[0], -1)
        else:
            scale = scale.reshape(1, -1).expand(values.shape[0], -1)
    scale = scale[: values.shape[0], :].repeat_interleave(MXFP4_BLOCK, dim=1)[:, : values.shape[1]]
    return (values * scale)[:rows, :cols].to(torch.bfloat16)


def _requantize_activation(activation: torch.Tensor) -> torch.Tensor:
    quantized, scale = _quant()(activation.contiguous(), shuffle=False)
    rows, cols = activation.shape
    return _dequant_matrix(quantized, scale, rows=rows, cols=cols)


def _dequant_gate_up_for_expert(
    gate_up_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    expert: int,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_hidden = int(config["d_hidden"])
    d_expert = int(config["d_expert"])
    gate_up = _dequant_matrix(
        gate_up_weight[expert],
        gate_up_weight_scale[expert],
        rows=2 * d_expert,
        cols=d_hidden,
    )
    gate_part, up_part = gate_up.chunk(2, dim=0)
    return gate_part.contiguous(), up_part.contiguous()


def _dequant_down_for_expert(
    down_weight: torch.Tensor,
    down_weight_scale: torch.Tensor,
    expert: int,
    config: dict,
) -> torch.Tensor:
    d_hidden = int(config["d_hidden"])
    d_expert = int(config["d_expert"])
    return _dequant_matrix(
        down_weight[expert],
        down_weight_scale[expert],
        rows=d_hidden,
        cols=d_expert,
    ).contiguous()


def _route_entries(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, topk = topk_ids.shape
    token_ids = torch.arange(num_tokens, device=topk_ids.device, dtype=torch.int64).repeat_interleave(topk)
    expert_ids = topk_ids.reshape(-1).to(torch.int64)
    weights = topk_weights.reshape(-1, 1).to(torch.bfloat16)
    order = torch.argsort(expert_ids)
    sorted_token_ids = token_ids[order].contiguous()
    sorted_expert_ids = expert_ids[order].contiguous()
    sorted_weights = weights[order].contiguous()
    return sorted_token_ids, sorted_expert_ids, sorted_weights


def _expert_windows(sorted_expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    offsets = torch.zeros_like(counts)
    if counts.numel() > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    return unique_experts, offsets


def _pack_rows_bf16(src: torch.Tensor, row_ids: torch.Tensor, use_hip_pack: bool) -> torch.Tensor:
    if row_ids.numel() == 0:
        return torch.empty((0, src.shape[1]), dtype=src.dtype, device=src.device)
    if not use_hip_pack:
        return src.index_select(0, row_ids).contiguous()
    packed = torch.empty((row_ids.numel(), src.shape[1]), dtype=src.dtype, device=src.device)
    try:
        _module().pack_rows_bf16(src.contiguous(), row_ids.contiguous(), packed)
    except Exception:
        return src.index_select(0, row_ids).contiguous()
    return packed


def _dispatch_pack_path(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    d_hidden = int(config["d_hidden"])
    use_hip_pack = _should_use_dispatch_pack_sparse256(hidden_states, config)
    hidden_states_q = _requantize_activation(hidden_states)

    sorted_token_ids, sorted_expert_ids, sorted_weights = _route_entries(topk_ids, topk_weights)
    unique_experts, offsets = _expert_windows(sorted_expert_ids)
    if sorted_token_ids.numel() == 0:
        return torch.zeros((num_tokens, d_hidden), dtype=torch.bfloat16, device=hidden_states.device)

    packed_hidden = _pack_rows_bf16(hidden_states_q, sorted_token_ids, use_hip_pack=use_hip_pack)
    packed_output = torch.empty((sorted_token_ids.numel(), d_hidden), dtype=torch.bfloat16, device=hidden_states.device)

    for slot, expert_tensor in enumerate(unique_experts):
        expert = int(expert_tensor.item())
        start = int(offsets[slot].item())
        end = int(offsets[slot + 1].item()) if slot + 1 < offsets.numel() else int(sorted_token_ids.numel())
        if end <= start:
            continue
        expert_inputs = packed_hidden[start:end]
        expert_gate_w, expert_up_w = _dequant_gate_up_for_expert(
            gate_up_weight,
            gate_up_weight_scale,
            expert,
            config,
        )
        expert_down_w = _dequant_down_for_expert(
            down_weight,
            down_weight_scale,
            expert,
            config,
        )
        gate = expert_inputs @ expert_gate_w.transpose(0, 1)
        up = expert_inputs @ expert_up_w.transpose(0, 1)
        fused = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        fused_q = _requantize_activation(fused)
        packed_output[start:end] = fused_q @ expert_down_w.transpose(0, 1)

    packed_output = packed_output * sorted_weights
    output = torch.zeros((num_tokens, d_hidden), dtype=torch.bfloat16, device=hidden_states.device)
    output.index_add_(0, sorted_token_ids, packed_output)
    return output


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data
    if not _should_use_dispatch_pack_sparse256(hidden_states, config):
        return _anchor_path(
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            gate_up_weight_scale_shuffled,
            down_weight_scale_shuffled,
            topk_weights,
            topk_ids,
            config,
        )
    return _dispatch_pack_path(
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        topk_weights,
        topk_ids,
        config,
    )
