#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# Candidate Card:
# shape: re256_de256_bs512_topk8
# regime_tag: re256_de256_bs512_topk8
# lane: dispatch_pack
# deleted cost center: generic 9-way sparse scheduling that mixes routed work with the always-on shared expert
# expected upside source: routed-only exact sparse path plus a separate dense shared-expert path should cut metadata and scheduling overhead on sparse256 bs512
# why larger than noise: bs512 sparse256 carries 4608 expert slots through the generic path; removing the shared slot from sparse scheduling deletes a whole metadata bucket
# forbidden edits: router/topk changes, stage1/stage2 ownership changes, fused_moe in the target hot path, Python all-expert rebuilds
# success_gate: clear re256_de256_bs512_topk8 win and global <170 us
# AGENT_LOOP_META: {"candidate_card": {"deleted_cost_center": "generic 9-way sparse scheduling that mixes routed work with the always-on shared expert", "expected_upside_source": "routed-only exact sparse path plus a separate dense shared-expert path should cut metadata and scheduling overhead on sparse256 bs512", "forbidden_edits": ["router/topk changes", "stage1/stage2 ownership changes", "fused_moe in the target hot path", "Python all-expert rebuilds"], "lane": "dispatch_pack", "motivation_refs": ["/root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/scattermoe.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/sonicmoe.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/deepseek_v3.md", "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/competition-rules.md"], "regime_tag": "re256_de256_bs512_topk8", "retrieval_queries": ["q29-fused-moe-padding-free-packing", "shared expert split routed sparse expert schedule", "sorted_token_ids sorted_expert_ids num_valid_ids"], "success_gate": "clear re256_de256_bs512_topk8 win and global <170 us", "why_larger_than_noise": "bs512 sparse256 carries 4608 expert slots through the generic path; removing the shared slot from sparse scheduling deletes a whole metadata bucket"}, "generator": {"kind": "manual_phase2"}, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "policy_profile": {"family": "hip_explore", "name": "dispatch_pack_sparse256_v6"}, "problem": "moe_mxfp4", "variant": {"ARCH": "gfx950", "HOT_PATH_STATE": "partial-native", "LANE": "dispatch_pack", "REGIME_HINT": "re256_de256_bs512_topk8", "family": "hip_explore", "strategy": "dispatch_pack_split_shared", "variant_name": "dispatch_pack_sparse256_v6"}}

import hashlib
import os
import tempfile
from pathlib import Path

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe_2stages, get_2stage_cfgs, get_padded_M, moe_sorting
from aiter.utility import fp4_utils
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


CONFIG = {
    "variant_name": "dispatch_pack_sparse256_v6",
    "family": "hip_explore",
    "strategy": "dispatch_pack_split_shared",
    "LANE": "dispatch_pack",
    "HOT_PATH_STATE": "partial-native",
    "REGIME_HINT": "re256_de256_bs512_topk8",
    "ARCH": "gfx950",
}
MXFP4_BLOCK = 32
MODULE = None

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
    global MODULE
    if MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "moe_dispatch_pack_sparse256_v6_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode("utf-8")).hexdigest()[:12]
        module_name = f"moe_dispatch_pack_sparse256_v6_{digest}"
        MODULE = load_inline(
            name=module_name,
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["pack_rows_bf16"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++17", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return MODULE


def _should_use_dispatch_pack_sparse256(hidden_states: torch.Tensor, config: dict) -> bool:
    return (
        int(config["n_routed_experts"]) == 256
        and int(config["d_expert"]) == 256
        and int(hidden_states.shape[0]) == 16
        and int(config["n_shared_experts"]) == 1
        and int(config["n_experts_per_token"]) == 8
    )


def _route_entries(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, topk = topk_ids.shape
    token_ids = torch.arange(num_tokens, device=topk_ids.device, dtype=torch.int64).repeat_interleave(topk)
    expert_ids = topk_ids.reshape(-1).to(torch.int64)
    weights = topk_weights.reshape(-1, 1).to(torch.float32)
    order = torch.argsort(expert_ids)
    return token_ids[order].contiguous(), expert_ids[order].contiguous(), weights[order].contiguous()


def _pack_rows_bf16(src: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    if row_ids.numel() == 0:
        return torch.empty((0, src.shape[1]), dtype=src.dtype, device=src.device)
    packed = torch.empty((row_ids.numel(), src.shape[1]), dtype=src.dtype, device=src.device)
    try:
        _module().pack_rows_bf16(src.contiguous(), row_ids.contiguous(), packed)
    except Exception:
        return src.index_select(0, row_ids).contiguous()
    return packed


def _dequant_matrix(weight_fp4: torch.Tensor, scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    values = fp4_utils.mxfp4_to_f32(weight_fp4.contiguous())
    scale = fp4_utils.e8m0_to_f32(scale_e8m0.contiguous())
    if scale.ndim == 0:
        scale = scale.reshape(1, 1)
    elif scale.ndim == 1:
        if scale.numel() % max(values.shape[0], 1) == 0:
            scale = scale.reshape(values.shape[0], -1)
        else:
            scale = scale.reshape(1, -1).expand(values.shape[0], -1)
    scale = scale[: values.shape[0], :].repeat_interleave(MXFP4_BLOCK, dim=1)[:, : values.shape[1]]
    return (values * scale)[:rows, :cols].to(torch.bfloat16).contiguous()


def _shared_dense_path(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    if topk_weights.numel() == 0:
        return torch.zeros(
            (hidden_states.shape[0], int(config["d_hidden"])),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
    d_hidden = int(config["d_hidden"])
    d_expert = int(config["d_expert"])
    gate_up = _dequant_matrix(gate_up_weight[0], gate_up_weight_scale[0], rows=2 * d_expert, cols=d_hidden)
    down = _dequant_matrix(down_weight[0], down_weight_scale[0], rows=d_hidden, cols=d_expert)
    gate_part, up_part = gate_up.chunk(2, dim=0)
    hidden = hidden_states.to(torch.float32)
    gate = hidden @ gate_part.transpose(0, 1).to(torch.float32)
    up = hidden @ up_part.transpose(0, 1).to(torch.float32)
    fused = F.silu(gate) * up
    out = fused @ down.transpose(0, 1).to(torch.float32)
    return (out * topk_weights[:, :1].to(torch.float32)).to(torch.bfloat16)


def _green_two_stage_exact(
    hidden_states: torch.Tensor,
    gate_up_weight_shuffled: torch.Tensor,
    down_weight_shuffled: torch.Tensor,
    gate_up_weight_scale_shuffled: torch.Tensor,
    down_weight_scale_shuffled: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    m = int(hidden_states.shape[0])
    d_hidden = int(config["d_hidden"])
    d_expert = int(config["d_expert"])
    hidden_pad = int(config["d_hidden_pad"]) - d_hidden
    intermediate_pad = int(config["d_expert_pad"]) - d_expert
    topk = int(topk_ids.shape[1])
    num_experts = int(gate_up_weight_shuffled.shape[0])
    metadata = get_2stage_cfgs(
        get_padded_M(m),
        d_hidden,
        d_expert,
        num_experts,
        topk,
        hidden_states.dtype,
        dtypes.fp4x2,
        gate_up_weight_shuffled.dtype,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        hidden_pad,
        intermediate_pad,
        bool(getattr(gate_up_weight_shuffled, "is_shuffled", False)),
    )
    block_size_m = int(metadata.block_m)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        d_hidden,
        hidden_states.dtype,
        block_size_m,
    )
    return fused_moe_2stages(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        True,
        block_size_m,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=dtypes.fp4x2,
        q_dtype_w=gate_up_weight_shuffled.dtype,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        num_local_tokens=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=None,
        bias2=None,
    )


def _dispatch_split_shared_path(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    gate_up_weight_shuffled: torch.Tensor,
    down_weight_shuffled: torch.Tensor,
    gate_up_weight_scale_shuffled: torch.Tensor,
    down_weight_scale_shuffled: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    routed_topk = int(config["n_experts_per_token"])
    routed_ids = topk_ids[:, :routed_topk].contiguous()
    routed_weights = topk_weights[:, :routed_topk].contiguous()
    routed_token_ids, _, _ = _route_entries(routed_ids, routed_weights)
    _ = _pack_rows_bf16(hidden_states, routed_token_ids)

    # Keep the routed IDs/weights narrowed to the sparse top-k, but preserve the full
    # shuffled expert bank so the benchmark path sees the same scale/layout contract
    # as the known-good exact baseline.
    routed_out = _green_two_stage_exact(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        routed_weights,
        routed_ids,
        config,
    )

    shared_out = _shared_dense_path(
        hidden_states,
        gate_up_weight[int(config["n_routed_experts"]):].contiguous(),
        down_weight[int(config["n_routed_experts"]):].contiguous(),
        gate_up_weight_scale[int(config["n_routed_experts"]):].contiguous(),
        down_weight_scale[int(config["n_routed_experts"]):].contiguous(),
        topk_weights[:, routed_topk:].contiguous(),
        config,
    )
    return (routed_out.to(torch.float32) + shared_out.to(torch.float32)).to(torch.bfloat16)


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
        return _green_two_stage_exact(
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            gate_up_weight_scale_shuffled,
            down_weight_scale_shuffled,
            topk_weights,
            topk_ids,
            config,
        )

    return _dispatch_split_shared_path(
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
    )
