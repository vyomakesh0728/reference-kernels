#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# Candidate Card:
# shape: re256_de256_bs512_topk8
# regime_tag: re256_de256_bs512_topk8
# lane: full_pipeline
# deleted cost center: anchor-backed fused_moe hot path plus opaque stage ownership across dispatch, stage1, and stage2
# expected upside source: correctness-first native gfx950 scaled-MFMA stage kernels over the live topk contract
# why larger than noise: replacing fused_moe with explicit expert grouping and native stage math deletes a whole semantic bucket, not wrapper jitter
# forbidden edits: router semantics, topk visibility, fused_moe fallback, Python all-expert rebuilds, mixed-lane rewrites
# success_gate: passes test on the active MXFP4 fused_moe contract and becomes the first non-anchor scaled-MFMA correctness baseline
# AGENT_LOOP_META: {"candidate_card": {"deleted_cost_center": "anchor-backed fused_moe hot path plus opaque stage ownership across dispatch, stage1, and stage2", "expected_upside_source": "correctness-first native gfx950 scaled-MFMA stage kernels over the live topk contract", "forbidden_edits": ["router semantics", "topk visibility", "fused_moe fallback", "Python all-expert rebuilds", "mixed-lane rewrites"], "lane": "full_pipeline", "motivation_refs": ["/root/reference-kernels/problems/amd/skills/amd-live-reference-correctness/SKILL.md", "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/fused-moe-multiplier.md", "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md", "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md", "/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-through-v45.md"], "regime_tag": "re256_de256_bs512_topk8", "retrieval_queries": ["moe scaled mfma correctness baseline", "gfx950 mxfp4 scale exact m16", "topk_ids topk_weights expert grouping", "shared experts appended in topk_ids"], "success_gate": "passes test on the active MXFP4 fused_moe contract and becomes the first non-anchor scaled-MFMA correctness baseline", "why_larger_than_noise": "replacing fused_moe with explicit expert grouping and native stage math deletes a whole semantic bucket, not wrapper jitter"}, "generator": {"kind": "manual_tune"}, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "policy_profile": {"family": "manual_tune", "name": "moe_scaled_mfma_correct_v1"}, "problem": "moe_mxfp4", "variant": {"ARCH": "gfx950", "HOT_PATH_STATE": "native", "LANE": "full_pipeline", "REGIME_HINT": "re256_de256_bs512_topk8", "family": "manual_tune", "strategy": "scaled_mfma_correctness", "variant_name": "moe_scaled_mfma_correct_v1"}}

import hashlib
import os
import tempfile
from pathlib import Path

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import aiter
from aiter import QuantType, dtypes
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


_MODULE = None
_TORCH_QUANT = None


CPP_WRAPPER = r"""
void moe_mxfp4_scaled_mfma_exact_m16(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
"""


HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

using i32x8_t = int __attribute__((ext_vector_type(8)));
using floatx4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int pack_scale_e8m0x4_lane(const uint8_t* scale_ptr, int group4) {
    return static_cast<int>(scale_ptr[group4])
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__global__ void moe_mxfp4_scaled_mfma_exact_m16_kernel(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int m,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;

    union {
        i32x8_t v;
        unsigned char b[32];
    } a_buf;
    union {
        i32x8_t v;
        unsigned char b[32];
    } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const int a_row = tile_row + lane16;
        if (a_row < m) {
            const unsigned char* ldg_a = a_packed + a_row * a_bytes_per_row + tile_k / 2 + group4 * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_buf.b[i] = ldg_a[i];
            }
        }

        const int b_row = tile_col + lane16;
        if (b_row < n) {
            const unsigned char* ldg_b = b_packed + b_row * b_bytes_per_row + tile_k / 2 + group4 * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                b_buf.b[i] = ldg_b[i];
            }
        }

        const int scale_block = tile_k / 32;
        const int scale_a = (a_row < m)
            ? pack_scale_e8m0x4_lane(a_scale + a_row * a_scale_stride + scale_block, group4)
            : (127 | (127 << 8) | (127 << 16) | (127 << 24));
        const int scale_b = (b_row < n)
            ? pack_scale_e8m0x4_lane(b_scale + b_row * b_scale_stride + scale_block, group4)
            : (127 | (127 << 8) | (127 << 16) | (127 << 24));
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = tile_row + group4 * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < m && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void moe_mxfp4_scaled_mfma_exact_m16(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor c
) {
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda() && a_scale.is_cuda() && b_scale.is_cuda() && c.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(a_packed.scalar_type() == at::kByte, "a_packed must be uint8");
    TORCH_CHECK(b_packed.scalar_type() == at::kByte, "b_packed must be uint8");
    TORCH_CHECK(a_scale.scalar_type() == at::kByte, "a_scale must be uint8");
    TORCH_CHECK(b_scale.scalar_type() == at::kByte, "b_scale must be uint8");
    TORCH_CHECK(c.scalar_type() == at::kBFloat16, "c must be bf16");
    TORCH_CHECK(
        a_packed.dim() == 2 && b_packed.dim() == 2 && a_scale.dim() == 2 && b_scale.dim() == 2 && c.dim() == 2,
        "rank-2 expected, got a_packed=", a_packed.dim(),
        " b_packed=", b_packed.dim(),
        " a_scale=", a_scale.dim(),
        " b_scale=", b_scale.dim(),
        " c=", c.dim()
    );
    TORCH_CHECK(a_packed.is_contiguous() && b_packed.is_contiguous() && a_scale.is_contiguous() && b_scale.is_contiguous() && c.is_contiguous(), "all tensors must be contiguous");

    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK((k % 128) == 0, "scaled-MFMA exact m16 requires K multiple of 128");
    TORCH_CHECK(a_scale.size(0) == m, "a_scale rows must match output M");
    TORCH_CHECK(b_packed.size(0) == n, "b_packed rows must match output N");
    TORCH_CHECK(b_scale.size(0) == n, "b_scale rows must match output N");

    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);
    hipLaunchKernelGGL(
        moe_mxfp4_scaled_mfma_exact_m16_kernel,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const unsigned char*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<const unsigned char*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<const uint8_t*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<const uint8_t*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}
"""


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "moe_scaled_mfma_correct_v1_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode("utf-8")).hexdigest()[:12]
        module_name = f"moe_scaled_mfma_correct_v1_{digest}"
        _MODULE = load_inline(
            name=module_name,
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["moe_mxfp4_scaled_mfma_exact_m16"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _quant():
    global _TORCH_QUANT
    if _TORCH_QUANT is None:
        _TORCH_QUANT = aiter.get_torch_quant(QuantType.per_1x32)
    return _TORCH_QUANT


def _quantize_activation_mxfp4(activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed, scale = _quant()(activation.contiguous(), quant_dtype=dtypes.fp4x2)
    rows = activation.shape[0]
    cols = activation.shape[1]
    return (
        _packed_weight_matrix(packed, rows, cols),
        _scale_matrix(scale, rows, cols),
    )


def _packed_weight_matrix(weight_packed: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    packed = weight_packed.contiguous().view(torch.uint8)
    cols = k // 2
    if packed.numel() == rows * cols:
        return packed.reshape(rows, cols).contiguous()
    if packed.numel() % cols == 0:
        packed_rows = packed.numel() // cols
        matrix = packed.reshape(packed_rows, cols)
        if packed_rows >= rows:
            return matrix[:rows].contiguous()
    raise RuntimeError(f"cannot normalize packed matrix with numel={packed.numel()} rows={rows} k={k}")


def _scale_matrix(weight_scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    scale = weight_scale.contiguous().view(torch.uint8)
    scale_cols = k // 32
    if scale.numel() == rows * scale_cols:
        return scale.reshape(rows, scale_cols).contiguous()
    if scale.numel() == scale_cols:
        return scale.reshape(1, scale_cols).expand(rows, scale_cols).contiguous()
    if scale.numel() % scale_cols == 0:
        scale_rows = scale.numel() // scale_cols
        matrix = scale.reshape(scale_rows, scale_cols)
        if scale_rows >= rows:
            return matrix[:rows].contiguous()
        repeats = (rows + scale_rows - 1) // scale_rows
        return matrix.repeat(repeats, 1)[:rows].contiguous()
    raise RuntimeError(f"cannot normalize scale matrix with numel={scale.numel()} rows={rows} k={k}")


def _route_entries(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, topk = topk_ids.shape
    token_ids = torch.arange(num_tokens, device=topk_ids.device, dtype=torch.int64).repeat_interleave(topk)
    expert_ids = topk_ids.reshape(-1).to(torch.int64)
    weights = topk_weights.reshape(-1).to(torch.float32)
    order = torch.argsort(expert_ids)
    return token_ids[order].contiguous(), expert_ids[order].contiguous(), weights[order].contiguous()


def _expert_windows(sorted_expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    starts = torch.zeros_like(counts)
    if counts.numel() > 1:
        starts[1:] = torch.cumsum(counts[:-1], dim=0)
    return unique_experts, starts, counts


def _scaled_mfma_nt(
    activations_bf16: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if activations_bf16.numel() == 0:
        return torch.empty((0, weight_packed.shape[0]), dtype=torch.bfloat16, device=activations_bf16.device)
    k = activations_bf16.shape[1]
    n = int(weight_packed.shape[0])
    a_packed, a_scale = _quantize_activation_mxfp4(activations_bf16)
    b_packed = _packed_weight_matrix(weight_packed, n, k)
    b_scale = _scale_matrix(weight_scale, n, k)
    c = torch.empty((activations_bf16.shape[0], n), dtype=torch.bfloat16, device=activations_bf16.device)
    _module().moe_mxfp4_scaled_mfma_exact_m16(
        a_packed,
        b_packed,
        a_scale,
        b_scale,
        c,
    )
    return c


def _pad_hidden(hidden_states: torch.Tensor, d_hidden_pad: int) -> torch.Tensor:
    hidden_tokens = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.bfloat16).contiguous()
    if hidden_tokens.shape[1] == d_hidden_pad:
        return hidden_tokens
    pad = d_hidden_pad - hidden_tokens.shape[1]
    return F.pad(hidden_tokens, (0, pad)).contiguous()


def _moe_scaled_mfma_correct_path(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    d_hidden = int(config["d_hidden"])
    d_hidden_pad = int(config["d_hidden_pad"])
    d_expert = int(config["d_expert"])
    d_expert_pad = int(config["d_expert_pad"])

    hidden_inputs = _pad_hidden(hidden_states, d_hidden_pad)
    num_tokens = hidden_inputs.shape[0]
    token_ids, expert_ids, routed_weights = _route_entries(topk_ids, topk_weights)
    unique_experts, starts, counts = _expert_windows(expert_ids)
    output = torch.zeros((num_tokens, d_hidden), dtype=torch.float32, device=hidden_inputs.device)

    for idx, expert in enumerate(unique_experts.tolist()):
        start = int(starts[idx].item())
        end = start + int(counts[idx].item())
        expert_token_ids = token_ids[start:end]
        expert_inputs = hidden_inputs.index_select(0, expert_token_ids)

        stage1_full = _scaled_mfma_nt(
            expert_inputs,
            gate_up_weight[expert],
            gate_up_weight_scale[expert],
        )
        gate = F.silu(stage1_full[:, :d_expert].to(torch.float32))
        up = stage1_full[:, d_expert_pad : d_expert_pad + d_expert].to(torch.float32)
        fused = (gate * up).to(torch.bfloat16)
        if d_expert_pad != d_expert:
            fused = F.pad(fused, (0, d_expert_pad - d_expert)).contiguous()
        else:
            fused = fused.contiguous()

        stage2_full = _scaled_mfma_nt(
            fused,
            down_weight[expert],
            down_weight_scale[expert],
        )
        weighted = stage2_full[:, :d_hidden].to(torch.float32) * routed_weights[start:end].unsqueeze(1)
        output.index_add_(0, expert_token_ids, weighted)

    output_bf16 = output.to(torch.bfloat16)
    if hidden_states.ndim > 2:
        return output_bf16.reshape(*hidden_states.shape[:-1], d_hidden).contiguous()
    return output_bf16


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

    del gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled, down_weight_scale_shuffled

    _module()
    return _moe_scaled_mfma_correct_path(
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        topk_weights,
        topk_ids,
        config,
    )
