#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# Candidate Card:
# shape: re256_de256_bs512_topk8
# regime_tag: re256_de256_bs512_topk8
# lane: stage1_core
# deleted cost center: repeated stage1 launch plus unfused routed gate/up materialization after dispatch_pack
# expected upside source: native grouped hidden->2*d_expert compute on touched experts with a stage1-owned output layout
# why larger than noise: sparse256 bs512 creates many tiny expert windows where the stage1 boundary and intermediate layout overhead compound across touched experts
# forbidden edits: router/topk changes, stage2_reduce changes, shared-expert rewrites, fused_moe in the hot path, Python all-expert rebuilds
# success_gate: native stage1 path beats the control on re256_de256_bs512_topk8 and global <150 us
# AGENT_LOOP_META: {"candidate_card": {"deleted_cost_center": "repeated stage1 launch plus unfused routed gate/up materialization after dispatch_pack", "expected_upside_source": "native grouped hidden->2*d_expert compute on touched experts with a stage1-owned output layout", "forbidden_edits": ["router/topk changes", "stage2_reduce changes", "shared-expert rewrites", "fused_moe in the hot path", "Python all-expert rebuilds"], "lane": "stage1_core", "motivation_refs": ["/root/reference-kernels/problems/amd/important_papers/fused_moe/README.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/links.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/scattermoe.md", "/root/reference-kernels/problems/amd/important_papers/fused_moe/sonicmoe.md"], "regime_tag": "re256_de256_bs512_topk8", "retrieval_queries": ["q30-fused-moe-persistent-pipeline", "stage1 grouped bf16 gate up swiglu fused expert tile pipeline", "ck_moe_stage1 block_m sorted_weights shuffled scale-aware", "persistent expert tile pipeline stage1 swiglu stage2 overlap io compute"], "success_gate": "native stage1 path beats the control on re256_de256_bs512_topk8 and global <150 us", "why_larger_than_noise": "sparse256 bs512 creates many tiny expert windows where the stage1 boundary and intermediate layout overhead compound across touched experts"}, "generator": {"kind": "manual_phase2"}, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "policy_profile": {"family": "kernel_explore", "name": "stage1_grouped_bf16_sparse256_v1"}, "problem": "moe_mxfp4", "variant": {"ARCH": "gfx950", "HOT_PATH_STATE": "partial-native", "LANE": "stage1_core", "REGIME_HINT": "re256_de256_bs512_topk8", "family": "kernel_explore", "strategy": "stage1_grouped_bf16", "variant_name": "stage1_grouped_bf16_sparse256_v1"}}
import os

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import torch
import torch.nn.functional as F
from aiter.utility import fp4_utils
from task import input_t, output_t


CONFIG = {
    "variant_name": "stage1_grouped_bf16_sparse256_v1",
    "family": "kernel_explore",
    "strategy": "stage1_grouped_bf16",
    "LANE": "stage1_core",
    "HOT_PATH_STATE": "partial-native",
    "REGIME_HINT": "re256_de256_bs512_topk8",
    "ARCH": "gfx950",
}
MXFP4_BLOCK = 32


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
def _route_entries(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, topk = topk_ids.shape
    token_ids = torch.arange(num_tokens, device=topk_ids.device, dtype=torch.int64).repeat_interleave(topk)
    expert_ids = topk_ids.reshape(-1).to(torch.int64)
    weights = topk_weights.reshape(-1, 1).to(torch.float32)
    order = torch.argsort(expert_ids)
    return token_ids[order].contiguous(), expert_ids[order].contiguous(), weights[order].contiguous()


def _expert_windows(sorted_expert_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    offsets = torch.zeros_like(counts)
    if counts.numel() > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    return unique_experts, offsets


def _gemm_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(
        a.to(torch.bfloat16).contiguous(),
        b.to(torch.bfloat16).transpose(0, 1).contiguous(),
    ).to(torch.bfloat16)


def _shared_dense_path(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    shared_weights: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    if shared_weights.numel() == 0 or gate_up_weight.numel() == 0:
        return torch.zeros(
            (hidden_states.shape[0], int(config["d_hidden"])),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    d_hidden = int(config["d_hidden"])
    d_expert = int(config["d_expert"])
    hidden = hidden_states.to(torch.float32)
    output = torch.zeros((hidden_states.shape[0], d_hidden), dtype=torch.float32, device=hidden_states.device)
    num_shared = min(shared_weights.shape[1], gate_up_weight.shape[0], down_weight.shape[0])

    for shared_idx in range(num_shared):
        gate_up = _dequant_matrix(
            gate_up_weight[shared_idx],
            gate_up_weight_scale[shared_idx],
            rows=2 * d_expert,
            cols=d_hidden,
        )
        down = _dequant_matrix(
            down_weight[shared_idx],
            down_weight_scale[shared_idx],
            rows=d_hidden,
            cols=d_expert,
        )
        gate_part, up_part = gate_up.chunk(2, dim=0)
        gate = hidden @ gate_part.transpose(0, 1).to(torch.float32)
        up = hidden @ up_part.transpose(0, 1).to(torch.float32)
        fused = F.silu(gate) * up
        output += fused @ down.transpose(0, 1).to(torch.float32) * shared_weights[:, shared_idx : shared_idx + 1].to(torch.float32)

    return output.to(torch.bfloat16)


def _stage1_grouped_path(
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
    d_expert = int(config["d_expert"])
    n_routed_experts = int(config["n_routed_experts"])
    routed_topk = int(config["n_experts_per_token"])

    routed_ids = topk_ids[:, :routed_topk].contiguous()
    routed_weights = topk_weights[:, :routed_topk].contiguous()
    token_ids, expert_ids, weights = _route_entries(routed_ids, routed_weights)
    unique_experts, offsets = _expert_windows(expert_ids)
    _, counts = torch.unique_consecutive(expert_ids, return_counts=True)

    output = torch.zeros((num_tokens, d_hidden), dtype=torch.bfloat16, device=hidden_states.device)
    stage1_layout = torch.empty((token_ids.numel(), 2 * d_expert), dtype=torch.bfloat16, device=hidden_states.device)
    hidden_bf16 = hidden_states.to(torch.bfloat16).contiguous()

    for idx, expert in enumerate(unique_experts.tolist()):
        start = int(offsets[idx].item())
        end = start + int(counts[idx].item())
        if end <= start:
            continue
        if expert >= n_routed_experts:
            continue

        expert_gate_up_w = _dequant_matrix(
            gate_up_weight[expert],
            gate_up_weight_scale[expert],
            rows=2 * d_expert,
            cols=d_hidden,
        )
        expert_down_w = _dequant_matrix(
            down_weight[expert],
            down_weight_scale[expert],
            rows=d_hidden,
            cols=d_expert,
        )

        expert_token_ids = token_ids[start:end]
        expert_inputs = hidden_bf16.index_select(0, expert_token_ids)

        # Stage1 owns the packed gate/up output layout for touched experts.
        gate_up = _gemm_nt(expert_inputs, expert_gate_up_w)
        stage1_layout[start:end].copy_(gate_up)
        expert_stage1 = stage1_layout[start:end]

        gate = F.silu(expert_stage1[:, :d_expert])
        up = expert_stage1[:, d_expert:]
        fused_q = (gate * up).to(torch.bfloat16)

        expert_out = _gemm_nt(fused_q, expert_down_w)
        expert_out = (expert_out * weights[start:end]).to(output.dtype)
        output.index_add_(0, expert_token_ids, expert_out)

    output = (
        output.to(torch.float32)
        + _shared_dense_path(
            hidden_states,
            gate_up_weight[n_routed_experts:].contiguous(),
            down_weight[n_routed_experts:].contiguous(),
            gate_up_weight_scale[n_routed_experts:].contiguous(),
            down_weight_scale[n_routed_experts:].contiguous(),
            topk_weights[:, routed_topk:].contiguous(),
            config,
        ).to(torch.float32)
    ).to(torch.bfloat16)

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

    return _stage1_grouped_path(
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        topk_weights,
        topk_ids,
        config,
    )
