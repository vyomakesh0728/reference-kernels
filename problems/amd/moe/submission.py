#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"generator": {"kind": "manual_tune"}, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "policy_profile": {"family": "kernel_explore", "name": "moe_blockm_tp8_bs512_v1"}, "problem": "moe_mxfp4"}

import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from task import input_t, output_t


def _select_block_size_m(hidden_states: torch.Tensor, config: dict) -> int | None:
    # Large-`bs` TP=8 still leaves a sparse per-expert distribution.
    # Shrinking the routing/sorting block trims padded regroup work in that case.
    if (
        int(config["n_routed_experts"]) == 256
        and int(config["d_expert"]) == 256
        and hidden_states.shape[0] == 512
    ):
        return 32
    return None


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        _,
        _,
        _,
        _,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
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
