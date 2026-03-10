#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"attempt": 2, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "problem": "moe_mxfp4", "variant": {"family": "anchor", "strategy": "contract_anchor", "variant_name": "fused_moe_contract_anchor"}, "variant_index": 0}
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from task import input_t, output_t


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
    del gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale
    hidden_pad = int(config["d_hidden_pad"]) - int(config["d_hidden"])
    intermediate_pad = int(config["d_expert_pad"]) - int(config["d_expert"])
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
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )
