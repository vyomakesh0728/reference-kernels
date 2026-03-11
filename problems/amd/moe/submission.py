#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
# AGENT_LOOP_META: {"attempt": 40, "generator": {"kind": "llm", "model": "(codex default)", "parallel_agents": 3, "provider": "codex_cli", "use_plan": true}, "gpu": "MI355X", "leaderboard": "amd-moe-mxfp4", "policy_profile": {"family": "kernel_explore", "focus": "stabilize routing and shuffled-weight semantics", "name": "contract_repair", "trigger_signals": ["contract_repair", "runtime_repair", "submission_repair"]}, "problem": "moe_mxfp4", "variant": {"BLOCK_SIZE": 256, "NUM_WARPS": 4, "SORT_BY_EXPERT": true, "family": "kernel_explore", "strategy": "routing_prototype", "variant_name": "routing_swiglu_256"}, "variant_index": 2}

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from task import input_t, output_t

_ACTIVATION = ActivationType.Silu
_QUANT_TYPE = QuantType.per_1x32
_FUSED_MOE = fused_moe


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

    hidden_pad = down_weight_shuffled.shape[1] - hidden_states.shape[1]
    intermediate_pad = (gate_up_weight_shuffled.shape[1] >> 1) - config["d_expert"]

    return _FUSED_MOE(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=_ACTIVATION,
        quant_type=_QUANT_TYPE,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )
