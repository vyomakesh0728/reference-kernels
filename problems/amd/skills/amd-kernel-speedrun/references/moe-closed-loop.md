# MoE Closed-Loop Notes

Use this when the active problem is `moe_mxfp4`.

## Lane Order

1. `dispatch_pack`
2. `stage1_core`
3. `stage2_reduce`
4. `shared_expert`
5. `full_pipeline`

## Regime Tags

- `re256_de256_bs16_topk8`
- `re256_de256_bs128_topk8`
- `re256_de256_bs512_topk8`
- `re32_de512_bs16_topk8`
- `re32_de512_bs128_topk8`
- `re32_de512_bs512_topk8`
- `re32_de2048_bs512_topk8`

## Candidate Rules

- one lane per candidate
- one regime tag per candidate
- one deleted cost center per candidate
- one `why_larger_than_noise` claim per candidate
- one success gate per candidate
- no Python all-expert rebuilds on benchmark lanes
- no `fused_moe(` in non-anchor hot paths
- no anchor fallback in a native candidate
- no remote spend without a complete Candidate Card and evidence pack

## Active Milestone

- First milestone: stable `<=140 us` geomean.
- Keep iterating until the milestone lands or a concrete blocker is proven.
- Current lane order inside the milestone push:
  1. `test -> benchmark` on `stage1_grouped_bf16_sparse256_v1`
  2. sibling `stage1_core` branch for `re32_de512_bs512_topk8`
  3. only after a stable `stage1_core` win: MFMA/LDS tuning

## Shared-Expert Contract

- `topk_ids` includes both routed and shared experts.
- First `n_experts_per_token` columns are routed experts.
- Last `n_shared_experts` columns are shared experts with IDs `[n_routed_experts, n_routed_experts + n_shared_experts)`.
- Shared experts are always selected with weight `1.0`.

## Quota Handling

- Use `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/scripts/quota_watch_resume.py` to auto-resume `test` and `benchmark` stages after reset windows.
- Do not let the loop stall just because the current hour is exhausted.

## Candidate Card

Every non-baseline MoE branch must carry:

- `lane`
- `regime_tag`
- `deleted_cost_center`
- `expected_upside_source`
- `why_larger_than_noise`
- `forbidden_edits`
- `success_gate`
- `motivation_refs`
- `retrieval_queries`

Use:

- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-subagent-prompt.md`

## Raw Motivation Links

- `https://github.com/microsoft/tutel`
- `https://github.com/deepseek-ai/DeepSeek-V3`
- `https://github.com/shawntan/scattermoe`
- `https://github.com/osayamenja/FlashMoE`
- `https://github.com/Dao-AILab/sonic-moe`
