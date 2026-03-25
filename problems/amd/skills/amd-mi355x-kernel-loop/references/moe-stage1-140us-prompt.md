# MoE Stage1 To 140us Prompt

Use this when the active MoE goal is the first stable `<=140 us` geomean path.

## Current Objective

- Keep iterating in a closed feedback loop until one of these is true:
  - stable `<=140 us` geomean is reached
  - a concrete blocker is proven with evidence
  - the user redirects the work
- Do not stop after one green run or one fast run.
- Do not spend time on wrapper tuning or anchor-backed convenience paths.

## Immediate Branch Order

1. `test -> benchmark` on `stage1_grouped_bf16_sparse256_v1`
2. sibling `stage1_core` branch for `re32_de512_bs512_topk8`
3. only after a stable `stage1_core` win: MFMA/LDS tuning

## Hard Constraints

- keep `topk_ids` and `topk_weights` fixed and visible
- one lane per branch
- no anchor fallback in the native candidate
- no `fused_moe(` in the non-anchor hot path
- no Python all-expert rebuilds
- shared experts must be handled natively

## Shared-Expert Contract

- `topk_ids` includes both routed and shared experts
- first `n_experts_per_token` columns are routed experts
- last `n_shared_experts` columns are shared experts
- shared expert IDs are `[n_routed_experts, n_routed_experts + n_shared_experts)`
- shared expert weights are always `1.0`

## Stage1 Guidance

- Own `hidden -> 2*d_expert` first.
- Keep `SwiGLU`, requant, stage2, weighting, and final reduction structurally unchanged until stage1 is stable.
- Use grouped/touched-expert stage1 over the packed dispatch order.
- The first scaled-MFMA work belongs inside a stable stage1/stage2 structure, not before it.

## Quota Discipline

- Use `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/scripts/quota_watch_resume.py` for automatic resume after reset windows.
- Keep `test -> benchmark` moving; do not wait passively for the next hour.
