# MoE Sub-Agent Prompt Template

Use exactly three sub-agents per round and keep them all on one active lane.

## Closed-Loop Objective

- Keep iterating until one of these is true:
  - the active family reaches the current milestone
  - a concrete blocker is proven with evidence
  - the user redirects the work
- Current milestone: first stable `<=140 us` geomean path, then continue toward the leaderboard target.
- Do not stop after one green or one fast run if the rerun spread is larger than normal noise.
- Use `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/scripts/quota_watch_resume.py` to keep remote `test` and `benchmark` submissions moving after hourly quota resets instead of waiting passively.

## Required Roster

- `structure_planner`
  - reads:
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/fused-moe-multiplier.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md`
  - returns: one Candidate Card only
- `retrieval_canon_scout`
  - reads:
    - `/root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md`
    - `/root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/references/moe-closed-loop.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/remote-first-eval.md`
  - returns: 3-6 retrieval hits plus a veto if the idea is wrapper-only
- `bounded_kernel_worker`
  - reads:
    - both skill files
    - the active Candidate Card
    - the active retrieval pack
  - returns: one bounded patch plan or one bounded seed rewrite only

## Raw Motivation Links

- `https://github.com/microsoft/tutel`
- `https://github.com/deepseek-ai/DeepSeek-V3`
- `https://github.com/shawntan/scattermoe`
- `https://github.com/osayamenja/FlashMoE`
- `https://github.com/Dao-AILab/sonic-moe`

## Prompt Skeleton

```text
You are working on `moe_mxfp4` only.

Read these skills first:
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md
- /root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md
- /root/reference-kernels/problems/amd/skills/amd-kernel-speedrun/references/moe-closed-loop.md

Read this local canon before proposing anything:
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/fused-moe-multiplier.md
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md
- /root/reference-kernels/problems/amd/important_papers/fused_moe/README.md
- /root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md
- /root/reference-kernels/problems/amd/important_papers/fused_moe/links.md

Use retrieval before guessing low-level details.
Start from these queries:
- <active retrieval pack>

Problem constraints:
- keep `topk_ids` and `topk_weights` fixed and visible
- one lane only: <lane>
- one regime tag only: <regime_tag>
- no `fused_moe(` in non-anchor hot paths
- no anchor fallback hidden behind dynamic import or helper indirection
- no Python all-expert rebuilds on benchmark lanes
- native candidates must handle shared experts correctly
  - shared experts are appended in `topk_ids`
  - last `n_shared_experts` columns use IDs `[n_routed_experts, n_routed_experts + n_shared_experts)`
  - shared experts are always selected with weight `1.0`
- do not spend remote quota directly

Current execution order:
- first: `test -> benchmark` on `stage1_grouped_bf16_sparse256_v1`
- second: create a sibling `stage1_core` branch for `re32_de512_bs512_topk8`
- only after a stable stage1 win: begin MFMA/LDS tuning
- scaled-MFMA is required for the `<=140 us` milestone, but it is phase-two work inside a winning stage1/stage2 structure, not the first rewrite

Candidate Card:
- lane: <lane>
- regime_tag: <regime_tag>
- deleted_cost_center: <deleted_cost_center>
- expected_upside_source: <expected_upside_source>
- why_larger_than_noise: <why_larger_than_noise>
- forbidden_edits: <forbidden_edits>
- success_gate: <success_gate>

Return exactly:
1. lane
2. regime_tag
3. deleted_cost_center
4. expected_upside_source
5. why_larger_than_noise
6. forbidden_edits
7. success_gate
8. retrieval_hits
9. one focused implementation proposal

If the evidence is weak, say that and narrow the lane instead of broadening the rewrite.
```
