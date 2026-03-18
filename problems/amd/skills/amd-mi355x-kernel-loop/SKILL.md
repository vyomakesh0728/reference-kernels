---
name: amd-mi355x-kernel-loop
description: Optimize AMD MI355X competition kernels in /Users/v/reference-kernels/problems/amd across mxfp4-mm (legacy dir fp8-mm), moe, mla-decode, and identity using the repo's remote-first agent loop, kernelbot/popcorn harness, AMD retrieval stack, dataset-mining priors, and team-results snapshots. Use when iterating on submission.py candidates, recovering correctness, designing manual or handrolled experiments, comparing harness results, or promoting wins without replay/cache tricks. When the user names a specific problem such as mxfp4-mm, moe, or mla-decode, scope the work to that single problem only unless the user explicitly asks for cross-problem parallel work.
---

# AMD MI355X Kernel Loop

## Start Here

1. Run `python3 scripts/problem_snapshot.py --repo /Users/v/reference-kernels/problems/amd --problem <problem>`.
2. Read only the minimum start set from the snapshot:
   - `task.py`
   - `reference.py`
   - `task.yml`
   - the current `submission.py`
3. If the problem is `mxfp4_mm`, read [references/mxfp4-through-v45.md](references/mxfp4-through-v45.md) before planning experiments.
4. If the problem is `moe_mxfp4` or `mixed_mla`, read [references/problem-transfer.md](references/problem-transfer.md) after the snapshot.

## Scope Rules

Work one problem at a time by default.

- If the user says `mxfp4-mm`, work only in [/Users/v/reference-kernels/problems/amd/fp8-mm](/Users/v/reference-kernels/problems/amd/fp8-mm) and its matching `.agent-loop` artifacts.
- If the user says `moe`, work only in [/Users/v/reference-kernels/problems/amd/moe](/Users/v/reference-kernels/problems/amd/moe) and its matching artifacts.
- If the user says `mla` or `mixed-mla`, work only in [/Users/v/reference-kernels/problems/amd/mla-decode](/Users/v/reference-kernels/problems/amd/mla-decode) and its matching artifacts.
- If the user says `identity`, work only in [/Users/v/reference-kernels/problems/amd/identity](/Users/v/reference-kernels/problems/amd/identity).

Do not run parallel optimization campaigns across multiple problems unless the user explicitly asks for cross-problem work.

When the user combines this skill with a problem name, treat that problem as the only active optimization target for the turn.

## Core Workflow

1. Preserve purity first.
   - Recompute outputs from current inputs on every call.
   - Reject pointer-keyed cache, stale-output reuse, replay, and benchmark-only shortcuts.
2. Separate correctness from speed.
   - If `test` is failing, read [/Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/SKILL.md).
   - If `test` is green and the task is speed, read [/Users/v/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md).
3. Keep one hypothesis per candidate.
   - Prefer new candidate files under `/Users/v/reference-kernels/problems/amd/.agent-loop/manual/`.
   - Do not combine semantic repair, data-movement changes, and scheduling changes in one remote run.
4. Use retrieval before guessing low-level contracts.
   - Use `amd_kernel_rag` for intrinsics, ISA, operand/feed layout, LLVM builtin mapping, and CDNA4 docs.
   - Use [references/repo-map.md](references/repo-map.md) for exact entrypoints and index locations.
5. Use dataset mining as prior memory, not live oracle.
   - `dataset_mining/kernelbot_data` is an older AMD snapshot.
   - Use it for repeated failure signatures, benchmark-only failures, and historical pattern lookup.
6. Stay remote-first.
   - Use `test` before `benchmark`.
   - Use `benchmark` before `leaderboard`.
   - For `mxfp4_mm`, prefer the quota-aware `mxfp4-closed-loop` path.
   - For other problems, use `harness-run`, `harness-summary`, `harness-resume`, or the broader `agent_loop` campaign commands.
7. Record what matters.
   - Keep transient experiment state in `.agent-loop/`.
   - Write shareable wins and turning points into `team_results/`.

## Problem Map

- `mxfp4_mm` uses the legacy directory [/Users/v/reference-kernels/problems/amd/fp8-mm](/Users/v/reference-kernels/problems/amd/fp8-mm)
- `moe_mxfp4` uses [/Users/v/reference-kernels/problems/amd/moe](/Users/v/reference-kernels/problems/amd/moe)
- `mixed_mla` uses [/Users/v/reference-kernels/problems/amd/mla-decode](/Users/v/reference-kernels/problems/amd/mla-decode)
- `identity` uses [/Users/v/reference-kernels/problems/amd/identity](/Users/v/reference-kernels/problems/amd/identity)

If a prompt mentions stale `problems/amd_202602/...` paths, map them to these live directories.

## Resources

- [references/repo-map.md](references/repo-map.md)
  Use for repo layout, retrieval, dataset mining, closed-loop state, and companion skills.
- [references/mxfp4-through-v45.md](references/mxfp4-through-v45.md)
  Use for the proven `mxfp4_mm` path from stable trunk through `v45`.
- [references/optimization.md](references/optimization.md)
  Use for the MI355X optimization landscape: launch geometry, occupancy/resource tradeoffs, arithmetic intensity, coalescing, bandwidth, vectorization, cache/LDS behavior, and thin-vs-wide regime choices.
- [references/problem-transfer.md](references/problem-transfer.md)
  Use to transfer the `mxfp4` playbook to `moe` and `mla-decode`.
- [references/remote-first-eval.md](references/remote-first-eval.md)
  Use for harness commands, closed-loop commands, quota discipline, and promotion rules.
- `scripts/problem_snapshot.py`
  Use first to build a compact per-problem snapshot.
- [$commit-push](/Users/v/.codex/skills/commit-push/SKILL.md)
  Use only when the user explicitly asks to commit or push.
- [$create-plan](/Users/v/.codex/skills/create-plan/SKILL.md)
  Use only when the user explicitly asks for a plan.

## Current Canon

- Current promoted `mxfp4_mm` winner:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py)
- Current pure recovery anchor:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py)

Start from the current winner or the current repo `submission.py`. Do not rediscover old dead ends unless you have a specific contract or performance reason.
