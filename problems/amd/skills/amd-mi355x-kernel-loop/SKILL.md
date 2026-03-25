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
3. If the problem is `mxfp4_mm`, read [references/mxfp4-through-v45.md](references/mxfp4-through-v45.md) and [references/amd-blog-insights.md](references/amd-blog-insights.md) before planning experiments.
4. If the problem is `moe_mxfp4` or `mixed_mla`, read [references/problem-transfer.md](references/problem-transfer.md) after the snapshot.
5. If the problem is `moe_mxfp4`, also read [references/fused-moe-multiplier.md](references/fused-moe-multiplier.md) before planning experiments or spawning sub-agents.
6. If the problem is `moe_mxfp4`, also read [references/moe-cost-center-gate.md](references/moe-cost-center-gate.md) and [references/moe-branch-queue.md](references/moe-branch-queue.md) before opening a branch.
7. If the problem is `moe_mxfp4`, read [references/moe-subagent-prompt.md](references/moe-subagent-prompt.md) before spawning lane-local sub-agents.

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
4. Enforce the cost-center branch gate.
   - Every candidate must begin with a Candidate Card that states:
     - shape
     - deleted cost center
     - expected upside source
     - why the gain should be larger than noise
     - forbidden edits
   - Reject any candidate that sounds like cleanup, hoist, fast path, prep improvement, or a cleaner version of the same path unless it deletes a whole bucket of work.
   - Reject any branch that touches more than one exact shape or changes both prep and scheduling together.
4. Use retrieval before guessing low-level contracts.
   - Use `amd_kernel_rag` for intrinsics, ISA, operand/feed layout, LLVM builtin mapping, and CDNA4 docs.
   - Use [references/repo-map.md](references/repo-map.md) for exact entrypoints and index locations.
5. Use dataset mining as prior memory, not live oracle.
   - `dataset_mining/kernelbot_data` is an older AMD snapshot.
   - Use it for repeated failure signatures, benchmark-only failures, and historical pattern lookup.
6. Stay remote-first.
   - Use `test` before `benchmark`.
   - Use `profile_rocprof` only after a benchmark winner when you need hardware-counter evidence for the next Candidate Card.
   - Use `benchmark` before `leaderboard`.
   - For `mxfp4_mm`, prefer the quota-aware `mxfp4-closed-loop` path.
   - For `moe_mxfp4`, prefer the quota-aware `moe-closed-loop` path and run it through `/root/reference-kernels/problems/amd/.venv/bin/python`.
   - For other problems, use `harness-run`, `harness-summary`, `harness-resume`, or the broader `agent_loop` campaign commands.
   - Treat `leaderboard` as a separate seeded-distribution gate, not a formality after benchmark. For `mxfp4_mm`, ranked inputs are not the same population as `test`/`benchmark`, so a benchmark win is necessary but not sufficient.
   - If quota is exhausted, start the quota watcher helper from [references/remote-first-eval.md](references/remote-first-eval.md) instead of manually polling.
   - When a benchmark winner is unclear or a plateau feels real, prefer the `rocprofv3` profiling lane over intuition. The resulting `profile_summary.json` and `candidate_cards.json` become the next design input for shape-local branches.
   - Treat the downloaded kernelbot `profile_*.zip` artifact as mandatory evidence, not an optional attachment. The active profile lane now mines the zip directly when kernelbot returns its built-in rocPROF trace instead of the custom PMC payload.
   - Before opening the next branch after a profile run, read:
     - the zip-derived `profile_summary.json`
     - the zip-derived `candidate_cards.json`
     - the current queue note in [references/mxfp4-profile-branch-queue.md](references/mxfp4-profile-branch-queue.md)
   - The current exact-shape `v83` trunk already exposes ROCTx ranges for `m4`, `m8`, `m16`, `m32`, `m64`, and `m256`, plus `b_prep`, `a_pack`, and `kernel_launch` where the Python path owns those buckets.
7. Record what matters.
   - Keep transient experiment state in `.agent-loop/`.
   - Write shareable wins and turning points into `team_results/`.
8. Assimilate signal before the next branch.
   - Update [references/mxfp4-exact-shape-frontier.md](references/mxfp4-exact-shape-frontier.md) when a branch creates a new plateau signal, unlocks a previously broken path, or changes the next allowed branch order.
   - Prefer adding “recent signal” and “allowed next branches” updates over free-form notes, so future workers inherit a tighter search space.
   - When a result proves a whole class of edits is low-yield, say that explicitly in the frontier note and treat it as canon until contradicted by a structural win.
9. Route sub-agents through the canon explicitly.
  - Sub-agents do not automatically ingest the local skill corpus.
  - Every sub-agent prompt for `mxfp4_mm` must explicitly point at:
     - [references/mxfp4-exact-shape-frontier.md](references/mxfp4-exact-shape-frontier.md)
     - [references/amd-blog-insights.md](references/amd-blog-insights.md)
     - [references/mxfp4-cost-center-gate.md](references/mxfp4-cost-center-gate.md)
     - [references/mxfp4-profile-branch-queue.md](references/mxfp4-profile-branch-queue.md)
  - Require sub-agents to return a Candidate Card before proposing code.
  - Every sub-agent prompt for `moe_mxfp4` must explicitly point at:
    - [references/problem-transfer.md](references/problem-transfer.md)
    - [references/fused-moe-multiplier.md](references/fused-moe-multiplier.md)
    - [references/moe-cost-center-gate.md](references/moe-cost-center-gate.md)
    - [references/moe-branch-queue.md](references/moe-branch-queue.md)
    - [references/moe-subagent-prompt.md](references/moe-subagent-prompt.md)
    - `/root/reference-kernels/problems/amd/important_papers/fused_moe/links.md`
  - For `moe_mxfp4`, require sub-agents to return:
    - lane
    - regime tag
    - deleted cost center
    - expected upside source
    - why larger than noise
    - forbidden edits
    - success gate

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
- [references/amd-blog-insights.md](references/amd-blog-insights.md)
  Use for distilled AMD blog and HipKittens-adjacent insights that are specifically relevant to `mxfp4_mm` after the `v66 -> v73` wide-line breakthroughs.
- [references/mxfp4-exact-shape-frontier.md](references/mxfp4-exact-shape-frontier.md)
  Use for the current `mxfp4_mm` exact-shape dispatch frontier, per-shape anchors, ranked-seed caveats, and allowed next branches.
- [references/mxfp4-cost-center-gate.md](references/mxfp4-cost-center-gate.md)
  Use for the mandatory Candidate Card schema, branch veto rules, and per-shape cost-center ladder.
- [references/mxfp4-profile-branch-queue.md](references/mxfp4-profile-branch-queue.md)
  Use for the current zip-derived branch order, active Candidate Cards, and the required next exact-shape queue.
- [references/problem-transfer.md](references/problem-transfer.md)
  Use to transfer the `mxfp4` playbook to `moe` and `mla-decode`.
- [references/fused-moe-multiplier.md](references/fused-moe-multiplier.md)
  Use for the MoE-specific paper canon, raw GitHub links, and the current structural multiplier bets.
- [references/moe-cost-center-gate.md](references/moe-cost-center-gate.md)
  Use for the mandatory MoE Candidate Card schema, branch veto rules, and lane success gates.
- [references/moe-branch-queue.md](references/moe-branch-queue.md)
  Use for the current first-wave branch order and exact MoE lane/regime queue.
- [references/moe-subagent-prompt.md](references/moe-subagent-prompt.md)
  Use for the default MoE sub-agent prompt contract.
- [references/remote-first-eval.md](references/remote-first-eval.md)
  Use for harness commands, closed-loop commands, quota discipline, and promotion rules.
- `scripts/quota_watch_resume.py`
  Use when `mxfp4-closed-loop` quota is exhausted and there is a pending `test`, `benchmark`, or `leaderboard` stage that should auto-resume at the exact next slot.
- `scripts/problem_snapshot.py`
  Use first to build a compact per-problem snapshot.
- [$commit-push](/Users/v/.codex/skills/commit-push/SKILL.md)
  Use only when the user explicitly asks to commit or push.
- [$create-plan](/Users/v/.codex/skills/create-plan/SKILL.md)
  Use only when the user explicitly asks for a plan.

## Current Canon

- Current best measured `mxfp4_mm` frontier:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_pyprep_v83/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_pyprep_v83/submission.py)
- Current ranked `mxfp4_mm` anchor:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_three_regime_v76/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_three_regime_v76/submission.py)
- Current thin baseline:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m16_direct_entry_v54/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_m16_direct_entry_v54/submission.py)
- Current pure recovery anchor:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py)

For `mxfp4_mm`, the live dispatch frontier is now exact-shape first: `m == 4`, `8`, `16`, `32`, `64`, and `256` each have dedicated routes in `v83`, with a separate `other multiples of 32` path behind them. Start from that dispatch unless your single hypothesis is to change routing itself.

Current cost-center policy for `mxfp4_mm`:

- Treat prep-only `m16` and prep-only `m32` edits as plateau territory unless a new branch deletes a whole cost center instead of shaving address/setup work.
- Treat exact `m64` prep specialization as low-yield after `v87`; `v90` also showed that the first exact `m64` body-level deletion was not stable enough to promote, so do not reopen `m64` unless the next branch deletes a different full bucket such as exact `m64` A-pack.
- Spend the next serious optimization budget on `m256` direct-body robustness, then `m4` exact-path fixed-overhead deletion, and only then revisit `m64` with a new deleted bucket.
- Keep `m8` shape-isolated and test-green, but do not let it consume benchmark budget ahead of `m4`, `m64`, or `m256`.

Start from the current winner or the current repo `submission.py`. Do not rediscover old dead ends unless you have a specific contract or performance reason.
