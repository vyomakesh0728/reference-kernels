# `mxfp4_mm` Sub-Agent Prompt Template

Use exactly three sub-agents per round and keep them all on one exact-shape lane.
Use `gpt-5.2` with `reasoning_effort = xhigh` for every scout. Do not use `gpt-5.2-mini`, and do not use `low`, `medium`, or `high` reasoning for `mxfp4_mm` scout rounds.

Before choosing the next lane for any end-to-end latency, orchestration, or multi-shape prioritization question, also read:

- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-portfolio-ladder.md`

## Closed-Loop Objective

- Keep iterating until one of these is true:
  - the active exact-shape lane reaches the current milestone
  - a concrete blocker is proven with evidence
  - the user redirects the work
- Current milestone: keep deleting whole-call cost centers from the best measured trunk until the overall benchmark geomean is materially below `24 us`, then keep pushing toward the current `<7 us` overall-geomean target.
- Do not stop after one green run if the visible-shape gain is small enough to be noise.
- Do not spend remote quota directly from sub-agents.
- Bias toward deleting real whole-call buckets such as extra launches, temp tensors, contract repair, hot-loop addressing math, orchestration, or bytes moved for non-math work.
- Choose lanes by total geomean leverage across the hot exact-shape family (`m4/m16/m32/m64/m256`), not by whether one shape looks attractive in isolation.
- For any `A-pack` reopen, require an explicit reuse-beats-duplication proof before proposing code. A dominant `A-pack` profile bucket alone is not enough after `v121` and `v122`.
- For any reopened `A-pack` lane, require the sub-agent to state:
  - `reuse_factor`
  - `duplication_factor`
  - `saved_global_bytes_per_block`
  - one short proof that total quant work drops instead of only relocating the old temp bytes
- Reject any `A-pack` idea immediately if `duplication_factor > reuse_factor`, or if the proposal quantizes `A` independently in each output-column CTA with no cross-CTA reuse law.

## Required Roster

- `cost_center_scout`
  - reads:
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/amd-blog-insights.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md`
  - returns: one Candidate Card only
- `retrieval_canon_scout`
  - reads:
    - `/root/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md`
    - `/root/reference-kernels/problems/amd/skills/amd-live-reference-correctness/references/mxfp4-mm-live-contract.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/remote-first-eval.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md`
  - returns: 3-6 grounded hits plus one veto if the idea is wrapper-only, prep-only, or reopens a banned lane
- `bounded_kernel_worker`
  - reads:
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md`
    - `/root/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md`
    - `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md`
    - the active Candidate Card
    - the active retrieval pack
  - returns: one bounded patch plan or one bounded seed rewrite only

## `A-pack` Reopen Rule

If the active lane is any form of `A-pack` deletion, every sub-agent must also return:

- `reuse_factor`
- `duplication_factor`
- `saved_global_bytes_per_block`
- `why_total_quant_work_drops`

Reject the lane immediately if any of these are true:

- `duplication_factor > reuse_factor`
- the proposal re-quantizes `A` independently in each output-column CTA with no cross-CTA reuse law
- the main benefit is only “fewer launches” or “fewer temp bytes” without a proof that quant work is amortized
- the proposal preserves only local CTA reuse while duplication still scales with the full output-column CTA count

Current bounded-wide-family warning:

- under the live `v119` exact-wide grid, simple `C2` or `C4` neighboring-`N` cluster reuse is not enough
- public-shape arithmetic already closes that lane:
  - `m32, n=4096`: `C2 -> reuse 2 / duplication 64`, `C4 -> reuse 4 / duplication 32`
  - `m32, n=2880`: `C2 -> reuse 2 / duplication 45`, `C4 -> reuse 4 / duplication 23`
  - `m64, n=7168`: `C2 -> reuse 2 / duplication 112`, `C4 -> reuse 4 / duplication 56`
  - `m256, n=3072`: `C2 -> reuse 2 / duplication 48`, `C4 -> reuse 4 / duplication 24`
- veto any proposal whose ownership law is only “one producer serves 2-4 neighboring `N` tiles” on top of the current exact-wide dispatch
- if the proposal escalates to macro-cluster service, require it to state the producer span `S` across consecutive `N32` tiles and reject it unless it clears the live thresholds:
  - `m32, n=4096`: `S >= 12` for the first gate, `S >= 16` for the stronger practical gate
  - `m32, n=2880`: `S >= 12` for the first gate, `S >= 16` for the stronger practical gate
  - `m64, n=7168`: `S >= 16` for the first gate, `S >= 24` for the stronger practical gate
  - `m256, n=3072`: `S >= 12` for the first gate, `S >= 16` for the stronger practical gate
- even if the arithmetic clears, reject the lane unless it also names a legal handoff mechanism stronger than per-CTA LDS and weaker than whole-grid global scratch
- if no such mechanism is named, route the round back to non-`A-pack` whole-call buckets instead of broadening the ownership rewrite

## Portfolio Ladder Rule

When the active question is end-to-end latency rather than one exact-shape regression, rank the next spend in this order:

1. compiled direct-entry collapse on a hot exact shape
2. whole helper-launch deletion
3. temp-law deletion that does not increase duplicated work
4. public-shape constant-body deletion of setup/addressing work
5. paper-only `A-pack` duplication-law research

Reject any branch that violates this order unless it has a clearly stronger whole-call deletion story than the earlier ladder items.

## Raw Motivation Refs

- `/root/reference-kernels/problems/amd/important_papers/amd-instinct-cdna4-instruction-set-architecture.pdf`
- `/root/reference-kernels/problems/amd/important_papers/HipKittens Fast and Furious AMD Kernels.pdf`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/optimization.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/amd-blog-insights.md`

## Prompt Skeleton

```text
You are working on `mxfp4_mm` only.

Read these skills first:
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/SKILL.md
- /root/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md

Read this local canon before proposing anything:
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/amd-blog-insights.md
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-cost-center-gate.md
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md
- /root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-portfolio-ladder.md
- /root/reference-kernels/problems/amd/skills/amd-live-reference-correctness/references/mxfp4-mm-live-contract.md
- /root/reference-kernels/problems/amd/important_papers/amd-instinct-cdna4-instruction-set-architecture.pdf
- /root/reference-kernels/problems/amd/important_papers/HipKittens Fast and Furious AMD Kernels.pdf

Use retrieval before guessing low-level details.
Start from these queries:
- <active retrieval pack>

Problem constraints:
- one exact shape only: <shape>
- one regime tag only: <regime_tag>
- one deleted cost center only: <deleted_cost_center>
- keep the live tuple contract `(a, b, b_q, b_shuffle, b_scale_sh)` intact
- keep the current winning raw-contract path intact unless the Candidate Card deletes a whole bucket
- do not reopen banned `A-pack` ownership shapes
- do not reopen local-requant `A-pack` feeder swaps like `v121` or `v122` unless the Candidate Card proves reuse beats duplication
- reject ideas that only polish wrapper code unless they delete a whole-call bucket that has already shown portfolio impact
- do not spend remote quota directly

Candidate Card:
- shape: <shape>
- regime_tag: <regime_tag>
- deleted_cost_center: <deleted_cost_center>
- expected_upside_source: <expected_upside_source>
- why_larger_than_noise: <why_larger_than_noise>
- touched_symbols_or_regions: <touched_symbols_or_regions>
- forbidden_edits: <forbidden_edits>
- success_gate: <success_gate>

Return exactly:
1. shape
2. regime_tag
3. deleted_cost_center
4. expected_upside_source
5. why_larger_than_noise
6. touched_symbols_or_regions
7. forbidden_edits
8. success_gate
9. retrieval_hits
10. one focused implementation proposal

If the lane is an `A-pack` reopen, also return exactly:
11. reuse_factor
12. duplication_factor
13. saved_global_bytes_per_block
14. why_total_quant_work_drops

If the evidence is weak, say that and narrow the lane instead of broadening the rewrite.
```
