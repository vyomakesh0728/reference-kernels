---
name: amd-kernel-speedrun
description: Generate and iterate real AMD x GPU MODE MI355X kernel candidates for fp8-mm, mla-decode, or moe. Use when working in /Users/v/reference-kernels/problems/amd on Triton or HIP/load_inline paths, when editing agent_loop prompts/mutators, when reading .agent-loop memory, or when a Codex/LLM agent needs a compact playbook for producing real hot-path candidates instead of anchor-backed wrappers.
---

# AMD Kernel Speedrun

## Overview

Use this skill to work the MI355X competition loop in a disciplined way: read the current search state, focus one problem until the hot path is understood, generate a real HIP hot path, submit through the existing `agent_loop`, and learn from failures without re-deriving the same context.

## Workflow

1. Start with the current state, not assumptions.
   Run `python3 -m agent_loop --config agent_loop.toml status` from `/Users/v/reference-kernels/problems/amd`.
   For one problem, run `python3 /Users/v/reference-kernels/problems/amd/skills/amd-kernel-speedrun/scripts/problem_snapshot.py --repo /Users/v/reference-kernels/problems/amd --problem <problem>`.

2. Prefer one problem first when failures are repetitive.
   If one problem shows the same failure signature repeatedly, focus there before running all three in parallel.
   Current default priority is `mxfp4_mm`, then `moe_mxfp4`, then `mixed_mla`.

3. Keep the hot-path claim honest.
   For `hip_explore`, the hot path inside `custom_kernel` must actually execute HIP through `load_inline` on `gfx950`.
   Do not count candidates where the claimed optimized path is only dead scaffold and the real work still happens in `aiter.gemm_a4w4`, `fused_moe`, or `mla_decode_fwd`.

4. Use the existing closed loop, not ad hoc local benchmarking.
   The source of truth is the current `agent_loop` pipeline and ranked leaderboard submissions.
   Cached bootstrap is already implemented, so restarting a campaign should not resubmit the same anchor baseline when the source has not changed.

5. Learn from failures, then prune.
   Failed mutation candidates are compacted into `.agent-loop/problems/<problem>/pruned/*.json`.
   Read the pruned summaries and `knowledge.json` before proposing the next rewrite.
   Do not resurrect the same broken family without a concrete semantic change.

## Problem Rules

Read `/Users/v/reference-kernels/problems/amd/skills/amd-kernel-speedrun/references/problem-contracts.md` when writing or reviewing candidates.

High-level rules:

- `mxfp4_mm`: combine local shape envelopes with the live `amd-mxfp4-mm` contract; preserve shuffled MXFP4 semantics before tuning.
  Current preferred path is HIP-first on `gfx950` via `load_inline`, then CDNA4 scaled-MFMA.
- `moe_mxfp4`: keep routing/top-k semantics fixed while rewriting one expert-compute stage at a time.
- `mixed_mla`: treat decode as latency-first with `q_seq_len=1`; optimize actual Triton decode/attention work, not a wrapper around the anchor.

## Mutation Rules

Read `/Users/v/reference-kernels/problems/amd/skills/amd-kernel-speedrun/references/search-playbook.md` when planning a candidate family or changing the search loop.

Default expectations:

- Prefer small semantic repairs over wide rewrites when correctness is failing.
- When a repeated failure signature appears, make one structural change that directly addresses it.
- Promote only real ranked improvements.
- Stop calling a path “progress” if the claimed optimized hot path is still anchor-backed.

## Phase 2 Optimization Ladder

Once a problem has its first green HIP reference, optimize in this order:

1. Preserve correctness while removing expensive Python-side reconstruction from the hot path.
   For MM, the first target is reducing the calibrated `A_ref/B_ref` rebuild cost without changing outputs.

2. Improve ingress and data movement before heroic math changes.
   Use vectorized loads, better packing-aware reads, and cleaner global-to-LDS movement.

3. Optimize LDS layout and reuse.
   Introduce bank-conflict-aware layouts, swizzle where it helps, and double buffering only after correctness stays green.

4. Move the inner loop toward CDNA4-native math.
   For MM, the intended path is the scaled MFMA family on `gfx950`, especially:
   `V_MFMA_SCALE_F32_16X16X128_F8F6F4`
   `V_MFMA_SCALE_F32_32X32X64_F8F6F4`

5. Tune occupancy and wave mapping last.
   Only after the kernel uses the right math path and memory hierarchy should you spend submissions on wave/block shape tuning.

Use the AMD CDNA4 GEMM blog as the optimization ladder, not as a substitute for correctness. The first reference kernel should be slow and green; the second phase is about climbing from that point toward vectorized ingress, LDS tuning, double buffering, swizzle, and scaled MFMA.

## Resources

### scripts/

- `problem_snapshot.py`: summarize best candidate, recent repeated failures, and current history for one problem from `.agent-loop`.

### references/

- `problem-contracts.md`: shape and contract notes for all three problems.
- `search-playbook.md`: loop discipline, mutation priorities, and plateau rules.
