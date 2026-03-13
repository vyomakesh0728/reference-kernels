---
name: optimization-skill
description: Optimize a green HIP kernel on AMD MI355X/gfx950 without breaking correctness. Use when a problem already has a passing HIP reference and the next goal is to reduce Python-side overhead, improve ingress/LDS movement, tune memory hierarchy, and move toward CDNA4 scaled MFMA based on the AMD CDNA4 GEMM optimization blog.
---

# Optimization Skill

Use this skill for phase 2 and later on MI355X once correctness is already green.

## When To Use It

- A HIP `submission.py` already passes `test`
- The next task is to make it faster without breaking correctness
- The target is `gfx950` / CDNA4
- You want a disciplined path from slow green reference to real matrix-core throughput

Do not use this skill as a substitute for correctness recovery. If `test` is still failing, use `skills/amd-live-reference-correctness/SKILL.md` first.

## Workflow

1. Start from a known green HIP candidate.
   For `mxfp4_mm`, the current tracked reference is `fp8-mm/hip_reference_green.py`.

2. Preserve the exact semantics first.
   Every optimization round must keep the same outputs as the green reference.

3. Optimize one lever at a time.
   Prefer one focused change per round:
   - shrink Python-side reconstruction
   - vectorize ingress
   - improve global-to-LDS movement
   - change LDS layout/swizzle
   - add double buffering
   - replace the inner loop with scaled MFMA
   - tune occupancy/wave mapping

4. Follow the CDNA4 ladder in order.
   Read `references/cdna4-gemm-ladder.md`.

5. Keep benchmark fidelity intact.
   No stream tricks, event tricks, overlap hacks, sleep/gap hacks, or anything that distorts measured microseconds without improving real throughput.

6. Promote only after staged evidence.
   - `test` stays green
   - `benchmark` improves materially
   - only then consider `leaderboard`

## Phase 2 Priorities

For the current MM path, the first priorities are:

1. Remove or shrink Python-side `A_ref/B_ref` reconstruction cost while keeping outputs identical.
2. Preserve the simple tiled HIP GEMM body while improving data ingress.
3. Only then move toward LDS improvements and scaled MFMA on `gfx950`.

## Non-Negotiables

- Keep the hot path in HIP through `load_inline`
- Stay on one compilation path
- Do not claim progress if the optimized path is dead scaffold
- Do not jump to occupancy tuning before the math and memory path are right
