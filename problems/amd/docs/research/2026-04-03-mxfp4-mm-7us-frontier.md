# mxfp4-mm <=7us frontier research (MI355X / CDNA4)

Date: 2026-04-03
Status: research-only, no code changes in this note
Scope: custom HIP/CDNA4 only; no AITER lanes
Anchor: `fp8-mm/submission.py` == `fp8-mm/submission_anchor_13p406.py`

## Goal

Drive the exact-shape benchmark geomean from ~13.406 us to <=7.0 us across:
- m4 k512 n2880
- m16 k7168 n2112
- m32 k512 n2880
- m32 k512 n4096
- m64 k2048 n7168
- m256 k1536 n3072

This note consolidates the live code audit, profile evidence, CDNA4 ISA facts, dead lanes, and a high-confidence patch ladder.

## Executive conclusion

The current anchor already deleted most of the old B-side temp laws on the public exact paths.
The remaining dominant whole-call bucket is the separate A-pack launch family:

1. Python fast path still allocates a workspace and dispatches an owned-workspace wrapper.
2. The owned-workspace wrapper still launches `mxfp4_pack_a_fixed_kernel_wave`.
3. The exact kernel is then launched as a second GPU kernel.

At the current frontier, helper arithmetic micro-edits and wrapper-only cleanup are not enough.
The only credible route to <=7 us is to delete the two-launch exact-path law itself.

The highest-confidence aggressive lane is:
- keep custom HIP exact kernels,
- stop paying the standalone A-pack helper on the public exact paths,
- switch the exact public family to consume the already-available `b_shuffle`/`b_scale_sh` contract directly,
- replace the current two-launch law with a single-launch or retained-prepack law that preserves one-pack-per-call economics,
- use CDNA4 FP4/BF16 convert builtins only if they do not reintroduce per-CTA A-quant duplication.

Important guardrail:
- A branch is only valid if it avoids the already-dead fused-A family.
- If the implementation degenerates into per-output-CTA BF16->FP4 conversion of `A`, it is structurally the same failed lane and should be killed immediately.

## Current anchor: measured wall vs GPU self gap

Benchmark anchor:
- `./.agent-loop/harness_runs/mxfp4_mm/20260402-120426-p0p5-runtime-collapse-owned-scaleflat-r1-benchmark/stages/01_benchmark/parsed_metrics.json`

Fresh profile anchor:
- `./.agent-loop/harness_runs/mxfp4_mm/20260402-143032-baseline-recheck-profile-r1-harness/stages/01_profile_rocprof/profile/profile_summary.json`

Approximate current picture:

| shape | bench_us | gpu_self_us | delta_us | dominant profiled bucket |
|---|---:|---:|---:|---|
| m4 | 9.9 | 2.918 | ~7.0 | A-pack 2.079 us |
| m16 | 19.6 | 2.598 | ~17.0 | A-pack 2.119 us |
| m32 n2880 | 10.0 | 2.438 | ~7.5 | A-pack 1.959 us |
| m32 n4096 | 10.1 | 2.078 | ~8.0 | A-pack 1.599 us |
| m64 | 18.1 | 0.958 | ~17.1 | A-pack 0.479 + kernel 0.479 |
| m256 | 16.4 | 0.958 | ~15.4 | A-pack 0.479 + kernel 0.479 |

Interpretation:
- m64 and m256 are the strongest proof that the whole-call problem is mostly outside the currently-profiled compute bodies.
- m16 and m32 still show A-pack as the dominant profiled GPU bucket.
- Therefore <=7 us is not a body-only optimization target; it is a launch/runtime-law deletion target.

## What the live public exact paths still do

Top-level exact dispatch in `custom_kernel`:
- m4: `fp8-mm/submission.py:3778-3796`
- m16: `fp8-mm/submission.py:3797-3811`
- m32: `fp8-mm/submission.py:3812-3826`
- m64: `fp8-mm/submission.py:3827-3841`
- m256: `fp8-mm/submission.py:3842-3856`

Each public exact call still performs:
- `_module()` lookup
- contiguity/view normalization
- `torch.empty(...)` workspace allocation
- `workspace.narrow(...).view(...)` to carve `c`
- one helper launch via `launch_mxfp4_pack_a_fixed_raw(...)`
- one exact MFMA launch

Shared helper and launcher:
- `mxfp4_pack_a_fixed_kernel_wave`: `fp8-mm/submission.py:544-654`
- `launch_mxfp4_pack_a_fixed_raw`: `fp8-mm/submission.py:673-727`

Public exact owned-workspace wrappers:
- m16: `fp8-mm/submission.py:1700-1754`
- m4: `fp8-mm/submission.py:1799-1852`
- m32: `fp8-mm/submission.py:2986-3043`
- m64: `fp8-mm/submission.py:3097-3150`
- m256: `fp8-mm/submission.py:3208-3262`

Important live-state finding:
- The public exact paths already deleted most of the old B-side generic temp laws.
- `candidate_cards.json` is stale for m16/m32 in its wording; the live path no longer materializes the old generic B-side temps there.
- The remaining common structural tax is the standalone A-pack launch.

## Key live-code opportunity: `b_shuffle` exists but is ignored by exact HIP fast paths

Input contract at top level:
- `custom_kernel(data)` receives `(a, b, b_q, b_shuffle, b_scale_sh)` at `fp8-mm/submission.py:3772-3773`

But every exact fast path currently routes only `a`, `b_q`, and `b_scale_sh` into the public exact wrappers.
The already-present `b_shuffle` tensor is not used by the exact HIP public fast paths.

Relevant helper already present but currently unused by the exact public path:
- `_get_b_preshuffled_mfma_fp4(...)`: `fp8-mm/submission.py:3631-3638`

This is the strongest dormant asset in the current contract:
- it is already part of the live tuple,
- it can delete raw-B address shaping / repack-style logic inside a new exact kernel,
- and it pairs naturally with a direct BF16-A exact kernel that avoids the standalone A-pack helper.

## CDNA4 ISA facts that matter here

Primary ISA reference:
- `important_papers/amd-instinct-cdna4-instruction-set-architecture.txt`

### 1) The right scaled MFMA families are already known
Relevant instructions:
- `V_MFMA_SCALE_F32_16X16X128_F8F6F4`
- `V_MFMA_SCALE_F32_32X32X64_F8F6F4`

Reference:
- `...instruction-set-architecture.txt:6825-6829`

Scale contract facts:
- scale format is E8M0
- scaling is per K-block of 32
- `ABID[0]` must be set for these scale MFMA ops

Reference:
- `...instruction-set-architecture.txt:6151-6157`
- `...instruction-set-architecture.txt:6839-6859`

Implication:
- do not spend the next slot on MFMA-family churn.
- keep the current scaled-MFMA families and delete feed/runtime law around them.

### 2) CDNA4 exposes direct BF16<->FP4 scale-pack converts
BF16 -> packed FP4:
- `V_CVT_SCALEF32_PK_FP4_BF16`
- semantics described at `...instruction-set-architecture.txt:26026-26038`

Packed FP4 -> BF16:
- `V_CVT_SCALEF32_PK_BF16_FP4`
- semantics described at `...instruction-set-architecture.txt:26102-26115`

These are the ISA-supported bridge needed for a direct BF16-A / preshuffled-FP4-B exact kernel family.
They are the key reason the next aggressive lane can be a single exact kernel rather than another helper-centric design.

### 3) Direct memory-buffer -> LDS and FP4 transpose loads exist, but only matter if reused
Relevant facts:
- MUBUF can load directly into LDS, bypassing VGPRs
- FP4 transpose loads from LDS exist, including `DS_READ_B64_TR_B4`

Reference:
- `...instruction-set-architecture.txt:10246-10254`
- `...instruction-set-architecture.txt:11265-11269`
- `...instruction-set-architecture.txt:11428-11451`

Implication:
- this is a second-stage wide-shape lever after the main two-launch law is deleted.
- do not start with a new LDS-heavy service path unless it clearly replaces real work instead of duplicating it.

### 4) Lane-permute ops are real, but they are local repair tools, not a global answer
Relevant facts:
- `DS_SWIZZLE_B32`
- `DS_PERMUTE_B32`
- `DS_BPERMUTE_B32`
- `V_PERMLANE16_SWAP_B32`
- `V_PERMLANE32_SWAP_B32`

Reference:
- `...instruction-set-architecture.txt:30282-30345`
- `...instruction-set-architecture.txt:30399-30405`
- `...instruction-set-architecture.txt:30454-30460`
- `...instruction-set-architecture.txt:17389-17414`

Implication:
- use them only for narrow per-wave repair after a structural feed deletion lands.
- do not reopen broad lane-routing experiments as the primary bet.

### 5) Hazard/scheduling rules matter for a direct-convert exact kernel
Relevant facts:
- MFMA is not single-cycle and has operand/result hazard windows
- packed convert writes need spacing
- OPSEL/SDWA users need an independent instruction gap
- generic VGPR-touching interference after DL ops is expensive

Reference:
- `...instruction-set-architecture.txt:4563-4565`
- `...instruction-set-architecture.txt:7342-7346`
- `...instruction-set-architecture.txt:3176-3179`
- `...instruction-set-architecture.txt:8814-8819`
- `...instruction-set-architecture.txt:8840-8847`

Implication:
- if a direct BF16-A / FP4-B exact kernel is attempted, schedule pointer math, scale loads, or unrelated address work into those hazard windows.
- do not judge the lane only by instruction count; schedule legality and VGPR lifetime will decide whether it wins.

## Dead lanes: what recent evidence already killed

### Host-only or wrapper-only collapse is too small
Examples:
- `owned-workspace-cpp-return-r1` -> ~13.578 us
- `owned-workspace-nocheck-r1` -> ~13.551 us

See:
- `../../HISTORY.md:153-168`

### Helper arithmetic micro-edits are dead
Examples:
- `helper-bf16-srcdomain-r1` -> hard regression
- `helper-fastapprox-ws-cache-r1` -> regression to ~13.584 us

See:
- `../../HISTORY.md:85-100`
- `../../HISTORY.md:119-134`

### Python-side exact-path graph replay is dead on the current kernelbot MI355X runner
Example:
- `exact_public_graph_replay_r1` -> test green after a stream fix, benchmark catastrophic at ~1.37 s per shape because HIP graph capture produced empty graphs (`torch/cuda/graphs.py: CUDA Graph is empty`)

Interpretation:
- Treat Python-side `torch.cuda.CUDAGraph`/HIP graph replay as a hard veto for this frontier unless a runner-specific proof shows non-empty graph capture.

### Thin producer/consumer synchronization is dead
Example:
- `m4-oneshot-nodup-r1fix` -> ~15.146 us, severe m4 regression

See:
- `../../HISTORY.md:102-117`

### Per-CTA fused A-pack is structurally dead in current forms
Examples:
- all-shape fused-A branch -> severe regression
- `fused_a_in_kernel_r1` -> catastrophic regression
- persistent N-loop fused A-pack -> catastrophic regression
- exact m16 bundle2 reuse-aware fused-A -> catastrophic regression

See:
- `../../HISTORY.md:34-66`
- `../../HISTORY.md:225-240`
- `../../HISTORY.md:320-335`

Interpretation:
- The user’s instinct about bandwidth/occupancy/synchronization/register movement is correct, but the important part is the law of duplicated quant work.
- If A-pack work scales with output-tile count, the branch is structurally wrong for this problem family.

## High-confidence patch ladder

The ladder below is intentionally aggressive and shape-scoped.
It is designed around deleting a whole call family, not saving 0.05 us in helper math.

### Branch 1: `m32_singlelaunch_bpreshuffle_nodup_apack_k512`
Deleted cost center:
- standalone A-pack launch + `a_packed/a_scale` temp law on the public m32 shapes
- residual raw-B address shaping by switching the exact kernel to the already-available `b_shuffle` contract
- implementation must preserve one-pack-per-call economics; no per-CTA fused-A fallback is allowed

Touched regions:
- `fp8-mm/submission.py:3812-3826`
- `fp8-mm/submission.py:2986-3043`
- `fp8-mm/submission.py:3631-3638`
- builtin scaffold from `fp8-mm/hip_phase2_working.py:421,466-473`

Why larger than noise:
- both visible m32 cases benefit
- m32 still has a large wall/self gap
- public exact m32 already deleted most B-side temp laws, so the remaining structural target is clear

Expected gain range:
- about `-3.5 to -5.5 us` on each m32 case
- about `-1.4 to -2.2 us` geomean if both cases land

Fast falsifier:
- if the first prototype needs a new per-call B temp instead of consuming live `b_shuffle`, kill it
- if self CUDA rises above roughly `~2.7 us`, the single-launch thesis is probably gone

### Branch 2: `m16_singlelaunch_bpreshuffle_nodup_apack_k7168_n2112`
Deleted cost center:
- standalone A-pack launch family on the worst thin-shape gap
- implementation must preserve one-pack-per-call economics; no per-CTA fused-A fallback is allowed

Touched regions:
- `fp8-mm/submission.py:3797-3811`
- `fp8-mm/submission.py:1700-1754`
- new direct BF16-A / preshuffled-FP4-B exact kernel

Why larger than noise:
- m16 is the most painful thin-shape outlier
- helper dominates its current profiled self time
- no recent small-step branch moved this shape enough

Expected gain range:
- about `-7 to -10 us` on m16
- about `-1.2 to -1.8 us` geomean

Fast falsifier:
- if the new exact kernel self rises above roughly `~3.3 us`, the one-launch gain is probably not enough

### Branch 3: `m64_singlelaunch_bpreshuffle_nodup_apack_k2048_n7168`
Deleted cost center:
- standalone A-pack launch + A temp law on public m64
- implementation must preserve one-pack-per-call economics; no per-CTA fused-A fallback is allowed

Touched regions:
- `fp8-mm/submission.py:3827-3841`
- `fp8-mm/submission.py:3097-3150`
- existing rawscale m64 kernel family

Why larger than noise:
- m64 wall time is ~18.1 us while profiled self is only ~0.958 us
- that is the strongest evidence that a whole-call launch/runtime law is still dominating

Expected gain range:
- about `-6 to -8 us` on m64
- about `-1.0 to -1.5 us` geomean

Fast falsifier:
- if the new direct kernel becomes decode-heavy and loses occupancy, kill it quickly

### Branch 4: `m256_singlelaunch_bpreshuffle_nodup_apack_k1536_n3072`
Deleted cost center:
- standalone A-pack launch + A temp law on public m256
- implementation must preserve one-pack-per-call economics; no per-CTA fused-A fallback is allowed

Touched regions:
- `fp8-mm/submission.py:3842-3856`
- `fp8-mm/submission.py:3208-3262`
- existing fixed-domain m256 scale-load helpers

Why larger than noise:
- m256 has the same signature as m64: very large wall/self gap with already-flattened B-side public path

Expected gain range:
- about `-5 to -7 us` on m256
- about `-0.8 to -1.3 us` geomean

### Branch 5: `m4_singlelaunch_bpreshuffle_nodup_apack_k512`
Deleted cost center:
- standalone A-pack launch + `a_packed/a_scale` temp law on public m4
- implementation must preserve one-pack-per-call economics; no per-CTA fused-A fallback is allowed

Touched regions:
- `fp8-mm/submission.py:3778-3796`
- `fp8-mm/submission.py:1799-1852`

Why larger than noise:
- m4 still spends most of its total call time outside the current compute body
- recent m4 one-shot negative shows the fix must avoid producer/consumer serialization

Expected gain range:
- about `-3 to -5 us` on m4
- about `-0.5 to -0.9 us` geomean

### Branch 6: `wide_post_delete_body_retune_32x32x64`
Deleted cost center:
- residual per-iteration scale unpack/address recompute and suboptimal issue pattern after A-pack launch deletion lands on wide shapes

Touched regions:
- wide exact MFMA helpers and scale-load helpers
- only after at least one of branches 1/3/4 is a real win

Expected gain range:
- about `~0.4 to 1.0 us` extra geomean after structural deletion

## Recommended order

1. `m32_singlelaunch_bpreshuffle_nodup_apack_k512`
2. `m16_singlelaunch_bpreshuffle_nodup_apack_k7168_n2112`
3. `m64_singlelaunch_bpreshuffle_nodup_apack_k2048_n7168`
4. `m256_singlelaunch_bpreshuffle_nodup_apack_k1536_n3072`
5. `m4_singlelaunch_bpreshuffle_nodup_apack_k512`
6. wide post-delete retune only after a direct-kernel win exists

Why this order:
- m32 is the safest proof branch because it hits two benchmark cases and can use the live `b_shuffle` asset without the thin-shape duplication traps.
- m16 is the single biggest blocker once the direct-BF16-A thesis is proven once.
- m64/m256 are the strongest “whole-call gap” shapes after m16.
- m4 matters, but it has less portfolio leverage than double-counted m32.

## What would make <=7 us plausible

Using the anchor per-shape numbers, a portfolio roughly in this neighborhood would be enough:
- m4: ~9.9 -> ~4.9 to 5.5 us
- m16: ~19.6 -> ~9.6 to 10.5 us
- m32: ~10.0 -> ~4.5 to 5.0 us on both visible cases
- m64: ~18.1 -> ~10.0 us
- m256: ~16.4 -> ~9.4 us

That is not reachable by helper arithmetic or wrapper cleanup.
It is only reachable if the exact public family stops paying the extra helper launch and temp law.

## Quick vetoes for future work

Do not reopen these as next-slot bets:
- standalone helper arithmetic micro-edits
- host-only workspace/view cleanup as the main thesis
- per-CTA fused A-pack without a true non-duplicating handoff law
- producer/consumer ready-flag one-shot schemes
- lane-permute or scheduling-only campaigns before feed law deletion
- MFMA-family churn while the current wall/self gap is still dominated by non-body cost

## Suggested artifact usage

Pair this note with:
- `docs/research/2026-04-03-mxfp4-mm-7us-portfolio.json`

That JSON records the branch ladder in machine-readable form so future agents can turn it into candidate cards or closed-loop submissions without re-mining the repo.
