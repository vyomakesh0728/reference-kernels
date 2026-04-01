# mxfp4_mm MI355X Frontier Research Addendum (2026-04-01)

Scope: durable follow-up to `2026-04-01-mxfp4_mm_mi355x_isa_research.md`.
Goal context: push geomean from ~14.5 us toward <=7 us with policy-compliant exact-shape deletions.

## Current measured frontier

- New best measured benchmark (tiny edge):
  - run: `/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122947-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-benchmark`
  - metrics: `/stages/01_benchmark/parsed_metrics.json`
  - geomean: `14.5091555928 us`
- Previous: `t18` at `14.5096961834 us`
- Delta: about `0.004%` (noise-level).

Per-shape (t20):
- m4: 10.0 us
- m16: 19.7 us
- m32_n4096: 10.2 us
- m32_n2880: 10.1 us
- m64: 23.1 us
- m256: 19.9 us

## What was changed in t20

Candidate: `native_scaled_exact_shape_m4_fixedcost_fastdispatch_t20`
Lane: `m4-fixed-cost-delete`

Code changes:
- Added exact-m4 early fast dispatch in `custom_kernel` that bypasses tracing/ROCTX scaffolding and avoids redundant tensor canonicalization where possible.
- No MFMA body changes.
- No A-pack ownership-law changes.
- No B-scale contract changes.

Validation:
- Preflight: static/purity OK
- Remote test: OK
  - run: `/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122700-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-test`
- Remote benchmark: OK
  - workflow: `https://github.com/gpu-mode/kernelbot/actions/runs/23848620457`

Interpretation:
- This confirms the fixed-cost lane is real but mostly exhausted.
- Additional m4-only fixed-cost polishing is unlikely to deliver meaningful geomean movement.

## ISA-mined optimization opportunities (mapped to this codebase)

### High-confidence

1) m64 address-law deletion (non-naive shuffled-scale path)
- Target regions: `mxfp4_mm_kernel_mfma_scale_exact_m64_rawb`, `mxfp4_mm_hip_mfma_scale_exact_m64_direct_entry`
- Objective: delete standalone b_scale materialization without paying back cost as per-iteration address rebuild.
- Why: m64 still has large `b_scale_decode` share in t18 profile prior.

2) Helper instruction-count cuts in wave A-pack
- Target region: `mxfp4_pack_a_fixed_kernel_wave`
- Objective: reduce lane-exchange + packing overhead (`__shfl` gather chain), keep ownership law unchanged.

3) Vectorized ingress cleanup in exact m4/m16 kernels
- Target regions: exact m4/m16 kernel byte load loops
- Objective: reduce scalar byte move count and address arithmetic.

### Speculative / high-risk

4) Native CDNA4 FP4 conversion builtins in helper
- Must satisfy: replace existing quant work; preserve contract correctness; avoid dual-path register blowup.
- Prior warning signal exists from earlier builtin mismatch attempts.

5) Schedule-hint tuning (`sched_barrier` / `sched_group_barrier` / priority)
- Only after major bucket deletions; not first-order at current profile mix.

## <=7 us quantitative requirement

Baseline vector (us): [10.0, 20.0, 10.1, 10.1, 23.1, 19.8]
- Geomean: `14.509696 us`
- Required global factor to reach 7 us: `0.4824` (about 51.8% average cut)
- Largest contribution to remaining log-gap: `m64 + m16 + m256` (~75%).

Implication:
- Without large wins in m64/m16/m256, <=7 us is not realistic.
- m4 fixed-cost wins are tie-breakers, not trajectory drivers.

## Recommended next branch order

1) exact `m64-address-last` (shuffled-scale address-law deletion done correctly)
2) exact `m256` helper/materialization delete with strict whole-bucket proof
3) exact `m16` only when a whole bucket is named and deletion is larger than noise
4) helper microarchitecture wave-pack improvements that preserve ownership law

## New durable artifacts added in this pass

- `important_papers/amd-instinct-cdna4-instruction-set-architecture.txt`
- `important_papers/HipKittens Fast and Furious AMD Kernels.txt`
- Updated:
  - `.hermes/notes/2026-04-01-mxfp4_mm_mi355x_isa_research.md`
  - `skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md`
  - `skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md`
