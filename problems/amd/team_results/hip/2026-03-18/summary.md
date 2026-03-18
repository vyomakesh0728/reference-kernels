# HIP Wide-Path Direct-B Breakthrough - 2026-03-18

This snapshot records the next major pure-path `mxfp4_mm` win after `v44`.

## Candidate

- Promoted submission: `.agent-loop/manual/native_scaled_m32_bfrag_direct_v45/submission.py`
- Problem: `mxfp4_mm`
- Strategy: keep the working `32x32x64` scaled-MFMA math stable, but prepack wide-path `B_q` into direct per-lane fragments so the hot wide loop can use contiguous fragment loads instead of runtime nibble extraction/repacking

## What Passed

- `test` mode: success on MI355X
- `benchmark` mode: success on MI355X
- `leaderboard` mode: success on MI355X

## Workflow Links

- `test`: `23231412418`
- `benchmark`: `23231531793`
- `leaderboard`: `23238607447`

## Results

- cluster benchmark geometric mean: `39.355 us`
- leaderboard ranked geometric mean: `40.093 us`
- prior promoted pure baseline (`v44`): `46.964 us`
- improvement vs `v44` benchmark: about `16.2%`

Per-shape means from the winning cluster benchmark:
- `m=4, n=2880, k=512`: `27.2 us`
- `m=16, n=2112, k=7168`: `108 us`
- `m=32, n=4096, k=512`: `26.0 us`
- `m=32, n=2880, k=512`: `24.5 us`
- `m=64, n=7168, k=2048`: `54.7 us`
- `m=256, n=3072, k=1536`: `36.3 us`

Per-shape means from the leaderboard run:
- `m=4, n=2880, k=512`: `27.8 us`
- `m=16, n=2112, k=7168`: `108 us`
- `m=32, n=4096, k=512`: `26.4 us`
- `m=32, n=2880, k=512`: `25.2 us`
- `m=64, n=7168, k=2048`: `55.6 us`
- `m=256, n=3072, k=1536`: `37.4 us`

## What Changed

- `v44` had already moved A-pack, B repack, and B-scale unshuffle into compiled HIP, but still rebuilt wide-path fragments in the hot loop.
- The winning `v45` step was narrower and safer than the regressed `v47` scale rewrite:
  - keep the wide `32x32x64` MFMA contract unchanged
  - keep thin-family behavior unchanged
  - prepack wide-path `B` into the exact fragment layout consumed by the kernel
  - replace runtime nibble extraction/repacking with direct contiguous fragment loads

## What Regressed

- `v47` (`native_scaled_m32_abscale_direct_v47`) stayed correctness-green but benchmarked slower at `41.173 us`.
- The direct A/B-scale ABI rewrite was therefore not promoted.
- `v48` was started as a lower-risk wide load-path cleanup from `v45`, but it was not yet the promoted branch at the time of this summary.

## Workflow/Tooling Added

- Repo-local MI355X optimization skill:
  - `skills/amd-mi355x-kernel-loop/`
- Optimization reference pack under that skill, including:
  - `references/mxfp4-through-v45.md`
  - `references/problem-transfer.md`
  - `references/remote-first-eval.md`
  - `references/repo-map.md`
  - `references/optimization.md`
- Remote-first closed-loop tooling under `agent_loop/`:
  - quota-aware coordinator
  - static preflight worker
  - optional Docker preflight images
  - mutator guardrails to keep sub-agents from spending remote quota directly

## Current Regime Map

- `m=4,8,16`: native scaled-MFMA on the `16x16x128` family
- `m>=32` benchmark/test regimes: native scaled-MFMA on the direct `32x32x64` family
- No pointer-keyed cache or cross-call output/intermediate reuse is used in the promoted path

## What We Learned

- The next pure win did not come from wrapper reshaping alone.
- It came from cutting wide-path contract materialization in the hot loop while preserving the working MFMA semantics.
- Wide-path scheduling is still a later rung; data movement cleanup paid off first.
- Thin shapes remain dominated by fixed overhead, while wide shapes gave the clearest immediate payoff from direct fragment consumption.

## Next Direction

- Keep `v45` as the promoted base.
- Continue rapid remote `test` iteration against `v45` with single-hypothesis aggressive branches.
- Thin lane next:
  - reduce thin-path contract materialization further, especially around direct shuffled inputs
- Wide lane next:
  - keep the `32x32x64` compute core stable
  - continue cutting load/address overhead before retrying wave scheduling
