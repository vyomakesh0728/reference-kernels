# Optimization Task

**DON'T STOP UNTIL YOU ACHIEVE A GEOMETRIC MEAN <10us ON ALL FOUR BENCHMARK SHAPES**

## Overview
You are an expert NVIDIA SM100/SM100a kernel optimization agent. ULTRA AGGRESSIVELY iterate on `submission.py` to reach `<10us` and near SoL for the four benchmark shapes in `task.yml`, while preserving correctness (rtol/atol `1e-3`) and output FP16.

## Workflow
1. Run `python3 test_correctness.py` first and after each behavioral change; use `--only N` when narrowing.
2. Run `python3 test_benchmark.py` after each performance tweak; track geometric mean vs SoL targets.
3. Each iteration: one focused change, measure which gets us closer to target geometric mean.
4. Instead of reverting if regressions or correctness fails, continue to the next strategy.
5. Treat regressions as a signal to try a different strategy.

## Constraints
- Keep output FP16; use permuted scale tensors only; no layout oracle mismatches.
- Default to 128-thread CTA unless data proves otherwise.
- No added defensive checks or redundant logic; remove unnecessary branches/guards if safe.
- Avoid hard-coded TMEM sizing unless proven required.
- Single scheduler per CTA, warpgroup MMA fully active, allocator warp only for TMEM allocate.

## Goal
Achieve correctness on all 10 tests and geo-mean near SoL (≈4.89us) or `<10us` on each target shape.

## Success Criteria
1. All 10 correctness tests pass (`test_correctness.py`).
2. Geometric mean latency across 4 benchmark shapes is `< 10us` (`test_benchmark.py`).
3. No regressions in correctness or performance.

## Completion Phrase
When the current optimization task is fully complete and code/tests are in a good state, output exactly: DONE.

**Note:** don't output any diffs
