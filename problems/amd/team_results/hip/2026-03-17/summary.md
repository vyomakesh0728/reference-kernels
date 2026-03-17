# HIP Pure-Path Recovery - 2026-03-17

This snapshot records the first major recovery after enforcing strict purity rules on `mxfp4_mm`.

## Candidate

- Promoted submission: `.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py`
- Problem: `mxfp4_mm`
- Strategy: real scaled-MFMA with compiled A-pack, compiled B repack, and compiled B-scale unshuffle

## Why A New Base Was Needed

- The earlier `v34` snapshot was fast, but impure under the current rules.
- It used cross-call B-side caches keyed by tensor identity, including `data_ptr()`, to reuse prior contract work.
- That made `v34` invalid as a promotion target for purity-compliant optimization.

## What Passed

- `test` mode: success on MI355X
- `benchmark` mode: success on MI355X

## Workflow Links

- `test`: `23190736962`
- `benchmark`: `23190838773`

## Benchmark Result

- geometric mean: `46.964 us`
- prior pure baseline (`v40`): `54.890 us`
- improvement vs prior pure baseline: about `14.4%`
- old impure `v34`: `32.946 us`

Per-shape means from the promoted pure benchmark:
- `m=4, n=2880, k=512`: `26.9 us`
- `m=16, n=2112, k=7168`: `108 us`
- `m=32, n=4096, k=512`: `30.8 us`
- `m=32, n=2880, k=512`: `29.1 us`
- `m=64, n=7168, k=2048`: `79.4 us`
- `m=256, n=3072, k=1536`: `51.9 us`

## What Changed

- `v40` already moved packed-B rebuild into compiled HIP and established the first correctness-green pure baseline.
- `v41` tried a dedicated no-LDS two-wave wide kernel for `m>=64`, but regressed and was discarded.
- The winning `v44` fix moved the remaining Python/Torch B-scale unshuffle into compiled HIP:
  - `mxfp4_unshuffle_b_scale(...)`
- The key detail was matching the exact linearized `view(...).permute(...).contiguous()` layout used by the Python path, rather than a simplified block formula.

## Current Regime Map

- `m=4,8,16`: native scaled-MFMA on the `16x16x128` family
- `m>=32` benchmark/test regimes: native scaled-MFMA on the direct `32x32x64` family
- No pointer-keyed contract cache is used in the promoted path

## What We Learned

- The remaining legal bottleneck is still fixed overhead around prep and dispatch, not operand legality.
- Thin shapes are especially sensitive to Python-side work.
- Wide-path scheduling changes should stay small and evidence-driven; the first simple two-wave remap was not a win.

## Next Direction

- Lane A next: collapse thin-family prep plus launch into a single compiled wrapper for `16x16x128`
- Lane B next: do the same wrapper fusion around the unchanged `32x32x64` core before retrying wave-remap ideas
- Only after that should we revisit deeper CDNA4 scheduling such as multi-wave overlap or ping-pong
