# HIP Scaled-MFMA Breakthrough - 2026-03-16

This snapshot records the first correctness-green, no-GPUVM-fault, real scaled-MFMA `mxfp4_mm` submission that beats the stable trunk by a wide margin.

## Candidate

- Submission snapshot: `.agent-loop/manual/native_scaled_compiled_a_pack_m8_v34/submission.py`
- Problem: `mxfp4_mm`
- Strategy: native scaled-MFMA with compiled A-pack and thin small-M routes

## What Passed

- `test` mode: success on MI355X
- `benchmark` mode: success on MI355X

## Workflow links

- `test`: `23155422581`
- `benchmark`: `23155558839`

## Benchmark Result

- geometric mean: `32.946 us`
- prior stable trunk: `319.371 us`
- improvement vs stable trunk: about `9.7x`

Per-shape means from the winning benchmark:
- `m=4, n=2880, k=512`: `19.1 us`
- `m=16, n=2112, k=7168`: `77.9 us`
- `m=32, n=4096, k=512`: `22.2 us`
- `m=32, n=2880, k=512`: `22.2 us`
- `m=64, n=7168, k=2048`: `45.3 us`
- `m=256, n=3072, k=1536`: `38.5 us`

## What Changed

- The dominant wall was Python/Triton A-side prep, not the B-side contract path.
- The winning fix moved A quant + correction + MXFP4 repack into a compiled HIP helper:
  - `mxfp4_pack_a_fixed(...)`
- The direct compute paths stayed on real scaled-MFMA:
  - `16x16x128` family for thin regimes
  - `32x32x64` family for `m >= 32`
- `m8` was brought onto the same thin native scaled-MFMA route, removing the remaining served small-M fallback.

## Current Regime Map

- `m=4,8,16`: thin native scaled-MFMA on `16x16x128`
- `m>=32` benchmark/test regimes: native scaled-MFMA on direct `32x32x64`

## Why This Matters

- This is no longer a correctness-first Python reconstruction path.
- It is a real scaled-MFMA submission that is both green and leaderboard-fast.
- The previous end-to-end overhead issue was solved by taking the hot A-pack path out of Python.

## Next Direction

- Promote this snapshot as the new base for the team.
- Then go deeper into CDNA4 scheduling:
  - multi-wave / ping-pong around the correct `32x32x64` core
  - CTA / wave packing tuned to the benchmark `M/N/K` families
