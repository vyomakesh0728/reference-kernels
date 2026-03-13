# HIP Correctness Milestone - 2026-03-13

This snapshot records the first green HIP reference for `mxfp4_mm`.

## Candidate

- Tracked HIP template path: `fp8-mm/template-hip.py`
- Search seed path: `agent_loop/kernel_mutator.py`
- Manual reference candidate label: `hip-reference-a-bcalibrated-bq-rawscale`

## What Passed

- `test` mode: success on MI355X
- `benchmark` mode: success on MI355X

## Workflow links

- `test`: `23039639369`
- `benchmark`: `23039713999`

## Correctness note

- The green path is correctness-first, not leaderboard-ready.
- It reconstructs calibrated logical `A_ref/B_ref` in Python, then runs a tiled HIP GEMM body.
- This was the first proof that we can match the live `amd-mxfp4-mm` reference with a HIP submission.

## Benchmark note

- Current benchmark geometric mean is about `7.91 ms`.
- This is intentionally slow compared with the ranked anchor because it preserves correctness first and still pays heavy Python-side reconstruction cost.

## Next direction

- Remove or shrink Python-side reconstruction while preserving exact outputs.
- Then optimize ingress, LDS layout, double buffering/swizzle, and move toward CDNA4 scaled MFMA on `gfx950`.
