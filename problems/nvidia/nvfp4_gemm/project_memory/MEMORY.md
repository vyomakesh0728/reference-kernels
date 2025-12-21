---
name: nvfp4-gemm memory tracker
description: A memory block to store information about this coding project. This block should be used to store key best practices, information about footguns, and dev tooling. Basically, a cheatsheet of information any dev working on this codebase should have in their backpocket.
trigger: auto
---

## nvfp4_gemm status / notes
- Implemented a full direct port of CUTLASS persistent blockscaled GEMM (mainloop + epilog) into `submission.py`, adapted to competition rules (no cuda graph usage and no usage of the forbidden keyword).
- Correctness: `CUTE_DSL_ARCH=sm_100a python test_correctness.py` passes all 10 cases.
- Best benchmark so far: geometric mean ~21.3 μs (target 3.045 μs) via `CUTE_DSL_ARCH=sm_100a python test_benchmark.py`.
- Current best config in `CONFIGS`:
  - N=7168 shapes: `mma_tiler_mn=(128,192)`, `cluster_shape_mn=(1,1)`, `swizzle_size=1`, `raster_along_m=True`, `occupancy=1`
  - N=4096 shape: `mma_tiler_mn=(128,128)` with same cluster/swizzle/raster/occupancy
- Special-case handling for `cta_tile_shape_mnk[1] == 192` in SFB layout and TMEM offset (copied from the CUTLASS example) is required for correctness.
- Tried and found worse: `cluster_shape_mn=(1,2)` or `(1,4)`, `swizzle_size=2/4`, `occupancy=2`, `mma_tiler_mn=(128,256)`, and `raster_along_m=False`.

## Current debugging hypotheses
