## Performance Gains

You're still far off the "speed-of-light" band for the 4 benchmark shapes (SoL per-shape is ~2.1-8.7 µs). The next gains won't come from tiny tweaks, but from better per-shape tiling/cluster configurations and getting back overlap via accumulator staging and early release, plus a bit of epilogue vectorization.

### What to push next (highest ROI first)

1. **Bring back `num_acc_stage = 2` for the 128×128 family**

   In `nvfp4_gemm.py` you compute `num_acc_stage = 2` unless `N==256`, which enables overlap and early release in the epilogue pipeline. In your current `submission.py`, `num_acc_stage` is hard-pinned to `1`, which typically kills overlap and makes the kernel behave much more "serialized". Fix: mirror the single-GEMM rule (2 stages when not in the 2-CTA/large-N regime). This is often a big step toward SoL.

2. **Add per-shape configurations**

   Right now your config selection is too coarse. In the single GEMM you used multiple tuned configs (different `mma_tiler_mn`, `cluster_shape_mn`, `swizzle_size`, raster direction) based on the shape of the tensors. For Blackwell, CUTLASS explicitly leans on persistent tile scheduling and cluster shape selection for performance. NVIDIA Docs +1. Even provides a "preferred cluster" example. GitHub +1.

   Practical: create 3-4 configs keyed to the 4 benchmark shapes' `(N,K)`, similar to your `nvfp4_gemm.py` table.

If you do just two things next: (1) restore `num_acc_stage=2` where appropriate and (2) add per-benchmark configs with larger clusters/2-CTA UMMA where viable. That’s the most realistic path from ~33 µs toward the single-digit µs regime.