## Performance Gains

• Using nvfp4-dual-gemm-optimizer. Short answer: your 27 µs is dominated by bandwidth + overhead; target ~15 µs then ~9 µs by tightening
  tiles/overhead.

  - Cut CTA size to 128 threads: drop the dedicated TMA warp, shrink epilogue warps (e.g., 2), and keep the MMA warpgroup only; this
    reduces CTA overhead and raises occupancy.
  - Use larger M tiles where M=256/512: try 256x128 for N=4096 and 256x64 for N=3072 to reduce tile count; enable 2‑CTA tcgen05 only when
    it increases arithmetic per load.
  - Increase cluster_shape_mn to (2,1) on N=4096 to multicast A/SFA and halve A traffic; retune swizzle to keep coalescing.
  - Specialize away the runtime SFB layout shifts (the 64/192 branches) by compiling per‑shape kernels so the inner loop is branch‑free.
  - Maximize AB stages (3–4) and keep C stages minimal; overlap TMA→SMEM with MMA to hide latency.


To get **~15 µs → ~9 µs**, the biggest wins in your current DSL kernel are:

* **Kill the per-subtile CTA barrier in the epilogue** (`epilog_sync_barrier.arrive_and_wait()` inside the subtile loop): replace with **proper mbarrier/pipeline signaling** so only the TMA-store warp waits (this alone can be a *huge* chunk of your 27 µs).
* **Make the epilogue cheaper:** compute `sigmoid(acc1)` with a **faster approximation** (or exp2-based fast path) and **vectorize** the rmem ops; keep it fully warp-local (avoid extra fences).
* **Use a larger/clustered schedule to cut DRAM pressure:** set **cluster_shape_mn=(2,1) or (1,2)** and enable multicast where possible so A/B traffic is amortized across CTAs (you’re near memory-bound on these shapes). 
* **Turn on 2-CTA UMMA path for the big tiles** (try `mma_tiler_mn=(256,128)` or `(256,64)` where it fits) and retune swizzle/raster for those cases.
* **Trade stages for occupancy:** cap `num_ab_stage` to what’s needed, push **occupancy=2** (even if fewer stages) if regs/TMEM allow; for memory-edge kernels, 2 resident CTAs often beats deep staging.

If you do only one thing first: **remove that subtile barrier + pipeline the TMA store**. 
