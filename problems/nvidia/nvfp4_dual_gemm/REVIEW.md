## Performance Gains

the first ~5 TODOs (warpgroup MMA + fewer epilogue warps, single-warp TMEM alloc/free, right-sized TMEM, remove inner-loop epilogue barriers, cap stages / raise occupancy) are exactly the kind of fixes that can plausibly take you from ~27 µs geo-mean toward ~9 µs.

## TODOs 

- Cut CTA size to 128 threads: current threads_per_cta is 192 (6 warps) (submission.py:85–submission.py:90); restructure so the tcgen05 MMA runs on a full warpgroup and don’t reserve 4 dedicated epilogue warps + 1 dedicated TMA warp.
- Fix TMEM allocation fanout: tmem.allocate(self.num_tmem_alloc_cols) is executed by every epilogue warp (warp_idx < self.mma_warp_id) (submission.py:1098–submission.py:1101); make exactly one warp do allocate/free, others only wait_for_alloc/use the pointer.
- Right-size TMEM allocation: self.num_tmem_alloc_cols = 512 is a hard-coded over-allocation (submission.py:100); compute required cols from num_accumulator_tmem_cols + num_sf_tmem_cols (plus alignment) and allocate just that.
- Remove per-subtile CTA barriers in epilogue: epilog_sync_barrier.arrive_and_wait() is inside the inner subtile loop (submission.py:1183–submission.py:1217); replace with a design where only the store-warp synchronizes (proper mbarrier/TMA-store pipeline), and other warps don’t full-block twice per subtile.
- Stop letting staging consume the whole SMEM budget: _compute_stages() derives num_ab_stage from smem_capacity // occupancy and your configs set occupancy=1 everywhere (submission.py:1321–submission.py:1375, submission.py:16–submission.py:61); cap num_ab_stage (e.g., 3–4) and retune occupancy to target 2 resident CTAs if regs/TMEM allow.

---
## Second phase of TODOs 
- Verify TMA pipeline accounting matches actual bytes: tx_count=self.num_tma_load_bytes is derived from size_in_bytes(sf_dtype, …) while scale TMAs use internal_type=cutlass.Int16 (submission.py:340–submission.py:372, submission.py:432–submission.py:441); confirm the mbarrier tx-count is correct or you can end up with extra waiting/serialization.
- Avoid triple-running the persistent scheduler logic: TMA warp, MMA warp, and epilogue warps each create their own StaticPersistentTileScheduler and walk tiles independently (submission.py:736, submission.py:923, submission.py:1140); refactor so tile ownership is computed once and broadcast (or fuse roles) to cut instruction overhead and reduce divergence risk.

---

## Third phase of TODOs (optional)
- Increase tile arithmetic intensity / reduce tile count: for the benchmark shapes, 128×128 (or 128×64) produces many tiles; introduce per-shape kernels with larger mma_tiler_mn where it fits (e.g., 256×128 for N=4096, 256×64 for N=3072) and only enable 2-CTA tcgen05 when it actually reduces total bytes/tile.
- Revisit cluster/multicast only with a correctness-proof launch rule: current cluster_shape_mn=(1,1) disables multicast; if you try (2,1) to amortize A/SFA traffic, ensure the cluster is compatible with the tiled MMA and your scheduler/layout math (you already saw “cluster shape not divisible…” in MEMORY.md).
- Make the epilogue math cheaper: epilogue_op is x * sigmoid(x) with exp (submission.py:540–submission.py:542); consider a faster sigmoid approximation (still within 1e-3 tolerances) and keep it warp-local/vectorized to avoid SFU bottleneck dominating.
- Remove dead/unused state and over-structure (cleanup, not speed): iter_acc_early_release_in_epilogue is computed but unused (submission.py:235), and several “shifted_ptr” special-cases are legacy for tiles you don’t currently use; delete once configs are finalized to prevent accidental regressions.

