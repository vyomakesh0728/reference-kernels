
You are editing ONLY `submission.py`

Goal: lift the proven performance techniques from `nvfp4_gemm.py` (single NVFP4 GEMM ~12us) and apply them to the dual-GEMM kernel in `submission.py` to move from ~56us geom mean toward the task.yml target, WITHOUT breaking correctness vs `reference.py`.

Hard constraints:
- Do NOT change Python-side I/O, tuple structure, or eval harness behavior.
- Keep the same kernel interfaces, tensor shapes, and scale semantics.
- Do NOT introduce any new CUDA concurrency APIs or any banned keywords from SKILL.md.
- Preserve the existing “permuted” FP8 scale layout contract: consume sfa_permuted/sfb1_permuted/sfb2_permuted exactly (no extra permutation/compaction in the hot path unless byte-for-byte identical).

What to lift (copy the exact playbook from `nvfp4_gemm.py`):
1) Scheduler + tiling:
   - Copy the same Persistent tile-scheduler parameterization and defaults:
     cluster_shape_mn, swizzle_size, raster_along_m, and any static grid mapping logic.
   - Ensure the dual-GEMM uses the same work distribution and CTA rasterization scheme.

2) Pipeline structure:
   - Mirror `nvfp4_gemm.py`’s TMA→SMEM staging strategy (num_ab_stage / num_acc_stage),
     descriptor prefetch placement, and barrier usage patterns.
   - Match the same minimal synchronization structure (don’t add extra __syncthreads).

3) Mainloop reuse (dual-GEMM specific):
   - Reuse A exactly once per kblock for both GEMMs: load A fragments once, then run MMA twice
     (B1 path → Acc1, B2 path → Acc2) using the same A fragment.
   - Keep two independent FP32 accumulator tiles in TMEM, but avoid duplicating A-side loads/copies.

4) Scale placement:
   - Use the same TMEM scale placement approach as `nvfp4_gemm.py` (SFA then SFB),
     but extended for SFB1 and SFB2. Ensure offsets/col placement are consistent and non-overlapping.

5) Epilogue:
   - Copy `nvfp4_gemm.py`’s TMEM→reg load strategy and any epilogue tiling/copy atom choice.
   - Fuse silu(Acc1) * Acc2 and cast to FP16 as late as possible; no extra global round-trips.

Implementation approach:
- Open `nvfp4_gemm.py` and identify the exact sections implementing (a) tile scheduler params,
  (b) descriptor prefetch, (c) pipeline + barriers, (d) mainloop fragment/copy structure,
  (e) TMEM accumulator + scale placement, (f) epilogue load/store.
- Diff against `submission.py` and transplant these mechanisms into the dual-GEMM path.
- Keep correctness first: must match `reference.py` numerics (rtol/atol 1e-3). Large-magnitude mismatches usually mean scale layout/placement mismatch—do not “fix” by adding expensive repacking.

Deliverable:
- Output a single updated `submission.py` (no other files, no extra commentary).
- Prefer minimal, surgical changes: transplant exact techniques rather than inventing new ones.
