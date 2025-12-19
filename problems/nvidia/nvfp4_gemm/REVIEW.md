You are still doing rank‑5 SFA/SFB and that is now literally what is breaking: `encode_tma_matrix` for SFB returns code 1, so the SFB descriptor is invalid.[1][2]

## Why this SFB TMA is failing

- The descriptor checks assume SFB has shape `(32,4,rest_n,4,rest_k)` and very specific strides (mm4 stride 4, kk4 stride 1, mm32 stride 16), then pack that to a rank‑4 TMA plus a hardcoded box, exactly as before.[3]
- On at least one benchmark shape, those assumptions do not hold (layout, strides, or box vs tensor size), so `encode_tma_matrix` returns `CUDA_ERROR_INVALID_VALUE` (code 1), giving the runtime error you see.[1][3]

## What to change

- Drop the synthetic rank‑5→rank‑4 packing and instead treat SFA/SFB as **simple scale matrices** for TMA: encode them as rank‑2 (or at most rank‑3) with a box that just covers the contiguous scale region you actually use per tile, matching your `[TileN, SfaBoxK]` / `[TileM, SfaBoxK]` views.[4][3]
- That removes both:  
  - the descriptor creation failure (no fragile stride/shape assertions), and  
  - a major source of logical mismatch that was giving you wrong values from `(0,8,0)` onward.

With the current aggressive CTA/TMA/tcgen05 pipeline and per‑stage MMA waits in place, simplifying SFA/SFB TMA to this “flat scale rows” view is the next correctness step before chasing the ~3.04 µs geom mean.


Given that the mismatch still starts exactly at `(m=0, n=8, l=0)`, the remaining bug is almost certainly in **how scales are mapped to N‑columns (SFB) / epilogue layout**, not in tcgen05 ordering or TMA correctness in general.[1][2]

## What the `(0,8,0)` pattern is telling you

- `(0,0..7,0)` matching but `(0,8,0)` onward wrong means:  
  - The first 8 columns of the first row use the correct scale / accumulator mapping.  
  - Starting at the next 8‑column block, either:  
    - You switch to the wrong SFB scale index for that K‑block, or  
    - The TMEM accumulator → global D mapping starts reading from a misaligned tile.[3][1]
- That aligns with two likely culprits:  
  - **SFB indexing in `process_tile`** (the `sfb_c0`, `SfaBoxK`, `scale_col` arithmetic) vs how SFB is actually laid out and fetched by TMA.  
  - **Epilogue’s `tCtAcc` view vs TMEM layout**: if the tmem copy partitions don’t match your CTA’s \((M,N)\) tile, the first few lanes/columns can line up by accident but later columns shift.[2][3]

## Where to focus next

- Treat SFB as prime suspect: your mismatches begin along **N**, and SFB is the only per‑N structure unique to N‑side scaling. Verifying, for a single tile, that `global_k_scale → sfb_idx → scale_h` matches what the reference kernel uses for the same `(n_global, k)` would directly confirm this. [file:382dc8ae-e8fb-40f7-a174-a461f8f4ac1a][3]
- In parallel, confirm that your `tCtAcc` / `make_tmem_copy` epilogue is using the exact same shape and partitioning as the CUTLASS SM100 nvfp4 example; any deviation there can also produce the “first few lanes OK, rest wrong” pattern you see.[4][2]
