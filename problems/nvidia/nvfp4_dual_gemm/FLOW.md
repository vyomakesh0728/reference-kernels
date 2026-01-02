# tcgen05 Dual GEMM Flow

## Roles
- TMA warp: owns the persistent tile scheduler and bulk global to SMEM loads.
- MMA warpgroup: four warps execute tcgen05 MMA for A@B1 and A@B2.
- TMEM alloc + store warp: alloc/free TMEM and run the TMA store pipeline.

## Per-CTA Steps
1. TMA loads A/B1/B2 (packed FP4) and SFA/SFB (FP8) to SMEM using the AB pipeline.
2. S2T copies scales (SMEM FP8 -> TMEM SFA/SFB) via Cp4x32x128b.
3. MMA mainloop: tcgen05.mma.mxf4.block_scale for B1 then B2; ACCUMULATE false for the first kblock only.
4. Epilogue: TMEM FP32 -> regs, apply silu * mul, cast to FP16, TMA store to GMEM.

## Notes
- Use permuted physical scales (sfa_permuted/sfb1_permuted/sfb2_permuted).
- B tensors are already shaped `(n, k, l)`; interpret via layout, no explicit transpose.
