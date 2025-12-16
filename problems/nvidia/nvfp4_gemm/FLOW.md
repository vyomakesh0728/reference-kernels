--- 
MAIN GEMM tcgen05 FLOW 
----

# tcgen05 GEMM Flow

## Data Preparation (Python)
A, B: torch float4e2m1fn (Torch e2m1_x2 packed; host adjusts \(k = k * 2\))
SFA, SFB: torch float8e4m3fn, passed as sfa_permuted/sfb_permuted (already permuted to expected blocked/atom layout)
C: torch float16 output tensor

## Kernel Flow (per CTA tile)
1. TMA Load (Global → SMEM, pipelined)
cute.copy(TMA atoms):
  A GMEM → sA SMEM (staged layout, num_ab_stage)
  B GMEM → sB SMEM (staged layout, num_ab_stage)
  SFA GMEM → sSFA SMEM (staged layout, num_ab_stage; (optional) filter_zeros/compact view to match tcgen05 Cp4x32x128b expectations)
  SFB GMEM → sSFB SMEM (staged layout, num_ab_stage; (optional) filter_zeros/compact view to match tcgen05 Cp4x32x128b expectations)
Synchronization via PipelineTmaUmma (ab_producer/ab_consumer barriers)

2. TMEM Allocate + Scale Placement (SMEM → TMEM)
Allocate TMEM via TmemAllocator, retrieve FP32 accumulator base pointer
tCtAcc: TMEM tensor (FP32 accum) built from retrieved pointer + acc fragment layout
tCtSFA: TMEM tensor (FP8 scales) placed at col offset = find_tmem_tensor_col_offset(tCtAcc), using make_tmem_layout_sfa(...)
tCtSFB: TMEM tensor (FP8 scales) placed after SFA via additional find_tmem_tensor_col_offset(tCtSFA), using make_tmem_layout_sfb(...)
Copy scales each k_tile:
  tcgen05.make_s2t_copy(Cp4x32x128b) + get_s2t_smem_desc_tensor
  cute.copy: sSFA(stage=ab_full.index) → tCtSFA
  cute.copy: sSFB(stage=ab_full.index) → tCtSFB

3. MMA Mainloop (SMEM + TMEM scales → TMEM accum)
tCrA/tCrB are fragments sourced from SMEM staging (per kblock, ab_full.index)
For each kblock:
  tiled_mma.set(Field.SFA, tCtSFA[...].iterator)
  tiled_mma.set(Field.SFB, tCtSFB[...].iterator)
  cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
ACCUMULATE toggling:
  set(Field.ACCUMULATE, False) for first k_tile, then True after first kblock

4. Epilogue (TMEM → Reg → Global)
tiled_copy_t2r loads tCtAcc (FP32) → registers, then stores to output tensor mC_mnl (FP16)

---