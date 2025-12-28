--- 
MAIN GEMM tcgen05 FLOW 
----

## Execution Roles Mapping

- **TMA warps** – A single CTA warp owns all Tensor Memory Accelerator loads. This warp instantiates the persistent tile scheduler and feeds the ab pipeline so downstream warps see a consistent tile ordering.
- **MMA warpgroup** – Four warps (a full warpgroup) form the MMA compute team. They share the same tile scheduler output, march through kblocks together, and collectively execute every tcgen05 mma tile without gating on a specific `warpidx`.
- **TMEM allocation + TMAstore** – One warp (typically the lead epilogue warp) allocates TMEM (alloc/free) and hosts the TMA store pipeline that writes results back to GMEM. All other epilogue warps wait on the allocation, reuse the pointer, and tap the pipeline stages through the pipeline’s mbarrier semantics.

# tcgen05 GEMM Flow

## Data Preparation (Python)
A, B1, B2: torch float4e2m1fn (Torch e2m1_x2 packed; host adjusts \(k = k * 2\))
SFA, SFB1, SFB2: torch float8e4m3fn, passed as sfa_permuted/sfb1_permuted/sfb2_permuted (already permuted to expected blocked/atom layout)
C: torch float16 output tensor (C = silu(A@B1) * (A@B2))

## Kernel Flow (per CTA tile)
1. TMA Load (Global → SMEM, pipelined)
cute.copy(TMA atoms):
  A GMEM → sA SMEM (staged layout, num_ab_stage)
  B1 GMEM → sB1 SMEM (staged layout, num_ab_stage)
  B2 GMEM → sB2 SMEM (staged layout, num_ab_stage)
  SFA GMEM → sSFA SMEM (staged layout, num_ab_stage; 
  SFB1 GMEM → sSFB1 SMEM (staged layout, num_ab_stage; 
  SFB2 GMEM → sSFB2 SMEM (staged layout, num_ab_stage; 
Synchronization via PipelineTmaUmma (ab_producer/ab_consumer barriers)

2. TMEM Allocate + Scale Placement (SMEM → TMEM)
Allocate TMEM via TmemAllocator, retrieve FP32 accumulator base pointer
tCtAcc1/tCtAcc2: TMEM tensors (FP32 accum) built from retrieved pointer + acc fragment layout
tCtSFA: TMEM tensor (FP8 scales) placed at col offset = find_tmem_tensor_col_offset(tCtAcc1), using make_tmem_layout_sfa(...)
tCtSFB1: TMEM tensor (FP8 scales) placed after SFA via additional find_tmem_tensor_col_offset(tCtSFA), using make_tmem_layout_sfb(...)
tCtSFB2: TMEM tensor (FP8 scales) placed after SFB1 via additional find_tmem_tensor_col_offset(tCtSFB1), using make_tmem_layout_sfb(...)
Copy scales each k_tile:
  tcgen05.make_s2t_copy(Cp4x32x128b) + get_s2t_smem_desc_tensor
  cute.copy: sSFA(stage=ab_full.index) → tCtSFA
  cute.copy: sSFB1(stage=ab_full.index) → tCtSFB1
  cute.copy: sSFB2(stage=ab_full.index) → tCtSFB2

3. MMA Mainloop (SMEM + TMEM scales → TMEM accum)
tCrA/tCrB1/tCrB2 are fragments sourced from SMEM staging (per kblock, ab_full.index)
For each kblock:
  tiled_mma.set(Field.SFA, tCtSFA[...].iterator)
  tiled_mma.set(Field.SFB, tCtSFB1[...].iterator)
  cute.gemm(tiled_mma, tCtAcc1, tCrA, tCrB1, tCtAcc1)
  tiled_mma.set(Field.SFB, tCtSFB2[...].iterator)
  cute.gemm(tiled_mma, tCtAcc2, tCrA, tCrB2, tCtAcc2)
ACCUMULATE toggling:
  set(Field.ACCUMULATE, False) for first k_tile, then True after first kblock

4. Epilogue (TMEM → Reg → Global)
tiled_copy_t2r loads tCtAcc1/tCtAcc2 (FP32) → registers, then applies silu+mul and stores to output tensor mC_mnl (FP16)
  - Avoid CTA-wide barriers per subtile; let the epilogue store warp/pipeline provide the necessary synchronization (per-tile mbarrier semantics) so other warps are never blocked twice per subtile.

---