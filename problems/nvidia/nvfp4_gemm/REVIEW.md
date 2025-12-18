## Status

- The `sfa_permuted/sfb_permuted` stride fix + rank-4 packed16 TMA view is correct for the atom-tiled physical layout.
- The tcgen05 scale path is the most likely remaining correctness blocker when outputs are “deterministically wrong everywhere”.

## Implemented Fixes (Correctness First)

### 1) Remove manual TMEM replication

- `tcgen05.cp ... 32x128b.warpx4` is a broadcast op; the previous `SF_REP/SF_SUBPART_DPS` loop was redundant and could corrupt TMEM.
- The kernel now issues exactly one `tcgen05.cp...warpx4` per 512B scale panel per k-block.

### 2) Mirror CUTLASS TMEM placement (no hardcoded “256”)

- TMEM scale placement now follows CUTLASS’ `find_tmem_tensor_col_offset` scheme:
  - `tmem_sfa_base = tmem_c + find_tmem_tensor_col_offset(tCtAcc)`
  - `tmem_sfb_base = tmem_sfa_base + find_tmem_tensor_col_offset(tCtSFA)`
- The per-kblock `tsfa_addr/tsfb_addr` used by `tcgen05.mma` is derived from `tmem_sf_frg` tensor slices, not linear column math.
- Note: CUTLASS’ SM100 block-scaled descriptors expect scale type `float_ue4m3_t` (ScaleFormat UE4M3), not `float_e4m3_t`.

### 3) Undo TMEM address bit-splicing

- Scale TMEM destinations for UTCCP and scale TMEM operands for MMA now use the raw CuTe `tmem_ptr` slice addresses (no `(base_hi | col_lo)` reconstruction).

### 4) Use UMMA-compatible SMEM descriptors for UTCCP + MMA

- Scale copy source descriptors now use a SM100 UMMA descriptor encoding (Blackwell `version=1`) via `make_umma_smem_desc_addr` instead of the old scale-only descriptor helper.
- A/B SMEM descriptors are no longer built from raw pointer offsets; they are derived from canonical CuTe UMMA layouts via `UMMA::make_umma_desc<UMMA::Major::K>` on per-K-block tensor slices.
- A/B TMA tensor maps now use `CU_TENSOR_MAP_SWIZZLE_128B` so the SMEM physical layout matches `UMMA::Layout_K_SW128_Atom` addressing.

### 5) Use CTA-wide TMEM accumulator tensor (epilogue correctness)

- The TMEM accumulator is now constructed as a CTA-wide MMA-partitioned tensor via `partition_shape_C(...)` + `FrgTypeC` (not `partition_fragment_C`), so epilogue reads cover the full `TileN` instead of only the first MMA sub-tile.

## Next Suspects (If Mismatches Persist)

- A/B descriptor semantics: confirm the chosen `leading_byte_offset` and `stride_byte_offset` match what `tcgen05.mma...mxf4nvf4.block_scale.block16` expects for packed FP4 in SMEM.
- SMEM swizzle/layout: if UMMA expects a swizzled/interleaved shared layout for A/B, update the TMA tensor maps and descriptors to use that canonical layout (avoid “linear SMEM + guessed descriptor”).
- Lane participation: gating UTCCP + MMA to `lane_id == 0` is not how CUTLASS issues these ops (`cute::elect_one_sync()` is warp-uniform). If results are still wrong, change the issuer condition to be warp-uniform (e.g. `warp_id == 0` with `elect_one_sync()` semantics).
- Instruction descriptor: if needed, switch to CUTLASS’ `UMMA::make_runtime_instr_desc_block_scaled` to derive SF IDs from the actual TMEM addresses.
