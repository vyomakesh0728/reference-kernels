---
name: nvfp4-gemm memory tracker
description: A memory block to store information about this coding project. This block should be used to store key best practices, information about footguns, and dev tooling. Basically, a cheatsheet of information any dev working on this codebase should have in their backpocket.
trigger: auto
---

## nvfp4_gemm status / notes
- We are rewriting the tcgen05 (SM100/B200) epilogue to get correctness vs `reference.py` (torch._scaled_mm) while using `sm100_bf16_gemm.cuh` only as an *oracle* for the TMEM address-walk / fence pattern.
- Important: `sm100_bf16_gemm.cuh` is a **BF16** GEMM reference; our kernel in `submission.py` is **NVFP4 block-scaled** GEMM. Use the BF16 kernel only to copy low-level patterns (TMEM load/fence, barrier choreography), not datatype/layout/math semantics.
- Key pitfall: the lane/bank-group swizzle math in `sm100_bf16_gemm.cuh` is for **SMEM/TMA layout**, not the mathematical row-major output used by `reference.py`. Using it directly for global `gn` caused massive OOB writes (can show up as illegal memory access on one cloud and inf/-inf/mismatches on another).
- Current direction: keep TMEM load pattern (STORE_BLOCK_N=32, kNumElemsPerBankGroup=4, kNumStores=4, fence after each TMEM load), but compute global indices in row-major: `gm = m_tile + (warp_id*32 + lane_id)` and `gn = n_tile + (w*BLOCK_N + s*STORE_BLOCK_N + i*kNumElemsPerBankGroup)`.
- We previously hit multiple syntax/bracing issues in `submission.py` while editing epilogue; brace mismatches can manifest as nvcc parse errors. Ensure braces balance after edits.
- Different outputs across clouds likely indicate undefined behavior from OOB writes / wrong addressing; fix bounds/indices first before tuning.
- Process rule: after every patch, record the change summary here.
- Latest patch: scale compaction now direct byte copy from TMA tiles (no load_sf_tile_byte_2048), and tcgen05 mbarrier commit moved inside warp<4 before wait.
- Latest patch: scale compaction loop restored using load_sf_tile_byte_2048 with packed16 index mapping (mm32/packed16/kb/mm4/kk4 → row_global/scale_col).
- Latest patch: UMMA SMEM descriptors now interpret A/B as packed bytes (uint8) with TileKPacked and kKBlockPacked to match packed FP4x2 layout.
- Latest patch: UMMA SMEM tensor shapes now match TMA dim order (Kpacked, M/N), and K-block tiling slices along mode-0.
- Latest patch: UMMA SMEM layouts switched to (Kpacked, M/N) with K-block slicing on mode-0 for A/B descriptors.
- Latest patch: local_tile coord uses runtime `make_coord(kb, 0)` to fix non-constexpr `Int<kb>` compile error.
- Latest patch: scale compaction uses explicit bitwise packed16 mapping with a computed dst_idx for clarity.
- Latest patch: added NVFP4_DEBUG_DUMP prints for idescE and first 32 bytes of sfa_stage/sfa_compact/sfb_compact.
- Latest patch: added `--only` flag to `test_correctness.py` to run a single 1-based test case.
- Latest patch: `test_correctness.py --only` now prints first 32 bytes of `to_blocked` SFA/SFB for layout comparison.
- Latest observation: sfa_compact/sfb_compact match to_blocked for first 32 bytes; likely remaining mismatches are from A/B descriptor layout or MMA/epilogue path.
- Latest patch: `test_correctness.py` adds `--debug-umma` to set `NVFP4_DEBUG_DUMP=1` and print first 32 packed bytes of A/B.
- Latest patch: fix `--debug-umma` A/B byte print by flattening to 1D before formatting.
- Latest patch: added NVFP4_DEBUG_DUMP prints for desc_a_smem_sh[0] and desc_b_smem_sh[0].
- Latest patch: added NVFP4_DEBUG_DUMP prints for A/B SMEM base addresses and D[0..7] epilogue values.
- Latest patch: added NVFP4_DEBUG_DUMP stride deltas for A/B SMEM and first TMEM load values before epilogue write.
- Latest patch: switched UMMA A/B layout to cutlass::detail::float_e2m1_unpacksmem_t with TileK/kKBlock to satisfy canonical 4-bit layout constraints.
- Latest patch: switched A/B UMMA layout atom to Layout_K_SW64_Atom for 4-bit canonical tiling with TileM/TileN=128.
- Latest patch: added NVFP4_DEBUG_DUMP prints for sA_full/sB_full nibble values and packed stage bytes for k=0 on CTA(0,0,0).
- Latest patch: sA_full/sB_full debug now prints float values (subbyte_reference has no raw()).
- Latest patch: sA_full/sB_full debug now uses subbyte_reference.get().raw() for nibble print.
- Latest patch: A/B SMEM layout order switched to (M/N, K) with k-block slicing along K (mode-1).
- Latest patch: UMMA A/B SMEM descriptors now use uint8_t with (M/N, Kpacked) shape and kKBlockPacked slicing (packed-byte units) plus updated debug prints for k-byte values.
- Latest patch: epilogue debug now prints gm/gn mapping with TMEM values, and test_correctness prints ref D[0,0..3] under --debug-umma for mapping checks.
- Latest patch: idescE now built per K-block using tmem_sfa_kb_sh/tmem_sfb_kb_sh, with debug decoding of format/major/sf_id bits and tmem scale addresses.
- Latest patch: UMMA instruction descriptor now uses float_e2m1_unpacksmem_t for A/B to set a_format/b_format to MXF8F6F4 E2M1 (5).
- Latest patch: UMMA A/B SMEM descriptors reverted to 4-bit element layout (float_e2m1_unpacksmem_t, TileK/kKBlock) to match MMA element size; debug prints reverted to nibble view.
- Latest patch: MmaOp now uses float_e2m1_unpacksmem_t for A/B to keep MMA traits consistent with idescE E2M1 format.
- Latest patch: added debug prints for MMA scale vector/opcode and TMEM scale addresses; removed TMEM load debug that caused a hang.
- Latest patch: TMA SFA/SFB box now loads full 2048B per tile (rest_k chunks), and UMMA A/B SMEM descriptor slicing uses kKBlockPacked.
- Latest patch: SFA/SFB TMA now issued as a single 2048B 4D load per tile (no 4x512 overflow), and added NVFP4_DEBUG_DUMP prints for SFA/SFB sizes/strides and kScaleChunksPerTile.

## Current debugging hypotheses
- SFA/SFB TMA uses rank-4 (packed16, mm32, rest_m/n, rest_k) because inputs are strided atom-tiled layouts (`sfa_permuted/sfb_permuted`); flattening to rank-2/1D is not inherently simpler and risks wrong re-linearization.
- Persistent inf/-inf across clouds suggests systematic issues in tcgen05 path (e.g., MMA instruction descriptor bits or SMEM→TMEM scale copy), not minor numeric drift.
