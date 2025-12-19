# Epilogue Address Calculation Plan

## Understanding the Reference Pattern (sm100_bf16_gemm.cuh)

### Reference Setup:
- **STORE_BLOCK_N** = kSwizzleCDMode / sizeof(cd_dtype_t) = 128 / 4 = 32 (for FP32)
- **kNumElemsPerBankGroup** = 16 / sizeof(cd_dtype_t) = 16 / 4 = 4 (for FP32)
- **kNumStores** = BLOCK_N / STORE_BLOCK_N = 128 / 32 = 4
- **kNumMWaves** = BLOCK_M / 128 = 1 (for BLOCK_M = 128)
- **kSwizzleCDMode** = 128 bytes
- **kHasShortcut** = (128 / 16) == 8 = true

### Reference Flow:
1. **TMEM → SMEM** (swizzled layout)
   - Reads from TMEM using address: `accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup`
   - Writes to SMEM using swizzled row/col calculation
   
2. **SMEM → Global** (via TMA)
   - TMA handles the layout conversion from swizzled SMEM to row-major global

### Our Problem:
We want to skip SMEM and write **directly from TMEM → Global**.

## TMEM Layout for tcgen05.mma Accumulator

The accumulator in TMEM is organized as:
- **128 columns** (for TileN=128, kNumMWaves=1, accum_stage_idx=0)
- Each column contains **FP32 values** for one output element

### TMEM Address Mapping:
For output position (m, n) in the tile:
- We need to figure out which TMEM column holds that result

## Thread Assignment in Epilogue

### Reference Uses:
- **epilogue_warp_idx**: Which warp in the epilogue warpgroup (0-3 for 4 warps)
- **lane_idx**: Lane within warp (0-31)

Each thread in the epilogue processes:
- Loops: w ∈ [0, kNumMWaves), s ∈ [0, kNumStores), i ∈ [0, STORE_BLOCK_N/kNumElemsPerBankGroup)
- That's 1 × 4 × 8 = 32 iterations per thread
- Each iteration loads 4 FP32 values
- Total: 32 × 4 = 128 values per thread

But we have 4 warps × 32 threads = 128 threads, each loading 128 values = 16384 total values.
For a 128×128 tile, we have 128 × 128 = 16384 elements. ✓

## Correct Mapping: TMEM Column → Output (m, n)

### Step 1: TMEM Address to Column Index
```
tmem_addr = tmem_c + accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup
tmem_column = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup
```
For our case (accum_stage_idx=0, kNumMWaves=1):
```
tmem_column = w * 128 + s * 32 + i * 4
```

### Step 2: Which thread processes which output row?
Looking at reference line 346: `epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32)`

In our case, we have warp_id ∈ [0, 3] for the 4 warps.
- row_base = warp_id * 32 + lane_id gives rows 0-127 ✓

### Step 3: The Critical Issue - Output Column Calculation

The reference calculates `row` and `col` for **SMEM addressing** (swizzled):
```cpp
uint32_t bank_group_index = i + lane_id * (kSwizzleCDMode / kNumBankGroupBytes);  // i + lane_id * 8
uint32_t row = kHasShortcut ? (i / 8 + lane_id) : (bank_group_index / 8);
uint32_t col = kHasShortcut ? i : (bank_group_index % 8);
col ^= row % (kSwizzleCDMode / 16);  // XOR swizzle
```

But these `row` and `col` are for the **shared memory swizzled layout**, NOT for global memory!

## The Fix

### For Direct TMEM → Global Write:

Each thread should write to:
- **M dimension**: `row_base` = warp_id * 32 + lane_id (this gives us 0-127)
- **N dimension**: We need to map the TMEM column to the output N coordinate

The TMEM column for a given (w, s, i) iteration is:
```
n_base = w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup
       = w * 128 + s * 32 + i * 4
```

And the 4 consecutive elements loaded go to columns:
```
n_base + 0, n_base + 1, n_base + 2, n_base + 3
```

### Global Memory Index:
```cpp
int gm = m_tile + row_base;  // NOT using the swizzled 'row'!
int gn = n_tile + (w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup);

D[(gm) * N + (gn + 0)] = __float2half_rn(__uint_as_float(v0));
D[(gm) * N + (gn + 1)] = __float2half_rn(__uint_as_float(v1));
D[(gm) * N + (gn + 2)] = __float2half_rn(__uint_as_float(v2));
D[(gm) * N + (gn + 3)] = __float2half_rn(__uint_as_float(v3));
```

## Summary of Changes Needed:
1. Use `row_base` (not the swizzled `row`) for M dimension
2. Calculate N dimension directly from loop indices (w, s, i)
3. Remove the swizzled row/col calculation as it's only for SMEM
