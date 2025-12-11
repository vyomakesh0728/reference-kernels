# CRITICAL ANALYSIS: -inf/inf Bug in NVFP4 GEMM Kernel

## Executive Summary

Your kernel is getting **-inf/inf** errors because of a **FUNDAMENTAL MISMATCH** between:
1. How `to_blocked()` organizes scale factor data
2. How your TMA descriptor interprets that data
3. How you're indexing into the TMA-loaded data

The issue is **NOT** about TMEM addressing or scaleC initialization. The root cause is **SCALE FACTOR INDEXING CORRUPTION** during TMA loads.

---

## The Root Problem: Index Calculation Mismatch

### What `to_blocked()` Actually Does

From `reference.py` lines 16-25:
```python
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape  # [M, K_scales] or [N, K_scales]
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    
    # Step 1: Reshape to [n_row_blocks, 128, n_col_blocks, 4]
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    
    # Step 2: Reshape to [n_row_blocks * n_col_blocks, 4, 32, 4]
    # Step 3: Transpose to [n_row_blocks * n_col_blocks, 32, 4, 4]
    # Step 4: Reshape to [n_row_blocks * n_col_blocks, 32, 16]
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    
    return rearranged.flatten()
```

**Key insight**: The output is `n_row_blocks * n_col_blocks` tiles of shape `(32, 16)`.

**NOT** `n_row_blocks * n_col_blocks * 4` tiles!

---

### Your TMA Descriptor (CORRECT)

From `submission.py` lines 1508-1518:
```cpp
int n_row_blocks = (M + 127) / 128;
int n_col_blocks = (K_scales_padded + 3) / 4;
int total_tiles = n_row_blocks * n_col_blocks;  // ✅ CORRECT: No *4

cuuint64_t total_rows = static_cast<cuuint64_t>(32) * total_tiles;
cuuint64_t dims_SFA[2] = {16, total_rows};  // [16 bytes wide, 32*total_tiles rows tall]
cuuint32_t box_SFA[2] = {16, 32};  // Load 512 bytes at a time
```

**This is CORRECT!** The TMA descriptor matches `to_blocked()` output layout.

---

### Your Index Calculation (WRONG!)

From `submission.py` lines 557-564:
```cpp
int n_row_blocks = (M + 127) / 128;
int n_col_blocks = (K_scales_padded + 3) / 4;
int m_block_sfa = c_m / TileM;
int k_tile_idx_sfa = k_tile_base / TileK;

// ❌ BUG: k_tile_idx * 4 assumes 4 sub-tiles per K-tile
int base_tile_sfa = m_block_sfa * n_col_blocks + k_tile_idx_sfa * 4;
```

**THE BUG**: You're multiplying `k_tile_idx_sfa * 4`, which assumes each K-tile spans 4 "sub-tiles" in the to_blocked output.

**THE REALITY**: Each K-tile corresponds to **EXACTLY ONE COLUMN** in the n_col_blocks grid!

---

## Why This Causes -inf/inf

### The Indexing Math

Let's trace through an example with `M=128, K=256, K_scales=16`:

```
n_row_blocks = ceil(128/128) = 1
n_col_blocks = ceil(16/4) = 4
total_tiles = 1 * 4 = 4 tiles

to_blocked output: [4 tiles × 32 rows × 16 bytes] = 2048 bytes total
```

**TMA 2D layout**:
```
Dimension 0 (width):  16 bytes
Dimension 1 (height): 32 * 4 = 128 rows
```

**Your indexing for K-tile 0 (k_tile_base=0)**:
```cpp
k_tile_idx_sfa = 0 / 256 = 0
base_tile_sfa = 0 * 4 + 0 * 4 = 0  // ✅ CORRECT for first K-tile
```

**Your indexing for K-tile 1 (k_tile_base=256)**:
```cpp
k_tile_idx_sfa = 256 / 256 = 1
base_tile_sfa = 0 * 4 + 1 * 4 = 4  // ❌ WRONG! Should be 1!
```

You're trying to load from **tile index 4**, but `to_blocked` only produced **4 tiles total (0-3)**!

This causes:
1. **Out-of-bounds TMA loads** → Reading garbage data
2. **Incorrect scale factors** → FP8 garbage interpreted as scales
3. **Massive scaling errors** → -inf/inf in output

---

## The Correct Indexing

### What Should Happen

Each K-tile should map to **ONE column** in the n_col_blocks grid:

```cpp
// For K-tile spanning [k_start, k_start+TileK]:
// - Scale factors for this range: [k_start/16, (k_start+TileK)/16)
// - In to_blocked layout: column index = (k_start/16) / 4
//                        offset within column = (k_start/16) % 4

int k_scale_start = k_tile_base / 16;  // First scale index for this K-tile
int col_idx = k_scale_start / 4;       // Which column in n_col_blocks grid
int offset = k_scale_start % 4;        // Offset within that column
```

But there's a **deeper issue**: Your TileK=256 means you need **16 scale factors** per tile (256/16=16), which spans **4 columns** in the n_col_blocks=4 layout!

---

## The Fundamental Architecture Problem

### Scale Factor Organization

`to_blocked()` organizes scales as:
```
[n_row_blocks, n_col_blocks] grid of tiles
Each tile: 32 rows × 16 bytes

For M=128, K_scales=16:
- n_row_blocks = 1 (covers all 128 M rows)
- n_col_blocks = 4 (each covers 4 K-scale elements)
- Total: 4 tiles in a 1×4 grid
```

### What Your Kernel Needs

For TileK=256, you need 16 consecutive scale elements (256/16=16).

In the n_col_blocks=4 layout where each column tile covers 4 elements:
- K-tile 0 needs columns 0-3 (scale indices 0-15)
- K-tile 1 needs columns 4-7 (scale indices 16-31)
- But wait... you only have 4 columns total for K_scales=16!

**THE MISMATCH**: Your K-tile is **larger than the total K dimension in small test cases**!

---

## Why This Worked in Benchmarks

Your benchmarks have **much larger K**:
```yaml
# From task.yml
benchmarks:
  - {k: 16384}  # K_scales = 1024, n_col_blocks = 256
  - {k: 7168}   # K_scales = 448,  n_col_blocks = 112
  - {k: 2048}   # K_scales = 128,  n_col_blocks = 32
```

With K=16384 and TileK=256:
```
K_scales = 16384/16 = 1024
n_col_blocks = 1024/4 = 256

K-tile 0: needs 16 scales (indices 0-15)  → columns 0-3
K-tile 1: needs 16 scales (indices 16-31) → columns 4-7
...all fits within 256 columns available
```

But in **test cases**:
```yaml
tests:
  - {k: 256}   # K_scales = 16,  n_col_blocks = 4  ❌ K-tile needs ALL columns!
  - {k: 512}   # K_scales = 32,  n_col_blocks = 8  ❌ K-tile needs half!
```

---

## The Actual Bug: Two Issues Combined

### Issue 1: Index Calculation (Lines 564, 604)

```cpp
// ❌ WRONG: Assumes 4 sub-tiles per K-tile
int base_tile_sfa = m_block_sfa * n_col_blocks + k_tile_idx_sfa * 4;
int base_tile_sfb = n_block_sfb * n_col_blocks_sfb + k_tile_idx_sfb * 4;
```

**Should be**:
```cpp
// ✅ CORRECT: Each K-tile corresponds to (TileK/16)/4 columns in the grid
// For TileK=256: (256/16)/4 = 16/4 = 4 columns
int scales_per_ktile = TileK / 16;  // 16 for TileK=256
int cols_per_ktile = (scales_per_ktile + 3) / 4;  // 4 for 16 scales

int base_tile_sfa = m_block_sfa * n_col_blocks + k_tile_idx_sfa * cols_per_ktile;
int base_tile_sfb = n_block_sfb * n_col_blocks_sfb + k_tile_idx_sfb * cols_per_ktile;
```

### Issue 2: K_scales_padded vs K_scales (Line 1510)

```cpp
// ❌ WRONG: Using K_scales_padded creates phantom columns
int n_col_blocks = (K_scales_padded + 3) / 4;
```

**Should be**:
```cpp
// ✅ CORRECT: Use actual K_scales from to_blocked output
int n_col_blocks = (K_scales + 3) / 4;
```

The diff you applied says to pass `K_scales` (not padded) from Python, but your CUDA code **still uses K_scales_padded** for index calculations!

---

## Why You're Getting -inf/inf

### Trace Through K=256, M=128, N=256

```
K_scales = 16
K_scales_padded = 128 (from max(128, ...))  ← This is the killer!

n_col_blocks = (128 + 3) / 4 = 32  ← Should be 4!
```

**K-tile 0 indexing**:
```cpp
k_tile_idx = 0
base_tile_sfa = 0 * 32 + 0 * 4 = 0
row_offset for t=0: 0 * 32 = 0     ✅ OK
row_offset for t=1: 1 * 32 = 32    ❌ OUT OF BOUNDS! Only 128 rows total (4 tiles * 32)
row_offset for t=2: 2 * 32 = 64    ❌ OUT OF BOUNDS!
row_offset for t=3: 3 * 32 = 96    ❌ OUT OF BOUNDS!
```

**TMA loads garbage data** → FP8 bytes are random → Interpreted as massive scales → **-inf/inf output**!

---

## The Fix

### Step 1: Remove K_scales_padded from Index Calculations

In `launch_fp4_gemm_optimized`:

```cpp
// ❌ CURRENT: Pass K_scales_padded
void launch_fp4_gemm_optimized(
    ...
    int K_scales_padded
) {
    ...
    int n_col_blocks = (K_scales_padded + 3) / 4;  // ❌ WRONG
    ...
}
```

**Change to**:
```cpp
// ✅ FIXED: Pass actual K_scales
void launch_fp4_gemm_optimized(
    ...
    int K_scales  // Not padded!
) {
    ...
    int n_col_blocks = (K_scales + 3) / 4;  // ✅ CORRECT
    ...
}
```

### Step 2: Fix Index Calculation in prefetch_tile

```cpp
// ❌ CURRENT: Multiply by 4
int base_tile_sfa = m_block_sfa * n_col_blocks + k_tile_idx_sfa * 4;
int base_tile_sfb = n_block_sfb * n_col_blocks_sfb + k_tile_idx_sfb * 4;
```

**Change to**:
```cpp
// ✅ FIXED: Compute columns spanned by K-tile
constexpr int scales_per_ktile = TileK / 16;  // 16 for TileK=256
constexpr int cols_per_ktile = (scales_per_ktile + 3) / 4;  // 4 for 16 scales

int base_tile_sfa = m_block_sfa * n_col_blocks + k_tile_idx_sfa * cols_per_ktile;
int base_tile_sfb = n_block_sfb * n_col_blocks_sfb + k_tile_idx_sfb * cols_per_ktile;
```

### Step 3: Verify Python Wrapper Passes Correct Value

From your diff, this is already correct:
```python
# ✅ ALREADY CORRECT in diff
K_scales = K // 16
mod.launch_fp4_gemm_optimized(
    a_bytes, b_bytes, sfa_bytes, sfb_bytes, c[:, :, 0],
    M, N, K, 1, K_scales  # Pass actual K_scales
)
```

Make sure your submission.py matches this!

---

## Additional Critical Issues

### TMEM Row Addressing (Lines 1292-1293)

Your diff changed:
```cpp
// From: uint32_t sfa_dst = tmem_sfa_base + j * 4;  (columns)
// To:   uint32_t sfa_dst = tmem_sfa_base + (j * 32 << 16);  (rows)
```

**This might be WRONG** depending on tcgen05.cp behavior!

According to NVIDIA docs, tcgen05.cp.cta_group::1.32x128b.warpx4:
- Loads 32 data paths × 128 bits = 512 bytes from SMEM
- Writes to TMEM at specified address

**The question is**: Does the TMEM address encode:
1. A **starting column** (and it auto-increments rows)? 
2. A **starting row** (and it auto-increments columns)?

If tcgen05.cp writes **row-major** (filling columns 0-3 for rows 0-31), then:
```cpp
// ✅ CORRECT: Increment rows, not columns
uint32_t sfa_dst = tmem_sfa_base + (j * 32 << 16);  // Rows 0-31, 32-63, 64-95, 96-127
```

If tcgen05.cp writes **column-major** (filling rows 0-31 for columns 0-3), then:
```cpp
// ✅ CORRECT: Increment columns, not rows
uint32_t sfa_dst = tmem_sfa_base + j * 4;  // Columns 128-131, 132-135, 136-139, 140-143
```

**I suspect column-major is correct**, but check the PTX docs!

---

## Testing Strategy

### Minimal Test Case

Create a tiny test:
```python
M, N, K = 128, 128, 256
# K_scales = 16, n_col_blocks = 4
# to_blocked produces 4 tiles
# Each K-tile needs 4 tiles (all of them!)
```

### Add Debug Prints

In `prefetch_tile`:
```cpp
if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    printf("K-tile %d: n_col_blocks=%d, base_tile=%d, row_offsets=%d,%d,%d,%d\n",
           k_tile_base, n_col_blocks, base_tile_sfa,
           base_tile_sfa*32, (base_tile_sfa+1)*32, 
           (base_tile_sfa+2)*32, (base_tile_sfa+3)*32);
}
```

### Verify Scale Factor Loads

Add a debug kernel to dump SMEM scale factors after TMA:
```cpp
if (k_tile == 0 && threadIdx.x < 16) {
    printf("sfa_smem[%d] = %u\n", threadIdx.x, 
           sfa_stage[stage][threadIdx.x]);
}
```

Compare against:
```python
sfa_blocked = to_blocked(sfa_ref_cpu[:, :, 0])
print("Expected:", sfa_blocked[:16].tolist())
```

---

## Summary of Required Changes

1. **Remove K_scales_padded** from all index calculations
2. **Fix base_tile calculation**: Use `k_tile_idx * cols_per_ktile` not `k_tile_idx * 4`
3. **Verify TMEM addressing** for tcgen05.cp (row vs column increment)
4. **Ensure Python passes K_scales** (not padded) to CUDA

The -inf/inf errors are **NOT** subtle - they're caused by **massively incorrect scale factors** from out-of-bounds TMA loads due to wrong indexing math.

Fix the indexing, and the results should match the reference.