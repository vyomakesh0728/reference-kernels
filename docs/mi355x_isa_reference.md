# AMD MI355X (CDNA4, gfx950) ISA Reference for MLA Kernels

This document provides a comprehensive reference of assembly instructions, techniques, and optimization strategies for writing state-of-the-art MLA (Multi-head Latent Attention) kernels on AMD MI355X GPUs.

**Target:** AMD Instinct MI355X (CDNA4 architecture, gfx950 ISA)

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [MFMA Instructions](#2-mfma-instructions)
3. [Memory Instructions](#3-memory-instructions)
4. [Conversion Instructions](#4-conversion-instructions)
5. [Math Instructions](#5-math-instructions)
6. [Synchronization & Control](#6-synchronization--control)
7. [AGPR/VGPR Register Management](#7-agprvgpr-register-management)
8. [Warp-Level Primitives](#8-warp-level-primitives)
9. [MLA Kernel Optimization Techniques](#9-mla-kernel-optimization-techniques)
10. [Code Examples](#10-code-examples)
11. [Novel Ideas for MLA Kernels](#11-novel-ideas-for-mla-kernels)

---

## 1. Architecture Overview

### MI355X Hardware Specifications

| Feature | MI355X (CDNA4) | MI325X (CDNA3) |
|---------|----------------|----------------|
| ISA | gfx950 | gfx942 |
| Compute Units | 256 (8 XCDs × 32 CUs) | 228 |
| Matrix Cores | 1024 | 912 |
| Wavefront Size | 64 | 64 |
| LDS per CU | **160 KB** | 64 KB |
| LDS Banks | **64** | 32 |
| LDS Read BW | **256 B/clk** | 128 B/clk |
| VGPRs per SIMD | 512 | 512 |
| AGPRs per SIMD | 512 | 512 |
| HBM3E Bandwidth | ~8 TB/s | ~5.3 TB/s |
| Max Engine Clock | ~2.1 GHz | ~2.1 GHz |

### Peak Performance (Theoretical)

| Precision | MI355X | Speedup vs FP32 |
|-----------|--------|-----------------|
| FP64 | 78.6 TF | 0.5× |
| FP32 | 157.3 TF | 1× |
| FP16/BF16 | **2.5 PF** | 16× |
| FP8 | **5.0 PF** | 32× |
| FP6 | **10 PF** | 64× |
| FP4 | **10 PF** | 64× |

---

## 2. MFMA Instructions

### 2.1 Standard MFMA Instructions

MFMA (Matrix Fused Multiply-Add) performs: `D := A × B + C`

| Instruction | MxNxK | Input Type | Output Type | Cycles | Use Case |
|-------------|-------|------------|-------------|--------|----------|
| `v_mfma_f32_16x16x4_f32` | 16×16×4 | FP32 | FP32 | 32 | High precision |
| `v_mfma_f32_32x32x2_f32` | 32×32×2 | FP32 | FP32 | 64 | Large tiles |
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | FP16 | FP32 | 16 | Inference |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | FP16 | FP32 | 32 | Inference |
| `v_mfma_f32_16x16x32_f16` | 16×16×32 | FP16 | FP32 | 16 | **CDNA4 NEW** |
| `v_mfma_f32_32x32x16_f16` | 32×32×16 | FP16 | FP32 | 32 | **CDNA4 NEW** |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | BF16 | FP32 | 16 | Training |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | BF16 | FP32 | 32 | Training |
| `v_mfma_f32_16x16x32_bf16` | 16×16×32 | BF16 | FP32 | 16 | **CDNA4 NEW** |
| `v_mfma_f32_32x32x16_bf16` | 32×32×16 | BF16 | FP32 | 32 | **CDNA4 NEW** |

### 2.2 FP8 MFMA Instructions

| Instruction | MxNxK | A Type | B Type | Output | Cycles |
|-------------|-------|--------|--------|--------|--------|
| `v_mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | E4M3 | E4M3 | FP32 | 16 |
| `v_mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | E4M3 | E4M3 | FP32 | 32 |
| `v_mfma_f32_16x16x32_fp8_bf8` | 16×16×32 | E4M3 | E5M2 | FP32 | 16 |
| `v_mfma_f32_32x32x16_fp8_bf8` | 32×32×16 | E4M3 | E5M2 | FP32 | 32 |
| `v_mfma_f32_16x16x32_bf8_fp8` | 16×16×32 | E5M2 | E4M3 | FP32 | 16 |
| `v_mfma_f32_32x32x16_bf8_bf8` | 32×32×16 | E5M2 | E5M2 | FP32 | 32 |

### 2.3 Scaled MFMA Instructions (CDNA4 EXCLUSIVE)

**These are the most powerful instructions for MLA with mixed precision KV caches!**

| Instruction | MxNxK | Types | Output | Cycles | Notes |
|-------------|-------|-------|--------|--------|-------|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16×16×128 | FP8/FP6/FP4 | FP32 | 16-32 | Native mxfp4! |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 32×32×64 | FP8/FP6/FP4 | FP32 | 32-64 | Native mxfp4! |

**Type encoding for scaled MFMA:**
- `0` = E4M3 (FP8)
- `1` = E5M2 (BF8)
- `2` = E2M3 (FP6)
- `3` = E3M2 (BF6)
- `4` = E2M1 (FP4)

**Scaling mechanism:**
- Uses E8M0 scale factors per 32-element block
- Scale applied after dot product, before accumulation
- Formula: `D[i,j] = Σ (A[i,k] × scale_a[i]) × (B[k,j] × scale_b[j]) + C[i,j]`

**Compiler intrinsic:**
```cpp
__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_reg,      // 256-bit packed input A
    b_reg,      // 256-bit packed input B  
    c_reg,      // FP32 accumulator
    Atype,      // 0-4 (fp8/bf8/fp6/bf6/fp4)
    Btype,      // 0-4
    OPSEL_A,    // typically 0
    scale_a,    // E8M0 scale factor
    OPSEL_B,    // typically 0
    scale_b     // E8M0 scale factor
);
```

### 2.4 Sparse MFMA Instructions (CDNA4)

| Instruction | MxNxK | Input | Output | Use Case |
|-------------|-------|-------|--------|----------|
| `v_smfmac_f32_16x16x128_fp8_fp8` | 16×16×128 | FP8 | FP32 | 2:4 sparsity |
| `v_smfmac_f32_16x16x128_bf8_fp8` | 16×16×128 | BF8+FP8 | FP32 | Mixed sparse |
| `v_smfmac_i32_32x32x64_i8` | 32×32×64 | INT8 | INT32 | Quantized |

### 2.5 MFMA Register Mapping (CRITICAL)

Understanding register mapping is essential for efficient data layout:

**For `v_mfma_f32_16x16x32_fp8_fp8`:**
- **D output:** `D[i][j]` where lane%16 = j (COLUMN!), i = 4*(lane/16) + gpr
- **A input:** `A[i][k]` where lane%16 = i (row), k = 8*(lane/16) + 4*gpr + byte
- **B input:** `B[k][j]` where lane%16 = j (col), k = 8*(lane/16) + 4*gpr + byte

**Key insight:** Lane%16 is the COLUMN index in D output, not the row!

**Per-lane data distribution (wave64):**
- 16×16×32 FP8: 8 bytes A, 8 bytes B, 4 floats D per lane
- 32×32×16 FP8: 8 bytes A, 8 bytes B, 16 floats D per lane
- 16×16×128 scaled: 32 bytes A, 32 bytes B, 4 floats D per lane

---

## 3. Memory Instructions

### 3.1 Global-to-LDS Direct Load (KEY OPTIMIZATION)

**This bypasses VGPRs entirely for maximum throughput!**

| Instruction | Bytes/lane | Use Case |
|-------------|-----------|----------|
| `buffer_load_dword ... lds` | 4 | Single dword |
| `buffer_load_dwordx2 ... lds` | 8 | Two dwords |
| `buffer_load_dwordx4 ... lds` | 16 | Four dwords (128-bit) |
| `global_load_lds_dwordx3` | 12 | Three dwords (CDNA4) |
| `global_load_lds_dwordx4` | 16 | Four dwords (CDNA4) |

**LLVM intrinsic:**
```cpp
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc,           // Buffer resource descriptor
    as3_uint32_ptr lds_ptr, // LDS destination
    int size,             // 4, 8, 12, or 16 bytes
    int voffset,          // Per-lane offset
    int soffset,          // Scalar offset
    int offset,           // Immediate offset
    int aux               // Auxiliary (usually 0)
) __asm("llvm.amdgcn.raw.buffer.load.lds");
```

**Buffer resource creation:**
```cpp
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;  // 0x110000 for typical usage
};

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = {
        reinterpret_cast<uint64_t>(ptr),
        range_bytes,
        0x110000
    };
    return *reinterpret_cast<i32x4*>(&rsrc);
}
```

### 3.2 LDS Instructions

| Instruction | Bytes | Destination | Use Case |
|-------------|-------|-------------|----------|
| `ds_read_b32` | 4 | VGPR | Single dword |
| `ds_read_b64` | 8 | VGPR | Two dwords |
| `ds_read_b128` | 16 | VGPR/AGPR | **Optimal for MFMA feed** |
| `ds_read2_b32` | 8 (2×4) | VGPR | Two separate addresses |
| `ds_read2_b64` | 16 (2×8) | VGPR | Two separate addresses |
| `ds_write_b32` | 4 | - | Single dword |
| `ds_write_b64` | 8 | - | Two dwords |
| `ds_write_b128` | 16 | - | Four dwords |
| `ds_write2_b32` | 8 (2×4) | - | Two separate addresses |
| `ds_write2_b64` | 16 (2×8) | - | Two separate addresses |

**Inline assembly for 128-bit LDS read to AGPRs:**
```cpp
// Read 128 bits (16 bytes) from LDS to AGPRs
asm volatile(
    "ds_read_b128 a[0:3], %0 offset:0\n"
    "ds_read_b128 a[4:7], %0 offset:64\n"
    "ds_read_b128 a[8:11], %0 offset:128\n"
    : : "v"(lds_addr) : "memory"
);
```

### 3.3 Global/Flat Load Instructions

| Instruction | Bytes | Notes |
|-------------|-------|-------|
| `global_load_dword` | 4 | Single dword |
| `global_load_dwordx2` | 8 | 64-bit load |
| `global_load_dwordx4` | 16 | 128-bit load (optimal) |
| `flat_load_dword` | 4 | Generic addressing |
| `flat_load_dwordx4` | 16 | Generic 128-bit |

### 3.4 Async Copy / Prefetch

**AMDGPU async operations use marks and waitcounts:**

```cpp
// Async load to LDS
llvm.amdgcn.raw.buffer.load.async.lds(rsrc, lds_ptr, size, voff, soff, off, aux);

// Wait for async completion
llvm.amdgcn.wait.asyncmark(mark);
```

---

## 4. Conversion Instructions

### 4.1 FP8 Pack/Unpack (CRITICAL for attention softmax → V MFMA)

| Instruction | Operation | Use Case |
|-------------|-----------|----------|
| `v_cvt_pk_fp8_f32` | 2×FP32 → 2×E4M3 | Pack softmax scores to FP8 |
| `v_cvt_pk_bf8_f32` | 2×FP32 → 2×E5M2 | Pack to BF8 |
| `v_cvt_f32_fp8` | E4M3 → FP32 | Unpack FP8 |
| `v_cvt_f32_bf8` | E5M2 → FP32 | Unpack BF8 |

**CDNA4 Scaled conversion (with per-tensor scale):**

| Instruction | Operation |
|-------------|-----------|
| `v_cvt_scalef32_pk_fp8_f32` | 2×FP32 → 2×FP8 with scale |
| `v_cvt_scalef32_pk_bf8_f32` | 2×FP32 → 2×BF8 with scale |
| `v_cvt_scalef32_pk_fp8_f16` | 2×FP16 → 2×FP8 with scale |
| `v_cvt_scalef32_pk_fp8_bf16` | 2×BF16 → 2×FP8 with scale |
| `v_cvt_scalef32_pk_bf8_f16` | 2×FP16 → 2×BF8 with scale |
| `v_cvt_scalef32_pk_bf8_bf16` | 2×BF16 → 2×BF8 with scale |

**Compiler intrinsics:**
```cpp
// Pack two floats to two FP8 values
__builtin_amdgcn_cvt_pk_fp8_f32(float a, float b);  // Returns 2×E4M3

// Pack with OPSEL for high/low 16 bits
v40 = __builtin_amdgcn_cvt_pk_fp8_f32(v40, v41);           // Low 16 bits
v40 = __builtin_amdgcn_cvt_pk_fp8_f32(v42, v43, /*opsel=*/1); // High 16 bits

// Scaled conversion (CDNA4)
__builtin_amdgcn_cvt_scalef32_pk_fp8_f32(float a, float b, float scale);
```

### 4.2 BF16 Conversions

| Instruction | Operation |
|-------------|-----------|
| `v_cvt_f32_bf16` | BF16 → FP32 |
| `v_cvt_pk_bf16_f32` | 2×FP32 → 2×BF16 |
| `v_cvt_pk_f16_f32` | 2×FP32 → 2×FP16 (CDNA4) |

### 4.3 FP4/FP6 Handling

FP4 values are packed 2 per byte. Helper functions:

```cpp
// Extract single FP4 from packed byte
uint8_t __amd_extract_fp4(const __amd_fp4x2_storage_t x, const size_t index) {
    if (index == 0) return (x & 0xFu);
    return (x >> 4);
}

// Create packed FP4x2 from two FP4 values
__amd_fp4x2_storage_t __amd_create_fp4x2(const uint8_t x, const uint8_t y) {
    return x | (y << 4);
}
```

---

## 5. Math Instructions

### 5.1 Transcendental Instructions

| Instruction | Operation | Throughput | Notes |
|-------------|-----------|------------|-------|
| `v_exp_f32` | exp2(x) | 1/4 per cycle | Base-2 exp |
| `v_log_f32` | log2(x) | 1/4 per cycle | Base-2 log |
| `v_rcp_f32` | 1/x | 1/4 per cycle | Reciprocal |
| `v_rsq_f32` | 1/√x | 1/4 per cycle | Inverse sqrt |
| `v_sqrt_f32` | √x | 1/4 per cycle | Square root |
| `v_sin_f32` | sin(x) | 1/4 per cycle | Sine |
| `v_cos_f32` | cos(x) | 1/4 per cycle | Cosine |

**For softmax exp:**
```cpp
// Hardware exp2 for softmax
// Note: Need to multiply by log2(e) ≈ 1.4427 for natural exp
float exp_approx = __builtin_amdgcn_exp_f32(x * 1.4426950408889634f);

// Or use v_exp_f32 directly in asm
asm volatile("v_exp_f32_e32 %0, %1" : "=v"(result) : "v"(input));
```

### 5.2 FMA Instructions

| Instruction | Operation | Throughput |
|-------------|-----------|------------|
| `v_fma_f32` | a×b+c | Full rate |
| `v_fmac_f32` | d = d×a+b | Full rate |
| `v_fma_f16` | a×b+c (FP16) | 2× rate |
| `v_pk_fma_f16` | Packed 2×FP16 FMA | 2× rate |

### 5.3 Min/Max/Compare

| Instruction | Operation |
|-------------|-----------|
| `v_max_f32` | max(a, b) |
| `v_min_f32` | min(a, b) |
| `v_max3_f32` | max(a, b, c) |
| `v_med3_f32` | median(a, b, c) |
| `v_cmp_*` | Various comparisons |

---

## 6. Synchronization & Control

### 6.1 Wait Count Instructions

| Instruction | Purpose |
|-------------|---------|
| `s_waitcnt vmcnt(N)` | Wait until ≤N vector memory ops remain |
| `s_waitcnt lgkmcnt(N)` | Wait until ≤N LDS/GDS/KMEM ops remain |
| `s_waitcnt expcnt(N)` | Wait until ≤N export ops remain |
| `s_waitcnt vmcnt(0) lgkmcnt(0)` | Full memory fence |

**Usage patterns:**
```cpp
// Issue loads
buffer_load_dwordx4 v[0:3], ...
buffer_load_dwordx4 v[4:7], ...

// Wait for first load only
asm volatile("s_waitcnt vmcnt(1)");  // 1 load still in flight

// Use v[0:3]
// ...

// Wait for second load
asm volatile("s_waitcnt vmcnt(0)");  // All loads complete
```

### 6.2 Barrier Instructions

| Instruction | Scope |
|-------------|-------|
| `s_barrier` | Workgroup barrier |
| `__builtin_amdgcn_s_barrier()` | HIP intrinsic |

### 6.3 Scheduling Control (CDNA4 Critical)

| Instruction | Purpose |
|-------------|---------|
| `s_setprio N` | Set wave priority (0-3) |
| `s_sched_barrier M` | Control instruction reordering |

**Intrinsics:**
```cpp
__builtin_amdgcn_s_setprio(1);     // Raise priority for MFMA
// ... MFMA instructions ...
__builtin_amdgcn_s_setprio(0);     // Lower priority

__builtin_amdgcn_sched_barrier(0); // Prevent reordering across this point
```

---

## 7. AGPR/VGPR Register Management

### 7.1 Overview

- **VGPRs:** 512 per SIMD, general purpose vector registers
- **AGPRs:** 512 per SIMD, accumulator GPRs optimized for MFMA output

### 7.2 AGPR ↔ VGPR Transfer

| Instruction | Operation |
|-------------|-----------|
| `v_accvgpr_read_b32` | AGPR → VGPR |
| `v_accvgpr_write_b32` | VGPR → AGPR |

**Key insight from aiter disassembly:**
- MFMA inputs can come from AGPRs directly: `v_mfma_f32_16x16x32_fp8_fp8 v[40:43], a[72:73], a[0:1], 0`
- LDS loads can target AGPRs: `ds_read_b128 a[0:3], v29`
- This eliminates VGPR pressure!

### 7.3 Efficient Patterns

```cpp
// Load directly to AGPRs (from aiter assembly)
asm volatile(
    "ds_read_b128 a[0:3], %0 offset:0\n"
    "ds_read_b128 a[4:7], %0 offset:64\n"
    : : "v"(lds_addr) : "memory"
);

// MFMA with AGPR inputs
asm volatile(
    "v_mfma_f32_16x16x32_fp8_fp8 v[40:43], a[72:73], a[0:1], 0\n"
    : : : "v40", "v41", "v42", "v43"
);
```

---

## 8. Warp-Level Primitives

### 8.1 DPP (Data Parallel Primitives)

| Operation | Code | Description |
|-----------|------|-------------|
| `row_shr` | 0x110-0x11F | Shift right within row (1-15) |
| `row_shl` | 0x100-0x10F | Shift left within row (1-15) |
| `row_bcast15` | 0x142 | Broadcast lane 15 to next row |
| `row_bcast31` | 0x143 | Broadcast lane 31 to rows 2,3 |
| `quad_perm` | - | Arbitrary 4-lane permute |
| `row_mirror` | 0x140 | Mirror within row |
| `row_half_mirror` | 0x141 | Half mirror |

**For wave64 reduction:**
```cpp
// Reduction using DPP
float sum = value;
sum += __builtin_amdgcn_mov_dpp(sum, 0x111, 0xF, 0xF, false);  // row_shr:1
sum += __builtin_amdgcn_mov_dpp(sum, 0x112, 0xF, 0xF, false);  // row_shr:2
sum += __builtin_amdgcn_mov_dpp(sum, 0x114, 0xF, 0xF, false);  // row_shr:4
sum += __builtin_amdgcn_mov_dpp(sum, 0x118, 0xF, 0xF, false);  // row_shr:8
// ... continue for cross-row reduction
```

### 8.2 Permute Lane Instructions (CDNA4 Optimized)

| Instruction | Use Case |
|-------------|----------|
| `v_permlane16_b32` | Cross-row permutation |
| `v_permlane16_var_b32` | Variable cross-row |
| `v_permlane*_swap` | **CDNA4 optimized reduction** |

**CDNA4 reduction optimization (from Triton PR):**
```cpp
// Use v_permlane*_swap for efficient reduction
__builtin_amdgcn_permlane_swap(value, ...);
```

### 8.3 Readlane/Writelane

| Instruction | Operation |
|-------------|-----------|
| `v_readlane_b32` | Read from specific lane to SGPR |
| `v_writelane_b32` | Write SGPR to specific lane |
| `v_readfirstlane_b32` | Read from first active lane |

---

## 9. MLA Kernel Optimization Techniques

### 9.1 LDS Bank Conflict Elimination (Swizzling)

**Problem:** 64 LDS banks on CDNA4; `ds_read_b128` executes in 4 phases.

**XOR swizzle pattern:**
```cpp
int swizzle_col(int row, int col) {
    const int pair = (row >> 1) & 7;
    const int perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    const int mask = perm << 4;
    return col ^ mask;
}
```

### 9.2 Double Buffering (Ping-Pong)

Overlap load and compute:
```cpp
shared fp8 A_lds[2][TILE_K][TILE_M];  // Two buffers
shared fp8 B_lds[2][TILE_K][TILE_N];

int cur = 0, nxt = 1;

// Prologue: load first tile
prefetch_tile(A_lds[cur], B_lds[cur], 0);
sync();

for (int t = 0; t < num_tiles; ++t) {
    if (t + 1 < num_tiles) {
        prefetch_tile_async(A_lds[nxt], B_lds[nxt], t + 1);
    }
    
    compute_mfma(A_lds[cur], B_lds[cur], accum);
    
    sync();
    cur ^= 1; nxt ^= 1;  // Swap buffers
}
```

### 9.3 8-Wave Ping-Pong Scheduling

**From HipKittens paper - maximum MFMA utilization:**
- 8 waves per block (2 per SIMD)
- Waves alternate between memory and compute
- Use barriers to stagger execution

```cpp
int waveid = threadIdx.x / 64;
int wave_m = waveid / 4;  // 0 or 1

// Stagger waves 4-7
if (wave_m == 1) {
    __builtin_amdgcn_s_barrier();
}

// Wave 0-3 loads while 4-7 waits
issue_loads();

__builtin_amdgcn_s_barrier();

// Now 4-7 loads while 0-3 computes
if (wave_m == 0) {
    __builtin_amdgcn_s_setprio(1);
    mfma_compute();
    __builtin_amdgcn_s_setprio(0);
}
__builtin_amdgcn_s_barrier();
// ... alternate
```

### 9.4 Online Softmax for Attention

Compute softmax incrementally to avoid storing full attention matrix:
```cpp
float m_prev = -INFINITY;  // Max so far
float l_prev = 0.0f;       // Sum of exp so far
float acc[V_DIM] = {0};    // Accumulator

for each KV tile:
    // Compute QK^T scores
    float scores[TILE_K];
    mfma_qk(Q, K_tile, scores);
    
    // Update running max
    float m_new = max(m_prev, max(scores));
    
    // Correction factor
    float correction = exp(m_prev - m_new);
    
    // Scale previous accumulator
    acc *= correction;
    l_prev *= correction;
    
    // Add new contributions
    for k in tile:
        float w = exp(scores[k] - m_new);
        l_prev += w;
        acc += w * V[k];
    
    m_prev = m_new;

// Final normalization
acc /= l_prev;
```

### 9.5 Split-K for Large KV Sequences

For long sequences, split K dimension across workgroups:
```cpp
// Stage 1: Each WG handles a portion of KV
partial_out[split_id] = attention(Q, K[start:end], V[start:end]);
partial_lse[split_id] = log_sum_exp;

// Stage 2: Reduce across splits
final_out = log_sum_exp_reduce(partial_out, partial_lse);
```

---

## 10. Code Examples

### 10.1 FP8 MFMA 16×16×32

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

using fp8_t = __hip_fp8_storage_t;
using fp8x8_t = __attribute__((vector_size(8))) fp8_t;
using fp32x4_t = __attribute__((vector_size(16))) float;

__global__ void mfma_fp8_16x16x32(
    const fp8_t* __restrict__ A,  // [16, 32]
    const fp8_t* __restrict__ B,  // [32, 16]
    float* __restrict__ C         // [16, 16]
) {
    fp8x8_t a_reg, b_reg;
    fp32x4_t c_reg = {0};
    
    const int lane = threadIdx.x;
    const int row = lane % 16;
    const int k_group = lane / 16;  // 0-3
    
    // Load A fragment: 8 FP8 values per lane
    a_reg = *reinterpret_cast<const fp8x8_t*>(
        A + row * 32 + k_group * 8
    );
    
    // Load B fragment: 8 FP8 values per lane
    for (int i = 0; i < 8; i++) {
        b_reg[i] = B[(k_group * 8 + i) * 16 + row];
    }
    
    // MFMA
    c_reg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
        (long)a_reg, (long)b_reg, c_reg, 0, 0, 0
    );
    
    // Store: lane%16 = column, 4*(lane/16)+i = row
    for (int i = 0; i < 4; i++) {
        C[(k_group * 4 + i) * 16 + row] = c_reg[i];
    }
}
```

### 10.2 Scaled MFMA with FP4 (MXFP4)

```cpp
#include <hip/hip_ext_ocp.h>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = __attribute__((vector_size(64))) float;

__global__ void mfma_mxfp4_32x32x64(
    const fp4x2_t* A,      // [32, 64] packed
    const fp4x2_t* B,      // [64, 32] packed
    const uint8_t* scaleA, // [32, 2] E8M0
    const uint8_t* scaleB, // [2, 32] E8M0
    float* C               // [32, 32]
) {
    fp4x64_t a_reg = {0};
    fp4x64_t b_reg = {0};
    fp32x16_t c_reg = {0};
    
    const int lane = threadIdx.x;
    const int row = lane % 32;
    const int group = lane / 32;  // 0 or 1
    
    // Load A: 32 FP4 values = 16 bytes
    const fp4x2_t* a_ptr = A + row * 32 + group * 16;
    for (int i = 0; i < 16; i++) {
        a_reg[i] = a_ptr[i];
    }
    
    // Load B with extraction
    const fp4x2_t* b_ptr = B + (row / 2) + group * 16 * 32;
    int extract_idx = row % 2;
    for (int i = 0; i < 16; i++) {
        uint8_t tmp0 = __amd_extract_fp4(b_ptr[16 * 2 * i], extract_idx);
        uint8_t tmp1 = __amd_extract_fp4(b_ptr[16 * (2 * i + 1)], extract_idx);
        b_reg[i] = __amd_create_fp4x2(tmp0, tmp1);
    }
    
    // Load scales
    uint8_t scale_a = scaleA[row * 2 + group];
    uint8_t scale_b = scaleB[group * 32 + row];
    
    // Scaled MFMA with type=4 (FP4)
    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg,
        4, 4,      // Atype=FP4, Btype=FP4
        0, scale_a,
        0, scale_b
    );
    
    // Store output (same as 32x32x16 layout)
    for (int i = 0; i < 4; i++) {
        C[(group * 4 + i) * 32 * 8 + row] = c_reg[i * 4];
        C[(group * 4 + i) * 32 * 8 + 32 + row] = c_reg[i * 4 + 1];
        C[(group * 4 + i) * 32 * 8 + 64 + row] = c_reg[i * 4 + 2];
        C[(group * 4 + i) * 32 * 8 + 96 + row] = c_reg[i * 4 + 3];
    }
}
```

### 10.3 Global-to-LDS Direct Load

```cpp
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_ptr lds, int size, int voff, int soff, int off, int aux
) __asm("llvm.amdgcn.raw.buffer.load.lds");

__device__ i32x4 make_buffer_rsrc(const void* ptr, uint32_t size) {
    struct { uint64_t addr; uint32_t range; uint32_t config; } r = {
        (uint64_t)ptr, size, 0x110000
    };
    return *(i32x4*)&r;
}

__global__ void load_to_lds(const float* src, float* dst) {
    __shared__ float lds[256];
    
    i32x4 rsrc = make_buffer_rsrc(src, 256 * sizeof(float));
    as3_ptr lds_ptr = (as3_ptr)((uint32_t*)lds);
    
    // Direct global -> LDS (16 bytes per lane)
    llvm_amdgcn_raw_buffer_load_lds(
        rsrc, lds_ptr, 16,
        threadIdx.x * 16,  // voffset
        0, 0, 0
    );
    
    asm volatile("s_waitcnt vmcnt(0)");
    __syncthreads();
    
    // Now use LDS data...
}
```

### 10.4 Hardware FP8 Conversion for Softmax Scores

```cpp
__device__ uint32_t pack_scores_to_fp8(float s0, float s1, float s2, float s3) {
    uint32_t packed;
    
    // Pack 4 floats to 4 FP8 values
    asm volatile(
        "v_cvt_pk_fp8_f32 %0, %1, %2\n"           // Low 2 bytes
        "v_cvt_pk_fp8_f32 %0, %3, %4 op_sel:[0,0,1]\n"  // High 2 bytes
        : "=v"(packed)
        : "v"(s0), "v"(s1), "v"(s2), "v"(s3)
    );
    
    return packed;
}
```

---

## 11. Novel Ideas for MLA Kernels

### 11.1 Native MXFP4 KV Cache

**Opportunity:** MI355X has native `v_mfma_scale_f32_16x16x128_f8f6f4` that handles FP4 natively!

**Current approach:** Software dequant (30-68× slower)
**Novel approach:** Use scaled MFMA directly with mxfp4 V values

```cpp
// Instead of: dequant(V_mxfp4) → fp16 → MFMA
// Use: v_mfma_scale_f32_16x16x128_f8f6f4 with type=4

// QK scores → FP8 via v_cvt_pk_fp8_f32
// V_mxfp4 + scales → direct MFMA input
// Accumulator: FP32 → final BF16 output
```

### 11.2 Single-Pass Attention (No Split-K)

For decode with small batch sizes, split-K overhead dominates.

**Novel approach:** Persistent kernel with:
- One workgroup per (batch, head) pair
- Iterate through all KV in registers
- Online softmax in registers
- No intermediate split buffers

### 11.3 Fused Softmax-to-FP8 Conversion

**Current:** softmax → fp32 scores → separate v_cvt_pk_fp8_f32 → V MFMA
**Novel:** Fuse exp, normalization, and FP8 pack in single loop:

```cpp
// After computing QK scores
float m = max(scores);
float sum = 0;

#pragma unroll
for (int i = 0; i < TILE; i += 4) {
    float e0 = __expf(scores[i] - m);
    float e1 = __expf(scores[i+1] - m);
    float e2 = __expf(scores[i+2] - m);
    float e3 = __expf(scores[i+3] - m);
    
    sum += e0 + e1 + e2 + e3;
    
    // Pack to FP8 immediately
    fp8_scores[i/4] = pack_scores_to_fp8(e0, e1, e2, e3);
}

// Normalize and rescale in V MFMA accumulator
```

### 11.4 AGPR-Resident Accumulator Pipeline

**Insight from aiter assembly:** They keep accumulators in AGPRs and feed MFMA from AGPRs.

**Pattern:**
1. Load Q/K/V tiles → LDS
2. `ds_read_b128` → AGPRs (not VGPRs!)
3. MFMA outputs to VGPRs
4. Move VGPRs → AGPRs for next iteration accumulation
5. Final reduce: AGPRs → VGPRs → global

### 11.5 8-Wave Interleaved Attention

**Adapt FP8 GEMM 8-wave ping-pong for attention:**
- 8 waves per workgroup
- Waves 0-3: Process even KV tiles
- Waves 4-7: Process odd KV tiles
- Interleave to hide memory latency

### 11.6 Warp-Specialized Attention

Different waves do different jobs:
- Wave 0-1: Global loads → LDS staging
- Wave 2-3: QK MFMA
- Wave 4-5: Softmax compute
- Wave 6-7: V MFMA

Use barriers and priority to coordinate.

### 11.7 Decode #1 "pg8" Hypothesis

Top leaderboard entry uses "pg8" in filename. Possible meanings:
- **Persistent Group 8:** 8-wave persistent scheduling
- **Page Size 8:** Different paging scheme for KV
- **Prefetch Group 8:** 8-tile lookahead prefetching

---

## References

1. [AMD CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
2. [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
3. [Matrix Core Programming on CDNA3/4](https://salykova.github.io/matrix-cores-cdna)
4. [FP8 GEMM Optimization on CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
5. [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
6. [LLVM AMDGPU Backend Documentation](https://llvm.org/docs/AMDGPUUsage.html)
7. [HipKittens: Fast and Furious AMD Kernels](https://arxiv.org/abs/2511.08083)
8. [Triton gfx950 Scaled MFMA PR](https://github.com/triton-lang/triton/pull/5845)
9. [ROCm FlashInfer](https://github.com/ROCm/flashinfer/)

---

*Last updated: April 2026*
*Target: AMD MI355X (CDNA4, gfx950)*
