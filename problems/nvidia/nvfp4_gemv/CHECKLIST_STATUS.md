# FP4 GEMV Submission Checklist Status
**Date**: 2025-11-13
**Branch**: `claude/optimize-fp4-gemv-cute-011CV2DZsnfGMiZ3qRCRbkLm`
**Implementation**: Two-stage FP4→FP16 decode + CUTLASS FP16 GEMM

---

## 📦 Input Tuple and Evaluation Specification

### Tensor Format Compliance
- ✅ **Input tuple structure**: Correctly unpacks `(a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)`
- ✅ **Tensor shapes**: Match specification (submission.py:311-314)
  - `a`: M × (K/2) × L (packed FP4) → internally M × K × L
  - `b`: 1 × (K/2) × L (packed FP4) → internally 1 × K × L
  - `sfa`: M × (K//16) × L in FP8 E4M3
  - `sfb`: 1 × (K//16) × L in FP8 E4M3
  - `c`: M × 1 × L in FP16
- ✅ **K-major layout**: Preserved through permute operations (line 317-318)
- ✅ **FP16 output**: Kernel directly produces FP16 output (line 143, 176-177)

### Shape Constraints
- ⚠️ **M divisibility**: ThreadblockShape<128, 8, 64> implies M should be divisible by 128
  - **Status**: Not explicitly validated in code
  - **Risk**: CUTLASS GEMM may handle this internally with padding
- ✅ **K divisibility by 64**: Guaranteed by problem specification (task.yml)

### Benchmark Targets
- ⏳ **Testing required**: Need B200 hardware to measure
  - 7168×16384×1: target 8.622μs
  - 4096×7168×8: target 17.275μs
  - 7168×2048×4: target 4.317μs

---

## 🐞 Common Compilation / Kernel Issues

### A. Register Pressure
- ✅ **maxrregcount=128**: Set in compilation flags (line 291)
- ⚠️ **Potential issue**: May be too low for complex kernels
- **Recommendation**: Monitor for "Internal Error (7)" during compilation

### B. Input Shapes / Permute / Layout
- ✅ **Permutation logic** (submission.py:317-319):
  ```python
  a = a.permute(2, 0, 1)  # M×(K/2)×L → L×M×(K/2)
  b = b.permute(2, 0, 1)  # 1×(K/2)×L → L×1×(K/2)
  c = c.permute(2, 0, 1)  # M×1×L → L×M×1
  ```
- ✅ **Scale factor permutation** (submission.py:326-327):
  ```python
  sfa = sfa_ref_cpu.permute(2, 0, 1)  # M×(K//16)×L → L×M×(K//16)
  sfb = sfb_ref_cpu.permute(2, 0, 1)  # 1×(K//16)×L → L×1×(K//16)
  ```
- ✅ **Layout maintained**: Decode kernel respects batch-first layout

### C. Data Type Mismatches
- ⚠️ **uint8 view usage** (submission.py:322-323, 328-329):
  ```python
  a_bytes = a.view(torch.uint8)  # Reinterpret FP4 packed as uint8
  ```
  - **Justification**: FP4 packed format is 2 nibbles per byte, uint8 view is correct
- ✅ **CUTLASS FP8 type**: Using `cutlass::float_e4m3_t` (line 40-42)
- ✅ **FP16 types**: Using `cutlass::half_t` and CUDA `half` (line 94-96)

### D. PTX / ISA / Arch
- ✅ **SM100 target**: `-gencode=arch=compute_100,code=sm_100` (line 286)
- ✅ **Architecture specification**: `cutlass::arch::Sm100` (line 121)

### E. Blockwise Epilogue API / Template Issues
- ⚠️ **NOT using Example 91 API**: Implementation uses standard CUTLASS device::Gemm
- ✅ **Rationale**: Two-stage approach explicitly approved by user
  - Stage 1: Custom decode kernel (FP4+FP8 → FP16)
  - Stage 2: Standard CUTLASS FP16 GEMM
- ✅ **Template configuration**: Uses documented CUTLASS types (line 115-128)

---

## ✅ Input / Output Tensor Formats

- ✅ **All formats match specification** (see table above)
- ✅ **Decode before GEMM**: FP4 decoded to FP16 in Stage 1 (line 154-162)
- ✅ **Output format**: FP16 M×1×L (line 336)

---

## 🧩 Preprocessing & Packing

### Decode Implementation (submission.py:47-88)
- ✅ **FP4 decode to FP16**: Using constant memory LUT (line 30-36)
- ✅ **FP8 scale factor decode**: Using CUTLASS float_e4m3_t (line 39-43)
- ✅ **Block scaling**: 16 FP4 elements per scale factor (line 65, 74-76)
- ✅ **GEMM input fragments**: All FP16 (line 94-96)

### Unpacking Logic (submission.py:63-71)
```cuda
const int packed_col = col / 2;           // Which byte
const int nibble_idx = col % 2;           // Which nibble (0=lower, 1=upper)
uint8_t nibble = (nibble_idx == 0) ?
    (packed_byte & 0x0F) :                // Lower nibble
    ((packed_byte >> 4) & 0x0F);          // Upper nibble
```
- ✅ **Nibble extraction**: Lower nibble first, upper nibble second
- ⚠️ **Verification needed**: Confirm this matches CUTLASS reference packing order

### Shape/Stride Verification
- ✅ **A matrix stride** (line 67): `batch * M * K_packed + row * K_packed + packed_col`
- ✅ **Scale factor stride** (line 74): `batch * M * K_scales + row * K_scales + sf_idx`
- ✅ **Output stride** (line 85): `batch * M * K + row * K + col`

---

## ⚙️ Kernel & API Requirements

### Tensor Core Usage
- ✅ **CUTLASS GEMM**: OpClassTensorOp with SM100 (line 120-121)
- ✅ **No PyTorch fallbacks**: Pure CUDA/CUTLASS implementation
- ✅ **InstructionShape<16, 8, 16>**: SM100 canonical MMA tile (line 106)
- ⚠️ **Decode kernel**: Uses scalar operations, NOT tensor cores
  - **Impact**: Decode latency may dominate for small shapes
  - **Mitigation**: Could use vectorized loads/stores

### API Compliance
- ✅ **CUTLASS device::Gemm**: Standard API (line 115-128)
- ⚠️ **Not using GemvBlockScaled**: Different from Example 91
- ✅ **Pointer access**: Direct CUDA pointer arithmetic (line 139-143)

---

## 🏆 Direct Output Requirement

- ✅ **FP16 output**: Kernel writes directly to FP16 tensor (line 176-177)
- ⚠️ **Multi-stage pipeline**: DOES use two stages (decode + GEMM)
  - **Checklist conflict**: Says "No multi-stage pipelines"
  - **User approval**: Explicitly approved two-stage approach
  - **Justification**: All computation in single kernel launch sequence, no post-processing
- ✅ **No post-kernel decode**: Output is final FP16 result
- ✅ **Reference match**: Output format matches reference spec

---

## 📐 Canonicalization & Tiling

### CUTLASS Configuration (submission.py:104-106)
- ✅ **ThreadblockShape<128, 8, 64>**: M=128, N=8, K=64 per CTA
- ✅ **WarpShape<64, 8, 64>**: M=64, N=8, K=64 per warp
- ✅ **InstructionShape<16, 8, 16>**: SM100 canonical MMA tile
- ⚠️ **N=8 dimension**: GEMV has N=1, may have inefficiency with N=8 tile

### Layout Compliance
- ✅ **LayoutA**: RowMajor (line 99)
- ✅ **LayoutB**: ColumnMajor (line 100)
- ✅ **LayoutC**: RowMajor (line 101)
- ✅ **K-major input**: Maintained through permutations

### Batch Handling
- ⚠️ **Sequential batch loop** (line 171-194): Processes batches sequentially
  - **Impact**: L=8 case may underutilize GPU
  - **Optimization**: Should use batched GEMM API or fuse batch into M dimension

---

## 🚀 Leaderboard / "Speed of Light" Targets

- ⏳ **Hardware testing required**: Need B200/SM100 GPU
- ⏳ **Performance measurement**: Latency targets not yet validated
- ✅ **Shape compatibility**: All leaderboard shapes should be supported
- ⚠️ **Known bottleneck**: Sequential batch processing for L=8 case

### Expected Performance Issues
1. **Decode kernel latency**: Scalar decode may add 1-3μs overhead
2. **Batch loop overhead**: L=8 case launches 8 separate GEMMs
3. **Small N dimension**: GEMV (N=1) may underutilize tensor cores

---

## 🧮 Validation & Correctness

- ⏳ **Correctness testing**: Requires B200 hardware
- ⏳ **Multi-seed validation**: Not yet performed
- ⏳ **Reference comparison**: Pending test execution

### Known Correctness Risks
1. **Nibble order**: Need to verify lower/upper nibble extraction matches reference
2. **Scale factor layout**: Using reference format (not CUTLASS permuted)
3. **Leading dimensions**: Fixed in commit 760b69d (B matrix ld=K)

---

## 🧾 Documentation & Traceability

### Implementation References
- ✅ **CUTLASS source**: Standard device::Gemm API from CUTLASS 3.x/4.x
- ⚠️ **Not using Example 91**: Different approach (two-stage vs single-kernel)
- ✅ **FP4 E2M1 LUT**: Standard IEEE-like format (submission.py:30-33)
- ✅ **FP8 E4M3**: Using `cutlass::float_e4m3_t` standard type

### Environment Details
- **GPU**: B200 / SM100 Blackwell (target, not yet tested)
- **CUDA/CUTLASS**: CUTLASS 4.2.1+ required (submission.py:8)
- **Compiler flags** (submission.py:282-297):
  - `-O3 --use_fast_math`
  - `-gencode=arch=compute_100,code=sm_100`
  - `-maxrregcount=128`
  - `-DNDEBUG`

### Kernel Launch Parameters
- **Decode kernel**:
  - Block: dim3(16, 16) = 256 threads
  - Grid: dim3((K+15)/16, (M+15)/16, L)
- **CUTLASS GEMM**: Managed internally by CUTLASS device API

---

## Summary Status

| Category | Status | Notes |
|----------|--------|-------|
| Input/Output Format | ✅ PASS | All tensors match spec |
| Compilation Flags | ✅ PASS | SM100 target, reasonable register limit |
| Data Types | ✅ PASS | Correct FP4/FP8/FP16 handling |
| Tensor Core Usage | ⚠️ PARTIAL | GEMM uses tensor cores, decode is scalar |
| API Compliance | ⚠️ DIVERGENT | Two-stage approach vs Example 91 single-kernel |
| Correctness | ⏳ PENDING | Requires hardware testing |
| Performance | ⏳ PENDING | Requires benchmarking |
| Documentation | ✅ PASS | Implementation well-documented |

---

## Critical Action Items Before Submission

1. **[CRITICAL] Test correctness** on all 10 test shapes with B200 hardware
2. **[CRITICAL] Verify nibble unpacking order** matches reference
3. **[HIGH] Optimize batch processing** for L=8 case (4096×7168×8 benchmark)
4. **[MEDIUM] Profile decode kernel** to quantify overhead
5. **[MEDIUM] Consider vectorized decode** for better throughput
6. **[LOW] Validate M divisibility** handling by CUTLASS

---

## Performance Optimization Opportunities

### Immediate (Pre-Submission)
1. **Fuse batch dimension**: Treat L×M as single M dimension, launch once
2. **Vectorized decode**: Use float4/uint4 loads for 8× throughput

### Post-Submission (If Time Permits)
1. **Single-kernel approach**: Fuse decode into CUTLASS prologue
2. **Shared memory caching**: Cache scale factors in SMEM
3. **Warp-level decode**: Use warp intrinsics for cooperative decode

---

## Checklist Conflicts / Clarifications

### "No multi-stage pipelines" (Section 🏆)
- **Checklist**: "❌ No multi-stage pipelines: do not emit intermediate FP4/FP8 outputs"
- **Implementation**: DOES use two-stage (decode → GEMM)
- **User clarification**: "we can do two-stage pipeline" (explicit approval)
- **Resolution**: Two-stage is acceptable as all stages complete before kernel return

### "Canonical CUTLASS blockwise launches" (Section ⚙️)
- **Checklist**: Use CUTLASS blockwise GEMM
- **Implementation**: Uses standard device::Gemm, not GemvBlockScaled
- **Rationale**: GemvBlockScaled only supports FP4→FP4 output, not FP16
- **Resolution**: Standard GEMM with FP16 inputs achieves tensor core goal

---

**Overall Status**: ✅ Implementation complete, ⏳ Hardware validation pending
