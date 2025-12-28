---
trigger: auto
---

You have additional ONE SKILL documented in the current directory containing a "SKILL.md" file

- nvfp4-dual-gemm-optimizer -> "skills/nvfp4-dual-gemm-optimizer/SKILL.md"

IMPORTANT: You MUST read the SKILL.md file whenever the description of the skills matches the user intent, or may help accomplish their task.

<available_skills>
   nvfp4-dual-gemm-optimizer: `Optimize NVIDIA SM100/SM100a FP4 block-scaled dual GEMM kernels with silu activation for B200`
</available_skills>

# NVFP4 Block-Scaled Dual GEMM Kernel Optimization

Optimize block-scaled dual matrix multiplication kernels for NVIDIA B200 (SM100/SM100a) using FP4 quantization to achieve competitive geometric mean latency across benchmark shapes.

## Target Performance

Speed-of-light analysis (1.5GHz clock):
  M   N   K   L time[us]
256 4096 7168 1 4.708
512 4096 7168 1 8.714
256 3072 4096 1 2.125
512 3072 7168 1 6.535

Geometric mean of those four times ≈ 4.89 µs.

## Optimization Strategy

### Memory Bandwidth Priority
The kernel is **memory bandwidth bound**, not compute bound. Focus optimization efforts on:
- TMA (Tensor Memory Accelerator) for efficient bulk loading
- Minimizing DRAM transactions
- Optimal data layout and swizzle patterns
- Coalesced memory access

### Hardware Utilization
- Use tcgen05 instruction set for Blackwell architecture
- Leverage hardware-fused FP4→FP16 decode in tensor cores
- Apply FP8 scales from TMEM during MMA operations
- Target peak occupancy while respecting register/SMEM limits

### Data Flow (tcgen05_FLOW)
```
Global A/B1/B2 (packed FP4) → TMA → SMEM (packed FP4)
Global SF (FP8, atom-tiled) → TMA → SMEM
→ tcgen05 S2T copy (SMEM FP8 → TMEM SFA/SFB1/SFB2)
→ tcgen05.mma.mxf4.block_scale (run twice: A@B1 and A@B2)
  ├─ Hardware FP4→FP16 decode
  ├─ Hardware FP8 scale application
  └─ MMA (ACCUMULATE=false first k_tile, then true)
→ TMEM (FP32 accum) → Registers → silu+mul → FP16 → Global D
```

## Correctness Requirements

### Reference Semantics
- The evaluation entrypoint uses a 10-tuple input like `reference.py` / `70.py`:
  `(a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_permuted, sfb1_permuted, sfb2_permuted, c)`.
- Compute semantics: `C = silu(A @ B1) * (A @ B2)`.
- Per `reference.py`, the reference math uses `torch._scaled_mm(..., out_dtype=torch.float32)` for each GEMM path, then applies `silu` and multiply, then casts to FP16.
- Physical B1/B2 inputs are already shaped/layouted as `(n, k, l)` for the kernel; interpret accordingly via layouts/iterators (do not explicit-transpose in-kernel).
- Scale factors for the kernel must use the permuted physical layout: `sfa_permuted/sfb1_permuted/sfb2_permuted`.

### Numerical Tolerance
- rtol=1e-3, atol=1e-3
- Output tensor `c` must be FP16 (internal accumulation/compute may be FP32)
- All 10 test cases must pass correctness checks

### Input Constraints
- M divisible by mma_tiler_mn[0]
- N divisible by mma_tiler_mn[1]  
- K divisible by 256
- L=1 for all benchmark shapes
- Scale factor blocks: 16 elements per block

## Strict Competition Rules

### **BANNED** - **Zero Tolerance**
- ❌ No CUDA streams (cudaStreamCreate, etc.)
- ❌ No stream APIs (cudaStreamSynchronize, etc.)
- ❌ No PyTorch stream references (c10::cuda::getCurrentCUDAStream, etc.)
- ❌ No cross-run caching beyond compilation/autotune
- ❌ Code containing word "stream" anywhere
- ❌ No caching of inputs between runs
- ❌ No graph replay for a single kernel

**Everything runs on default stream.** Benchmarking only syncs default stream.

### Optimization Checklist

### NOTE:
- No micro-optimization work is allowed until the user specified that all 10 correctness tests and 4 benchmark shape tests pass (rtol/atol)
- Prefer 128-thread CTAs for tcgen05 path unless there is a proven need for extra warps; extra warps otherwise waste occupancy/bandwidth headroom

### Red flags
- Single-warp MMA gating.
- Multiple independent persistent schedulers.
- Hard-coded `num_tmem_alloc_cols=512`.
- `arrive_and_wait()` inside the subtile loop.

### Kernel Launch Configuration
- [ ] Grid/block dimensions tuned per shape
- [ ] Register usage under spill threshold
- [ ] SMEM usage optimized for occupancy
- [ ] Thread block size balances occupancy vs resources

### Memory Access Patterns
- [ ] TMA descriptors configured correctly
- [ ] Swizzle modes optimized for bank conflicts
- [ ] Coalesced global memory access
- [ ] Minimal DRAM bandwidth consumption

### Computation Pipeline
- [ ] K-loop tiling optimized
- [ ] Double buffering for overlap (if beneficial)
- [ ] MMA accumulate flags set correctly
- [ ] TMEM load/store minimized

### Profiling Targets
- [ ] Achieved DRAM bandwidth > 95% theoretical
- [ ] SM occupancy > 75%
- [ ] Zero bank conflicts
- [ ] Minimal warp divergence

## Tasks
- Double check for redundancy and over protective code? an environment built with PTX inline is a narrow process for a specific thing (GEMM here), there should be little to no SEMANTIC/LAYOUT oracle MISMATCHES because there should NOT be unknowns in this program.

- Check with the official cutlass files or supporting files at /usr/local/cutlass/*  ** only if needed ** 

- Check `submission.py` files for 
   - consistency 
   - proper encapsulation
   - proper accumulator initialization
   - no over-scaling or applying scales multiple times 
   - no OVERHEAD
   - no defensive coding
   - not overcomplicating the code
   - no over-protective if statements
   - not repeating the same code patterns separately 
   - no silent failing funcs 
   - structurally perfect
   - production ready 

## Key Files 

- `submission.py`: Your optimized kernel implementation
- `reference.py`: PyTorch reference using torch._scaled_mm
- `task.yml`: Test/benchmark shape definitions
- `SKILL.md`: Extended context and constraints
- `FLOW.md`: Detailed kernel flow description

## Failed Attempts to Avoid

| Approach | Why It Failed |
|----------|---------------|
| Explicit B transpose in-kernel | Violates reference semantics; use layout interpretation |
| Using sfa_ref_cpu directly | Wrong physical layout; must use sfa_permuted |
| FP32 output | Task requires FP16 output dtype |
| Custom stream management | Violates competition rules; invalid timing |
| Porting/embedding CuTe reference kernel (70.py) into submission.py | violates the intended PTX kernel path |

## Success Criteria

1. ✅ Pass all 10 correctness tests (rtol=1e-3, atol=1e-3)
2. ✅ Achieve strong geometric mean performance on the 4 benchmark shapes in `task.yml`
3. ✅ Memory bandwidth utilization > 95% of theoretical peak
4. ✅ Zero competition rule violations
