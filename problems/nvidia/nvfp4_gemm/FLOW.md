---
trigger: auto
---

# NVFP4 Block‑Scaled GEMM Flow (CUTLASS / CuTe / cuTile)

## What changed

We are **no longer** pursuing a “pure PTX” pipeline (manual TMA descriptors, tcgen05, TMEM management).

**New default plan:** implement and tune the kernel using **CUTLASS** (preferred), with **CuTe** and **cuTile** as needed for tiling/layout/pipeline composition.

Goals:
- **Correctness:** match `reference.py` within rtol=1e‑3, atol=1e‑3
- **Performance:** **~3.04 µs geometric mean** on the `task.yml` benchmark shapes

---

## Inputs and layouts

The harness passes a tuple:
`(a, b, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c)`

Semantics:
- `a`: A ∈ [M, K, L], packed NVFP4 e2m1, **K‑major**
- `b`: B ∈ [N, K, L], packed NVFP4 e2m1, **K‑major**
- `sfa_permuted`: SFA ∈ [M, K//16, L], FP8 e4m3, **already permuted to the blocked layout**
- `sfb_permuted`: SFB ∈ [N, K//16, L], FP8 e4m3, **already permuted to the blocked layout**
- `c`: C ∈ [M, N, L], FP16 output

**Hard rule:** kernels must consume `sfa_permuted` / `sfb_permuted` directly.  
No `to_blocked()` or scale repacking in the timed path.

---

## Reference behavior (what we must match)

`reference.py` represents a scaled matmul where scale factors apply per **K‑block of 16**:

- Scale axis size is `K//16`
- Every group of 16 K‑elements shares one FP8 scale factor for A and one for B
- Accumulate in FP32 (conceptually), store FP16

Important nuance:
- The reference logically treats B as if used transposed (depending on how it calls the scaled matmul), but **the input tensor `b` arrives as shape (N, K, L)**.
- We match semantics by selecting the correct **layout interpretation** in CUTLASS/CuTe (not by explicitly transposing B).

---

## High‑level pipeline

### 1) Compile / load (first call)

1. `submission.py` builds a small set of candidate kernels (CUTLASS device GEMM instantiations, or CuTe‑composed variants).
2. Compile via `torch.utils.cpp_extension.load_inline` (or the project’s existing compilation pathway).
3. Cache compiled modules (keyed by arch + dtype + kernel config).

Constraints:
- No handwritten PTX in the kernel body.
- No custom descriptor math unless CUTLASS/CuTe cannot express the layout.

### 2) Select a kernel (per call or per shape)

Given (M, N, K, L), choose a tuned config:
- Prefer a **tiny dispatch table** specialized for the `task.yml` shapes
- Otherwise pick a general config that remains correct

Selection logic must be:
- outside the timed region
- minimal branching
- no extra allocations

### 3) Launch (runtime)

1. Bind pointers to A/B/SFA/SFB/C (using `sfa_permuted` / `sfb_permuted`).
2. Launch the selected CUTLASS kernel.
3. Kernel handles:
   - global memory loads (library-managed)
   - scale application (block‑scaled path)
   - FP32 accumulate
   - FP16 epilogue store

---

## Kernel configuration strategy (aggressive)

We care about **throughput**: M is small (often 128), N is large (4096–7168), K is large.

### What to tune
- Threadblock tile (MNK) shapes (favor wide N tiles)
- Pipeline staging depth (as exposed by CUTLASS/CuTe)
- Cluster shape (if exposed/beneficial in the chosen CUTLASS path)
- Epilogue: keep it minimal (FP32→FP16, store)

### What not to do
- Don’t add a parallel custom “repacking kernel”
- Don’t do manual loads/permutes that duplicate CUTLASS movement
- Don’t keep debug printing or alternate slow paths

---

## Correctness checkpoints

Before claiming perf wins, verify these invariants:

1. **Scale tensors used**
   - You are reading `sfa_permuted` / `sfb_permuted` (not the CPU reference tensors).

2. **Scale granularity**
   - Indexing matches blocks of **16 K‑elements** (`K//16`).

3. **Packed FP4 interpretation**
   - Nibble order is consistent with the project’s reference (low nibble = element 0, high nibble = element 1) if/when you interpret packed data.
   - Ideally: let CUTLASS handle decode for the chosen type.

4. **B semantics**
   - No explicit transpose; correct layout selection instead.

5. **Epilogue**
   - No second scaling pass, no “scale twice” bugs.

---

## Performance checklist (toward ~3.04 µs geom mean)

- [ ] Use one main kernel path (avoid hybrid pipelines).
- [ ] Keep scale loads coalesced and aligned to the permuted layout.
- [ ] Prefer a small tuned dispatch set for the three benchmark shapes.
- [ ] Remove redundant checks/branches from the hot path.
- [ ] Confirm you are not launching hidden conversion/repacking ops.

---

## Debug mode policy

Debugging is allowed, but must be easy to disable:
- Wrap debug prints / extra verification in a compile‑time flag.
- Ensure the benchmark path runs with debug disabled.

