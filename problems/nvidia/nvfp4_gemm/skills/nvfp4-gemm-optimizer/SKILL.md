---
name: nvfp4-gemm-optimizer
description: Optimize NVFP4 block-scaled GEMM on NVIDIA B200 (SM100a) using CUTLASS/CuTe + cuTile to target ~3.04 µs geometric mean on task.yml shapes while matching reference.py exactly.
trigger: auto
---

# nvfp4-gemm-optimizer

## Goal

Deliver a **numerically correct** and **near speed-of-light** NVFP4 block‑scaled GEMM for B200 (SM100a).

Target benchmark shapes (from `task.yml`):

- (M=128, N=7168, K=16384, L=1)
- (M=128, N=4096, K=7168,  L=1)
- (M=128, N=7168, K=2048,  L=1)

Target geometric mean: **~3.04 µs**.

## Inputs and semantics

Evaluation tuple:
`(a, b, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c)`

You must:
- Read A/B from `a` and `b`
- Read scale factors from `sfa_permuted` and `sfb_permuted`
- Write FP16 output to `c`

Reference semantics are defined by `reference.py` (scaled matmul).

## Required approach (new direction)

We are **not** doing a pure inline‑PTX implementation anymore.

Use high‑level GPU libraries:
- **CUTLASS** for the GEMM kernel (preferred)
- **CuTe** for layouts/tiling/pipeline composition as needed
- **cuTile** utilities if available in the environment

### Hard constraints

- Do not introduce handwritten PTX for tcgen05 / manual tensor memory movement.
- Do not repack scales in the hot path.
- Do not change the Python/ABI surface of `submission.py`.
- Do not create extra CUDA execution plumbing; default execution only.
- Do not add cross‑run caching beyond compile/autotune caches already present.

## What “correct” means

- Output dtype must be **FP16**
- rtol=1e‑3, atol=1e‑3 against `reference.py`
- All tests in `task.yml` must pass

Correctness invariants to preserve:
- B is provided as (N,K,L); match reference behavior via layout interpretation (not explicit transpose).
- Scale factors correspond to blocks of **16 K‑elements** (axis size K//16).

## Performance priorities

This workload is primarily **memory‑throughput dominated**:
- Keep A/B loads coalesced for packed FP4
- Ensure scale loads match the physical permuted layout expected by the kernel
- Choose tile shapes that sweep large N efficiently with pipelined K

## Execution recipe

1. **Read the files before proposing changes**
   - `submission.py`, `reference.py`, `task.yml`, and any helper headers.

2. **Pick a canonical CUTLASS kernel path**
   - Prefer a CUTLASS block‑scaled GEMM path that naturally matches:
     - packed 4‑bit inputs
     - FP8 scales
     - FP32 accumulate + FP16 output
   - Use CuTe for composition/tuning, not to re‑implement kernels.

3. **Shape‑aware specialization**
   - Allow a small, explicit set of tuned configs selected by (M,N,K).
   - Keep selection logic simple and outside the timed kernel body.

4. **Lean epilogue**
   - One conversion FP32→FP16 and store.
   - Avoid extra post‑ops unless fused by the library.

## Pitfalls to avoid

- Consuming `sfa_ref_cpu` / `sfb_ref_cpu` instead of permuted tensors.
- Applying scales twice.
- Hidden repacking/conversion kernels in the benchmark path.
- Implementing partial “manual” movement that duplicates what CUTLASS already does.
- Debug prints or fallback slow paths left enabled.

## Deliverable expectations

When asked to change code:
- Provide a patch that is minimal, compiles, and preserves correctness.
- Prefer removing complexity over adding it.
- Keep `submission.py` production‑ready (no dead code paths, no “just in case” branches).
