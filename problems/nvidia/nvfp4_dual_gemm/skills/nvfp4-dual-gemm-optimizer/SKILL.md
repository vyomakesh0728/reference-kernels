---
name: nvfp4-dual-gemm-optimizer
description: Optimize NVIDIA SM100/SM100a FP4 block-scaled dual GEMM kernels with silu activation for B200
trigger: auto
---
# DEFAULT SYSTEM_PROMPT AND INSTRUCTIONS
"""
You are an expert NVIDIA GPU SM100/SM100a kernel optimization assistant.
Prioritize memory bandwidth, occupancy, and TMA usage for Blackwell FP4 dual GEMM.
Always preserve correctness; add minimal, well-instrumented changes and benchmark them.
"""

## system-reminder
   YOUR RESPONSES SHOULD ALWAYS BE EITHER CONCISE OR IN PATCH-CODE FORMAT

   IMPORTANT: this context is relevant to your tasks. 
   You should always respond to this context unless it is highly relevant to your task.

## DO_NOT
   - WRITE SUMMARIES, DOCUMENTATION, EXPLANATIONS, AND GUIDES
   - RUN TESTS UNLESS SPECIFICALLY INSTRUCTED NOT TO
   - DO NOT import or depend on `kutte.py`, `cutlass.cute`, or any CuTe runtime in `submission.py`.
   - CuTe/CUTLASS may be used only as a conceptual reference for descriptor/layout reasoning; the implementation must remain the current inline-CUDA/inline-PTX approach in `submission.py`.​



## STRICT_COMPETITION_RULES_DO_NOT_VIOLATE

1. NO CUDA STREAMS - Zero tolerance policy
   - Do NOT create any CUDA streams (cudaStreamCreate, cudaStreamCreateWithFlags, etc.)
   - Do NOT use stream-related APIs (cudaStreamSynchronize, cudaStreamWaitEvent, etc.)
   - Do NOT import or reference stream functionality from PyTorch (c10::cuda::getCurrentCUDAStream, at::cuda::CUDAStreamGuard, etc.)
   - Everything runs on the default stream automatically

2. BANNED KEYWORDS AND PATTERNS
   - Your code will be rejected if it contains the word "stream" anywhere
   - Do NOT try to circumvent this by importing stream functionality without using the word
   - Do NOT attempt any workarounds to create concurrent execution

- Do not create or use additional CUDA streams; everything must run on the default stream. The benchmarking script only syncs the default stream.

- DO NOT ADD CROSS-RUN CACHING BEYOND COMPILATION/AUTOTUNE.

WHY: The benchmarking script only synchronizes the default stream. Using custom streams 
or explicit cross-run caching produces invalid timing results and violates competition rules.

CONSEQUENCE: Submissions violating these rules will be automatically rejected or deleted.

## INVESTIGATE_BEFORE_ANSWERING
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.


## CODE_STYLE
Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please inform me rather than working around them. The solution should be robust, maintainable, and extendable.

---

# CLEAN_UP
  ## TODO: clean up legacy code

--- 

# tcgen05_FLOW

** Hardware Fused HIGH LEVEL flow **:
Global A/B1/B2 (packed FP4) → TMA → staged SMEM (packed FP4, NO DECODE!)
Global SF (FP8) in atom-tiled physical layout (`sfa_permuted/sfb1_permuted/sfb2_permuted`) → TMA → staged SMEM
→ CuTe tcgen05 S2T copy (Cp4x32x128b) (staged SMEM FP8 scales → TMEM tensor views for SFA/SFB1/SFB2)
→ tcgen05.mma.mxf4.block_scale / CuTe tiled MMA (run twice for B1 and B2 paths)
   ├─ Hardware decodes FP4→FP16 inside tensor core
   ├─ Hardware applies FP8 scales from TMEM
   └─ Hardware performs MMA (ACCUMULATE=false for first k_tile / first kblock, then true)
→ TMEM (FP32 accumulator tiles for each GEMM)
→ tcgen05 TMEM.load → Registers → apply silu + multiply → FP16 → Global D

IMPORTANT: refer to FULL tcgen05 GEMM Flow, `FLOW.md`

---

# tcgen05_TMEM_BUG_DIAGNOSIS

CURRENT INF/NAN BUG - Likely causes:

1. **Instruction Descriptor Format Bits**
   - Verify bits [7:10] = 5 (E2M1 format)
   - Verify bits [23:24] = 0 (E4M3 scale format)
   - Print `idescE` value in hex to confirm

2. **TMA vs TMEM Scale Layout Mismatch**
   - TMA loads atom-tiled layout (32,4,rest_m,4,rest_k) flattened
   - tcgen05.cp expects contiguous 32×16B chunks
   - These may not match! Need to verify byte-for-byte correctness


3. **Scale Copy Synchronization**
   - tcgen05.cp must complete BEFORE tcgen05.mma
   - Add explicit `__syncthreads()` after scale copy loop
   - Current code has sync, but verify warp_id<4 condition

4. **ACCUMULATE Flag Logic**
   - scaleC=0 only for k_tile==0 && kb==0
   - scaleC=1 for all other K-blocks
   - Verify predicate `p` is set correctly

DEBUG NEXT STEPS:
1. print `idescE` and compare against the CuTe/CUTLASS construction for (M=128, N=128)
2. Print first 32 bytes of sfa_stage[0] after TMA load
3. Compare with first 32 bytes of sfa_permuted on CPU
4. Add `__syncthreads()` immediately after scale copy loop before MMA

---

 # COMPLIANCE_CHECK

  Your `submission.py` kernel implementation MUST be compliant with:
   - The kernel must **produce** FP16 output tensor `c/c_ref` (not FP32).
   - Evaluation/reference uses dual GEMM semantics: `C = silu(A @ B1) * (A @ B2)`.
   - The kernel must consume the permuted physical FP8 scale tensors (`sfa_permuted`, `sfb1_permuted`, `sfb2_permuted`) and be semantically equivalent to applying `to_blocked(sfa_ref_cpu)`, `to_blocked(sfb1_ref_cpu)`, `to_blocked(sfb2_ref_cpu)` as in reference.py.

  1. reference.py
     - Match torch._scaled_mm(..., out_dtype=torch.float32) behavior exactly for each GEMM path, then apply `silu` and multiply, and finally cast to FP16.
     - Match reference semantics where reference.py computes with B^T (it uses
       `b1_ref[:, :, l_idx].transpose(0, 1)` and `b2_ref[:, :, l_idx].transpose(0, 1)`). Physical B1/B2 inputs are already
       shaped/layouted as (n, k, l) for the kernel; interpret them accordingly
       via layouts/iterators (do not perform an explicit transpose in-kernel).
     - Semantic scale truth is `sfa_ref_cpu` / `sfb1_ref_cpu` / `sfb2_ref_cpu` as used by reference.py.
   
     - Correctness condition:
         The values applied by the kernel (via permuted physical layout) must be
         exactly equivalent to the blocked semantics used by reference.py
         (i.e., `to_blocked(sfa_ref_cpu[:, :, l_idx])`, `to_blocked(sfb1_ref_cpu[:, :, l_idx])`, `to_blocked(sfb2_ref_cpu[:, :, l_idx])`),
         without re-running `to_blocked()` in the hot path.
         - **IMPORTANT**: Confirm tcgen05 warpgroup participation: choose which 4 warps participate in cp + mma + epilogue; make gating and the asm predicate consistent.
     - Meet output tolerance: rtol=1e‑3, atol=1e‑3.
   

  2. task.yml
     - task.yml text describes a 7-tuple, but the evaluation harness/reference code uses a 10-tuple input:
       `(a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_permuted, sfb1_permuted, sfb2_permuted, c)`.
     - M divisible by mma_tiler_mn[0] (e.g., 16 or 32 factors).
     - N divisible by mma_tiler_mn[1] (e.g., 8 or 16 factors).
     - K divisible by 256.
     - All benchmark shapes have L=1.
     - Ranking by geometric mean runtime over test cases.
     - Target SoL latencies (1.5Ghz):
       - (256, 4096, 7168, 1): 4.708 us
       - (512, 4096, 7168, 1): 8.714 us
       - (256, 3072, 4096, 1): 2.125 us
       - (512, 3072, 7168, 1): 6.535 us

---
