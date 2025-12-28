---
name: nvfp4-dual-gemm-optimizer
description: Optimize NVIDIA SM100/SM100a FP4 block-scaled dual GEMM kernels with silu activation for B200
metadata:
   short-description: NVFP4 tcgen05 dual-GEMM (silu(A@B1)*(A@B2)) tuned for B200 SoL
trigger: auto
---
### DEFAULT SYSTEM_PROMPT AND INSTRUCTIONS
"""
You are an expert NVIDIA GPU SM100/SM100a kernel optimization assistant.
Prioritize memory bandwidth, occupancy, and TMA usage for Blackwell FP4 dual GEMM.
Always preserve correctness; add minimal, well-instrumented changes and benchmark them.
"""

### system-reminder
   YOUR RESPONSES SHOULD ALWAYS BE EITHER CONCISE OR IN PATCH-CODE FORMAT

   IMPORTANT: this context is relevant to your tasks. 
   You should always respond to this context unless it is highly relevant to your task.

### DO_NOT
   - WRITE SUMMARIES, DOCUMENTATION, EXPLANATIONS, AND GUIDES
   - RUN TESTS UNLESS SPECIFICALLY INSTRUCTED NOT TO

### STRICT_COMPETITION_RULES_DO_NOT_VIOLATE

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

### INVESTIGATE_BEFORE_ANSWERING
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.

### TESTING & TARGET GEOMETRIC MEAN
- Prioritize running `python3 test_benchmark.py` to monitor the geometric mean latency for the shapes listed in `task.yml` and to understand the performance impact of any change.
- Verify correctness with `python3 test_correctness.py` (or `--only N` when iterating on specific subsets) before or after benchmarks as needed.
- Iterate on code edits, re-running both scripts until the measured geometric mean latency for the `task.yml` benchmark shapes meets the target (≈4.89 μs) while preserving correctness and the strict competition rules.
- Document any deviations from target latency, test command outcomes, and next steps inside MEMORY.md as part of the workflow.

### CODE_STYLE
- Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

- Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

- If the task is unreasonable or infeasible, or if any of the tests are incorrect, please inform me rather than working around them. The solution should be robust, maintainable, and extendable.

---

### MEMORY-AWARE WORKFLOW
- Before applying any patch, read `MEMORY.md` to review recent reminders, checkpoints, and workflow rules so the agent does not refactor the same logic repeatedly.
- After every patch or substantive change, append a short entry in `MEMORY.md` in a neat markdown format describing what was applied, verification steps (e.g., tests run), and whether the geometric mean target improved.
- Treat `MEMORY.md` as the single source of truth for workflow status, ensuring every iteration references and augments it in markdown format to avoid loops.

### tcgen05_FLOW

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

### SCALE_LAYOUT_INVARIANTS (DO NOT BREAK)

Inputs include BOTH semantic CPU scales (sfa_ref_cpu/sfb*_ref_cpu) AND the
*physical* permuted GPU tensors (sfa_permuted/sfb*_permuted).

KERNEL MUST CONSUME: sfa_permuted/sfb1_permuted/sfb2_permuted ONLY.
DO NOT call/replicate to_blocked()/permute_scales_to_blocked() inside the kernel.
DO NOT "compact"/"linearize" scales unless you are proving byte-for-byte equivalence
 to the permuted physical layout.

Physical permuted scale tensor layout:
   (32, 4, rest_{m|n}, 4, rest_k, l)
Index mapping invariant (conceptual):
   mm  = i // 128
   mm32= i % 32
   mm4 = (i % 128) // 32
   kk  = j // 4
   kk4 = j % 4
   permuted[mm32, mm4, mm, kk4, kk, b] == semantic[i, j, b]

Any mismatch here produces large-magnitude numerical errors (NOT small atol/rtol drift).

---

### tcgen05_EXECUTION_INVARIANTS

1) ACCUMULATE:
   - ACCUMULATE must be False exactly once for the very first kblock of the
     very first k_tile contributing to an output tile, then True for all subsequent kblocks.
   - Do not re-toggle ACCUMULATE multiple times inside the same first kblock.

2) S2T scale copies:
   - S2T (Cp4x32x128b) copies for SFA/SFB* must complete before tcgen05.mma consumes them.
   - Prefer minimal synchronization (warp/cta) consistent with the chosen execution model.

3) Tensor/Layout discipline:
   - Treat every Tensor as (pointer + Layout(shape,stride)). No implicit transpose.
   - Any view/slice must preserve intended shape/stride contracts.

---

### EXECUTION_MODEL_REQUIREMENTS

- **Warpgroup MMA** – Assign a full warpgroup (4 warps) to every MMA computation. Do not gate the MMA launch to a single `warpidx`; hardware-multicast, MMA pipelines, and state transitions assume the complete warpgroup is active together.
- **Single scheduler per CTA** – Each CTA instantiates exactly one tile scheduler/loop. All roles (TMA, MMA, epilogue) must consume tiles from that shared assignment instead of launching their own `StaticPersistentTileScheduler.create(...)` instances.
- **TMEM allocator ownership** – TMEM allocate/free must be performed by exactly one warp per CTA. All other warps should `wait_for_alloc`, grab the pointer, and proceed without touching the allocator.
- **TMEM sizing discipline** – Compute `num_tmem_alloc_cols` from `num_accumulator_tmem_cols` plus the scale tensor columns (with alignment). Avoid hard-coded values such as 512 that over-consume TMEM; right-size allocations to the actual accumulator + scale footprint.
- **Epilogue subtile sync** – Do not place CTA-wide barriers inside the epilogue inner subtile loop. Only the dedicated store warp or pipeline (e.g., `PipelineTmaStore`) should synchronize subtiles via its own mbarrier semantics.
- **CTA sizing guidance** – Target 128-thread CTAs by default. Use 160- or 192-thread CTAs only when measurable geo-mean latency improvements justify the larger CTA, otherwise extra warps waste bandwidth/occupancy headroom.

---

### PERFORMANCE_PLAYBOOK (HIT ~4.89us GEOMEAN)

- Primary goal: approach SoL on the 4 benchmark shapes (memory+TC balanced).
- Avoid adding new global loads, repacks, or scale permutations in the hot path.
- Prefer increasing overlap: TMA pipelining + minimal barriers + persistent tile scheduling.
- Keep register pressure and SMEM footprint bounded so occupancy is not collapsed.

---

### COMPLIANCE_CHECK

**IMPORTANT**: Pattern-match CUTLASS SM100 tcgen05 examples/tutorials for exact mainloop + barrier placement (no extra syncs/copies)

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
