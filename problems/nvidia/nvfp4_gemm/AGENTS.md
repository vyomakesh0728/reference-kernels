<system-reminder>
      CRITICAL: ALWAYS USE mcp @morph tools, for code base search, file edits etc. 
      IMPORTANT: this context is relevant to your tasks. 
      You should always respond to this context unless it is highly relevant to your task.
</system-reminder>

<DO_NOT>
- WRITE SUMMARIES, DOCUMENTATION, EXPLANATIONS, AND GUIDES
</DO_NOT>

<STRICT_COMPETITION_RULES_DO_NOT_VIOLATE>

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
</STRICT_COMPETITION_RULES_DO_NOT_VIOLATE>

<INVESTIGATE_BEFORE_ANSWERING>
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
</INVESTIGATE_BEFORE_ANSWERING>

<CODE_STYLE>
Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please inform me rather than working around them. The solution should be robust, maintainable, and extendable.
</CODE_STYLE>

<CLEAN_UP>
The current submission.py contains legacy GEMV code and RANK-3 (L=4, L=8) cluster logic that must be removed:

REMOVE ALL RANK-3 LOGIC:
  - tma_load_3d_cluster_noarrive() function
  - tma_load_3d() wrapper with L parameter
  - sync_cluster_or_block(int L) conditional function
  - All "if (L == 1) {...} else {...}" branches that handle rank-3
  - cooperative_groups::cluster_group cluster = cg::this_cluster()
  - cluster.sync() calls
  - All comments mentioning "RANK-3", "L=4", "L=8", "cluster"
  - cudaLaunchKernelExC cluster launch path (keep only cudaLaunchKernel)
  - cudaFuncAttributeNonPortableClusterSizeAllowed attribute setting
  - cudaLaunchAttribute launchattr[1] and cluster dimension code
  - prefetchtile<> Ltemplate parameter (hardcode to 1)

REMOVE GEMV-SPECIFIC CODE:
  - bvecsmem (B vector broadcast buffer)
  - All "B is 1 × K vector" comments
  - "For GEMV B is broadcast" output writeback logic
  - "if (col == 0)" output guard (write ALL columns for GEMM)
  - Single cfrag[4] accumulators (need N-tiling for GEMM)

REMOVE SWIZZLE_NONE LOGIC:
  - encode_tma_vector() function (uses SWIZZLE_NONE)
  - All SWIZZLE_NONE references
  - "int SfaBoxK = L == 1 ? 16 : 128" conditionals (always use 128)
  - BOX_K 16-byte box configurations

KEEP ONLY:
  - Pure rank-2 (L=1) CTA-scope TMA code
  - SWIZZLE_128B configuration
  - BOX_K = 128 bytes
  - mma.sync.aligned PTX instructions
  - FP4/FP8 decode LUTs
  - Shared memory alignment helpers
  - mbarrier functions
</CLEAN_UP>
# NVFP4 GEMM Agent Plan (Updated)

<TODOs>

In `problems/nvidia/nvfp4_gemm/submission.py`, update the `#else  // USE_WGMMA_MAINLOOP` branch of `fp4_gemm_rank2_cta` to use **Blackwell tcgen05 tensor core instructions** instead of Hopper WGMMA, while leaving the classic `mma.sync` path (`#if !USE_WGMMA_MAINLOOP`) completely unchanged.

**Context / constraints**
- The current `#else` block contains a WGMMA stub:
  ```cpp
  // WGMMA-based mainloop stub body.
  // Chosen WGMMA tile (logical): WG_M=128, WG_N=128, WG_K=64.
  // Consumer warpgroup: warps 0..3 (128 threads). Other warps are idle in this branch.
  ...
  if (warp_id < 4) {
      uint32_t smem_a_addr = cvta_to_shared_u32(smem_A_wgmma);
      uint32_t smem_b_addr = cvta_to_shared_u32(smem_B_wgmma);
      asm volatile(
          "wgmma.fence.sync.aligned;\\n"
          "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
          "{%0, %1, %2, %3}, %4, %5, %6, 1, 1, 0;\\n"
          "wgmma.commit_group.sync.aligned;\\n"
          "wgmma.wait_group.sync.aligned 0;\\n"
          : "+f"(wg_accum[0]), "+f"(wg_accum[1]), "+f"(wg_accum[2]), "+f"(wg_accum[3])
          : "r"(smem_a_addr), "r"(smem_b_addr), "r"(0)
      );
  }
  ```
- This does not assemble for `.target sm_100a` because WGMMA is Hopper‑only; Blackwell uses **tcgen05.mma** instead.[2][4]
- Keep `#define USE_WGMMA_MAINLOOP 0` as the default in this repo so the contest harness never tries to compile or run this path; the tcgen05 branch is for local SM100/SM100a testing only.[1]

**Goal**
Replace the WGMMA inline PTX inside the `#else  // USE_WGMMA_MAINLOOP` branch with a **minimal, compiling tcgen05.mma stub** that:
- Only compiles when `__CUDA_ARCH__ = 1000` (SM100/SM100a).  
- Issues a single **synchronous** `tcgen05.mma` op per K‑tile from a designated consumer warpgroup (warps 0..3).  
- Accumulates into the existing `wg_accum[4]` array but does **not** write anything to `D` yet.  
- Leaves everything else (TMA prologue, mbarriers, classic path) untouched.

**Step A: Clean up macro + guards**
1. Ensure the global macro at the top reads:
  ```cpp
  // Enable advanced tcgen05-based mainloop when non-zero.
  #ifndef USE_WGMMA_MAINLOOP
  #define USE_WGMMA_MAINLOOP 0
  #endif
  ```
  Do not change any `#if __CUDA_ARCH__ = 900` guards around the kernel; just add an extra `#if __CUDA_ARCH__ = 1000` around the tcgen05 inline PTX inside the `#else` branch.
2. Inside the `#else  // USE_WGMMA_MAINLOOP` block, wrap the tcgen05 PTX with:
  ```cpp
  #if __CUDA_ARCH__ = 1000
    // tcgen05.mma stub here
  #endif
  ```
  so that older architectures see a no‑op mainloop (just TMA + waits + syncs, no tensor core instructions).

**Step B: Replace WGMMA with tcgen05.mma**
1. In the `if (warp_id < 4)` block, delete the WGMMA inline PTX (`wgmma.fence`, `wgmma.mma_async`, `wgmma.commit_group`, `wgmma.wait_group`).[1]
2. Replace it with a **single synchronous tcgen05.mma call** modeled after CUTLASS’s SM100 GEMM examples (e.g. `72a_blackwell_nvfp4_bf16_gemm`):[3][2]
  - Use an instruction form that:
    - Multiplies A and B tiles in SMEM (or TMEM) and accumulates into FP32 or FP16 C.  
    - Has a compact M×N×K shape (e.g. 64×128×16 or 64×64×32) supported by tcgen05.  
  - Use `smem_A_wgmma` and `smem_B_wgmma` as the base pointers for A and B operands. It is acceptable for this stub to treat the existing layout as row‑major [WG_M, WG_K] and [WG_K, WG_N] and ignore swizzle/optimal layout for now.
  - Map the tcgen05 C fragment to `wg_accum[0..3]` as the accumulator operands (similar to how the WGMMA stub did, but adjusted to tcgen05’s operand ordering).
  - Use a **synchronous** variant (e.g. `tcgen05.mma.sp.sync.aligned.*`) so you do **not** need explicit fence/commit/wait group calls in this stub.
3. Add comments that reference the exact CUTLASS example and PTX ISA section you’re following, so it’s clear how to expand this later.

**Step C: Keep behavior and layout assumptions**
1. Preserve the existing structure in the `#else` branch:
  - TMA prefetch prologue over `s = 0..StageCount-2`.  
  - Main `for (int k_tile = 0; k_tile < K; k_tile += TileK)` loop with prefetch of `next_k`, mbarrier waits on `mbar_a` and `mbar_b`, and `__syncthreads()` before and after the compute block.  
  - Phase flipping at the end of each full pipeline cycle.[1]
2. Keep the role of warps unchanged:
  - Warps 0–3 are the tcgen05 consumer warpgroup (`if (warp_id < 4)`).  
  - Other warps are effectively idle in this branch for now.
3. Do **not** change or touch:
  - The classic `mma.sync` path and its epilogue under `#if !USE_WGMMA_MAINLOOP`.  
  - Any TMA descriptor setup or `prefetch_tile` implementation.  
  - The FP4 decode path or shared‑memory layout used by the classic `process_tile` function.

**Step D: No epilogue yet**
1. Leave `wg_accum` unused outside the WGMMA/tcgen05 compute block; do not write it into `c_accum` or `D` yet.
2. Keep the existing TODO:
  ```cpp
  // TODO: Map WGMMA/tcgen05 accumulators to c_accum / D epilogue once layout is defined.
  ```
  and update the comment to mention tcgen05 instead of WGMMA.

**Step E: Sanity checks (local, not in contest harness)**
- Ensure the kernel **compiles for `sm_100` / `sm_100a`** with `USE_WGMMA_MAINLOOP` set to 1 using a recent CUDA toolchain that supports tcgen05 (check against the PTX ISA docs and CUTLASS Blackwell examples).[2][3]
- With `USE_WGMMA_MAINLOOP` left at 0, confirm that:
  - The PTX for tcgen05 is not emitted (no tcgen05 mnemonics in the generated SASS/PTX for the contest build).  
  - All correctness tests and benchmarks behave exactly as before.

</TODOs>


<COMPLIANCE_CHECK>

  Your kernel implementation MUST be compliant with:

  1. reference.py
     - Match torch._scaled_mm behavior exactly for all test cases.
     - Handle B matrix transposition: b_ref[:, :, l_idx].transpose(0, 1).
     - Use sfa_ref_cpu / sfb_ref_cpu as the *semantic* source of scales.
     - Ensure that for every FP4 K‑block, the scale applied by your kernel
       is equal to the scale that would be used by:
         to_blocked(sfa_ref_cpu[:, :, l_idx]) and
         to_blocked(sfb_ref_cpu[:, :, l_idx])
       in ref_kernel.
     - Meet output tolerance: rtol=1e‑3, atol=1e‑3.

  2. task.yml
     - Input tuple has 7 elements as described above.
     - M divisible by mma_tiler_mn[0] (e.g., 16 or 32 factors).
     - N divisible by mma_tiler_mn[1] (e.g., 8 or 16 factors).
     - K divisible by 256.
     - All benchmark shapes have L=1.
     - Ranking by geometric mean runtime over test cases.
     - Target SoL latencies: ~8.994 μs, 2.354 μs, 1.333 μs for the official
       benchmark shapes (as documented in the competition materials).

</COMPLIANCE_CHECK>
