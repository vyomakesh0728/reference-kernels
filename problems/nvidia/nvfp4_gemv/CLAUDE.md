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

<IMPLEMENTATION_PLAN>
  <PREVIOUS_VERSION>
    sahasra_copy.py is the previous, working single-kernel version.
    It contains one kernel that handles both:
      - RANK-2, L=1: TMA CTA + SWIZZLE_NONE + box_k = 16 (bytes / scales).
      - RANK-3, L=4 or 8: TMA CLUSTER (non-multicast) + SWIZZLE_128B + 1024-byte swizzled regions.
    All decode, math, tensor layouts, and TMA coordinate logic in sahasra_copy.py are trusted and must be preserved.
  </PREVIOUS_VERSION>

  <KERNEL_SPLIT_REQUIREMENTS>
    We now want TWO kernels, but with the SAME core behavior as sahasra_copy.py:

    1) fp4gemv_rank2_cta<TileM, TileK, Threads>  (RANK-2, L == 1)
       - CTA-only execution, no cluster usage.
       - Uses RANK-2 TMA descriptors:
         * TMA CTA, SWIZZLE_NONE, SfaBoxK = 16.
       - Uses .shared::cta TMA only and __syncthreads() for sync.
       - No cooperative_groups::this_cluster, no .shared::cluster or ctagroup2 TMA, no cluster mbarrier peer-mask.
       - Its decode + MMA + writeback logic MUST be identical to the RANK-2 path in sahasra_copy.py,
         aside from trivial parameterization (no new indexing formulas).

    2) fp4gemv_rank3_cluster<TileM, TileK, Threads>  (RANK-3, L > 1 : L=4 or L=8)
       - Cluster launch, 2 CTAs per cluster, same as sahasra_copy.py.
       - Uses RANK-3 TMA descriptors:
         * TMA CLUSTER NON-MULTICAST, SWIZZLE_128B, SfaBoxK = 128,
           1024-byte aligned SWIZZLE regions in shared memory.
       - Uses the existing cluster TMA pattern from sahasra_copy.py:
         ctagroup2.shared::cluster.global.mbarriercomplete_txbytes,
         mbarrier peer-mask (Sm100MmaPeerBitMask), and syncclusterorblock(L).
       - Its decode + MMA + writeback logic MUST be identical to the RANK-3 path in sahasra_copy.py.

    The ONLY structural change allowed is splitting the single kernel from sahasra_copy.py into
    these two entry kernels and updating the host launch code to dispatch:

      - If L == 1:
          Launch fp4gemv_rank2_cta<kTileM, kTileK, kThreads> with a CTA-only launch
          (no clusterDim, no cluster attributes).

      - If L > 1 (L = 4 or 8):
          Launch fp4gemv_rank3_cluster<kTileM, kTileK, kThreads> as a cluster kernel
          with cudaLaunchKernelExC and clusterDim = dim3(2,1,1) plus
          cudaFuncSetAttribute(...NonPortableClusterSizeAllowed, 1), exactly as in sahasra_copy.py.

    DO NOT change:
      - FP4 / FP8 decode implementations or math.
      - TMA descriptor dims/strides/box settings.
      - SFA/SFB indexing or shared-memory layouts.
      - ldmatrix + mma.sync fragment mapping.
    Reuse the sahasra_copy.py logic verbatim wherever possible; only factor it into shared helpers
    and two kernels without altering behavior.
  </KERNEL_SPLIT_REQUIREMENTS>
</IMPLEMENTATION_PLAN>

<TOOLS>
  Always diff sahasra_copy.py and submission.py before and after edits.
  Treat sahasra_copy.py as the source of truth for decode, SFA/SFB indexing,
  TMA dims/strides/box, and PTX mnemonics; copy those sections directly instead
  of rewriting them from scratch.
</TOOLS>
