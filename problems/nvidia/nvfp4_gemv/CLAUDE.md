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

<CONCISE_CRITICAL_INSTRUCTIONS>

  NVFP4 GEMV KERNEL FIX: CLUSTER DEADLOCK IN RANK-3 KERNEL

  CRITICAL_CONSTRAINT
    <CRITICAL>
      ABSOLUTE RULE: ZERO HOST/PYTHON/ABI CHANGES ALLOWED

      LOCKED INTERFACES (DO NOT TOUCH):
        - C++ launcher:
            launch_fp4_gemv_optimized(A, B, SFA, SFB, D, M, K, L, K_scales_padded)

        - Kernel signatures:
            fp4_gemv_rank2_cta(..., M, K, L, K_scales_padded)
            fp4_gemv_rank3_cluster(..., M, K, L, K_scales_padded)

        - Problem layout:
            a   : [M, K, L]
            b   : [1, K, L]
            sfa : [M, K/16, L]
            sfb : [1, K/16, L]
            c   : [M, 1, L]

        - Python custom_kernel interface
        - TMA descriptors and shared-memory allocation

      FORBIDDEN ACTIONS:
        - Adding stride parameters anywhere
        - Remapping L/N on host or Python side
        - Changing cluster launch logic
        - Modifying any host-side code
    </CRITICAL>

  CURRENT_STATE
    <OBSERVATION>
      Official Benchmark Shapes (from test_kernel_only.py):
        - Shape 1: M=7168, K=16384, L=1 (rank-2 CTA) - WORKS, but 2753x slower
        - Shape 2: M=4096, K=7168,  L=8 (rank-3 cluster) - DEADLOCKS
        - Shape 3: M=7168, K=2048,  L=4 (rank-3 cluster) - UNTESTED

      Status (from cuda-gdb):
        - Shape 1 (L=1): Uses fp4_gemv_rank2_cta, executes but numerically wrong (2753x slower)
        - Shape 2 (L=8): Uses fp4_gemv_rank3_cluster, DEADLOCKS in cluster synchronization
        - All blocks stuck alternating between two PCs (0x7ffea15f9a70 and 0x7ffea15f61f0)
        - Launch config: grid=(32,8,1), block=(320,1,1), shared_bytes=189440
    </OBSERVATION>

  IMPLEMENTATION_PLAN
    <PLAN>
      PRIORITY 1: Fix Cluster Deadlock in fp4_gemv_rank3_cluster (L=8, L=4)

        DEADLOCK SYMPTOMS (from cuda-gdb):
          - Blocks alternating between two PCs
          - Likely stuck in cluster.sync() or barrier.arrive_tx/wait
          - Grid=(32,8,1) means 256 blocks across 8 batches

        POSSIBLE CAUSES:
          1. Mismatched barrier/cluster sync counts
          2. Incorrect cluster dimensions or launch config
          3. TMA fences not properly ordered
          4. Producer-consumer barrier mismatch

        DEBUG APPROACH:
          - Add printf in fp4_gemv_rank3_cluster at each sync point
          - Check if all blocks in cluster reach each barrier
          - Verify barrier phase/expected_tx match across all blocks
          - Check TMA descriptor batch indexing for L>1

      PRIORITY 2: Fix Numerical Issues in fp4_gemv_rank2_cta (L=1)

        ISSUE: Executes but 2753x slower than target (23.7ms vs 8.6μs)

        POSSIBLE CAUSES:
          1. Wrong indexing causing cache thrashing
          2. Incorrect SFA/SFB scale indexing
          3. FP4 decode nibble order wrong
          4. Accumulator writeback misaligned

        DEBUG APPROACH:
          - Use M=16, K=64, L=1 test case
          - Add printf for decoded A[0, 0:16], B[0, 0:16]
          - Compare with reference.py for same seed
          - Fix indexing bugs one by one
    </PLAN>

  PREVIOUS_MISTAKES
    <PREVIOUS_VERSION>
      - Misunderstood that L=1 for all benchmarks (WRONG - actually L=1, L=8, L=4)
      - Thought rank-3 cluster was unreachable (WRONG - it's used but deadlocks)
      - Focused only on numerics, missed cluster synchronization bug
    </PREVIOUS_VERSION>

  NON_GOALS
    <NON_GOALS>
      - Do NOT modify host-side launch logic
      - Do NOT add stride parameters
      - Do NOT change TMA descriptor creation (unless cluster batch indexing wrong)
    </NON_GOALS>

  TOOLS
    <TOOLS>
      - compute-sanitizer:
          Already clean, confirms launch/TMA stable
      - printf:
          Device-side conditional printing for row-0 debug
      - reference.py:
          Element-wise numerical ground truth
      - Target:
          <55.577 μs geomean latency after numerical fix
    </TOOLS>

</CONCISE_CRITICAL_INSTRUCTIONS>

