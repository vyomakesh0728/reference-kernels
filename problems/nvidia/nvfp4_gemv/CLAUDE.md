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

  NVFP4 GEMV KERNEL FIX: DEVICE-SIDE NUMERICS ONLY

  CRITICAL_CONSTRAINT
    <CRITICAL>
      ABSOLUTE RULE: ZERO HOST/PYTHON/ABI CHANGES ALLOWED

      Hammond harness ALWAYS passes L=1 in c.shape[2], so runtime L cannot decide rank-2 vs rank-3.
      All three benchmark shapes route through fp4_gemv_rank2_cta and ALL THREE MISMATCH
      because of device-side indexing bugs.

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
      Hammond Benchmark Shapes (all with harness L=1):
        - Shape 1: k=16384, l=1, m=128, n=7168
        - Shape 2: k=7168,  l=1, m=128, n=4096
        - Shape 3: k=2048,  l=1, m=128, n=7168

      Status:
        - Launch/TMA/cluster config is STABLE (compute-sanitizer clean)
        - Only device-side numerical bugs remain in fp4_gemv_rank2_cta
    </OBSERVATION>

  IMPLEMENTATION_PLAN
    <PLAN>
      STEP 1: Tiny Debug Case
        - Use harness with:
            M=16, K=64, L=1
        - Single CTA launch of:
            fp4_gemv_rank2_cta

      STEP 2: Device Printf Instrumentation
        - Add to fp4_gemv_rank2_cta under:
            blockIdx == (0,0), consumer warp, k_tile == 0

        - Print:
            // decoded A[0, 0:16] after SFA scaling
            // decoded B[0, 0:16] after SFB scaling
            // c_frag accumulator for row 0 before writeback

      STEP 3: Reference Comparison
        - In reference.py for same tiny shape/seed, print:
            # decoded a[0, 0:16] using fp4_lut + sfa scaling
            # decoded b[0, 0:16] using fp4_lut + sfb scaling
            # row-0 dot product

      STEP 4: Fix Device Numerics Line-by-Line
        <FIXES>

          SFA indexing in process_tile:
            // For sfa_stride == 16 (rank-2):
            sfa_idx = row * 16 + (col_packed >> 3)
              // col_packed counts FP4 packed bytes along K

          SFB indexing / broadcast:
            // Ensure k_idx // 8 mapping (one scale per 16 FP4 elements)
            // Must match reference.py's sfb[0, k//16, 0] indexing

          B decode nibble order:
            // Confirm low/high nibble extraction matches fp4_lut usage in reference.py

          Accumulator writeback:
            // Verify c_frag_* maps to correct D_batch[row_idx] with NO swaps

        </FIXES>

      STEP 5: Progressive Validation
        - Tiny case row 0 matches
            -> expand to M=128, K=64, L=1
        - Validate multiple rows
            -> run three official Hammond shapes
        - NO HOST CHANGES during validation
    </PLAN>

  PREVIOUS_MISTAKES
    <PREVIOUS_VERSION>
      - Tried adding stride parameters (VIOLATED ABI)
      - Attempted L→N remapping on host (FORBIDDEN)
      - Modified TMA/cluster logic (UNNECESSARY - already stable)
      - Assumed rank-3 kernel was reachable (WRONG - harness only uses rank-2)
    </PREVIOUS_VERSION>

  NON_GOALS
    <NON_GOALS>
      - Do NOT try to "enable" rank-3 kernel (unreachable under harness L=1)
      - Do NOT reconstruct conceptual L=4,8 shapes (harness doesn't expose them)
      - Do NOT touch anything outside fp4_gemv_rank2_cta device code
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

