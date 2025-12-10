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

<tcgen05_FLOW>
(Hardware Fused):
Global A/B (packed FP4) → TMA → SMEM (packed FP4, NO DECODE!)
Global SF (FP8) → TMA → SMEM (FP8 scales)
→ tcgen05.cp (SMEM FP8 scales → TMEM)
→ tcgen05.mma.mxf4.block_scale (reads SMEM packed FP4 + TMEM scales)
   ├─ Hardware decodes FP4→FP16 inside tensor core
   ├─ Hardware applies FP8 scales from TMEM
   └─ Hardware performs MMA
→ TMEM (FP32 accumulator, 128×128 tile)
→ TMEM.load → Registers → Global D
</tcgen05_FLOW>

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
