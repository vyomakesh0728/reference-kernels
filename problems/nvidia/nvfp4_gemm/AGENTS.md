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
use @morph tools to ease search/edits. 

Workspace context:
- This repo is the NVFP4 GEMM competition harness.  
- There is a **full, up-to-date CUTLASS repo** at `./cutlass`
- I want a **standalone minimal .cu file** to experiment with `tcgen05.mma` + TMEM on SM100/SM100a, outside of the main sub kernel.

### Goal
Create a single CUDA source file `tcgen05_playground.cu` that:
- Compiles with:
  ```bash
  nvcc -std=c++17 -O3 -arch=sm_100a -I./cutlass/include tcgen05_playground.cu -o tcgen05_playground
  ```
- Launches a **tiny GEMM-like kernel** that:
  - Allocates TMEM with `tcgen05.alloc`.[2][3]
  - Loads small A/B tiles (e.g. 128×16 and 16×128 of `half`) from GMEM into SMEM.  
  - Issues a **single `tcgen05.mma.cta_group::1`** instruction to compute C = A×B, accumulating into TMEM.[4][3]
  - Uses `tcgen05.ld` (or the CUTLASS “UMMA” abstractions) to read accumulators back from TMEM into registers/SMEM.[2]
  - Writes the final C tile to GMEM so the host can print a few values.
- The primary objective is: **ptxas must accept the `tcgen05.mma` and `tcgen05.ld` instructions with correct operands and descriptors** on SM100/SM100a.

### Constraints / preferences
- Keep it **as small and explicit as possible**:
  - One `.cu` file, one kernel, one `main()`.  
  - Hard-code a single tile, e.g. C is 128×128, K=16.  
  - Initialize A and B on the host to all ones so C should be a constant (e.g. 16) everywhere.
- It is fine (and preferred) to follow the Colfax / CUTLASS Blackwell tutorials and examples, but please:
  - Inline the minimal needed code into this `.cu` instead of wiring the full CUTLASS GEMM stack.  
  - Use CUTLASS headers only for **type helpers or macros**, not for launching a whole GEMM.[5][2]
- Do **not** modify any existing files in this repo (especially not @sub ); just add `tcgen05_playground.cu`.

### Implementation sketch
Please implement roughly this flow (filling in the real PTX signatures and descriptors from the docs/examples):

1. **Host side (`main`)**
  - Allocate device buffers:
    - `A_dev`: shape, dtype `half`.[6]
    - `B_dev`: shape, dtype `half`.[6]
    - `C_dev`: shape, dtype `float` or `half`.[6]
  - Initialize A and B on the host to 1.0, copy to device.
  - Launch `tcgen05_kernel<<<1, 128` (one CTA of 128 threads).
  - Copy `C_dev` back and print the first few elements (e.g. `C[0,0]`, `C[0,1]`) to verify they equal K.

2. **Device side kernel (`tcgen05_kernel`)**
  - Use dynamic shared memory to allocate A/B tiles:  
    - `extern __shared__ uint8_t smem_raw[];`  
    - Carve out `half* smem_A` and `half* smem_B` with simple contiguous layouts.
  - Have one warp (or one thread) copy A and B from GMEM into SMEM with plain `ld.global` / `st.shared` loops (no TMA in this playground).  
  - Synchronize the CTA.
  - Have a **single elected thread** (e.g. `if (threadIdx.x == 0)`) perform:
    - `tcgen05.alloc` to allocate TMEM for the accumulator.[2]
    - Construction of A/B descriptors and the instruction descriptor (`idesc`) required by `tcgen05.mma`, using the PTX ISA 9.0 docs and/or Colfax tutorial as a guide.[7][2]
    - A single `tcgen05.mma.cta_group::1.kind::f16 ...` call that:
      - Reads operands from SMEM (A and B tiles).  
      - Accumulates into TMEM.  
    - A `tcgen05.ld` to copy the accumulator tile from TMEM into registers or SMEM.[2]
    - A simple loop to write the resulting C tile to `C_dev`.
  - Synchronize again if needed, then return.

3. **PTX details**
  - Use **inline PTX** for `tcgen05.alloc`, `tcgen05.mma`, and `tcgen05.ld`.  
  - The tcgen05 instructions and operand lists must be copied from **working examples**:
    - The Colfax CUTLASS tutorial “Writing GEMM Kernels Using Tensor Memory for NVIDIA Blackwell GPUs” (UMMA / Tensor Memory).[8][2]
    - CUTLASS example 04_mma_tma_2sm_sm100.cu or similar, already present under @cut
    - The CUTLASS “Blackwell SM100 GEMMs” docs for supported `tcgen05.mma` kinds and operand conventions.
  - **Important:** The current inline PTX in `submission.py` for `tcgen05.mma` causes  
    `ptxas ... error   : Arguments mismatch for instruction 'tcgen05.mma'`.
    In this playground, you must:
    - Use a tcgen05 variant that ptxas actually accepts on `sm_100a`.  
    - Ensure the number and types of operands (TMEM handle, A/B descriptors, instruction descriptor, lane disable mask, scale flags, etc.) exactly match the PTX ISA example.

4. **Sanity checks**
  - Add a small comment block at the top of `tcgen05_playground.cu` describing:
    - Which CUTLASS/Colfax example and PTX ISA section the tcgen05 usage is based on (file name + section heading).[3][2]
  - After building and running:
    - `./tcgen05_playground` should print that `C[0,0]`, `C[0,1]`, etc. are equal to K (e.g. 16.0f), proving the tcgen05 path works.

### Deliverable
- A single new file `tcgen05_playground.cu` checked into the repo root, with all the above behavior implemented and comments pointing back to the relevant CUTLASS / Colfax / PTX docs for tcgen05 and TMEM.
- Do **not** touch any other files; I will later transplant the working inline PTX sequence into my `USE_tcgen05_MAINLOOP` branch in `submission.py` myself.

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
