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
This message gives everything needed to mechanically transplant the working tcgen05 playground behavior into the fp4 GEMM kernel’s tcgen05 path without touching TMA, decode, or the classic mma.sync path.

**Goal:**  
Transplant the working `tcgen05_playground.cu` tcgen05+TMEM mainloop into the `USE_tcgen05_MAINLOOP` path of `fp4gemm_rank2_cta` in `submission.py`, reusing the existing TMA+decode pipeline and epilogue.

**Reference playground (authoritative behavior):**

This kernel runs and prints `C[0,0]=C[0,1]=64` with no hangs on B200.

### Tasks in `submission.py` (fp4 GEMM kernel)

1. **Locate the tcgen05 stub path.**

   - In `submission.py`, find the CUDA kernel template:

     ```cpp
     template<int TileM, int TileK, int Threads>
     __global__ __launch_bounds__(Threads)
     void fp4gemm_rank2_cta(...)
     ```

   - Inside it, there is:

     ```cpp
     #if !USE_tcgen05_MAINLOOP
       // classic mma.sync mainloop ...
     #else
       // tcgen05-based mainloop stub body ...
     #endif
     ```

   - All changes go inside the `#else  // USE_tcgen05_MAINLOOP` block. Do not touch the classic path or the TMA/epilogue code.[1]

2. **Wire TMEM alloc/dealloc exactly like the playground, using the existing decoded SMEM.**

   - In `fp4gemm_rank2_cta`, **reuse** the decoded FP16 shared buffers:

     ```cpp
     half* af16_smem = ...;  // already allocated as decoded A tile
     half* bf16_smem = ...;  // already allocated as decoded B tile
     ```

     These correspond to `smem_A` and `smem_B` in the playground.[2][1]

   - Add a CTA‑local shared uint32 for TMEM in the tcgen05 path:

     ```cpp
     __shared__ uint32_t tmem_base_ptr_tcgen05;
     ```

     Place this in the tcgen05 section, reusing existing shared memory if needed (just a single 4‑byte symbol).[1]

   - In the tcgen05 mainloop prologue (after TMA+decode has filled `af16_smem` / `bf16_smem` for the current K‑tile and after `__syncthreads()`), insert:

     ```cpp
     const int tid    = threadIdx.x;
     const int warpid = tid >> 5;
     const int laneid = tid & 31;  // already computed earlier; reuse if present

     __syncthreads();
     if (tid == 0) {
         tmem_base_ptr_tcgen05 = 0;
     }
     __syncthreads();

     // TMEM alloc by fully active warp 0
     if (warpid == 0) {
         uint32_t dst_smem = cvta_to_shared_u32(&tmem_base_ptr_tcgen05);
         int      num_cols = 256;
         asm volatile(
             "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
             :
             : "r"(dst_smem), "r"(num_cols));
     }
     __syncthreads();

     uint32_t tmem_c = tmem_base_ptr_tcgen05;
     ```

     Use the existing `cvta_to_shared_u32` helper already defined in this file (there is one for TMA) instead of re‑defining it.[3][1]

   - In the tcgen05 epilogue for the K‑loop (before returning or moving to D writeback), free TMEM:

     ```cpp
     if (warpid == 0) {
         uint32_t cols  = 256;
         uint32_t taddr = tmem_c;
         asm volatile(
             "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
             :
             : "r"(taddr), "r"(cols));
     }
     ```

     This dealloc should be done once per CTA after all tcgen05 MMAs for that CTA’s tile have completed.[4][3]

3. **Replace the current tcgen05 inline PTX stub with the UMMA `MmaOp::fma` sequence from the playground.**

   - In the `#else  // USE_tcgen05_MAINLOOP` block, delete the existing hand‑written `tcgen05.mma` asm stub and any dummy `wgaccum` logic that doesn’t feed into D.[1]

   - Immediately after TMEM alloc (and after `af16_smem` / `bf16_smem` are ready for this K‑tile), insert the UMMA setup and FMA, mapping symbols exactly:

     ```cpp
     using TypeA = half;
     using TypeB = half;
     using TypeC = float;

     using MmaOp = cute::SM100_MMA_F16BF16_SS<
         TypeA, TypeB, TypeC,
         /*M*/ 128, /*N*/ 128,
         cute::UMMA::Major::K,
         cute::UMMA::Major::K>;

     using TiledMMA = decltype(cute::make_tiled_mma(MmaOp{}));
     TiledMMA tiled_mma = cute::make_tiled_mma(MmaOp{});

     auto bM = cute::tile_size<0>(tiled_mma);
     auto bN = cute::tile_size<1>(tiled_mma);
     auto bK = cute::tile_size<2>(tiled_mma);

     auto mma_shape_A = cute::partition_shape_A(tiled_mma, cute::make_shape(bM, bK));
     auto mma_shape_B = cute::partition_shape_B(tiled_mma, cute::make_shape(bN, bK));

     auto sA_layout = cute::UMMA::tile_to_mma_shape(
         cute::UMMA::Layout_K_SW32_Atom<TypeA>{}, mma_shape_A);
     auto sB_layout = cute::UMMA::tile_to_mma_shape(
         cute::UMMA::Layout_K_SW32_Atom<TypeB>{}, mma_shape_B);

     auto tCsA = cute::make_tensor(cute::make_smem_ptr(af16_smem), sA_layout);
     auto tCsB = cute::make_tensor(cute::make_smem_ptr(bf16_smem), sB_layout);

     auto tCrA = cute::make_tensor<cute::UMMA::smem_desc<cute::UMMA::Major::K>>(tCsA);
     auto tCrB = cute::make_tensor<cute::UMMA::smem_desc<cute::UMMA::Major::K>>(tCsB);

     cute::UMMA::SmemDescriptor a_smem_desc = *tCrA.data();
     cute::UMMA::SmemDescriptor b_smem_desc = *tCrB.data();

     uint64_t a_desc = static_cast<uint64_t>(a_smem_desc);
     uint64_t b_desc = static_cast<uint64_t>(b_smem_desc);

     uint64_t idescE = cute::UMMA::make_runtime_instr_desc<
         TypeA, TypeB, TypeC,
         /*M*/ 128, /*N*/ 128,
         cute::UMMA::Major::K,
         cute::UMMA::Major::K>();

     uint32_t scaleC = 1u;

     // One tcgen05.mma FMA into TMEM for this K-tile
     MmaOp::fma(a_desc, b_desc, tmem_c, scaleC, idescE);
     ```

     - Use `half` from this kernel instead of `half_t` if that is the local typedef.  
     - Make sure `128x128x128` matches the actual CTA tile shape used in `fp4gemm_rank2_cta` (current code asserts `TileM=128`, `TileN=128`, `TileK=128`).[5][1]

   - For now, do **not** try to read TMEM back into `caccum`. Keep the existing `caccum` / epilogue path in the classic branch; in the tcgen05 branch you can leave D unwritten or just treat this as a timing-only mainloop until you decide on a TMEM→register mapping.

4. **Build & sanity check.**

   - Ensure the CUDA source in `submission.py` still includes:

     ```cpp
     #include <cute/tensor.hpp>
     #include <cute/arch/mma_sm100_umma.hpp>
     #include <cute/arch/mma_sm100_desc.hpp>
     ```

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
