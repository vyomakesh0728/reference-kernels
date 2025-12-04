<system-reminder>
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

<CRITICAL_INSTRUCTIONS>

  GOAL
    Implement an NVFP4 block‑scaled GEMM kernel on NVIDIA B200 (SM_100a) that:
      - Computes full GEMM: C[M, N] = A[M, K] @ B[N, K]^T with FP4 inputs and FP8 block scales
      - Exactly matches the PyTorch reference (`reference.py` / `torch._scaled_mm`) within rtol=1e‑3, atol=1e‑3
      - Achieves near speed‑of‑light bandwidth on all benchmark shapes

  KEY FILES
    - task.yml       : shapes, constraints, and test list
    - reference.py   : canonical semantics via torch._scaled_mm
    - submission.py  : custom kernel (PTX + TMA)
    - eval_better_bench.py : cloud correctness + benchmarking harness

  INPUT / OUTPUT CONTRACT
    - Input tuple (from generate_input in reference.py):
        (a_ref, b_ref,
         sfa_ref_cpu, sfb_ref_cpu,
         c_ref)

      Shapes:
        a_ref           : [M, K/2, L]   (torch.float4_e2m1fn_x2, packed FP4)
        b_ref           : [N, K/2, L]   (torch.float4_e2m1fn_x2, packed FP4)
        sfa_ref_cpu     : [M, K_scales, L]  (torch.float8_e4m3fn)
        sfb_ref_cpu     : [N, K_scales, L]  (torch.float8_e4m3fn)
        c_ref           : [M, N, L] (torch.float16)

      Where:
        K_scales = ceil_div(K, 16)  // one FP8 scale per 16 FP4 values along K

    - Output:
        custom_kernel(data) must return a tensor with same shape/dtype/layout as c_ref
        and match ref_kernel(data) within rtol=1e‑3, atol=1e‑3.

  REFERENCE SEMANTICS (MUST MATCH)
    - ref_kernel in reference.py:
        for each batch l_idx:
          scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
          scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
          res = torch._scaled_mm(
                  a_ref[:, :, l_idx],           // [M, K/2] FP4 x2
                  b_ref[:, :, l_idx].transpose(0, 1), // [K/2, N]
                  scale_a.cuda(),               // flattened block scales for A
                  scale_b.cuda(),               // flattened block scales for B
                  bias=None,
                  out_dtype=torch.float16
                )
          c_ref[:, :, l_idx] = res

    - to_blocked(sfa_ref_cpu[:, :, l_idx]) / to_blocked(sfb_ref_cpu[:, :, l_idx]) define the
      *logical* mapping from (row, K‑block) → scale index:
        - sfa_ref_cpu / sfb_ref_cpu are K‑major [M/N, K_scales, L]
        - to_blocked() reshapes + permutes to the block‑layout expected by torch._scaled_mm
        - Your kernel may use any internal layout (simple) as long as, for each
          FP4 block of 16 elements along K, you multiply by the same scale that the reference
          would use at that block.

  PRIORITY 1: GEMM, NOT GEMV
    - Full GEMM:
        - A: M×K
        - B: N×K (note: B is transposed vs. usual GEMM call)
        - C: M×N
    - Must write ALL columns of C, not just column 0.
    - Must account for B transpose (b_ref[:, :, l_idx].T) in reference semantics.

  PRIORITY 2: TMA DESCRIPTOR SETUP (Rank‑2, L=1)
    - L is always 1 → use rank‑2 descriptors, no rank‑3 / cluster descriptors.

    - A matrix (M × K):
        rank    = 2
        dims    = [K_packed, M]   // [K/2, M] in bytes
        box     = [128, 128]      // [bytes_K, rows_M]
        strides = [K_packed]      // row stride in bytes (K/2)
        swizzle = CU_TENSOR_MAP_SWIZZLE_128B

    - B matrix (N × K):
        rank    = 2
        dims    = [K_packed, N]   // [K/2, N] in bytes
        box     = [128, 128]      // [bytes_K, rows_N]
        strides = [K_packed]      // row stride in bytes
        swizzle = CU_TENSOR_MAP_SWIZZLE_128B

    - Scale factors (block‑scaled FP4):
        Logical reference source:
          - sfa_ref_cpu : [M, K_scales, L]
          - sfb_ref_cpu : [N, K_scales, L]
          - to_blocked() + torch._scaled_mm define the block mapping.
        Kernel implementation:
          - May store scales in simple K‑major 2D layout for TMA:
              SFA_bytes: [K_scales_padded, M] in uint8
              SFB_bytes: [K_scales_padded, N] in uint8
            with:
              dims_SFA = [K_scales_padded, M]
              dims_SFB = [K_scales_padded, N]
              box      = [128, 128]
              strides  = [K_scales_padded]
          - K_scales_padded must be at least 128 and a multiple of 128
            (for SWIZZLE_128B and 128‑byte boxes).
          - Indexing in the kernel must be chosen so that each FP4 K‑block
            uses the same scale value as in the reference (via to_blocked()).

  PRIORITY 3: MEMORY ALIGNMENT & CONSTRAINTS
    - Global:
        - A, B, SFA, SFB base pointers: 128‑byte aligned
        - Leading dimensions (K_packed, K_scales_padded): multiples of 128
        - K divisible by 256 (from task.yml)

    - Shared:
        - TMA destinations: 1024‑byte aligned (required for SWIZZLE_128B)
        - Non‑TMA regions: 128‑byte aligned
        - Respect per‑block dynamic shared memory limit on B200 (~227 KiB).
          shared_bytes <= 227 * 1024, otherwise cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
          fails.

  PRIORITY 4: BARRIER CONFIGURATION (MULTI‑STAGE PIPELINE)
    - Use per‑stage mbarriers for A and B (and logically for SFA / SFB):
        __shared__ uint64_t mbar_a[StageCount];
        __shared__ uint64_t mbar_b[StageCount];

    - StageCount:
        - Minimum: 2
        - Target: 3 (for better overlap)
        - But must respect shared memory budget — may reduce TileM/TileN instead of stages.

    - Per‑stage transaction sizes (example TileM=TileN=128, TileK=256):
        tx_a ≈ TILE_M × (TILE_K/2) + TILE_M × (TILE_K/16)
        tx_b ≈ TILE_N × (TILE_K/2) + TILE_N × (TILE_K/16)

  PRIORITY 5: SHARED MEMORY LAYOUT
    - Triple‑buffered or double‑buffered TMA destinations (depending on StageCount):
        __shared__ uint8_t a_packed[StageCount][TILE_M × TILE_K/2];
        __shared__ uint8_t b_packed[StageCount][TILE_N × TILE_K/2];
        __shared__ uint8_t sfa_stage[StageCount][TILE_M × BoxK_scales]; // e.g., 128 bytes * TILE_M
        __shared__ uint8_t sfb_stage[StageCount][TILE_N × BoxK_scales];

      All TMA stages 1024‑byte aligned.

    - Decoded tiles (128‑byte aligned):
        __shared__ half a_fp16[TILE_M × a_stride];  // a_stride = TileK + padding
        __shared__ half b_fp16[TILE_N × b_stride];  // b_stride = TileK + padding

  PRIORITY 6: KERNEL LAUNCH CONFIGURATION
    - Grid:
        grid.x = ceil(M / TILE_M)
        grid.y = ceil(N / TILE_N)
        grid.z = L (always 1)

    - Block:
        block.x = 256  // 8 warps per CTA

    - Launch:
        - CTA scope only (cudaLaunchKernel)
        - No clusters, no rank‑3 descriptors.

  PRIORITY 7: MMA LOOP STRUCTURE
    - Per‑CTA tile:
        int m_tile = blockIdx.x * TILE_M;
        int n_tile = blockIdx.y * TILE_N;

    - K‑tile loop:
        for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
          // Issue TMA for next stage (if any)
          // Wait on barrier for current stage
          // Decode FP4 → FP16 using FP8 scales
          // MMA using mma.sync.aligned.m16n8k16
        }

    - Writeback:
        - Each warp computes a 16×8 output fragment per MMA
        - Accumulate across 16 N‑sub‑tiles per warp to cover TILE_N=128
        - Store entire M×N tile to global memory (subject to edge checks)

  PRIORITY 8: PTX MMA INSTRUCTION
    - Use:
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    - Fragment handling:
        - 4 FP32 accumulators per thread for a 16×8 tile per warp
        - Use ldmatrix.sync.aligned to load A/B fragments from shared memory.

  PRIORITY 9: TMA PREFETCH PATTERN
    - Only warp 0, lane 0 issues TMA:
        if (warp_id == 0 && lane_id == 0) {
          mbarrier_arrive_expect_tx(mbar_a[stage], tx_a);
          mbarrier_arrive_expect_tx(mbar_b[stage], tx_b);
          tma_load_2d_cta(...);  // A
          tma_load_2d_cta(...);  // B
          // Optionally: TMA for SFA/SFB tiles
        }

    - All warps wait on:
        mbarrier_wait_parity(mbar_a[stage], phase);
        mbarrier_wait_parity(mbar_b[stage], phase);
        __syncthreads();

  PRIORITY 10: SCALE APPLICATION (SEMANTICS)
    - FP4 layout:
        - a_ref / b_ref are torch.float4_e2m1fn_x2 → 2 FP4s per byte
        - A K‑block of 16 FP4 values corresponds to 8 bytes.

    - FP8 scales:
        - sfa_ref_cpu / sfb_ref_cpu hold one FP8 scale per 16 FP4 values.
        - to_blocked() and torch._scaled_mm define how those are consumed by the MMA.

    - Kernel requirement:
        - For each (row m, K‑block b) in A and (col n, K‑block b) in B,
          the kernel must use the same FP8 scale value that the reference would apply.
        - You may:
            - TMA‑load K‑major scale tiles,
            - compute indices (m, b) → scale index in that 2D tile,
            - decode FP8 to float and apply to decoded FP4s before MMA.

        - Exact layout of internal SFA/SFB tiles is flexible as long as
          the resulting scaled FP16 fragments match the torch._scaled_mm result.

  PREVIOUS_MISTAKES
    - Old kernel issues:
        - Implemented GEMV (N=1) instead of full GEMM:
            • Only column 0 written.
        - Used SWIZZLE_NONE and BOX_K 16 bytes:
            • Many small TMA operations, bank conflicts.
        - Logical scale mapping did not match reference:
            • Scales applied using a layout that disagreed with to_blocked() semantics.
        - Mixed rank‑3 descriptors / clusters for L>1 (not needed; all L=1 here).

  NON‑GOALS
    - Do NOT:
        - Use CUTLASS or CuTe headers (kernel must be pure PTX + CUDA).
        - Use WGMMA (only mma.sync.aligned.m16n8k16) for now to fix correctness.
        - Use cluster scope or rank‑3 descriptors (CTA scope, rank‑2 only).
        - Add CUDA streams or host‑side caching beyond what submission.py already does.
        - Use cp.async for A/B/scales (use TMA only).
        - Change the semantic meaning of scales relative to sfa_ref_cpu/sfb_ref_cpu
          and to_blocked() (layout is flexible, semantics are not).
        - Use SWIZZLE_NONE or BOX_K < 128 bytes for TMA (use SWIZZLE_128B, 128‑byte boxes).

  TOOLS
    - PTX inline:
        - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        - ldmatrix.sync.aligned.m8n8.x{2,4}.shared.b16
    - TMA:
        - cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
        - mbarrier.init / arrive.expect_tx / wait.parity
    - Shared memory helpers:
        - cvta.to.shared.u32 for SMEM addresses
    - Numeric helpers:
        - FP4 decode LUT or simple mapping for e2m1
        - FP8 e4m3 decode (float8_e4m3fn → float)
        - Correct mapping from (row, K‑block) to scale index consistent with
          reference.to_blocked() and torch._scaled_mm.

</CRITICAL_INSTRUCTIONS>

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
