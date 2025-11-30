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

<CRITICAL_INSTRUCTIONS>

  CRITICAL_CONSTRAINT
    <CRITICAL>
      Target: FP4 block-scaled GEMM (M × N, not M × 1) on NVIDIA B200 SM_100a
      Goal: ~3μs geometric mean across benchmark shapes
      Speed of light analysis targets (1.5GHz clock):
        - Shape 1: 8.994 μs  (M=128, N=7168, K=16384)
        - Shape 2: 2.354 μs  (M=128, N=4096, K=7168)
        - Shape 3: 1.333 μs  (M=128, N=7168, K=2048)
      Architecture: Pure PTX inline assembly + TMA + CTA scope (no CUTLASS/CuTe headers)
      
      Benchmark shapes (all L=1):
      - M=128, N=7168, K=16384
      - M=128, N=4096, K=7168
      - M=128, N=7168, K=2048

      Scale factor layouts:
        - Reference provides TWO scale formats per input
        - Simple: [M/N, K/16, L] for reference kernel
        - Permuted: [32, 4, rest_m/n, 4, rest_k, L] for YOUR kernel
        - Use sfa_ref_permuted and sfb_ref_permuted (indices 4,5 in input tuple)
        - Permuted layout matches CuTe/CUTLASS atom tiling (32, 4) blocks
        
      Input tuple structure (7 elements):
        0: a_ref [M, K, L] - FP4 matrix A (nvfp4, K-major)
        1: b_ref [N, K, L] - FP4 matrix B (nvfp4, K-major)
        2: sfa_ref_cpu [M, K/16, L] - Simple scales (for reference, DO NOT USE)
        3: sfb_ref_cpu [N, K/16, L] - Simple scales (for reference, DO NOT USE)
        4: sfa_ref_permuted [32,4,rest_m,4,rest_k,L] - Use THIS for A scales
        5: sfb_ref_permuted [32,4,rest_n,4,rest_k,L] - Use THIS for B scales
        6: c_ref [M, N, L] - Output (fp16)

      CRITICAL: Must match torch._scaled_mm reference behavior:
        - B matrix is TRANSPOSED in reference: b.transpose(0, 1)
        - Your kernel must account for this transposition
        - Scale factors go through to_blocked() permutation
        - Output tolerance: rtol=1e-03, atol=1e-03

    </CRITICAL>

  CURRENT_STATE
    <OBSERVATION>
      Problem: GEMM (matrix-matrix multiply), not GEMV
      B is N × K full matrix, not 1 × K vector
      Output C is M × N matrix, not M × 1 vector
      Must tile over both M and N dimensions
      All benchmark shapes have L=1 (no batching)
    </OBSERVATION>

  IMPLEMENTATION_PLAN
    <PLAN>
      PRIORITY 1: TILE CONFIGURATION
        TILE_M = 128              // Rows of A per CTA
        TILE_N = 128              // Columns of B per CTA
        TILE_K = 256              // Elements (128 bytes packed FP4)
        THREADS_PER_BLOCK = 256   // 8 warps × 32 threads
        NUM_STAGES = 3            // Triple-buffering

      PRIORITY 2: TMA DESCRIPTOR SETUP (Rank-2 only, L=1)
        A matrix (M × K):
          rank = 2
          dims = [K_packed, M]           // [K/2, M] in bytes
          box = [128, 128]               // [bytes_K, rows_M]
          strides = [K_packed]           // Row stride in bytes
          swizzle = CU_TENSOR_MAP_SWIZZLE_128B
          
        B matrix (N × K):
          rank = 2
          dims = [K_packed, N]           // [K/2, N] in bytes
          box = [128, 128]               // [bytes_K, rows_N]
          strides = [K_packed]           // Row stride in bytes
          swizzle = CU_TENSOR_MAP_SWIZZLE_128B
          
        SFA scales (M × K/16) - use permuted format:
          Decode permuted layout [32,4,rest_m,4,rest_k,L]
          Access pattern matches (32, 4) atom tiling
          
        SFB scales (N × K/16) - use permuted format:
          Decode permuted layout [32,4,rest_n,4,rest_k,L]
          Access pattern matches (32, 4) atom tiling

      PRIORITY 3: MEMORY ALIGNMENT
        Global memory base pointers: 128-byte aligned
        Shared memory TMA destinations: 1024-byte aligned (required for SWIZZLE_128B)
        Shared memory non-TMA regions: 128-byte aligned
        Leading dimensions: multiples of 128 elements
        K dimension: divisible by 256 (per task.yml constraint)

      PRIORITY 4: BARRIER CONFIGURATION
        Per-stage barriers (3 stages):
          __shared__ uint64_t mbar_a[3];     // A matrix TMA tracking
          __shared__ uint64_t mbar_b[3];     // B matrix TMA tracking
        
        Transaction bytes per stage:
          tx_a = TILE_M × (TILE_K/2) + TILE_M × (TILE_K/16)
               = 128 × 128 + 128 × 16 = 18,432 bytes
          tx_b = TILE_N × (TILE_K/2) + TILE_N × (TILE_K/16)
               = 128 × 128 + 128 × 16 = 18,432 bytes

      PRIORITY 5: SHARED MEMORY LAYOUT
        Triple-buffered TMA destinations (1024-byte aligned):
          __shared__ uint8_t a_packed[3][TILE_M × TILE_K/2];
          __shared__ uint8_t b_packed[3][TILE_N × TILE_K/2];
          __shared__ uint8_t sfa_permuted[3][...];  // Permuted scale layout
          __shared__ uint8_t sfb_permuted[3][...];  // Permuted scale layout
        
        Decoded tiles (128-byte aligned):
          __shared__ half a_fp16[TILE_M × TILE_K/8];
          __shared__ half b_fp16[TILE_N × TILE_K/8];

      PRIORITY 6: KERNEL LAUNCH CONFIGURATION
        Grid dimensions:
          grid.x = ceil(M / TILE_M)          // e.g., ceil(128/128) = 1
          grid.y = ceil(N / TILE_N)          // e.g., ceil(7168/128) = 56
          grid.z = L                         // Always 1
        
        Block dimensions:
          block.x = 256                      // 8 warps
        
        Launch: CTA scope only (cudaLaunchKernel, NO cluster)

      PRIORITY 7: MMA LOOP STRUCTURE
        Nested loops:
          1. M-tile: int mtile = blockIdx.x * TILE_M
          2. N-tile: int ntile = blockIdx.y * TILE_N
          3. K-tile loop: for (int ktile = 0; ktile < K; ktile += TILE_K)
             - Prefetch stage (k+2) via TMA
             - Wait on barrier for stage k
             - Decode FP4→FP16 + apply permuted scales
             - MMA: 16×8×16 per warp using mma.sync.aligned
          4. Writeback: Store full M × N tile to global memory

      PRIORITY 8: PTX MMA INSTRUCTION
        Use mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        Each warp processes 16 rows × 8 cols
        Accumulate across 16 N-sub-tiles per warp (cover TILE_N=128)
        Write ALL 8 columns per thread (not just col==0)

      PRIORITY 9: TMA PREFETCH PATTERN
        Stage pipeline (3 stages):
          Stage 0: TMA load A[k+2], B[k+2]
          Stage 1: Decode A[k+1], B[k+1] (FP4→FP16 + permuted scale)
          Stage 2: MMA compute A[k] × B[k]
        
        Only warp 0 lane 0 issues TMA:
          if (warpid == 0 && laneid == 0) {
              mbarrier_arrive_expect_tx(mbar_a[stage], tx_a);
              mbarrier_arrive_expect_tx(mbar_b[stage], tx_b);
              tma_load_2d_cta(...);  // Rank-2 only
          }
        
        All threads wait:
          mbarrier_wait_parity(mbar_a[stage], phase);
          mbarrier_wait_parity(mbar_b[stage], phase);
          __syncthreads();

      PRIORITY 10: OUTPUT WRITEBACK
        Each warp computes 16 rows × (16 × 8) cols of C
        Fragment layout: 4 floats per thread cover 16×8 output
        Write ALL columns (full GEMM output, not just col 0)
        Account for B transposition in reference
        Global store: C[m_global][n_global] = __float2half(cfrag[...])
    </PLAN>

  PREVIOUS_MISTAKES
    <PREVIOUS_VERSION>
      Previous implementation: GEMV (M × 1 output)
      Architecture: PTX inline + TMA + CTA scope + SWIZZLE_NONE + BOX_K 16 bytes
      Issues:
        - B was 1 × K vector (broadcast)
        - SWIZZLE_NONE left bank conflicts
        - BOX_K 16 bytes caused many TMA operations
        - Only processed column 0 of MMA output
        - Single N=1 output dimension
        - Had RANK-3 cluster logic for L=4, L=8 (unused for this task)
        - Used simple scale format instead of permuted
    </PREVIOUS_VERSION>

  NON_GOALS
    <NON_GOALS>
      - Do NOT use CUTLASS or CuTe headers (pure PTX only)
      - Do NOT use WGMMA (stick with mma.sync.aligned)
      - Do NOT use cluster scope (CTA scope only)
      - Do NOT use Rank-3 descriptors (Rank-2 only for L=1)
      - Do NOT add CUDA streams or caching
      - Do NOT use cp.async (use TMA only)
      - Do NOT use simple scale layout (use permuted)
      - Do NOT use SWIZZLE_NONE (use SWIZZLE_128B)
      - Do NOT use BOX_K 16 bytes (use 128 bytes)
    </NON_GOALS>

  TOOLS
    <TOOLS>
      - PTX inline assembly for MMA (mma.sync.aligned.m16n8k16...)
      - TMA bulk copy (cp.async.bulk.tensor.2d.shared::cta.global.mbarrier...)
      - Barriers (mbarrier.init, mbarrier.arrive.expect_tx, mbarrier.wait.parity)
      - Shared memory pointer conversion (cvta.to.shared.u32)
      - ldmatrix.sync.aligned for loading decoded tiles
      - FP4 decode LUT + FP8 scale decode
      - Manual permuted scale indexing (32, 4) atom pattern
    </TOOLS>

</CRITICAL_INSTRUCTIONS>

<COMPLIANCE_CHECK>
Your kernel implementation MUST be compliant with these files:

1. reference.py:
   - Match torch._scaled_mm behavior exactly
   - Handle B matrix transposition: b.transpose(0, 1)
   - Use permuted scale formats (sfa_ref_permuted, sfb_ref_permuted)
   - Decode permuted layout: [32, 4, rest_m/n, 4, rest_k, L]
   - Apply to_blocked() scale transformation logic
   - Output tolerance: rtol=1e-03, atol=1e-03

2. task.yml:
   - Input tuple has 7 elements (not 5)
   - M divisible by mma_tiler_mn[0]
   - N divisible by mma_tiler_mn[1]
   - K divisible by 256
   - All benchmark shapes have L=1
   - Ranking by geometric mean
   - Speed of light targets: 8.994μs, 2.354μs, 1.333μs

3. utils.py:
   - Pass verbose_allclose validation
   - Handle rtol=1e-03, atol=1e-03
   - No NaN, Inf, or size mismatches
   - Return empty list [] for success

4. eval.py:
   - No CUDA streams (default stream only)
   - No cross-run caching
   - Synchronous execution
   - Pass both correctness and benchmark tests

VERIFICATION:
- Run all 10 test shapes successfully
- Pass 3 benchmark shapes with correct output
- Achieve geom_mean close to 3μs target
- Zero tolerance for stream usage
</COMPLIANCE_CHECK>
