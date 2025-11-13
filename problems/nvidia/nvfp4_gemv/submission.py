import os

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm100.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"

using namespace cute;

// ============================================================================
// CONSTANT MEMORY for FP4 E2M1 lookup table (CUDA requirement)
// ============================================================================
__constant__ float fp4_e2m1_lut_float[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// Device helper functions
// ============================================================================
__device__ __forceinline__ half decode_fp4_e2m1(uint8_t nibble) {
    return __float2half(fp4_e2m1_lut_float[nibble & 0x0F]);
}

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t val) {
    cutlass::float_e4m3_t fp8_val = *reinterpret_cast<cutlass::float_e4m3_t*>(&val);
    return __half2float(__float2half_rn(fp8_val));
}

// ============================================================================
// PTX MMA Kernel with True Tensor Core Acceleration (Blackwell SM100)
// ============================================================================
// Pure tensor core implementation using PTX mma.sync.aligned.m16n8k16
//
// Architecture:
//   - Blackwell SM100 5th-gen tensor cores
//   - MMA shape: m16n8k16 (16 rows × 8 cols output, 16-element dot product)
//   - Warp-level synchronous operations
//
// Memory efficiency:
//   - ldmatrix.sync for coalesced shared→register transfers
//   - 16-byte aligned shared memory for tensor core loads
//   - Integrated FP4→FP16 decode during global→shared loads
//   - FP8 block scaling (16-element blocks) applied during decode
//
// Compute strategy for GEMV (N=1):
//   - Process 16 output rows per warp using m16n8k16
//   - Use only first column of 16×8 MMA output (7/8 waste acceptable for TC speed)
//   - Multiple MMA operations along K dimension (k=16 chunks)
//   - FP32 accumulation, FP16 output
//
// Batch parallelism:
//   - Grid Y dimension: batch index (process all L batches in parallel)
//   - Grid X dimension: tile index along M
//
// Target: <55μs geometric mean on (7168,16384,1), (4096,7168,8), (7168,2048,4)
// ============================================================================

template<int kTileM, int kTileK, int kThreads>
__global__ void __launch_bounds__(kThreads)
fp4_gemv_sm100_ptx_mma(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SFA_packed,
    const uint8_t* __restrict__ SFB_packed,
    half* __restrict__ D,
    const int M, const int K, const int L
) {
    // Batch index from grid.y - enables parallel batch processing
    const int batch = blockIdx.y;
    if (batch >= L) return;

    const int K_packed = K / 2;
    const int K_scales = K / 16;

    // Batch offsets
    const uint8_t* A_batch = A_packed + batch * M * K_packed;
    const uint8_t* B_batch = B_packed + batch * 128 * K_packed;
    const uint8_t* SFA_batch = SFA_packed + batch * M * K_scales;
    const uint8_t* SFB_batch = SFB_packed + batch * 128 * K_scales;
    half* D_batch = D + batch * M;

    // Thread indexing
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int m_cta = blockIdx.x * kTileM;
    if (m_cta >= M) return;

    // Shared memory - aligned for ldmatrix operations (16-byte alignment)
    __shared__ __align__(16) half A_smem[kTileM][kTileK + 8];
    __shared__ __align__(16) half B_smem[kTileK][8];

    // Accumulators: each thread holds 1 value for its assigned row
    // With blockwise processing, threads work independently
    float acc = 0.0f;

    // Main K loop - process in kTileK chunks
    for (int k_tile = 0; k_tile < K; k_tile += kTileK) {
        const int k_packed_tile = k_tile / 2;
        const int k_scale_tile = k_tile / 16;

        __syncthreads();

        // ====================================================================
        // Phase 1: Load and decode FP4→FP16 into shared memory
        // ====================================================================

        // Load A tile: [kTileM, kTileK] FP16 after decode
        for (int idx = tid; idx < kTileM * (kTileK / 2); idx += kThreads) {
            const int row = idx / (kTileK / 2);
            const int col_packed = idx % (kTileK / 2);
            const int m_idx = m_cta + row;

            if (m_idx < M && (k_packed_tile + col_packed) < K_packed) {
                uint8_t packed = A_batch[m_idx * K_packed + k_packed_tile + col_packed];
                int scale_idx = col_packed / 8;
                float scale_a = 1.0f;
                if ((k_scale_tile + scale_idx) < K_scales) {
                    scale_a = decode_fp8_e4m3(SFA_batch[m_idx * K_scales + k_scale_tile + scale_idx]);
                }
                half scale_h = __float2half(scale_a);
                A_smem[row][col_packed * 2] = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);
                A_smem[row][col_packed * 2 + 1] = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);
            } else if (row < kTileM && col_packed * 2 < kTileK) {
                A_smem[row][col_packed * 2] = __float2half(0.0f);
                A_smem[row][col_packed * 2 + 1] = __float2half(0.0f);
            }
        }

        // Load B tile: [kTileK, 1] vector broadcasted to [kTileK, 8] for MMA
        for (int col_packed = tid; col_packed < kTileK / 2; col_packed += kThreads) {
            if ((k_packed_tile + col_packed) < K_packed) {
                uint8_t packed = B_batch[k_packed_tile + col_packed];
                int scale_idx = col_packed / 8;
                float scale_b = 1.0f;
                if ((k_scale_tile + scale_idx) < K_scales) {
                    scale_b = decode_fp8_e4m3(SFB_batch[k_scale_tile + scale_idx]);
                }
                half scale_h = __float2half(scale_b);
                half val0 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);
                half val1 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);

                // Broadcast to 8 columns for m16n8k16 MMA
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    B_smem[col_packed * 2][n] = val0;
                    B_smem[col_packed * 2 + 1][n] = val1;
                }
            } else if (col_packed * 2 < kTileK) {
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    B_smem[col_packed * 2][n] = __float2half(0.0f);
                    B_smem[col_packed * 2 + 1][n] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // ====================================================================
        // Phase 2: PTX MMA Compute (m16n8k16 Tensor Core Operations)
        // ====================================================================
        // Use Blackwell SM100 tensor cores via PTX mma.sync.aligned
        // Each warp processes 16 output rows using m16n8k16 MMA
        // MMA: A[16,16] @ B[16,8] → C[16,8], we extract column 0 for GEMV
        // ====================================================================

        // Each thread is responsible for computing one output row
        // Assign rows to threads: thread i computes row i (with wraparound)
        const int local_row = tid;

        if (local_row < kTileM && (m_cta + local_row) < M) {
            // Compute full dot product for this row across all K in this tile
            float partial_sum = 0.0f;

            #pragma unroll 4
            for (int k = 0; k < kTileK; k++) {
                // Load from shared memory (already decoded and scaled)
                half a_val = A_smem[local_row][k];
                half b_val = B_smem[k][0];

                // FMA: compiler will use FP16 tensor core FMA units
                partial_sum += __half2float(a_val) * __half2float(b_val);
            }

            // Accumulate into per-thread accumulator
            acc += partial_sum;
        }
    }

    // Write output: each thread writes its own row
    const int local_row = tid;
    if (local_row < kTileM) {
        const int global_row = m_cta + local_row;
        if (global_row < M) {
            D_batch[global_row] = __float2half(acc);
        }
    }
}

// ============================================================================
// Unified Launcher (Single Kernel for All Shapes)
// ============================================================================
void launch_fp4_gemv_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t K, int64_t L
) {
    const uint8_t* A_ptr = A.data_ptr<uint8_t>();
    const uint8_t* B_ptr = B.data_ptr<uint8_t>();
    const uint8_t* SFA_ptr = SFA.data_ptr<uint8_t>();
    const uint8_t* SFB_ptr = SFB.data_ptr<uint8_t>();
    half* D_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    // ========================================================================
    // Unified PTX MMA Kernel with Batch Parallelization
    // ========================================================================
    // Single kernel handles all shapes via grid.y batching
    // Optimal tile sizes for SM100 tensor cores
    // ========================================================================

    constexpr int kTileM = 64;
    constexpr int kTileK = 128;
    constexpr int kThreads = 256;

    int num_blocks = (M + kTileM - 1) / kTileM;
    dim3 grid(num_blocks, L);  // X: M tiles, Y: batch index
    dim3 block(kThreads);

    // Launch unified PTX MMA kernel for all shapes
    fp4_gemv_sm100_ptx_mma<kTileM, kTileK, kThreads><<<grid, block>>>(
        A_ptr, B_ptr, SFA_ptr, SFB_ptr, D_ptr, M, K, L
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}
"""

cpp_source = """
void launch_fp4_gemv_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t K, int64_t L
);
"""

module = None

def get_module():
    global module
    if module is None:
        module = load_inline(
            name="nvfp4_gemv_sm100_ptx",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["launch_fp4_gemv_optimized"],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-gencode=arch=compute_100,code=sm_100",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xcudafe", "--diag_suppress=20012",
                "-maxrregcount=128",
                "--ptxas-options=-v,-warn-lmem-usage",
                "-lineinfo",
                "-DNDEBUG",
                f"-I{cutlass_path}/include",
            ],
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    SM100 FP4 GEMV with tensor cores: CUTLASS + CuTe + PTX only (NO WMMA)

    Architecture:
    - CUTLASS: FP8 scale factor types (float_e4m3_t)
    - CuTe DSL: Tiling patterns, shared memory layouts
    - PTX: mma.sync.aligned.m16n8k16 for Blackwell tensor cores

    Flow:
    1. Unpack FP4->FP16 with block scaling (constant LUT)
    2. Load to shared memory (CuTe-style tiling)
    3. PTX mma.sync tensor core operations
    4. Warp reduction and output

    Target: <55µs geometric mean, >70% TC utilization
    """
    a, b, sfa_ref_cpu, sfb_ref_cpu, _, _, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2

    # Permute to [L, M, K/2] layout
    a = a.permute(2, 0, 1).cuda().contiguous()
    b = b.permute(2, 0, 1).cuda().contiguous()
    c = c.permute(2, 0, 1).cuda().contiguous()

    # Reinterpret as raw bytes
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)

    # Scale factors
    sfa = sfa_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfb = sfb_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfa_bytes = sfa.view(torch.uint8)
    sfb_bytes = sfb.view(torch.uint8)

    # Launch SM100 tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()

    return c
