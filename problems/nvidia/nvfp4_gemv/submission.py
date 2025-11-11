import os

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

// SM100 FP4 Block-Scaled GEMV with TRUE Tensor Core Usage
// Strategy: Unpack FP4->FP16 and use native FP16 tensor core MMA

using namespace nvcuda;

// FP4 E2M1 lookup table - decode to FP16 directly
__device__ const half fp4_e2m1_to_fp16[16] = {
    __float2half_rn(0.0f),  __float2half_rn(0.5f),  __float2half_rn(1.0f),  __float2half_rn(1.5f),
    __float2half_rn(2.0f),  __float2half_rn(3.0f),  __float2half_rn(4.0f),  __float2half_rn(6.0f),
    __float2half_rn(-0.0f), __float2half_rn(-0.5f), __float2half_rn(-1.0f), __float2half_rn(-1.5f),
    __float2half_rn(-2.0f), __float2half_rn(-3.0f), __float2half_rn(-4.0f), __float2half_rn(-6.0f)
};

// Decode FP8 E4M3 scale factor to float
__device__ __forceinline__ float decode_fp8_e4m3(uint8_t val) {
    cutlass::float_e4m3_t fp8_val = *reinterpret_cast<cutlass::float_e4m3_t*>(&val);
    return __half2float(__float2half_rn(fp8_val));
}

// Optimized kernel using tensor cores via wmma (FP16 path)
// Each CTA processes 16 rows, full K dimension
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__(128)
fp4_gemv_wmma_kernel(
    const uint8_t* __restrict__ A_packed,     // [M, K/2]
    const uint8_t* __restrict__ B_packed,     // [128, K/2] (padded)
    const uint8_t* __restrict__ SFA_packed,   // [M, K/16]
    const uint8_t* __restrict__ SFB_packed,   // [128, K/16]
    half* __restrict__ D,                     // [M]
    const int M,
    const int K
) {
    // Each CTA handles 16 rows (WMMA_M)
    const int m_base = blockIdx.x * WMMA_M;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (m_base >= M) return;

    // Shared memory for unpacked FP16 data
    // A: [16, 32] per iteration, B: [32, 8] per iteration
    __shared__ half A_smem[WMMA_M][WMMA_K + 8];  // +8 for bank conflict avoidance
    __shared__ half B_smem[WMMA_K][WMMA_N + 8];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    const int K_packed = K / 2;
    const int K_scales = K / 16;

    // Process K dimension in WMMA_K chunks
    for (int k_base = 0; k_base < K; k_base += WMMA_K) {
        const int k_size = min(WMMA_K, K - k_base);
        const int k_packed_base = k_base / 2;
        const int k_scale_base = k_base / 16;

        __syncthreads();

        // Load and unpack A tile: [WMMA_M, WMMA_K] with scaling
        // Each thread loads multiple elements
        for (int idx = tid; idx < WMMA_M * (WMMA_K / 2); idx += blockDim.x) {
            const int row = idx / (WMMA_K / 2);
            const int col_packed = idx % (WMMA_K / 2);
            const int m_idx = m_base + row;

            if (m_idx < M && (k_packed_base + col_packed) < K_packed) {
                uint8_t packed = A_packed[m_idx * K_packed + k_packed_base + col_packed];

                // Get scale factor (one per 16 elements, so per 8 packed bytes)
                int scale_idx = col_packed / 8;
                float scale_a = 1.0f;
                if (k_scale_base + scale_idx < K_scales) {
                    scale_a = decode_fp8_e4m3(SFA_packed[m_idx * K_scales + k_scale_base + scale_idx]);
                }

                // Unpack both nibbles with scaling
                half val0 = __hmul(fp4_e2m1_to_fp16[packed & 0x0F], __float2half(scale_a));
                half val1 = __hmul(fp4_e2m1_to_fp16[(packed >> 4) & 0x0F], __float2half(scale_a));

                A_smem[row][col_packed * 2] = val0;
                A_smem[row][col_packed * 2 + 1] = val1;
            } else {
                A_smem[row][col_packed * 2] = __float2half(0.0f);
                A_smem[row][col_packed * 2 + 1] = __float2half(0.0f);
            }
        }

        // Load and unpack B tile: [WMMA_K, WMMA_N] with scaling
        // B is a vector (N=1) but replicated to N=8 for WMMA
        for (int idx = tid; idx < (WMMA_K / 2); idx += blockDim.x) {
            const int col_packed = idx;

            if ((k_packed_base + col_packed) < K_packed) {
                uint8_t packed = B_packed[0 * K_packed + k_packed_base + col_packed];

                int scale_idx = col_packed / 8;
                float scale_b = 1.0f;
                if (k_scale_base + scale_idx < K_scales) {
                    scale_b = decode_fp8_e4m3(SFB_packed[0 * K_scales + k_scale_base + scale_idx]);
                }

                half val0 = __hmul(fp4_e2m1_to_fp16[packed & 0x0F], __float2half(scale_b));
                half val1 = __hmul(fp4_e2m1_to_fp16[(packed >> 4) & 0x0F], __float2half(scale_b));

                // Broadcast to all N columns
                for (int n = 0; n < WMMA_N; n++) {
                    B_smem[col_packed * 2][n] = val0;
                    B_smem[col_packed * 2 + 1][n] = val1;
                }
            } else {
                for (int n = 0; n < WMMA_N; n++) {
                    B_smem[col_packed * 2][n] = __float2half(0.0f);
                    B_smem[col_packed * 2 + 1][n] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Load fragments and perform WMMA tensor core operation
        // Each warp does one WMMA operation
        if (warp_id == 0) {
            wmma::load_matrix_sync(a_frag, &A_smem[0][0], WMMA_K + 8);
            wmma::load_matrix_sync(b_frag, &B_smem[0][0], WMMA_N + 8);

            // TENSOR CORE MMA OPERATION!
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store results (only first column matters for GEMV)
    if (warp_id == 0) {
        __shared__ float out_smem[WMMA_M][WMMA_N];
        wmma::store_matrix_sync(&out_smem[0][0], acc_frag, WMMA_N, wmma::mem_row_major);

        __syncthreads();

        // Write first column to output
        if (lane_id < WMMA_M) {
            int m_idx = m_base + lane_id;
            if (m_idx < M) {
                D[m_idx] = __float2half(out_smem[lane_id][0]);
            }
        }
    }
}

// Launcher
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

    const int64_t K_packed = K / 2;
    const int64_t K_scales = K / 16;

    // WMMA configuration: 16x8x16 (M, N, K)
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 8;
    constexpr int WMMA_K = 16;

    const int num_blocks = (M + WMMA_M - 1) / WMMA_M;
    dim3 grid(num_blocks);
    dim3 block(128);  // 4 warps

    for (int64_t batch = 0; batch < L; batch++) {
        const uint8_t* A_batch = A_ptr + batch * M * K_packed;
        const uint8_t* B_batch = B_ptr + batch * 128 * K_packed;
        const uint8_t* SFA_batch = SFA_ptr + batch * M * K_scales;
        const uint8_t* SFB_batch = SFB_ptr + batch * 128 * K_scales;
        half* D_batch = D_ptr + batch * M;

        fp4_gemv_wmma_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
            A_batch, B_batch, SFA_batch, SFB_batch, D_batch, M, K
        );
    }

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
            name="nvfp4_gemv_wmma_tensorcore",
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
                "-Xcudafe", "--diag_suppress=20012",
                "-maxrregcount=64",  # Tight register budget for occupancy
                "-lineinfo",
                "-DNDEBUG",
                f"-I{cutlass_path}/include",
            ],
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    FP4 GEMV with REAL tensor core usage via WMMA.

    Strategy:
    1. Unpack FP4->FP16 with block scaling during load
    2. Use native FP16 tensor core MMA via wmma::mma_sync
    3. Target: <55 µs with high TC utilization
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

    # Launch tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()

    return c
