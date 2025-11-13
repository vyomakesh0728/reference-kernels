import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# ============================================================================
# Two-Stage Pipeline: FP4→FP16 Decode + CUTLASS FP16 Blockwise GEMV
# Stage 1: Decode FP4+FP8 scale factors to FP16 (custom kernel)
# Stage 2: FP16 GEMV using CUTLASS blockwise GEMM (tensor cores)
# ============================================================================
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ============================================================================
// Stage 1: FP4+FP8 → FP16 Decode Kernel
// ============================================================================

__constant__ float fp4_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ half decode_fp4_e2m1(uint8_t nibble) {
    return __float2half(fp4_e2m1_lut[nibble & 0x0F]);
}

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t val) {
    cutlass::float_e4m3_t fp8_val;
    reinterpret_cast<uint8_t&>(fp8_val) = val;
    return static_cast<float>(fp8_val);
}

// Decode FP4 matrix with FP8 scale factors to FP16
// Layout: A is [L, M, K/2] packed, SFA is [L, M, K/16]
__global__ void decode_fp4_to_fp16(
    const uint8_t* __restrict__ fp4_packed,
    const uint8_t* __restrict__ scale_factors,
    half* __restrict__ output_fp16,
    int M, int K, int L, bool transpose_output
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.z;

    if (batch >= L || row >= M || col >= K) return;

    const int K_packed = K / 2;
    const int K_scales = K / 16;

    // Input layout: [L, M, K/2] for packed FP4
    const int packed_col = col / 2;
    const int nibble_idx = col % 2;
    const int sf_idx = col / 16;

    const int input_offset = batch * M * K_packed + row * K_packed + packed_col;
    const uint8_t packed_byte = fp4_packed[input_offset];

    uint8_t nibble = (nibble_idx == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
    half decoded = decode_fp4_e2m1(nibble);

    // Apply scale factor
    const int sf_offset = batch * M * K_scales + row * K_scales + sf_idx;
    float scale = decode_fp8_e4m3(scale_factors[sf_offset]);
    half result = __hmul(decoded, __float2half(scale));

    // Output layout: row-major [L, M, K] or transposed [L, K, M] for GEMM
    if (transpose_output) {
        // For B matrix: transpose to [L, K, 1] (column-major for GEMM)
        const int output_offset = batch * K * 1 + col * 1 + 0;  // Treat as Kx1
        output_fp16[output_offset] = result;
    } else {
        // For A matrix: keep row-major [L, M, K]
        const int output_offset = batch * M * K + row * K + col;
        output_fp16[output_offset] = result;
    }
}

// ============================================================================
// Stage 2: CUTLASS FP16 Blockwise GEMV (A @ B = C, where B is Kx1 vector)
// ============================================================================

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // Kx1 vector as column
using LayoutC = cutlass::layout::RowMajor;

// SM100 Blockwise GEMM configuration
using ThreadblockShape = cutlass::gemm::GemmShape<128, 8, 64>;  // M, N, K per threadblock
using WarpShape = cutlass::gemm::GemmShape<64, 8, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // SM100 MMA

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3  // Stages
>;

// ============================================================================
// Launcher
// ============================================================================
void launch_fp4_gemv_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t K, int64_t L
) {
    const uint8_t* A_fp4 = A.data_ptr<uint8_t>();
    const uint8_t* B_fp4 = B.data_ptr<uint8_t>();
    const uint8_t* SFA_ptr = SFA.data_ptr<uint8_t>();
    const uint8_t* SFB_ptr = SFB.data_ptr<uint8_t>();
    half* D_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    // Allocate temp FP16 buffers for decoded A and B
    half* A_fp16 = nullptr;
    half* B_fp16 = nullptr;
    cudaMalloc(&A_fp16, L * M * K * sizeof(half));
    cudaMalloc(&B_fp16, L * K * 1 * sizeof(half));

    // Stage 1: Decode FP4+FP8 → FP16
    dim3 block_decode(16, 16);
    dim3 grid_A((K + 15) / 16, (M + 15) / 16, L);
    decode_fp4_to_fp16<<<grid_A, block_decode>>>(
        A_fp4, SFA_ptr, A_fp16, M, K, L, false
    );

    // Decode B (1 x K x L) - treat first dimension as 1
    dim3 grid_B((K + 15) / 16, 1, L);
    decode_fp4_to_fp16<<<grid_B, block_decode>>>(
        B_fp4, SFB_ptr, B_fp16, 1, K, L, true
    );

    cudaDeviceSynchronize();

    // Stage 2: CUTLASS FP16 GEMV - Process all batches
    // Note: For optimal performance on L>1, consider using batch-parallel grid
    Gemm gemm_op;
    ElementAccumulator alpha = 1.0f;
    ElementAccumulator beta = 0.0f;

    // Process batches sequentially (simple but correct approach)
    // TODO: Optimize with strided batched GEMM or grid-level batch parallelism
    for (int batch = 0; batch < L; ++batch) {
        typename Gemm::Arguments arguments{
            {static_cast<int>(M), 1, static_cast<int>(K)},  // M x 1 = (M x K) @ (K x 1)
            {A_fp16 + batch * M * K, static_cast<int>(K)},
            {B_fp16 + batch * K, static_cast<int>(K)},  // ld=K for column-major K×1
            {D_ptr + batch * M, 1},
            {D_ptr + batch * M, 1},
            {alpha, beta}
        };

        cutlass::Status status = gemm_op.initialize(arguments);
        if (status != cutlass::Status::kSuccess) {
            cudaFree(A_fp16);
            cudaFree(B_fp16);
            throw std::runtime_error("CUTLASS GEMM initialization failed");
        }

        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            cudaFree(A_fp16);
            cudaFree(B_fp16);
            throw std::runtime_error("CUTLASS GEMM execution failed");
        }
    }

    cudaFree(A_fp16);
    cudaFree(B_fp16);

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
            name="nvfp4_gemv_sm100",
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
                "-Xcudafe",
                "--diag_suppress=20012",
                "-maxrregcount=128",
                "--ptxas-options=-v,-warn-lmem-usage",
                "-lineinfo",
                "-DNDEBUG",
                f"-I{cutlass_path}/include",
                f"-I{cutlass_path}/tools/util/include",
            ],
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    Two-stage FP4 GEMV: Decode FP4→FP16 + CUTLASS FP16 blockwise GEMM

    Stage 1: Custom kernel decodes FP4+FP8 to FP16
    Stage 2: CUTLASS blockwise GEMM on FP16 inputs (tensor cores)
    """
    # Use reference scale factors (simple M×(K//16)×L layout)
    a, b, sfa_ref_cpu, sfb_ref_cpu, _, _, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2  # Correct K dimension from packed FP4

    # Move to GPU and permute to [L, M, K/2] for batch-parallel processing
    a = a.permute(2, 0, 1).cuda().contiguous()
    b = b.permute(2, 0, 1).cuda().contiguous()
    c = c.permute(2, 0, 1).cuda().contiguous()  # [L, M, 1]

    # Reinterpret as raw bytes (packed format)
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)

    # Scale factors in reference format: [M, K//16, L] → [L, M, K//16]
    sfa = sfa_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfb = sfb_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfa_bytes = sfa.view(torch.uint8)
    sfb_bytes = sfb.view(torch.uint8)

    # Launch two-stage pipeline
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back to [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()

    return c
