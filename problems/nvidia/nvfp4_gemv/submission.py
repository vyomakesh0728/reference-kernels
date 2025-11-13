import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# ============================================================================
# CUTLASS GemvBlockScaled with FP16 Output (Based on Example 91)
# Single-kernel approach: FP4 decode + scale + GEMV using tensor cores
# ============================================================================
cuda_source = r"""
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemv_block_scaled.h>
#include <cutlass/gemm/kernel/gemv_block_scaled.h>
#include <cutlass/epilogue/threadblock/gemv_epilogue_with_scaling_factor.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

// FP4 and FP8 data types
using ElementA = cutlass::float_e2m1_t;  // FP4 E2M1 for input A
using ElementB = cutlass::float_e2m1_t;  // FP4 E2M1 for input B
using ElementSFA = cutlass::float_e4m3_t;  // FP8 E4M3 scale factors for A
using ElementSFB = cutlass::float_e4m3_t;  // FP8 E4M3 scale factors for B
using ElementD = cutlass::half_t;  // FP16 output (CHANGED from FP4)
using ElementC = cutlass::half_t;  // FP16 accumulator
using ElementAccumulator = float;  // Float accumulation for precision
using ElementCompute = float;  // Float for epilogue computations

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using LayoutSFA = cutlass::layout::ColumnMajor;
using LayoutSFB = cutlass::layout::ColumnMajor;

// SM100 configuration
static constexpr int kVectorSize = 16;  // Block size for scale factors
static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;

// Thread shape for epilogue
using ThreadShape = cutlass::gemm::GemmShape<16, 8, 1>;

// Epilogue configuration - modified for FP16 output
using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,  // FP16 output
    ElementSFA,  // FP8 scale factors
    LayoutD,
    LayoutSFA
>;

// GemvBlockScaled kernel configuration
using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA,
    LayoutA,
    ElementB,
    ElementD,  // FP16 output
    ElementAccumulator,
    EpilogueOp,
    kElementsPerAccess
>;

// Device-level GEMV operator
using Gemv = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

// Launch CUTLASS FP4 GEMV with FP16 output
void launch_fp4_gemv_optimized(
    torch::Tensor A_fp4,      // [L, M, K/2] packed FP4
    torch::Tensor B_fp4,      // [L, 1, K/2] packed FP4
    torch::Tensor SFA,        // [L, M, K//16] FP8 scale factors
    torch::Tensor SFB,        // [L, 1, K//16] FP8 scale factors
    torch::Tensor D,          // [L, M, 1] FP16 output
    int M, int K, int L
) {
    // Get raw pointers
    uint8_t* A_ptr = reinterpret_cast<uint8_t*>(A_fp4.data_ptr());
    uint8_t* B_ptr = reinterpret_cast<uint8_t*>(B_fp4.data_ptr());
    uint8_t* SFA_ptr = reinterpret_cast<uint8_t*>(SFA.data_ptr());
    uint8_t* SFB_ptr = reinterpret_cast<uint8_t*>(SFB.data_ptr());
    cutlass::half_t* D_ptr = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

    // Process each batch
    for (int batch = 0; batch < L; ++batch) {
        // Compute offsets for this batch
        int64_t a_offset = batch * M * (K / 2);  // Packed FP4: K/2 bytes per row
        int64_t b_offset = batch * 1 * (K / 2);
        int64_t sfa_offset = batch * M * (K / 16);
        int64_t sfb_offset = batch * 1 * (K / 16);
        int64_t d_offset = batch * M * 1;

        // CUTLASS GemvBlockScaled arguments
        typename Gemv::Arguments arguments{
            {M, 1, K},  // Problem size (M x 1 GEMV, K reduction dimension)
            reinterpret_cast<ElementA*>(A_ptr + a_offset),  // ptr_A
            K,  // lda (stride for A)
            reinterpret_cast<ElementB*>(B_ptr + b_offset),  // ptr_B
            K,  // ldb (stride for B)
            reinterpret_cast<ElementSFA*>(SFA_ptr + sfa_offset),  // ptr_SFA
            K / kVectorSize,  // ldSFA
            reinterpret_cast<ElementSFB*>(SFB_ptr + sfb_offset),  // ptr_SFB
            K / kVectorSize,  // ldSFB
            D_ptr + d_offset,  // ptr_C (unused)
            1,  // ldc
            D_ptr + d_offset,  // ptr_D (output)
            1,  // ldd
            {1.0f, 0.0f},  // {alpha, beta} for linear combination
            1  // batch_count (processing one at a time)
        };

        // Initialize CUTLASS GEMV operator
        Gemv gemv_op;
        cutlass::Status status = gemv_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GemvBlockScaled cannot implement this problem");
        }

        status = gemv_op.initialize(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GemvBlockScaled initialization failed");
        }

        // Execute kernel
        status = gemv_op();
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GemvBlockScaled execution failed");
        }
    }

    // Synchronize
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}
"""

# Compile the inline CUDA extension
def get_module():
    return load_inline(
        name="fp4_gemv_cutlass_blockscaled",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["launch_fp4_gemv_optimized"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-I{cutlass_path}/include",
            f"-I{cutlass_path}/tools/util/include",
            "-gencode=arch=compute_100,code=sm_100",  # SM100 Blackwell
            "-maxrregcount=128",
            "-DNDEBUG",
        ],
        with_cuda=True,
        verbose=True,
    )


def custom_kernel(data: input_t) -> output_t:
    """
    FP4 GEMV using CUTLASS GemvBlockScaled with FP16 output.

    Based on CUTLASS Example 91 but modified for FP16 output instead of FP4.
    All decode, scaling, and GEMV occur in single tensor-core blockwise kernel.
    """
    # Unpack input tuple: (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)
    a, b, sfa_ref_cpu, sfb_ref_cpu, _, _, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2  # Packed FP4: K/2 elements per dimension

    # Permute to [L, M, K/2] layout for batch-parallel processing
    a = a.permute(2, 0, 1).cuda().contiguous()
    b = b.permute(2, 0, 1).cuda().contiguous()
    c = c.permute(2, 0, 1).cuda().contiguous()  # [L, M, 1]

    # Reinterpret as uint8 (packed FP4 format)
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)

    # Scale factors: [M, K//16, L] → [L, M, K//16]
    sfa = sfa_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfb = sfb_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfa_bytes = sfa.view(torch.uint8)
    sfb_bytes = sfb.view(torch.uint8)

    # Launch CUTLASS GemvBlockScaled kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back to [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()

    return c
