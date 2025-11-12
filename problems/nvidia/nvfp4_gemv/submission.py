import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# ============================================================================
# CUTLASS GEMV BlockScaled API - Direct FP16 Output (No Decode Stage)
# Uses CUTLASS blockwise GEMM with FP4 inputs, FP16 output directly
# ============================================================================
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/tensor_ref.h"

// ============================================================================
// Kernel Configuration (Back to scaling factor epilogue, but we'll decode on GPU)
// ============================================================================

// Element types: FP4 (float_e2m1_t) with FP8 scale factors (float_e4m3_t)
using ElementA = cutlass::float_e2m1_t;
using ElementSFA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;

using ElementB = cutlass::float_e2m1_t;
using ElementSFB = cutlass::float_e4m3_t;

// GEMV produces FP16 output directly (no intermediate FP4 stage)
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;  // FP16 final output
using LayoutD = cutlass::layout::ColumnMajor;

// No scale factors needed for output (applied during mainloop)
using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

// Vector size for block scaling (must match scale factor block size)
static constexpr int kVectorSize = 16;
static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;

// Thread shape for GEMV (MUST be 16x8)
using ThreadShape = cutlass::gemm::GemmShape<16, 8>;
static_assert(kVectorSize == ThreadShape::kM, "vector size mismatch");

// Simple epilogue that outputs FP16 directly (no intermediate FP4 quantization)
// Uses standard linear combination: D = alpha * accumulator + beta * C
using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueLinearCombination<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,
    LayoutD
>;

// Main GEMV kernel using CUTLASS BlockScaled API
using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA,
    LayoutA,
    ElementB,
    ElementC,
    ElementAccumulatorMainloop,
    EpilogueOp,
    kElementsPerAccess
>;

using Gemv = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

// ============================================================================
// Launcher (Direct FP16 output from GEMV - no decode stage needed)
// ============================================================================
void launch_fp4_gemv_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,  // FP16 output
    int64_t M, int64_t K, int64_t L
) {
    // Get raw pointers
    auto A_ptr = reinterpret_cast<ElementA*>(A.data_ptr<uint8_t>());
    auto B_ptr = reinterpret_cast<ElementB*>(B.data_ptr<uint8_t>());
    auto SFA_ptr = reinterpret_cast<ElementSFA*>(SFA.data_ptr<uint8_t>());
    auto SFB_ptr = reinterpret_cast<ElementSFB*>(SFB.data_ptr<uint8_t>());
    auto D_fp16_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    const int gemm_m = static_cast<int>(M);
    const int gemm_k = static_cast<int>(K);
    const int gemm_batch = static_cast<int>(L);

    // Calculate batch strides
    const int k_blks = (gemm_k + kVectorSize - 1) / kVectorSize;  // ceil(K/16)
    const int m_blks = (gemm_m + 127) / 128;                       // ceil(M/128)

    // For permuted scale factors with shape [L, 32, 4, ceil(M/128), 4, ceil(K/64)]
    const int k_blks_sf = (gemm_k + 63) / 64;  // ceil(K/64) for atom_k=4 blocking
    const int64_t sf_elements_per_batch = 32 * 4 * m_blks * 4 * k_blks_sf;

    const int64_t batch_stride_A = gemm_m * (gemm_k / 2);
    const int64_t batch_stride_B = gemm_k / 2;
    const int64_t batch_stride_C = 0;
    const int64_t batch_stride_D = gemm_m;  // FP16 output (not packed)
    const int64_t batch_stride_SFA = sf_elements_per_batch;
    const int64_t batch_stride_SFB = 32 * 4 * 1 * 4 * k_blks_sf;  // For B: M=1 (padded to 128)

    // Construct TensorRefs
    cutlass::TensorRef<ElementA, LayoutA> ref_A(
        A_ptr,
        LayoutA::packed({gemm_m, gemm_k / 2})
    );

    cutlass::TensorRef<ElementD, LayoutD> ref_D(
        reinterpret_cast<ElementD*>(D_fp16_ptr),
        LayoutD::packed({gemm_m, 1})
    );

    ElementCompute alpha = ElementCompute(1.0f);
    ElementCompute beta = ElementCompute(0.0f);

    // Setup GEMV arguments
    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord{gemm_m, gemm_k},
        gemm_batch,
        typename EpilogueOp::Params{
            ref_D,
            alpha,
            beta
        },
        ref_A,
        B_ptr,
        nullptr,  // ptr_C
        reinterpret_cast<ElementD*>(D_fp16_ptr),
        SFA_ptr,
        SFB_ptr,
        gemm_k / 2,
        batch_stride_A,
        batch_stride_B,
        batch_stride_C,
        batch_stride_D,
        batch_stride_SFA,
        batch_stride_SFB
    };

    // Execute GEMV (produces FP16 output directly)
    Gemv gemv_op;
    cutlass::Status status = gemv_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMV initialization failed");
    }

    status = gemv_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMV execution failed");
    }

    // Synchronize
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
    SM100 FP4 GEMV using CUTLASS BlockScaled API with FP16 output
    """
    # CRITICAL: Use CUTLASS-permuted scale factors (indices 4,5), NOT reference format (2,3)
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2  # Correct K dimension from packed FP4

    # Move to GPU and permute to [L, M, K/2] for batch-parallel processing
    a = a.permute(2, 0, 1).cuda().contiguous()
    b = b.permute(2, 0, 1).cuda().contiguous()
    c = c.permute(2, 0, 1).cuda().contiguous()  # [L, M, 1]

    # Reinterpret as raw bytes (packed format)
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)

    # Scale factors - need to move batch dimension from innermost to outermost
    # Current shape: [32, 4, ceil(M/128), 4, ceil(K/64), L]
    # CUTLASS expects: [L, 32, 4, ceil(M/128), 4, ceil(K/64)]
    sfa_reordered = sfa_permuted.permute(5, 0, 1, 2, 3, 4).cuda().contiguous()
    sfb_reordered = sfb_permuted.permute(5, 0, 1, 2, 3, 4).cuda().contiguous()
    sfa_bytes = sfa_reordered.view(torch.uint8)
    sfb_bytes = sfb_reordered.view(torch.uint8)

    # Launch CUTLASS GEMV - outputs FP16 directly (no decode stage)
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back to [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()

    return c
