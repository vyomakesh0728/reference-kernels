import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# ============================================================================
# CUTLASS GEMV BlockScaled API - Production SM100 FP4 GEMV
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

// GEMV produces FP4 output with scale factors, we'll decode to FP16 after
using ElementC = cutlass::float_e2m1_t;
using ElementD = cutlass::float_e2m1_t;  // FP4 intermediate output
using LayoutD = cutlass::layout::ColumnMajor;

using ElementSFD = cutlass::float_e4m3_t;
using LayoutSFD = cutlass::layout::ColumnMajor;

using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

// Vector size for block scaling (must match scale factor block size)
static constexpr int kVectorSize = 16;
static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;

// Thread shape for GEMV (MUST be 16x8)
using ThreadShape = cutlass::gemm::GemmShape<16, 8>;
static_assert(kVectorSize == ThreadShape::kM, "vector size mismatch");

// Epilogue with scaling factor output
using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,
    ElementSFD,
    LayoutD,
    LayoutSFD
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
// GPU Decode Kernel: FP4 + FP8 Scale Factors → FP16
// ============================================================================

// FP4 E2M1 lookup table (1 sign, 2 exp, 1 mantissa)
__constant__ float fp4_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t byte) {
    // FP8 E4M3: 1 sign, 4 exp, 3 mantissa (bias = 7)
    // Use CUTLASS type for proper decoding
    cutlass::float_e4m3_t fp8_val;
    reinterpret_cast<uint8_t&>(fp8_val) = byte;
    return static_cast<float>(fp8_val);
}

__global__ void decode_fp4_to_fp16_kernel(
    const uint8_t* __restrict__ fp4_packed,
    const uint8_t* __restrict__ scale_factors,
    half* __restrict__ output_fp16,
    int M, int L
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.y;

    if (idx >= M || batch >= L) return;

    const int packed_idx = idx / 2;
    const int nibble_idx = idx % 2;
    const int sf_idx = idx / kVectorSize;

    // Get FP4 packed byte
    const int batch_offset_fp4 = batch * (M / 2);
    const uint8_t packed_byte = fp4_packed[batch_offset_fp4 + packed_idx];

    // Extract nibble (low nibble = even idx, high nibble = odd idx)
    uint8_t nibble;
    if (nibble_idx == 0) {
        nibble = packed_byte & 0x0F;
    } else {
        nibble = (packed_byte >> 4) & 0x0F;
    }

    // Decode FP4 E2M1 using lookup table
    float decoded_val = fp4_e2m1_lut[nibble];

    // Decode FP8 E4M3 scale factor
    const int batch_offset_sf = batch * (M / kVectorSize);
    uint8_t sf_byte = scale_factors[batch_offset_sf + sf_idx];
    float sf_decoded = decode_fp8_e4m3(sf_byte);

    // FIX: Scale factors from epilogue already incorporate st parameter
    // Simply multiply dequantized FP4 value by the scale factor
    float result = decoded_val * sf_decoded;
    const int output_idx = batch * M + idx;
    output_fp16[output_idx] = __float2half(result);
}

// ============================================================================
// Launcher (FP4 output from GEMV, then GPU decode to FP16)
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

    // Allocate temporary FP4 output and scale factors
    uint8_t* D_fp4_ptr = nullptr;
    uint8_t* SFD_ptr = nullptr;
    cudaMalloc(&D_fp4_ptr, gemm_batch * (gemm_m / 2) * sizeof(uint8_t));
    cudaMalloc(&SFD_ptr, gemm_batch * (gemm_m / kVectorSize) * sizeof(uint8_t));

    // Calculate batch strides
    const int k_blks = (gemm_k + kVectorSize - 1) / kVectorSize;  // ceil(K/16)
    const int m_blks = (gemm_m + 127) / 128;                       // ceil(M/128)

    // For permuted scale factors with shape [L, 32, 4, ceil(M/128), 4, ceil(K/64)]
    const int k_blks_sf = (gemm_k + 63) / 64;  // ceil(K/64) for atom_k=4 blocking
    const int64_t sf_elements_per_batch = 32 * 4 * m_blks * 4 * k_blks_sf;

    const int64_t batch_stride_A = gemm_m * (gemm_k / 2);
    const int64_t batch_stride_B = gemm_k / 2;
    const int64_t batch_stride_C = 0;
    const int64_t batch_stride_D = gemm_m / 2;  // FP4 packed
    const int64_t batch_stride_SFA = sf_elements_per_batch;
    const int64_t batch_stride_SFB = 32 * 4 * 1 * 4 * k_blks_sf;  // For B: M=1 (padded to 128)
    const int64_t batch_stride_SFD = gemm_m / kVectorSize;

    // Construct TensorRefs
    cutlass::TensorRef<ElementA, LayoutA> ref_A(
        A_ptr,
        LayoutA::packed({gemm_m, gemm_k / 2})
    );

    cutlass::TensorRef<ElementD, LayoutD> ref_D(
        reinterpret_cast<ElementD*>(D_fp4_ptr),
        LayoutD::packed({gemm_m, 1})
    );

    ElementCompute alpha = ElementCompute(1.0f);
    ElementCompute beta = ElementCompute(0.0f);
    float epilogue_st = 2.0f;  // Deterministic quantization scale

    // Setup GEMV arguments
    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord{gemm_m, gemm_k},
        gemm_batch,
        typename EpilogueOp::Params{
            ref_D,
            reinterpret_cast<ElementSFD*>(SFD_ptr),
            alpha,
            beta,
            epilogue_st,
            batch_stride_SFD,
            gemm_m
        },
        ref_A,
        B_ptr,
        nullptr,  // ptr_C
        reinterpret_cast<ElementD*>(D_fp4_ptr),
        SFA_ptr,
        SFB_ptr,
        gemm_k / 2,
        batch_stride_A,
        batch_stride_B,
        batch_stride_C,
        batch_stride_D,
        batch_stride_SFA,
        batch_stride_SFB,
        batch_stride_SFD
    };

    // Execute GEMV (produces FP4 + scale factors)
    Gemv gemv_op;
    cutlass::Status status = gemv_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMV initialization failed");
    }

    status = gemv_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMV execution failed");
    }

    // Decode FP4 + scale factors → FP16 on GPU
    dim3 block(256);
    dim3 grid((gemm_m + 255) / 256, gemm_batch);
    decode_fp4_to_fp16_kernel<<<grid, block>>>(
        D_fp4_ptr, SFD_ptr, D_fp16_ptr, gemm_m, gemm_batch
    );

    // Free temporary buffers
    cudaFree(D_fp4_ptr);
    cudaFree(SFD_ptr);

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

    # Launch CUTLASS GEMV (produces FP16 output directly!)
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back to [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()

    return c
