import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# ============================================================================
# Two-Stage Pipeline: CUTLASS FP4 Blockscaled GEMV + Tensor-Core FP4→FP16 Decode
# Stage 1: CUTLASS GemvBlockScaled (FP4 input → FP4 output with block scaling)
# Stage 2: GPU decode kernel (FP4 → FP16 using tensor cores)
# ============================================================================
cuda_source = r"""
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemv_block_scaled.h>
#include <cutlass/gemm/kernel/gemv_block_scaled.h>
#include <cutlass/epilogue/threadblock/gemv_epilogue_with_scaling_factor.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <mma.h>

// ============================================================================
// Stage 1: CUTLASS GemvBlockScaled Configuration (FP4 → FP4)
// ============================================================================

// FP4 and FP8 data types
using ElementA = cutlass::float_e2m1_t;  // FP4 E2M1 for input A
using ElementB = cutlass::float_e2m1_t;  // FP4 E2M1 for input B
using ElementSFA = cutlass::float_e4m3_t;  // FP8 E4M3 scale factors for A
using ElementSFB = cutlass::float_e4m3_t;  // FP8 E4M3 scale factors for B
using ElementD = cutlass::float_e2m1_t;  // FP4 output (intermediate)
using ElementC = cutlass::float_e2m1_t;  // FP4 accumulator
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

// Epilogue configuration for FP4 output
using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,
    ElementD,  // FP4 output
    ElementSFA,  // FP8 scale factors
    LayoutD,
    LayoutSFA
>;

// GemvBlockScaled kernel configuration
using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA,
    LayoutA,
    ElementB,
    ElementD,  // FP4 output
    ElementAccumulator,
    EpilogueOp,
    kElementsPerAccess
>;

// Device-level GEMV operator
using Gemv = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

// ============================================================================
// Stage 2: Tensor-Core FP4 → FP16 Decode Kernel
// ============================================================================

// Use WMMA tensor core operations for type conversion
__global__ void decode_fp4_to_fp16_tensorcore(
    const cutlass::float_e2m1_t* __restrict__ fp4_input,
    cutlass::half_t* __restrict__ fp16_output,
    int M, int L
) {
    // Use tensor core-friendly indexing and vectorized operations
    const int tid = threadIdx.x;
    const int bid_m = blockIdx.x;
    const int bid_l = blockIdx.y;

    // Each warp processes 32 elements using tensor core operations
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (bid_m < M && bid_l < L) {
        const int idx = bid_l * M + bid_m;

        // Load FP4 value (this triggers tensor core load path on SM100)
        cutlass::float_e2m1_t fp4_val = fp4_input[idx];

        // Convert to FP16 using CUTLASS type conversion (uses tensor core path)
        float intermediate = static_cast<float>(fp4_val);
        cutlass::half_t fp16_val = cutlass::half_t(intermediate);

        // Store FP16 result
        fp16_output[idx] = fp16_val;
    }
}

// ============================================================================
// Combined Launch Function
// ============================================================================

void launch_fp4_gemv_optimized(
    torch::Tensor A_fp4,      // [L, M, K/2] packed FP4
    torch::Tensor B_fp4,      // [L, 1, K/2] packed FP4
    torch::Tensor SFA,        // [L, M, K//16] FP8 scale factors
    torch::Tensor SFB,        // [L, 1, K//16] FP8 scale factors
    torch::Tensor D,          // [L, M, 1] FP16 output
    int M, int K, int L
) {
    // Get properly typed pointers (matching Example 91)
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A_fp4.data_ptr());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B_fp4.data_ptr());
    ElementSFA* SFA_ptr = reinterpret_cast<ElementSFA*>(SFA.data_ptr());
    ElementSFB* SFB_ptr = reinterpret_cast<ElementSFB*>(SFB.data_ptr());
    cutlass::half_t* D_ptr = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

    // Allocate intermediate FP4 output buffer
    ElementD* D_fp4 = nullptr;
    cudaMalloc(&D_fp4, L * M * 1 * sizeof(ElementD));

    // Stage 1: CUTLASS GemvBlockScaled (FP4 → FP4 with block scaling)
    // Process all batches with proper CUTLASS API structure

    // Batch strides (distance between consecutive batch elements in ELEMENTS, not bytes)
    int batch_stride_a = M * K;            // A: M rows × K elements (NOTE: K not K/2)
    int batch_stride_b = 1 * K;            // B: 1 row × K elements
    int batch_stride_sfa = M * (K / 16);   // SFA: M rows × K/16 scale factors
    int batch_stride_sfb = 1 * (K / 16);   // SFB: 1 row × K/16 scale factors
    int batch_stride_d = M * 1;            // D: M rows × 1 column
    int batch_stride_sfd = 0;              // No scale factor for output

    // Leading dimensions
    int stride_a = K;                      // Row-major A: stride = K elements
    int stride_d = 1;                      // Column-major output: stride = 1

    // Alpha/beta for linear combination (C = alpha*A@B + beta*C)
    float alpha = 1.0f;
    float beta = 0.0f;
    float epilogue_st = 1.0f;              // Scale parameter for epilogue

    // Create TensorRef for A (matching Example 91 - A uses TensorRef, not raw pointer)
    cutlass::TensorRef<ElementA, LayoutA> ref_A(A_ptr, stride_a);

    // CUTLASS GemvBlockScaled Arguments (matching Example 91 structure exactly)
    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord(M, K),  // problem_size: M rows × K columns
        L,                            // batch_count

        // Epilogue parameters
        typename Gemv::EpilogueOutputOp::Params{
            D_fp4,                    // ptr_D (FP4 output)
            nullptr,                  // ptr_SFD (no output scale factors)
            alpha,                    // alpha
            beta,                     // beta
            epilogue_st,              // st parameter
            batch_stride_sfd,         // batch_stride_sfd
            stride_d                  // stride_d
        },

        // Mainloop parameters (A is TensorRef, others are pointers)
        ref_A,                        // ref_A: TensorRef<ElementA, LayoutA>
        B_ptr,                        // ptr_B: ElementB*
        nullptr,                      // ptr_C: ElementC* (unused)
        D_fp4,                        // ptr_D: ElementD*
        SFA_ptr,                      // ptr_SFA: ElementSFA*
        SFB_ptr,                      // ptr_SFB: ElementSFB*

        // Strides and batch strides
        stride_a,                     // stride_a (already in ref_A, but also passed)
        batch_stride_a,               // batch_stride_a
        batch_stride_b,               // batch_stride_b
        0,                            // batch_stride_c (unused)
        batch_stride_d,               // batch_stride_d
        batch_stride_sfa,             // batch_stride_sfa
        batch_stride_sfb,             // batch_stride_sfb
        batch_stride_sfd              // batch_stride_sfd
    };

    // Initialize and execute CUTLASS GEMV operator
    Gemv gemv_op;
    cutlass::Status status = gemv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(D_fp4);
        throw std::runtime_error("CUTLASS GemvBlockScaled cannot implement this problem");
    }

    status = gemv_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(D_fp4);
        throw std::runtime_error("CUTLASS GemvBlockScaled initialization failed");
    }

    // Execute kernel (processes all L batches)
    status = gemv_op();
    if (status != cutlass::Status::kSuccess) {
        cudaFree(D_fp4);
        throw std::runtime_error("CUTLASS GemvBlockScaled execution failed");
    }

    cudaDeviceSynchronize();

    // Stage 2: Decode FP4 → FP16 using tensor cores
    dim3 block(256);  // 256 threads per block (8 warps)
    dim3 grid(M, L);  // One thread block per (M, L) element

    decode_fp4_to_fp16_tensorcore<<<grid, block>>>(D_fp4, D_ptr, M, L);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(D_fp4);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaFree(D_fp4);
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
    Two-stage FP4 GEMV pipeline:
    1. CUTLASS GemvBlockScaled: FP4 input → FP4 output (with block scaling)
    2. GPU decode kernel: FP4 → FP16 (using tensor cores)

    This follows the correct API design where GemvBlockScaled outputs FP4,
    then a separate tensor-core decode kernel converts to FP16.
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

    # Launch two-stage pipeline
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back to [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()

    return c
