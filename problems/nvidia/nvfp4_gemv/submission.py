import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

# Clean C++ header declaration
cpp_source = r"""
#include <torch/extension.h>
void launch_fp4_gemv_optimized(
    torch::Tensor A_fp4,
    torch::Tensor B_fp4,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor D,
    int M, int K, int L
);
"""

# Fixed CUDA implementation
cuda_source = r"""
#include <cute/tensor.hpp>
#include <cutlass/arch/memory_sm100.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/gemm/device/gemv_blockscaled.h>
#include <cutlass/gemm/kernel/gemv_blockscaled.h>
#include <cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <mma.h>
#include <torch/extension.h>



using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementSFA = cutlass::float_e4m3_t;
using ElementSFB = cutlass::float_e4m3_t;
using ElementD = cutlass::float_e2m1_t;
using ElementC = cutlass::float_e2m1_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using LayoutSFA = cutlass::layout::ColumnMajor;
using LayoutSFB = cutlass::layout::ColumnMajor;

static constexpr int kVectorSize = 16;
static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;
using ThreadShape = cutlass::gemm::GemmShape<16, 8, 1>;

using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
    kVectorSize, ThreadShape, ElementCompute, ElementAccumulator,
    ElementC, ElementD, ElementSFA, LayoutD, LayoutSFA>;

using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA, LayoutA, ElementB, ElementD, ElementAccumulator, EpilogueOp, kElementsPerAccess>;

using Gemv = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;

// Tensor-core accelerated FP4→FP16 decode for Blackwell SM100
__global__ void decode_fp4_to_fp16_tensorcore(
    const cutlass::float_e2m1_t* __restrict__ fp4_input,
    cutlass::half_t* __restrict__ fp16_output,
    int total_elements
) {
    using ElementFP4 = cutlass::float_e2m1_t;
    using ElementFP16 = cutlass::half_t;

    // SM100 optimal vector width: 16 FP4 elements (8 bytes) per thread
    constexpr int kVectorWidth = 16;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = tid * kVectorWidth;

    // Main vectorized path - maps to SM100 tensor-core LDG.128/STG.128
    if (offset + kVectorWidth <= total_elements) {
        // 1. Vectorized load: 8 bytes (16 FP4 values) per thread
        cute::Array<ElementFP4, kVectorWidth> fp4_vec;
        cutlass::arch::global_load<kVectorWidth / 2, cutlass::arch::CacheOperation::LastUse>(
            reinterpret_cast<uint32_t*>(&fp4_vec),
            reinterpret_cast<const uint32_t*>(fp4_input + offset / 2),
            true
        );

        // 2. Tensor-core accelerated conversion: CVT.FP4.FP16
        cute::Array<ElementFP16, kVectorWidth> fp16_vec;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kVectorWidth; ++i) {
            fp16_vec[i] = cutlass::NumericConverter<ElementFP16, ElementFP4>::convert(fp4_vec[i]);
        }

        // 3. Vectorized store: 32 bytes (16 FP16 values) per thread
        cutlass::arch::global_store<kVectorWidth * sizeof(ElementFP16), cutlass::arch::CacheOperation::Writeback>(
            reinterpret_cast<const uint32_t*>(&fp16_vec),
            reinterpret_cast<uint32_t*>(fp16_output + offset),
            true
        );
    }
    // Tail processing for non-vectorized elements
    else if (offset < total_elements) {
        const int remaining = total_elements - offset;
        for (int i = 0; i < remaining; ++i) {
            fp16_output[offset + i] =
                cutlass::NumericConverter<ElementFP16, ElementFP4>::convert(fp4_input[offset / 2 + i / 2]);
        }
    }
}

void launch_fp4_gemv_optimized(
    torch::Tensor A_fp4,
    torch::Tensor B_fp4,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor D,
    int M, int K, int L
) {
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A_fp4.data_ptr());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B_fp4.data_ptr());
    ElementSFA* SFA_ptr = reinterpret_cast<ElementSFA*>(SFA.data_ptr());
    ElementSFB* SFB_ptr = reinterpret_cast<ElementSFB*>(SFB.data_ptr());
    cutlass::half_t* D_ptr = reinterpret_cast<cutlass::half_t*>(D.data_ptr());

    ElementD* D_fp4 = nullptr;
    const size_t fp4_elements = M * L;
    cudaMalloc(&D_fp4, fp4_elements * sizeof(ElementD));

    int batch_stride_a = M * K;
    int batch_stride_b = 1 * K;
    int batch_stride_sfa = M * (K / 16);
    int batch_stride_sfb = 1 * (K / 16);
    int batch_stride_d = M * 1;
    int batch_stride_sfd = 0;

    int stride_a = K;
    int stride_d = 1;
    float alpha = 1.0f;
    float beta = 0.0f;
    float epilogue_st = 1.0f;

    cutlass::TensorRef<ElementA, LayoutA> ref_A(A_ptr, stride_a);
    cutlass::TensorRef<ElementD, LayoutD> ref_D(D_fp4, stride_d);

    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord(M, K),
        L,
        typename Gemv::EpilogueOutputOp::Params{
            ref_D, nullptr, alpha, beta, epilogue_st, batch_stride_sfd, stride_d},
        ref_A, B_ptr, nullptr, D_fp4, SFA_ptr, SFB_ptr,
        stride_a, batch_stride_a, batch_stride_b, 0, batch_stride_d,
        batch_stride_sfa, batch_stride_sfb, batch_stride_sfd
    };

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

    status = gemv_op();
    if (status != cutlass::Status::kSuccess) {
        cudaFree(D_fp4);
        throw std::runtime_error("CUTLASS GemvBlockScaled execution failed");
    }

    cudaDeviceSynchronize();

    const int threads_per_block = 128;
    const int blocks_needed = (total_elements + kVectorWidth * threads_per_block - 1) /
                              (kVectorWidth * threads_per_block);

    decode_fp4_to_fp16_tensorcore<<<blocks_needed, threads_per_block>>>(
        D_fp4, D_ptr, total_elements);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(D_fp4);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaFree(D_fp4);
}
"""


def get_module():
    return load_inline(
        name="fp4_gemv_cutlass_blockscaled",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["launch_fp4_gemv_optimized"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-I{cutlass_path}/include",
            f"-I{cutlass_path}/tools/util/include",
            "-gencode=arch=compute_100,code=sm_100",
            "-maxrregcount=128",
            "-DNDEBUG",
        ],
        with_cuda=True,
        verbose=True,
    )


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa_ref_cpu, sfb_ref_cpu, _, _, c = data
    M, _, L = c.shape

    # FIX 1: Use logical K dimension directly from input shape
    # According to checklist, a has shape [M, K, L], so a.shape[1] is K
    K = a.shape[1]

    # Debug: Print input tensor shapes for verification
    print(f"DEBUG: Input shapes - a: {a.shape}, b: {b.shape}, c: {c.shape}")
    print(
        f"DEBUG: Expected shapes - a: ({M}, {K}, {L}), b: (1, {K}, {L}), c: ({M}, 1, {L})"
    )

    # FIX 2: Enforce strict shape conformance per checklist
    # These are LOGICAL shapes; physical packing is handled by .view(torch.uint8)
    assert a.shape == (M, K, L), f"A shape mismatch: {a.shape} != ({M}, {K}, {L})"
    assert b.shape == (1, K, L), f"B shape mismatch: {b.shape} != (1, {K}, {L})"
    assert c.shape == (M, 1, L), f"C shape mismatch: {c.shape} != ({M}, 1, {L})"
    assert sfa_ref_cpu.shape == (M, K // 16, L), f"SFA shape mismatch"
    assert sfb_ref_cpu.shape == (1, K // 16, L), f"SFB shape mismatch"

    # FIX 3: Simplified FP4 packing - view as uint8 (2 FP4 values per byte)
    # No complex conditionals; input shapes are guaranteed by assertions above
    a_bytes = a.view(torch.uint8)  # Physical shape: [M, K//2, L]
    b_bytes = b.view(torch.uint8)  # Physical shape: [1, K//2, L]

    # FIX 4: Canonical batch dimension permutation
    # Result: [L, M, K//2] and [L, 1, K//2] for batched GEMV
    a_bytes = a_bytes.permute(2, 0, 1).cuda().contiguous()
    b_bytes = b_bytes.permute(2, 0, 1).cuda().contiguous()

    # Scale factors: [M, K//16, L] → [L, M, K//16] (FP8 is byte-aligned, no packing)
    sfa = sfa_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfb = sfb_ref_cpu.permute(2, 0, 1).cuda().contiguous()
    sfa_bytes = sfa.view(torch.uint8)
    sfb_bytes = sfb.view(torch.uint8)

    # FIX 5: Verify physical byte counts after packing
    expected_a_bytes = L * M * (K // 2)
    expected_b_bytes = L * 1 * (K // 2)

    assert a_bytes.numel() == expected_a_bytes, (
        f"A byte count mismatch: {a_bytes.numel()} != {expected_a_bytes}"
    )
    assert b_bytes.numel() == expected_b_bytes, (
        f"B byte count mismatch: {b_bytes.numel()} != {expected_b_bytes}"
    )

    # Prepare output tensor: [M, 1, L] → [L, M, 1]
    c = c.permute(2, 0, 1).cuda().contiguous()

    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Restore original output layout: [L, M, 1] → [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()
    return c
