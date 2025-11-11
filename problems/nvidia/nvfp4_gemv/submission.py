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
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// SM100 FP4 Block-Scaled GEMV configuration
// Uses specialized GEMV kernel from CUTLASS example 91

using ElementA = cutlass::float_e2m1_t;      // FP4 E2M1 for matrix A
using ElementB = cutlass::float_e2m1_t;      // FP4 E2M1 for vector B
using ElementC = cutlass::half_t;            // FP16 for output
using ElementAccumulator = float;            // FP32 accumulator
using ElementSFA = cutlass::float_e4m3_t;    // FP8 E4M3 scale factors for A
using ElementSFB = cutlass::float_e4m3_t;    // FP8 E4M3 scale factors for B

using LayoutA = cutlass::layout::RowMajor;   // A is row-major (M x K)

// Epilogue operation: D = alpha*accumulator + beta*C
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                // Output element type
    1,                       // Elements per access
    ElementAccumulator,      // Accumulator type
    ElementAccumulator       // Compute type
>;

// Define the specialized GEMV kernel for block-scaled FP4
// Parameters: ElementsPerAccess=32 (for FP4), ThreadCount=128, ThreadsPerRow=16
using GemvKernel = cutlass::gemm::kernel::GemvBlockScaled<
    ElementA,                // Matrix A element type
    LayoutA,                 // Matrix A layout (RowMajor)
    ElementB,                // Vector B element type
    ElementC,                // Output vector C element type
    ElementAccumulator,      // Accumulator type
    EpilogueOutputOp,        // Epilogue operation
    32,                      // kElementsPerAccess (required for FP4)
    128,                     // kThreadCount (threads per block)
    16,                      // kThreadsPerRow (threads in K dimension)
    ElementSFA,              // Scale factor A type (FP8 E4M3)
    ElementSFB               // Scale factor B type (FP8 E4M3)
>;

// Host function to launch CUTLASS FP4 GEMV
void launch_cutlass_fp4_gemv(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t K, int64_t L
) {
    // Get pointers
    const ElementA* A_ptr = reinterpret_cast<const ElementA*>(A.data_ptr<uint8_t>());
    const ElementB* B_ptr = reinterpret_cast<const ElementB*>(B.data_ptr<uint8_t>());
    const ElementSFA* SFA_ptr = reinterpret_cast<const ElementSFA*>(SFA.data_ptr<uint8_t>());
    const ElementSFB* SFB_ptr = reinterpret_cast<const ElementSFB*>(SFB.data_ptr<uint8_t>());
    ElementC* D_ptr = reinterpret_cast<ElementC*>(D.data_ptr<at::Half>());

    // Dimensions
    const int64_t K_packed = K / 2;      // FP4 is packed 2 per byte
    const int64_t K_scales = K / 16;     // Scale factors every 16 elements

    // Setup epilogue params
    typename EpilogueOutputOp::Params epilogue_params{
        ElementAccumulator(1.0f),  // alpha
        ElementAccumulator(0.0f)   // beta
    };

    // Run batched GEMV
    for (int64_t batch = 0; batch < L; batch++) {
        // Batch offsets
        const ElementA* A_batch = A_ptr + batch * M * K_packed;
        const ElementB* B_batch = B_ptr + batch * 128 * K_packed;  // B padded to 128, use first row
        const ElementSFA* SFA_batch = SFA_ptr + batch * M * K_scales;
        const ElementSFB* SFB_batch = SFB_ptr + batch * 128 * K_scales;
        ElementC* D_batch = D_ptr + batch * M;

        // Create TensorRef for matrix A
        cutlass::TensorRef<ElementA const, LayoutA> ref_A(A_batch, K_packed);

        // GEMV kernel arguments
        typename GemvKernel::Arguments arguments{
            cutlass::MatrixCoord(M, K),     // problem_size (M rows, K columns)
            1,                               // batch_count
            epilogue_params,                 // epilogue params
            ref_A,                           // ref_A
            B_batch,                         // ptr_B
            nullptr,                         // ptr_C (not used, beta=0)
            D_batch,                         // ptr_D
            K_packed,                        // stride_A (row stride in packed elements)
            0,                               // batch_stride_A
            0,                               // batch_stride_B
            0,                               // batch_stride_C
            0,                               // batch_stride_D
            SFA_batch,                       // ptr_SFA
            SFB_batch,                       // ptr_SFB
            0,                               // batch_stride_SFA
            0,                               // batch_stride_SFB
            0                                // batch_stride_SFD
        };

        // Allocate shared memory (if needed)
        size_t workspace_size = GemvKernel::get_workspace_size(arguments);

        // Check if kernel can be implemented
        cutlass::Status status = GemvKernel::can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMV cannot be implemented with these parameters!");
        }

        // Launch kernel
        dim3 grid = GemvKernel::get_grid_shape(arguments);
        dim3 block = GemvKernel::get_block_shape();

        // Allocate shared memory
        int smem_size = int(sizeof(typename GemvKernel::SharedStorage));

        // Launch
        cutlass::Kernel<GemvKernel><<<grid, block, smem_size>>>(arguments);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    // Synchronize to catch any runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA sync error: ") + cudaGetErrorString(err));
    }
}
"""

cpp_source = """
void launch_cutlass_fp4_gemv(
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
            name="nvfp4_gemv_blockscaled",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["launch_cutlass_fp4_gemv"],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "-arch=sm_100a",
                "--expt-relaxed-constexpr",
                "-Xcudafe", "--diag_suppress=20012",
                f"-I{cutlass_path}/include",
                f"-I{cutlass_path}/tools/util/include",
            ],
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    B200-optimized FP4 GEMV using CUTLASS GemvBlockScaled kernel.

    Leverages SM100 tensor cores via specialized GEMV kernel with block-scaled FP4.
    Target: < 10 µs geom_mean
    """
    a, b, sfa_ref_cpu, sfb_ref_cpu, _, _, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2

    # Permute to [L, M, K/2] layout for matrix data
    a = a.permute(2, 0, 1).cuda().contiguous()
    b = b.permute(2, 0, 1).cuda().contiguous()
    c = c.permute(2, 0, 1).cuda().contiguous()

    # Use simple scale factor format [M, K_scales, L] -> [L, M, K_scales]
    # CUTLASS GEMV kernel expects simple layout, not CuTe-permuted format
    sfa = sfa_ref_cpu.permute(2, 0, 1).cuda().contiguous().to(dtype=torch.float8_e4m3fn)
    sfb = sfb_ref_cpu.permute(2, 0, 1).cuda().contiguous().to(dtype=torch.float8_e4m3fn)

    # Launch CUTLASS GEMV
    mod = get_module()
    mod.launch_cutlass_fp4_gemv(a, b, sfa, sfb, c, M, K, L)

    # Permute output back to [M, 1, L]
    c = c.permute(1, 2, 0).contiguous()

    return c
