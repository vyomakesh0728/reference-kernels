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
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

using namespace cute;

// CUTLASS 3.x SM100 FP4 GEMM configuration
// Treating GEMV as GEMM with N=8 for tensor core efficiency

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;
using ElementScaleFactor = cutlass::float_ue8m0_t;

using LayoutA = cutlass::layout::RowMajor;  // M x K
using LayoutB = cutlass::layout::ColumnMajor;  // K x N (for TN layout)
using LayoutC = cutlass::layout::RowMajor;  // M x N

// SM100 architecture
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shape: use 128x8x256 for GEMV (small N)
using TileShape = Shape<_128, _8, _256>;  // M, N, K
using ClusterShape = Shape<_1, _1, _1>;   // No cluster for GEMV

// Kernel schedule
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1SmNvf4;

// Stage count for pipeline
constexpr int Stages = 4;

// Use CUTLASS CollectiveBuilder for mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA, LayoutA, 32,  // Alignment 32 for FP4
    ElementB, LayoutB, 32,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename cutlass::epilogue::collective::detail::Sm90TmaWarpSpecialized1SmInternalParams<
            ElementC, LayoutC
        >::SmemLayoutAtom))
    >,
    KernelSchedule
>::CollectiveOp;

// Epilogue: simple passthrough (D = C)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, 32,
    ElementC, LayoutC, 32,
    EpilogueSchedule
>::CollectiveOp;

// Define the GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // Problem shape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Host function to launch CUTLASS GEMM
void launch_cutlass_fp4_gemv(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t K, int64_t L
) {
    // Get pointers
    const ElementA* A_ptr = reinterpret_cast<const ElementA*>(A.data_ptr<uint8_t>());
    const ElementB* B_ptr = reinterpret_cast<const ElementB*>(B.data_ptr<uint8_t>());
    const ElementScaleFactor* SFA_ptr = reinterpret_cast<const ElementScaleFactor*>(SFA.data_ptr<uint8_t>());
    const ElementScaleFactor* SFB_ptr = reinterpret_cast<const ElementScaleFactor*>(SFB.data_ptr<uint8_t>());
    ElementC* D_ptr = reinterpret_cast<ElementC*>(D.data_ptr<at::Half>());

    // For GEMV, treat as GEMM with N=8 then reduce
    // This allows tensor cores to work efficiently
    const int64_t N = 8;

    // Problem size for batch
    typename Gemm::Arguments arguments;

    // Run batched GEMM
    for (int64_t batch = 0; batch < L; batch++) {
        // Batch offsets
        const int64_t K_packed = K / 2;  // FP4 is packed 2 per byte
        const int64_t K_scales = K / 16;  // Scale factors every 16 elements

        const ElementA* A_batch = A_ptr + batch * M * K_packed;
        const ElementB* B_batch = B_ptr + batch * 128 * K_packed;  // B padded to 128
        const ElementScaleFactor* SFA_batch = SFA_ptr + batch * M * K_scales;
        const ElementScaleFactor* SFB_batch = SFB_ptr + batch * 128 * K_scales;
        ElementC* D_batch = D_ptr + batch * M;

        // Set up GEMM arguments
        // Problem shape: M x N x K x batch (batch=1 for this iteration)
        arguments.problem_shape = cutlass::gemm::GemmCoord(M, N, K);
        arguments.mainloop.ptr_A = A_batch;
        arguments.mainloop.ptr_B = B_batch;
        arguments.mainloop.ptr_scale_A = SFA_batch;
        arguments.mainloop.ptr_scale_B = SFB_batch;
        arguments.epilogue.ptr_C = D_batch;
        arguments.epilogue.ptr_D = D_batch;

        // Strides
        arguments.mainloop.dA = cutlass::make_cute_packed_stride(
            typename GemmKernel::StrideA{}, {M, K, 1}
        );
        arguments.mainloop.dB = cutlass::make_cute_packed_stride(
            typename GemmKernel::StrideB{}, {N, K, 1}
        );
        arguments.epilogue.dC = cutlass::make_cute_packed_stride(
            typename GemmKernel::StrideC{}, {M, N, 1}
        );
        arguments.epilogue.dD = cutlass::make_cute_packed_stride(
            typename GemmKernel::StrideD{}, {M, N, 1}
        );

        // Create GEMM operator
        Gemm gemm_op;

        // Check if arguments are valid
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM cannot be implemented!");
        }

        // Initialize
        status = gemm_op.initialize(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM initialization failed!");
        }

        // Run
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM execution failed!");
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
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
            name="nvfp4_gemv_cutlass_v6",
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
                "-Xcudafe", "--diag_suppress=20012",  # Suppress CuTe warnings
                f"-I{cutlass_path}/include",
                f"-I{cutlass_path}/tools/util/include",
            ],
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    B200-optimized FP4 GEMV using CUTLASS CollectiveBuilder.

    Leverages SM100 tensor cores via CUTLASS 3.x API with block-scaled FP4.
    Target: < 10 µs geom_mean
    """
    a, b, sfa, sfb, _, _, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2

    # Permute to [L, M, K/2] layout
    a = a.permute(2, 0, 1).cuda().contiguous()
    b = b.permute(2, 0, 1).cuda().contiguous()
    sfa = sfa.permute(2, 0, 1).cuda().contiguous()
    sfb = sfb.permute(2, 0, 1).cuda().contiguous()
    c = c.permute(2, 0, 1).cuda().contiguous()

    # Launch CUTLASS GEMM
    mod = get_module()
    mod.launch_cutlass_fp4_gemv(a, b, sfa, sfb, c, M, K, L)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()

    return c
