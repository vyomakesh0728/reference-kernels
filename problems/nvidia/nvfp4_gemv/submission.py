import os

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

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
    const uint8_t* B_batch = B_packed + batch * K_packed;
    const uint8_t* SFA_batch = SFA_packed + batch * M * K_scales;
    const uint8_t* SFB_batch = SFB_packed + batch * K_scales;
    half* D_batch = D + batch * M;

    // Thread indexing and warp constants
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;   // warp index within the block (0-7)
    const int lane_id = tid & 31;   // lane index within the warp (0-31)
    const int m_cta = blockIdx.x * kTileM;
    if (m_cta >= M) return;
    // Number of warps that collaboratively compute one M tile
    const int warps_per_tile = kTileM / 16;

    // Shared memory allocations.  We align by 128 bytes to satisfy the
    // requirements of ldmatrix (16x8 half tile = 256B).  A_smem holds the
    // decoded FP4 values for a tile of A (kTileM rows × kTileK columns).  We
    // allocate a few extra columns (+8) to prevent bank conflicts when
    // loading 16×16 fragments.  B_smem holds the decoded FP4 values for a
    // tile of B (kTileK rows) and broadcasts them across 8 columns so that
    // mma.sync can consume a 16×8 tile each iteration.
    __shared__ __align__(128) half A_smem[kTileM][kTileK + 8];
    __shared__ __align__(128) half B_smem[kTileK][8];

    // Accumulators for the 2×2 tile produced by mma.sync.  Each warp lane
    // maintains four float accumulators corresponding to a 2×2 sub‑tile of
    // the 16×8 output.  We initialize them once here and accumulate over
    // the entire K dimension.
    float c_frag_0 = 0.0f;
    float c_frag_1 = 0.0f;
    float c_frag_2 = 0.0f;
    float c_frag_3 = 0.0f;

    // Loop over K in tiles of kTileK elements.  Each iteration loads and
    // decodes a slice of A and B into shared memory, performs all mma.sync
    // operations for that slice, and then advances to the next K tile.
    for (int k_tile = 0; k_tile < K; k_tile += kTileK) {
        const int k_packed_tile = k_tile / 2;   // offset in packed FP4 arrays
        const int k_scale_tile = k_tile / 16;   // offset in FP8 scale arrays

        // --------------------------------------------------------------------
        // Phase 1: Decode and load A into shared memory
        //
        // Each thread cooperatively decodes FP4 values from A_batch into
        // half-precision and applies the per‑16‑element blockscale.  A tile
        // covers kTileM rows and kTileK columns; there are kTileK/2 packed
        // bytes per row because each byte holds two FP4 values.  We stride by
        // kThreads to ensure all threads participate in the load.  Out‑of‑range
        // indices are padded with zero.
        for (int idx = tid; idx < kTileM * (kTileK / 2); idx += kThreads) {
            int row = idx / (kTileK / 2);
            int col_packed = idx % (kTileK / 2);
            int m_idx = m_cta + row;
            if (m_idx < M && (k_packed_tile + col_packed) < K_packed) {
                uint8_t packed = A_batch[m_idx * K_packed + k_packed_tile + col_packed];
                int scale_idx = col_packed / 8;
                half scale_h = __float2half(1.0f);
                if ((k_scale_tile + scale_idx) < K_scales) {
                    float scale_val = decode_fp8_e4m3(SFA_batch[m_idx * K_scales + k_scale_tile + scale_idx]);
                    scale_h = __float2half(scale_val);
                }
                // Unpack two FP4s from the byte and multiply by scale
                half v0 = decode_fp4_e2m1(packed & 0x0F);
                half v1 = decode_fp4_e2m1((packed >> 4) & 0x0F);
                A_smem[row][col_packed * 2]     = __hmul(v0, scale_h);
                A_smem[row][col_packed * 2 + 1] = __hmul(v1, scale_h);
            } else if (row < kTileM && col_packed * 2 < kTileK) {
                // Pad out-of-range entries with zero
                A_smem[row][col_packed * 2]     = __float2half(0.0f);
                A_smem[row][col_packed * 2 + 1] = __float2half(0.0f);
            }
        }

        // --------------------------------------------------------------------
        // Phase 1: Decode and load B into shared memory
        //
        // B is a GEMV vector (N=1), but mma.sync expects an 8‑column tile.  We
        // decode each pair of FP4 values, apply the per‑block scale, and
        // broadcast the resulting half across all 8 columns.  Out‑of‑range
        // indices are padded with zero.
        for (int col_packed = tid; col_packed < kTileK / 2; col_packed += kThreads) {
            if ((k_packed_tile + col_packed) < K_packed) {
                uint8_t packed = B_batch[k_packed_tile + col_packed];
                int scale_idx = col_packed / 8;
                half scale_h = __float2half(1.0f);
                if ((k_scale_tile + scale_idx) < K_scales) {
                    float scale_val = decode_fp8_e4m3(SFB_batch[k_scale_tile + scale_idx]);
                    scale_h = __float2half(scale_val);
                }
                half v0 = decode_fp4_e2m1(packed & 0x0F);
                half v1 = decode_fp4_e2m1((packed >> 4) & 0x0F);
                v0 = __hmul(v0, scale_h);
                v1 = __hmul(v1, scale_h);
                // Broadcast both values to all eight columns
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    B_smem[col_packed * 2][n]     = v0;
                    B_smem[col_packed * 2 + 1][n] = v1;
                }
            } else if (col_packed * 2 < kTileK) {
                // Pad with zero
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    B_smem[col_packed * 2][n]     = __float2half(0.0f);
                    B_smem[col_packed * 2 + 1][n] = __float2half(0.0f);
                }
            }
        }

        // Ensure all decoded values are visible before MMA
        __syncthreads();

        // --------------------------------------------------------------------
        // Phase 2: PTX MMA Compute (m16n8k16 Tensor Core Operations)
        //
        // Each warp computes a 16×8 tile of C using tensor cores.  We only
        // extract the first column (GEMV), but performing the full MMA is
        // necessary to utilize the tensor core throughput.  Warps beyond
        // warps_per_tile remain idle.
        if (warp_id < warps_per_tile) {
            int row_offset = warp_id * 16;
            // Iterate over the K dimension in 16‑element chunks.  Each
            // iteration loads a 16×16 fragment from A_smem and a 16×8 fragment
            // from B_smem via ldmatrix, then issues one mma.sync.
            for (int kk = 0; kk < kTileK; kk += 16) {
                // Load A fragment: 16×16 halves -> four 32‑bit registers per lane
                const half* a_base = &A_smem[row_offset][kk];
                unsigned long long a_ptr = __cvta_generic_to_shared((void*)a_base);
                unsigned a_reg_0, a_reg_1, a_reg_2, a_reg_3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
                    : "=r"(a_reg_0), "=r"(a_reg_1), "=r"(a_reg_2), "=r"(a_reg_3)
                    : "l"(a_ptr)
                );

                // Load B fragment: two 8×8 halves -> two 32‑bit registers per lane
                const half* b_base = &B_smem[kk][0];
                unsigned long long b_ptr_sh = __cvta_generic_to_shared((void*)b_base);
                int b_quad = lane_id >> 3;        // 0..3 (eight‑lane groups)
                int b_row = lane_id & 7;          // 0..7
                int b_k_block = (b_quad & 1) * 8; // selects upper or lower half of B tile
                unsigned long long b_addr = b_ptr_sh + ((unsigned long long)(b_k_block + b_row) * 8ULL) * sizeof(half);
                if (b_quad > 1) {
                    // Remap lanes 16–31: treat them as a second warp on the same 16×8 tile
                    int lower = lane_id & 15;
                    int lg    = lower >> 3;
                    int lr    = lower & 7;
                    int lrBlk = (lg & 1) * 8;
                    b_addr = b_ptr_sh + ((unsigned long long)(lrBlk + lr) * 8ULL) * sizeof(half);
                }
                unsigned b_reg_0, b_reg_1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [%2];\n"
                    : "=r"(b_reg_0), "=r"(b_reg_1)
                    : "l"(b_addr)
                );

                // MMA accumulate: multiply A and B fragments and accumulate into c_frag
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };\n"
                    : "+f"(c_frag_0), "+f"(c_frag_1), "+f"(c_frag_2), "+f"(c_frag_3)
                    : "r"(a_reg_0), "r"(a_reg_1), "r"(a_reg_2), "r"(a_reg_3),
                      "r"(b_reg_0), "r"(b_reg_1)
                );
            }
        }

        // Synchronize before loading the next K‑tile
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Phase 3: Scatter the accumulated results back to global memory
    //
    // After processing all K tiles, each lane owns a 2×2 sub‑tile of the 16×8
    // result.  We only need the first column (GEMV), so lanes with
    // col_in_quad==0 store their two row results.  Warps beyond
    // warps_per_tile remain idle.
    if (warp_id < warps_per_tile) {
        int row_offset = warp_id * 16;
        int quad = lane_id >> 2;       // 0..7, identifies which pair of rows
        int col_in_quad = lane_id & 3; // 0..3, identifies which pair of columns
        if (col_in_quad == 0) {
            int r0 = quad;
            int global_row0 = m_cta + row_offset + r0;
            int global_row1 = global_row0 + 8;
            if (global_row0 < M) {
                D_batch[global_row0] = __float2half(c_frag_0);
            }
            if (global_row1 < M) {
                D_batch[global_row1] = __float2half(c_frag_2);
            }
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
                "-Xcudafe",
                "--diag_suppress=20012",
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

    # ========================================================================
    # DEBUG: Print shapes BEFORE permute
    # ========================================================================
    print("=" * 80)
    print("BEFORE PERMUTE:")
    print(f"  A shape: {a.shape}, dtype: {a.dtype}")
    print(f"  B shape: {b.shape}, dtype: {b.dtype}")
    print(f"  C shape: {c.shape}, dtype: {c.dtype}")
    print(f"  SFA shape: {sfa_ref_cpu.shape}, dtype: {sfa_ref_cpu.dtype}")
    print(f"  SFB shape: {sfb_ref_cpu.shape}, dtype: {sfb_ref_cpu.dtype}")
    print(f"  M={M}, K={K}, L={L}")
    print("=" * 80)

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

    # ========================================================================
    # DEBUG: Print shapes AFTER permute (BEFORE kernel launch)
    # ========================================================================
    print("AFTER PERMUTE (BEFORE KERNEL):")
    print(f"  a_bytes shape: {a_bytes.shape}, dtype: {a_bytes.dtype}")
    print(f"  b_bytes shape: {b_bytes.shape}, dtype: {b_bytes.dtype}")
    print(f"  c shape: {c.shape}, dtype: {c.dtype}")
    print(f"  sfa_bytes shape: {sfa_bytes.shape}, dtype: {sfa_bytes.dtype}")
    print(f"  sfb_bytes shape: {sfb_bytes.shape}, dtype: {sfb_bytes.dtype}")
    print(f"  Kernel params: M={M}, K={K}, L={L}")
    print("=" * 80)

    # Launch SM100 tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # ========================================================================
    # DEBUG: Print shapes AFTER kernel
    # ========================================================================
    print("AFTER KERNEL (BEFORE PERMUTE BACK):")
    print(f"  c shape: {c.shape}, dtype: {c.dtype}")
    print("=" * 80)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()  # [M, 1, L]

    # ========================================================================
    # CRITICAL FIX: Squeeze out the dummy middle dimension
    # ========================================================================
    # GEMV output should be [M, L] not [M, 1, L]
    # For L=1: [M, 1, 1] → [M, 1] → [M]
    # For L>1: [M, 1, L] → [M, L]
    c = c.squeeze(1)  # Remove dimension 1 (the dummy "1" column dimension)

    # ========================================================================
    # DEBUG: Print final output shape
    # ========================================================================
    print("FINAL OUTPUT (AFTER PERMUTE BACK + SQUEEZE):")
    print(f"  c shape: {c.shape}, dtype: {c.dtype}")
    print("=" * 80)

    return c
