import os

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
# No longer need to_blocked - using sfa_permuted (atom-tiled) directly

def _u8_strided_view(t: torch.Tensor) -> torch.Tensor:
    st = t.untyped_storage()
    off = t.storage_offset()
    out = torch.empty((0,), device=t.device, dtype=torch.uint8)
    out.set_(st, off, t.size(), t.stride())
    return out

cutlass_path = os.environ.get("CUTLASS_PATH", "/usr/local/cutlass")

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include <cute/arch/copy_sm100.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

using cutlass::half_t;

// ===== HELPER FUNCTIONS =====

inline void check_cuda(cudaError_t code, const char* msg) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + msg + " - " + cudaGetErrorString(code));
    }
}

CUresult encode_tma_matrix(
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    const void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim,
    CUtensorMapSwizzle swizzle
) {
    CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    cuuint32_t elementStrides[5];
    for (cuuint32_t i = 0; i < tensorRank && i < 5; ++i) {
        elementStrides[i] = 1;
    }

    return cuTensorMapEncodeTiled(
        tensorMap,
        tensorDataType,
        tensorRank,
        const_cast<void*>(globalAddress),
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,  // elementStrides (default)
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

// ===== END HELPER FUNCTIONS =====

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr32;
    asm volatile(
        "{\n"
        "  .reg .u64 addr64;\n"
        "  cvta.to.shared.u64 addr64, %1;\n"
        "  cvt.u32.u64 %0, addr64;\n"
        "}\n"
        : "=r"(addr32)
        : "l"(ptr)
    );
    return addr32;
}

// tcgen05 mainloop only.

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar) {
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n"
        :
        : "r"(mbar_addr), "r"(1)
    );
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_addr), "r"(bytes)
    );
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase) {
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    asm volatile(
        "{\n"
        "  .reg .pred P;\n"
        "WAIT:\n"
        "  mbarrier.try_wait.parity.shared.b64 P, [%0], %1, 0;\n"
        "  @!P bra.uni WAIT;\n"
        "}\n"
        :
        : "r"(mbar_addr), "r"(phase)
    );
}

#if __CUDA_ARCH__ >= 1000
__device__ __forceinline__ void tcgen05_commit_mbarrier(uint64_t* mbar) {
    uint32_t bar_intptr = cvta_to_shared_u32(mbar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        :
        : "r"(bar_intptr)
        : "memory"
    );
}
#endif

// TMA load for CTA (rank-2, L=1) - no cluster
__device__ __forceinline__ void tma_load_2d_cta_no_arrive(void* smem_ptr,
                                                           const CUtensorMap* desc,
                                                           uint32_t coord0,
                                                           uint32_t coord1,
                                                           uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(coord0),
          "r"(coord1)
        : "memory"
    );
}

// TMA load for CTA (rank-4)
__device__ __forceinline__ void tma_load_4d_cta_no_arrive(void* smem_ptr,
                                                           const CUtensorMap* desc,
                                                           uint32_t coord0,
                                                           uint32_t coord1,
                                                           uint32_t coord2,
                                                           uint32_t coord3,
                                                           uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);

    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5, %6}], [%2];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(coord0),
          "r"(coord1),
          "r"(coord2),
          "r"(coord3)
        : "memory"
    );
}

// TMA load for CTA (rank-5) - used for permuted scale-factor views
__device__ __forceinline__ void tma_load_5d_cta_no_arrive(void* smem_ptr,
                                                           const CUtensorMap* desc,
                                                           uint32_t coord0,
                                                           uint32_t coord1,
                                                           uint32_t coord2,
                                                           uint32_t coord3,
                                                           uint32_t coord4,
                                                           uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);

    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(coord0),
          "r"(coord1),
          "r"(coord2),
          "r"(coord3),
          "r"(coord4)
        : "memory"
    );
}

// TMA load for 1D tensor (rank-1) - for pre-permuted scale factors
__device__ __forceinline__ void tma_load_1d_cta_no_arrive(void* smem_ptr,
                                                           const CUtensorMap* desc,
                                                           uint32_t offset,
                                                           uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);

    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3}], [%2];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(offset)
        : "memory"
    );
}

// Simplified - L=1 only (no cluster)
__device__ __forceinline__ void tma_load_2d_no_arrive(void* smem_ptr,
                                                       const CUtensorMap* desc,
                                                       uint32_t coord0,
                                                       uint32_t coord1,
                                                       uint64_t* mbar,
                                                       int L) {
    // L=1 always for this competition
    tma_load_2d_cta_no_arrive(smem_ptr, desc, coord0, coord1, mbar);
}


__device__ __forceinline__ void tma_load_2d(void* smem_ptr,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t coord1,
                                            uint32_t bytes,
                                            uint64_t* mbar,
                                            int L) {
    mbarrier_arrive_expect_tx(mbar, bytes);
    tma_load_2d_no_arrive(smem_ptr, desc, coord0, coord1, mbar, L);
}

__device__ __forceinline__ uint64_t* mbar_stage(uint64_t* base, int stage) {
    // Each mbarrier occupies 16 bytes (two uint64_t slots)
    return base + stage * 2;
}

// ===== tcgen05.mma HELPER FUNCTIONS =====

// SMEM Descriptor for tcgen05.mma (matches CuTe UMMA::SmemDescriptor)
// Bitfield layout (64 bits):
//   [0:14)   - start_address >> 4 (4 LSB not included)
//   [16:30)  - leading_byte_offset >> 4 
//   [32:46)  - stride_byte_offset >> 4
//   [48:49)  - version (0)
//   [49:52)  - base_offset
//   [52:53)  - lbo_mode
//   [61:64)  - layout_type (swizzle): 0=NONE, 1=128B_BASE32B, 2=128B, 4=64B, 6=32B
__device__ __forceinline__ uint64_t make_smem_desc_tcgen05(
    const void* smem_ptr,
    int leading_byte_offset,   // Stride between consecutive M/N rows (in bytes)
    int stride_byte_offset,    // Stride for K dimension (usually 0 for contiguous K)
    int swizzle_type           // 0=NONE, 2=SWIZZLE_128B, 4=SWIZZLE_64B, 6=SWIZZLE_32B
) {
    // Get 32-bit shared memory address
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);

    // Build descriptor
    uint64_t desc = 0;
    // start_address: bits [0:14), value is smem_addr >> 4
    desc |= ((uint64_t)(smem_addr >> 4) & 0x3FFF);
    // leading_byte_offset: bits [16:30), value is leading_byte_offset >> 4
    desc |= ((uint64_t)((leading_byte_offset >> 4) & 0x3FFF) << 16);
    // stride_byte_offset: bits [32:46), value is stride_byte_offset >> 4
    desc |= ((uint64_t)((stride_byte_offset >> 4) & 0x3FFF) << 32);
    // layout_type (swizzle): bits [61:64)
    desc |= ((uint64_t)(swizzle_type & 0x7) << 61);

    return desc;
}

__device__ __forceinline__ uint64_t make_smem_desc_tcgen05_addr(
    uint32_t smem_addr,
    int leading_byte_offset,
    int stride_byte_offset,
    int swizzle_type
) {
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_addr >> 4) & 0x3FFF);
    desc |= ((uint64_t)((leading_byte_offset >> 4) & 0x3FFF) << 16);
    desc |= ((uint64_t)((stride_byte_offset >> 4) & 0x3FFF) << 32);
    desc |= ((uint64_t)(swizzle_type & 0x7) << 61);
    return desc;
}

// Build a SM100 UMMA-compatible SMEM descriptor (same 64b encoding as CUTLASS/CuTe UMMA::SmemDescriptor).
// NOTE: tcgen05.cp and tcgen05.mma both consume this descriptor encoding on SM100.
__device__ __forceinline__ uint64_t make_umma_smem_desc_addr(
    uint32_t smem_addr,
    int leading_byte_offset,
    int stride_byte_offset,
    int swizzle_type
) {
    cute::UMMA::SmemDescriptor desc;
    desc.desc_ = 0;
    desc.start_address_ = static_cast<uint16_t>(smem_addr >> 4);
    desc.leading_byte_offset_ = static_cast<uint16_t>((leading_byte_offset >> 4) & 0x3FFF);
    desc.stride_byte_offset_ = static_cast<uint16_t>((stride_byte_offset >> 4) & 0x3FFF);
    desc.version_ = 1;     // Blackwell requires version=1
    desc.base_offset_ = 0;
    desc.lbo_mode_ = 0;
    desc.layout_type_ = static_cast<uint8_t>(swizzle_type & 0x7);
    return static_cast<uint64_t>(desc);
}

// Instruction Descriptor for tcgen05.mma.kind::mxf4nvf4.block_scale
// For NVFP4 (e2m1) with FP8 (e4m3) block scales
// InstrDescriptorBlockScaled bitfield (32 bits, matches CuTe exactly):
//   [0:2)   - sparse_id2 (0 for non-sparse)
//   [2:3)   - sparse_flag (0=dense)
//   [3:4)   - reserved
//   [4:6)   - b_sf_id (derived from TMEM address high bits)
//   [6:7)   - reserved
//   [7:10)  - a_format (5=E2M1 for NVFP4)
//   [10:13) - b_format (5=E2M1 for NVFP4)
//   [13:14) - a_negate (0)
//   [14:15) - b_negate (0)
//   [15:16) - a_major (0=K-major)
//   [16:17) - b_major (0=K-major)
//   [17:23) - n_dim (N >> 3)
//   [23:24) - scale_format (0=E4M3)
//   [24:29) - m_dim (M >> 4)
//   [29:31) - a_sf_id (derived from TMEM address high bits)
//   [31:32) - k_size (0 for dense K64 on MXF4)
__device__ __forceinline__ uint64_t make_instr_desc_mxf4(
    int tile_m,          // Must be 128 for SM100
    int tile_n,          // 8-256, multiple of 8
    int a_major,         // 0=K-major, 1=MN-major
    int b_major,         // 0=K-major, 1=MN-major
    int sf_format,       // 0=E4M3, 1=E8M0
    uint32_t tmem_sfa,       // Used to derive SF ID bits
    uint32_t tmem_sfb        // Used to derive SF ID bits
) {
    uint32_t desc = 0;
    
    // Format value for E2M1 (NVFP4) is 5
    constexpr uint32_t E2M1_FORMAT = 5;
    
    // [0:2) sparse_id2 = 0
    // [2:3) sparse_flag = 0
    // [3:4) reserved
    // [4:6) b_sf_id = 0 (always 0 for single CTA)
    // [6:7) reserved
    // SF IDs are encoded from the top bits of the TMEM addresses (CuTe convention).
    // These are not the TMEM base addresses themselves; they select which SF region
    // the instruction uses.
    uint32_t a_sf_id = (tmem_sfa & 0xC0000000u) >> 30;
    uint32_t b_sf_id = (tmem_sfb & 0xC0000000u) >> 30;
    // [4:6) b_sf_id
    desc |= ((b_sf_id & 0x3u) << 4);

    // [7:10) a_format = 5 (E2M1)
    desc |= (E2M1_FORMAT << 7);
    // [10:13) b_format = 5 (E2M1)
    desc |= (E2M1_FORMAT << 10);
    // [13:14) a_negate = 0
    // [14:15) b_negate = 0
    // [15:16) a_major
    desc |= ((a_major & 0x1) << 15);
    // [16:17) b_major
    desc |= ((b_major & 0x1) << 16);
    // [17:23) n_dim = tile_n >> 3
    desc |= (((tile_n >> 3) & 0x3F) << 17);
    // [23:24) scale_format
    desc |= ((sf_format & 0x1) << 23);
    // [24:29) m_dim = tile_m >> 4
    desc |= (((tile_m >> 4) & 0x1F) << 24);
    // [29:31) a_sf_id
    desc |= ((a_sf_id & 0x3u) << 29);
    // [31:32) k_size = 0 (K64 for MXF4 dense)
    
    // idescE: upper 32 bits are the instruction descriptor
    return ((uint64_t)desc << 32);
}

// Prefetch tile using TMA - simplified for Rank-2 (L=1) GEMM
template<int TileM, int TileK>
__device__ __forceinline__ void prefetch_tile(
    int stage, int k_tile_base,
    bool use_tma_a, bool is_producer, int warp_id, int lane_id,
    int m_tile, int n_tile, int K_packed, int M, int N,
    uint8_t** a_packed_stage, uint8_t** b_packed_stage, uint8_t** sfa_stage, uint8_t** sfb_stage,
    uint64_t* mbar,
    const CUtensorMap* desc_A, const CUtensorMap* desc_B,
    const CUtensorMap* desc_SFA, const CUtensorMap* desc_SFB
) {
    constexpr int TileKPacked = TileK / 2;
    if (use_tma_a && is_producer) {
        if (warp_id == 0 && lane_id == 0) {
            // Element-space coordinates
            uint32_t c_m = static_cast<uint32_t>(m_tile);
            uint32_t c_n = static_cast<uint32_t>(n_tile);
            uint32_t c_k_packed = static_cast<uint32_t>(k_tile_base >> 1);
            // Relaxed guards: Allow partial tiles
            // TMA handles OOB by zero-filling (if configured) or we rely on padding.
            // Since we padded the tensors in Python, we can safely load full tiles.
            // Just check if the start of the tile is within bounds.
            bool valid_m = (c_m < M);
            bool valid_k = (c_k_packed < K_packed);

            // --- TMA Load A (M x K) ---
            uint32_t c0_a = c_k_packed;
            uint32_t c1_a = c_m;
            // The original valid_k and valid_m checks were:
            // bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);
            // bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);

            // --- TMA Load SFA (atom-tiled layout: 2048 bytes per K-tile) ---
            // Atom-tiled physical: (mm32=32, mm4=4, rest_m, kk4=4, rest_k)
            // Reason about it as (rest_m, rest_k, 32, 4, 4) where (32,4,4) is the
            // natural contiguous 512B scale panel.
            //
            // For TMA encoding, the fastest-changing (minor) box dimension must be
            // at least 16B-wide, so we use a packed16 = (mm4*4 + kk4) dimension:
            // box = (packed16=16B, mm32=32, 1, 1) => 16*32 = 512 bytes.
            int m_block_sfa = c_m / TileM;
            int k_tile_idx_sfa = k_tile_base / TileK;
            bool valid_sfa = valid_m;

            if (valid_m && valid_k) {
                tma_load_2d_cta_no_arrive(
                    a_packed_stage[stage], desc_A, c0_a, c1_a, mbar_stage(mbar, stage)
                );
            }
            if (valid_sfa) {
                // SFA comes in as the permuted physical layout (32, 4, rest_m, 4, rest_k).
                // The TMA descriptor exposes the natural contiguous 512B panel as (packed16, mm32).
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int rest_k_idx = k_tile_idx_sfa * 4 + t;
                    tma_load_4d_cta_no_arrive(
                        sfa_stage[stage] + t * 512,
                        desc_SFA,
                        0, 0,
                        static_cast<uint32_t>(m_block_sfa),
                        static_cast<uint32_t>(rest_k_idx),
                        mbar_stage(mbar, stage)
                    );
                }
            }

            // --- TMA Load B (N x K) ---
            // B is N x K. TMA dims: [K_packed, N].
            uint32_t c0_b = c_k_packed;
            uint32_t c1_b = c_n;
            // Relaxed guard for N
            bool valid_n = (c_n < N);

            // --- TMA Load SFB (atom-tiled layout: 2048 bytes per K-tile) ---
            int n_block_sfb = c_n / TileM;  // TileN = TileM = 128
            int k_tile_idx_sfb = k_tile_base / TileK;
            bool valid_sfb = valid_n;

            if (valid_n && valid_k) {
                tma_load_2d_cta_no_arrive(
                    b_packed_stage[stage], desc_B, c0_b, c1_b, mbar_stage(mbar, stage)
                );
            }
            if (valid_sfb) {
                // SFB comes in as the permuted physical layout (32, 4, rest_n, 4, rest_k).
                // Use the same packed16×mm32 512B box as SFA.
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int rest_k_idx = k_tile_idx_sfb * 4 + t;
                    tma_load_4d_cta_no_arrive(
                        sfb_stage[stage] + t * 512,
                        desc_SFB,
                        0, 0,
                        static_cast<uint32_t>(n_block_sfb),
                        static_cast<uint32_t>(rest_k_idx),
                        mbar_stage(mbar, stage)
                    );
                }
            }

            // Arrive after issuing all bulk tensor copies (A+SFA+B+SFB).
            uint32_t bytes_total = 0;
            if (valid_m && valid_k) bytes_total += TileM * TileKPacked;
            if (valid_sfa) bytes_total += 2048;  // SF_TILE_BYTES
            if (valid_n && valid_k) bytes_total += TileM * TileKPacked; // TileN=TileM=128
            if (valid_sfb) bytes_total += 2048;  // SF_TILE_BYTES
            mbarrier_arrive_expect_tx(mbar_stage(mbar, stage), bytes_total);
        }
    }
}
#endif

// RANK2 CTA + SWIZZLE_128B + BOX_K 128 BYTE (NO CLUSTER)
template<int TileM, int TileK, int Threads>
__global__ void __launch_bounds__(Threads)
fp4_gemm_rank2_cta(
    const CUtensorMap* __restrict__ desc_A,
    const CUtensorMap* __restrict__ desc_SFA,
    const CUtensorMap* __restrict__ desc_B,
    const CUtensorMap* __restrict__ desc_SFB,
    half* __restrict__ D,
    const int M, const int N, const int K, const int L, const int K_scales
) {
#if __CUDA_ARCH__ >= 900
    constexpr int TileKPacked = TileK / 2;
    constexpr int TileN = 128; // Fixed TileN matching TileM
    constexpr int StageCount = 2; 

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const bool is_producer = warp_id == 0; // Warp 0 issues TMA

    // Grid: x=M_tile, y=N_tile, z=Batch(L)
    const int batch = blockIdx.z;
    const int m_tile = blockIdx.x * TileM;
    const int n_tile = blockIdx.y * TileN;
    
    if (batch >= L || m_tile >= M || n_tile >= N) return;

    const int K_packed = K >> 1;

    extern __shared__ uint8_t smem[];

    auto align_up = [] __device__(size_t x, size_t align) {
        return (x + align - 1) & ~(align - 1);
    };

    size_t offset = 0;

    auto alloc_mbar_array = [&] __device__(int count) -> uint64_t* {
        offset = align_up(offset, 16);
        uint8_t* base = smem + offset;
        offset += static_cast<size_t>(count) * 2 * sizeof(uint64_t);
        return reinterpret_cast<uint64_t*>(base);
    };

    // Helper to align to 1024 bytes (required for SWIZZLE_128B TMA)
    auto align_up_smem_1024 = [&] __device__() {
        uint32_t addr = cvta_to_shared_u32(smem + offset);
        uint32_t aligned = (addr + 1023u) & ~1023u;
        offset += static_cast<size_t>(aligned - addr);
    };
    

    uint64_t* mbar = alloc_mbar_array(StageCount);
    uint64_t* mbar_cp = alloc_mbar_array(StageCount);
    uint64_t* mbar_mma = alloc_mbar_array(StageCount);

    // TMA destinations (SWIZZLE_128B -> 1024 byte alignment)
    align_up_smem_1024();
    uint8_t* a_packed_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        a_packed_stage[s] = smem + offset;
        offset += TileM * TileKPacked;
    }

    align_up_smem_1024();
    uint8_t* b_packed_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        b_packed_stage[s] = smem + offset;
        offset += TileN * TileKPacked;
    }

    align_up_smem_1024();
    // SFA/SFB: Pre-permuted atom-tiled layout - 2048 bytes per M-tile (128 rows × 16 K-scales)
    // For tcgen05 path: loaded directly in blocked format, no permutation needed
    constexpr int SF_TILE_BYTES = 2048;  // 128 * 16 = 2048 bytes per tile
    uint8_t* sfa_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfa_stage[s] = smem + offset;
        offset += SF_TILE_BYTES;
    }
    
    align_up_smem_1024();
    uint8_t* sfb_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfb_stage[s] = smem + offset;
        offset += SF_TILE_BYTES;
    }

    (void)offset;

    const bool use_tma_a = (desc_A != nullptr);

    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            mbarrier_init(mbar_stage(mbar, s));
            mbarrier_init(mbar_stage(mbar_cp, s));
            mbarrier_init(mbar_stage(mbar_mma, s));
        }
        __threadfence_block();
    }
    __syncthreads();

    // ========================================================================
    // tcgen05.mma.kind::mxf4nvf4.block_scale MAINLOOP for NVFP4 GEMM
    // ========================================================================
    // This path uses Blackwell's native FP4 tensor cores:
    // - Consumes packed FP4 (e2m1) data directly from SMEM
    // - Applies FP8 (e4m3) block scales from TMEM
    // - Fuses decode + scale + MMA in hardware
    // - No manual FP4->FP16 decode needed!

    // tcgen05.alloc writes one b32 result per warp in the warpgroup; provide
    // 16B-aligned storage for 4 warps (Threads=128 in tcgen05 path).
    __shared__ __align__(16) uint32_t tmem_base_ptr_tcgen05[4];

    // Prologue: prefetch stages 0..StageCount-2 (same as classic path)
    for (int s = 0; s < StageCount - 1; ++s) {
        int k_tile = s * TileK;
        if (k_tile < K) {
            prefetch_tile<TileM, TileK>(
                s, k_tile, use_tma_a, is_producer, warp_id, lane_id,
                m_tile, n_tile, K_packed, M, N,
                a_packed_stage, b_packed_stage, sfa_stage, sfb_stage,
                mbar,
                desc_A, desc_B, desc_SFA, desc_SFB
            );
        }
    }

    // Allocate TMEM once per CTA (b32 columns)
    // PTX: nCols must be power-of-2, in [32,512]. Allocate full 512 columns.
    constexpr uint32_t TMEM_COLS_TOTAL = 512u;
    __syncthreads();
    if (tid < 4) {
        tmem_base_ptr_tcgen05[tid] = 0;
    }
    __syncthreads();
    
    uint32_t dst_smem = cvta_to_shared_u32(tmem_base_ptr_tcgen05);
    uint32_t num_cols = TMEM_COLS_TOTAL;
    // PTX: allocation must be performed by a single warp in the CTA.
    if (warp_id == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                    :
                    : "r"(dst_smem), "r"(num_cols)
                    : "memory");
    }
    __syncthreads();
    // Use a CTA-uniform base pointer; alloc writes the same base for each warp.
    uint32_t tmem_c = tmem_base_ptr_tcgen05[0];

    // Place SFA/SFB in TMEM using the same column-offset rules as CUTLASS.
    // This avoids hardcoding "256" and avoids assuming linear column packing.
    #if __CUDA_ARCH__ >= 1000
    using namespace cute;
    using MmaOp = SM100_MMA_MXF4_SS<
        cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
        TileM, TileN, 16, UMMA::Major::K, UMMA::Major::K
    >;
    auto tiled_mma = make_tiled_mma(MmaOp{});

    // CTA-wide accumulator tensor in TMEM (MMA-partitioned), matching CUTLASS' partition_shape_C flow.
    // NOTE: Do NOT use partition_fragment_C here; that yields a per-thread fragment and breaks epilogue coverage.
    auto acc_shape = partition_shape_C(tiled_mma, make_shape(Int<TileM>{}, Int<TileN>{}));  // ((MMA_TILE_M,MMA_TILE_N), MMA_M, MMA_N)
    auto tCtAcc = make_tensor<typename MMA_Traits<MmaOp>::FrgTypeC>(acc_shape);
    tCtAcc.data() = make_tmem_ptr<float>(tmem_c);

    constexpr int kKBlock = 64;
    static_assert(TileK % kKBlock == 0, "TileK must be divisible by kKBlock");
    constexpr int kNumKBlocks = TileK / kKBlock;

    // tmem_sf_frg expects ((MMA_MN, (VecSize, NSF)), num_MMA_MN, num_MMA_K)
    constexpr int kSfVecSize = 16;
    constexpr int kSfNSF = kKBlock / kSfVecSize;  // 4 for K=64, vec=16
    static_assert(kSfVecSize * kSfNSF == kKBlock, "Scale-factor NSF mismatch");
    auto tmem_shape_sf = make_shape(
        make_shape(Int<TileM>{}, make_shape(Int<kSfVecSize>{}, Int<kSfNSF>{})),
        Int<1>{},
        Int<kNumKBlocks>{}
    );

    auto tCtSFA = make_tensor<typename MMA_Traits<MmaOp>::FrgTypeSFA>(tmem_shape_sf);
    auto tCtSFB = make_tensor<typename MMA_Traits<MmaOp>::FrgTypeSFB>(tmem_shape_sf);

    uint32_t tmem_sfa_base = tmem_c + cutlass::detail::find_tmem_tensor_col_offset(tCtAcc);
    tCtSFA.data() = make_tmem_ptr<cutlass::float_ue4m3_t>(tmem_sfa_base);
    uint32_t tmem_sfb_base = tmem_sfa_base + cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);
    tCtSFB.data() = make_tmem_ptr<cutlass::float_ue4m3_t>(tmem_sfb_base);
    #endif

    // NOTE: TMEM accumulator initialization is handled by scaleC=0 on the first MMA.
    // tcgen05.mma with scaleC=0 (predicate p=false) does D = A*B, ignoring existing C.

    #if __CUDA_ARCH__ >= 1000
    // Build instruction descriptor (constant for all K-tiles)
    // sf_format = 0 for E4M3 scale format
    // a_sf_id and b_sf_id are set to 0 (default ID, not derived from addresses)
    uint64_t idescE = make_instr_desc_mxf4(
        TileM, TileN,
        0, 0,       // K-major for both A and B
        0,          // sf_format = E4M3
        tmem_sfa_base, tmem_sfb_base
    );
    uint32_t idescE_hi = uint32_t(idescE >> 32);
    #endif

    int num_k_tiles = (K + TileK - 1) / TileK;
    int last_tile_iter = num_k_tiles - 1;
    int last_stage = last_tile_iter % StageCount;
    uint32_t last_phase = uint32_t((last_tile_iter / StageCount) & 1);
    int prev_tile_iter = last_tile_iter - 1;
    int prev_stage = prev_tile_iter % StageCount;
    uint32_t prev_phase = uint32_t((prev_tile_iter / StageCount) & 1);

    // Main K-loop
    for (int k_tile = 0; k_tile < K; k_tile += TileK) {
        int tile_iter = (k_tile / TileK);
        int stage = tile_iter % StageCount;
        uint32_t phase = uint32_t((tile_iter / StageCount) & 1);
        int next_k = k_tile + (StageCount - 1) * TileK;
        
        // Issue TMA prefetch for next tile
        if (next_k < K) {
            // Wait for tcgen05.mma to finish consuming the stage we are about to overwrite.
            int write_stage = (stage + StageCount - 1) % StageCount;
            if (warp_id == 0 && lane_id == 0 && tile_iter > 0) {
                int prev_iter_for_write = tile_iter - 1;
                uint32_t wait_phase = uint32_t((prev_iter_for_write / StageCount) & 1);
                mbarrier_wait_parity(mbar_stage(mbar_mma, write_stage), wait_phase);
            }
            prefetch_tile<TileM, TileK>(
                (stage + StageCount - 1) % StageCount, next_k, use_tma_a, is_producer, warp_id, lane_id,
                m_tile, n_tile, K_packed, M, N,
                a_packed_stage, b_packed_stage, sfa_stage, sfb_stage,
                mbar,
                desc_A, desc_B, desc_SFA, desc_SFB
            );
        }

        // Wait for current TMA to complete (reduce spin to a single warp)
        if (warp_id == 0) {
            mbarrier_wait_parity(mbar_stage(mbar, stage), phase);
        }
        __syncthreads();

        #if __CUDA_ARCH__ >= 1000
        // =========================================================================
        // Scale SMEM -> TMEM (Cp4x32x128b): one copy per K-block (K=64)
        // =========================================================================
        // Each K-block holds 32 rows × 16B/row = 512B of FP8 scale data in SMEM.
        // sfa_stage/sfb_stage are laid out as 4 contiguous 512B chunks (one per K-block).
        //
        // NOTE: Match kutte.py oracle issue scope: issue tcgen05.cp from a single warp
        // (warp 0) and a single lane (lane 0). This avoids redundant copies from
        // multiple warps in the CTA that can corrupt TMEM scale panels.
        uint32_t sfa_stage_smem = cvta_to_shared_u32(sfa_stage[stage]);
        uint32_t sfb_stage_smem = cvta_to_shared_u32(sfb_stage[stage]);
        if (warp_id < 4) {
            // tcgen05.cp ... warpx4 expects warpgroup participation.
            #pragma unroll
            for (int kb = 0; kb < kNumKBlocks; ++kb) {
                uint64_t desc_sfa = make_umma_smem_desc_addr(
                    sfa_stage_smem + uint32_t(kb) * 512u,
                    16, 0, 0
                );
                uint32_t sfa_dst = raw_pointer_cast(tCtSFA(_, _, kb).data());
                asm volatile(
                    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
                    :: "r"(sfa_dst), "l"(desc_sfa)
                    : "memory"
                );

                uint64_t desc_sfb = make_umma_smem_desc_addr(
                    sfb_stage_smem + uint32_t(kb) * 512u,
                    16, 0, 0
                );
                uint32_t sfb_dst = raw_pointer_cast(tCtSFB(_, _, kb).data());
                asm volatile(
                    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
                    :: "r"(sfb_dst), "l"(desc_sfb)
                    : "memory"
                );
            }
        }
        if (warp_id == 0 && lane_id == 0) {
            tcgen05_commit_mbarrier(mbar_stage(mbar_cp, stage));
        }

        // Ensure scale-panel TMEM stores are visible before tcgen05.mma reads them.
        // Use all warps in the warpgroup to avoid relying on lane-level execution for fences.
        if (warp_id < 4) {
            asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
        }

        __syncthreads();

        // =========================================================================
        // MMA: one tcgen05.mma per K-block (K=64), with ACCUMULATE disabled only
        // for the first K-block of the full GEMM.
        // =========================================================================
        using namespace cute;
        using ElementAB = cutlass::float_e2m1_t;
        using SmemLayoutAtomAB = UMMA::Layout_K_SW128_Atom<ElementAB>;
        auto smem_layout_a = tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<TileM>{}, Int<TileK>{}));
        auto smem_layout_b = tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<TileN>{}, Int<TileK>{}));

        auto sA_full = make_tensor(make_smem_ptr<ElementAB>(a_packed_stage[stage]), smem_layout_a);
        auto sB_full = make_tensor(make_smem_ptr<ElementAB>(b_packed_stage[stage]), smem_layout_b);

        auto mma_kb = [&](int kb, bool accum) {
            // Derive per-K-block SMEM descriptors from canonical UMMA layouts (no raw pointer offsets).
            // Each K-block is 64 elements in K (i.e. 32 bytes packed).
            auto sA_kb = local_tile(
                sA_full,
                make_shape(Int<TileM>{}, Int<kKBlock>{}),
                make_coord(Int<0>{}, kb)
            );
            auto sB_kb = local_tile(
                sB_full,
                make_shape(Int<TileN>{}, Int<kKBlock>{}),
                make_coord(Int<0>{}, kb)
            );
            uint64_t desc_a_smem = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(sA_kb));
            uint64_t desc_b_smem = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(sB_kb));

            uint32_t tmem_sfa_kb = raw_pointer_cast(tCtSFA(_, _, kb).data());
            uint32_t tmem_sfb_kb = raw_pointer_cast(tCtSFB(_, _, kb).data());

            // NOTE: tcgen05.mma expects warpgroup participation.
            if (warp_id < 4) {
                if (accum) {
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, 1, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                        "}\n"
                        :
                        : "r"(tmem_c), "l"(desc_a_smem), "l"(desc_b_smem),
                          "r"(idescE_hi), "r"(tmem_sfa_kb), "r"(tmem_sfb_kb)
                        : "memory"
                    );
                } else {
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, 0, 0;\n\t"
                        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                        "}\n"
                        :
                        : "r"(tmem_c), "l"(desc_a_smem), "l"(desc_b_smem),
                          "r"(idescE_hi), "r"(tmem_sfa_kb), "r"(tmem_sfb_kb)
                        : "memory"
                    );
                }
            }
        };

        if (k_tile == 0) {
            mma_kb(0, false);
            #pragma unroll
            for (int kb = 1; kb < kNumKBlocks; ++kb) {
                mma_kb(kb, true);
            }
        } else {
            #pragma unroll
            for (int kb = 0; kb < kNumKBlocks; ++kb) {
                mma_kb(kb, true);
            }
        }

        // Signal completion of in-flight tcgen05.mma operations for this stage.
        if (warp_id == 0 && lane_id == 0) {
            tcgen05_commit_mbarrier(mbar_stage(mbar_mma, stage));
        }
        #endif  // __CUDA_ARCH__ >= 1000

        __syncthreads();
    }

    // Ensure all in-flight tcgen05.mma operations are complete before reading TMEM.
    #if __CUDA_ARCH__ >= 1000
    if (warp_id == 0 && num_k_tiles > 0) {
        mbarrier_wait_parity(mbar_stage(mbar_mma, last_stage), last_phase);
        if (num_k_tiles > 1) {
            mbarrier_wait_parity(mbar_stage(mbar_mma, prev_stage), prev_phase);
        }
    }
    __syncthreads();
    #endif

    // ========================================================================
    // EPILOGUE: TMEM -> register -> global D
    // ========================================================================
    {
        // Correctness-first TMEM epilogue:
        // Use CuTe's TMEM->register copy partitioning so we don't guess TMEM address math.
        using namespace cute;
        using X = Underscore;

        Tensor mD = make_tensor(make_gmem_ptr<cutlass::half_t>(reinterpret_cast<cutlass::half_t*>(D)),
                                make_shape(M, N),
                                make_stride(N, 1));
        Tensor gD = local_tile(mD, make_shape(Int<TileM>{}, Int<TileN>{}), make_coord(blockIdx.x, blockIdx.y));

        auto tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b1x{}, tensor<0>(tCtAcc));
        int t2r_threads = int(size(tiled_t2r));
        if (tid < t2r_threads) {
            auto thread_t2r = tiled_t2r.get_slice(tid);
            Tensor tAcc = tensor<0>(tCtAcc);
            Tensor tTR_tAcc = thread_t2r.partition_S(tAcc);
            Tensor tTR_gD = thread_t2r.partition_D(gD);
            Tensor tTR_rAcc = make_tensor<float>(shape(tTR_gD));

            copy(tiled_t2r, tTR_tAcc, tTR_rAcc);
            #if __CUDA_ARCH__ >= 1000
            asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
            #endif

            #pragma unroll
            for (int i = 0; i < size(tTR_rAcc); ++i) {
                tTR_gD(i) = cutlass::half_t(__float2half_rn(tTR_rAcc(i)));
            }
        }
    }

    // Free TMEM allocation (must be explicit before CTA exit)
    __syncthreads();  // ensure no warp still uses TMEM
    if (warp_id == 0) {
        uint32_t taddr = tmem_c;
        uint32_t ncols = TMEM_COLS_TOTAL;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                    :
                    : "r"(taddr), "r"(ncols)
                    : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
                    :
                    :
                    : "memory");
    }
#endif
}

// launch_fp4_gemm_optimized starts from here 
void launch_fp4_gemm_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t N, int64_t K, int64_t L, int64_t K_scales
) {
    const uint8_t* A_ptr = A.data_ptr<uint8_t>();
    const uint8_t* B_ptr = B.data_ptr<uint8_t>();
    const uint8_t* SFA_ptr = SFA.data_ptr<uint8_t>();
    const uint8_t* SFB_ptr = SFB.data_ptr<uint8_t>();
    half* D_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    constexpr int kTileM = 128;
    constexpr int kTileN = 128; // GEMM tiling
    constexpr int kTileK = 256;  // Must be 256 to match CuTe mma_tiler_mnk and tcgen05.mma
    constexpr int kThreads = 128; // 4 warps
    constexpr int kTileKPacked = kTileK / 2;  // 128 bytes per tile row
    constexpr int StageCount = 2;
    
    // TMA hardware constraint
    constexpr int kTMABoxLimit = 256;
    static_assert(kTileM <= kTMABoxLimit, "kTileM exceeds TMA box limit");
    static_assert(kTileN <= kTMABoxLimit, "kTileN exceeds TMA box limit");

    // --- TMA Descriptors ---
    alignas(64) CUtensorMap map_A;
    alignas(64) CUtensorMap map_B;
    alignas(64) CUtensorMap map_SFA;
    alignas(64) CUtensorMap map_SFB;
    CUtensorMap *d_map_A = nullptr, *d_map_B = nullptr, *d_map_SFA = nullptr, *d_map_SFB = nullptr;
    CUtensorMap *map_A_ptr = &map_A, *map_B_ptr = &map_B, *map_SFA_ptr = &map_SFA, *map_SFB_ptr = &map_SFB;
    bool tma_ok = true;
    int tma_fail_code = 0;
    const char* tma_fail_which = nullptr;

    // A: M x K (Rank-2)
    {
        cuuint64_t dims_A[2] = {static_cast<cuuint64_t>(K/2), static_cast<cuuint64_t>(M)};
        cuuint32_t box_A[2] = {static_cast<cuuint32_t>(kTileKPacked), static_cast<cuuint32_t>(kTileM)};
        cuuint64_t strides_A[1] = {static_cast<cuuint64_t>(K/2)};
        
        CUresult resA = encode_tma_matrix(map_A_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                     2, A_ptr, dims_A, strides_A, box_A,
                                     CU_TENSOR_MAP_SWIZZLE_128B);
        if (resA != CUDA_SUCCESS) {
             printf("ERROR: TMA A descriptor failed with code %d\n", resA);
             tma_ok = false;
             tma_fail_code = int(resA);
             tma_fail_which = "A";
        } else {
            check_cuda(cudaMalloc(&d_map_A, sizeof(CUtensorMap)), "cudaMalloc d_map_A");
            check_cuda(cudaMemcpy(d_map_A, map_A_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_A");
        }
    }

    // B: N x K (Rank-2) - GEMM
    {
        cuuint64_t dims_B[2] = {static_cast<cuuint64_t>(K/2), static_cast<cuuint64_t>(N)};
        cuuint32_t box_B[2] = {static_cast<cuuint32_t>(kTileKPacked), static_cast<cuuint32_t>(kTileN)};
        cuuint64_t strides_B[1] = {static_cast<cuuint64_t>(K/2)};
        
        CUresult resB = encode_tma_matrix(map_B_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                     2, B_ptr, dims_B, strides_B, box_B,
                                     CU_TENSOR_MAP_SWIZZLE_128B);
        if (resB != CUDA_SUCCESS) {
             printf("ERROR: TMA B descriptor failed with code %d\n", resB);
             tma_ok = false;
             tma_fail_code = int(resB);
             tma_fail_which = "B";
        } else {
            check_cuda(cudaMalloc(&d_map_B, sizeof(CUtensorMap)), "cudaMalloc d_map_B");
            check_cuda(cudaMemcpy(d_map_B, map_B_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_B");
        }
    }

    // SFA: permuted physical layout tensor passed as a rank-5 uint8 view:
    //   (mm32=32, mm4=4, rest_m, kk4=4, rest_k)
    //
    // Reason about it as (rest_m, rest_k, 32, 4, 4) where (32,4,4) is the natural
    // contiguous 512B panel, but encode TMA with a 16B-wide minor dimension:
    //   packed16 = (mm4*4 + kk4) (16 bytes), then mm32 (32 rows) => 512 bytes.
    {
        TORCH_CHECK(SFA.is_cuda(), "SFA must be CUDA");
        TORCH_CHECK(SFA.scalar_type() == torch::kUInt8, "SFA must be uint8");
        TORCH_CHECK(SFA.dim() == 5, "SFA must be rank-5");
        TORCH_CHECK(SFA.size(0) == 32 && SFA.size(1) == 4 && SFA.size(3) == 4,
                    "SFA shape must be (32,4,rest_m,4,rest_k)");
        TORCH_CHECK(SFA.stride(3) == 1, "SFA expects contiguous kk4 dimension");
        TORCH_CHECK(SFA.stride(1) == 4, "SFA expects mm4 stride 4");
        TORCH_CHECK(SFA.stride(0) == 16, "SFA expects mm32 stride 16");

        cuuint64_t dims_SFA[4] = {
            16ull,                               // packed16 (mm4*4 + kk4)
            static_cast<cuuint64_t>(SFA.size(0)), // mm32 (32)
            static_cast<cuuint64_t>(SFA.size(2)), // rest_m
            static_cast<cuuint64_t>(SFA.size(4)), // rest_k
        };
        cuuint64_t strides_SFA[3] = {
            static_cast<cuuint64_t>(SFA.stride(0)), // mm32
            static_cast<cuuint64_t>(SFA.stride(2)), // rest_m
            static_cast<cuuint64_t>(SFA.stride(4)), // rest_k
        };
        cuuint32_t box_SFA[4] = {16, 32, 1, 1};      // 16B x 32 rows = 512B

        CUresult resSFA = encode_tma_matrix(map_SFA_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                     4, const_cast<void*>(static_cast<const void*>(SFA_ptr)),
                                     dims_SFA, strides_SFA, box_SFA,
                                     CU_TENSOR_MAP_SWIZZLE_NONE);
        if (resSFA != CUDA_SUCCESS) {
             printf("ERROR: TMA SFA descriptor failed with code %d\\n", resSFA);
             tma_ok = false;
             tma_fail_code = int(resSFA);
             tma_fail_which = "SFA";
        } else {
            check_cuda(cudaMalloc(&d_map_SFA, sizeof(CUtensorMap)), "cudaMalloc d_map_SFA");
            check_cuda(cudaMemcpy(d_map_SFA, map_SFA_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFA");
        }
    }

    // SFB: permuted physical layout tensor passed as a rank-5 uint8 view.
    // Same packed16×mm32 encoding as SFA (box=16B×32).
    {
        TORCH_CHECK(SFB.is_cuda(), "SFB must be CUDA");
        TORCH_CHECK(SFB.scalar_type() == torch::kUInt8, "SFB must be uint8");
        TORCH_CHECK(SFB.dim() == 5, "SFB must be rank-5");
        TORCH_CHECK(SFB.size(0) == 32 && SFB.size(1) == 4 && SFB.size(3) == 4,
                    "SFB shape must be (32,4,rest_n,4,rest_k)");
        TORCH_CHECK(SFB.stride(3) == 1, "SFB expects contiguous kk4 dimension");
        TORCH_CHECK(SFB.stride(1) == 4, "SFB expects mm4 stride 4");
        TORCH_CHECK(SFB.stride(0) == 16, "SFB expects mm32 stride 16");

        cuuint64_t dims_SFB[4] = {
            16ull,                               // packed16 (mm4*4 + kk4)
            static_cast<cuuint64_t>(SFB.size(0)), // mm32 (32)
            static_cast<cuuint64_t>(SFB.size(2)), // rest_n
            static_cast<cuuint64_t>(SFB.size(4)), // rest_k
        };
        cuuint64_t strides_SFB[3] = {
            static_cast<cuuint64_t>(SFB.stride(0)), // mm32
            static_cast<cuuint64_t>(SFB.stride(2)), // rest_n
            static_cast<cuuint64_t>(SFB.stride(4)), // rest_k
        };
        cuuint32_t box_SFB[4] = {16, 32, 1, 1};

        CUresult resSFB = encode_tma_matrix(map_SFB_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                     4, const_cast<void*>(static_cast<const void*>(SFB_ptr)),
                                     dims_SFB, strides_SFB, box_SFB,
                                     CU_TENSOR_MAP_SWIZZLE_NONE);
        if (resSFB != CUDA_SUCCESS) {
             printf("ERROR: TMA SFB descriptor failed with code %d\\n", resSFB);
             tma_ok = false;
             tma_fail_code = int(resSFB);
             tma_fail_which = "SFB";
        } else {
            check_cuda(cudaMalloc(&d_map_SFB, sizeof(CUtensorMap)), "cudaMalloc d_map_SFB");
            check_cuda(cudaMemcpy(d_map_SFB, map_SFB_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFB");
        }
    }

    struct TensorMapScope {
        CUtensorMap* a = nullptr;
        CUtensorMap* b = nullptr;
        CUtensorMap* sfa = nullptr;
        CUtensorMap* sfb = nullptr;
        ~TensorMapScope() {
            if (a) cudaFree(a);
            if (sfa) cudaFree(sfa);
            if (b) cudaFree(b);
            if (sfb) cudaFree(sfb);
        }
    } maps;
    maps.a = d_map_A;
    maps.b = d_map_B;
    maps.sfa = d_map_SFA;
    maps.sfb = d_map_SFB;

    if (!tma_ok) {
        std::string msg = "TMA descriptor creation failed";
        if (tma_fail_which) {
            msg += " for ";
            msg += tma_fail_which;
        }
        msg += " (code ";
        msg += std::to_string(tma_fail_code);
        msg += ")";
        throw std::runtime_error(msg);
    }

    // Kernel Launch
    void const* kernel_ptr = (void const*)fp4_gemm_rank2_cta<kTileM, kTileK, kThreads>;
    
    size_t shared_bytes = 0; 
    size_t offset = 0;
    auto align = [&](size_t x, size_t a) { return (x + a - 1) & ~(a - 1); };
    
    // Mbarriers (StageCount)
    offset = align(offset, 16);
    offset += StageCount * 16; // mbar
    offset += StageCount * 16; // mbar_cp
    offset += StageCount * 16; // mbar_mma
    
    // TMA A (StageCount)
    offset = align(offset, 1024);
    offset += StageCount * kTileM * kTileKPacked;
    
    // TMA B (StageCount)
    offset = align(offset, 1024);
    offset += StageCount * kTileN * kTileKPacked;
    
    // TMA SFA (StageCount)
    offset = align(offset, 1024);
    offset += StageCount * kTileM * 128;
    
    // TMA SFB (StageCount)
    offset = align(offset, 1024);
    offset += StageCount * kTileN * 128;
    
    shared_bytes = offset;

    check_cuda(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)), "MaxDynamicSharedMemorySize");

    int grid_x = (M + kTileM - 1) / kTileM;
    int grid_y = (N + kTileN - 1) / kTileN;
    dim3 grid(grid_x, grid_y, L);
    dim3 block(kThreads);

    int M_int = static_cast<int>(M);
    int N_int = static_cast<int>(N);
    int K_int = static_cast<int>(K);
    int L_int = static_cast<int>(L);
    int K_scales_int = static_cast<int>(K_scales);

    void* kernel_args[] = {
        &d_map_A,
        &d_map_SFA,
        &d_map_B,
        &d_map_SFB,
        &D_ptr,
        &M_int,
        &N_int,
        &K_int,
        &L_int,
        &K_scales_int
    };

    check_cuda(cudaLaunchKernel(kernel_ptr, grid, block, kernel_args, shared_bytes, 0), "cudaLaunchKernel");
} 

"""

cpp_source = """
void launch_fp4_gemm_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t N, int64_t K, int64_t L, int64_t K_scales_padded
);

"""

module = None


def get_module():
    global module
    if module is None:
        module_name = "nvfp4_gemm_sm100_ptx"
        extra_cuda_cflags = [
            "-O3",  # Enable optimizations to fix lambda stack issues
            "-DNDEBUG",  # Disable debug output for performance testing
            "--use_fast_math",  # Disabled for debugging
            "--ftz=true",  # Flush denormals to zero
            "--prec-div=false",  # Faster division (less precise)
            "--prec-sqrt=false",  # Faster sqrt (less precise)
            "--fmad=true",  # Enable fused multiply-add
            "-std=c++17",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcudafe",
            "--diag_suppress=20012",
            "-maxrregcount=128",
            "--ptxas-options=-v,-warn-lmem-usage",
            "-lineinfo",
            f"-I{cutlass_path}/include",
        ]
        module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=[
                "launch_fp4_gemm_optimized",
            ],
            verbose=False,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    SM100 FP4 GEMM with tensor cores: Pure PTX implementation
    """
    # Workaround for multiprocessing setting sys.stdout to None
    import sys

    if sys.stdout is None:
        sys.stdout = open("/dev/null", "w")
    if sys.stderr is None:
        sys.stderr = open("/dev/null", "w")

    a, b, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c = data

    M, N, L = c.shape
    K = a.shape[1] * 2  # a.shape[1] is K/2 (packed)
    k_packed = K // 2
    K_scales = K // 16

    # Extract 2D slices (L=1)
    a_2d = a[:, :, 0].contiguous()
    b_2d = b[:, :, 0].contiguous()
    
    # Convert to uint8 view
    a_bytes = a_2d.view(torch.uint8)
    b_bytes = b_2d.view(torch.uint8)

    # SCALE FACTORS: Use atom-tiled layout (sfa_permuted/sfb_permuted)
    # =========================================================================  
    
    # Preserve the logical addressing of the permuted view (non-contiguous)
    sfa_bytes = _u8_strided_view(sfa_permuted[..., 0])
    sfb_bytes = _u8_strided_view(sfb_permuted[..., 0])
    
    mod = get_module()
    mod.launch_fp4_gemm_optimized(
        a_bytes, b_bytes, sfa_bytes, sfb_bytes, c[:, :, 0],
        M, N, K, L, K_scales
    )

    return c
