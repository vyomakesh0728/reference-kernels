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
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cute/arch/copy_sm100.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

using cutlass::half_t;

// Debug-only: targeted dump to localize mismatches around TMEM->reg epilogue.
// Default off; enable by compiling with -DNVFP4_DEBUG_DUMP=1.
#ifndef NVFP4_DEBUG_DUMP
#define NVFP4_DEBUG_DUMP 0
#endif

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

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t val) {
    cutlass::float_e4m3_t fp8_val = *reinterpret_cast<cutlass::float_e4m3_t*>(&val);
    return __half2float(__float2half_rn(fp8_val));
}

// Scale factors come from the permuted (atom-tiled) layout:
//   (mm32=32, mm4=4, rest_mn, kk4=4, rest_k)
// For a CTA tile (128 rows, TileK=256 => 16 scale columns), we TMA-load 2048B:
//   4 chunks × 512B, where each 512B chunk is a 32×16 panel:
//     rows = mm32 (0..31)
//     cols = packed16 = (mm4*4 + kk4) (0..15)
// This helper fetches the correct byte for a logical row (0..127) and local
// scale column (0..15) from that 2048B tile.
template<int TileK>
__device__ __forceinline__ uint8_t load_sf_tile_byte_2048(const uint8_t* __restrict__ sf_tile,
                                                          int row, int scale_col) {
    static_assert(TileK % 256 == 0 || TileK == 256, "Expected TileK=256 for current kernel");
    constexpr int SF_STRIDE = TileK / 16;  // 16 scale columns for TileK=256
    (void)SF_STRIDE;
    // row: 0..127 => mm32 in [0,31], mm4 in [0,3]
    int mm32 = row & 31;
    int mm4  = row >> 5;
    // scale_col: 0..15 => rest_k in [0,3], kk4 in [0,3]
    int rest_k = scale_col >> 2;
    int kk4    = scale_col & 3;
    int packed16 = (mm4 << 2) | kk4;
    // 512B chunk per rest_k, 16B per mm32 row
    return sf_tile[rest_k * 512 + mm32 * 16 + packed16];
}

// Hardware FP4 decode helper using cvt.rn.f16x2.e2m1x2, adapted from gau.py.
// Decodes a byte containing two FP4 values (float4_e2m1fn_x2) into a half2.
__device__ __forceinline__ void fp4x8_to_fp16x2x4(int *out, int in) {
    asm volatile(
        "{\n\t"
        ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n\t"
        "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4;\n\t" // unpack 32-bit to 4x fp4x2
        "cvt.rn.f16x2.e2m1x2 %0, tmp0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n\t"
        "}"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "r"(in)
    );
}

__device__ __forceinline__ void decode_fp4x2_hw(uint8_t packed, half &h0, half &h1) {
    // Hardware cvt.rn.f16x2.e2m1x2 decodes: low nibble first, high nibble second
    // But Python does: hi = byte >> 4, lo = byte & 0xF (hi first, lo second)
    // So we need to swap: h0 = hi nibble (>>4), h1 = lo nibble (&0xF)
    int out_i32[4];
    int in32 = static_cast<int>(packed);
    fp4x8_to_fp16x2x4(out_i32, in32);
    half2 pair = *reinterpret_cast<half2*>(&out_i32[0]);
    // Swap order to match Python: hi nibble -> h0, lo nibble -> h1
    h0 = __high2half(pair);  // was __low2half - this is the hi nibble (>>4)
    h1 = __low2half(pair);   // was __high2half - this is the lo nibble (&0xF)
}

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

#ifndef NDEBUG
// Global debug caps: overall and per-thread (flattened grid thread id, hashed).
__device__ unsigned int g_debug_error_count = 0;
__device__ unsigned int g_debug_thread_error_counts[1024];
#define DEBUG_MAX_ERRORS 64u
#define DEBUG_MAX_ERRORS_PER_THREAD 2u

#define DEBUG_PRINT_ERROR(fmt, ...)                                                        \
    do {                                                                                   \
        unsigned int _tid_flat =                                                           \
            threadIdx.x + blockDim.x *                                                     \
            (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));             \
        unsigned int _slot = _tid_flat & 1023u;                                            \
        unsigned int _t = atomicAdd(&g_debug_thread_error_counts[_slot], 1u);              \
        if (_t < DEBUG_MAX_ERRORS_PER_THREAD) {                                            \
            unsigned int _g = atomicAdd(&g_debug_error_count, 1u);                         \
            if (_g < DEBUG_MAX_ERRORS) {                                                   \
                // printf(fmt, __VA_ARGS__);                                                  \
            }                                                                              \
        }                                                                                  \
    } while (0)

#define DEBUG_OOB_GLOBAL_1D(name, idx, size, base_ptr)                                     \
    do {                                                                                   \
        long long _i = static_cast<long long>(idx);                                        \
        long long _n = static_cast<long long>(size);                                       \
        if (_i < 0 || _i >= _n) {                                                          \
            DEBUG_PRINT_ERROR("OOB GLOBAL %s idx=%lld size=%lld base=%p at %s:%d\n",       \
                              name, _i, _n, (const void*)(base_ptr), __FILE__, __LINE__);  \
        }                                                                                  \
    } while (0)

#define DEBUG_OOB_SMEM_1D(name, idx, size, base_ptr)                                       \
    do {                                                                                   \
        long long _i = static_cast<long long>(idx);                                        \
        long long _n = static_cast<long long>(size);                                       \
        if (_i < 0 || _i >= _n) {                                                          \
            uint32_t _addr = cvta_to_shared_u32(base_ptr);                                 \
            DEBUG_PRINT_ERROR("OOB SMEM %s idx=%lld size=%lld base_smem_addr=%u at %s:%d\n",\
                              name, _i, _n, _addr, __FILE__, __LINE__);                    \
        }                                                                                  \
    } while (0)

#else
#define DEBUG_PRINT_ERROR(fmt, ...) do { } while (0)
#define DEBUG_OOB_GLOBAL_1D(name, idx, size, base_ptr) do { } while (0)
#define DEBUG_OOB_SMEM_1D(name, idx, size, base_ptr) do { } while (0)
#define DEBUG_MAX_ERRORS 0u
#endif



// Enable advanced tcgen05-based mainloop when non-zero.
#ifndef USE_tcgen05_MAINLOOP
#define USE_tcgen05_MAINLOOP 1
#endif

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
    // Canonical descriptor construction + runtime SF-ID derivation.
    (void)sf_format;
    using AType = cutlass::float_e2m1_t;
    using BType = cutlass::float_e2m1_t;
    using CType = float;
    using SFType = cutlass::float_ue4m3_t;

    auto desc_i = cute::UMMA::make_instr_desc_block_scaled<
        AType, BType, CType, SFType,
        128, 128,
        cute::UMMA::Major::K, cute::UMMA::Major::K
    >();
    desc_i.m_dim_ = uint32_t(tile_m >> 4);
    desc_i.n_dim_ = uint32_t(tile_n >> 3);
    desc_i.a_major_ = uint8_t(a_major & 0x1);
    desc_i.b_major_ = uint8_t(b_major & 0x1);

    // Populate a_sf_id/b_sf_id from TMEM addresses (as CUTLASS does).
    return cute::UMMA::make_runtime_instr_desc_block_scaled(desc_i, tmem_sfa, tmem_sfb);
}

// Prefetch tile using TMA - simplified for Rank-2 (L=1) GEMM
template<int TileM, int TileK>
__device__ __forceinline__ void prefetch_tile(
    int stage, int k_tile_base,
    bool use_tma_a, bool is_producer, int warp_id, int lane_id,
    int m_tile, int n_tile, int K_packed, int K_scales, int M, int N,
    uint8_t** a_packed_stage, uint8_t** b_packed_stage, uint8_t** sfa_stage, uint8_t** sfb_stage,
    uint64_t* mbar,
    const CUtensorMap* desc_A, const CUtensorMap* desc_B,
    const CUtensorMap* desc_SFA, const CUtensorMap* desc_SFB
) {
    constexpr int TileKPacked = TileK / 2;
    // SfaBoxK is 128 for SWIZZLE_128B (Rank-2 GEMM now uses SWIZZLE_128B)
    constexpr int SfaBoxK = 128; 

    if (use_tma_a && is_producer) {
        if (warp_id == 0 && lane_id == 0) {
            // Element-space coordinates
            uint32_t c_m = static_cast<uint32_t>(m_tile);
            uint32_t c_n = static_cast<uint32_t>(n_tile);
            uint32_t c_k_packed = static_cast<uint32_t>(k_tile_base >> 1);
            uint32_t c_k_scales = static_cast<uint32_t>(k_tile_base >> 4);

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

// Minimal per-tile helper: decode + MMA only (no K-loop, no prefetch, no wait/sync)
// Lifted directly from sahasra_copy.py lines 1117-1347
// Minimal per-tile helper: decode + MMA
template<int TileM, int TileN, int TileK, int Threads>
__device__ __forceinline__ void process_tile(
    int k_tile, int stage, int tile_rows, int tile_cols,
    int m_tile, int n_tile,
    uint8_t** a_packed_stage, uint8_t** b_packed_stage,
    uint8_t** sfa_stage, uint8_t** sfb_stage,
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SFA_packed,
    const uint8_t* __restrict__ SFB_packed,
    half* a_f16_smem, half* b_f16_smem,
    const int M, const int N, const int K, const int K_scales,
    const int tid, const int warp_id, const int lane_id,
    const bool is_producer, const bool is_consumer,
    float c_accum[16][4]
) {
    constexpr int TileKPacked = TileK / 2;
    constexpr int a_stride = TileK + 8;
    constexpr int b_stride = TileK + 8;
    // Use a simple [TileK, TileN] K-major layout for decoded B so each K-row
    // has TileN contiguous columns. This gives a per-row pitch of TileN * 2
    // bytes, which is a multiple of 16 bytes (TileN=128), satisfying
    // ldmatrix alignment requirements.
    constexpr int b_k_stride = TileN;
    constexpr int SfaBoxK = 128;
    // Correct stride for scale factor tiles: 2048 bytes = 128 rows × 16 cols
    constexpr int SF_STRIDE = TileK / 16;  // 16 scales per row (256/16=16)

    const int K_packed = K >> 1;
    // K_scales is already passed as parameter

    // Base K indices for this tile
    const int k_packed_base = k_tile >> 1;
    const int k_scales_base = k_tile >> 4;
    // This stage's scale tile corresponds exactly to global scale columns
    // [k_scales_base, k_scales_base + SF_STRIDE).

    // TMA stage tiles interpreted as row-major
    uint8_t* a_tile  = a_packed_stage[stage];   // [TileM, TileKPacked]
    uint8_t* b_tile  = b_packed_stage[stage];   // [TileN, TileKPacked]
    uint8_t* sfa_tile = sfa_stage[stage];       // [TileM, SF_STRIDE] = [128, 16]
    uint8_t* sfb_tile = sfb_stage[stage];       // [TileN, SF_STRIDE] = [128, 16]

    int curr_k = (K - k_tile) < TileK ? (K - k_tile) : TileK;
    int curr_cols_a = (curr_k + 1) >> 1;
    int curr_cols_b = (curr_k + 1) >> 1;
    int scale_count = (curr_k + 15) >> 4;
 
     // --- DECODE A (M x K) into row-major [TileM, TileK] ---
    {
        // Each thread processes groups of 8 packed bytes (one FP8 scale block) per row
        int total_blocks_a = tile_rows * scale_count;
        for (int idx = tid; idx < total_blocks_a; idx += Threads) {
            int row       = idx / scale_count;
            int scale_col = idx - row * scale_count;   // K-block within the tile

            int m_global        = m_tile + row;
            int global_k_scale  = k_scales_base + scale_col;

            // Load FP8 scale once per 16 FP4 values (8 packed bytes)
            half scale_h = __float2half(0.0f);
            if (m_global < M && global_k_scale < K_scales) {
                // scale_col is the *local* scale column within this TileK slice (0..SF_STRIDE-1)
                // NOTE: sf_tile is in packed16×mm32 chunks (not a simple row-major [128,16]).
                uint8_t sfa_byte = load_sf_tile_byte_2048<TileK>(sfa_tile, row, scale_col);
                scale_h = __float2half(decode_fp8_e4m3(sfa_byte));
            }

            // Columns of packed FP4 covered by this scale block
            int col_packed_start = scale_col << 3;     // 8 packed bytes per 16 FP4
            int col_packed_end   = col_packed_start + 8;
            if (col_packed_start >= curr_cols_a) {
                continue;
            }
            if (col_packed_end > curr_cols_a) {
                col_packed_end = curr_cols_a;
            }

            int   k_base_block = k_tile + (col_packed_start << 1);
            half* a_dst        = a_f16_smem + row * a_stride;

            for (int col_packed = col_packed_start; col_packed < col_packed_end; ++col_packed) {
                int k_base          = k_base_block + ((col_packed - col_packed_start) << 1);
                int k_packed_global = (k_base >> 1);

                uint8_t packed = 0;
                if (m_global < M && k_packed_global < K_packed) {
                    int a_tile_idx = row * TileKPacked + col_packed;
                    packed = a_tile[a_tile_idx];
                }

                half v0, v1;
                decode_fp4x2_hw(packed, v0, v1);
                v0 = __hmul(v0, scale_h);
                v1 = __hmul(v1, scale_h);

                a_dst[col_packed * 2]     = v0;
                a_dst[col_packed * 2 + 1] = v1;
            }
        }
    }

    // --- DECODE B (N x K) into K-major layout [TileK, TileN] ---
    {
        // Symmetric decode for B: reuse one FP8 scale across 8 packed bytes
        int total_blocks_b = tile_cols * scale_count;
        for (int idx = tid; idx < total_blocks_b; idx += Threads) {
            int row       = idx / scale_count;          // N dimension within tile
            int scale_col = idx - row * scale_count;    // K-block within tile

            int n_global       = n_tile + row;
            int global_k_scale = k_scales_base + scale_col;

            half scale_h = __float2half(0.0f);
            if (n_global < N && global_k_scale < K_scales) {
                uint8_t sfb_byte = load_sf_tile_byte_2048<TileK>(sfb_tile, row, scale_col);
                scale_h = __float2half(decode_fp8_e4m3(sfb_byte));
            }

            int col_packed_start = scale_col << 3;
            int col_packed_end   = col_packed_start + 8;
            if (col_packed_start >= curr_cols_b) {
                continue;
            }
            if (col_packed_end > curr_cols_b) {
                col_packed_end = curr_cols_b;
            }

            int k_base_block = k_tile + (col_packed_start << 1);

            for (int col_packed = col_packed_start; col_packed < col_packed_end; ++col_packed) {
                int k_base          = k_base_block + ((col_packed - col_packed_start) << 1);
                int k_packed_global = (k_base >> 1);

                uint8_t packed = 0;
                if (n_global < N && k_packed_global < K_packed) {
                    int b_tile_idx = row * TileKPacked + col_packed;
                    packed = b_tile[b_tile_idx];
                }

                half v0, v1;
                decode_fp4x2_hw(packed, v0, v1);
                v0 = __hmul(v0, scale_h);
                v1 = __hmul(v1, scale_h);

                int k0 = col_packed * 2;
                int k1 = k0 + 1;

                if (row < tile_cols) {
                    if (k0 < TileK) {
                        half* b_row0 = b_f16_smem + k0 * b_k_stride;
                        b_row0[row] = v0;
                    }
                    if (k1 < TileK) {
                        half* b_row1 = b_f16_smem + k1 * b_k_stride;
                        b_row1[row] = v1;
                    }
                }
            }
        }
    }

    __syncthreads();

    // --- MMA LOOP: CUTLASS-style m16n8k16 layout ---
    if (!is_consumer) {
        return;
    }

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int WARP_TILE_M = 4;  // 4x4 warp tiles -> 128x128 per CTA
    constexpr int WARP_TILE_N = 4;

    const int warp_m = warp_id % 2; // 0..1
    const int warp_n = warp_id / 2; // 0..3

    for (int kk = 0; kk < curr_k; kk += MMA_K) {
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

        // Load A fragments: row-major [TileM, TileK]
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M; // 0..127
            int lane_smem_a_m = warp_smem_a_m + (lane_id % 16);            // 0..15 within tile
            if (lane_smem_a_m < tile_rows) {
                int lane_smem_a_k = kk + (lane_id / 16) * 8;               // 0 or 8 within this k-block
                if (lane_smem_a_k < curr_k) {
                    uint32_t a_base = cvta_to_shared_u32(
                        a_f16_smem + lane_smem_a_m * a_stride + lane_smem_a_k);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
                        : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])
                        : "r"(a_base)
                    );
                } else {
                    RA[i][0] = RA[i][1] = RA[i][2] = RA[i][3] = 0u;
                }
            } else {
                RA[i][0] = RA[i][1] = RA[i][2] = RA[i][3] = 0u;
            }
        }

        // Load B fragments: treat shared B as column-major [TileK, TileN]
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N; // 0..127
            int lane_smem_b_k = kk + (lane_id % 16);                        // 0..15 within k-block
            if (lane_smem_b_k < curr_k && warp_smem_b_n < tile_cols) {
                uint32_t b_base = cvta_to_shared_u32(
                    b_f16_smem + lane_smem_b_k * b_k_stride + warp_smem_b_n);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [%2];\n"
                    : "=r"(RB[j][0]), "=r"(RB[j][1])
                    : "r"(b_base)
                );
            } else {
                RB[j][0] = RB[j][1] = 0u;
            }
        }

        // MMA compute: accumulate into c_accum flattened as [i * WARP_TILE_N + j][0..3]
        #pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int row_block = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            if (row_block >= tile_rows) continue;
            #pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int col_block = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                if (col_block >= tile_cols) continue;

                int acc_idx = i * WARP_TILE_N + j; // 0..15
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{ %0, %1, %2, %3 }, "
                    "{ %4, %5, %6, %7 }, "
                    "{ %8, %9 }, "
                    "{ %0, %1, %2, %3 };\n"
                    : "+f"(c_accum[acc_idx][0]), "+f"(c_accum[acc_idx][1]),
                      "+f"(c_accum[acc_idx][2]), "+f"(c_accum[acc_idx][3])
                    : "r"(RA[i][0]), "r"(RA[i][1]), "r"(RA[i][2]), "r"(RA[i][3]),
                      "r"(RB[j][0]), "r"(RB[j][1])
                );
            }
        }
    }
}

// RANK2 CTA + SWIZZLE_128B + BOX_K 128 BYTE (NO CLUSTER)
template<int TileM, int TileK, int Threads>
__global__ void __launch_bounds__(Threads)
fp4_gemm_rank2_cta(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SFA_packed,
    const uint8_t* __restrict__ SFB_packed,
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
    constexpr int SfaBoxK = 128;  // Rank-2 GEMM: SWIZZLE_128B
    constexpr int StageCount = 2; 
    constexpr int a_stride = TileK + 8;
    constexpr int b_stride = TileK + 8;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const bool is_producer = warp_id == 0; // Warp 0 issues TMA
    const bool is_consumer = true; // All warps compute (including warp 0 after barrier)

    // Grid: x=M_tile, y=N_tile, z=Batch(L)
    const int batch = blockIdx.z;
    const int m_tile = blockIdx.x * TileM;
    const int n_tile = blockIdx.y * TileN;
    
    if (batch >= L || m_tile >= M || n_tile >= N) return;

    const int K_packed = K >> 1;
    const int tile_rows = (M - m_tile) < TileM ? (M - m_tile) : TileM;
    const int tile_cols = (N - n_tile) < TileN ? (N - n_tile) : TileN;

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
    
    // Helper to align to 128 bytes
    auto align_up_smem_128 = [&] __device__() {
        uint32_t addr = cvta_to_shared_u32(smem + offset);
        uint32_t aligned = (addr + 127u) & ~127u;
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

#if USE_tcgen05_MAINLOOP
    // Compact scale tiles for tcgen05.cp: ensure byte-for-byte packed16 order.
    align_up_smem_128();
    uint8_t* sfa_compact_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfa_compact_stage[s] = smem + offset;
        offset += SF_TILE_BYTES;
    }

    align_up_smem_128();
    uint8_t* sfb_compact_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfb_compact_stage[s] = smem + offset;
        offset += SF_TILE_BYTES;
    }
#endif

    #if !USE_tcgen05_MAINLOOP
    // =========================================================================
    // FP16 decode buffers - ONLY needed for non-tcgen05 path
    // tcgen05.mma consumes packed FP4 directly, no decode buffers needed!
    // =========================================================================
    align_up_smem_128();
    half* a_f16_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileM) * a_stride * sizeof(half);
    
    align_up_smem_128();
    half* b_f16_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileN) * b_stride * sizeof(half);
    
    align_up_smem_128();
    float* c_smem_from_tmem = reinterpret_cast<float*>(smem + offset);
    offset += static_cast<size_t>(TileM) * static_cast<size_t>(TileN) * sizeof(float);
    #endif  // !USE_tcgen05_MAINLOOP

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

    #if !USE_tcgen05_MAINLOOP
    // Accumulators - ONLY for non-tcgen05 path (tcgen05 uses TMEM)
    float c_accum[16][4]; // 16 steps of N=8, 4 floats each
    #pragma unroll
    for(int i=0; i<16; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) c_accum[i][j] = 0.0f;
    }
    #endif  // !USE_tcgen05_MAINLOOP

    #if !USE_tcgen05_MAINLOOP
    // Main Loop
    int phase = 0;
    
    // Prologue: Prefetch stages 0 to StageCount-2
    for (int s = 0; s < StageCount - 1; ++s) {
        int k_tile = s * TileK;
        if (k_tile < K) {
            prefetch_tile<TileM, TileK>(
                s, k_tile, use_tma_a, is_producer, warp_id, lane_id,
                m_tile, n_tile, K_packed, K_scales, M, N,
                a_packed_stage, b_packed_stage, sfa_stage, sfb_stage,
                mbar,
                desc_A, desc_B, desc_SFA, desc_SFB
            );
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += TileK) {
        int stage = (k_tile / TileK) % StageCount;
        int next_k = k_tile + (StageCount - 1) * TileK;

        // Issue prefetch for next_k
        if (next_k < K) {
            prefetch_tile<TileM, TileK>(
                (stage + StageCount - 1) % StageCount, next_k, use_tma_a, is_producer, warp_id, lane_id,
                m_tile, n_tile, K_packed, K_scales, M, N,
                a_packed_stage, b_packed_stage, sfa_stage, sfb_stage,
                mbar,
                desc_A, desc_B, desc_SFA, desc_SFB
            );
        }

        // Wait for current stage
        mbarrier_wait_parity(mbar_stage(mbar, stage), phase);
        __syncthreads(); // Ensure all threads see data

        // Process tile
        process_tile<TileM, TileN, TileK, Threads>(
            k_tile, stage, tile_rows, tile_cols,
            m_tile, n_tile,
            a_packed_stage, b_packed_stage, sfa_stage, sfb_stage,
            A_packed, B_packed, SFA_packed, SFB_packed,
            a_f16_smem, b_f16_smem,
            M, N, K, K_scales,
            tid, warp_id, lane_id,
            is_producer, is_consumer,
            c_accum
        );
        
        __syncthreads(); // Ensure consumers done before recycling stage?
        
        if ((k_tile / TileK) % StageCount == StageCount - 1) {
            phase ^= 1;
        }
    }

    // Epilogue: Writeback using canonical m16n8k16 lane mapping
    {
        constexpr int MMA_M = 16;
        constexpr int MMA_N = 8;
        constexpr int WARP_TILE_M = 4;
        constexpr int WARP_TILE_N = 4;

        int warp_m = warp_id % 2;
        int warp_n = warp_id / 2;

        int lane_row_group = lane_id / 4; // 0..7
        int lane_col_group = lane_id % 4; // 0..3

        int row0 = lane_row_group;
        int row1 = lane_row_group + 8;
        int col0 = lane_col_group * 2;
        int col1 = col0 + 1;

        for (int i = 0; i < WARP_TILE_M; ++i) {
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int acc_idx = i * WARP_TILE_N + j; // 0..15

                int tile_row_base = m_tile + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int tile_col_base = n_tile + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;

                int global_row0 = tile_row_base + row0;
                int global_row1 = tile_row_base + row1;
                int global_col0 = tile_col_base + col0;
                int global_col1 = tile_col_base + col1;

                if (global_row0 < M) {
                    if (global_col0 < N) {
                        D[global_row0 * N + global_col0] = __float2half(c_accum[acc_idx][0]);
                    }
                    if (global_col1 < N) {
                        D[global_row0 * N + global_col1] = __float2half(c_accum[acc_idx][1]);
                    }
                }
                if (global_row1 < M) {
                    if (global_col0 < N) {
                        D[global_row1 * N + global_col0] = __float2half(c_accum[acc_idx][2]);
                    }
                    if (global_col1 < N) {
                        D[global_row1 * N + global_col1] = __float2half(c_accum[acc_idx][3]);
                    }
                }
            }
        }
    }
    #else  // USE_tcgen05_MAINLOOP
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
                m_tile, n_tile, K_packed, K_scales, M, N,
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

    // Canonical TMEM scale tensors: match CUTLASS' blockscaled collective.
    // Allocate the TMEM scale fragments using the derived SMEM scale layout atoms.
    using BlkScaledCfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;
    using TileShape_MNK = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
    using SmemLayoutAtomSFA = decltype(BlkScaledCfg::deduce_smem_layoutSFA(tiled_mma, TileShape_MNK{}));
    using SmemLayoutAtomSFB = decltype(BlkScaledCfg::deduce_smem_layoutSFB(tiled_mma, TileShape_MNK{}));

    auto tCtSFA = make_tensor<typename MMA_Traits<MmaOp>::FrgTypeSFA>(shape(SmemLayoutAtomSFA{}));
    auto tCtSFB = make_tensor<typename MMA_Traits<MmaOp>::FrgTypeSFB>(shape(SmemLayoutAtomSFB{}));

    // Place SFA/SFB after the accumulator region in TMEM.
    tCtSFA.data() = tCtAcc.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tCtAcc);
    tCtSFB.data() = tCtSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);

    __shared__ __align__(16) uint64_t desc_a_smem_sh[kNumKBlocks];
    __shared__ __align__(16) uint64_t desc_b_smem_sh[kNumKBlocks];
    __shared__ __align__(16) uint32_t tmem_sfa_kb_sh[kNumKBlocks];
    __shared__ __align__(16) uint32_t tmem_sfb_kb_sh[kNumKBlocks];
    __shared__ __align__(4) uint32_t idescE_hi_sh[kNumKBlocks];

    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int kb = 0; kb < kNumKBlocks; ++kb) {
            tmem_sfa_kb_sh[kb] = raw_pointer_cast(tCtSFA(_, _, kb).data());
            tmem_sfb_kb_sh[kb] = raw_pointer_cast(tCtSFB(_, _, kb).data());
        }
    }
    #endif

    // NOTE: TMEM accumulator initialization is handled by scaleC=0 on the first MMA.
    // tcgen05.mma with scaleC=0 (predicate p=false) does D = A*B, ignoring existing C.

    #if __CUDA_ARCH__ >= 1000
    // Build instruction descriptor (constant for all K-tiles)
    // sf_format = 0 for E4M3 scale format
    // a_sf_id and b_sf_id are set to 0 (default ID, not derived from addresses)
    #endif
    #if __CUDA_ARCH__ >= 1000
    if (warp_id == 0 && lane_id == 0) {
        #pragma unroll
        for (int kb = 0; kb < kNumKBlocks; ++kb) {
            uint64_t idescE = make_instr_desc_mxf4(
                TileM, TileN,
                0, 0,       // K-major for both A and B
                0,          // sf_format = E4M3
                tmem_sfa_kb_sh[kb], tmem_sfb_kb_sh[kb]
            );
            idescE_hi_sh[kb] = uint32_t(idescE >> 32);
#if NVFP4_DEBUG_DUMP
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                uint32_t idesc_hi = uint32_t(idescE >> 32);
                uint32_t a_fmt = (idesc_hi >> 7) & 0x7;
                uint32_t b_fmt = (idesc_hi >> 10) & 0x7;
                uint32_t a_maj = (idesc_hi >> 15) & 0x1;
                uint32_t b_maj = (idesc_hi >> 16) & 0x1;
                uint32_t n_dim = (idesc_hi >> 17) & 0x3F;
                uint32_t sf_fmt = (idesc_hi >> 23) & 0x1;
                uint32_t m_dim = (idesc_hi >> 24) & 0x1F;
                uint32_t b_sf = (idesc_hi >> 4) & 0x3;
                uint32_t a_sf = (idesc_hi >> 29) & 0x3;
                printf("idescE_kb%d=0x%016llx fmt(a,b)=%u,%u major(a,b)=%u,%u dims(m,n)=%u,%u sf_fmt=%u sf_id(a,b)=%u,%u tmem_sfa=0x%08x tmem_sfb=0x%08x\n",
                       kb, (unsigned long long)idescE, a_fmt, b_fmt, a_maj, b_maj,
                       m_dim, n_dim, sf_fmt, a_sf, b_sf,
                       tmem_sfa_kb_sh[kb], tmem_sfb_kb_sh[kb]);
            }
#endif
        }
    }
#if NVFP4_DEBUG_DUMP
    __syncthreads();
#endif
    __syncthreads();
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
                m_tile, n_tile, K_packed, K_scales, M, N,
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
        // Compact scale tiles into packed16 order (byte-for-byte match to to_blocked)
        // =========================================================================
        {
            constexpr int kScalePanelBytes = 32 * 16;
            constexpr int kTotalScaleBytes = kNumKBlocks * kScalePanelBytes;
            uint8_t* sfa_src = sfa_stage[stage];
            uint8_t* sfb_src = sfb_stage[stage];
            uint8_t* sfa_dst = sfa_compact_stage[stage];
            uint8_t* sfb_dst = sfb_compact_stage[stage];
            for (int idx = tid; idx < kTotalScaleBytes; idx += Threads) {
                int mm32 = (idx & 511) >> 4;
                int packed16 = (idx & 511) & 15;
                int kb = idx >> 9;
                int mm4 = packed16 >> 2;
                int kk4 = packed16 & 3;
                int row_global = mm32 + (mm4 << 5);
                int scale_col = (kb << 2) + kk4;
                int dst_idx = (kb << 9) + (mm32 << 4) + packed16;
                sfa_dst[dst_idx] = load_sf_tile_byte_2048<TileK>(sfa_src, row_global, scale_col);
                sfb_dst[dst_idx] = load_sf_tile_byte_2048<TileK>(sfb_src, row_global, scale_col);
            }
        }
        __syncthreads();
#if NVFP4_DEBUG_DUMP
        if (warp_id == 0 && lane_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("sfa_stage[0..31]:");
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                printf(" %02x", sfa_stage[stage][i]);
            }
            printf("\n");
            printf("sfa_compact[0..31]:");
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                printf(" %02x", sfa_compact_stage[stage][i]);
            }
            printf("\n");
            printf("sfb_compact[0..31]:");
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                printf(" %02x", sfb_compact_stage[stage][i]);
            }
            printf("\n");
        }
#endif
        __syncthreads();

        // =========================================================================
        // Scale SMEM -> TMEM (Cp4x32x128b): one copy per K-block (K=64)
        // =========================================================================
        // Each K-block holds 32 rows × 16B/row = 512B of FP8 scale data in SMEM.
        // sfa_stage/sfb_stage are laid out as 4 contiguous 512B chunks (one per K-block).
        //
        // NOTE: Match kutte.py oracle issue scope: issue tcgen05.cp from a single warp
        // (warp 0) and a single lane (lane 0). This avoids redundant copies from
        // multiple warps in the CTA that can corrupt TMEM scale panels.
        uint32_t sfa_stage_smem = cvta_to_shared_u32(sfa_compact_stage[stage]);
        uint32_t sfb_stage_smem = cvta_to_shared_u32(sfb_compact_stage[stage]);
        // Scale SMEM->TMEM copy is warpgroup-scoped (.warpx4). All threads in the
        // 4-warp warpgroup must participate (no lane_id gating).
        if (warp_id < 4) {
            // Issue one UTCCP per 512B scale panel (per K-block).
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
            if (warp_id == 0 && lane_id == 0) {
                tcgen05_commit_mbarrier(mbar_stage(mbar_cp, stage));
            }
        }

        // Wait for scale copy completion before MMA consumes TMEM scales.
        // Use the same parity as the current stage.
        __syncthreads();
        if (warp_id == 0 && lane_id == 0) {
            mbarrier_wait_parity(mbar_stage(mbar_cp, stage), phase);
        }
        __syncthreads();

        // =========================================================================
        // MMA: one tcgen05.mma per K-block (K=64), with ACCUMULATE disabled only
        // for the first K-block of the full GEMM.
        // =========================================================================
        using namespace cute;
        using ElementAB = uint8_t;
        using SmemLayoutAtomAB = UMMA::Layout_K_SW128_Atom<ElementAB>;
        constexpr int TileKPacked = TileK / 2;
        constexpr int kKBlockPacked = kKBlock / 2;

        // A is (M, Kpacked bytes)
        auto smem_layout_a = tile_to_shape(SmemLayoutAtomAB{},
            make_shape(Int<TileM>{}, Int<TileKPacked>{}),
            cute::GenRowMajor{});
        auto sA_full = make_tensor(make_smem_ptr<ElementAB>(a_packed_stage[stage]), smem_layout_a);

        // B is (N, Kpacked bytes)
        auto smem_layout_b = tile_to_shape(SmemLayoutAtomAB{},
            make_shape(Int<TileN>{}, Int<TileKPacked>{}),
            cute::GenRowMajor{});
        auto sB_full = make_tensor(make_smem_ptr<ElementAB>(b_packed_stage[stage]), smem_layout_b);

#if NVFP4_DEBUG_DUMP
        if (warp_id == 0 && lane_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && k_tile == 0) {
            uint32_t a_base = cvta_to_shared_u32(a_packed_stage[stage]);
            uint32_t b_base = cvta_to_shared_u32(b_packed_stage[stage]);
            uint32_t a_k1 = cvta_to_shared_u32(a_packed_stage[stage] + 1);
            uint32_t a_m1 = cvta_to_shared_u32(a_packed_stage[stage] + TileKPacked);
            uint32_t b_k1 = cvta_to_shared_u32(b_packed_stage[stage] + 1);
            uint32_t b_n1 = cvta_to_shared_u32(b_packed_stage[stage] + TileKPacked);
            printf("a_smem_base=0x%08x a_k1=0x%08x a_m1=0x%08x\n", a_base, a_k1, a_m1);
            printf("b_smem_base=0x%08x b_k1=0x%08x b_n1=0x%08x\n", b_base, b_k1, b_n1);
            printf("a_stride_k=%u a_stride_m=%u\n", a_k1 - a_base, a_m1 - a_base);
            printf("b_stride_k=%u b_stride_n=%u\n", b_k1 - b_base, b_n1 - b_base);
            printf("a_packed_stage[0..3]: %02x %02x %02x %02x\n",
                   a_packed_stage[stage][0], a_packed_stage[stage][1],
                   a_packed_stage[stage][2], a_packed_stage[stage][3]);
            printf("b_packed_stage[0..3]: %02x %02x %02x %02x\n",
                   b_packed_stage[stage][0], b_packed_stage[stage][1],
                   b_packed_stage[stage][2], b_packed_stage[stage][3]);
            printf("sA_full kbyte[0..7], m0:");
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                auto v = sA_full(make_coord(0, i));
                printf(" %02x", static_cast<unsigned>(v));
            }
            printf("\n");
            printf("sB_full kbyte[0..7], n0:");
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                auto v = sB_full(make_coord(0, i));
                printf(" %02x", static_cast<unsigned>(v));
            }
            printf("\n");
        }
#endif


        if (warp_id == 0 && lane_id == 0) {
            #pragma unroll
            for (int kb = 0; kb < kNumKBlocks; ++kb) {
                auto sA_kb = local_tile(sA_full, make_shape(Int<TileM>{}, Int<kKBlockPacked>{}), make_coord(0, kb));
                auto sB_kb = local_tile(sB_full, make_shape(Int<TileN>{}, Int<kKBlockPacked>{}), make_coord(0, kb));
                desc_a_smem_sh[kb] = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(sA_kb));
                desc_b_smem_sh[kb] = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(sB_kb));
            }
        }
        __syncthreads();
#if NVFP4_DEBUG_DUMP
        if (warp_id == 0 && lane_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("desc_a_smem[0]=0x%016llx\n", (unsigned long long)desc_a_smem_sh[0]);
            printf("desc_b_smem[0]=0x%016llx\n", (unsigned long long)desc_b_smem_sh[0]);
        }
#endif

        auto mma_kb = [&](int kb, bool accum) {
            uint64_t desc_a_smem = desc_a_smem_sh[kb];
            uint64_t desc_b_smem = desc_b_smem_sh[kb];
            uint32_t tmem_sfa_kb = tmem_sfa_kb_sh[kb];
            uint32_t tmem_sfb_kb = tmem_sfb_kb_sh[kb];
            uint32_t idesc_hi = idescE_hi_sh[kb];

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
                      "r"(idesc_hi), "r"(tmem_sfa_kb), "r"(tmem_sfb_kb)
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
                      "r"(idesc_hi), "r"(tmem_sfa_kb), "r"(tmem_sfb_kb)
                    : "memory"
                );
            }
        };

        if (warp_id < 4) {
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
    // Match sm100_bf16_gemm.cuh epilogue pattern exactly for correct TMEM addressing
    {
        constexpr uint32_t LAYOUT_AD_M = 128;
        constexpr uint32_t BLOCK_N = TileN;
        constexpr uint32_t kNumMWaves = 1;
        constexpr uint32_t kSwizzleCDMode = 128;
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
        constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(float);
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
        constexpr uint32_t accum_stage_idx = 0;  // Single accumulator stage for our case

        // Each thread handles one row (M dimension)
        int row_base = warp_id * 32 + lane_id;
        bool full_tile = (m_tile + TileM <= M) && (n_tile + TileN <= N);
        if (warp_id < 4 && row_base < TileM) {
            // Global memory M coordinate (same for all iterations)
            int gm = m_tile + row_base;
            
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++w) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++s) {
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
                        // TMEM address: N-dimension offset from base
                        // SM100_TMEM_LOAD is warp-collective: same address gives different rows to different threads
                        // TMEM load address: column offset.
                        // For SM100 TMEM load ops (as used in sm100_bf16_gemm.cuh), the address is an offset
                        // in TMEM columns; do not add the tcgen05.alloc base pointer.
                        uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +
                                             w * BLOCK_N +
                                             s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;

                        // Load 4 consecutive FP32 values from TMEM
                        uint32_t v0, v1, v2, v3;
                        cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_c + tmem_addr, v0, v1, v2, v3);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Global memory N coordinate (row-major):
                        // This must match the mathematical output layout used by reference.py.
                        // NOTE: the lane/bank-group swizzle in sm100_bf16_gemm.cuh is for SMEM/TMA layout;
                        // when writing directly to global memory we use the logical N coordinate.
                        int gn = n_tile + int(w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup);
#if NVFP4_DEBUG_DUMP
                        if (warp_id == 0 && lane_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                            w == 0 && s == 0 && i == 0) {
                            printf("map gm=%d gn=%d tmem_addr=%u v=%f %f %f %f\n",
                                   gm, gn, tmem_addr,
                                   __uint_as_float(v0), __uint_as_float(v1),
                                   __uint_as_float(v2), __uint_as_float(v3));
                        }
#endif

                        // Write to global memory
                        if (full_tile) {
                            int base = gm * N + gn;
                            D[base + 0] = __float2half_rn(__uint_as_float(v0));
                            D[base + 1] = __float2half_rn(__uint_as_float(v1));
                            D[base + 2] = __float2half_rn(__uint_as_float(v2));
                            D[base + 3] = __float2half_rn(__uint_as_float(v3));
                        } else {
                            if (gm < M) {
                                if (gn + 0 < N) D[gm * N + gn + 0] = __float2half_rn(__uint_as_float(v0));
                                if (gn + 1 < N) D[gm * N + gn + 1] = __float2half_rn(__uint_as_float(v1));
                                if (gn + 2 < N) D[gm * N + gn + 2] = __float2half_rn(__uint_as_float(v2));
                                if (gn + 3 < N) D[gm * N + gn + 3] = __float2half_rn(__uint_as_float(v3));
                            }
                        }
                    }
                }
            }
        }
    }
#if NVFP4_DEBUG_DUMP
    if (warp_id == 0 && lane_id == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("D[0..7]:");
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = __half2float(D[i]);
            printf(" %.3f", v);
        }
        printf("\n");
    }
#endif

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
    #endif  // !USE_tcgen05_MAINLOOP
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
    #if USE_tcgen05_MAINLOOP
    constexpr int kThreads = 128; // 4 warps
    #else
    constexpr int kThreads = 256; // 8 warps
    #endif
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
    
    // Decoded buffers (FP16) are NOT used in tcgen05 mainloop
    // Removing them saves ~132KB of shared memory, avoiding MaxDynamicSharedMemorySize error
    // offset = align(offset, 128);
    // offset += kTileM * (kTileK + 8) * 2;
    // offset = align(offset, 128);
    // offset += kTileN * (kTileK + 8) * 2;
    
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

    const uint8_t* A_ptr_arg = A_ptr;
    const uint8_t* B_ptr_arg = B_ptr;
    const uint8_t* SFA_ptr_arg = SFA_ptr;
    const uint8_t* SFB_ptr_arg = SFB_ptr;
    void* kernel_args[] = {
        const_cast<const uint8_t**>(&A_ptr_arg),
        const_cast<const uint8_t**>(&B_ptr_arg),
        const_cast<const uint8_t**>(&SFA_ptr_arg),
        const_cast<const uint8_t**>(&SFB_ptr_arg),
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

template<int TileM, int TileK, int Threads>
__global__ void __launch_bounds__(Threads)
fp4_scale_debug_rank2_cta(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SFA_packed,
    const uint8_t* __restrict__ SFB_packed,
    uint8_t* __restrict__ OUT_SFA,
    uint8_t* __restrict__ OUT_SFB,
    half* __restrict__ OUT_A,
    half* __restrict__ OUT_B,
    const int M, const int N, const int K, const int L, const int K_scales_padded
) {
#if __CUDA_ARCH__ >= 900
    constexpr int TileN = 128;
    const int tid = threadIdx.x;

    const int batch  = blockIdx.z;
    const int m_tile = blockIdx.x * TileM;
    const int n_tile = blockIdx.y * TileN;

    if (batch >= L || m_tile >= M || n_tile >= N) return;

    int tile_rows   = (M - m_tile) < TileM ? (M - m_tile) : TileM;
    int tile_cols   = (N - n_tile) < TileN ? (N - n_tile) : TileN;
    int K_scales    = K / 16;
    int K_packed    = K / 2;

    // --- Copy A-side scales: [M, K_scales_padded] ---
    // Only CTAs with n_tile == 0 to avoid write races.
    if (n_tile == 0) {
        for (int idx = tid; idx < tile_rows * K_scales; idx += Threads) {
            int row      = idx / K_scales;
            int j        = idx - row * K_scales;
            int m_global = m_tile + row;
            if (m_global < M) {
                int src_idx = m_global * K_scales_padded + j;
                OUT_SFA[src_idx] = SFA_packed[src_idx];
            }
        }
    }

    // --- Copy B-side scales: [N, K_scales_padded] ---
    // Only CTAs with m_tile == 0 to avoid write races.
    if (m_tile == 0) {
        for (int idx = tid; idx < tile_cols * K_scales; idx += Threads) {
            int row      = idx / K_scales;
            int j        = idx - row * K_scales;
            int n_global = n_tile + row;
            if (n_global < N) {
                int src_idx = n_global * K_scales_padded + j;
                OUT_SFB[src_idx] = SFB_packed[src_idx];
            }
        }
    }

    // --- Decode A: OUT_A [M, K] ---
    // Only CTAs with n_tile == 0 write A rows.
    if (n_tile == 0) {
        for (int idx = tid; idx < tile_rows * K; idx += Threads) {
            int row      = idx / K;
            int k        = idx - row * K;
            int m_global = m_tile + row;
            if (m_global >= M) continue;

            int  k_packed = k >> 1;               // K/2 index
            uint8_t packed = A_packed[m_global * K_packed + k_packed];
            half v0, v1;
            decode_fp4x2_hw(packed, v0, v1);
            half v = (k & 1) ? v1 : v0;

            OUT_A[m_global * K + k] = v;
        }
    }

    // --- Decode B: OUT_B [N, K] ---
    // Only CTAs with m_tile == 0 write B rows.
    if (m_tile == 0) {
        for (int idx = tid; idx < tile_cols * K; idx += Threads) {
            int row      = idx / K;
            int k        = idx - row * K;
            int n_global = n_tile + row;
            if (n_global >= N) continue;

            int  k_packed = k >> 1;
            uint8_t packed = B_packed[n_global * K_packed + k_packed];
            half v0, v1;
            decode_fp4x2_hw(packed, v0, v1);
            half v = (k & 1) ? v1 : v0;

            OUT_B[n_global * K + k] = v;
        }
    }
#endif
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
        debug_dump = os.environ.get("NVFP4_DEBUG_DUMP", "0") not in ("", "0", "false", "False")
        module_name = "nvfp4_gemm_sm100_ptx_dbg" if debug_dump else "nvfp4_gemm_sm100_ptx"
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
        if debug_dump:
            extra_cuda_cflags.append("-DNVFP4_DEBUG_DUMP=1")

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
