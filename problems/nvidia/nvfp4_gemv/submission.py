import os

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

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
__constant__ float fp4_e2m1_lut_float[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ half decode_fp4_e2m1(uint8_t nibble) {
    return __float2half(fp4_e2m1_lut_float[nibble & 0x0F]);
}

__device__ __forceinline__ float decode_fp8_e4m3(uint8_t val) {
    cutlass::float_e4m3_t fp8_val = *reinterpret_cast<cutlass::float_e4m3_t*>(&val);
    return __half2float(__float2half_rn(fp8_val));
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
                printf(fmt, __VA_ARGS__);                                                  \
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

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void cp_async_16b(void* dst, const void* src, bool pred) {
    if (pred) {
        uint32_t smem_addr = cvta_to_shared_u32(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(src));
    }
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n");
}
#else
__device__ __forceinline__ void cp_async_16b(void* dst, const void* src, bool pred) {
    if (pred) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    }
}
__device__ __forceinline__ void cp_async_commit() {}
__device__ __forceinline__ void cp_async_wait() {}
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

// TMA load for CTA (rank-2, L=1) - no cluster
__device__ __forceinline__ void tma_load_2d_cta_no_arrive(void* smem_ptr,
                                                           const CUtensorMap* desc,
                                                           uint32_t coord0,
                                                           uint32_t coord1,
                                                           uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    uint64_t cache_hint = 0;

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

// TMA load for cluster (rank-3, L>1) - with cta_group::2
__device__ __forceinline__ void tma_load_2d_cluster_no_arrive(void* smem_ptr,
                                                                const CUtensorMap* desc,
                                                                uint32_t coord0,
                                                                uint32_t coord1,
                                                                uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    // SM100 cluster TMA: clear peer bit so transaction bytes update CTA0's barrier
    constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;
    uint32_t mbar_addr = cvta_to_shared_u32(mbar) & Sm100MmaPeerBitMask;
    uint64_t cache_hint = 0;

    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%3, %4}], [%2], %5;\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(coord0),
          "r"(coord1),
          "l"(cache_hint)
        : "memory"
    );
}

// Wrapper that chooses CTA or cluster based on runtime L
__device__ __forceinline__ void tma_load_2d_no_arrive(void* smem_ptr,
                                                       const CUtensorMap* desc,
                                                       uint32_t coord0,
                                                       uint32_t coord1,
                                                       uint64_t* mbar,
                                                       int L) {
    if (L == 1) {
        tma_load_2d_cta_no_arrive(smem_ptr, desc, coord0, coord1, mbar);
    } else {
        tma_load_2d_cluster_no_arrive(smem_ptr, desc, coord0, coord1, mbar);
    }
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

// TMA 3D load for CTA (rank-2, L=1) - no cluster
__device__ __forceinline__ void tma_load_3d_cta_no_arrive(void* smem_ptr,
                                                           const CUtensorMap* desc,
                                                           uint32_t coord0,
                                                           uint32_t coord1,
                                                           uint32_t coord2,
                                                           uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    uint64_t cache_hint = 0;

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5}], [%2];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(coord0),
          "r"(coord1),
          "r"(coord2)
        : "memory"
    );
}

// TMA 3D load for cluster (rank-3, L>1) - with cta_group::2
__device__ __forceinline__ void tma_load_3d_cluster_no_arrive(void* smem_ptr,
                                                                const CUtensorMap* desc,
                                                                uint32_t coord0,
                                                                uint32_t coord1,
                                                                uint32_t coord2,
                                                                uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    // SM100 cluster TMA: clear peer bit so transaction bytes update CTA0's barrier
    constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;
    uint32_t mbar_addr = cvta_to_shared_u32(mbar) & Sm100MmaPeerBitMask;
    uint64_t cache_hint = 0;

    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%3, %4, %5}], [%2], %6;\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(mbar_addr),
          "r"(coord0),
          "r"(coord1),
          "r"(coord2),
          "l"(cache_hint)
        : "memory"
    );
}

// Wrapper that chooses CTA or cluster based on runtime L
__device__ __forceinline__ void tma_load_3d_no_arrive(void* smem_ptr,
                                                       const CUtensorMap* desc,
                                                       uint32_t coord0,
                                                       uint32_t coord1,
                                                       uint32_t coord2,
                                                       uint64_t* mbar,
                                                       int L) {
    if (L == 1) {
        tma_load_3d_cta_no_arrive(smem_ptr, desc, coord0, coord1, coord2, mbar);
    } else {
        tma_load_3d_cluster_no_arrive(smem_ptr, desc, coord0, coord1, coord2, mbar);
    }
}

__device__ __forceinline__ void tma_load_3d(void* smem_ptr,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t coord1,
                                            uint32_t coord2,
                                            uint32_t bytes,
                                            uint64_t* mbar,
                                            int L) {
    mbarrier_arrive_expect_tx(mbar, bytes);
    tma_load_3d_no_arrive(smem_ptr, desc, coord0, coord1, coord2, mbar, L);
}

__device__ __forceinline__ uint64_t* mbar_stage(uint64_t* base, int stage) {
    // Each mbarrier occupies 16 bytes (two uint64_t slots)
    return base + stage * 2;
}

// Prefetch tile using TMA - extracted from lambda for better performance
template<int TileM, int TileK>
__device__ __forceinline__ void prefetch_tile(
    int stage, int k_tile_base,
    bool use_tma_a, bool is_producer, int warp_id, int lane_id,
    int m_tile, int K_packed, int K_scales_padded, int M, int L, int batch,
    uint8_t** a_packed_stage, uint8_t** sfa_stage, uint64_t* mbar_a,
    const CUtensorMap* desc_A, const CUtensorMap* desc_SFA
) {
    constexpr int TileKPacked = TileK / 2;

    if (use_tma_a && is_producer) {
        if (warp_id == 0 && lane_id == 0) {
            // Element-space coordinates: row and packed-K index
            uint32_t c_m = static_cast<uint32_t>(m_tile);
            uint32_t c_k_packed = static_cast<uint32_t>(k_tile_base >> 1);
            uint32_t c_k_scales = static_cast<uint32_t>(k_tile_base >> 4);

            // SFA box size must match TMA descriptor configuration:
            // rank-2 (L=1, CTA, SWIZZLE_NONE): box_k=16
            // rank-3 (L=4,8, cluster, SWIZZLE_128B): box_k=128
            int SfaBoxK = (L == 1) ? 16 : 128;

            if (L == 1) {
                // Load A matrix tile (2D)
                uint32_t c0 = c_k_packed;
                uint32_t c1 = c_m;
                bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);
                bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);

                // Load SFA scales tile (2D) using element-space coordinates
                // TMA requires coordinates to be aligned to box size
                // For rank-2: align to SfaBoxK (16) and load the correct slice for this K-tile
                uint32_t sfa_c0 = (c_k_scales / SfaBoxK) * SfaBoxK;  // Align to 16-scale boundaries
                uint32_t sfa_c1 = c_m;
                bool valid_sfa_k = (sfa_c0 + SfaBoxK) <= static_cast<uint32_t>(K_scales_padded);
                bool valid_sfa_m = (c_m + TileM) <= static_cast<uint32_t>(M);

                // Calculate total bytes for mbarrier
                uint32_t total_bytes = 0;
                if (valid_m && valid_k) {
                    total_bytes += (TileM * TileKPacked + 15) & ~15;
                }
                if (valid_sfa_m && valid_sfa_k) {
                    total_bytes += TileM * SfaBoxK;
                }

                // Set expected bytes for mbarrier
                mbarrier_arrive_expect_tx(mbar_stage(mbar_a, stage), total_bytes);

                // Issue TMA loads (they will complete to the same mbarrier)
                if (valid_m && valid_k) {
#ifndef NDEBUG
                    if (k_tile_base == 256 && stage == 1) {  // Second tile with TileK=256
                        printf("DEBUG prefetch: About to TMA load A, c0=%u c1=%u valid_k=%d valid_m=%d\n",
                               c0, c1, valid_k, valid_m);
                    }
#endif
                    tma_load_2d_no_arrive(
                        a_packed_stage[stage],
                        desc_A,
                        c0, c1,
                        mbar_stage(mbar_a, stage),
                        L
                    );
#ifndef NDEBUG
                    if (k_tile_base == 256 && stage == 1) {  // Second tile with TileK=256
                        printf("DEBUG prefetch: A TMA load completed\n");
                    }
#endif
                }

                if (valid_sfa_m && valid_sfa_k) {
#ifndef NDEBUG
                    if (k_tile_base == 256 && stage == 1) {  // Second tile with TileK=256
                        printf("DEBUG prefetch: About to TMA load SFA, sfa_c0=%u sfa_c1=%u valid_sfa_k=%d valid_sfa_m=%d\n",
                               sfa_c0, sfa_c1, valid_sfa_k, valid_sfa_m);
                    }
#endif
                    tma_load_2d_no_arrive(
                        sfa_stage[stage],
                        desc_SFA,
                        sfa_c0, sfa_c1,
                        mbar_stage(mbar_a, stage),
                        L
                    );
#ifndef NDEBUG
                    if (k_tile_base == 256 && stage == 1) {  // Second tile with TileK=256
                        printf("DEBUG prefetch: SFA TMA load completed\n");
                    }
#endif
                }
            } else {
                // Load A matrix tile (3D)
                uint32_t c0 = c_k_packed;
                uint32_t c1 = c_m;
                uint32_t c2 = static_cast<uint32_t>(batch);
                bool valid_batch = c2 < static_cast<uint32_t>(L);
                bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);
                bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);

                // Load SFA scales tile (3D) using element-space coordinates
                // TMA requires coordinates to be aligned to box size
                uint32_t sfa_c0 = (c_k_scales / SfaBoxK) * SfaBoxK;  // Align to SfaBoxK
                uint32_t sfa_c1 = c_m;
                uint32_t sfa_c2 = c2;
                bool valid_sfa_k = (sfa_c0 + SfaBoxK) <= static_cast<uint32_t>(K_scales_padded);
                bool valid_sfa_m = (c_m + TileM) <= static_cast<uint32_t>(M);

                // Calculate total bytes for mbarrier
                uint32_t total_bytes = 0;
                if (valid_batch && valid_m && valid_k) {
                    total_bytes += TileM * TileKPacked;
                }
                if (valid_batch && valid_sfa_m && valid_sfa_k) {
                    total_bytes += TileM * SfaBoxK;
                }

                // Set mbarrier expected bytes
                mbarrier_arrive_expect_tx(mbar_stage(mbar_a, stage), total_bytes);

                // Issue TMA loads (they will complete to the same mbarrier)
                if (valid_batch && valid_m && valid_k) {
                    tma_load_3d_no_arrive(
                        a_packed_stage[stage],
                        desc_A,
                        c0, c1, c2,
                        mbar_stage(mbar_a, stage),
                        L
                    );
                }

                if (valid_batch && valid_sfa_m && valid_sfa_k) {
                    tma_load_3d_no_arrive(
                        sfa_stage[stage],
                        desc_SFA,
                        sfa_c0, sfa_c1, sfa_c2,
                        mbar_stage(mbar_a, stage),
                        L
                    );
                }
            }
        }
    }
}
#endif

template<int TileM, int TileK, int Threads>
__global__ void __launch_bounds__(Threads)
fp4_gemv_streaming(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SFA_packed,
    const uint8_t* __restrict__ SFB_packed,
    const CUtensorMap* __restrict__ desc_A,
    const CUtensorMap* __restrict__ desc_SFA,
    const CUtensorMap* __restrict__ desc_B,
    const CUtensorMap* __restrict__ desc_SFB,
    half* __restrict__ D,
    const int M, const int K, const int L, const int K_scales_padded
) {
#if __CUDA_ARCH__ >= 700
    constexpr int TileKPacked = TileK / 2;
    constexpr int TileScaleCount = TileK / 16;
    constexpr int SfaBoxK = 16;  // TMA minimum box size constraint  // TMA box size - reverted to 16 due to TMA descriptor constraints
    constexpr int StageCount = 3;
    constexpr int a_stride = TileK + 8;
    constexpr int ProducerThreads = 64;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const bool is_producer = warp_id < 2;
    const bool is_consumer = warp_id >= 2;

    const int batch = blockIdx.y;
    const int m_tile = blockIdx.x * TileM;
    if (batch >= L || m_tile >= M) return;

    const int K_packed = K >> 1;
    // K_scales_padded is passed as parameter - use it for tensor access and validity checks
    // Keep K_scales as local variable for lambda capture compatibility (though we use K_scales_padded for SFA)
    const int K_scales = K >> 4;  // Used for SFB (not padded) - lambda needs this
    const int tile_rows = (M - m_tile) < TileM ? (M - m_tile) : TileM;

    const uint8_t* A_batch = A_packed + static_cast<size_t>(batch) * M * K_packed;
    const uint8_t* B_batch = B_packed + static_cast<size_t>(batch) * 128 * K_packed;
    const uint8_t* SFA_batch = SFA_packed + static_cast<size_t>(batch) * M * K_scales_padded;
    const uint8_t* SFB_batch = SFB_packed + static_cast<size_t>(batch) * 128 * (K >> 4);
    half* D_batch = D + static_cast<size_t>(batch) * M;

    extern __shared__ uint8_t smem[];

#ifndef NDEBUG
    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        DEBUG_PRINT_ERROR("KERNEL STARTING: grid=(%d,%d,%d) blockDim.x=%d M=%d K=%d L=%d\n",
                          gridDim.x, gridDim.y, gridDim.z, blockDim.x, M, K, L);
    }
    if (tid == 0) {
        DEBUG_PRINT_ERROR("DBG block=(%d,%d,%d) tid=%d warp_id=%d lane_id=%d batch=%d m_tile=%d tile_rows=%d M=%d K=%d L=%d K_packed=%d K_scales_padded=%d\n",
                          blockIdx.x, blockIdx.y, blockIdx.z,
                          tid, warp_id, lane_id,
                          batch, m_tile, tile_rows,
                          M, K, L, K_packed, K_scales_padded);
        DEBUG_PRINT_ERROR("DBG A_batch=%p B_batch=%p SFA_batch=%p SFB_batch=%p D_batch=%p smem=%p\n",
                          (const void*)A_batch, (const void*)B_batch,
                          (const void*)SFA_batch, (const void*)SFB_batch,
                          (const void*)D_batch, (void*)smem);
    }
#endif

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

    // Helper to align the *absolute* shared address (cvta_to_shared) to 128 bytes
    auto align_up_smem_128 = [&] __device__() {
        uint32_t addr = cvta_to_shared_u32(smem + offset);
        uint32_t aligned = (addr + 127u) & ~127u;
        offset += static_cast<size_t>(aligned - addr);
    };

    // Helper to align to 1024 bytes (required for SWIZZLE_128B TMA)
    auto align_up_smem_1024 = [&] __device__() {
        uint32_t addr = cvta_to_shared_u32(smem + offset);
        uint32_t aligned = (addr + 1023u) & ~1023u;
        offset += static_cast<size_t>(aligned - addr);
    };

    uint64_t* mbar_a = alloc_mbar_array(StageCount);
    uint64_t* mbar_b = alloc_mbar_array(1);
    uint64_t* mbar_sfb = alloc_mbar_array(1);

    // TMA requires 128-byte alignment for destination addresses
    align_up_smem_128();

    uint8_t* b_packed_smem = smem + offset;
    offset += K_packed;

    align_up_smem_128();

    uint8_t* sfb_smem = smem + offset;
    offset += (K >> 4);  // SFB scale factors (not padded)
    offset = align_up(offset, 16);

    half* b_vec_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(K) * sizeof(half);
    offset = align_up(offset, 16);

    // Ensure TMA destinations with SWIZZLE_128B are 1024-byte aligned in shared memory
    align_up_smem_1024();
    uint8_t* a_packed_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        a_packed_stage[s] = smem + offset;
        offset += TileM * TileKPacked;
    }

    // Scales with SWIZZLE_128B TMA also need 1024-byte alignment
    align_up_smem_1024();
    uint8_t* sfa_stage[StageCount];
    // For rank-2 (L=1): allocate TileM * 16 (matching TMA box_k)
    // For rank-3 (L>1): allocate TileM * 128 (matching TMA box_k)
    int sfa_stage_stride = (L == 1) ? (TileM * 16) : (TileM * 128);
    for (int s = 0; s < StageCount; ++s) {
        sfa_stage[s] = smem + offset;
        offset += sfa_stage_stride;
    }

    // Tensor Core operand tiles in shared memory: also 128-byte aligned
    align_up_smem_128();
    half* a_f16_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileM) * a_stride * sizeof(half);
    align_up_smem_128();
    half* b_tile_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileK) * 8 * sizeof(half);

    (void)offset;

#if __CUDA_ARCH__ >= 900
    const bool use_tma_a = (desc_A != nullptr);
#ifndef NDEBUG
    if (tid == 0 && blockIdx.x == 0) {
        DEBUG_PRINT_ERROR("CUDA_ARCH=%d use_tma_a=%d desc_A=%p\n", __CUDA_ARCH__, use_tma_a, (const void*)desc_A);
    }
#endif
#else
    const bool use_tma_a = false;
#ifndef NDEBUG
    if (tid == 0 && blockIdx.x == 0) {
        DEBUG_PRINT_ERROR("CUDA_ARCH=%d < 900, TMA disabled\n", __CUDA_ARCH__);
    }
#endif
#endif

    __shared__ uint32_t stage_phase_smem[StageCount];

    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            stage_phase_smem[s] = 0;
        }
        __threadfence_block();  // Ensure stage_phase init visible to all warps
    }

#ifndef NDEBUG
    // Verify alignment of shared-memory regions
    // SWIZZLE_128B TMA regions need 1024-byte alignment
    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            uint32_t addr = cvta_to_shared_u32(a_packed_stage[s]);
            if (addr % 1024 != 0) {
                printf("SMEM ALIGN ERROR: a_packed_stage[%d] addr=%u not 1024-byte aligned (mod1024=%u)\n",
                       s, addr, addr % 1024);
            }
            uint32_t sfa_addr = cvta_to_shared_u32(sfa_stage[s]);
            if (sfa_addr % 1024 != 0) {
                printf("SMEM ALIGN ERROR: sfa_stage[%d] addr=%u not 1024-byte aligned (mod1024=%u)\n",
                       s, sfa_addr, sfa_addr % 1024);
            }
        }
        uint32_t a_f16_addr = cvta_to_shared_u32(a_f16_smem);
        if (a_f16_addr % 128 != 0) {
            DEBUG_PRINT_ERROR("SMEM ALIGN ERROR: a_f16_smem addr=%u not 128-byte aligned (mod128=%u)\n",
                              a_f16_addr, a_f16_addr % 128);
        }
        uint32_t b_tile_addr = cvta_to_shared_u32(b_tile_smem);
        if (b_tile_addr % 128 != 0) {
            DEBUG_PRINT_ERROR("SMEM ALIGN ERROR: b_tile_smem addr=%u not 128-byte aligned (mod128=%u)\n",
                              b_tile_addr, b_tile_addr % 128);
        }
    }
#endif

#if __CUDA_ARCH__ >= 900
    if (tid == 0 && use_tma_a) {
        // RANK-3 ONLY: Cluster fence (CUTLASS pattern from barrier.h)
        if (L > 1) {
            asm volatile(
                "{\n\t"
                "fence.mbarrier_init.release.cluster; \n"
                "}"
                ::
                : "memory"
            );
        }

        for (int s = 0; s < StageCount; ++s) {
            mbarrier_init(mbar_stage(mbar_a, s));
        }
        // Initialize barriers for B and SFB
        mbarrier_init(mbar_b);
        mbarrier_init(mbar_sfb);
    }
#endif
    __syncthreads();

    // Verify TMA descriptors are valid
    if (!desc_A || !desc_SFA || !desc_B || !desc_SFB) {
#ifndef NDEBUG
        if (tid == 0) {
            DEBUG_PRINT_ERROR("ERROR: TMA descriptors are null! desc_A=%p desc_SFA=%p desc_B=%p desc_SFB=%p\n",
                              (const void*)desc_A, (const void*)desc_SFA,
                              (const void*)desc_B, (const void*)desc_SFB);
        }
#endif
        return;  // Abort kernel execution
    }

#if __CUDA_ARCH__ >= 900
    // Use TMA for B and SFB - need proper barrier management for multi-tile loads
    constexpr int kTMABoxLimit = 256;
    const int b_chunks = (K_packed + (kTMABoxLimit - 1)) / kTMABoxLimit;
    const int sfb_chunks = ((K >> 4) + (kTMABoxLimit - 1)) / kTMABoxLimit;  // SFB not padded

    // Calculate total expected bytes for all chunks
    uint32_t total_b_bytes = static_cast<uint32_t>(K_packed);
    uint32_t total_sfb_bytes = static_cast<uint32_t>(K >> 4);  // SFB not padded

    // RANK-2 (L=1): Original code path - DO NOT TOUCH
    if (L == 1) {
        if (warp_id == 0 && lane_id == 0) {
            mbarrier_arrive_expect_tx(mbar_b, total_b_bytes);
            mbarrier_arrive_expect_tx(mbar_sfb, total_sfb_bytes);
        }
    }
    // RANK-3 (L>1): Cluster barrier setup - Only CTA0 sets expect_tx
    else {
        uint32_t cta_rank = blockIdx.x % 2;
        if (warp_id == 0 && lane_id == 0 && cta_rank == 0) {
            // Only CTA0 sets expect_tx on its local barrier
            // TMA from both CTAs will route to CTA0's barrier via peer mask
            // Apply Sm100MmaPeerBitMask to barrier addresses (FIX #2)
            constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;
            uint32_t mbar_b_addr = cvta_to_shared_u32(mbar_b) & Sm100MmaPeerBitMask;
            uint32_t mbar_sfb_addr = cvta_to_shared_u32(mbar_sfb) & Sm100MmaPeerBitMask;

            asm volatile(
                "{\n\t"
                "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1; \n\t"
                "}"
                :
                : "r"(mbar_b_addr), "r"(total_b_bytes)
                : "memory"
            );

            asm volatile(
                "{\n\t"
                "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1; \n\t"
                "}"
                :
                : "r"(mbar_sfb_addr), "r"(total_sfb_bytes)
                : "memory"
            );
        }
    }

    // Issue TMA loads
    if (warp_id == 0 && lane_id == 0) {

        // Issue all TMA loads (barrier expects total bytes from all loads)
        for (int tile_idx = 0; tile_idx < b_chunks; ++tile_idx) {
            int elem_offset = tile_idx * kTMABoxLimit;
            int chunk_size = (elem_offset + kTMABoxLimit <= K_packed) ? kTMABoxLimit : (K_packed - elem_offset);
            (void)chunk_size;
            uint8_t* dst = b_packed_smem + elem_offset;

            uint32_t smem_addr = cvta_to_shared_u32(dst);

            if (L == 1) {
                // Rank-2: CTA launch (no cluster)
                uint32_t mbar_addr = cvta_to_shared_u32(mbar_b);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%3, %4}], [%2];\n"
                    :
                    : "r"(smem_addr), "l"(desc_B), "r"(mbar_addr),
                      "r"(static_cast<uint32_t>(elem_offset)), "r"(0u)
                    : "memory"
                );
            } else {
                // Rank-3: Cluster launch with cta_group::2
                constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;
                uint32_t mbar_addr = cvta_to_shared_u32(mbar_b) & Sm100MmaPeerBitMask;
                uint64_t cache_hint = 0;
                asm volatile(
                    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                    "[%0], [%1, {%3, %4, %5}], [%2], %6;\n"
                    :
                    : "r"(smem_addr), "l"(desc_B), "r"(mbar_addr),
                      "r"(static_cast<uint32_t>(elem_offset)), "r"(0u), "r"(static_cast<uint32_t>(batch)),
                      "l"(cache_hint)
                    : "memory"
                );
            }
        }

        for (int tile_idx = 0; tile_idx < sfb_chunks; ++tile_idx) {
            int elem_offset = tile_idx * kTMABoxLimit;
            const int K_sfb = (K >> 4);  // SFB not padded
            int chunk_size = (elem_offset + kTMABoxLimit <= K_sfb) ? kTMABoxLimit : (K_sfb - elem_offset);
            (void)chunk_size;
            uint8_t* dst = sfb_smem + elem_offset;

            uint32_t smem_addr = cvta_to_shared_u32(dst);

            if (L == 1) {
                // Rank-2: CTA launch (no cluster)
                uint32_t mbar_addr = cvta_to_shared_u32(mbar_sfb);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%3, %4}], [%2];\n"
                    :
                    : "r"(smem_addr), "l"(desc_SFB), "r"(mbar_addr),
                      "r"(static_cast<uint32_t>(elem_offset)), "r"(0u)
                    : "memory"
                );
            } else {
                // Rank-3: Cluster launch with cta_group::2
                constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;
                uint32_t mbar_addr = cvta_to_shared_u32(mbar_sfb) & Sm100MmaPeerBitMask;
                uint64_t cache_hint = 0;
                asm volatile(
                    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                    "[%0], [%1, {%3, %4, %5}], [%2], %6;\n"
                    :
                    : "r"(smem_addr), "l"(desc_SFB), "r"(mbar_addr),
                      "r"(static_cast<uint32_t>(elem_offset)), "r"(0u), "r"(static_cast<uint32_t>(batch)),
                      "l"(cache_hint)
                    : "memory"
                );
            }
        }
    }

    // All threads wait on barriers
    if (L == 1) {
        // RANK-2: Wait locally (DO NOT TOUCH)
        mbarrier_wait_parity(mbar_b, 0);
        mbarrier_wait_parity(mbar_sfb, 0);
    } else {
        // RANK-3: All CTAs wait on CTA0's barrier using mapa
        uint32_t cta_rank = blockIdx.x % 2;
        uint32_t mbar_b_addr_cta0;
        uint32_t mbar_sfb_addr_cta0;
        constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFF;

        if (cta_rank == 0) {
            // CTA0: use local address with peer mask (FIX #3)
            mbar_b_addr_cta0 = cvta_to_shared_u32(mbar_b) & Sm100MmaPeerBitMask;
            mbar_sfb_addr_cta0 = cvta_to_shared_u32(mbar_sfb) & Sm100MmaPeerBitMask;
        } else {
            // CTA1: use mapa to get CTA0's remote barrier address (FIXED - NO MASKING!)
            uint32_t mbar_b_local = cvta_to_shared_u32(mbar_b);  // ✓ NO MASKING for mapa!
            uint32_t mbar_sfb_local = cvta_to_shared_u32(mbar_sfb);  // ✓ NO MASKING for mapa!
            uint32_t cta0_id = 0;

            // mapa will handle the CTA ID translation internally
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(mbar_b_addr_cta0)
                : "r"(mbar_b_local), "r"(cta0_id)
            );

            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(mbar_sfb_addr_cta0)
                : "r"(mbar_sfb_local), "r"(cta0_id)
            );
        }

        // Wait on CTA0's barrier - address from mapa encodes cluster scope, use generic .shared
        asm volatile(
            "{\n\t"
            ".reg .pred P1; \n\t"
            "LAB_WAIT_B: \n\t"
            "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1; \n\t"
            "@!P1 bra.uni LAB_WAIT_B; \n\t"
            "}"
            :
            : "r"(mbar_b_addr_cta0), "r"(0)
            : "memory"
        );

        asm volatile(
            "{\n\t"
            ".reg .pred P1; \n\t"
            "LAB_WAIT_SFB: \n\t"
            "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1; \n\t"
            "@!P1 bra.uni LAB_WAIT_SFB; \n\t"
            "}"
            :
            : "r"(mbar_sfb_addr_cta0), "r"(0)
            : "memory"
        );
    }
#endif
    __syncthreads();

        // DEBUG: Print first 4 SFB scale bytes and first 4 B packed bytes (first tile only)
        // DISABLED FOR PERFORMANCE: Excessive kernel printf statements cause hang
        // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        // printf("=== KERNEL DEBUG (block 0,0) ===\n");
        // printf("SFB raw bytes [0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
        //        sfb_smem[0], sfb_smem[1], sfb_smem[2], sfb_smem[3]);
        // printf("SFB decoded FP8 [0-3]: %.6f %.6f %.6f %.6f\n",
        //        decode_fp8_e4m3(sfb_smem[0]), decode_fp8_e4m3(sfb_smem[1]),
        //        decode_fp8_e4m3(sfb_smem[2]), decode_fp8_e4m3(sfb_smem[3]));
        // printf("B packed bytes [0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
        //        b_packed_smem[0], b_packed_smem[1], b_packed_smem[2], b_packed_smem[3]);
        //
        // const int K_sfb = (K >> 4);  // SFB not padded
        // for (int i = 0; i < 4; i++) {
        //     size_t sfb_idx = static_cast<size_t>(batch) * 128 * K_sfb + static_cast<size_t>(i);
        //     printf("SFB contiguous_idx[row=0,col=%d,batch=%d] = %llu\n",
        //            i, batch, (unsigned long long)sfb_idx);
        // }
        // }
    __syncthreads();

    // Decode B vector
    const int K_sfb_decode = (K >> 4);  // SFB not padded
    for (int idx = tid; idx < K_packed; idx += Threads) {
        int k_base = idx * 2;
        int scale_idx = idx >> 3;
        DEBUG_OOB_SMEM_1D("b_packed_smem", idx, K_packed, b_packed_smem);
        uint8_t packed = b_packed_smem[idx];
        half scale_h = __float2half(0.0f);
        if (scale_idx < K_sfb_decode) {
            DEBUG_OOB_SMEM_1D("sfb_smem", scale_idx, K_sfb_decode, sfb_smem);
            scale_h = __float2half(decode_fp8_e4m3(sfb_smem[scale_idx]));
        }
        // FIXED: LOW nibble = element 0, HIGH nibble = element 1 (matches reference implementation)
        half v0 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);         // LOW nibble -> element 0
        half v1 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);  // HIGH nibble -> element 1

#ifndef NDEBUG
        // Debug: print first few B decode values
        if (blockIdx.x == 0 && blockIdx.y == 0 && idx < 4) {
            DEBUG_PRINT_ERROR("B_decode: idx=%d packed=0x%02x scale_idx=%d scale=%.4f v0=%.4f v1=%.4f\n",
                              idx, packed, scale_idx, __half2float(scale_h), __half2float(v0), __half2float(v1));
        }
#endif

        if (k_base < K) {
            DEBUG_OOB_SMEM_1D("b_vec_smem", k_base, K, b_vec_smem);
            b_vec_smem[k_base] = v0;
        } else {
#ifndef NDEBUG
            DEBUG_PRINT_ERROR("WARN: b_vec_smem store OOB: k_base=%d K=%d at %s:%d\n",
                              k_base, K, __FILE__, __LINE__);
#endif
        }
        if (k_base + 1 < K) {
            DEBUG_OOB_SMEM_1D("b_vec_smem", k_base + 1, K, b_vec_smem);
            b_vec_smem[k_base + 1] = v1;
        } else {
#ifndef NDEBUG
            DEBUG_PRINT_ERROR("WARN: b_vec_smem store OOB: k_base+1=%d K=%d at %s:%d\n",
                              k_base + 1, K, __FILE__, __LINE__);
#endif
        }
    }
    __syncthreads();

    // DEBUG: Print first 8 decoded b_vec_smem values (first tile only)
    // DISABLED FOR PERFORMANCE: Excessive kernel printf statements cause hang
    // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    //     printf("b_vec_smem decoded [0-7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
    //            __half2float(b_vec_smem[0]), __half2float(b_vec_smem[1]),
    //            __half2float(b_vec_smem[2]), __half2float(b_vec_smem[3]),
    //            __half2float(b_vec_smem[4]), __half2float(b_vec_smem[5]),
    //            __half2float(b_vec_smem[6]), __half2float(b_vec_smem[7]));
    //     printf("DEBUG: After b_vec_smem decode print, about to syncthreads\n");
    // }
    __syncthreads();

#ifndef NDEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("DEBUG: About to call first prefetch_tile, use_tma_a=%d is_producer=%d m_tile=%d K_packed=%d\n",
               use_tma_a, is_producer, m_tile, K_packed);
    }
#endif

    // Prefetch first tiles using TMA
    prefetch_tile<TileM, TileK>(
        0, 0,
        use_tma_a, is_producer, warp_id, lane_id,
        m_tile, K_packed, K_scales_padded, M, L, batch,
        a_packed_stage, sfa_stage, mbar_a,
        desc_A, desc_SFA
    );

#ifndef NDEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("DEBUG: First prefetch_tile completed\n");
    }
#endif
    if (TileK < K) {
#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("DEBUG: About to call second prefetch_tile, TileK=%d K=%d\n", TileK, K);
        }
#endif
        prefetch_tile<TileM, TileK>(
            1, TileK,
            use_tma_a, is_producer, warp_id, lane_id,
            m_tile, K_packed, K_scales_padded, M, L, batch,
            a_packed_stage, sfa_stage, mbar_a,
            desc_A, desc_SFA
        );
#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("DEBUG: Second prefetch_tile completed\n");
        }
#endif
    }

#ifndef NDEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("DEBUG: About to enter K-tile loop\n");
    }
#endif

    float c_frag_0 = 0.0f;
    float c_frag_1 = 0.0f;
    float c_frag_2 = 0.0f;
    float c_frag_3 = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TileK) {
        int tile_idx = k_tile / TileK;
        int stage = tile_idx % StageCount;

#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("K-TILE LOOP START: k_tile=%d tile_idx=%d stage=%d K=%d TileK=%d\n",
                   k_tile, tile_idx, stage, K, TileK);
        }
#endif

        int next_stage = (stage + 2) % StageCount;
        int next_k = k_tile + 2 * TileK;

#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("K-TILE: About to prefetch, next_k=%d next_stage=%d (condition: %d)\n",
                   next_k, next_stage, next_k < K);
        }
#endif

        if (next_k < K) {
            prefetch_tile<TileM, TileK>(
                next_stage, next_k,
                use_tma_a, is_producer, warp_id, lane_id,
                m_tile, K_packed, K_scales_padded, M, L, batch,
                a_packed_stage, sfa_stage, mbar_a,
                desc_A, desc_SFA
            );
#ifndef NDEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
                printf("K-TILE: Prefetch completed for next_k=%d\n", next_k);
            }
#endif
        }

#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("K-TILE: About to wait on mbarrier, use_tma_a=%d stage=%d\n", use_tma_a, stage);
        }
#endif

        if (use_tma_a) {
#if __CUDA_ARCH__ >= 900
            mbarrier_wait_parity(mbar_stage(mbar_a, stage), stage_phase_smem[stage]);
#ifndef NDEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
                printf("K-TILE: Mbarrier wait completed, about to syncthreads\n");
            }
#endif
            __syncthreads();
#ifndef NDEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
                printf("K-TILE: After syncthreads, updating phase\n");
            }
#endif
            if (tid == 0) {
                stage_phase_smem[stage] ^= 1;
            }
#endif
        } else {
            cp_async_wait();
        }
        __syncthreads();

#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("K-TILE: After final syncthreads, starting decode at k_tile=%d\n", k_tile);
        }
#endif

#ifndef NDEBUG
        // Dump first few bytes of SFA shared memory after TMA load
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0) {
            printf("SFA_SMEM_DUMP stage=%d:\n", stage);
            printf("  [0-15]  = %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                   sfa_stage[stage][0], sfa_stage[stage][1], sfa_stage[stage][2], sfa_stage[stage][3],
                   sfa_stage[stage][4], sfa_stage[stage][5], sfa_stage[stage][6], sfa_stage[stage][7],
                   sfa_stage[stage][8], sfa_stage[stage][9], sfa_stage[stage][10], sfa_stage[stage][11],
                   sfa_stage[stage][12], sfa_stage[stage][13], sfa_stage[stage][14], sfa_stage[stage][15]);
            printf("  [16-31] = %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                   sfa_stage[stage][16], sfa_stage[stage][17], sfa_stage[stage][18], sfa_stage[stage][19],
                   sfa_stage[stage][20], sfa_stage[stage][21], sfa_stage[stage][22], sfa_stage[stage][23],
                   sfa_stage[stage][24], sfa_stage[stage][25], sfa_stage[stage][26], sfa_stage[stage][27],
                   sfa_stage[stage][28], sfa_stage[stage][29], sfa_stage[stage][30], sfa_stage[stage][31]);
            printf("  [32-47] = %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                   sfa_stage[stage][32], sfa_stage[stage][33], sfa_stage[stage][34], sfa_stage[stage][35],
                   sfa_stage[stage][36], sfa_stage[stage][37], sfa_stage[stage][38], sfa_stage[stage][39],
                   sfa_stage[stage][40], sfa_stage[stage][41], sfa_stage[stage][42], sfa_stage[stage][43],
                   sfa_stage[stage][44], sfa_stage[stage][45], sfa_stage[stage][46], sfa_stage[stage][47]);
        }
#endif

        int curr_k = (K - k_tile) < TileK ? (K - k_tile) : TileK;
        int curr_cols = (curr_k + 1) >> 1;
        int scale_count = (curr_k + 15) >> 4;

        {
            uint8_t* a_stage = a_packed_stage[stage];

#ifndef NDEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
                printf("DECODE LOOP START: k_tile=%d tile_rows=%d curr_cols=%d loop_limit=%d\n",
                       k_tile, tile_rows, curr_cols, tile_rows * curr_cols);
            }
#endif

            for (int idx = tid; idx < tile_rows * curr_cols; idx += Threads) {
                int row = idx / curr_cols;
                int col_packed = idx - row * curr_cols;
                int k_base = k_tile + col_packed * 2;
                int a_smem_idx = row * TileKPacked + col_packed;
                DEBUG_OOB_SMEM_1D("a_packed_stage", a_smem_idx, TileM * TileKPacked, a_stage);
                uint8_t packed = a_stage[a_smem_idx];
                half scale_h = __float2half(0.0f);
                if (row < tile_rows) {
                    int scale_col = col_packed >> 3;
                    if (scale_col < scale_count) {
                        // For rank-2 (L=1): sfa_stage stride is SfaBoxK=16 (loaded width)
                        // For rank-3 (L>1): sfa_stage stride is SfaBoxK=128 (loaded width)
                        int sfa_stride = (L == 1) ? 16 : 128;
                        int sfa_idx = row * sfa_stride + scale_col;
                        int sfa_size = TileM * sfa_stride;
                        DEBUG_OOB_SMEM_1D("sfa_stage", sfa_idx, sfa_size, sfa_stage[stage]);
                        scale_h = __float2half(decode_fp8_e4m3(sfa_stage[stage][sfa_idx]));
                    }
                }
                half v0 = __float2half(0.0f);
                half v1 = __float2half(0.0f);
                // FIXED: LOW nibble = element 0, HIGH nibble = element 1 (matches reference implementation)
                if (row < tile_rows && k_base < K) {
                    v0 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);         // LOW nibble -> element 0
                }
                if (row < tile_rows && (k_base + 1) < K) {
                    v1 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);  // HIGH nibble -> element 1
                }

#ifndef NDEBUG
                // Debug: print A decode for first few rows at col=0
                if (blockIdx.x == 0 && blockIdx.y == 0 && col_packed == 0 && row < 5) {
                    int sfa_idx_check = row * K_scales_padded + (col_packed >> 3);
                    uint8_t sfa_byte = sfa_stage[stage][sfa_idx_check];
                    DEBUG_PRINT_ERROR("A_decode k_tile=%d stage=%d: row=%d col=%d k_base=%d packed=0x%02x scale=%.4f sfa_idx=%d sfa_byte=0x%02x v0=%.4f v1=%.4f\n",
                                      k_tile, stage, row, col_packed, k_base, packed, __half2float(scale_h), sfa_idx_check, sfa_byte, __half2float(v0), __half2float(v1));
                }
#endif

                half* a_dst = a_f16_smem + row * a_stride;
                int dst_idx0 = col_packed * 2;
                int dst_idx1 = col_packed * 2 + 1;
                DEBUG_OOB_SMEM_1D("a_f16_smem", row * a_stride + dst_idx0, TileM * a_stride, a_f16_smem);
                a_dst[dst_idx0] = v0;
                DEBUG_OOB_SMEM_1D("a_f16_smem", row * a_stride + dst_idx1, TileM * a_stride, a_f16_smem);
                a_dst[dst_idx1] = v1;
            }
        }
        __syncthreads();

        // DEBUG: Print first 8 decoded A[row=0-7] values and first 4 SFA raw bytes (first tile only)
        // DISABLED FOR PERFORMANCE: Excessive kernel printf statements cause hang
        // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0) {
        //     printf("SFA raw bytes [row=0, 0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
        //            sfa_stage[stage][0], sfa_stage[stage][1],
        //            sfa_stage[stage][2], sfa_stage[stage][3]);
        //     printf("SFA decoded FP8 [row=0, 0-3]: %.6f %.6f %.6f %.6f\n",
        //            decode_fp8_e4m3(sfa_stage[stage][0]), decode_fp8_e4m3(sfa_stage[stage][1]),
        //            decode_fp8_e4m3(sfa_stage[stage][2]), decode_fp8_e4m3(sfa_stage[stage][3]));
        //     printf("A packed bytes [row=0, 0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
        //            a_packed_stage[stage][0], a_packed_stage[stage][1],
        //            a_packed_stage[stage][2], a_packed_stage[stage][3]);
        //
        //     // Print decoded A for rows 0-7 (first 8 elements of each row)
        //     for (int r = 0; r < 8; r++) {
        //         half* a_row = a_f16_smem + r * a_stride;
        //         printf("A_f16_smem[row=%d, 0-7]: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
        //                r,
        //                __half2float(a_row[0]), __half2float(a_row[1]),
        //                __half2float(a_row[2]), __half2float(a_row[3]),
        //                __half2float(a_row[4]), __half2float(a_row[5]),
        //                __half2float(a_row[6]), __half2float(a_row[7]));
        //     }
        //
        //     // Print CuTe blocked indices for SFA at this tile origin
        //     int nmblocks_sfa = (M + 127) / 128;
        //     int nkblocks_sfa = (K_scales_padded + 3) / 4;
        // for (int i = 0; i < 4; i++) {
        // size_t sfa_idx = static_cast<size_t>(batch) * M * K_scales_padded
        //                + static_cast<size_t>(m_tile) * K_scales_padded
        //                + static_cast<size_t>(i);
        // printf("SFA contiguous_idx[row=%d,col=%d,batch=%d] = %llu\n",
        //        m_tile, i, batch, (unsigned long long)sfa_idx);
        // }
        //     printf("=== END KERNEL DEBUG ===\n");
        // }
        __syncthreads();

        if (is_producer && warp_id == 1) {
            for (int kk = lane_id; kk < curr_k; kk += 32) {
                half v = __float2half(0.0f);
                if (k_tile + kk < K) {
                    int b_vec_idx = k_tile + kk;
                    DEBUG_OOB_SMEM_1D("b_vec_smem", b_vec_idx, K, b_vec_smem);
                    v = b_vec_smem[b_vec_idx];
                }
                half* b_row = b_tile_smem + kk * 8;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    int b_tile_idx = kk * 8 + n;
                    DEBUG_OOB_SMEM_1D("b_tile_smem", b_tile_idx, TileK * 8, b_tile_smem);
                    b_row[n] = v;
                }
            }
        }
        __syncthreads();

        int active_warps = (tile_rows + 15) / 16;
        if (is_consumer && (warp_id - 2) < active_warps) {
            int warp_row = (warp_id - 2) * 16;
            for (int kk = 0; kk < curr_k; kk += 16) {
                // Load A matrix tile (16x16) using ldmatrix
                // A is stored row-major with stride a_stride in shared memory
                // ldmatrix.m8n8.x4 loads 4 8x8 matrices
                //
                // For m16n8k16 MMA, A is 16 rows x 16 cols, arranged as:
                //   [A00 A01]  where each Aij is 8x8
                //   [A10 A11]
                //
                // ldmatrix.x4 addressing: thread T loads row (T%8) of matrix (T/8)
                // So threads 0-7: A00 rows 0-7, threads 8-15: A01 rows 0-7
                //    threads 16-23: A10 rows 0-7, threads 24-31: A11 rows 0-7

                const half* a_tile_ptr = a_f16_smem + warp_row * a_stride + kk;
                int a_tile_base_idx = warp_row * a_stride + kk;
                DEBUG_OOB_SMEM_1D("a_f16_smem_tile", a_tile_base_idx, TileM * a_stride, a_f16_smem);
                uint32_t a_base = cvta_to_shared_u32(a_tile_ptr);

                // Original ldmatrix.x4 addressing pattern
                int matrix_id = lane_id / 8;  // 0-3
                int row_in_matrix = lane_id % 8;  // 0-7
                int block_row = (matrix_id & 1) ? 8 : 0;
                int block_col = (matrix_id >= 2) ? 8 : 0;

                int a_row = block_row + row_in_matrix;
                int a_col = block_col;

                uint32_t a_addr = a_base + (static_cast<uint32_t>(a_row) * a_stride + a_col) * sizeof(half);
                unsigned a0, a1, a2, a3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(a_addr)
                );

#ifndef NDEBUG
                if (blockIdx.x == 0 && blockIdx.y == 0 && k_tile == 0 && kk == 0 && warp_id == 2 && lane_id < 16) {
                    // Print ldmatrix fragments for all lanes 0-15 to see full distribution pattern
                    // This helps understand SM_100a specific ldmatrix behavior
                    half* a0_h = reinterpret_cast<half*>(&a0);
                    half* a1_h = reinterpret_cast<half*>(&a1);
                    half* a2_h = reinterpret_cast<half*>(&a2);
                    half* a3_h = reinterpret_cast<half*>(&a3);
                    printf("LDMTX_SM100: lane=%d octet=%d a_row=%d a0=[%.2f,%.2f] a1=[%.2f,%.2f] a2=[%.2f,%.2f] a3=[%.2f,%.2f]\n",
                           lane_id, lane_id/4, a_row,
                           __half2float(a0_h[0]), __half2float(a0_h[1]),
                           __half2float(a1_h[0]), __half2float(a1_h[1]),
                           __half2float(a2_h[0]), __half2float(a2_h[1]),
                           __half2float(a3_h[0]), __half2float(a3_h[1]));
                }
#endif

                // Load B matrix tile (16x8) using ldmatrix with transpose
                // B is stored as 16 rows x 8 cols (but we broadcast the vector to 8 cols)
                // ldmatrix.x2.trans loads 2 8x8 matrices with transpose
                //
                // For m16n8k16, B is 16 cols x 8 rows (k=16, n=8), stored column-major
                // With .trans, it reads row-major data and transposes

                const half* b_tile_ptr = b_tile_smem + kk * 8;
                DEBUG_OOB_SMEM_1D("b_tile_smem_tile", kk * 8, TileK * 8, b_tile_smem);
                uint32_t b_base = cvta_to_shared_u32(b_tile_ptr);

                // For ldmatrix.x2.trans: threads 0-7 load first 8x8, threads 8-15 load second 8x8
                // But we only use first 16 threads for the 2 matrices
                int b_matrix_id = (lane_id / 8) % 2;  // 0 or 1
                int b_row_in_matrix = lane_id % 8;
                int b_row = b_matrix_id * 8 + b_row_in_matrix;

                uint32_t b_addr = b_base + static_cast<uint32_t>(b_row) * 8 * sizeof(half);
                unsigned b0, b1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [%2];\n"
                    : "=r"(b0), "=r"(b1)
                    : "r"(b_addr)
                );

                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{ %0, %1, %2, %3 }, "
                    "{ %4, %5, %6, %7 }, "
                    "{ %8, %9 }, "
                    "{ %0, %1, %2, %3 };\n"
                    : "+f"(c_frag_0), "+f"(c_frag_1), "+f"(c_frag_2), "+f"(c_frag_3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );

                // DEBUG: Print accumulator after first MMA in first K tile
                if (blockIdx.x == 0 && blockIdx.y == 0 && k_tile == 0 && kk == 0 && warp_id == 2 && lane_id == 0) {
                    printf("MMA DEBUG: k_tile=%d kk=%d warp=%d lane=%d c_frag=[%.4f, %.4f, %.4f, %.4f]\n",
                           k_tile, kk, warp_id, lane_id, c_frag_0, c_frag_1, c_frag_2, c_frag_3);
                }
            }
        }

        __syncthreads();

#ifndef NDEBUG
        // Debug accumulated values after each K tile
        if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 2 && lane_id == 0) {
            printf("ACCUM after k_tile=%d: c_frag_0=%.4f c_frag_2=%.4f\n",
                   k_tile, c_frag_0, c_frag_2);
        }
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("K-TILE LOOP END: Completed k_tile=%d, next will be k_tile=%d\n",
                   k_tile, k_tile + TileK);
        }
#endif
    }

#ifndef NDEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("EXITED K-TILE LOOP: All K tiles completed\n");
    }
#endif

    // DEBUG: Print ALL fragments for lanes 0,4,8 to understand output distribution
    // DISABLED FOR PERFORMANCE: Excessive kernel printf statements cause hang
    // if (blockIdx.x == 0 && blockIdx.y == 0 && is_consumer && warp_id == 2) {
    //     int octet = lane_id / 4;
    //     int tid_in_octet = lane_id % 4;
    //     if (tid_in_octet == 0 && (lane_id == 0 || lane_id == 4 || lane_id == 8)) {
    //         printf("FINAL_SM100: lane=%d octet=%d ALL_FRAGS: c0=%.2f c1=%.2f c2=%.2f c3=%.2f (rows %d,%d)\n",
    //                lane_id, octet, c_frag_0, c_frag_1, c_frag_2, c_frag_3,
    //                m_tile + (warp_id-2)*16 + octet, m_tile + (warp_id-2)*16 + octet + 8);
    //     }
    // }

    // m16n8k16.row.col.f32 output fragment layout per PTX ISA:
    // C matrix is 16 rows x 8 cols, distributed across 32 threads with 4 floats each.
    //
    // For thread T (0-31), the 4 output registers map to C matrix positions:
    //   Let octet = T / 4  (0-7, which octet of threads)
    //   Let tid_in_octet = T % 4  (0-3, position within octet)
    //
    //   c_frag_0: C[octet + 0][tid_in_octet * 2 + 0]  -> row=octet,     col=tid_in_octet*2
    //   c_frag_1: C[octet + 0][tid_in_octet * 2 + 1]  -> row=octet,     col=tid_in_octet*2+1
    //   c_frag_2: C[octet + 8][tid_in_octet * 2 + 0]  -> row=octet+8,   col=tid_in_octet*2
    //   c_frag_3: C[octet + 8][tid_in_octet * 2 + 1]  -> row=octet+8,   col=tid_in_octet*2+1
    //
    // So threads 0-3 cover rows 0,8 cols 0-7; threads 4-7 cover rows 1,9 cols 0-7; etc.
    //
    // For GEMV: B is broadcast to all 8 columns, so all columns of C are identical.
    // We only need col 0 values, held by threads where tid_in_octet == 0 (i.e., T % 4 == 0).
    // Those threads are: 0, 4, 8, 12, 16, 20, 24, 28
    // They hold rows: 0,8 | 1,9 | 2,10 | 3,11 | 4,12 | 5,13 | 6,14 | 7,15

    int active_warps_total = (tile_rows + 15) / 16;
    if (is_consumer && (warp_id - 2) < active_warps_total) {
        int warp_row_offset = (warp_id - 2) * 16;

        int octet = lane_id / 4;           // 0-7
        int tid_in_octet = lane_id % 4;    // 0-3

        // Row this thread's c_frag_0 and c_frag_2 correspond to
        int row0 = octet;      // c_frag_0 -> row 0-7
        int row1 = octet + 8;  // c_frag_2 -> row 8-15

        // Column this thread holds (for c_frag_0/c_frag_2)
        int col = tid_in_octet * 2;  // 0, 2, 4, or 6

#ifndef NDEBUG
        if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 2 && lane_id < 4) {
            DEBUG_PRINT_ERROR("PRE_STORE_CHECK: lane=%d octet=%d tid_in_oct=%d col=%d\n",
                              lane_id, octet, tid_in_octet, col);
        }
#endif

        // Since B was broadcast, all columns have same value.
        // We only write from threads holding col 0 to avoid duplication.
        if (col == 0) {
            int global_row0 = m_tile + warp_row_offset + row0;
            int global_row1 = m_tile + warp_row_offset + row1;

#ifndef NDEBUG
            // Debug ALL lanes that write (col==0) to see what they're storing
            if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 2) {
                DEBUG_PRINT_ERROR("D_STORE: warp=%d lane=%d octet=%d tid_in_oct=%d col=%d row0=%d row1=%d g0=%d g1=%d c0=%.2f c2=%.2f\n",
                                  warp_id, lane_id, octet, tid_in_octet, col, row0, row1,
                                  global_row0, global_row1, c_frag_0, c_frag_2);
            }
#endif

            if (global_row0 < M) {
                DEBUG_OOB_GLOBAL_1D("D_batch", global_row0, M, D_batch);
                D_batch[global_row0] = __float2half(c_frag_0);
            }
            if (global_row1 < M) {
                DEBUG_OOB_GLOBAL_1D("D_batch", global_row1, M, D_batch);
                D_batch[global_row1] = __float2half(c_frag_2);
            }
        }
    }

#ifndef NDEBUG
    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        DEBUG_PRINT_ERROR("KERNEL FINISHED: grid=(%d,%d,%d) blockDim.x=%d\n",
                          gridDim.x, gridDim.y, gridDim.z, blockDim.x);
    }
#endif
#endif
}

void launch_fp4_gemv_optimized(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor D,
    int64_t M, int64_t K, int64_t L, int64_t K_scales_padded
) {
    const uint8_t* A_ptr = A.data_ptr<uint8_t>();
    const uint8_t* B_ptr = B.data_ptr<uint8_t>();
    const uint8_t* SFA_ptr = SFA.data_ptr<uint8_t>();
    const uint8_t* SFB_ptr = SFB.data_ptr<uint8_t>();
    half* D_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    constexpr int kTileM = 128;
    constexpr int kTileK = 256;  // 256 elements = 128 packed bytes (required for SWIZZLE_128B)
    constexpr int kThreads = 320;
    constexpr int kTileKPacked = kTileK / 2;  // 128 bytes
    constexpr int kTileScaleCount = kTileK / 16;  // 16 scales
    // SFA box size is determined at runtime based on K_scales_padded (not a constant)
    constexpr int kStageCount = 3;
    constexpr int kAStride = kTileK + 8;

    // TMA hardware constraint: box dimensions must be <= 256 elements per dimension
    constexpr int kTMABoxLimit = 256;
    static_assert(kTileM <= kTMABoxLimit, "kTileM exceeds TMA box limit of 256");
    static_assert(kTileKPacked <= kTMABoxLimit, "kTileKPacked exceeds TMA box limit of 256");

    auto align_up = [](size_t x, size_t align) {
        return (x + align - 1) & ~(align - 1);
    };

    constexpr size_t kMbarBytes = 16;

    size_t shared_bytes = 0;

    auto alloc_mbar_array = [&](int count) {
        shared_bytes = align_up(shared_bytes, 16);
        shared_bytes += static_cast<size_t>(count) * kMbarBytes;
    };

    alloc_mbar_array(kStageCount); // mbar_a
    alloc_mbar_array(1);           // mbar_b
    alloc_mbar_array(1);           // mbar_sfb

    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 2;    // b_packed_smem
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 16;   // sfb_smem
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) * sizeof(__half); // b_vec_smem
    shared_bytes = align_up(shared_bytes, 16);

    // 1024-byte alignment for SWIZZLE_128B TMA regions
    shared_bytes = align_up(shared_bytes, 1024); // for a_packed_stage[0] (SWIZZLE_128B)
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileKPacked; // a_packed_stage[s]
    }

    shared_bytes = align_up(shared_bytes, 1024); // for sfa_stage[0] (SWIZZLE_128B)
    // SFA box size: use K_scales_padded (16 for rank-2, 128 for rank-3)
    // Allocate worst-case: 128 bytes per row
    constexpr int kSfaBoxKWorstCase = 128;
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kSfaBoxKWorstCase; // sfa_stage[s]
    }

    // 128-byte alignment for non-swizzled regions
    shared_bytes = align_up(shared_bytes, 128); // for a_f16_smem
    shared_bytes += static_cast<size_t>(kTileM) * kAStride * sizeof(__half); // a_f16_smem
    shared_bytes = align_up(shared_bytes, 128); // for b_tile_smem
    shared_bytes += static_cast<size_t>(kTileK) * 8 * sizeof(__half); // b_tile_smem

    auto check_cuda = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    };

    const uint64_t K_packed = static_cast<uint64_t>(K / 2);
    const uint64_t K_scales = static_cast<uint64_t>(K / 16);

    bool use_tma_a = true;

    // 1. Use heap-aligned allocation for CUtensorMap descriptor
    std::vector<uint8_t> map_buf(sizeof(CUtensorMap) + 128);
    void* raw_ptr = map_buf.data();
    size_t space = map_buf.size();
    void* aligned_ptr = std::align(128, sizeof(CUtensorMap), raw_ptr, space);
    if (!aligned_ptr) throw std::runtime_error("Failed to align CUtensorMap");

    // Zero-initialize the memory to prevent garbage values
    std::memset(aligned_ptr, 0, sizeof(CUtensorMap));
    CUtensorMap* map_A_ptr = reinterpret_cast<CUtensorMap*>(aligned_ptr);

    CUtensorMap* d_map_A = nullptr;
    CUtensorMap* d_map_SFA = nullptr;
    CUtensorMap* d_map_B = nullptr;
    CUtensorMap* d_map_SFB = nullptr;
    bool tma_ok = true;

    // 2. Ensure tensor's device pointer is 128-byte aligned for TMA
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(A_ptr);
    if ((base_addr % 128) != 0) {
        printf("ERROR: base pointer A not 128-byte aligned: %p (mod128=%llu)\n",
               A_ptr, (unsigned long long)(base_addr % 128));
        tma_ok = false;
    } else {
        printf("✅A_ptr 128-byte alignment check passed: %p\n", A_ptr);
    }

    // 3. Check that strides are properly 128-byte aligned
    if (K_packed % 128 != 0) {
        printf("WARNING: K_packed (%llu) is not 128-byte aligned, may cause TMA issues\n",
               (unsigned long long)K_packed);
    } else {
        printf("✅K_packed 128-byte stride alignment check passed: %llu bytes\n",
               (unsigned long long)K_packed);
    }

    // TMA encoder with SWIZZLE_128B for large 2D matrices (A, SFA)
    auto encode_tma_matrix = [&](CUtensorMap* out,
                          CUtensorMapDataType type,
                          cuuint32_t rank,
                          const void* base,
                          const cuuint64_t* dims,
                          const cuuint64_t* globalStrides,
                          const cuuint32_t* box) -> CUresult {

        cuuint32_t elementStrides[5] = {1, 1, 1, 1, 1};
        return cuTensorMapEncodeTiled(
            out,
            type,
            rank,
            const_cast<void*>(base),
            dims,
            globalStrides,
            box,
            elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,  // Optimize for memory bandwidth
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
    };

    // TMA encoder with SWIZZLE_NONE for vectors/small dimensions (B, SFB)
    auto encode_tma_vector = [&](CUtensorMap* out,
                          CUtensorMapDataType type,
                          cuuint32_t rank,
                          const void* base,
                          const cuuint64_t* dims,
                          const cuuint64_t* globalStrides,
                          const cuuint32_t* box) -> CUresult {

        cuuint32_t elementStrides[5] = {1, 1, 1, 1, 1};
        return cuTensorMapEncodeTiled(
            out,
            type,
            rank,
            const_cast<void*>(base),
            dims,
            globalStrides,
            box,
            elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,  // No swizzle for small dimensions
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
    };

    if (tma_ok) {
        // Ensure box dimensions don't exceed TMA hardware limit (256) or tensor dimensions
        // Both rank-2 and rank-3 use kTileKPacked (128 bytes) for box_k
        // The difference is in swizzle mode: SWIZZLE_NONE for rank-2, SWIZZLE_128B for rank-3
        cuuint32_t box_k = static_cast<cuuint32_t>(
            kTileKPacked < K_packed ?
                (kTileKPacked < kTMABoxLimit ? kTileKPacked : kTMABoxLimit) :
                (K_packed < kTMABoxLimit ? K_packed : kTMABoxLimit)
        );
        cuuint32_t box_m = static_cast<cuuint32_t>(
            kTileM < M ?
                (kTileM < kTMABoxLimit ? kTileM : kTMABoxLimit) :
                (M < kTMABoxLimit ? M : kTMABoxLimit)
        );

        printf("TMA Debug: A_ptr = %p, map_A_ptr = %p\n", A_ptr, (void*)map_A_ptr);

        // 4. Check box dimensions for good alignment (we prefer 128-byte multiples)
        if (box_k % 128 != 0) {
            printf("WARNING: box_k (%u) is not a multiple of 128 elements, may reduce TMA efficiency\n", box_k);
        }
        // box_m doesn't need to be 128-aligned since it's the number of rows, not bytes

        // 5. Check that tile sizes result in 128-byte aligned transfers
        uint32_t tile_bytes = box_m * box_k;
        if (tile_bytes % 128 != 0) {
            printf("WARNING: tile_bytes (%u) is not 128-byte aligned\n", tile_bytes);
        } else {
            printf("✅Tile size 128-byte alignment check passed: %u bytes\n", tile_bytes);
        }

        if (L == 1) {
            // Use 2D descriptor when no batching (L=1)
            cuuint64_t dims_A[2] = {
                static_cast<cuuint64_t>(K_packed),  // Innermost (fastest)
                static_cast<cuuint64_t>(M)          // Outermost (slowest)
            };

            cuuint32_t box_A[2] = {box_k, box_m};  // Match dims order

            // For 2D, globalStrides has 1 element: stride between rows (in bytes)
            cuuint64_t strides_A[1] = {
                static_cast<cuuint64_t>(K_packed)  // Row stride in bytes
            };

            // Rank-2 descriptor creation (silent)

            // Validate box dimensions
            if (box_A[0] > 256 || box_A[1] > 256) {
                printf("ERROR: TMA box dimension exceeds 256 limit! box=[%u, %u]\n",
                       box_A[0], box_A[1]);
                tma_ok = false;
            }

            if (tma_ok) {
                // For rank-2 (L=1): use SWIZZLE_NONE (CTA launch)
                CUresult resA = encode_tma_vector(map_A_ptr,
                                           CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                           2,  // rank=2 for 2D access
                                           A_ptr,
                                           dims_A,
                                           strides_A,  // Explicit stride, NOT nullptr
                                           box_A);

                printf("TMA Encode A Result: %d\n", (int)resA);
                if (resA != CUDA_SUCCESS) {
                    const char* err_str = nullptr;
                    cuGetErrorString(resA, &err_str);
                    printf("TMA Encode A failed: %s\n", err_str ? err_str : "unknown error");
                    tma_ok = false;
                } else {
                    // Rank-2 success (silent)
                }
            }
        }
        else {
            // Use 3D descriptor when batching (L > 1)
            // Tensor is [L, M, K_packed] row-major, TMA wants (innermost, ..., outermost)
            cuuint64_t dims_A[3] = {
                static_cast<cuuint64_t>(K_packed),  // Innermost (fastest varying)
                static_cast<cuuint64_t>(M),         // Middle
                static_cast<cuuint64_t>(L)          // Outermost (slowest varying)
            };

            // Box in same order as dims
            cuuint32_t box_A[3] = {box_k, box_m, 1u};

            // strides[0] = stride for dim 1 (M), strides[1] = stride for dim 2 (L)
            cuuint64_t strides_A[2] = {
                static_cast<cuuint64_t>(K_packed),           // Stride between rows (M dim)
                static_cast<cuuint64_t>(M) * K_packed        // Stride between batches (L dim)
            };

            printf("TMA Debug: Using RANK=3 for L=%lld\n", (long long)L);
            printf("TMA Debug: dims = [K_packed=%llu, M=%llu, L=%llu], box = [%u, %u, %u]\n",
                   (unsigned long long)dims_A[0], (unsigned long long)dims_A[1], (unsigned long long)dims_A[2],
                   box_A[0], box_A[1], box_A[2]);
            printf("TMA Debug: strides_A = [%llu, %llu]\n",
                   (unsigned long long)strides_A[0], (unsigned long long)strides_A[1]);

            // Validate box dimensions
            if (box_A[0] > 256 || box_A[1] > 256 || box_A[2] > 256) {
                printf("ERROR: TMA box dimension exceeds 256 limit! box=[%u, %u, %u]\n",
                       box_A[0], box_A[1], box_A[2]);
                tma_ok = false;
            }

            if (tma_ok) {
                // For rank-3 (L>1): use SWIZZLE_128B (cluster launch)
                CUresult resA = encode_tma_matrix(map_A_ptr,
                                           CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                           3,
                                           A_ptr,
                                           dims_A,
                                           strides_A,
                                           box_A);

                printf("TMA Encode A Result: %d\n", (int)resA);
                if (resA != CUDA_SUCCESS) {
                    const char* err_str = nullptr;
                    cuGetErrorString(resA, &err_str);
                    printf("TMA Encode A failed: %s\n", err_str ? err_str : "unknown error");
                    tma_ok = false;
                } else {
                    printf("✅TMA Encode A (rank=3, SWIZZLE_128B) SUCCESS!\n");
                }
            }
        }
    }

    if (tma_ok) {
        check_cuda(cudaMalloc(&d_map_A, sizeof(CUtensorMap)), "cudaMalloc d_map_A");
        check_cuda(cudaMemcpy(d_map_A, map_A_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_A");

        // Create TMA descriptors for B (vector broadcast to all batches)
        std::vector<uint8_t> map_B_buf(sizeof(CUtensorMap) + 128);
        void* map_B_raw = map_B_buf.data();
        size_t map_B_space = map_B_buf.size();
        void* map_B_aligned = std::align(128, sizeof(CUtensorMap), map_B_raw, map_B_space);
        if (!map_B_aligned) throw std::runtime_error("Failed to align map_B");
        std::memset(map_B_aligned, 0, sizeof(CUtensorMap));
        CUtensorMap* map_B_ptr = reinterpret_cast<CUtensorMap*>(map_B_aligned);

        if (L == 1) {
            // B: rank-2, actual layout is [1, 128, K_packed] but L=1 collapses to [128, K_packed]
            // TMA dims order: [K_packed, 128] (innermost to outermost)
            cuuint64_t dims_B[2] = {K_packed, 128};
            cuuint32_t box_B[2] = {static_cast<cuuint32_t>(K_packed < 256ULL ? K_packed : 256ULL), 1};  // Clamp to TMA 256 limit
            cuuint64_t strides_B[1] = {K_packed};  // Stride between rows

            CUresult resB = encode_tma_vector(map_B_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                       2, B_ptr, dims_B, strides_B, box_B);
            if (resB != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resB, &err_str);
                printf("TMA Encode B failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_B, sizeof(CUtensorMap)), "cudaMalloc d_map_B");
                check_cuda(cudaMemcpy(d_map_B, map_B_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_B");
                // Rank-2 B success (silent)
            }
        } else {
            // B: rank-3, actual layout is [L, 128, K_packed]
            // TMA dims order: [K_packed, 128, L] (innermost to outermost)
            cuuint64_t dims_B[3] = {K_packed, 128, static_cast<cuuint64_t>(L)};
            cuuint32_t box_B[3] = {static_cast<cuuint32_t>(K_packed < 256ULL ? K_packed : 256ULL), 1, 1};  // Clamp to TMA 256 limit
            cuuint64_t strides_B[2] = {K_packed, 128 * K_packed};  // Row stride, batch stride

            CUresult resB = encode_tma_vector(map_B_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                       3, B_ptr, dims_B, strides_B, box_B);
            if (resB != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resB, &err_str);
                printf("TMA Encode B failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_B, sizeof(CUtensorMap)), "cudaMalloc d_map_B");
                check_cuda(cudaMemcpy(d_map_B, map_B_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_B");
                printf("✅TMA Encode B (rank=3) SUCCESS!\n");
            }
        }

        // Create TMA descriptors for SFB (scale factors for B)
        std::vector<uint8_t> map_SFB_buf(sizeof(CUtensorMap) + 128);
        void* map_SFB_raw = map_SFB_buf.data();
        size_t map_SFB_space = map_SFB_buf.size();
        void* map_SFB_aligned = std::align(128, sizeof(CUtensorMap), map_SFB_raw, map_SFB_space);
        if (!map_SFB_aligned) throw std::runtime_error("Failed to align map_SFB");
        std::memset(map_SFB_aligned, 0, sizeof(CUtensorMap));
        CUtensorMap* map_SFB_ptr = reinterpret_cast<CUtensorMap*>(map_SFB_aligned);

        if (L == 1) {
            // SFB: rank-2, actual layout is [1, 128, K_scales] but L=1 collapses to [128, K_scales]
            // TMA dims order: [K_scales, 128] (innermost to outermost)
            cuuint64_t dims_SFB[2] = {K_scales, 128};
            cuuint32_t box_SFB[2] = {static_cast<cuuint32_t>(K_scales < 256ULL ? K_scales : 256ULL), 1};
            cuuint64_t strides_SFB[1] = {K_scales};

            CUresult resSFB = encode_tma_vector(map_SFB_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         2, SFB_ptr, dims_SFB, strides_SFB, box_SFB);
            if (resSFB != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resSFB, &err_str);
                printf("TMA Encode SFB failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_SFB, sizeof(CUtensorMap)), "cudaMalloc d_map_SFB");
                check_cuda(cudaMemcpy(d_map_SFB, map_SFB_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFB");
                // Rank-2 SFB success (silent)
            }
        } else {
            // SFB: rank-3, actual layout is [L, 128, K_scales]
            // TMA dims order: [K_scales, 128, L] (innermost to outermost)
            cuuint64_t dims_SFB[3] = {K_scales, 128, static_cast<cuuint64_t>(L)};
            cuuint32_t box_SFB[3] = {static_cast<cuuint32_t>(K_scales < 256ULL ? K_scales : 256ULL), 1, 1};
            cuuint64_t strides_SFB[2] = {K_scales, 128 * K_scales};

            CUresult resSFB = encode_tma_vector(map_SFB_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         3, SFB_ptr, dims_SFB, strides_SFB, box_SFB);
            if (resSFB != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resSFB, &err_str);
                printf("TMA Encode SFB failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_SFB, sizeof(CUtensorMap)), "cudaMalloc d_map_SFB");
                check_cuda(cudaMemcpy(d_map_SFB, map_SFB_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFB");
                printf("✅TMA Encode SFB (rank=3) SUCCESS!\n");
            }
        }

        // Create TMA descriptors for SFA (scale factors for A) - tiled access
        std::vector<uint8_t> map_SFA_buf(sizeof(CUtensorMap) + 128);
        void* map_SFA_raw = map_SFA_buf.data();
        size_t map_SFA_space = map_SFA_buf.size();
        void* map_SFA_aligned = std::align(128, sizeof(CUtensorMap), map_SFA_raw, map_SFA_space);
        if (!map_SFA_aligned) throw std::runtime_error("Failed to align map_SFA");
        std::memset(map_SFA_aligned, 0, sizeof(CUtensorMap));
        CUtensorMap* map_SFA_ptr = reinterpret_cast<CUtensorMap*>(map_SFA_aligned);

        // For rank-2 (L=1): Use SWIZZLE_NONE with box_k=16 (no padding needed, CTA launch)
        // For rank-3 (L=4,8): Use SWIZZLE_128B with box_k=128 (optimal for swizzling, cluster launch)
        // TMA hardware limit: box dimensions must not exceed 256
        cuuint32_t box_sfa_k = (L == 1) ? 16u : 128u;
        cuuint32_t box_sfa_m = static_cast<cuuint32_t>(
            kTileM < M ?
                (kTileM < 16ULL ? 16ULL : (kTileM < 256ULL ? kTileM : 256ULL)) :
                (M < 16ULL ? 16ULL : (M < 256ULL ? M : 256ULL))
        );

        if (L == 1) {
            // SFA: rank-2, use SWIZZLE_NONE (encode_tma_vector) for small dimensions
            // TMA load passes (K_coord, M_coord) so dims must be [K, M]
            // Global memory is row-major: element[row,k] at offset row*K_scales_padded + k
            cuuint64_t dims_SFA[2] = {static_cast<cuuint64_t>(K_scales_padded), static_cast<cuuint64_t>(M)};
            cuuint32_t box_SFA[2] = {box_sfa_k, box_sfa_m};
            cuuint64_t strides_SFA[1] = {static_cast<cuuint64_t>(K_scales_padded)};

            // Rank-2 SFA descriptor creation (silent)

            uintptr_t sfa_addr = reinterpret_cast<uintptr_t>(SFA_ptr);
            if ((sfa_addr % 128) != 0) {
                printf("ERROR: SFA_ptr not 128-byte aligned: %p (mod128=%llu)\n",
                       SFA_ptr, (unsigned long long)(sfa_addr % 128));
            }

            // Use SWIZZLE_NONE for rank-2 (CTA launch)
            CUresult resSFA = encode_tma_vector(map_SFA_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         2, SFA_ptr, dims_SFA, strides_SFA, box_SFA);
            if (resSFA != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resSFA, &err_str);
                printf("TMA Encode SFA failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_SFA, sizeof(CUtensorMap)), "cudaMalloc d_map_SFA");
                check_cuda(cudaMemcpy(d_map_SFA, map_SFA_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFA");
                // Rank-2 SFA success (silent)
            }
        } else {
            // SFA: rank-3, actual layout is [L, M, K_scales_padded]
            // TMA dims order: [K_scales_padded, M, L] (innermost to outermost)
            cuuint64_t dims_SFA[3] = {static_cast<cuuint64_t>(K_scales_padded), static_cast<cuuint64_t>(M), static_cast<cuuint64_t>(L)};
            cuuint32_t box_SFA[3] = {box_sfa_k, box_sfa_m, 1};
            cuuint64_t strides_SFA[2] = {static_cast<cuuint64_t>(K_scales_padded), static_cast<cuuint64_t>(M * K_scales_padded)};

            CUresult resSFA = encode_tma_matrix(map_SFA_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         3, SFA_ptr, dims_SFA, strides_SFA, box_SFA);
            if (resSFA != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resSFA, &err_str);
                printf("TMA Encode SFA failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_SFA, sizeof(CUtensorMap)), "cudaMalloc d_map_SFA");
                check_cuda(cudaMemcpy(d_map_SFA, map_SFA_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFA");
                printf("✅TMA Encode SFA (rank=3) SUCCESS!\n");
            }
        }
    }

    // Verify all TMA descriptors were created successfully
    if (!d_map_A || !d_map_SFA || !d_map_B || !d_map_SFB) {
        printf("ERROR: Not all TMA descriptors were created successfully!\n");
        printf("  d_map_A=%p d_map_SFA=%p d_map_B=%p d_map_SFB=%p\n",
               (void*)d_map_A, (void*)d_map_SFA, (void*)d_map_B, (void*)d_map_SFB);
        throw std::runtime_error("TMA descriptor creation failed");
    }

    cudaFuncAttributes attr;
    cudaError_t attr_err = cudaFuncGetAttributes(&attr, fp4_gemv_streaming<kTileM, kTileK, kThreads>);
    if (attr_err != cudaSuccess) throw std::runtime_error(std::string("cudaFuncGetAttributes failed"));

    cudaError_t set_err = cudaFuncSetAttribute(
        fp4_gemv_streaming<kTileM, kTileK, kThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)
    );
    if (set_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaFuncSetAttribute MaxDynamicSharedMemorySize failed: ") + cudaGetErrorString(set_err));
    }
    if (L > 1) {
        printf("✅cudaFuncSetAttribute MaxDynamicSharedMemorySize=%zu SUCCESS\n", shared_bytes);
    }

    int num_blocks = static_cast<int>((M + kTileM - 1) / kTileM);
    void const* kernel_ptr = (void const*)fp4_gemv_streaming<kTileM, kTileK, kThreads>;

    int grid_x, grid_y;
    dim3 grid, block, cluster;
    cudaLaunchConfig_t launch_config = {0};
    cudaLaunchAttribute launch_attr[1] = {};  // Declare outside if/else to persist

    if (L == 1) {
        // Rank-2: Regular CTA launch (no cluster)
        grid_x = num_blocks;
        grid_y = 1;
        grid = dim3(grid_x, grid_y);
        block = dim3(kThreads);

        launch_config.gridDim = grid;
        launch_config.blockDim = block;
        launch_config.dynamicSmemBytes = shared_bytes;
        launch_config.stream = 0;
        launch_config.attrs = nullptr;
        launch_config.numAttrs = 0;
    } else {
        // Rank-3: Cluster launch with cta_group::2
        // Round up grid_x to be divisible by cluster.x=2
        grid_x = ((num_blocks + 1) / 2) * 2;
        grid_y = static_cast<int>(L);
        grid = dim3(grid_x, grid_y);
        block = dim3(kThreads);
        cluster = dim3(2, 1, 1);

        // Enable non-portable cluster size
        cudaError_t cluster_enable = cudaFuncSetAttribute(
            kernel_ptr,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1
        );
        if (cluster_enable != cudaSuccess) {
            printf("ERROR: cudaFuncSetAttribute NonPortableCluster failed: %s\n", cudaGetErrorString(cluster_enable));
            throw std::runtime_error(std::string("cudaFuncSetAttribute NonPortableCluster failed: ") + cudaGetErrorString(cluster_enable));
        }
        printf("✅cudaFuncSetAttribute NonPortableClusterSizeAllowed=1 SUCCESS\n");

        // Set up cluster launch configuration
        launch_attr[0].id = cudaLaunchAttributeClusterDimension;
        launch_attr[0].val.clusterDim.x = cluster.x;
        launch_attr[0].val.clusterDim.y = cluster.y;
        launch_attr[0].val.clusterDim.z = cluster.z;

        printf("DEBUG: Immediately after setting launch_attr:\n");
        printf("  cudaLaunchAttributeClusterDimension enum value=%d\n", (int)cudaLaunchAttributeClusterDimension);
        printf("  launch_attr[0].id=%d\n", (int)launch_attr[0].id);
        printf("  launch_attr[0].val.clusterDim=(%u,%u,%u)\n",
               launch_attr[0].val.clusterDim.x,
               launch_attr[0].val.clusterDim.y,
               launch_attr[0].val.clusterDim.z);

        launch_config.gridDim = grid;
        launch_config.blockDim = block;
        launch_config.dynamicSmemBytes = shared_bytes;
        launch_config.stream = 0;
        launch_config.attrs = launch_attr;
        launch_config.numAttrs = 1;

        printf("DEBUG: After setting launch_config.attrs:\n");
        printf("  launch_config.attrs=%p, launch_attr=%p\n", (void*)launch_config.attrs, (void*)launch_attr);
        printf("  launch_config.attrs[0].id=%d\n", (int)launch_config.attrs[0].id);
        printf("  launch_config.attrs[0].val.clusterDim=(%u,%u,%u)\n",
               launch_config.attrs[0].val.clusterDim.x,
               launch_config.attrs[0].val.clusterDim.y,
               launch_config.attrs[0].val.clusterDim.z);

        // Debug output for rank-3 only
        printf("DEBUG launch (rank-3 cluster):\n");
        printf("  grid=(%d,%d,%d)\n", grid_x, grid_y, 1);
        printf("  block=(%d,%d,%d)\n", kThreads, 1, 1);
        printf("  shared_bytes=%zu (%.1f KB)\n", shared_bytes, shared_bytes/1024.0);
        printf("  cluster=(%d,%d,%d)\n", cluster.x, cluster.y, cluster.z);
        printf("  M=%lld K=%lld L=%lld K_scales_padded=%lld\n",
               (long long)M, (long long)K, (long long)L, (long long)K_scales_padded);
        printf("  box_sfa_k=%d\n", (L == 1) ? 16 : 128);

        // Check if grid is divisible by cluster
        if (grid_x % cluster.x != 0) {
            printf("WARNING: grid.x=%d not divisible by cluster.x=%d\n", grid_x, cluster.x);
        }
        if (grid_y % cluster.y != 0) {
            printf("WARNING: grid.y=%d not divisible by cluster.y=%d\n", grid_y, cluster.y);
        }
    }

    int M_int = static_cast<int>(M);
    int K_int = static_cast<int>(K);
    int L_int = static_cast<int>(L);
    int K_scales_padded_int = static_cast<int>(K_scales_padded);

    void* kernel_args[] = {
        const_cast<uint8_t**>(&A_ptr),
        const_cast<uint8_t**>(&B_ptr),
        const_cast<uint8_t**>(&SFA_ptr),
        const_cast<uint8_t**>(&SFB_ptr),
        &d_map_A,
        &d_map_SFA,
        &d_map_B,
        &d_map_SFB,
        &D_ptr,
        &M_int,
        &K_int,
        &L_int,
        &K_scales_padded_int
    };

    // For rank-3, query device properties before launch
    if (L > 1) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("Device limits:\n");
        printf("  sharedMemPerBlock: %zu (%.1f KB)\n",
               prop.sharedMemPerBlock, prop.sharedMemPerBlock/1024.0);
        printf("  sharedMemPerBlockOptin: %zu (%.1f KB)\n",
               prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin/1024.0);
        printf("  maxBlocksPerMultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  clusterDimSupported: %d\n", prop.clusterLaunch);
        printf("  Requested shared_bytes: %zu (%.1f KB)\n",
               shared_bytes, shared_bytes/1024.0);

        // Check if shared memory exceeds limits
        if (shared_bytes > prop.sharedMemPerBlockOptin) {
            printf("ERROR: Requested shared memory exceeds device limit!\n");
        }

        // Check cluster support
        if (!prop.clusterLaunch) {
            printf("ERROR: Device does not support cluster launch!\n");
        }

        // Verify launch config
        printf("Launch config verification:\n");
        printf("  kernel_ptr=%p\n", kernel_ptr);
        printf("  launch_config.gridDim=(%u,%u,%u)\n",
               launch_config.gridDim.x, launch_config.gridDim.y, launch_config.gridDim.z);
        printf("  launch_config.blockDim=(%u,%u,%u)\n",
               launch_config.blockDim.x, launch_config.blockDim.y, launch_config.blockDim.z);
        printf("  launch_config.dynamicSmemBytes=%zu\n", launch_config.dynamicSmemBytes);
        printf("  launch_config.stream=%p\n", launch_config.stream);
        printf("  launch_config.numAttrs=%d\n", launch_config.numAttrs);
        if (launch_config.numAttrs > 0) {
            printf("  launch_attr[0].id=%d (should be %d for ClusterDimension)\n",
                   launch_config.attrs[0].id, cudaLaunchAttributeClusterDimension);
            printf("  cluster dims=(%u,%u,%u)\n",
                   launch_config.attrs[0].val.clusterDim.x,
                   launch_config.attrs[0].val.clusterDim.y,
                   launch_config.attrs[0].val.clusterDim.z);
        }
    }

    cudaError_t launch_err = cudaLaunchKernelExC(&launch_config, kernel_ptr, kernel_args);
    if (launch_err != cudaSuccess) {
        if (L > 1) {
            printf("Launch failed with error: %s\n", cudaGetErrorString(launch_err));
            printf("This was a rank-3 (L=%lld) cluster launch\n", (long long)L);
        }
        throw std::runtime_error(std::string("cudaLaunchKernelExC failed: ") + cudaGetErrorString(launch_err));
    }

    if (L > 1) {
        printf("✓ Rank-3 kernel launched successfully!\n");
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (d_map_A) cudaFree(d_map_A);
    if (d_map_SFA) cudaFree(d_map_SFA);
    if (d_map_B) cudaFree(d_map_B);
    if (d_map_SFB) cudaFree(d_map_SFB);
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
    int64_t M, int64_t K, int64_t L, int64_t K_scales_padded
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
            functions=[
                "launch_fp4_gemv_optimized",
            ],
            verbose=True,
            extra_cuda_cflags=[
                "-O2",  # Enable optimizations to fix lambda stack issues
                "-DNDEBUG",  # Disable debug output for performance testing
                # "--use_fast_math",  # Disabled for debugging
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
            ],
            extra_ldflags=["-lcuda"],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    """
    SM100 FP4 GEMV with tensor cores: CUTLASS + CuTe + PTX only (NO WMMA)


    """
    # Workaround for multiprocessing setting sys.stdout to None
    import sys
    if sys.stdout is None:
        sys.stdout = open('/dev/null', 'w')
    if sys.stderr is None:
        sys.stderr = open('/dev/null', 'w')

    a, b, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c = data

    M, _, L = c.shape
    K = a.shape[1] * 2
    K_scales = K // 16

    # Permute to [L, M, K/2] layout
    # CRITICAL: Clone first to avoid tensor aliasing/reuse between test calls
    a = a.clone().permute(2, 0, 1).contiguous().cuda()
    b = b.clone().permute(2, 0, 1).contiguous().cuda()
    c = c.clone().permute(2, 0, 1).contiguous().cuda()

    # Shape assertions to catch corruption early
    assert a.shape == (L, M, K // 2), (
        f"Shape mismatch: a.shape={a.shape}, expected=({L}, {M}, {K // 2})"
    )
    assert b.shape[0] == L and b.shape[2] == K // 2, (
        f"Shape mismatch: b.shape={b.shape}"
    )

    # Scale factors: use simple LINEAR layout [L, rows, K_scales]
    # sfa_ref_cpu: [M, K_scales, L] -> permute to [L, M, K_scales]
    # sfb_ref_cpu: [128, K_scales, L] -> permute to [L, 128, K_scales]
    K_scales = K // 16
    print(
        f"\n[SCALE DEBUG] sfa_ref_cpu shape={sfa_ref_cpu.shape}, device={sfa_ref_cpu.device}"
    )
    print(
        f"[SCALE DEBUG] sfb_ref_cpu shape={sfb_ref_cpu.shape}, device={sfb_ref_cpu.device}"
    )

    # Permute to [L, M, K_scales] and [L, 128, K_scales] for simple linear indexing
    sfa_linear = sfa_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda()
    sfb_linear = sfb_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda()

    # Pad SFA tensor based on rank:
    # rank-2 (L=1): Minimal padding for SWIZZLE_NONE (just round to 16)
    # rank-3 (L=4,8): Pad to at least 128 for SWIZZLE_128B
    if L == 1:
        # rank-2: Minimal padding for SWIZZLE_NONE
        K_scales_padded = ((K_scales + 15) // 16) * 16
    else:
        # rank-3: Pad to at least 128 for SWIZZLE_128B
        K_scales_padded = max(128, ((K_scales + 15) // 16) * 16)

    if K_scales_padded > K_scales:
        pad_amount = K_scales_padded - K_scales
        # Pad with zeros in the K_scales dimension (dimension 2 of [L, M, K_scales])
        sfa_linear = torch.nn.functional.pad(sfa_linear, (0, pad_amount), value=0)
        print(f"[DEBUG] Padded SFA from K_scales={K_scales} to K_scales_padded={K_scales_padded}")
    else:
        K_scales_padded = K_scales
        print(f"[DEBUG] No padding needed, K_scales={K_scales}")

    # Reinterpret as raw bytes for the CUDA kernel (uint8 view of float8 storage)
    sfa_bytes = sfa_linear.view(torch.uint8)
    sfb_bytes = sfb_linear.view(torch.uint8)

    # Reinterpret as raw bytes
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)

    def dump_tensor_info(name: str, t: torch.Tensor):
        elem_size = t.element_size()
        numel = t.numel()
        ptr = t.data_ptr()
        print(
            f"[DEBUG] {name}: shape={tuple(t.shape)}, stride={tuple(t.stride())}, "
            f"elem_size={elem_size}, numel={numel}, bytes={numel * elem_size}, "
            f"data_ptr={hex(ptr)}"
        )

    dump_tensor_info("a_bytes", a_bytes)
    dump_tensor_info("b_bytes", b_bytes)
    dump_tensor_info("sfa_bytes", sfa_bytes)
    dump_tensor_info("sfb_bytes", sfb_bytes)

    # Debug: Print first bytes of scale factors for multiple rows
    for r in range(min(5, sfa_bytes.shape[1])):
        print(f"[DEBUG] sfa_bytes[0,{r},:8] (batch 0, row {r}, first 8 scales): {sfa_bytes[0,r,:min(8,sfa_bytes.shape[2])].cpu().tolist()}")
    print(f"[DEBUG] sfb_bytes[0,0,:8] (batch 0, row 0, first 8 scales): {sfb_bytes[0,0,:min(8,sfb_bytes.shape[2])].cpu().tolist()}")

    # Compute base/end ranges and check for overlap between all byte tensors
    tensors = {
        "a_bytes": a_bytes,
        "b_bytes": b_bytes,
        "sfa_bytes": sfa_bytes,
        "sfb_bytes": sfb_bytes,
    }
    ranges = {}
    for name, t in tensors.items():
        base = t.data_ptr()
        size_bytes = t.numel() * t.element_size()
        ranges[name] = (base, base + size_bytes)
        print(
            f"[DEBUG] {name} range: [{hex(base)}, {hex(base + size_bytes)}) "
            f"({size_bytes} bytes)"
        )

    names = list(ranges.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            b1, e1 = ranges[n1]
            b2, e2 = ranges[n2]
            if max(b1, b2) < min(e1, e2):
                print(
                    f"[WARNING] Host buffer overlap detected between {n1} and {n2}: "
                    f"[{hex(b1)}, {hex(e1)}) vs [{hex(b2)}, {hex(e2)})"
                )

    # Verify tensor alignment / stride expectations for TMA-esque access
    a_ptr = a_bytes.data_ptr()
    b_ptr = b_bytes.data_ptr()
    K_packed_val = K // 2
    print(
        f"[DEBUG] a shape={tuple(a.shape)}, stride={tuple(a.stride())}, "
        f"data_ptr={hex(a.data_ptr())}"
    )
    print(
        f"[DEBUG] b shape={tuple(b.shape)}, stride={tuple(b.stride())}, "
        f"data_ptr={hex(b.data_ptr())}"
    )
    if a_ptr % 128 != 0:
        print(
            f"WARNING (Python): a_bytes data_ptr {hex(a_ptr)} is not 128-byte aligned "
            f"(mod128={a_ptr % 128})"
        )
    else:
        print(f"✅(Python) a_bytes 128-byte alignment check passed: {hex(a_ptr)}")
    if b_ptr % 128 != 0:
        print(
            f"WARNING (Python): b_bytes data_ptr {hex(b_ptr)} is not 128-byte aligned "
            f"(mod128={b_ptr % 128})"
        )
    else:
        print(f"✅(Python) b_bytes 128-byte alignment check passed: {hex(b_ptr)}")
    if K_packed_val % 128 != 0:
        print(f"WARNING (Python): K_packed ({K_packed_val}) is not 128-byte aligned")

    # Launch SM100 tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L, K_scales_padded)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()  # [M, 1, L]

    return c
