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

__device__ __forceinline__ void tma_load_2d(void* smem_ptr,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t coord1,
                                            uint32_t bytes,
                                            uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    mbarrier_arrive_expect_tx(mbar, bytes);

    uint32_t c0 = coord0;
    uint32_t c1 = coord1;

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(c0),
          "r"(c1),
          "r"(mbar_addr)
    );
}

__device__ __forceinline__ void tma_load_3d(void* smem_ptr,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t coord1,
                                            uint32_t coord2,
                                            uint32_t bytes,
                                            uint64_t* mbar) {
    uint32_t smem_addr = cvta_to_shared_u32(smem_ptr);
    uint32_t mbar_addr = cvta_to_shared_u32(mbar);
    mbarrier_arrive_expect_tx(mbar, bytes);

    uint32_t c0 = coord0;
    uint32_t c1 = coord1;
    uint32_t c2 = coord2;

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];\n"
        :
        : "r"(smem_addr),
          "l"(desc),
          "r"(c0),
          "r"(c1),
          "r"(c2),
          "r"(mbar_addr)
    );
}

__device__ __forceinline__ uint64_t* mbar_stage(uint64_t* base, int stage) {
    // Each mbarrier occupies 16 bytes (two uint64_t slots)
    return base + stage * 2;
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
    const CUtensorMap* __restrict__ desc_B,
    const CUtensorMap* __restrict__ desc_SFB,
    half* __restrict__ D,
    const int M, const int K, const int L
) {
#if __CUDA_ARCH__ >= 700
    constexpr int TileKPacked = TileK / 2;
    constexpr int TileScaleCount = TileK / 16;
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
    const int K_scales = K >> 4;
    const int tile_rows = (M - m_tile) < TileM ? (M - m_tile) : TileM;

    const uint8_t* A_batch = A_packed + static_cast<size_t>(batch) * M * K_packed;
    const uint8_t* B_batch = B_packed + static_cast<size_t>(batch) * 128 * K_packed;
    const uint8_t* SFA_batch = SFA_packed + static_cast<size_t>(batch) * M * K_scales;
    const uint8_t* SFB_batch = SFB_packed + static_cast<size_t>(batch) * 128 * K_scales;
    half* D_batch = D + static_cast<size_t>(batch) * M;

    extern __shared__ uint8_t smem[];

#ifndef NDEBUG
    if (tid == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        DEBUG_PRINT_ERROR("KERNEL STARTING: grid=(%d,%d,%d) blockDim.x=%d M=%d K=%d L=%d\n",
                          gridDim.x, gridDim.y, gridDim.z, blockDim.x, M, K, L);
    }
    if (tid == 0) {
        DEBUG_PRINT_ERROR("DBG block=(%d,%d,%d) tid=%d warp_id=%d lane_id=%d batch=%d m_tile=%d tile_rows=%d M=%d K=%d L=%d K_packed=%d K_scales=%d\n",
                          blockIdx.x, blockIdx.y, blockIdx.z,
                          tid, warp_id, lane_id,
                          batch, m_tile, tile_rows,
                          M, K, L, K_packed, K_scales);
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

    uint64_t* mbar_a = alloc_mbar_array(StageCount);
    uint64_t* mbar_b = alloc_mbar_array(1);
    uint64_t* mbar_sfb = alloc_mbar_array(1);

    offset = align_up(offset, 16);

    uint8_t* b_packed_smem = smem + offset;
    offset += K_packed;
    offset = align_up(offset, 16);

    uint8_t* sfb_smem = smem + offset;
    offset += K_scales;
    offset = align_up(offset, 16);

    half* b_vec_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(K) * sizeof(half);
    offset = align_up(offset, 16);

    // Ensure TMA destinations are 128-byte aligned in shared memory
    align_up_smem_128();
    uint8_t* a_packed_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        a_packed_stage[s] = smem + offset;
        offset += TileM * TileKPacked;
    }

    // Scales used alongside TMA data ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“ keep them 128-byte aligned as well
    align_up_smem_128();
    uint8_t* sfa_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfa_stage[s] = smem + offset;
        offset += TileM * TileScaleCount;
    }

    // Tensor Core operand tiles in shared memory: also 128-byte aligned
    align_up_smem_128();
    half* a_f16_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileM) * a_stride * sizeof(half);
    align_up_smem_128();
    half* b_tile_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileK) * 8 * sizeof(half);
    align_up_smem_128();
    half* a_scale_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileM) * TileScaleCount * sizeof(half);

    (void)offset;

#if __CUDA_ARCH__ >= 900
    const bool use_tma_a = (desc_A != nullptr);
#else
    const bool use_tma_a = false;
#endif

    __shared__ uint32_t stage_phase_smem[StageCount];

    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            stage_phase_smem[s] = 0;
        }
    }

#ifndef NDEBUG
    // Verify 128-byte alignment of key shared-memory regions used by TMA / Tensor Cores
    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            uint32_t addr = cvta_to_shared_u32(a_packed_stage[s]);
            if (addr % 128 != 0) {
                printf("SMEM ALIGN ERROR: a_packed_stage[%d] addr=%u not 128-byte aligned (mod128=%u)\n",
                       s, addr, addr % 128);
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
        for (int s = 0; s < StageCount; ++s) {
            mbarrier_init(mbar_stage(mbar_a, s));
        }
    }
#endif
    __syncthreads();

    // Robust fallback: Always use software path for B and SFB
    {
#ifndef NDEBUG
        __syncthreads();
        DEBUG_PRINT_ERROR("IN_LOOP B/SFB cp_async START block=(%d,%d,%d) tid=%d\n",
                          blockIdx.x, blockIdx.y, blockIdx.z, tid);
        __syncthreads();
#endif

        // Total sizes (in bytes) of B and SFB packed tensors across all batches.
        const size_t B_total_bytes = static_cast<size_t>(L) * 128 * K_packed;
        const size_t SFB_total_bytes = static_cast<size_t>(L) * 128 * K_scales;

        int b_segments = (K_packed + 15) / 16;
        for (int idx = tid; idx < b_segments; idx += Threads) {
            int byte_idx = idx * 16;
            bool full = (byte_idx + 16) <= K_packed;
            uint8_t* dst = b_packed_smem + byte_idx;
            const uint8_t* src = B_batch + byte_idx;

#ifndef NDEBUG
            {
                // Global index into full B_packed tensor (bytes)
                size_t B_index = static_cast<size_t>(batch) * 128 * K_packed
                                 + static_cast<size_t>(byte_idx);
                DEBUG_OOB_GLOBAL_1D("B_packed_cp_async", B_index, B_total_bytes, B_packed);
                // Shared-memory index for b_packed_smem
                DEBUG_OOB_SMEM_1D("b_packed_smem_cp_async", byte_idx, K_packed, b_packed_smem);

                size_t start = static_cast<size_t>(byte_idx);
                size_t end = start + 16;
                if (start >= static_cast<size_t>(K_packed) || B_index >= B_total_bytes) {
                    DEBUG_PRINT_ERROR("OOB: B_batch cp_async start OOB: byte_idx=%d K_packed=%d B_index=%zu B_total=%zu src=%p dst=%p at %s:%d\n",
                                      byte_idx, K_packed, B_index, B_total_bytes,
                                      (const void*)src, (void*)dst, __FILE__, __LINE__);
                    return;
                }
                if (end > static_cast<size_t>(K_packed) ||
                    (B_index + 16) > B_total_bytes) {
                    DEBUG_PRINT_ERROR("INFO: B_batch cp_async partial tail: byte_idx=%d K_packed=%d B_index=%zu B_total=%zu src=%p dst=%p at %s:%d\n",
                                      byte_idx, K_packed, B_index, B_total_bytes,
                                      (const void*)src, (void*)dst, __FILE__, __LINE__);
                }

                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                    tid == 0 && idx < 4) {
                    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
                    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
                    DEBUG_PRINT_ERROR("B cp_async: batch=%d idx=%d byte_idx=%d B_total=%zu B_packed=%p B_batch=%p src=%p dst=%p src_mod16=%llu dst_mod16=%llu at %s:%d\n",
                                      batch, idx, byte_idx,
                                      B_total_bytes,
                                      (const void*)B_packed, (const void*)B_batch,
                                      (const void*)src, (void*)dst,
                                      (unsigned long long)(src_addr % 16),
                                      (unsigned long long)(dst_addr % 16),
                                      __FILE__, __LINE__);
                }
            }
#endif

            cp_async_16b(dst, src, full && (byte_idx < K_packed));
            if (!full && byte_idx < K_packed) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int g = byte_idx + i;
                    if (g < K_packed) {
#ifndef NDEBUG
                        // Global index for this byte
                        size_t B_idx_g = static_cast<size_t>(batch) * 128 * K_packed
                                         + static_cast<size_t>(g);
                        DEBUG_OOB_GLOBAL_1D("B_packed_tail", B_idx_g, B_total_bytes, B_packed);
                        DEBUG_OOB_SMEM_1D("b_packed_smem_tail", g, K_packed, b_packed_smem);
                        if (B_idx_g >= B_total_bytes) {
                            DEBUG_PRINT_ERROR("OOB: B_tail copy global index OOB: g=%d B_idx=%zu B_total=%zu at %s:%d\n",
                                              g, B_idx_g, B_total_bytes, __FILE__, __LINE__);
                            return;
                        }
#endif
                        dst[i] = src[i];
                    } else {
#ifndef NDEBUG
                        DEBUG_PRINT_ERROR("WARN: B_batch tail copy would be OOB: g=%d K_packed=%d base=%p at %s:%d\n",
                                          g, K_packed, (const void*)B_batch, __FILE__, __LINE__);
#endif
                    }
                }
            }
        }
        int s_segments = (K_scales + 15) / 16;
        for (int idx = tid; idx < s_segments; idx += Threads) {
            int byte_idx = idx * 16;
            bool full = (byte_idx + 16) <= K_scales;
            uint8_t* dst = sfb_smem + byte_idx;
            const uint8_t* src = SFB_batch + byte_idx;

#ifndef NDEBUG
            {
                size_t SFB_index = static_cast<size_t>(batch) * 128 * K_scales
                                   + static_cast<size_t>(byte_idx);
                DEBUG_OOB_GLOBAL_1D("SFB_packed_cp_async", SFB_index, SFB_total_bytes, SFB_packed);
                DEBUG_OOB_SMEM_1D("sfb_smem_cp_async", byte_idx, K_scales, sfb_smem);
                size_t start = static_cast<size_t>(byte_idx);
                size_t end = start + 16;
                if (start >= static_cast<size_t>(K_scales) || SFB_index >= SFB_total_bytes) {
                    DEBUG_PRINT_ERROR("OOB: SFB_batch cp_async start OOB: byte_idx=%d K_scales=%d SFB_index=%zu SFB_total=%zu src=%p dst=%p at %s:%d\n",
                                      byte_idx, K_scales, SFB_index, SFB_total_bytes,
                                      (const void*)src, (void*)dst, __FILE__, __LINE__);
                    return;
                }
                if (end > static_cast<size_t>(K_scales) ||
                    (SFB_index + 16) > SFB_total_bytes) {
                    DEBUG_PRINT_ERROR("INFO: SFB_batch cp_async partial tail: byte_idx=%d K_scales=%d SFB_index=%zu SFB_total=%zu src=%p dst=%p at %s:%d\n",
                                      byte_idx, K_scales, SFB_index, SFB_total_bytes,
                                      (const void*)src, (void*)dst, __FILE__, __LINE__);
                }

                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                    tid == 0 && idx < 4) {
                    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
                    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
                    DEBUG_PRINT_ERROR("SFB cp_async: batch=%d idx=%d byte_idx=%d SFB_total=%zu SFB_packed=%p SFB_batch=%p src=%p dst=%p src_mod16=%llu dst_mod16=%llu at %s:%d\n",
                                      batch, idx, byte_idx,
                                      SFB_total_bytes,
                                      (const void*)SFB_packed, (const void*)SFB_batch,
                                      (const void*)src, (void*)dst,
                                      (unsigned long long)(src_addr % 16),
                                      (unsigned long long)(dst_addr % 16),
                                      __FILE__, __LINE__);
                }
            }
#endif

            cp_async_16b(dst, src, full && (byte_idx < K_scales));
            if (!full && byte_idx < K_scales) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int g = byte_idx + i;
                    if (g < K_scales) {
                        size_t SFB_idx_g = static_cast<size_t>(batch) * 128 * K_scales
                                           + static_cast<size_t>(g);
                        DEBUG_OOB_GLOBAL_1D("SFB_packed_tail", SFB_idx_g, SFB_total_bytes, SFB_packed);
                        DEBUG_OOB_SMEM_1D("sfb_smem_tail", g, K_scales, sfb_smem);
                        if (SFB_idx_g >= SFB_total_bytes) {
                            DEBUG_PRINT_ERROR("OOB: SFB_tail copy global index OOB: g=%d SFB_idx=%zu SFB_total=%zu at %s:%d\n",
                                              g, SFB_idx_g, SFB_total_bytes, __FILE__, __LINE__);
                            return;
                        }
                        dst[i] = src[i];
                    } else {
#ifndef NDEBUG
                        DEBUG_PRINT_ERROR("WARN: SFB_batch tail copy would be OOB: g=%d K_scales=%d base=%p at %s:%d\n",
                                          g, K_scales, (const void*)SFB_batch, __FILE__, __LINE__);
#endif
                    }
                }
            }
        }
        cp_async_commit();
        cp_async_wait();
#ifndef NDEBUG
        __syncthreads();
        DEBUG_PRINT_ERROR("IN_LOOP B/SFB cp_async END block=(%d,%d,%d) tid=%d\n",
                          blockIdx.x, blockIdx.y, blockIdx.z, tid);
        __syncthreads();
#endif
    }
    __syncthreads();

    // Decode B vector
    for (int idx = tid; idx < K_packed; idx += Threads) {
        int k_base = idx * 2;
        int scale_idx = idx >> 3;
        DEBUG_OOB_SMEM_1D("b_packed_smem", idx, K_packed, b_packed_smem);
        uint8_t packed = b_packed_smem[idx];
        half scale_h = __float2half(0.0f);
        if (scale_idx < K_scales) {
            DEBUG_OOB_SMEM_1D("sfb_smem", scale_idx, K_scales, sfb_smem);
            scale_h = __float2half(decode_fp8_e4m3(sfb_smem[scale_idx]));
        }
        // NVFP4 convention: low nibble = element 0, high nibble = element 1
        half v0 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);         // LOW nibble → first element
        half v1 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);  // HIGH nibble → second element

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

    auto prefetch_tile = [&](int stage, int k_tile_base) {
        if (use_tma_a) {
            if (warp_id == 0 && lane_id == 0) {

                // Element-space coordinates: row and packed-K index
                uint32_t c_m = static_cast<uint32_t>(m_tile);            // row in [0, M)
                uint32_t c_k_packed = static_cast<uint32_t>(k_tile_base >> 1); // packed K index in [0, K/2)

                if (L == 1) {
                    // rank = 2: TMA coords are (k_packed, m) to match tensor layout
                    // cuTensorMapEncodeTiled dims are [M, K_packed] with stride K_packed
                    // So coord0 indexes into the fastest-varying dimension (K_packed)
                    // and coord1 indexes into the slower dimension (M)
                    uint32_t c0 = c_k_packed;  // K dimension (fastest varying)
                    uint32_t c1 = c_m;         // M dimension (slower varying)

                    // Bounds check for TMA tile in element space
                    bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);
                    bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);
#ifndef NDEBUG
                    if (!valid_m || !valid_k) {
                        DEBUG_PRINT_ERROR("WARN: TMA 2D coords may be OOB: c0(k)=%u c1(m)=%u TileKPacked=%d TileM=%d K_packed=%d M=%d at %s:%d\n",
                                          c0, c1, TileKPacked, TileM, K_packed, M, __FILE__, __LINE__);
                    }
                    if (blockIdx.x == 0) {
                        DEBUG_PRINT_ERROR("TMA load 2D: dst=%p c0(k)=%u c1(m)=%u bytes=%u batch=%d m_tile=%d k_tile_base=%d at %s:%d\n",
                                          (void*)a_packed_stage[stage],
                                          c0, c1,
                                          static_cast<uint32_t>((TileM * TileKPacked + 15) & ~15),
                                          batch, m_tile, k_tile_base, __FILE__, __LINE__);
                    }
#endif

                    if (valid_m && valid_k) {
                        tma_load_2d(
                            a_packed_stage[stage],
                            desc_A,
                            c0,
                            c1,
                            static_cast<uint32_t>((TileM * TileKPacked + 15) & ~15),
                            mbar_stage(mbar_a, stage)
                        );
                    } else {
                        // If bounds check fails, arrive at barrier without TX to avoid hang
                        mbarrier_arrive_expect_tx(mbar_stage(mbar_a, stage), 0);
                    }
                } else {
                    // For rank=3: coordinates are (k_packed, m, batch) to match tensor layout
                    // Tensor is [L, M, K_packed] with strides [M*K_packed, K_packed, 1]
                    uint32_t c0 = c_k_packed;  // K dimension (fastest varying)
                    uint32_t c1 = c_m;         // M dimension
                    uint32_t c2 = static_cast<uint32_t>(batch);  // Batch dimension (slowest)

                    bool valid_batch = c2 < static_cast<uint32_t>(L);
                    bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);
                    bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);
#ifndef NDEBUG
                    if (!valid_batch || !valid_m || !valid_k) {
                        DEBUG_PRINT_ERROR("WARN: TMA 3D coords may be OOB: c0(k)=%u c1(m)=%u c2(batch)=%u TileKPacked=%d TileM=%d L=%d M=%d K_packed=%d at %s:%d\n",
                                          c0, c1, c2, TileKPacked, TileM, L, M, K_packed, __FILE__, __LINE__);
                    }
                    if (blockIdx.x == 0) {
                        DEBUG_PRINT_ERROR("TMA load 3D: dst=%p c0(k)=%u c1(m)=%u c2(batch)=%u bytes=%u batch=%d m_tile=%d k_tile_base=%d at %s:%d\n",
                                          (void*)a_packed_stage[stage],
                                          c0, c1, c2,
                                          static_cast<uint32_t>(TileM * TileKPacked),
                                          batch, m_tile, k_tile_base, __FILE__, __LINE__);
                    }
#endif

                    if (valid_batch && valid_m && valid_k) {
                        tma_load_3d(
                            a_packed_stage[stage],
                            desc_A,
                            c0,
                            c1,
                            c2,
                            static_cast<uint32_t>(TileM * TileKPacked),
                            mbar_stage(mbar_a, stage)
                        );
                    } else {
                        // If bounds check fails, arrive at barrier without TX to avoid hang
                        mbarrier_arrive_expect_tx(mbar_stage(mbar_a, stage), 0);
                    }
                }
            }
        } else if (is_producer) {
            size_t bytes = static_cast<size_t>(TileM) * TileKPacked;
            int segments = static_cast<int>((bytes + 15) / 16);
            const uint8_t* src_base = A_batch + static_cast<size_t>(m_tile) * K_packed + (k_tile_base >> 1);
            uint8_t* dst_base = a_packed_stage[stage];
            for (int idx = tid; idx < segments; idx += ProducerThreads) {
                size_t linear = static_cast<size_t>(idx) * 16;
                int row = linear / TileKPacked;
                int col = linear - row * TileKPacked;
                const uint8_t* src = src_base + row * K_packed + col;
                uint8_t* dst = dst_base + row * TileKPacked + col;
                bool valid_row = row < tile_rows;
                bool full = (col + 16) <= TileKPacked;
                bool valid_col = (k_tile_base + col * 2) < K;
#ifndef NDEBUG
                if (!valid_row || !valid_col) {
                    DEBUG_PRINT_ERROR("WARN: A_batch cp_async candidate OOB: row=%d col=%d tile_rows=%d K=%d src=%p dst=%p at %s:%d\n",
                                      row, col, tile_rows, K, (const void*)src, (void*)dst, __FILE__, __LINE__);
                }
#endif
                cp_async_16b(dst, src, valid_row && full && valid_col);
                if (valid_row && !full && col < TileKPacked && valid_col) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        int g = col + i;
                        if (g < TileKPacked && (k_tile_base + g * 2) < K) {
                            dst[i] = src[i];
                        } else {
#ifndef NDEBUG
                            DEBUG_PRINT_ERROR("WARN: A_batch tail copy would be OOB: g=%d TileKPacked=%d K=%d base=%p at %s:%d\n",
                                              g, TileKPacked, K, (const void*)A_batch, __FILE__, __LINE__);
#endif
                        }
                    }
                }
            }
            cp_async_commit();
        }

        if (is_producer && warp_id == 1) {
            int scale_offset = k_tile_base >> 4;
            int total = TileM * TileScaleCount;
            for (int idx = lane_id; idx < total; idx += 32) {
                int row = idx / TileScaleCount;
                int col = idx - row * TileScaleCount;
                int global_scale = scale_offset + col;
                uint8_t val = 0;
                if (row < tile_rows && global_scale < K_scales) {
                    size_t sfa_idx = (static_cast<size_t>(m_tile + row) * K_scales) + global_scale;
                    DEBUG_OOB_GLOBAL_1D("SFA_batch", sfa_idx, static_cast<size_t>(M) * K_scales, SFA_batch);
                    val = SFA_batch[sfa_idx];
                } else {
#ifndef NDEBUG
                    DEBUG_PRINT_ERROR("WARN: SFA_batch index OOB: row=%d global_scale=%d tile_rows=%d K_scales=%d at %s:%d\n",
                                      row, global_scale, tile_rows, K_scales, __FILE__, __LINE__);
#endif
                }
                int sfa_smem_idx = row * TileScaleCount + col;
                DEBUG_OOB_SMEM_1D("sfa_stage", sfa_smem_idx, TileM * TileScaleCount, sfa_stage[stage]);
                sfa_stage[stage][sfa_smem_idx] = val;
            }
        }
    };

    prefetch_tile(0, 0);
    if (TileK < K) {
        prefetch_tile(1, TileK);
    }

    float c_frag_0 = 0.0f;
    float c_frag_1 = 0.0f;
    float c_frag_2 = 0.0f;
    float c_frag_3 = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TileK) {
        int tile_idx = k_tile / TileK;
        int stage = tile_idx % StageCount;

        int next_stage = (stage + 2) % StageCount;
        int next_k = k_tile + 2 * TileK;

        if (next_k < K) {
            prefetch_tile(next_stage, next_k);
        }

        if (use_tma_a) {
#if __CUDA_ARCH__ >= 900
            mbarrier_wait_parity(mbar_stage(mbar_a, stage), stage_phase_smem[stage]);
            __syncthreads();
            if (tid == 0) {
                stage_phase_smem[stage] ^= 1;
            }
#endif
        } else {
            cp_async_wait();
        }
        __syncthreads();

        int curr_k = (K - k_tile) < TileK ? (K - k_tile) : TileK;
        int curr_cols = (curr_k + 1) >> 1;
        int scale_count = (curr_k + 15) >> 4;

        for (int idx = tid; idx < tile_rows * scale_count; idx += Threads) {
            int row = idx / scale_count;
            int s = idx - row * scale_count;
            half scale_h = __float2half(0.0f);
            if (row < tile_rows) {
                int sfa_idx = row * TileScaleCount + s;
                DEBUG_OOB_SMEM_1D("sfa_stage", sfa_idx, TileM * TileScaleCount, sfa_stage[stage]);
                scale_h = __float2half(decode_fp8_e4m3(sfa_stage[stage][sfa_idx]));
            }
            int a_scale_idx = row * TileScaleCount + s;
            DEBUG_OOB_SMEM_1D("a_scale_smem", a_scale_idx, TileM * TileScaleCount, a_scale_smem);
            a_scale_smem[a_scale_idx] = scale_h;
        }
        __syncthreads();

        {
            uint8_t* a_stage = a_packed_stage[stage];
            for (int idx = tid; idx < tile_rows * curr_cols; idx += Threads) {
                int row = idx / curr_cols;
                int col_packed = idx - row * curr_cols;
                int k_base = k_tile + col_packed * 2;
                int a_smem_idx = row * TileKPacked + col_packed;
                DEBUG_OOB_SMEM_1D("a_packed_stage", a_smem_idx, TileM * TileKPacked, a_stage);
                uint8_t packed = a_stage[a_smem_idx];
                int scale_idx = row * TileScaleCount + (col_packed >> 3);
                DEBUG_OOB_SMEM_1D("a_scale_smem", scale_idx, TileM * TileScaleCount, a_scale_smem);
                half scale_h = a_scale_smem[scale_idx];
                half v0 = __float2half(0.0f);
                half v1 = __float2half(0.0f);
                // NVFP4 convention: low nibble = element 0, high nibble = element 1
                if (row < tile_rows && k_base < K) {
                    v0 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);         // LOW nibble → first element
                }
                if (row < tile_rows && (k_base + 1) < K) {
                    v1 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);  // HIGH nibble → second element
                }

#ifndef NDEBUG
                // Debug: print first few A decode values for row 0
                if (blockIdx.x == 0 && blockIdx.y == 0 && row == 0 && col_packed < 4 && k_tile == 0) {
                    DEBUG_PRINT_ERROR("A_decode: row=%d col=%d packed=0x%02x scale=%.4f v0=%.4f v1=%.4f\n",
                                      row, col_packed, packed, __half2float(scale_h), __half2float(v0), __half2float(v1));
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

                // Which 8x8 sub-matrix this thread addresses (0-3)
                int matrix_id = lane_id / 8;  // 0, 1, 2, or 3
                int row_in_matrix = lane_id % 8;  // 0-7

                // Matrix layout: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
                int block_row = (matrix_id >= 2) ? 8 : 0;  // 0 for matrices 0,1; 8 for matrices 2,3
                int block_col = (matrix_id & 1) ? 8 : 0;   // 0 for matrices 0,2; 8 for matrices 1,3

                int a_row = block_row + row_in_matrix;
                int a_col = block_col;

                uint32_t a_addr = a_base + (static_cast<uint32_t>(a_row) * a_stride + a_col) * sizeof(half);
                unsigned a0, a1, a2, a3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(a_addr)
                );

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
            }
        }

        __syncthreads();
    }

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

        // Since B was broadcast, all columns have same value.
        // We only write from threads holding col 0 to avoid duplication.
        if (col == 0) {
            int global_row0 = m_tile + warp_row_offset + row0;
            int global_row1 = m_tile + warp_row_offset + row1;

#ifndef NDEBUG
            if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 2) {
                DEBUG_PRINT_ERROR("D_STORE: warp=%d lane=%d octet=%d row0=%d row1=%d g0=%d g1=%d c0=%.2f c2=%.2f\n",
                                  warp_id, lane_id, octet, row0, row1,
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
    int64_t M, int64_t K, int64_t L
) {
    const uint8_t* A_ptr = A.data_ptr<uint8_t>();
    const uint8_t* B_ptr = B.data_ptr<uint8_t>();
    const uint8_t* SFA_ptr = SFA.data_ptr<uint8_t>();
    const uint8_t* SFB_ptr = SFB.data_ptr<uint8_t>();
    half* D_ptr = reinterpret_cast<half*>(D.data_ptr<at::Half>());

    constexpr int kTileM = 128;
    constexpr int kTileK = 128;
    constexpr int kThreads = 320;
    constexpr int kTileKPacked = kTileK / 2;
    constexpr int kTileScaleCount = kTileK / 16;
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

    // Match 128-byte alignment for TMA/TC regions in shared memory
    shared_bytes = align_up(shared_bytes, 128); // for a_packed_stage[0]
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileKPacked; // a_packed_stage[s]
    }

    shared_bytes = align_up(shared_bytes, 128); // for sfa_stage[0]
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount; // sfa_stage[s]
    }

    shared_bytes = align_up(shared_bytes, 128); // for a_f16_smem
    shared_bytes += static_cast<size_t>(kTileM) * kAStride * sizeof(__half); // a_f16_smem
    shared_bytes = align_up(shared_bytes, 128); // for b_tile_smem
    shared_bytes += static_cast<size_t>(kTileK) * 8 * sizeof(__half); // b_tile_smem
    shared_bytes = align_up(shared_bytes, 128); // for a_scale_smem
    shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount * sizeof(__half); // a_scale_smem

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
        printf("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ A_ptr 128-byte alignment check passed: %p\n", A_ptr);
    }

    // 3. Check that strides are properly 128-byte aligned
    if (K_packed % 128 != 0) {
        printf("WARNING: K_packed (%llu) is not 128-byte aligned, may cause TMA issues\n",
               (unsigned long long)K_packed);
    } else {
        printf("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ K_packed 128-byte stride alignment check passed: %llu bytes\n",
               (unsigned long long)K_packed);
    }

    auto encode_tma = [&](CUtensorMap* out,
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
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
    };

    if (tma_ok) {
        // Ensure box dimensions don't exceed TMA hardware limit (256) or tensor dimensions
        cuuint32_t box_k = static_cast<cuuint32_t>(std::min({(uint64_t)kTileKPacked, K_packed, (uint64_t)kTMABoxLimit}));
        cuuint32_t box_m = static_cast<cuuint32_t>(std::min({(uint64_t)kTileM, (uint64_t)M, (uint64_t)kTMABoxLimit}));

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
            printf("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ Tile size 128-byte alignment check passed: %u bytes\n", tile_bytes);
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

            printf("TMA Debug: Using RANK=2 for L=1\n");
            printf("TMA Debug: dims = [K_packed=%llu, M=%llu], box = [%u, %u]\n",
                   (unsigned long long)dims_A[0], (unsigned long long)dims_A[1],
                   box_A[0], box_A[1]);
            printf("TMA Debug: strides_A[0] = %llu bytes\n", (unsigned long long)strides_A[0]);

            // Validate box dimensions
            if (box_A[0] > 256 || box_A[1] > 256) {
                printf("ERROR: TMA box dimension exceeds 256 limit! box=[%u, %u]\n",
                       box_A[0], box_A[1]);
                tma_ok = false;
            }

            if (tma_ok) {
                CUresult resA = encode_tma(map_A_ptr,
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
                    printf("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ TMA Encode A (rank=2) SUCCESS!\n");
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
                CUresult resA = encode_tma(map_A_ptr,
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
                    printf("ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ TMA Encode A (rank=3) SUCCESS!\n");
                }
            }
        }
    }

    if (tma_ok) {
        check_cuda(cudaMalloc(&d_map_A, sizeof(CUtensorMap)), "cudaMalloc d_map_A");
        check_cuda(cudaMemcpy(d_map_A, map_A_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_A");
    }

    cudaFuncAttributes attr;
    cudaError_t attr_err = cudaFuncGetAttributes(&attr, fp4_gemv_streaming<kTileM, kTileK, kThreads>);
    if (attr_err != cudaSuccess) throw std::runtime_error(std::string("cudaFuncGetAttributes failed"));

    cudaError_t set_err = cudaFuncSetAttribute(
        fp4_gemv_streaming<kTileM, kTileK, kThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)
    );
    if (set_err != cudaSuccess) throw std::runtime_error(std::string("cudaFuncSetAttribute failed"));

    int num_blocks = static_cast<int>((M + kTileM - 1) / kTileM);
    int grid_x = num_blocks;
    int grid_y = static_cast<int>(L);
#ifndef NDEBUG
    // Debug mode: keep full grid to process all data, just add logging
    printf("DEBUG launch grid=(%d,%d) blockDim.x=%d shared_bytes=%zu M=%lld K=%lld L=%lld\n",
           grid_x, grid_y, kThreads, shared_bytes, (long long)M, (long long)K, (long long)L);
#endif
    dim3 grid(grid_x, grid_y);
    dim3 block(kThreads);

    fp4_gemv_streaming<kTileM, kTileK, kThreads><<<grid, block, shared_bytes>>>(
        A_ptr, B_ptr, SFA_ptr, SFB_ptr,
        d_map_A, d_map_B, d_map_SFB,
        D_ptr, static_cast<int>(M), static_cast<int>(K), static_cast<int>(L)
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (d_map_A) cudaFree(d_map_A);
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
            functions=[
                "launch_fp4_gemv_optimized",
            ],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
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


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    """Convert scale factor tensor to blocked format (matches reference.py exactly)."""
    rows, cols = input_matrix.shape

    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def custom_kernel(data: input_t) -> output_t:
    """
    SM100 FP4 GEMV with tensor cores: CUTLASS + CuTe + PTX only (NO WMMA)


    """
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

    # Scale factors: apply blocked packing exactly like reference.py
    # sfa_ref_cpu: [M, K_scales, L], sfb_ref_cpu: [128, K_scales, L]
    print(
        f"\n[SCALE DEBUG] sfa_ref_cpu shape={sfa_ref_cpu.shape}, device={sfa_ref_cpu.device}"
    )
    print(
        f"[SCALE DEBUG] sfb_ref_cpu shape={sfb_ref_cpu.shape}, device={sfb_ref_cpu.device}"
    )

    # Apply blocked packing per batch (same as reference.py)
    sfa_blocked_list = []
    sfb_blocked_list = []
    for l_idx in range(L):
        # sfa_ref_cpu[:, :, l_idx] is [M, K_scales]
        sfa_blocked = to_blocked(sfa_ref_cpu[:, :, l_idx])
        sfb_blocked = to_blocked(sfb_ref_cpu[:, :, l_idx])
        sfa_blocked_list.append(sfa_blocked)
        sfb_blocked_list.append(sfb_blocked)

    # Stack into [L, blocked_size] then view as bytes
    sfa_stacked = torch.stack(sfa_blocked_list, dim=0).cuda()  # [L, blocked_size]
    sfb_stacked = torch.stack(sfb_blocked_list, dim=0).cuda()  # [L, blocked_size]

    # Reinterpret as raw bytes for the CUDA kernel (uint8 view of float8 storage)
    sfa_bytes = sfa_stacked.view(torch.uint8)
    sfb_bytes = sfb_stacked.view(torch.uint8)

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
        print(
            f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ (Python) a_bytes 128-byte alignment check passed: {hex(a_ptr)}"
        )
    if b_ptr % 128 != 0:
        print(
            f"WARNING (Python): b_bytes data_ptr {hex(b_ptr)} is not 128-byte aligned "
            f"(mod128={b_ptr % 128})"
        )
    else:
        print(
            f"ÃƒÂ¢Ã…â€œÃ¢â‚¬Å“ (Python) b_bytes 128-byte alignment check passed: {hex(b_ptr)}"
        )
    if K_packed_val % 128 != 0:
        print(f"WARNING (Python): K_packed ({K_packed_val}) is not 128-byte aligned")

    # Launch SM100 tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()  # [M, 1, L]

    return c
