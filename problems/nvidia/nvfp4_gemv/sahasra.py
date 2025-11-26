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
    const CUtensorMap* __restrict__ desc_SFA,
    const CUtensorMap* __restrict__ desc_B,
    const CUtensorMap* __restrict__ desc_SFB,
    half* __restrict__ D,
    const int M, const int K, const int L
) {
#if __CUDA_ARCH__ >= 700
    constexpr int TileKPacked = TileK / 2;
    constexpr int TileScaleCount = TileK / 16;
    constexpr int SfaBoxK = 16;  // Must match host-side box_sfa_k used for TMA
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

    // TMA requires 128-byte alignment for destination addresses
    align_up_smem_128();

    uint8_t* b_packed_smem = smem + offset;
    offset += K_packed;

    align_up_smem_128();

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

    // Scales used alongside TMA data keep them 128-byte aligned as well
    align_up_smem_128();
    uint8_t* sfa_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfa_stage[s] = smem + offset;
        offset += TileM * SfaBoxK;
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
#else
    const bool use_tma_a = false;
#endif

    __shared__ uint32_t stage_phase_smem[StageCount];

    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            stage_phase_smem[s] = 0;
        }
        __threadfence_block();  // Ensure stage_phase init visible to all warps
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
    const int sfb_chunks = (K_scales + (kTMABoxLimit - 1)) / kTMABoxLimit;

    if (warp_id == 0 && lane_id == 0) {
        // Calculate total expected bytes for all chunks
        uint32_t total_b_bytes = static_cast<uint32_t>(K_packed);
        uint32_t total_sfb_bytes = static_cast<uint32_t>(K_scales);

        // Set barrier to expect all bytes at once
        mbarrier_arrive_expect_tx(mbar_b, total_b_bytes);
        mbarrier_arrive_expect_tx(mbar_sfb, total_sfb_bytes);

        // Issue all TMA loads (barrier expects total bytes from all loads)
        for (int tile_idx = 0; tile_idx < b_chunks; ++tile_idx) {
            int elem_offset = tile_idx * kTMABoxLimit;
            int chunk_size = (elem_offset + kTMABoxLimit <= K_packed) ? kTMABoxLimit : (K_packed - elem_offset);
            (void)chunk_size;
            uint8_t* dst = b_packed_smem + elem_offset;

            uint32_t smem_addr = cvta_to_shared_u32(dst);
            if (L == 1) {
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3}], [%4];\n"
                    :
                    : "r"(smem_addr), "l"(desc_B),
                      "r"(static_cast<uint32_t>(tile_idx)), "r"(0u),
                      "r"(cvta_to_shared_u32(mbar_b))
                );
            } else {
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];\n"
                    :
                    : "r"(smem_addr), "l"(desc_B),
                      "r"(static_cast<uint32_t>(tile_idx)), "r"(0u), "r"(static_cast<uint32_t>(batch)),
                      "r"(cvta_to_shared_u32(mbar_b))
                );
            }
        }

        for (int tile_idx = 0; tile_idx < sfb_chunks; ++tile_idx) {
            int elem_offset = tile_idx * kTMABoxLimit;
            int chunk_size = (elem_offset + kTMABoxLimit <= K_scales) ? kTMABoxLimit : (K_scales - elem_offset);
            (void)chunk_size;
            uint8_t* dst = sfb_smem + elem_offset;

            uint32_t smem_addr = cvta_to_shared_u32(dst);
            if (L == 1) {
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3}], [%4];\n"
                    :
                    : "r"(smem_addr), "l"(desc_SFB),
                      "r"(static_cast<uint32_t>(tile_idx)), "r"(0u),
                      "r"(cvta_to_shared_u32(mbar_sfb))
                );
            } else {
                asm volatile(
                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2, %3, %4}], [%5];\n"
                    :
                    : "r"(smem_addr), "l"(desc_SFB),
                      "r"(static_cast<uint32_t>(tile_idx)), "r"(0u), "r"(static_cast<uint32_t>(batch)),
                      "r"(cvta_to_shared_u32(mbar_sfb))
                );
            }
        }
    }

    // All threads wait on barriers
    mbarrier_wait_parity(mbar_b, 0);
    mbarrier_wait_parity(mbar_sfb, 0);
#endif
    __syncthreads();

        // DEBUG: Print first 4 SFB scale bytes and first 4 B packed bytes (first tile only)
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("=== KERNEL DEBUG (block 0,0) ===\n");
        printf("SFB raw bytes [0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
               sfb_smem[0], sfb_smem[1], sfb_smem[2], sfb_smem[3]);
        printf("SFB decoded FP8 [0-3]: %.6f %.6f %.6f %.6f\n",
               decode_fp8_e4m3(sfb_smem[0]), decode_fp8_e4m3(sfb_smem[1]),
               decode_fp8_e4m3(sfb_smem[2]), decode_fp8_e4m3(sfb_smem[3]));
        printf("B packed bytes [0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
               b_packed_smem[0], b_packed_smem[1], b_packed_smem[2], b_packed_smem[3]);

        for (int i = 0; i < 4; i++) {
            size_t sfb_idx = static_cast<size_t>(batch) * 128 * K_scales + static_cast<size_t>(i);
            printf("SFB contiguous_idx[row=0,col=%d,batch=%d] = %llu\n",
                   i, batch, (unsigned long long)sfb_idx);
        }
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
        // FIXED: HIGH nibble = element 0, LOW nibble = element 1 (matches torch._scaled_mm)
        half v0 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);  // HIGH nibble -> element 0
        half v1 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);         // LOW nibble -> element 1

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
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("b_vec_smem decoded [0-7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               __half2float(b_vec_smem[0]), __half2float(b_vec_smem[1]),
               __half2float(b_vec_smem[2]), __half2float(b_vec_smem[3]),
               __half2float(b_vec_smem[4]), __half2float(b_vec_smem[5]),
               __half2float(b_vec_smem[6]), __half2float(b_vec_smem[7]));
    }
    __syncthreads();

    auto prefetch_tile = [&](int stage, int k_tile_base) {
        if (use_tma_a && is_producer) {
            if (warp_id == 0 && lane_id == 0) {
                // Element-space coordinates: row and packed-K index
                uint32_t c_m = static_cast<uint32_t>(m_tile);
                uint32_t c_k_packed = static_cast<uint32_t>(k_tile_base >> 1);
                uint32_t c_k_scales = static_cast<uint32_t>(k_tile_base >> 4);

                if (L == 1) {
                    // Load A matrix tile (2D)
                    uint32_t c0 = c_k_packed;
                    uint32_t c1 = c_m;
                    bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);
                    bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);

                    if (valid_m && valid_k) {
                        tma_load_2d(
                            a_packed_stage[stage],
                            desc_A,
                            c0, c1,
                            static_cast<uint32_t>((TileM * TileKPacked + 15) & ~15),
                            mbar_stage(mbar_a, stage)
                        );
                    } else {
                        mbarrier_arrive_expect_tx(mbar_stage(mbar_a, stage), 0);
                    }

                    // Load SFA scales tile (2D) using element-space coordinates
                    uint32_t sfa_c0 = c_k_scales;
                    uint32_t sfa_c1 = c_m;
                    bool valid_sfa_k = (c_k_scales + SfaBoxK) <= static_cast<uint32_t>(K_scales);
                    bool valid_sfa_m = (c_m + TileM) <= static_cast<uint32_t>(M);

                    if (valid_sfa_m && valid_sfa_k) {
                        tma_load_2d(
                            sfa_stage[stage],
                            desc_SFA,
                            sfa_c0, sfa_c1,
                            static_cast<uint32_t>(TileM * SfaBoxK),
                            mbar_stage(mbar_a, stage)
                        );
                    }
                } else {
                    // Load A matrix tile (3D)
                    uint32_t c0 = c_k_packed;
                    uint32_t c1 = c_m;
                    uint32_t c2 = static_cast<uint32_t>(batch);
                    bool valid_batch = c2 < static_cast<uint32_t>(L);
                    bool valid_m = (c_m + TileM) <= static_cast<uint32_t>(M);
                    bool valid_k = (c_k_packed + TileKPacked) <= static_cast<uint32_t>(K_packed);

                    if (valid_batch && valid_m && valid_k) {
                        tma_load_3d(
                            a_packed_stage[stage],
                            desc_A,
                            c0, c1, c2,
                            static_cast<uint32_t>(TileM * TileKPacked),
                            mbar_stage(mbar_a, stage)
                        );
                    } else {
                        mbarrier_arrive_expect_tx(mbar_stage(mbar_a, stage), 0);
                    }

                    // Load SFA scales tile (3D) using element-space coordinates
                    uint32_t sfa_c0 = c_k_scales;
                    uint32_t sfa_c1 = c_m;
                    uint32_t sfa_c2 = c2;
                    bool valid_sfa_k = (c_k_scales + SfaBoxK) <= static_cast<uint32_t>(K_scales);
                    bool valid_sfa_m = (c_m + TileM) <= static_cast<uint32_t>(M);

                    if (valid_batch && valid_sfa_m && valid_sfa_k) {
                        tma_load_3d(
                            sfa_stage[stage],
                            desc_SFA,
                            sfa_c0, sfa_c1, sfa_c2,
                            static_cast<uint32_t>(TileM * SfaBoxK),
                            mbar_stage(mbar_a, stage)
                        );
                    }
                }
            }
        }
        // All scale loading now done via TMA - no manual copies needed
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

        {
            uint8_t* a_stage = a_packed_stage[stage];
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
                        int sfa_idx = row * SfaBoxK + scale_col;
                        DEBUG_OOB_SMEM_1D("sfa_stage", sfa_idx, TileM * SfaBoxK, sfa_stage[stage]);
                        scale_h = __float2half(decode_fp8_e4m3(sfa_stage[stage][sfa_idx]));
                    }
                }
                half v0 = __float2half(0.0f);
                half v1 = __float2half(0.0f);
                // FIXED: HIGH nibble = element 0, LOW nibble = element 1 (matches torch._scaled_mm)
                if (row < tile_rows && k_base < K) {
                    v0 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);  // HIGH nibble -> element 0
                }
                if (row < tile_rows && (k_base + 1) < K) {
                    v1 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);         // LOW nibble -> element 1
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
        __syncthreads();

        // DEBUG: Print first 8 decoded A[row=0] values and first 4 SFA raw bytes (first tile only)
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0) {
            printf("SFA raw bytes [row=0, 0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
                   sfa_stage[stage][0], sfa_stage[stage][1],
                   sfa_stage[stage][2], sfa_stage[stage][3]);
            printf("SFA decoded FP8 [row=0, 0-3]: %.6f %.6f %.6f %.6f\n",
                   decode_fp8_e4m3(sfa_stage[stage][0]), decode_fp8_e4m3(sfa_stage[stage][1]),
                   decode_fp8_e4m3(sfa_stage[stage][2]), decode_fp8_e4m3(sfa_stage[stage][3]));
            printf("A packed bytes [row=0, 0-3]: 0x%02x 0x%02x 0x%02x 0x%02x\n",
                   a_packed_stage[stage][0], a_packed_stage[stage][1],
                   a_packed_stage[stage][2], a_packed_stage[stage][3]);
            half* a_row0 = a_f16_smem;
            printf("A decoded [row=0, 0-7]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   __half2float(a_row0[0]), __half2float(a_row0[1]),
                   __half2float(a_row0[2]), __half2float(a_row0[3]),
                   __half2float(a_row0[4]), __half2float(a_row0[5]),
                   __half2float(a_row0[6]), __half2float(a_row0[7]));

            // Print CuTe blocked indices for SFA at this tile origin
            int nmblocks_sfa = (M + 127) / 128;
            int nkblocks_sfa = (K_scales + 3) / 4;
        for (int i = 0; i < 4; i++) {
        size_t sfa_idx = static_cast<size_t>(batch) * M * K_scales
                       + static_cast<size_t>(m_tile) * K_scales
                       + static_cast<size_t>(i);
        printf("SFA contiguous_idx[row=%d,col=%d,batch=%d] = %llu\n",
               m_tile, i, batch, (unsigned long long)sfa_idx);
        }
            printf("=== END KERNEL DEBUG ===\n");
        }
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

                // Which 8x8 sub-matrix this thread addresses (0-3)
                int matrix_id = lane_id / 8;  // 0, 1, 2, or 3
                int row_in_matrix = lane_id % 8;  // 0-7

                // PTX layout: LSB = row block, MSB = column block. Swap fixes A01/A10 mixup.
                //   matrix_id 0 -> rows 0-7,  cols 0-7
                //   matrix_id 1 -> rows 8-15, cols 0-7
                //   matrix_id 2 -> rows 0-7,  cols 8-15
                //   matrix_id 3 -> rows 8-15, cols 8-15
                int block_row = (matrix_id & 1) * 8;        // LSB chooses top/bottom
                int block_col = (matrix_id >> 1) * 8;       // MSB chooses left/right

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

                // DEBUG: Print accumulator after first MMA in first K tile
                if (blockIdx.x == 0 && blockIdx.y == 0 && k_tile == 0 && kk == 0 && warp_id == 2 && lane_id == 0) {
                    printf("MMA DEBUG: k_tile=%d kk=%d warp=%d lane=%d c_frag=[%.4f, %.4f, %.4f, %.4f]\n",
                           k_tile, kk, warp_id, lane_id, c_frag_0, c_frag_1, c_frag_2, c_frag_3);
                }
            }
        }

        __syncthreads();
    }

    // DEBUG: Print final accumulator before store
    if (blockIdx.x == 0 && blockIdx.y == 0 && is_consumer && (warp_id - 2) < ((tile_rows + 15) / 16)) {
        int octet = lane_id / 4;
        int tid_in_octet = lane_id % 4;
        if (tid_in_octet == 0 && octet == 0 && warp_id == 2) {
            printf("FINAL ACCUM: warp=%d lane=%d c_frag_0=%.4f c_frag_2=%.4f (rows %d,%d)\n",
                   warp_id, lane_id, c_frag_0, c_frag_2,
                   m_tile + (warp_id-2)*16 + octet, m_tile + (warp_id-2)*16 + octet + 8);
        }
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
    constexpr int kSfaBoxK = 16;  // Must match device-side SfaBoxK
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
        shared_bytes += static_cast<size_t>(kTileM) * kSfaBoxK; // sfa_stage[s]
    }

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
                    printf("✅TMA Encode A (rank=2) SUCCESS!\n");
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
                    printf("✅TMA Encode A (rank=3) SUCCESS!\n");
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

            CUresult resB = encode_tma(map_B_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                       2, B_ptr, dims_B, strides_B, box_B);
            if (resB != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resB, &err_str);
                printf("TMA Encode B failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_B, sizeof(CUtensorMap)), "cudaMalloc d_map_B");
                check_cuda(cudaMemcpy(d_map_B, map_B_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_B");
                printf("✅TMA Encode B (rank=2) SUCCESS!\n");
            }
        } else {
            // B: rank-3, actual layout is [L, 128, K_packed]
            // TMA dims order: [K_packed, 128, L] (innermost to outermost)
            cuuint64_t dims_B[3] = {K_packed, 128, static_cast<cuuint64_t>(L)};
            cuuint32_t box_B[3] = {static_cast<cuuint32_t>(K_packed < 256ULL ? K_packed : 256ULL), 1, 1};  // Clamp to TMA 256 limit
            cuuint64_t strides_B[2] = {K_packed, 128 * K_packed};  // Row stride, batch stride

            CUresult resB = encode_tma(map_B_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
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

            CUresult resSFB = encode_tma(map_SFB_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         2, SFB_ptr, dims_SFB, strides_SFB, box_SFB);
            if (resSFB != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resSFB, &err_str);
                printf("TMA Encode SFB failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_SFB, sizeof(CUtensorMap)), "cudaMalloc d_map_SFB");
                check_cuda(cudaMemcpy(d_map_SFB, map_SFB_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFB");
                printf("✅TMA Encode SFB (rank=2) SUCCESS!\n");
            }
        } else {
            // SFB: rank-3, actual layout is [L, 128, K_scales]
            // TMA dims order: [K_scales, 128, L] (innermost to outermost)
            cuuint64_t dims_SFB[3] = {K_scales, 128, static_cast<cuuint64_t>(L)};
            cuuint32_t box_SFB[3] = {static_cast<cuuint32_t>(K_scales < 256ULL ? K_scales : 256ULL), 1, 1};
            cuuint64_t strides_SFB[2] = {K_scales, 128 * K_scales};

            CUresult resSFB = encode_tma(map_SFB_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
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

        cuuint32_t box_sfa_k = static_cast<cuuint32_t>(std::min<int64_t>(kSfaBoxK, K_scales));
        cuuint32_t box_sfa_m = static_cast<cuuint32_t>(
            kTileM < M ?
                (kTileM < 16ULL ? 16ULL : (kTileM < 256ULL ? kTileM : 256ULL)) :
                (M < 16ULL ? 16ULL : (M < 256ULL ? M : 256ULL))
        );

        if (L == 1) {
            // SFA: rank-2, actual layout is [1, M, K_scales] but L=1 collapses to [M, K_scales]
            // TMA dims order: [K_scales, M] (innermost to outermost)
            cuuint64_t dims_SFA[2] = {K_scales, static_cast<cuuint64_t>(M)};
            cuuint32_t box_SFA[2] = {box_sfa_k, box_sfa_m};
            cuuint64_t strides_SFA[1] = {K_scales};

            printf("TMA SFA (rank=2): dims=[%llu,%llu] box=[%u,%u] stride=[%llu] ptr=%p\n",
                   (unsigned long long)dims_SFA[0], (unsigned long long)dims_SFA[1],
                   box_SFA[0], box_SFA[1], (unsigned long long)strides_SFA[0], SFA_ptr);

            uintptr_t sfa_addr = reinterpret_cast<uintptr_t>(SFA_ptr);
            if ((sfa_addr % 128) != 0) {
                printf("ERROR: SFA_ptr not 128-byte aligned: %p (mod128=%llu)\n",
                       SFA_ptr, (unsigned long long)(sfa_addr % 128));
            }

            CUresult resSFA = encode_tma(map_SFA_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                         2, SFA_ptr, dims_SFA, strides_SFA, box_SFA);
            if (resSFA != CUDA_SUCCESS) {
                const char* err_str = nullptr;
                cuGetErrorString(resSFA, &err_str);
                printf("TMA Encode SFA failed: %s\n", err_str ? err_str : "unknown");
                tma_ok = false;
            } else {
                check_cuda(cudaMalloc(&d_map_SFA, sizeof(CUtensorMap)), "cudaMalloc d_map_SFA");
                check_cuda(cudaMemcpy(d_map_SFA, map_SFA_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFA");
                printf("✅TMA Encode SFA (rank=2) SUCCESS!\n");
            }
        } else {
            // SFA: rank-3, actual layout is [L, M, K_scales]
            // TMA dims order: [K_scales, M, L] (innermost to outermost)
            cuuint64_t dims_SFA[3] = {K_scales, static_cast<cuuint64_t>(M), static_cast<cuuint64_t>(L)};
            cuuint32_t box_SFA[3] = {box_sfa_k, box_sfa_m, 1};
            cuuint64_t strides_SFA[2] = {K_scales, M * K_scales};

            CUresult resSFA = encode_tma(map_SFA_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
        d_map_A, d_map_SFA, d_map_B, d_map_SFB,
        D_ptr, static_cast<int>(M), static_cast<int>(K), static_cast<int>(L)
    );

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
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()  # [M, 1, L]

    return c
