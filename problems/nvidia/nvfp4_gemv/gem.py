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

    // We expect M to be padded, so m_tile is always valid for TMA.
    if (batch >= L) return;

    const int K_packed = K >> 1;
    const int K_scales = K >> 4;
    const int tile_rows = (M - m_tile) < TileM ? (M - m_tile) : TileM;

    const uint8_t* A_batch = A_packed + static_cast<size_t>(batch) * M * K_packed;
    const uint8_t* B_batch = B_packed + static_cast<size_t>(batch) * 128 * K_packed;
    const uint8_t* SFA_batch = SFA_packed + static_cast<size_t>(batch) * M * K_scales;
    const uint8_t* SFB_batch = SFB_packed + static_cast<size_t>(batch) * 128 * K_scales;
    half* D_batch = D + static_cast<size_t>(batch) * M;

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

    // Force 128-byte alignment for TMA destination buffers
    {
        uintptr_t current_ptr = (uintptr_t)(smem + offset);
        uintptr_t aligned_ptr = (current_ptr + 127) & ~127ULL;
        size_t padding = aligned_ptr - current_ptr;
        offset += padding;
    }

    uint8_t* a_packed_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        a_packed_stage[s] = smem + offset;
        offset += TileM * TileKPacked;
    }

    offset = align_up(offset, 16);

    uint8_t* sfa_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfa_stage[s] = smem + offset;
        offset += TileM * TileScaleCount;
    }

    half* a_f16_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileM) * a_stride * sizeof(half);
    offset = align_up(offset, 64);

    half* b_tile_smem = reinterpret_cast<half*>(smem + offset);
    offset += static_cast<size_t>(TileK) * 8 * sizeof(half);
    offset = align_up(offset, 16);

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

#if __CUDA_ARCH__ >= 900
    if (tid == 0 && use_tma_a) {
        for (int s = 0; s < StageCount; ++s) {
            mbarrier_init(mbar_stage(mbar_a, s));
        }
    }
#endif
    __syncthreads();

    // Load B and SFB (Full B is loaded since K is assumed to fit in Shared Memory budget for this benchmark)
    // NOTE: This assumes K is relatively small (e.g. 8192).
    {
        int b_segments = (K_packed + 15) / 16;
        for (int idx = tid; idx < b_segments; idx += Threads) {
            int byte_idx = idx * 16;
            bool full = (byte_idx + 16) <= K_packed;
            uint8_t* dst = b_packed_smem + byte_idx;
            const uint8_t* src = B_batch + byte_idx;
            cp_async_16b(dst, src, full);
            if (!full && byte_idx < K_packed) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int g = byte_idx + i;
                    if (g < K_packed) {
                        dst[i] = src[i];
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
            cp_async_16b(dst, src, full);
            if (!full && byte_idx < K_scales) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int g = byte_idx + i;
                    if (g < K_scales) {
                        dst[i] = src[i];
                    }
                }
            }
        }
        cp_async_commit();
        cp_async_wait();
    }
    __syncthreads();

    // Decode B
    for (int idx = tid; idx < K_packed; idx += Threads) {
        int k_base = idx * 2;
        int scale_idx = idx >> 3;
        uint8_t packed = b_packed_smem[idx];
        half scale_h = __float2half(0.0f);
        if (scale_idx < K_scales) {
            scale_h = __float2half(decode_fp8_e4m3(sfb_smem[scale_idx]));
        }
        half v0 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);
        half v1 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);
        if (k_base < K) b_vec_smem[k_base] = v0;
        if (k_base + 1 < K) b_vec_smem[k_base + 1] = v1;
    }
    __syncthreads();

    auto prefetch_tile = [&](int stage, int k_tile_base) {
        if (use_tma_a) {
            if (warp_id == 0 && lane_id == 0) {
                uint32_t c_m = static_cast<uint32_t>(m_tile);
                uint32_t c_k_packed = static_cast<uint32_t>(k_tile_base >> 1);

                if (L == 1) {
                    tma_load_2d(
                        a_packed_stage[stage],
                        desc_A,
                        c_m,
                        c_k_packed,
                        static_cast<uint32_t>((TileM * TileKPacked + 15) & ~15),
                        mbar_stage(mbar_a, stage)
                    );
                } else {
                    tma_load_3d(
                        a_packed_stage[stage],
                        desc_A,
                        static_cast<uint32_t>(batch),
                        c_m,
                        c_k_packed,
                        static_cast<uint32_t>(TileM * TileKPacked),
                        mbar_stage(mbar_a, stage)
                    );
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
                bool valid_row = (row < TileM);
                bool full = (col + 16) <= TileKPacked;
                cp_async_16b(dst, src, valid_row && full);
                if (valid_row && !full && col < TileKPacked) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        int g = col + i;
                        if (g < TileKPacked) dst[i] = src[i];
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
                if (row < TileM && global_scale < K_scales) {
                    val = SFA_batch[(static_cast<size_t>(m_tile + row) * K_scales) + global_scale];
                }
                sfa_stage[stage][row * TileScaleCount + col] = val;
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
                scale_h = __float2half(decode_fp8_e4m3(sfa_stage[stage][row * TileScaleCount + s]));
            }
            a_scale_smem[row * TileScaleCount + s] = scale_h;
        }
        __syncthreads();

        {
            uint8_t* a_stage = a_packed_stage[stage];
            for (int idx = tid; idx < tile_rows * curr_cols; idx += Threads) {
                int row = idx / curr_cols;
                int col_packed = idx - row * curr_cols;
                uint8_t packed = a_stage[row * TileKPacked + col_packed];
                half scale_h = a_scale_smem[row * TileScaleCount + (col_packed >> 3)];
                half v0 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);
                half v1 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);

                half* a_dst = a_f16_smem + row * a_stride;
                a_dst[col_packed * 2] = v0;
                a_dst[col_packed * 2 + 1] = v1;
            }
        }

        if (is_producer && warp_id == 1) {
            for (int kk = lane_id; kk < curr_k; kk += 32) {
                half v = __float2half(0.0f);
                if (k_tile + kk < K) {
                    v = b_vec_smem[k_tile + kk];
                }
                half* b_row = b_tile_smem + kk * 8;
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    b_row[n] = v;
                }
            }
        }
        __syncthreads();

        int active_warps = (tile_rows + 15) / 16;
        if (is_consumer && (warp_id - 2) < active_warps) {
            int warp_row = (warp_id - 2) * 16;
            for (int kk = 0; kk < curr_k; kk += 16) {
                const half* a_tile_ptr = a_f16_smem + warp_row * a_stride + kk;
                uint32_t a_base = cvta_to_shared_u32(a_tile_ptr);
                int a_row = lane_id & 15;
                int a_col_block = (lane_id >> 4) & 0x1;
                uint32_t a_addr = a_base + (static_cast<uint32_t>(a_row) * a_stride + a_col_block * 8) * sizeof(half);
                unsigned a0, a1, a2, a3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "r"(a_addr)
                );

                const half* b_tile_ptr = b_tile_smem + kk * 8;
                uint32_t b_base = cvta_to_shared_u32(b_tile_ptr);
                int b_row = lane_id & 15;
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

    int active_warps_total = (tile_rows + 15) / 16;
    if (is_consumer && (warp_id - 2) < active_warps_total) {
        int quad = lane_id >> 2;
        int col_in_quad = lane_id & 3;

        if (col_in_quad == 0) {
            int warp_row_offset = (warp_id - 2) * 16;
            int row0 = quad;
            int row1 = quad + 8;

            int global_row0 = m_tile + warp_row_offset + row0;
            int global_row1 = m_tile + warp_row_offset + row1;

            if (global_row0 < M) {
                D_batch[global_row0] = __float2half(c_frag_0);
            }
            if (global_row1 < M) {
                D_batch[global_row1] = __float2half(c_frag_2);
            }
        }
    }
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
    constexpr int kTMABoxLimit = 256;

    size_t shared_bytes = 0;
    auto align_up = [](size_t x, size_t align) { return (x + align - 1) & ~(align - 1); };

    // Alloc calculation
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(kStageCount) * 16; // mbar_a
    shared_bytes += 16; // mbar_b
    shared_bytes += 16; // mbar_sfb

    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 2;    // b_packed_smem
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 16;   // sfb_smem
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) * sizeof(__half); // b_vec_smem

    // Add extra padding for 128-byte alignment
    shared_bytes += 256;

    for (int s = 0; s < kStageCount; ++s) shared_bytes += static_cast<size_t>(kTileM) * kTileKPacked;
    for (int s = 0; s < kStageCount; ++s) shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount;

    shared_bytes += static_cast<size_t>(kTileM) * kAStride * sizeof(__half);
    shared_bytes = align_up(shared_bytes, 64);
    shared_bytes += static_cast<size_t>(kTileK) * 8 * sizeof(__half);
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount * sizeof(__half);

    auto check_cuda = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    };

    const uint64_t K_packed = static_cast<uint64_t>(K / 2);
    bool use_tma_a = true;

    // Use heap-aligned allocation for CUtensorMap
    std::vector<uint8_t> map_buf(sizeof(CUtensorMap) + 64);
    void* ptr = map_buf.data();
    size_t space = map_buf.size();
    void* aligned_ptr = std::align(64, sizeof(CUtensorMap), ptr, space);
    if (!aligned_ptr) throw std::runtime_error("Failed to align CUtensorMap");

    std::memset(aligned_ptr, 0, sizeof(CUtensorMap));
    CUtensorMap* map_A_ptr = reinterpret_cast<CUtensorMap*>(aligned_ptr);
    CUtensorMap* d_map_A = nullptr;

    uintptr_t base_addr = reinterpret_cast<uintptr_t>(A_ptr);
    if ((base_addr % 16) != 0) use_tma_a = false;

    if (use_tma_a) {
        cuuint32_t box_k = static_cast<cuuint32_t>(std::min({(uint64_t)kTileKPacked, K_packed, (uint64_t)kTMABoxLimit}));
        cuuint32_t box_m = static_cast<cuuint32_t>(std::min({(uint64_t)kTileM, (uint64_t)M, (uint64_t)kTMABoxLimit}));

        // Element strides
        cuuint32_t elementStrides[5] = {1, 1, 1, 1, 1};
        CUresult res;

        if (L == 1) {
            cuuint64_t dims_A[2] = {static_cast<cuuint64_t>(M), static_cast<cuuint64_t>(K_packed)};
            cuuint32_t box_A[2] = {box_m, box_k};
            cuuint64_t strides_A[1] = {static_cast<cuuint64_t>(K_packed)};
            res = cuTensorMapEncodeTiled(
                map_A_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>((void*)A_ptr),
                dims_A, strides_A, box_A, elementStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
        } else {
            cuuint64_t dims_A[3] = {static_cast<cuuint64_t>(L), static_cast<cuuint64_t>(M), static_cast<cuuint64_t>(K_packed)};
            cuuint32_t box_A[3] = {1u, box_m, box_k};
            cuuint64_t strides_A[2] = {static_cast<cuuint64_t>(M) * K_packed, static_cast<cuuint64_t>(K_packed)};
            res = cuTensorMapEncodeTiled(
                map_A_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>((void*)A_ptr),
                dims_A, strides_A, box_A, elementStrides,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
            );
        }

        if (res != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(res, &err_str);
            throw std::runtime_error(std::string("TMA Encode failed: ") + (err_str ? err_str : "unknown"));
        }

        check_cuda(cudaMalloc(&d_map_A, sizeof(CUtensorMap)), "cudaMalloc d_map_A");
        check_cuda(cudaMemcpy(d_map_A, map_A_ptr, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_A");
    }

    cudaFuncAttributes attr;
    cudaError_t attr_err = cudaFuncGetAttributes(&attr, fp4_gemv_streaming<kTileM, kTileK, kThreads>);
    if (attr_err != cudaSuccess) throw std::runtime_error("cudaFuncGetAttributes failed");

    cudaError_t set_err = cudaFuncSetAttribute(fp4_gemv_streaming<kTileM, kTileK, kThreads>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
    if (set_err != cudaSuccess) throw std::runtime_error("cudaFuncSetAttribute failed");

    int num_blocks = static_cast<int>((M + kTileM - 1) / kTileM);
    dim3 grid(num_blocks, static_cast<int>(L));
    dim3 block(kThreads);

    fp4_gemv_streaming<kTileM, kTileK, kThreads><<<grid, block, shared_bytes>>>(
        A_ptr, B_ptr, SFA_ptr, SFB_ptr,
        d_map_A, nullptr, nullptr,
        D_ptr, static_cast<int>(M), static_cast<int>(K), static_cast<int>(L)
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (d_map_A) cudaFree(d_map_A);
    if (err != cudaSuccess) check_cuda(err, "Kernel execution failed");
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
                "-gencode=arch=compute_100a,code=sm_100a",
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
    a, b, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c = data
    M, _, L = c.shape
    K = a.shape[1] * 2
    K_scales = K // 16

    # 1. Pad dimensions to multiples of 128
    pad_m = (128 - M % 128) % 128
    pad_k = (128 - K % 128) % 128  # Logical K padding

    # K is in units of FP4 elements.
    # a is [L, M, K/2]. We need to pad K dimension of a (dim 2) by pad_k/2.
    # We must also pad M dimension (dim 1).

    a_in = a.clone().permute(2, 0, 1).contiguous()  # [L, M, K/2]

    if pad_m > 0 or pad_k > 0:
        a_in = torch.nn.functional.pad(a_in, (0, pad_k // 2, 0, pad_m), value=0)

    a_bytes = a_in.contiguous().cuda().view(torch.uint8)

    # 2. Pad and Broadcast B
    # b is [L, 1, K/2] ideally.
    # We need to pad K, and broadcast to 128 rows for the kernel.
    b_in = (
        b.clone().permute(2, 0, 1).contiguous()
    )  # [L, 1, K/2] assuming input b matches

    if pad_k > 0:
        b_in = torch.nn.functional.pad(b_in, (0, pad_k // 2), value=0)

    # Broadcast to [L, 128, K_padded/2]
    b_expanded = b_in.expand(-1, 128, -1).contiguous()
    b_bytes = b_expanded.cuda().view(torch.uint8)

    # 3. SFA padding
    if sfa_permuted is not None and list(sfa_permuted.shape) == [L, M, K_scales]:
        sfa_in = sfa_permuted.contiguous()
    else:
        sfa_in = sfa_ref_cpu.clone().permute(2, 0, 1).contiguous()

    # SFA is [L, M, K_scales]. Pad M and K_scales.
    # K_scales padding corresponds to pad_k / 16.
    pad_scales = pad_k // 16
    if pad_m > 0 or pad_scales > 0:
        sfa_in = torch.nn.functional.pad(sfa_in, (0, pad_scales, 0, pad_m), value=0)

    sfa_bytes = sfa_in.cuda().view(torch.uint8)

    # 4. SFB padding and broadcast
    if sfb_permuted is not None and list(sfb_permuted.shape) == [L, 128, K_scales]:
        sfb_in = sfb_permuted.contiguous()
    else:
        sfb_in = sfb_ref_cpu.clone().permute(2, 0, 1).contiguous()
        # Ensure sfb has 128 dim if ref is [L, 1, K]
        if sfb_in.size(1) == 1:
            sfb_in = sfb_in.expand(-1, 128, -1).contiguous()

    # Pad scales
    if pad_scales > 0:
        sfb_in = torch.nn.functional.pad(sfb_in, (0, pad_scales), value=0)

    sfb_bytes = sfb_in.cuda().view(torch.uint8)

    M_padded = M + pad_m
    K_padded = K + pad_k

    # Allocate output with padded M
    d_out = torch.empty((L, M_padded), dtype=torch.float16, device="cuda")

    mod = get_module()
    mod.launch_fp4_gemv_optimized(
        a_bytes, b_bytes, sfa_bytes, sfb_bytes, d_out, M_padded, K_padded, L
    )

    # Slice and permute back to [M, 1, L]
    d_sliced = d_out[:, :M]  # [L, M]
    c_out = d_sliced.permute(1, 0).unsqueeze(1).contiguous()  # [M, 1, L]

    return c_out
