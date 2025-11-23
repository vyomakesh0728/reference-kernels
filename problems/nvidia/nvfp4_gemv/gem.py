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

__device__ __forceinline__ uint64_t cvta_to_shared_u64(const void* ptr) {
    uint64_t addr;
    asm volatile("cvta.to.shared.u64 %0, %1;\n"
                 : "=l"(addr)
                 : "l"(ptr));
    return addr;
}

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void cp_async_16b(void* dst, const void* src, bool pred) {
    if (pred) {
        uint32_t smem_addr = static_cast<uint32_t>(cvta_to_shared_u64(dst));
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
    uint64_t mbar_addr = cvta_to_shared_u64(mbar);
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n"
        :
        : "l"(mbar_addr), "r"(1)
    );
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint64_t mbar_addr = cvta_to_shared_u64(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "l"(mbar_addr), "r"(bytes)
    );
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase) {
    uint64_t mbar_addr = cvta_to_shared_u64(mbar);
    asm volatile(
        "{\n"
        "  .reg .pred P;\n"
        "WAIT:\n"
        "  mbarrier.try_wait.parity.shared.b64 P, [%0], %1, 0;\n"
        "  @!P bra.uni WAIT;\n"
        "}\n"
        :
        : "l"(mbar_addr), "r"(phase)
    );
}

__device__ __forceinline__ void tma_load_2d(void* smem_ptr,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t coord1,
                                            uint32_t bytes,
                                            uint64_t* mbar) {
    uint64_t smem_addr = cvta_to_shared_u64(smem_ptr);
    uint64_t mbar_addr = cvta_to_shared_u64(mbar);
    mbarrier_arrive_expect_tx(mbar, bytes);

    uint32_t c0 = coord0;
    uint32_t c1 = coord1;

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :
        : "l"(smem_addr),
          "l"(desc),
          "r"(c0),
          "r"(c1),
          "l"(mbar_addr)
    );
}

__device__ __forceinline__ void tma_load_3d(void* smem_ptr,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t coord1,
                                            uint32_t coord2,
                                            uint32_t bytes,
                                            uint64_t* mbar) {
    uint64_t smem_addr = cvta_to_shared_u64(smem_ptr);
    uint64_t mbar_addr = cvta_to_shared_u64(mbar);
    mbarrier_arrive_expect_tx(mbar, bytes);

    uint32_t c0 = coord0;
    uint32_t c1 = coord1;
    uint32_t c2 = coord2;

    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];\n"
        :
        : "l"(smem_addr),
          "l"(desc),
          "r"(c0),
          "r"(c1),
          "r"(c2),
          "l"(mbar_addr)
    );
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
    auto align_up = [] __device__(size_t x, size_t align) {
        return (x + align - 1) & ~(align - 1);
    };

    size_t offset = 0;
    offset = align_up(offset, 16);
    uint64_t* mbar_a = reinterpret_cast<uint64_t*>(smem + offset);
    offset += StageCount * sizeof(uint64_t);
    offset = align_up(offset, 16);
    uint64_t* mbar_b = reinterpret_cast<uint64_t*>(smem + offset);
    offset += sizeof(uint64_t);
    offset = align_up(offset, 16);
    uint64_t* mbar_sfb = reinterpret_cast<uint64_t*>(smem + offset);
    offset += sizeof(uint64_t);
    offset = align_up(offset, 16);

    uint8_t* smem_base = smem;
    uint8_t* b_packed_smem = smem_base + offset;
    offset += K_packed;
    offset = align_up(offset, 16);

    uint8_t* sfb_smem = smem_base + offset;
    offset += K_scales;
    offset = align_up(offset, 16);

    offset = align_up(offset, 16);
    half* b_vec_smem = reinterpret_cast<half*>(smem_base + offset);
    offset += static_cast<size_t>(K) * sizeof(half);
    offset = align_up(offset, 16);

    uint8_t* a_packed_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        a_packed_stage[s] = smem_base + offset;
        offset += TileM * TileKPacked;
        offset = align_up(offset, 128);
    }

    uint8_t* sfa_stage[StageCount];
    for (int s = 0; s < StageCount; ++s) {
        sfa_stage[s] = smem_base + offset;
        offset += TileM * TileScaleCount;
        offset = align_up(offset, 16);
    }

    half* a_f16_smem = reinterpret_cast<half*>(smem_base + offset);
    offset += static_cast<size_t>(TileM) * a_stride * sizeof(half);
    offset = align_up(offset, 64);

    half* b_tile_smem = reinterpret_cast<half*>(smem_base + offset);
    offset += static_cast<size_t>(TileK) * 8 * sizeof(half);
    offset = align_up(offset, 16);

    half* a_scale_smem = reinterpret_cast<half*>(smem_base + offset);
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
            mbarrier_init(&mbar_a[s]);
        }
    }
#endif
    __syncthreads();

    // Robust fallback: Always use software path for B and SFB
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

    // Decode B vector
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
                if (L == 1) {
                    // For rank=2: coordinates are (m_tile, k_tile_coord)
                    uint32_t c0 = static_cast<uint32_t>(m_tile);
                    uint32_t c1 = static_cast<uint32_t>(k_tile_base >> 1);
                    tma_load_2d(
                        a_packed_stage[stage],
                        desc_A,
                        c0,
                        c1,
                        static_cast<uint32_t>(TileM * TileKPacked),
                        &mbar_a[stage]
                    );
                } else {
                    // For rank=3: coordinates are (batch, m_tile, k_tile_coord)
                    uint32_t c0 = static_cast<uint32_t>(batch);
                    uint32_t c1 = static_cast<uint32_t>(m_tile);
                    uint32_t c2 = static_cast<uint32_t>(k_tile_base >> 1);
                    tma_load_3d(
                        a_packed_stage[stage],
                        desc_A,
                        c0,
                        c1,
                        c2,
                        static_cast<uint32_t>(TileM * TileKPacked),
                        &mbar_a[stage]
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
                bool valid_row = row < tile_rows;
                bool full = (col + 16) <= TileKPacked;
                bool valid_col = (k_tile_base + col * 2) < K;
                cp_async_16b(dst, src, valid_row && full && valid_col);
                if (valid_row && !full && col < TileKPacked && valid_col) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        int g = col + i;
                        if (g < TileKPacked && (k_tile_base + g * 2) < K) {
                            dst[i] = src[i];
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

    int k_tiles = (K + TileK - 1) / TileK;
    for (int tile_idx = 0; tile_idx < k_tiles; ++tile_idx) {
        int k_tile = tile_idx * TileK;
        int stage = tile_idx % StageCount;
        int next_stage = (stage + 2) % StageCount;
        int next_k = k_tile + 2 * TileK;

        if (next_k < K) {
            prefetch_tile(next_stage, next_k);
        }

        if (use_tma_a) {
#if __CUDA_ARCH__ >= 900
            mbarrier_wait_parity(&mbar_a[stage], stage_phase_smem[stage]);
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
                int k_base = k_tile + col_packed * 2;
                uint8_t packed = a_stage[row * TileKPacked + col_packed];
                half scale_h = a_scale_smem[row * TileScaleCount + (col_packed >> 3)];
                half v0 = __float2half(0.0f);
                half v1 = __float2half(0.0f);
                if (row < tile_rows && k_base < K) {
                    v0 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);
                }
                if (row < tile_rows && (k_base + 1) < K) {
                    v1 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);
                }
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
                uint64_t a_base = cvta_to_shared_u64(a_tile_ptr);
                int a_row = lane_id & 15;
                int a_col_block = (lane_id >> 4) & 0x1;
                uint64_t a_addr = a_base + (static_cast<uint64_t>(a_row) * a_stride + a_col_block * 8) * sizeof(half);
                unsigned a0, a1, a2, a3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [%4];\n"
                    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                    : "l"(a_addr)
                );

                const half* b_tile_ptr = b_tile_smem + kk * 8;
                uint64_t b_base = cvta_to_shared_u64(b_tile_ptr);
                int b_row = lane_id & 15;
                uint64_t b_addr = b_base + static_cast<uint64_t>(b_row) * 8 * sizeof(half);
                unsigned b0, b1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 { %0, %1 }, [%2];\n"
                    : "=r"(b0), "=r"(b1)
                    : "l"(b_addr)
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

    // TMA hardware constraint: box dimensions must be <= 256 elements per dimension
    constexpr int kTMABoxLimit = 256;
    static_assert(kTileM <= kTMABoxLimit, "kTileM exceeds TMA box limit of 256");
    static_assert(kTileKPacked <= kTMABoxLimit, "kTileKPacked exceeds TMA box limit of 256");

    auto align_up = [](size_t x, size_t align) {
        return (x + align - 1) & ~(align - 1);
    };

    size_t shared_bytes = 0;
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += kStageCount * sizeof(uint64_t);
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += 2 * sizeof(uint64_t);
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 2;
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 16;
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) * sizeof(__half);
    shared_bytes = align_up(shared_bytes, 16);
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileKPacked;
        shared_bytes = align_up(shared_bytes, 128);
    }
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount;
        shared_bytes = align_up(shared_bytes, 16);
    }
    shared_bytes += static_cast<size_t>(kTileM) * kAStride * sizeof(__half);
    shared_bytes = align_up(shared_bytes, 64);
    shared_bytes += static_cast<size_t>(kTileK) * 8 * sizeof(__half);
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount * sizeof(__half);

    auto check_cuda = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    };

    const uint64_t K_packed = static_cast<uint64_t>(K / 2);
    const uint64_t K_scales = static_cast<uint64_t>(K / 16);

    // 1. Use heap-aligned allocation for CUtensorMap descriptor
    std::vector<uint8_t> map_buf(sizeof(CUtensorMap) + 64);
    void* raw_ptr = map_buf.data();
    size_t space = map_buf.size();
    void* aligned_ptr = std::align(64, sizeof(CUtensorMap), raw_ptr, space);
    if (!aligned_ptr) throw std::runtime_error("Failed to align CUtensorMap");

    // Zero-initialize the memory to prevent garbage values
    std::memset(aligned_ptr, 0, sizeof(CUtensorMap));
    CUtensorMap* map_A_ptr = reinterpret_cast<CUtensorMap*>(aligned_ptr);

    CUtensorMap* d_map_A = nullptr;
    CUtensorMap* d_map_B = nullptr;
    CUtensorMap* d_map_SFB = nullptr;
    bool tma_ok = true;

    // 2. Ensure tensor's device pointer is 64-byte aligned
    if ((uintptr_t)A_ptr % 16 != 0) {
        printf("WARNING: A_ptr is not 16-byte aligned: %p\n", A_ptr);
        tma_ok = false;
    }

    auto encode_tma = [&](CUtensorMap* out,
                          CUtensorMapDataType type,
                          cuuint32_t rank,
                          const void* base,
                          const cuuint64_t* dims,
                          const cuuint64_t* globalStrides,
                          const cuuint32_t* box) {
        return cuTensorMapEncodeTiled(
            out,
            type,
            rank,
            const_cast<void*>(base),
            dims,
            globalStrides,
            box,
            nullptr,  // elementStrides
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

        if (L == 1) {
            // Use 2D descriptor when no batching (L=1)
            // TMA doesn't handle degenerate dimensions well
            cuuint64_t dims_A[2] = {
                static_cast<cuuint64_t>(M),
                static_cast<cuuint64_t>(K_packed)
            };

            cuuint32_t box_A[2] = {box_m, box_k};

            printf("TMA Debug: Using RANK=2 for L=1\n");
            printf("TMA Debug: dims = [M=%llu, K_packed=%llu], box = [%u, %u]\n",
                   (unsigned long long)dims_A[0], (unsigned long long)dims_A[1],
                   box_A[0], box_A[1]);

            // Validate box dimensions
            if (box_A[0] > 256 || box_A[1] > 256) {
                printf("ERROR: TMA box dimension exceeds 256 limit! box=[%u, %u]\n",
                       box_A[0], box_A[1]);
                tma_ok = false;
            }

            if (tma_ok) {
                // For contiguous 2D row-major, pass nullptr for globalStrides
                CUresult resA = encode_tma(map_A_ptr,
                                           CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                           2,  // rank=2 for 2D access
                                           A_ptr,
                                           dims_A,
                                           nullptr,  // nullptr for contiguous row-major layout
                                           box_A);

                printf("TMA Encode A Result: %d\n", (int)resA);
                if (resA != CUDA_SUCCESS) {
                    const char* err_str = nullptr;
                    cuGetErrorString(resA, &err_str);
                    printf("TMA Encode A failed: %s\n", err_str ? err_str : "unknown error");
                    tma_ok = false;
                }
            }
        }
        else {
            // Use 3D descriptor when batching (L > 1)
            cuuint64_t dims_A[3] = {
                static_cast<cuuint64_t>(L),
                static_cast<cuuint64_t>(M),
                static_cast<cuuint64_t>(K_packed)
            };

            cuuint32_t box_A[3] = {1u, box_m, box_k};

            cuuint64_t strides_A[2] = {
                static_cast<cuuint64_t>(M) * K_packed,
                static_cast<cuuint64_t>(K_packed)
            };

            printf("TMA Debug: Using RANK=3 for L=%lld\n", (long long)L);
            printf("TMA Debug: dims = [L=%llu, M=%llu, K_packed=%llu], box = [%u, %u, %u]\n",
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
    dim3 grid(num_blocks, static_cast<int>(L));
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
                "-DNDEBUG",
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

    # Reinterpret as raw bytes
    a_bytes = a.view(torch.uint8)
    b_bytes = b.view(torch.uint8)

    # Debug: Print tensor shape and stride for TMA verification
    print(f"TMA Debug (Python): a shape={a.shape}, stride={a.stride()}")
    print(f"TMA Debug (Python): a_bytes shape={a_bytes.shape}, stride={a_bytes.stride()}")

    # Scale factors: prefer pre-permuted tensors if provided
    # sfa_permuted should already be [L, M, K_scales], and sfb_permuted [L, 128, K_scales]
    # Use these directly when available to avoid re-permuting on every call
    # NOTE: sfa_permuted from reference.py is actually [32, 4, rest_m, 4, rest_k, L] which is NOT [L, M, K_scales]
    # So we'll always use the fallback path for now
    if sfa_permuted is not None and list(sfa_permuted.shape) == [L, M, K_scales]:
        sfa_bytes = sfa_permuted.contiguous().cuda().view(torch.uint8)
    else:
        # Fall back to permuting sfa_ref_cpu on the fly
        sfa_bytes = (
            sfa_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda().view(torch.uint8)
        )
    if sfb_permuted is not None and list(sfb_permuted.shape) == [L, 128, K_scales]:
        sfb_bytes = sfb_permuted.contiguous().cuda().view(torch.uint8)
    else:
        sfb_bytes = (
            sfb_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda().view(torch.uint8)
        )

    # Launch SM100 tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()  # [M, 1, L]

    return c
