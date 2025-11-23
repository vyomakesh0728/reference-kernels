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

__device__ __forceinline__ void tma_load_1d(void* dst,
                                            const CUtensorMap* desc,
                                            uint32_t coord0,
                                            uint32_t bytes,
                                            uint64_t* mbar) {
    uint64_t smem_addr = cvta_to_shared_u64(dst);
    uint64_t mbar_addr = cvta_to_shared_u64(mbar);

    mbarrier_arrive_expect_tx(mbar, bytes);

    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2}], [%3];\n"
        :
        : "l"(smem_addr),
          "l"(desc),
          "r"(coord0),
          "l"(mbar_addr)
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
    const bool use_tma = (desc_A != nullptr) && (desc_B != nullptr);
#else
    const bool use_tma = false;
#endif

    // Phase tracking in shared memory so all threads see same values
    __shared__ uint32_t stage_phase_smem[StageCount];
    __shared__ uint32_t b_phase_smem;
    __shared__ uint32_t sfb_phase_smem;

    if (tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            stage_phase_smem[s] = 0;
        }
        b_phase_smem = 0;
        sfb_phase_smem = 0;
    }

#if __CUDA_ARCH__ >= 900
    if (use_tma && tid == 0) {
        for (int s = 0; s < StageCount; ++s) {
            mbarrier_init(&mbar_a[s]);
        }
        mbarrier_init(mbar_b);
        mbarrier_init(mbar_sfb);
    }
#endif
    __syncthreads();

    // Cache B and SFB once using TMA (fallback cp.async when unavailable)
    if (use_tma) {
        if (warp_id == 0 && lane_id == 0) {
            // Coordinates match dims [K_packed, L*128]
            uint32_t coord0 = 0u;
            uint32_t coord1 = static_cast<uint32_t>(batch * 128u);
            tma_load_2d(b_packed_smem, desc_B, coord0, coord1, static_cast<uint32_t>(K_packed), mbar_b);
        }
        if (warp_id == 0 && lane_id == 0) {
            const CUtensorMap* desc_sfb_local = desc_SFB ? desc_SFB : desc_B;
            // Coordinates match dims [K_scales, L*128]
            uint32_t coord0 = 0u;
            uint32_t coord1 = static_cast<uint32_t>(batch * 128u);
            tma_load_2d(sfb_smem, desc_sfb_local, coord0, coord1, static_cast<uint32_t>(K_scales), mbar_sfb);
        }
#if __CUDA_ARCH__ >= 900
        // Wait with current phase, then flip for next use
        mbarrier_wait_parity(mbar_b, b_phase_smem);
        mbarrier_wait_parity(mbar_sfb, sfb_phase_smem);
        __syncthreads();
        if (tid == 0) {
            b_phase_smem ^= 1;
            sfb_phase_smem ^= 1;
        }
        __syncthreads();
#endif
    } else {
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

    // Decode B vector once: FP4 -> FP16 with FP8 block scale
    for (int idx = tid; idx < K_packed; idx += Threads) {
        int k_base = idx * 2;
        int scale_idx = idx >> 3;
        uint8_t packed = b_packed_smem[idx];
        half scale_h = __float2half(0.0f);
        if (scale_idx < K_scales) {
            scale_h = __float2half(decode_fp8_e4m3(sfb_smem[scale_idx]));
        }
        // FP4 nibble order: high nibble is first element (even), low nibble is second (odd)
        half v0 = __hmul(decode_fp4_e2m1((packed >> 4) & 0x0F), scale_h);
        half v1 = __hmul(decode_fp4_e2m1(packed & 0x0F), scale_h);
        if (k_base < K) b_vec_smem[k_base] = v0;
        if (k_base + 1 < K) b_vec_smem[k_base + 1] = v1;
    }
    __syncthreads();

    auto prefetch_tile = [&](int stage, int k_tile_base) {
        if (use_tma) {
            if (warp_id == 0 && lane_id == 0) {
                // Coordinates match dims [K_packed, M, L]
                uint32_t c0 = static_cast<uint32_t>(k_tile_base >> 1);
                uint32_t c1 = static_cast<uint32_t>(m_tile);
                uint32_t c2 = static_cast<uint32_t>(batch);
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
        } else if (is_producer) {
            size_t bytes = static_cast<size_t>(TileM) * TileKPacked;
            int segments = static_cast<int>((bytes + 15) / 16);
            const uint8_t* src_base = A_batch + static_cast<size_t>(m_tile) * K_packed + (k_tile_base >> 1);
            uint8_t* dst_base = a_packed_stage[stage];
            for (int idx = tid; idx < segments; idx += Threads) {
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

        // Prefetch SFA scales for this tile (producer warp 1)
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

        if (use_tma) {
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

        // ALL threads decode scales (not just consumers) to cover all indices
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

        // ALL threads decode A tile (not just consumers) to cover all indices
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
                // FP4 nibble order: high nibble is first element (even), low nibble is second (odd)
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
        // For m16n8k16 MMA, thread t holds rows (t % 8) and (t % 8) + 8
        // Threads 0-7 have column 0 results, which we need for GEMV
        if (lane_id < 8) {
            int row0 = (warp_id - 2) * 16 + lane_id;
            int row1 = row0 + 8;
            int global_row0 = m_tile + row0;
            int global_row1 = m_tile + row1;
            if (row0 < tile_rows && global_row0 < M) {
                D_batch[global_row0] = __float2half(c_frag_0);
            }
            if (row1 < tile_rows && global_row1 < M) {
                D_batch[global_row1] = __float2half(c_frag_2);
            }
        }
    }
#endif
}


// ============================================================================
// Launcher
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

    constexpr int kTileM = 128;
    constexpr int kTileK = 128;
    constexpr int kThreads = 256;
    constexpr int kTileKPacked = kTileK / 2;
    constexpr int kTileScaleCount = kTileK / 16;
    constexpr int kStageCount = 3;
    constexpr int kAStride = kTileK + 8;

    auto align_up = [](size_t x, size_t align) {
        return (x + align - 1) & ~(align - 1);
    };

    size_t shared_bytes = 0;
    shared_bytes = align_up(shared_bytes, 16);                  // mbar alignment
    shared_bytes += kStageCount * sizeof(uint64_t);             // mbarriers for A stages
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += 2 * sizeof(uint64_t);                       // mbar for B + SFB
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 2;                 // B packed
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) / 16;                // SFB packed
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(K) * sizeof(__half);    // B decoded
    shared_bytes = align_up(shared_bytes, 16);
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileKPacked; // A packed stage
        shared_bytes = align_up(shared_bytes, 128);
    }
    for (int s = 0; s < kStageCount; ++s) {
        shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount; // SFA stage
        shared_bytes = align_up(shared_bytes, 16);
    }
    shared_bytes += static_cast<size_t>(kTileM) * kAStride * sizeof(__half); // decoded A
    shared_bytes = align_up(shared_bytes, 64);
    shared_bytes += static_cast<size_t>(kTileK) * 8 * sizeof(__half); // B tile
    shared_bytes = align_up(shared_bytes, 16);
    shared_bytes += static_cast<size_t>(kTileM) * kTileScaleCount * sizeof(__half); // decoded scales

    auto check_cuda = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    };

    const uint64_t K_packed = static_cast<uint64_t>(K / 2);
    const uint64_t K_scales = static_cast<uint64_t>(K / 16);

    CUtensorMap map_A{};
    CUtensorMap map_B{};
    CUtensorMap map_SFB{};
    CUtensorMap* d_map_A = nullptr;
    CUtensorMap* d_map_B = nullptr;
    CUtensorMap* d_map_SFB = nullptr;
    bool tma_ok = true;

    auto encode_tma = [&](CUtensorMap& out,
                          CUtensorMapDataType type,
                          cuuint32_t rank,
                          const void* base,
                          const cuuint64_t* dims,
                          const cuuint64_t* strides,
                          const cuuint32_t* box) {
        cuuint32_t elem_strides[4] = {1, 1, 1, 1};
        return cuTensorMapEncodeTiled(
            &out,
            type,
            rank,
            const_cast<void*>(base),
            dims,
            strides,
            box,
            elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
    };


    if (tma_ok) {
        // TMA strides array should have rank-1 elements (innermost stride is implicit)
        cuuint64_t dims_A[3] = {static_cast<cuuint64_t>(K_packed), static_cast<cuuint64_t>(M), static_cast<cuuint64_t>(L)};
        cuuint64_t strides_A[2] = {static_cast<cuuint64_t>(K_packed), static_cast<cuuint64_t>(M) * static_cast<cuuint64_t>(K_packed)};
        cuuint32_t box_A[3] = {static_cast<cuuint32_t>(kTileKPacked), static_cast<cuuint32_t>(kTileM), 1u};
        CUresult resA = encode_tma(map_A, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, A_ptr, dims_A, strides_A, box_A);
        tma_ok = tma_ok && (resA == CUDA_SUCCESS);

        cuuint64_t dims_B[2] = {static_cast<cuuint64_t>(K_packed), static_cast<cuuint64_t>(L) * 128ull};
        cuuint64_t strides_B[1] = {static_cast<cuuint64_t>(K_packed)};
        // GEMV only needs row 0 of each batch (N=1), not all 128 rows
        cuuint32_t box_B[2] = {static_cast<cuuint32_t>(K_packed), 1u};
        CUresult resB = encode_tma(map_B, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, B_ptr, dims_B, strides_B, box_B);
        tma_ok = tma_ok && (resB == CUDA_SUCCESS);

        cuuint64_t dims_SFB[2] = {static_cast<cuuint64_t>(K_scales), static_cast<cuuint64_t>(L) * 128ull};
        cuuint64_t strides_SFB[1] = {static_cast<cuuint64_t>(K_scales)};
        // GEMV only needs row 0 of each batch (N=1), not all 128 rows
        cuuint32_t box_SFB[2] = {static_cast<cuuint32_t>(K_scales), 1u};
        CUresult resSFB = encode_tma(map_SFB, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, SFB_ptr, dims_SFB, strides_SFB, box_SFB);
        tma_ok = tma_ok && (resSFB == CUDA_SUCCESS);
    }

    if (tma_ok) {
        check_cuda(cudaMalloc(&d_map_A, sizeof(CUtensorMap)), "cudaMalloc d_map_A");
        check_cuda(cudaMalloc(&d_map_B, sizeof(CUtensorMap)), "cudaMalloc d_map_B");
        check_cuda(cudaMalloc(&d_map_SFB, sizeof(CUtensorMap)), "cudaMalloc d_map_SFB");
        check_cuda(cudaMemcpy(d_map_A, &map_A, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_A");
        check_cuda(cudaMemcpy(d_map_B, &map_B, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_B");
        check_cuda(cudaMemcpy(d_map_SFB, &map_SFB, sizeof(CUtensorMap), cudaMemcpyHostToDevice), "cudaMemcpy d_map_SFB");
    }

    // Enable >48KB dynamic shared memory on Hopper/Blackwell.
    cudaFuncAttributes attr;
    cudaError_t attr_err = cudaFuncGetAttributes(
        &attr,
        fp4_gemv_streaming<kTileM, kTileK, kThreads>
    );
    if (attr_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFuncGetAttributes failed: ") +
            cudaGetErrorString(attr_err)
        );
    }

    cudaError_t set_err = cudaFuncSetAttribute(
        fp4_gemv_streaming<kTileM, kTileK, kThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)
    );
    if (set_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaFuncSetAttribute failed: ") +
            cudaGetErrorString(set_err)
        );
    }

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

    # ========== DEBUG PRINTS ==========
    print("\n" + "="*60)
    print("DEBUG: classic.py custom_kernel")
    print("="*60)

    # Input shapes from generate_input
    print(f"\nInput tensor shapes (from generate_input):")
    print(f"  a (original): {a.shape} dtype={a.dtype}")
    print(f"  b (original): {b.shape} dtype={b.dtype}")
    print(f"  sfa_ref_cpu: {sfa_ref_cpu.shape} dtype={sfa_ref_cpu.dtype}")
    print(f"  sfb_ref_cpu: {sfb_ref_cpu.shape} dtype={sfb_ref_cpu.dtype}")
    print(f"  sfa_permuted: {sfa_permuted.shape if sfa_permuted is not None else None}")
    print(f"  sfb_permuted: {sfb_permuted.shape if sfb_permuted is not None else None}")
    print(f"  c (original): {c.shape} dtype={c.dtype}")
    print(f"  M={M}, K={K}, L={L}, K_scales={K_scales}")

    # Permute to [L, M, K/2] layout
    # CRITICAL: Clone first to avoid tensor aliasing/reuse between test calls
    a = a.clone().permute(2, 0, 1).contiguous().cuda()
    b = b.clone().permute(2, 0, 1).contiguous().cuda()
    c = c.clone().permute(2, 0, 1).contiguous().cuda()

    print(f"\nAfter permute(2, 0, 1):")
    print(f"  a: {a.shape} (expected [{L}, {M}, {K//2}])")
    print(f"  b: {b.shape} (expected [{L}, 128, {K//2}])")
    print(f"  c: {c.shape} (expected [{L}, {M}, 1])")

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

    # Scale factors: prefer pre-permuted tensors if provided
    # sfa_permuted should already be [L, M, K_scales], and sfb_permuted [L, 128, K_scales]
    # Use these directly when available to avoid re-permuting on every call
    # NOTE: sfa_permuted from reference.py is actually [32, 4, rest_m, 4, rest_k, L] which is NOT [L, M, K_scales]
    # So we'll always use the fallback path for now
    if sfa_permuted is not None and list(sfa_permuted.shape) == [L, M, K_scales]:
        sfa_bytes = sfa_permuted.contiguous().cuda().view(torch.uint8)
        print(f"\nUsing pre-permuted sfa_permuted")
    else:
        # Fall back to permuting sfa_ref_cpu on the fly
        sfa_bytes = (
            sfa_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda().view(torch.uint8)
        )
        print(f"\nUsing sfa_ref_cpu.permute(2, 0, 1): {sfa_ref_cpu.shape} -> {(L, M, K_scales)}")
    if sfb_permuted is not None and list(sfb_permuted.shape) == [L, 128, K_scales]:
        sfb_bytes = sfb_permuted.contiguous().cuda().view(torch.uint8)
        print(f"Using pre-permuted sfb_permuted")
    else:
        sfb_bytes = (
            sfb_ref_cpu.clone().permute(2, 0, 1).contiguous().cuda().view(torch.uint8)
        )
        print(f"Using sfb_ref_cpu.permute(2, 0, 1): {sfb_ref_cpu.shape} -> {(L, 128, K_scales)}")

    print(f"\nScale bytes shapes:")
    print(f"  sfa_bytes: {sfa_bytes.shape}")
    print(f"  sfb_bytes: {sfb_bytes.shape}")

    # Print first few raw values for debugging
    print(f"\nFirst 8 bytes of tensors (batch 0):")
    print(f"  a_bytes[0,0,:8]: {a_bytes[0, 0, :min(8, a_bytes.shape[2])].tolist()}")
    print(f"  b_bytes[0,0,:8]: {b_bytes[0, 0, :min(8, b_bytes.shape[2])].tolist()}")
    print(f"  sfa_bytes[0,0,:8]: {sfa_bytes[0, 0, :min(8, sfa_bytes.shape[2])].tolist()}")
    print(f"  sfb_bytes[0,0,:8]: {sfb_bytes[0, 0, :min(8, sfb_bytes.shape[2])].tolist()}")

    # Decode first FP4 value manually to verify nibble order
    if a_bytes.numel() > 0:
        packed_val = a_bytes[0, 0, 0].item()
        hi_nibble = (packed_val >> 4) & 0x0F
        lo_nibble = packed_val & 0x0F
        fp4_lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        print(f"\nFP4 decoding check (byte={packed_val:#04x}):")
        print(f"  High nibble (bits 7-4) = {hi_nibble:#x} -> FP4 value = {fp4_lut[hi_nibble]} (element 0)")
        print(f"  Low nibble (bits 3-0)  = {lo_nibble:#x} -> FP4 value = {fp4_lut[lo_nibble]} (element 1)")

    # Launch SM100 tensor core kernel
    mod = get_module()
    mod.launch_fp4_gemv_optimized(a_bytes, b_bytes, sfa_bytes, sfb_bytes, c, M, K, L)

    # Print output before permute back
    print(f"\nOutput c before permute back: {c.shape}")
    print(f"  c[0,:5,0] (first 5 rows of batch 0): {c[0, :min(5, c.shape[1]), 0].tolist()}")

    # Permute output back
    c = c.permute(1, 2, 0).contiguous()  # [M, 1, L]

    print(f"\nOutput c after permute: {c.shape}")
    print(f"  c[:5,0,0] (first 5 elements): {c[:min(5, c.shape[0]), 0, 0].tolist()}")
    print("="*60 + "\n")

    return c
