#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
# Candidate Card
# shape: exact m64 on top of the t18 wave-helper winner
# regime_tag: exact_m64_shuffled_scale_delete_t19
# deleted_cost_center: exact m64 per-call b_scale_decode materialization
# expected_upside_source: the t18 profile shows exact m64 still spends 45.5% of self CUDA time in mxfp4_unshuffle_b_scale_kernel before the raw-B MFMA body runs
# why_larger_than_noise: this removes a named whole bucket from the slowest exact shape in the current portfolio without reopening fused-A or touching the validated wave helper
# touched_symbols_or_regions: exact m64 raw-B kernel scale contract; exact m64 direct-entry wrapper
# forbidden_edits: do not reopen fused-A compute-path quantization; do not add LDS staging to A-pack; do not change m4/m16/m32/m256 paths in the same branch
# success_gate: geomean materially improves from the 14.5097 us t18 frontier
# AGENT_LOOP_META: {"generator": {"kind": "manual_phase2"}, "gpu": "MI355X", "leaderboard": "amd-mxfp4-mm", "policy_profile": {"family": "hip_explore", "name": "m32_k512_scaleaddr_v119"}, "problem": "mxfp4_mm"}
import ctypes
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import aiter
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CONFIG = {
    "variant_name": "native_scaled_exact_shape_m64_shscale_t19",
    "family": "hip_explore",
    "strategy": "hip_reference_oracle",
    "ARCH": "gfx950",
    "REFERENCE_INPUTS": True,
    "NAIVE_KERNEL": False,
    "TILE_M": 16,
    "TILE_N": 32,
    "TILE_K": 64,
}
SCALE_GROUP = 32
USE_FP4_WAVE_PACK = int(os.environ.get("MXFP4_USE_WAVE_PACK", "1"))
DIRECT_M32_EXPERIMENT = "REAL_A_REAL_SCALES_COMPILED_A_PACK_M8_V34"
FIXED_ADJUSTMENT_RULES: dict[float, tuple[str, float, float]] = {
    -6.0: ("gt", -5.03125, -4.0),
    -3.0: ("gt", -2.515625, -2.0),
    -1.5: ("gt", -1.2578125, -1.0),
    -0.5: ("gt", -0.251953125, -0.0),
    0.5: ("le", 0.25, 0.0),
    1.5: ("le", 1.25, 1.0),
    3.0: ("le", 2.5, 2.0),
    6.0: ("le", 5.0, 4.0),
}

CPP_WRAPPER = """
void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m16(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m16_dense(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m16_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry_public_k512(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m8_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m4m8_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor a_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m4m8_direct_entry_owned(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m32(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m32_only(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m32_rawb_only(torch::Tensor a_packed, torch::Tensor b_q, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m64_only(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m64_rawb_only(torch::Tensor a_packed, torch::Tensor b_q, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m256_only(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m256_rawb_only(torch::Tensor a_packed, torch::Tensor b_q, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m32_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m64_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_mm_hip_mfma_scale_exact_m256_direct_entry(torch::Tensor a, torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor c);
void mxfp4_pack_a_fixed(torch::Tensor a, torch::Tensor a_packed, torch::Tensor a_scale);
void mxfp4_repack_b_packed(torch::Tensor b_q, torch::Tensor b_packed);
void mxfp4_pack_b_m32_direct(torch::Tensor b_q, torch::Tensor b_direct);
void mxfp4_pack_b_m32_direct_with_scale(torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor b_direct, torch::Tensor b_scale);
void mxfp4_unshuffle_b_scale(torch::Tensor b_scale_sh, torch::Tensor b_scale);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <cstdio>
#include "opus/opus.hpp"

constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int TILE_K = 64;
constexpr int USE_FP4_WAVE_PACK = 1;

__global__ void canary_store_kernel(__hip_bfloat16* c, int numel) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && numel > 0) {
        c[0] = static_cast<__hip_bfloat16>(0.0f);
    }
}

inline void torch_check_hip_ok(hipError_t err, const char* phase) {
    TORCH_CHECK(
        err == hipSuccess,
        phase,
        " err=",
        static_cast<int>(err),
        " name=",
        hipGetErrorName(err),
        " msg=",
        hipGetErrorString(err)
    );
}

__global__ void mxfp4_mm_kernel(
    const float* a,
    const float* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int row = blockIdx.y * TILE_M + local_y;
    const int col = blockIdx.x * TILE_N + local_x;

    double acc = 0.0;
    __shared__ float a_tile[TILE_M][TILE_K + 1];
    __shared__ float b_tile[TILE_N][TILE_K + 1];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        if (local_x < TILE_K / 4) {
            const int k_vec = local_x * 4;
            const int global_k = tile_k + k_vec;
            if (row < m && global_k + 3 < k) {
                const float4 vec = *reinterpret_cast<const float4*>(a + row * k + global_k);
                a_tile[local_y][k_vec + 0] = vec.x;
                a_tile[local_y][k_vec + 1] = vec.y;
                a_tile[local_y][k_vec + 2] = vec.z;
                a_tile[local_y][k_vec + 3] = vec.w;
            } else {
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    const int kk = global_k + lane;
                    a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : 0.0f;
                }
            }
        }

        {
            const int k_vec = local_y * 4;
            const int global_k = tile_k + k_vec;
            if (col < n && global_k + 3 < k) {
                const float4 vec = *reinterpret_cast<const float4*>(b + col * k + global_k);
                b_tile[local_x][k_vec + 0] = vec.x;
                b_tile[local_x][k_vec + 1] = vec.y;
                b_tile[local_x][k_vec + 2] = vec.z;
                b_tile[local_x][k_vec + 3] = vec.w;
            } else {
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    const int kk = global_k + lane;
                    b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : 0.0f;
                }
            }
        }

        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll 4
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += static_cast<double>(a_tile[local_y][kk]) * static_cast<double>(b_tile[local_x][kk]);
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = static_cast<__hip_bfloat16>(static_cast<float>(acc));
    }
}



using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8
{
    _B16x4 xy[2];
} _B16x8;

__device__ __forceinline__ floatx4 gcn_mfma16x16x32_bf16_instr(
    const _B16x8& inpA,
    const _B16x8& inpB,
    const floatx4& inpC
) {
    bit16x8 tmpA = __builtin_shufflevector(inpA.xy[0], inpA.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    bit16x8 tmpB = __builtin_shufflevector(inpB.xy[0], inpB.xy[1], 0, 1, 2, 3, 4, 5, 6, 7);
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(tmpA, tmpB, inpC, 0, 0, 0);
}

__device__ __forceinline__ floatx4 gcn_mfma16x16x16bf16_1k_instr(
    const _B16x4& inpA,
    const _B16x4& inpB,
    const floatx4& inpC
) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, 0, 0, 0);
}

template <typename input_t>
__global__ void mxfp4_mm_kernel_bf16_scalar(
    const input_t* a,
    const input_t* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int row = blockIdx.y * TILE_M + local_y;
    const int col = blockIdx.x * TILE_N + local_x;

    float acc = 0.0f;
    __shared__ input_t a_tile[TILE_M][TILE_K + 1];
    __shared__ input_t b_tile[TILE_N][TILE_K + 1];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        if (local_x < TILE_K / 4) {
            const int k_vec = local_x * 4;
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int kk = tile_k + k_vec + lane;
                a_tile[local_y][k_vec + lane] = (row < m && kk < k) ? a[row * k + kk] : input_t(0.0f);
            }
        }

        {
            const int k_vec = local_y * 4;
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int kk = tile_k + k_vec + lane;
                b_tile[local_x][k_vec + lane] = (col < n && kk < k) ? b[col * k + kk] : input_t(0.0f);
            }
        }

        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll 4
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += static_cast<float>(a_tile[local_y][kk]) * static_cast<float>(b_tile[local_x][kk]);
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = static_cast<__hip_bfloat16>(acc);
    }
}

__launch_bounds__(64)
__global__ void mxfp4_mm_kernel_mfma_medium(
    const __hip_bfloat16* a,
    const __hip_bfloat16* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 16;

    const int lane = threadIdx.x;
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane_col = lane & 15;
    const int lane_group = lane >> 4;

    const auto* a_bits = reinterpret_cast<uint16_t const*>(a);
    const auto* b_bits = reinterpret_cast<uint16_t const*>(b);
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        _B16x4 a_reg{};
        _B16x4 b_reg{};
        #pragma unroll
        for (int lane_i = 0; lane_i < 4; ++lane_i) {
            const int kk = tile_k + lane_group * 4 + lane_i;
            const int a_row = tile_row + lane_col;
            const int b_col = tile_col + lane_col;
            a_reg[lane_i] = (a_row < m && kk < k) ? a_bits[a_row * k + kk] : uint16_t{0};
            b_reg[lane_i] = (b_col < n && kk < k) ? b_bits[b_col * k + kk] : uint16_t{0};
        }
        acc = gcn_mfma16x16x16bf16_1k_instr(a_reg, b_reg, acc);
    }

    const int out_col = tile_col + lane_col;
    const int out_row_base = tile_row + lane_group * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < m && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_medium(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));

    if ((m % 16 == 0) && (n % 16 == 0) && (k % 32 == 0) && m <= 128) {
        dim3 block(64);
        dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);
        hipLaunchKernelGGL(
            mxfp4_mm_kernel_mfma_medium,
            grid,
            block,
            0,
            0,
            reinterpret_cast<__hip_bfloat16 const*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16 const*>(b.data_ptr<at::BFloat16>()),
            reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
            m,
            n,
            k
        );
        return;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_bf16_scalar<__hip_bfloat16>,
        grid,
        block,
        0,
        0,
        reinterpret_cast<__hip_bfloat16 const*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16 const*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k
    );
}



using i32x8_t = int __attribute__((ext_vector_type(8)));
using i32x4_t = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ unsigned char fp4_extract(unsigned char packed, int idx) {
    return (idx == 0) ? (packed & 0xFu) : (packed >> 4);
}

__device__ __forceinline__ unsigned char fp4_pack(unsigned char lo, unsigned char hi) {
    return (lo & 0xFu) | ((hi & 0xFu) << 4);
}

__device__ __forceinline__ unsigned char quantize_fp4_scaled(float q) {
    const unsigned int bits = __builtin_bit_cast(unsigned int, q);
    const unsigned int sign = bits & 0x80000000u;
    unsigned int exponent = (bits >> 23) & 0xFFu;
    unsigned int mantissa = bits & 0x7FFFFFu;
    if (exponent < 127u) {
        const unsigned int adjusted = 127u - (exponent + 1u);
        const unsigned int denorm = 0x400000u | (mantissa >> 1);
        mantissa = (adjusted >= 32u) ? 0u : (denorm >> adjusted);
    }
    exponent = max(exponent, 126u) - 126u;
    const unsigned int e2m1 = min(((((exponent << 2) | (mantissa >> 21)) + 1u) >> 1), 0x7u);
    return static_cast<unsigned char>((sign >> 28) | e2m1);
}

__device__ __forceinline__ unsigned char apply_fixed_adjustment(unsigned char nibble, float q) {
    if (nibble == 0x1u && q <= 0.25f) {
        return 0x0u;
    }
    if (nibble == 0x3u && q <= 1.25f) {
        return 0x2u;
    }
    if (nibble == 0x5u && q <= 2.5f) {
        return 0x4u;
    }
    if (nibble == 0x7u && q <= 5.0f) {
        return 0x6u;
    }
    if (nibble == 0x9u && q > -0.251953125f) {
        return 0x8u;
    }
    if (nibble == 0xBu && q > -1.2578125f) {
        return 0xAu;
    }
    if (nibble == 0xDu && q > -2.515625f) {
        return 0xCu;
    }
    if (nibble == 0xFu && q > -5.03125f) {
        return 0xEu;
    }
    return nibble;
}

__global__ void mxfp4_pack_a_fixed_kernel_scalar(
    const __hip_bfloat16* __restrict__ a,
    unsigned char* __restrict__ a_packed,
    uint8_t* __restrict__ a_scale,
    int m,
    int k,
    int packed_stride,
    int scale_stride
) {
    const int row = blockIdx.y;
    const int scale_block = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_scale_blocks = k / 32;
    if (row >= m || scale_block >= num_scale_blocks) {
        return;
    }

    const int col0 = scale_block * 32;
    const __hip_bfloat16* a_row = a + row * k + col0;

    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(static_cast<float>(a_row[i])));
    }

    uint8_t scale_byte = 0;
    float quant_scale = 1.0f;
    if (amax > 0.0f) {
        const unsigned int rounded_bits = (__builtin_bit_cast(unsigned int, amax) + 0x200000u) & 0xFF800000u;
        const unsigned int rounded_exp = (rounded_bits >> 23) & 0xFFu;
        scale_byte = static_cast<uint8_t>(rounded_exp - 2u);
        const int scale_unbiased = static_cast<int>(scale_byte) - 127;
        quant_scale = ldexpf(1.0f, -scale_unbiased);
    }
    a_scale[row * scale_stride + scale_block] = scale_byte;

    unsigned char* packed_row = a_packed + row * packed_stride + scale_block * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float q0 = static_cast<float>(a_row[2 * i + 0]) * quant_scale;
        const float q1 = static_cast<float>(a_row[2 * i + 1]) * quant_scale;
        const unsigned char nib0 = apply_fixed_adjustment(quantize_fp4_scaled(q0), q0);
        const unsigned char nib1 = apply_fixed_adjustment(quantize_fp4_scaled(q1), q1);
        packed_row[i] = fp4_pack(nib0, nib1);
    }
}

__global__ void mxfp4_pack_a_fixed_kernel_wave(
    const __hip_bfloat16* __restrict__ a,
    unsigned char* __restrict__ a_packed,
    uint8_t* __restrict__ a_scale,
    int m,
    int k,
    int packed_stride,
    int scale_stride
) {
    const int row = blockIdx.y;
    const int subgroup = threadIdx.x >> 4;
    const int lane16 = threadIdx.x & 15;
    const int scale_blocks_per_block = blockDim.x >> 4;
    const int scale_block = blockIdx.x * scale_blocks_per_block + subgroup;
    const int num_scale_blocks = k / 32;
    if (row >= m || scale_block >= num_scale_blocks) {
        return;
    }

    const int col0 = scale_block * 32;
    const uint32_t* a_pairs = reinterpret_cast<const uint32_t*>(a + row * k + col0);
    const uint32_t pair_bits = a_pairs[lane16];

    union {
        uint32_t u32;
        __hip_bfloat16 bf16[2];
    } pair;
    pair.u32 = pair_bits;

    const float src0 = static_cast<float>(pair.bf16[0]);
    const float src1 = static_cast<float>(pair.bf16[1]);
    float amax = fmaxf(fabsf(src0), fabsf(src1));
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down(amax, offset, 16));
    }
    amax = __shfl(amax, 0, 16);

    int scale_word = 0;
    float quant_scale = 1.0f;
    if (lane16 == 0 && amax > 0.0f) {
        const unsigned int rounded_bits = (__builtin_bit_cast(unsigned int, amax) + 0x200000u) & 0xFF800000u;
        const unsigned int rounded_exp = (rounded_bits >> 23) & 0xFFu;
        const uint8_t scale_byte = static_cast<uint8_t>(rounded_exp - 2u);
        scale_word = static_cast<int>(scale_byte);
        const int scale_unbiased = static_cast<int>(scale_byte) - 127;
        quant_scale = ldexpf(1.0f, -scale_unbiased);
    }
    scale_word = __shfl(scale_word, 0, 16);
    quant_scale = __shfl(quant_scale, 0, 16);
    const uint8_t scale_byte = static_cast<uint8_t>(scale_word);

    const float q0 = src0 * quant_scale;
    const float q1 = src1 * quant_scale;
    const unsigned char nib0 = apply_fixed_adjustment(quantize_fp4_scaled(q0), q0);
    const unsigned char nib1 = apply_fixed_adjustment(quantize_fp4_scaled(q1), q1);
    const unsigned int packed_byte = static_cast<unsigned int>(fp4_pack(nib0, nib1));

    const int quad_lane = lane16 & ~3;
    const unsigned int b0 = static_cast<unsigned int>(__shfl(static_cast<int>(packed_byte), quad_lane + 0, 16)) & 0xFFu;
    const unsigned int b1 = static_cast<unsigned int>(__shfl(static_cast<int>(packed_byte), quad_lane + 1, 16)) & 0xFFu;
    const unsigned int b2 = static_cast<unsigned int>(__shfl(static_cast<int>(packed_byte), quad_lane + 2, 16)) & 0xFFu;
    const unsigned int b3 = static_cast<unsigned int>(__shfl(static_cast<int>(packed_byte), quad_lane + 3, 16)) & 0xFFu;

    if ((lane16 & 3) == 0) {
        uint32_t* packed_words = reinterpret_cast<uint32_t*>(a_packed + row * packed_stride + scale_block * 16);
        packed_words[lane16 >> 2] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    if (lane16 == 0) {
        a_scale[row * scale_stride + scale_block] = scale_byte;
    }
}

void mxfp4_pack_a_fixed(torch::Tensor a, torch::Tensor a_packed, torch::Tensor a_scale) {
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    TORCH_CHECK((k % 32) == 0, "A K must be a multiple of 32");

    if (USE_FP4_WAVE_PACK != 0) {
        dim3 block(256);
        dim3 grid((k / 32 + (block.x >> 4) - 1) / (block.x >> 4), m);
        hipLaunchKernelGGL(
            mxfp4_pack_a_fixed_kernel_wave,
            grid,
            block,
            0,
            0,
            reinterpret_cast<const __hip_bfloat16*>(a.data_ptr<at::BFloat16>()),
            reinterpret_cast<unsigned char*>(a_packed.data_ptr<uint8_t>()),
            reinterpret_cast<uint8_t*>(a_scale.data_ptr<uint8_t>()),
            m,
            k,
            static_cast<int>(a_packed.size(1)),
            static_cast<int>(a_scale.size(1))
        );
        return;
    }

    dim3 block(128);
    dim3 grid((k / 32 + block.x - 1) / block.x, m);
    hipLaunchKernelGGL(
        mxfp4_pack_a_fixed_kernel_scalar,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const __hip_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned char*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(a_scale.data_ptr<uint8_t>()),
        m,
        k,
        static_cast<int>(a_packed.size(1)),
        static_cast<int>(a_scale.size(1))
    );
}

__global__ void mxfp4_repack_b_packed_kernel(
    const uint8_t* __restrict__ b_q,
    uint8_t* __restrict__ b_packed,
    int n,
    int k_half
) {
    const int k_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_pair = blockIdx.y;
    if (k_idx >= k_half || n_pair >= (n / 2)) {
        return;
    }

    const int even_row = 2 * n_pair;
    const int odd_row = even_row + 1;
    const uint8_t even_byte = b_q[even_row * k_half + k_idx];
    const uint8_t odd_byte = b_q[odd_row * k_half + k_idx];
    const int out_stride = n / 2;
    b_packed[(2 * k_idx + 0) * out_stride + n_pair] = static_cast<uint8_t>((even_byte & 0x0F) | ((odd_byte & 0x0F) << 4));
    b_packed[(2 * k_idx + 1) * out_stride + n_pair] = static_cast<uint8_t>((even_byte >> 4) | (((odd_byte >> 4) & 0x0F) << 4));
}

void mxfp4_repack_b_packed(torch::Tensor b_q, torch::Tensor b_packed) {
    const int n = static_cast<int>(b_q.size(0));
    const int k_half = static_cast<int>(b_q.size(1));
    TORCH_CHECK((n % 2) == 0, "B row count must be even for packed row-major repack");

    dim3 block(128);
    dim3 grid((k_half + block.x - 1) / block.x, n / 2);
    hipLaunchKernelGGL(
        mxfp4_repack_b_packed_kernel,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const uint8_t*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(b_packed.data_ptr<uint8_t>()),
        n,
        k_half
    );
}

__global__ void mxfp4_pack_b_m32_direct_kernel(
    const uint8_t* __restrict__ b_q,
    uint8_t* __restrict__ b_direct,
    int n,
    int k_half
) {
    const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_idx = blockIdx.y;
    if (n_idx >= n || byte_idx >= k_half) {
        return;
    }

    const int k_tile = byte_idx / 32;
    const int inner = byte_idx & 31;
    b_direct[(k_tile * n + n_idx) * 32 + inner] = b_q[n_idx * k_half + byte_idx];
}

void mxfp4_pack_b_m32_direct(torch::Tensor b_q, torch::Tensor b_direct) {
    const int n = static_cast<int>(b_q.size(0));
    const int k_half = static_cast<int>(b_q.size(1));
    TORCH_CHECK((k_half % 32) == 0, "B K/2 must be divisible by 32 for direct m32 B packing");

    dim3 block(128);
    dim3 grid((k_half + block.x - 1) / block.x, n);
    hipLaunchKernelGGL(
        mxfp4_pack_b_m32_direct_kernel,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const uint8_t*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(b_direct.data_ptr<uint8_t>()),
        n,
        k_half
    );
}

__global__ void mxfp4_unshuffle_b_scale_kernel(
    const uint8_t* __restrict__ b_scale_sh,
    uint8_t* __restrict__ b_scale,
    int rows,
    int cols,
    int src_rows,
    int src_cols
) {
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_row >= rows || out_col >= cols) {
        return;
    }

    const int row_block = out_row >> 5;
    const int row_in_block = out_row & 31;
    const int d = row_in_block >> 4;
    const int b = row_in_block & 15;
    const int c0 = out_col >> 3;
    const int c = (out_col & 7) >> 2;
    const int a = out_col & 3;
    const int padded_cols = ((cols + 7) / 8) * 8;
    const int col_blocks = padded_cols / 8;
    const int source_linear = (((((row_block * col_blocks + c0) * 4 + a) * 16 + b) * 2 + c) * 2 + d);
    const int in_row = source_linear / padded_cols;
    const int in_col = source_linear % padded_cols;
    b_scale[out_row * cols + out_col] = (in_row < rows && in_col < cols && in_row < src_rows && in_col < src_cols)
        ? b_scale_sh[in_row * src_cols + in_col]
        : static_cast<uint8_t>(127);
}

void mxfp4_unshuffle_b_scale(torch::Tensor b_scale_sh, torch::Tensor b_scale) {
    const int rows = static_cast<int>(b_scale.size(0));
    const int cols = static_cast<int>(b_scale.size(1));
    const int src_rows = static_cast<int>(b_scale_sh.size(0));
    const int src_cols = static_cast<int>(b_scale_sh.size(1));
    TORCH_CHECK(src_rows >= rows, "B scale source rows must cover logical rows");
    TORCH_CHECK(src_cols >= cols, "B scale source cols must cover logical cols");

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    hipLaunchKernelGGL(
        mxfp4_unshuffle_b_scale_kernel,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const uint8_t*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(b_scale.data_ptr<uint8_t>()),
        rows,
        cols,
        src_rows,
        src_cols
    );
}

__device__ __forceinline__ uint8_t mxfp4_load_unshuffled_b_scale(
    const uint8_t* __restrict__ b_scale_sh,
    int rows,
    int cols,
    int src_rows,
    int src_cols,
    int out_row,
    int out_col
) {
    const int row_block = out_row >> 5;
    const int row_in_block = out_row & 31;
    const int d = row_in_block >> 4;
    const int b = row_in_block & 15;
    const int c0 = out_col >> 3;
    const int c = (out_col & 7) >> 2;
    const int a = out_col & 3;
    const int padded_cols = ((cols + 7) / 8) * 8;
    const int col_blocks = padded_cols / 8;
    const int source_linear = (((((row_block * col_blocks + c0) * 4 + a) * 16 + b) * 2 + c) * 2 + d);
    const int in_row = source_linear / padded_cols;
    const int in_col = source_linear % padded_cols;
    return (in_row < rows && in_col < cols && in_row < src_rows && in_col < src_cols)
        ? b_scale_sh[in_row * src_cols + in_col]
        : static_cast<uint8_t>(127);
}

__global__ void mxfp4_pack_b_m32_direct_with_scale_kernel(
    const uint8_t* __restrict__ b_q,
    const uint8_t* __restrict__ b_scale_sh,
    uint8_t* __restrict__ b_direct,
    uint8_t* __restrict__ b_scale,
    int n,
    int k_half,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    const int k_tile = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_idx = blockIdx.y;
    const int num_k_tiles = k_half / 32;
    if (n_idx >= n || k_tile >= num_k_tiles) {
        return;
    }

    const int byte_base = k_tile * 32;
    const uint8_t* src_b = b_q + n_idx * k_half + byte_base;
    uint8_t* dst_b = b_direct + (k_tile * n + n_idx) * 32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        dst_b[i] = src_b[i];
    }

    const int scale_col0 = k_tile * 2;
    const int scale_col1 = scale_col0 + 1;
    b_scale[n_idx * scale_cols + scale_col0] = mxfp4_load_unshuffled_b_scale(
        b_scale_sh, n, scale_cols, src_rows, src_cols, n_idx, scale_col0
    );
    if (scale_col1 < scale_cols) {
        b_scale[n_idx * scale_cols + scale_col1] = mxfp4_load_unshuffled_b_scale(
            b_scale_sh, n, scale_cols, src_rows, src_cols, n_idx, scale_col1
        );
    }
}

void mxfp4_pack_b_m32_direct_with_scale(torch::Tensor b_q, torch::Tensor b_scale_sh, torch::Tensor b_direct, torch::Tensor b_scale) {
    const int n = static_cast<int>(b_q.size(0));
    const int k_half = static_cast<int>(b_q.size(1));
    const int scale_cols = static_cast<int>(b_scale.size(1));
    const int src_rows = static_cast<int>(b_scale_sh.size(0));
    const int src_cols = static_cast<int>(b_scale_sh.size(1));
    TORCH_CHECK((k_half % 32) == 0, "B K/2 must be divisible by 32 for fused direct m32 B packing");

    dim3 block(128);
    dim3 grid((k_half / 32 + block.x - 1) / block.x, n);
    hipLaunchKernelGGL(
        mxfp4_pack_b_m32_direct_with_scale_kernel,
        grid,
        block,
        0,
        0,
        reinterpret_cast<const uint8_t*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<const uint8_t*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(b_direct.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(b_scale.data_ptr<uint8_t>()),
        n,
        k_half,
        scale_cols,
        src_rows,
        src_cols
    );
}

__device__ __forceinline__ int pack_scale_e8m0x4(const uint8_t* scale_ptr) {
    return static_cast<int>(scale_ptr[0])
        | (static_cast<int>(scale_ptr[1]) << 8)
        | (static_cast<int>(scale_ptr[2]) << 16)
        | (static_cast<int>(scale_ptr[3]) << 24);
}

__device__ __forceinline__ int pack_scale_e8m0x4_lane(const uint8_t* scale_ptr, int group4) {
    return static_cast<int>(scale_ptr[group4])
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__device__ __forceinline__ int pack_scale_e8m0x4_lane_from_shuffled(
    const uint8_t* __restrict__ b_scale_sh,
    int rows,
    int cols,
    int src_rows,
    int src_cols,
    int out_row,
    int scale_block,
    int group4
) {
    const uint8_t scale0 = mxfp4_load_unshuffled_b_scale(
        b_scale_sh, rows, cols, src_rows, src_cols, out_row, scale_block + group4
    );
    return static_cast<int>(scale0)
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m16(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int m,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const int a_row = tile_row + lane16;
        if (a_row < m) {
            const unsigned char* ldg_a = a_packed + a_row * a_bytes_per_row + tile_k / 2 + group4 * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_buf.b[i] = ldg_a[i];
            }
        }

        const unsigned char* ldg_b = b_packed + (tile_col + lane16) * b_bytes_per_row + tile_k / 2 + group4 * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            b_buf.b[i] = ldg_b[i];
        }

        const int scale_block = tile_k / 32;
        const int scale_a = (a_row < m)
            ? pack_scale_e8m0x4_lane(a_scale + a_row * a_scale_stride + scale_block, group4)
            : (127 | (127 << 8) | (127 << 16) | (127 << 24));
        const int scale_b = pack_scale_e8m0x4_lane(b_scale + (tile_col + lane16) * b_scale_stride + scale_block, group4);
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = tile_row + group4 * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < m && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m16(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);

    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, (m + 16 - 1) / 16);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m16,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m16_dense(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;
    const unsigned char* a_row_ptr = a_packed + lane16 * a_bytes_per_row + group4 * 16;
    const unsigned char* b_row_ptr = b_packed + (tile_col + lane16) * b_bytes_per_row + group4 * 16;
    const uint8_t* a_scale_ptr = a_scale + lane16 * a_scale_stride;
    const uint8_t* b_scale_ptr = b_scale + (tile_col + lane16) * b_scale_stride;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const unsigned char* ldg_a = a_row_ptr;
        const unsigned char* ldg_b = b_row_ptr;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a_buf.b[i] = ldg_a[i];
            b_buf.b[i] = ldg_b[i];
        }

        const int scale_a = pack_scale_e8m0x4_lane(a_scale_ptr, group4);
        const int scale_b = pack_scale_e8m0x4_lane(b_scale_ptr, group4);
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);

        a_row_ptr += 64;
        b_row_ptr += 64;
        a_scale_ptr += 4;
        b_scale_ptr += 4;
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = group4 * 4;
    c[(out_row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[0]);
    c[(out_row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[1]);
    c[(out_row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[2]);
    c[(out_row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[3]);
}

void mxfp4_mm_hip_mfma_scale_exact_m16_dense(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 16, "dense exact m16 path requires m == 16");

    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m16_dense,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m16_dense_rawscale(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int a_scale_stride,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;
    const unsigned char* a_row_ptr = a_packed + lane16 * a_bytes_per_row + group4 * 16;
    const unsigned char* b_row_ptr = b_packed + (tile_col + lane16) * b_bytes_per_row + group4 * 16;
    const uint8_t* a_scale_ptr = a_scale + lane16 * a_scale_stride;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const unsigned char* ldg_a = a_row_ptr;
        const unsigned char* ldg_b = b_row_ptr;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a_buf.b[i] = ldg_a[i];
            b_buf.b[i] = ldg_b[i];
        }

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x4_lane(a_scale_ptr, group4);
        const int scale_b = pack_scale_e8m0x4_lane_from_shuffled(
            b_scale_sh,
            n,
            scale_cols,
            src_rows,
            src_cols,
            tile_col + lane16,
            scale_block,
            group4
        );
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);

        a_row_ptr += 64;
        b_row_ptr += 64;
        a_scale_ptr += 4;
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = group4 * 4;
    c[(out_row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[0]);
    c[(out_row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[1]);
    c[(out_row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[2]);
    c[(out_row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[3]);
}

void mxfp4_mm_hip_mfma_scale_exact_m16_dense_rawscale(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    const int scale_cols = k / 32;
    TORCH_CHECK(m == 16, "dense exact m16 raw-scale path requires m == 16");

    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m16_dense_rawscale,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        scale_cols,
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

__device__ __forceinline__ int pack_scale_e8m0x4_lane_from_shuffled_exact_m16_k7168_fast(
    const uint8_t* __restrict__ b_scale_sh,
    int base_const,
    int step
) {
    constexpr int SCALE_COLS = 224;
    const int source_linear = base_const + (step >> 1) * 256 + (step & 1) * 2;
    const int in_row = source_linear / SCALE_COLS;
    const int in_col = source_linear - in_row * SCALE_COLS;
    const uint8_t scale0 = b_scale_sh[in_row * SCALE_COLS + in_col];
    return static_cast<int>(scale0)
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m16_k7168_n2112(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int src_rows,
    int src_cols
) {
    constexpr int N = 2112;
    constexpr int K = 7168;
    constexpr int A_SCALE_STRIDE = 224;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const int a_bytes_per_row = K / 2;
    const int b_bytes_per_row = K / 2;
    const unsigned char* a_row_ptr = a_packed + lane16 * a_bytes_per_row + group4 * 16;
    const unsigned char* b_row_ptr = b_packed + (tile_col + lane16) * b_bytes_per_row + group4 * 16;
    const uint8_t* a_scale_ptr = a_scale + lane16 * A_SCALE_STRIDE;

    const int scale_row = tile_col + lane16;
    const int row_block = scale_row >> 5;
    const int row_in_block = scale_row & 31;
    const int d = row_in_block >> 4;
    const int b = row_in_block & 15;
    const int base_const = row_block * 7168 + group4 * 64 + b * 4 + d;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int step = 0; step < (K / MFMA_K); ++step) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            a_buf.b[i] = a_row_ptr[i];
            b_buf.b[i] = b_row_ptr[i];
        }

        const int scale_a = pack_scale_e8m0x4_lane(a_scale_ptr, group4);
        const int scale_b = pack_scale_e8m0x4_lane_from_shuffled_exact_m16_k7168_fast(
            b_scale_sh, base_const, step
        );
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);

        a_row_ptr += 64;
        b_row_ptr += 64;
        a_scale_ptr += 4;
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = group4 * 4;
    c[(out_row_base + 0) * N + out_col] = static_cast<__hip_bfloat16>(acc[0]);
    c[(out_row_base + 1) * N + out_col] = static_cast<__hip_bfloat16>(acc[1]);
    c[(out_row_base + 2) * N + out_col] = static_cast<__hip_bfloat16>(acc[2]);
    c[(out_row_base + 3) * N + out_col] = static_cast<__hip_bfloat16>(acc[3]);
}

void mxfp4_mm_hip_mfma_scale_exact_m16_k7168_n2112(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 16, "exact m16 public path requires m == 16");
    TORCH_CHECK(n == 2112, "exact m16 public path requires n == 2112");
    TORCH_CHECK(k == 7168, "exact m16 public path requires k == 7168");

    dim3 block(64);
    dim3 grid(2112 / 16, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m16_k7168_n2112,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m4_dense(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int a_scale_stride,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 128;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const bool a_active = lane16 < 4;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        if (a_active) {
            const unsigned char* ldg_a = a_packed + lane16 * a_bytes_per_row + tile_k / 2 + group4 * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_buf.b[i] = ldg_a[i];
            }
        }

        const unsigned char* ldg_b = b_packed + (tile_col + lane16) * b_bytes_per_row + tile_k / 2 + group4 * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            b_buf.b[i] = ldg_b[i];
        }

        const int scale_block = tile_k / 32;
        const int scale_a = a_active
            ? pack_scale_e8m0x4_lane(a_scale + lane16 * a_scale_stride + scale_block, group4)
            : (127 | (127 << 8) | (127 << 16) | (127 << 24));
        const int scale_b = pack_scale_e8m0x4_lane_from_shuffled(
            b_scale_sh,
            n,
            scale_cols,
            src_rows,
            src_cols,
            tile_col + lane16,
            scale_block,
            group4
        );
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b);
    }

    const int out_col = tile_col + lane16;
    const int out_row_base = group4 * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < 4 && out_col < n) {
            c[out_row * n + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m4_dense(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale_sh, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    const int scale_cols = k / 32;
    TORCH_CHECK(m == 4, "dense exact m4 path requires m == 4");

    dim3 block(64);
    dim3 grid((n + 16 - 1) / 16, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m4_dense,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        scale_cols,
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

__device__ __forceinline__ uint8_t mxfp4_load_unshuffled_b_scale_exact_m4_k512(
    const uint8_t* __restrict__ b_scale_sh,
    int src_cols,
    int out_row,
    int scale_col
);

__device__ __forceinline__ int pack_scale_e8m0x4_lane_from_shuffled_exact_m4_k512(
    const uint8_t* __restrict__ b_scale_sh,
    int src_cols,
    int out_row,
    int scale_block,
    int group4
);

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m4_k512_dense(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int src_cols
) {
    constexpr int N = 2880;
    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_col = blockIdx.x * 16;
    const int lane16 = lane & 15;
    const int group4 = lane >> 4;
    const bool a_active = lane16 < 4;
    const unsigned char* a_row_ptr = a_packed + lane16 * 256 + group4 * 16;
    const int out_col = tile_col + lane16;
    const unsigned char* b_row_ptr = b_packed + out_col * 256 + group4 * 16;
    const uint8_t* a_scale_ptr = a_scale + lane16 * 16;

    const int row_block = out_col & ~31;
    const int base_row = row_block + (group4 << 2) + ((out_col >> 2) & 3);
    const int base_col = ((out_col & 3) << 2) + ((out_col >> 4) & 1);
    const int base_row0 = base_row * src_cols;
    const int base_row1 = (base_row + 16) * src_cols;
    const int base_col1 = base_col + 2;
    const int scale_b0 = static_cast<int>(b_scale_sh[base_row0 + base_col])
        | (127 << 8) | (127 << 16) | (127 << 24);
    const int scale_b1 = static_cast<int>(b_scale_sh[base_row0 + base_col1])
        | (127 << 8) | (127 << 16) | (127 << 24);
    const int scale_b2 = static_cast<int>(b_scale_sh[base_row1 + base_col])
        | (127 << 8) | (127 << 16) | (127 << 24);
    const int scale_b3 = static_cast<int>(b_scale_sh[base_row1 + base_col1])
        | (127 << 8) | (127 << 16) | (127 << 24);

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        if (a_active) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_buf.b[i] = a_row_ptr[i];
            }
        }

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            b_buf.b[i] = b_row_ptr[i];
        }

        const int scale_a = a_active
            ? pack_scale_e8m0x4_lane(a_scale_ptr, group4)
            : (127 | (127 << 8) | (127 << 16) | (127 << 24));
        const int scale_b = (step == 0)
            ? scale_b0
            : (step == 1)
                ? scale_b1
                : (step == 2) ? scale_b2 : scale_b3;
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_buf.v, b_buf.v, acc, 4, 4, 0, scale_a, 0, scale_b
        );

        a_row_ptr += 64;
        b_row_ptr += 64;
        a_scale_ptr += 4;
    }

    const int out_row_base = group4 * 4;
    #pragma unroll
    for (int row_i = 0; row_i < 4; ++row_i) {
        const int out_row = out_row_base + row_i;
        if (out_row < 4) {
            c[out_row * N + out_col] = static_cast<__hip_bfloat16>(acc[row_i]);
        }
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m4_k512_dense(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 4, "exact m4 k512 dense path requires m == 4");
    TORCH_CHECK(n == 2880, "exact m4 k512 dense path requires n == 2880");
    TORCH_CHECK(k == 512, "exact m4 k512 dense path requires k == 512");

    dim3 block(64);
    dim3 grid(2880 / 16, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m4_k512_dense,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        static_cast<int>(b_scale_sh.size(1))
    );
}

void mxfp4_mm_hip_mfma_scale_exact_m16_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for thin direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for thin direct entry");

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    if (static_cast<int>(a_in.size(1)) == 7168 && static_cast<int>(c.size(1)) == 2112) {
        mxfp4_mm_hip_mfma_scale_exact_m16_k7168_n2112(a_packed, b_q_u8, a_scale, b_scale_u8, c);
    } else {
        mxfp4_mm_hip_mfma_scale_exact_m16_dense_rawscale(a_packed, b_q_u8, a_scale, b_scale_u8, c);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(a_in.size(0) == 4, "exact m4 path requires m == 4");
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for exact m4 direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for exact m4 direct entry");

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    if (static_cast<int>(a_in.size(1)) == 512 && static_cast<int>(c.size(1)) == 2880) {
        mxfp4_mm_hip_mfma_scale_exact_m4_k512_dense(a_packed, b_q_u8, a_scale, b_scale_u8, c);
    } else {
        mxfp4_mm_hip_mfma_scale_exact_m4_dense(a_packed, b_q_u8, a_scale, b_scale_u8, c);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry_public_k512(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a;
    auto b_q_u8 = b_q;
    auto b_scale_u8 = b_scale_sh;

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    mxfp4_mm_hip_mfma_scale_exact_m4_k512_dense(a_packed, b_q_u8, a_scale, b_scale_u8, c);
}

void mxfp4_mm_hip_mfma_scale_exact_m8_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(a_in.size(0) == 8, "exact m8 path requires m == 8");
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for exact m8 direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for exact m8 direct entry");

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);
    auto b_scale = torch::empty({b_q_u8.size(0), (b_q_u8.size(1) * 2) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    mxfp4_unshuffle_b_scale(b_scale_u8, b_scale);
    mxfp4_mm_hip_mfma_scale_exact_m16(a_packed, b_q_u8, a_scale, b_scale, c);
}

void mxfp4_mm_hip_mfma_scale_exact_m4m8_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor a_packed,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    mxfp4_unshuffle_b_scale(b_scale_u8, b_scale);
    mxfp4_mm_hip_mfma_scale_exact_m16(a_packed, b_q_u8, a_scale, b_scale, c);
}

void mxfp4_mm_hip_mfma_scale_exact_m4m8_direct_entry_owned(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for thin direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for thin direct entry");

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);
    auto b_scale = torch::empty({b_q_u8.size(0), (b_q_u8.size(1) * 2) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    mxfp4_unshuffle_b_scale(b_scale_u8, b_scale);
    mxfp4_mm_hip_mfma_scale_exact_m16(a_packed, b_q_u8, a_scale, b_scale, c);
}

using floatx16 = float __attribute__((ext_vector_type(16)));

__device__ __forceinline__ int pack_scale_e8m0x2_lane(const uint8_t* scale_ptr, int group) {
    return static_cast<int>(scale_ptr[group])
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__device__ __forceinline__ int pack_scale_e8m0x2_lane_from_shuffled(
    const uint8_t* __restrict__ b_scale_sh,
    int rows,
    int cols,
    int src_rows,
    int src_cols,
    int out_row,
    int scale_block,
    int group
) {
    const uint8_t scale0 = mxfp4_load_unshuffled_b_scale(
        b_scale_sh, rows, cols, src_rows, src_cols, out_row, scale_block + group
    );
    return static_cast<int>(scale0)
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__device__ __forceinline__ uint8_t mxfp4_load_unshuffled_b_scale_exact_m32_k512(
    const uint8_t* __restrict__ b_scale_sh,
    int src_cols,
    int out_row,
    int scale_col
) {
    // Public k512 exact-m32 lane fixes cols=16, so the shuffled-scale remap
    // collapses to a closed form without the generic source_linear rebuild.
    const int in_row = (out_row & ~31)
        + ((scale_col >> 3) << 4)
        + ((scale_col & 3) << 2)
        + ((out_row >> 2) & 3);
    const int in_col = ((out_row & 3) << 2)
        + (((scale_col >> 2) & 1) << 1)
        + ((out_row >> 4) & 1);
    return b_scale_sh[in_row * src_cols + in_col];
}

__device__ __forceinline__ int pack_scale_e8m0x2_lane_from_shuffled_exact_m32_k512(
    const uint8_t* __restrict__ b_scale_sh,
    int src_cols,
    int out_row,
    int scale_block,
    int group
) {
    const uint8_t scale0 = mxfp4_load_unshuffled_b_scale_exact_m32_k512(
        b_scale_sh, src_cols, out_row, scale_block + group
    );
    return static_cast<int>(scale0)
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__device__ __forceinline__ uint8_t mxfp4_load_unshuffled_b_scale_exact_m256_k1536(
    const uint8_t* __restrict__ b_scale_sh,
    int src_rows,
    int src_cols,
    int out_row,
    int out_col
) {
    constexpr int COLS = 48;
    constexpr int PADDED_COLS = 48;
    constexpr int COL_BLOCKS = PADDED_COLS / 8;
    constexpr int ROWS = 3072;
    const int row_block = out_row >> 5;
    const int row_in_block = out_row & 31;
    const int d = row_in_block >> 4;
    const int b = row_in_block & 15;
    const int c0 = out_col >> 3;
    const int c = (out_col & 7) >> 2;
    const int a = out_col & 3;
    const int source_linear = (((((row_block * COL_BLOCKS + c0) * 4 + a) * 16 + b) * 2 + c) * 2 + d);
    const int in_row = source_linear / PADDED_COLS;
    const int in_col = source_linear % PADDED_COLS;
    return (in_row < ROWS && in_col < COLS && in_row < src_rows && in_col < src_cols)
        ? b_scale_sh[in_row * src_cols + in_col]
        : static_cast<uint8_t>(127);
}

__device__ __forceinline__ int pack_scale_e8m0x2_lane_from_shuffled_exact_m256_k1536(
    const uint8_t* __restrict__ b_scale_sh,
    int src_rows,
    int src_cols,
    int out_row,
    int scale_block,
    int group
) {
    const uint8_t scale0 = mxfp4_load_unshuffled_b_scale_exact_m256_k1536(
        b_scale_sh, src_rows, src_cols, out_row, scale_block + group
    );
    return static_cast<int>(scale0)
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__device__ __forceinline__ uint8_t mxfp4_load_unshuffled_b_scale_exact_m4_k512(
    const uint8_t* __restrict__ b_scale_sh,
    int src_cols,
    int out_row,
    int scale_col
) {
    // Exact m4 benchmark lane fixes cols=16, so the generic shuffled remap
    // reduces to the same closed form used by the public exact-m32,k512 path.
    const int in_row = (out_row & ~31)
        + ((scale_col >> 3) << 4)
        + ((scale_col & 3) << 2)
        + ((out_row >> 2) & 3);
    const int in_col = ((out_row & 3) << 2)
        + (((scale_col >> 2) & 1) << 1)
        + ((out_row >> 4) & 1);
    return b_scale_sh[in_row * src_cols + in_col];
}

__device__ __forceinline__ int pack_scale_e8m0x4_lane_from_shuffled_exact_m4_k512(
    const uint8_t* __restrict__ b_scale_sh,
    int src_cols,
    int out_row,
    int scale_block,
    int group4
) {
    const uint8_t scale0 = mxfp4_load_unshuffled_b_scale_exact_m4_k512(
        b_scale_sh, src_cols, out_row, scale_block + group4
    );
    return static_cast<int>(scale0)
        | (127 << 8)
        | (127 << 16)
        | (127 << 24);
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m32(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int m,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 64;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int a_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};
    const uint8_t* a_scale_row = a_scale + (tile_row + lane32) * a_scale_stride;
    const int out_col = tile_col + lane32;
    const uint8_t* b_scale_row = b_scale + out_col * b_scale_stride;

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        a_buf.v[4] = 0;
        a_buf.v[5] = 0;
        a_buf.v[6] = 0;
        a_buf.v[7] = 0;
        b_buf.v[4] = 0;
        b_buf.v[5] = 0;
        b_buf.v[6] = 0;
        b_buf.v[7] = 0;

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + tile_k / 2 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_a, 16));
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const int tile_index = tile_k / MFMA_K;
        const unsigned char* ldg_b = b_packed + ((tile_index * n + out_col) * 32) + group * 16;
        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_b, 16));
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane(b_scale_row + scale_block, group);
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        if (row_base + 0 < m && out_col < n) c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        if (row_base + 1 < m && out_col < n) c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        if (row_base + 2 < m && out_col < n) c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        if (row_base + 3 < m && out_col < n) c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m32_m64plus(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_packed,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale,
    __hip_bfloat16* __restrict__ c,
    int m,
    int n,
    int k,
    int a_scale_stride,
    int b_scale_stride
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 64;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int a_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};
    const uint8_t* a_scale_row = a_scale + (tile_row + lane32) * a_scale_stride;
    const int out_col = tile_col + lane32;
    const uint8_t* b_scale_row = b_scale + out_col * b_scale_stride;

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a_buf.v[i] = 0;
            b_buf.v[i] = 0;
        }

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + tile_k / 2 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_a, 16));
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const int tile_index = tile_k / MFMA_K;
        const unsigned char* ldg_b = b_packed + ((tile_index * n + out_col) * 32) + group * 16;
        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_b, 16));
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane(b_scale_row + scale_block, group);
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        if (row_base + 0 < m && out_col < n) c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        if (row_base + 1 < m && out_col < n) c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        if (row_base + 2 < m && out_col < n) c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        if (row_base + 3 < m && out_col < n) c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m32(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    auto* c_ptr = reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>());

    dim3 block(64);
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
    if (m == 64) {
        hipLaunchKernelGGL(
            mxfp4_mm_kernel_mfma_scale_exact_m32_m64plus,
            grid,
            block,
            0,
            0,
            reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
            reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
            reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
            reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
            c_ptr,
            m,
            n,
            k,
            static_cast<int>(a_scale.size(1)),
            static_cast<int>(b_scale.size(1))
        );
        return;
    }
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        c_ptr,
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

void mxfp4_mm_hip_mfma_scale_exact_m32_only(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 32, "exact m32 path requires m == 32");

    dim3 block(64);
    dim3 grid((n + 32 - 1) / 32, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m32_rawb(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_q,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int a_scale_stride,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 64;
    constexpr int EXACT_M = 32;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};
    const uint8_t* a_scale_row = a_scale + (tile_row + lane32) * a_scale_stride;
    const int out_col = tile_col + lane32;
    const unsigned char* b_row_ptr = b_q + out_col * b_bytes_per_row;

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        a_buf.v[4] = 0;
        a_buf.v[5] = 0;
        a_buf.v[6] = 0;
        a_buf.v[7] = 0;
        b_buf.v[4] = 0;
        b_buf.v[5] = 0;
        b_buf.v[6] = 0;
        b_buf.v[7] = 0;

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + tile_k / 2 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_a, 16));
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const unsigned char* ldg_b = b_row_ptr + tile_k / 2 + group * 16;
        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_b, 16));
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane_from_shuffled(
            b_scale_sh,
            n,
            scale_cols,
            src_rows,
            src_cols,
            out_col,
            scale_block,
            group
        );
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        if (row_base + 0 < EXACT_M && out_col < n) c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        if (row_base + 1 < EXACT_M && out_col < n) c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        if (row_base + 2 < EXACT_M && out_col < n) c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        if (row_base + 3 < EXACT_M && out_col < n) c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m32_rawb_only(
    torch::Tensor a_packed,
    torch::Tensor b_q,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    const int scale_cols = k / 32;
    TORCH_CHECK(m == 32, "exact m32 raw-b path requires m == 32");
    TORCH_CHECK((n % 32) == 0, "exact m32 raw-b path requires n to be divisible by 32");

    dim3 block(64);
    dim3 grid(n / 32, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32_rawb,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        scale_cols,
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m32_rawb_k512_unrolled(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_q,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int src_cols
) {
    constexpr int MFMA_N = 32;
    constexpr int EXACT_K = 512;
    constexpr int EXACT_K_HALF = EXACT_K / 2;
    constexpr int A_SCALE_STRIDE = EXACT_K / 32;
    constexpr int SCALE_COLS = EXACT_K / 32;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int tile_col = blockIdx.x * MFMA_N;
    const int out_col = tile_col + lane32;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};
    const unsigned char* a_row_ptr = a_packed + lane32 * EXACT_K_HALF;
    const unsigned char* b_row_ptr = b_q + out_col * EXACT_K_HALF;
    const uint8_t* a_scale_row = a_scale + lane32 * A_SCALE_STRIDE;

    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        a_buf.v[4] = 0;
        a_buf.v[5] = 0;
        a_buf.v[6] = 0;
        a_buf.v[7] = 0;
        b_buf.v[4] = 0;
        b_buf.v[5] = 0;
        b_buf.v[6] = 0;
        b_buf.v[7] = 0;

        const int byte_offset = step * 32 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(
            __builtin_assume_aligned(a_row_ptr + byte_offset, 16)
        );
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(
            __builtin_assume_aligned(b_row_ptr + byte_offset, 16)
        );
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = step * 2;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane_from_shuffled_exact_m32_k512(
            b_scale_sh,
            src_cols,
            out_col,
            scale_block,
            group
        );
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = group * 4 + i * 8;
        c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m32_rawb_k512_unrolled_only(
    torch::Tensor a_packed,
    torch::Tensor b_q,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 32, "exact m32 raw-b k512 path requires m == 32");
    TORCH_CHECK(k == 512, "exact m32 raw-b k512 path requires k == 512");
    TORCH_CHECK((n % 32) == 0, "exact m32 raw-b k512 path requires n to be divisible by 32");

    dim3 block(64);
    dim3 grid(n / 32, 1);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32_rawb_k512_unrolled,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        static_cast<int>(b_scale_sh.size(1))
    );
}

void mxfp4_mm_hip_mfma_scale_exact_m64_only(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 64, "exact m64 path requires m == 64");

    dim3 block(64);
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32_m64plus,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m64_rawb(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_q,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int a_scale_stride,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_K = 64;
    constexpr int EXACT_M = 64;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * 32;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int out_col = tile_col + lane32;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};

    const uint8_t* a_scale_row = a_scale + (tile_row + lane32) * a_scale_stride;
    const unsigned char* b_row_ptr = b_q + out_col * b_bytes_per_row;

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        a_buf.v[4] = 0;
        a_buf.v[5] = 0;
        a_buf.v[6] = 0;
        a_buf.v[7] = 0;
        b_buf.v[4] = 0;
        b_buf.v[5] = 0;
        b_buf.v[6] = 0;
        b_buf.v[7] = 0;

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + tile_k / 2 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_a, 16));
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const unsigned char* ldg_b = b_row_ptr + tile_k / 2 + group * 16;
        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_b, 16));
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane_from_shuffled(
            b_scale_sh,
            n,
            scale_cols,
            src_rows,
            src_cols,
            out_col,
            scale_block,
            group
        );
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        if (out_col < n) c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        if (out_col < n) c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        if (out_col < n) c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        if (out_col < n) c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m64_rawb_only(
    torch::Tensor a_packed,
    torch::Tensor b_q,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    const int scale_cols = k / 32;
    TORCH_CHECK(m == 64, "exact m64 raw-b path requires m == 64");
    TORCH_CHECK((n % 32) == 0, "exact m64 raw-b path requires n to be divisible by 32");

    dim3 block(64);
    dim3 grid(n / 32, 2);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m64_rawb,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        scale_cols,
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

void mxfp4_mm_hip_mfma_scale_exact_m256_only(torch::Tensor a_packed, torch::Tensor b_packed, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 256, "exact m256 path requires m == 256");

    dim3 block(64);
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m32,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_packed.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        static_cast<int>(b_scale.size(1))
    );
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m256_rawb(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_q,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int n,
    int k,
    int a_scale_stride,
    int scale_cols,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 64;
    constexpr int EXACT_M = 256;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int a_bytes_per_row = k / 2;
    const int b_bytes_per_row = k / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};
    const uint8_t* a_scale_row = a_scale + (tile_row + lane32) * a_scale_stride;
    const int out_col = tile_col + lane32;
    const unsigned char* b_row_ptr = b_q + out_col * b_bytes_per_row;

    for (int tile_k = 0; tile_k < k; tile_k += MFMA_K) {
        a_buf.v[4] = 0;
        a_buf.v[5] = 0;
        a_buf.v[6] = 0;
        a_buf.v[7] = 0;
        b_buf.v[4] = 0;
        b_buf.v[5] = 0;
        b_buf.v[6] = 0;
        b_buf.v[7] = 0;

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + tile_k / 2 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_a, 16));
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const unsigned char* ldg_b = b_row_ptr + tile_k / 2 + group * 16;
        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_b, 16));
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = tile_k / 32;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane_from_shuffled(
            b_scale_sh,
            n,
            scale_cols,
            src_rows,
            src_cols,
            out_col,
            scale_block,
            group
        );
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        if (row_base + 0 < EXACT_M && out_col < n) c[(row_base + 0) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        if (row_base + 1 < EXACT_M && out_col < n) c[(row_base + 1) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        if (row_base + 2 < EXACT_M && out_col < n) c[(row_base + 2) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        if (row_base + 3 < EXACT_M && out_col < n) c[(row_base + 3) * n + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

__global__ void mxfp4_mm_kernel_mfma_scale_exact_m256_k1536_n3072(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ b_q,
    const uint8_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b_scale_sh,
    __hip_bfloat16* __restrict__ c,
    int src_rows,
    int src_cols
) {
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 64;
    constexpr int EXACT_M = 256;
    constexpr int N = 3072;
    constexpr int K = 1536;
    constexpr int A_SCALE_STRIDE = 48;

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int tile_row = blockIdx.y * MFMA_M;
    const int tile_col = blockIdx.x * MFMA_N;
    const int lane32 = lane & 31;
    const int group = lane >> 5;
    const int a_bytes_per_row = K / 2;
    const int b_bytes_per_row = K / 2;

    union { i32x8_t v; unsigned char b[32]; } a_buf;
    union { i32x8_t v; unsigned char b[32]; } b_buf;
    opus::fp32x16_t acc{};
    auto mma = opus::mfma<opus::fp4_t, opus::fp4_t, opus::fp32_t, 32, 32, 64>{};
    const uint8_t* a_scale_row = a_scale + (tile_row + lane32) * A_SCALE_STRIDE;
    const int out_col = tile_col + lane32;
    const unsigned char* b_row_ptr = b_q + out_col * b_bytes_per_row;

    #pragma unroll
    for (int step = 0; step < (K / MFMA_K); ++step) {
        a_buf.v[4] = 0;
        a_buf.v[5] = 0;
        a_buf.v[6] = 0;
        a_buf.v[7] = 0;
        b_buf.v[4] = 0;
        b_buf.v[5] = 0;
        b_buf.v[6] = 0;
        b_buf.v[7] = 0;

        const unsigned char* ldg_a = a_packed + (tile_row + lane32) * a_bytes_per_row + step * 32 + group * 16;
        const i32x4_t* ldg_a_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_a, 16));
        *reinterpret_cast<i32x4_t*>(&a_buf.b[0]) = ldg_a_vec[0];

        const unsigned char* ldg_b = b_row_ptr + step * 32 + group * 16;
        const i32x4_t* ldg_b_vec = reinterpret_cast<const i32x4_t*>(__builtin_assume_aligned(ldg_b, 16));
        *reinterpret_cast<i32x4_t*>(&b_buf.b[0]) = ldg_b_vec[0];

        const int scale_block = step * 2;
        const int scale_a = pack_scale_e8m0x2_lane(a_scale_row + scale_block, group);
        const int scale_b = pack_scale_e8m0x2_lane_from_shuffled_exact_m256_k1536(
            b_scale_sh, src_rows, src_cols, out_col, scale_block, group
        );
        acc = mma(a_buf.v, b_buf.v, acc, scale_a, scale_b);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row_base = tile_row + group * 4 + i * 8;
        c[(row_base + 0) * N + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 0]);
        c[(row_base + 1) * N + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 1]);
        c[(row_base + 2) * N + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 2]);
        c[(row_base + 3) * N + out_col] = static_cast<__hip_bfloat16>(acc[i * 4 + 3]);
    }
}

void mxfp4_mm_hip_mfma_scale_exact_m256_k1536_n3072(
    torch::Tensor a_packed,
    torch::Tensor b_q,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    TORCH_CHECK(m == 256, "exact m256 public path requires m == 256");
    TORCH_CHECK(n == 3072, "exact m256 public path requires n == 3072");
    TORCH_CHECK(k == 1536, "exact m256 public path requires k == 1536");

    dim3 block(64);
    dim3 grid(3072 / 32, 256 / 32);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m256_k1536_n3072,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

void mxfp4_mm_hip_mfma_scale_exact_m256_rawb_only(
    torch::Tensor a_packed,
    torch::Tensor b_q,
    torch::Tensor a_scale,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    const int m = static_cast<int>(c.size(0));
    const int n = static_cast<int>(c.size(1));
    const int k = static_cast<int>(a_packed.size(1) * 2);
    const int scale_cols = k / 32;
    TORCH_CHECK(m == 256, "exact m256 raw-b path requires m == 256");
    TORCH_CHECK((n % 32) == 0, "exact m256 raw-b path requires n to be divisible by 32");

    dim3 block(64);
    dim3 grid(n / 32, m / 32);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel_mfma_scale_exact_m256_rawb,
        grid,
        block,
        0,
        0,
        reinterpret_cast<unsigned char const*>(a_packed.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char const*>(b_q.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(a_scale.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t const*>(b_scale_sh.data_ptr<uint8_t>()),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        n,
        k,
        static_cast<int>(a_scale.size(1)),
        scale_cols,
        static_cast<int>(b_scale_sh.size(0)),
        static_cast<int>(b_scale_sh.size(1))
    );
}

void mxfp4_mm_hip_mfma_scale_exact_m32_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(a_in.size(0) == 32, "exact m32 path requires m == 32");
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for exact m32 direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for exact m32 direct entry");
    const int k = static_cast<int>(a_in.size(1));

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    if (k == 512) {
        mxfp4_mm_hip_mfma_scale_exact_m32_rawb_k512_unrolled_only(a_packed, b_q_u8, a_scale, b_scale_u8, c);
        return;
    }
    mxfp4_mm_hip_mfma_scale_exact_m32_rawb_only(a_packed, b_q_u8, a_scale, b_scale_u8, c);
}

void mxfp4_mm_hip_mfma_scale_exact_m64_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(a_in.size(0) == 64, "exact m64 path requires m == 64");
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for exact m64 direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for exact m64 direct entry");

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    mxfp4_mm_hip_mfma_scale_exact_m64_rawb_only(a_packed, b_q_u8, a_scale, b_scale_u8, c);
}

void mxfp4_mm_hip_mfma_scale_exact_m256_direct_entry(
    torch::Tensor a,
    torch::Tensor b_q,
    torch::Tensor b_scale_sh,
    torch::Tensor c
) {
    auto a_in = a.contiguous();
    auto b_q_u8 = b_q.contiguous();
    auto b_scale_u8 = b_scale_sh.contiguous();
    TORCH_CHECK(a_in.size(0) == 256, "exact m256 path requires m == 256");
    TORCH_CHECK(b_q_u8.scalar_type() == torch::kUInt8, "b_q must be uint8-packed for exact m256 direct entry");
    TORCH_CHECK(b_scale_u8.scalar_type() == torch::kUInt8, "b_scale_sh must be uint8-packed for exact m256 direct entry");
    TORCH_CHECK((b_q_u8.size(0) % 32) == 0, "exact m256 path requires n to be divisible by 32");

    auto opts_u8 = torch::TensorOptions().device(a_in.device()).dtype(torch::kUInt8);
    auto a_packed = torch::empty({a_in.size(0), a_in.size(1) / 2}, opts_u8);
    auto a_scale = torch::empty({a_in.size(0), a_in.size(1) / 32}, opts_u8);

    mxfp4_pack_a_fixed(a_in, a_packed, a_scale);
    if (static_cast<int>(a_in.size(1)) == 1536 && static_cast<int>(c.size(1)) == 3072) {
        mxfp4_mm_hip_mfma_scale_exact_m256_k1536_n3072(a_packed, b_q_u8, a_scale, b_scale_u8, c);
        return;
    }
    mxfp4_mm_hip_mfma_scale_exact_m256_rawb_only(a_packed, b_q_u8, a_scale, b_scale_u8, c);
}

void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));
    dim3 block(TILE_N, TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel,
        grid,
        block,
        0,
        0,
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k
    );
}
"""

_MODULE = None
_TRITON_QUANT = None
_ROCTX_LIB = None
_ROCTX_READY = False


def _phase(name: str, **payload: object) -> None:
    return None


def _roctx_lib():
    global _ROCTX_LIB, _ROCTX_READY
    if _ROCTX_READY:
        return _ROCTX_LIB
    _ROCTX_READY = True
    if os.environ.get("MXFP4_ROCTX_ENABLE", "").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
        return None
    for name in ("libroctx64.so", "libroctx64.so.1"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        lib.roctxRangePushA.restype = ctypes.c_int
        lib.roctxRangePop.argtypes = []
        lib.roctxRangePop.restype = ctypes.c_int
        _ROCTX_LIB = lib
        break
    return _ROCTX_LIB


@contextmanager
def _roctx_range(name: str):
    lib = _roctx_lib()
    pushed = False
    if lib is not None:
        try:
            lib.roctxRangePushA(name.encode("utf-8"))
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed and lib is not None:
            try:
                lib.roctxRangePop()
            except Exception:
                pass


def _emit_error_stats(tag: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    got_f = got.to(torch.float32)
    ref_f = ref.to(torch.float32)
    abs_err = (got_f - ref_f).abs()
    rel_err = abs_err / ref_f.abs().clamp_min(1e-6)
    edges = torch.tensor([1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0], device=rel_err.device)
    bucket_ids = torch.bucketize(rel_err.reshape(-1), edges)
    bucket_counts = torch.bincount(bucket_ids, minlength=edges.numel() + 1).cpu().tolist()
    _phase(
        "error_stats",
        tag=tag,
        mae=float(abs_err.mean().item()),
        max_abs=float(abs_err.max().item()),
        mean_rel=float(rel_err.mean().item()),
        tol_hits={
            "atol_0.5_rtol_0.05": int((abs_err <= (0.5 + 0.05 * ref_f.abs())).sum().item()),
            "atol_1.0_rtol_0.10": int((abs_err <= (1.0 + 0.10 * ref_f.abs())).sum().item()),
        },
        rel_hist={
            "lt_1e-2": int(bucket_counts[0]),
            "1e-2_to_5e-2": int(bucket_counts[1]),
            "5e-2_to_1e-1": int(bucket_counts[2]),
            "1e-1_to_2e-1": int(bucket_counts[3]),
            "2e-1_to_5e-1": int(bucket_counts[4]),
            "5e-1_to_1e0": int(bucket_counts[5]),
            "ge_1e0": int(bucket_counts[6]),
        },
    )


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mxfp4_mm_hip_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode("utf-8")).hexdigest()[:12]
        module_name = f"mxfp4_mm_hip_{CONFIG['variant_name']}_{digest}"
        _phase("pre_module_build", module_name=module_name)
        _MODULE = load_inline(
            name=module_name,
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["mxfp4_mm_hip", "mxfp4_mm_hip_mfma_medium", "mxfp4_mm_hip_mfma_scale_exact_m16", "mxfp4_mm_hip_mfma_scale_exact_m16_dense", "mxfp4_mm_hip_mfma_scale_exact_m16_direct_entry", "mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry", "mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry_public_k512", "mxfp4_mm_hip_mfma_scale_exact_m8_direct_entry", "mxfp4_mm_hip_mfma_scale_exact_m4m8_direct_entry", "mxfp4_mm_hip_mfma_scale_exact_m4m8_direct_entry_owned", "mxfp4_mm_hip_mfma_scale_exact_m32", "mxfp4_mm_hip_mfma_scale_exact_m32_only", "mxfp4_mm_hip_mfma_scale_exact_m32_rawb_only", "mxfp4_mm_hip_mfma_scale_exact_m64_only", "mxfp4_mm_hip_mfma_scale_exact_m64_rawb_only", "mxfp4_mm_hip_mfma_scale_exact_m256_only", "mxfp4_mm_hip_mfma_scale_exact_m256_rawb_only", "mxfp4_mm_hip_mfma_scale_exact_m32_direct_entry", "mxfp4_mm_hip_mfma_scale_exact_m64_direct_entry", "mxfp4_mm_hip_mfma_scale_exact_m256_direct_entry", "mxfp4_pack_a_fixed", "mxfp4_repack_b_packed", "mxfp4_pack_b_m32_direct", "mxfp4_pack_b_m32_direct_with_scale", "mxfp4_unshuffle_b_scale"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3", "-I/home/runner/aiter/csrc/include"],
            build_directory=str(build_root),
            verbose=False,
        )
        _phase("post_module_build", module_name=module_name)
    return _MODULE


def _quant():
    global _TRITON_QUANT
    if _TRITON_QUANT is None:
        _TRITON_QUANT = aiter.get_triton_quant(QuantType.per_1x32)
    return _TRITON_QUANT


def _expand_scales(scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    scales = scale_e8m0.contiguous()[:rows]
    scales = scales.repeat_interleave(SCALE_GROUP, dim=1)[:, :cols]
    return fp4_utils.e8m0_to_f32(scales).to(torch.float32)


def _learn_adjustment_rules(
    norm: torch.Tensor,
    ref_vals: torch.Tensor,
    live_vals: torch.Tensor,
) -> dict[float, tuple[str, float, float]]:
    rules: dict[float, tuple[str, float, float]] = {}
    for q_tensor in torch.unique(ref_vals):
        q = float(q_tensor.item())
        mask = ref_vals == q
        if int(mask.sum().item()) == 0:
            continue
        labels = (live_vals != ref_vals)[mask]
        total = int(labels.numel())
        positives = int(labels.sum().item())
        if positives == 0:
            continue
        if positives == total:
            adjusted = float(torch.unique(live_vals[mask], return_counts=True)[0][0].item())
            rules[q] = ("all", 0.0, adjusted)
            continue

        values = norm[mask]
        live_subset = live_vals[mask]
        pos_live = live_subset[labels]
        uniq_live, cnt_live = torch.unique(pos_live, return_counts=True)
        adjusted = float(uniq_live[torch.argmax(cnt_live)].item())

        sorted_vals, order = torch.sort(values.reshape(-1))
        sorted_labels = labels.reshape(-1)[order].to(torch.int64)
        prefix_pos = torch.cumsum(sorted_labels, dim=0)
        prefix_idx = torch.arange(1, sorted_labels.numel() + 1, device=sorted_labels.device, dtype=torch.int64)
        prefix_neg = prefix_idx - prefix_pos
        total_pos = int(prefix_pos[-1].item())
        total_neg = sorted_labels.numel() - total_pos
        suffix_pos = total_pos - prefix_pos
        suffix_neg = total_neg - prefix_neg

        err_le = prefix_neg + suffix_pos
        err_gt = prefix_pos + suffix_neg
        best_le = int(torch.argmin(err_le).item())
        best_gt = int(torch.argmin(err_gt).item())
        err_le_val = int(err_le[best_le].item())
        err_gt_val = int(err_gt[best_gt].item())

        if err_le_val <= err_gt_val:
            rules[q] = ("le", float(sorted_vals[best_le].item()), adjusted)
        else:
            rules[q] = ("gt", float(sorted_vals[best_gt].item()), adjusted)
    return rules


def _apply_adjustment_rules(
    norm: torch.Tensor,
    ref_vals: torch.Tensor,
    rules: dict[float, tuple[str, float, float]],
) -> torch.Tensor:
    corrected = ref_vals.clone()
    for q, (direction, threshold, adjusted) in rules.items():
        mask = ref_vals == q
        if direction == "all":
            cond = mask
        elif direction == "le":
            cond = mask & (norm <= threshold)
        else:
            cond = mask & (norm > threshold)
        corrected = torch.where(cond, torch.full_like(corrected, adjusted), corrected)
    return corrected


def _get_b_contract(
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[dict[float, tuple[str, float, float]], torch.Tensor]:
    quant = _quant()
    public_b_q, b_scale = quant(b.contiguous(), shuffle=False)
    b_scale_f32 = _expand_scales(b_scale, rows=b.shape[0], cols=b.shape[1])
    b_ref_vals = fp4_utils.mxfp4_to_f32(b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)
    b_public_vals = fp4_utils.mxfp4_to_f32(public_b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)
    norm_b = (b.to(torch.float32) / b_scale_f32).contiguous()
    rules = _learn_adjustment_rules(norm_b, b_public_vals, b_ref_vals)
    b_ref = (b_ref_vals * b_scale_f32).to(torch.float32).contiguous()
    return rules, b_ref



def _get_b_contract_bf16(
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> torch.Tensor:
    _, b_ref = _get_b_contract(b, b_q, b_scale_sh)
    return b_ref.to(torch.bfloat16).contiguous()


def _get_b_contract_mfma_fp4(
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    _, b_scale = quant(b.contiguous(), shuffle=False)
    b_ref_vals = fp4_utils.mxfp4_to_f32(b_q.contiguous())[: b.shape[0], : b.shape[1]].to(torch.float32)
    b_packed = fp4_utils.f32_to_mxfp4(b_ref_vals.t().contiguous()).contiguous().view(torch.uint8)
    b_scale_u8 = b_scale.contiguous().view(torch.uint8)
    return b_packed, b_scale_u8


def _repack_mxfp4_nk_to_kn_packed(b_q_u8: torch.Tensor) -> torch.Tensor:
    src = b_q_u8.contiguous()
    rows_n, cols_k_half = src.shape
    torch._assert((rows_n % 2) == 0, "B row count must be even for packed row-major repack")
    lo = (src & 0x0F).transpose(0, 1).contiguous()
    hi = (src >> 4).transpose(0, 1).contiguous()
    packed_even_k = lo[:, 0::2] | (lo[:, 1::2] << 4)
    packed_odd_k = hi[:, 0::2] | (hi[:, 1::2] << 4)
    out = torch.empty((cols_k_half * 2, rows_n // 2), dtype=torch.uint8, device=src.device)
    out[0::2] = packed_even_k
    out[1::2] = packed_odd_k
    return out.contiguous()


def _e8m0_unshuffle(scale_sh_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    scale_src = scale_sh_u8.contiguous()[:rows, :cols]
    padded_rows = ((rows + 255) // 256) * 256
    padded_cols = ((cols + 7) // 8) * 8
    scale_padded = torch.full(
        (padded_rows, padded_cols),
        127,
        dtype=scale_src.dtype,
        device=scale_src.device,
    )
    scale_padded[:rows, :cols] = scale_src
    scale_unshuffled = scale_padded.view(
        padded_rows // 32,
        padded_cols // 8,
        4,
        16,
        2,
        2,
    )
    scale_unshuffled = scale_unshuffled.permute(0, 5, 3, 1, 4, 2).contiguous()
    return scale_unshuffled.view(padded_rows, padded_cols)[:rows, :cols].contiguous()


def _get_b_contract_mfma_fp4_live(
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b_q_u8 = b_q.contiguous().view(torch.uint8)
    b_packed = torch.empty((b_q_u8.shape[1] * 2, b_q_u8.shape[0] // 2), dtype=torch.uint8, device=b_q_u8.device)
    _module().mxfp4_repack_b_packed(b_q_u8, b_packed)
    b_scale_u8 = b_scale_sh.contiguous().view(torch.uint8)
    b_scale_rowmajor = torch.empty(
        (b_q_u8.shape[0], (b_q_u8.shape[1] * 2) // SCALE_GROUP),
        dtype=torch.uint8,
        device=b_q_u8.device,
    )
    _module().mxfp4_unshuffle_b_scale(b_scale_u8, b_scale_rowmajor)
    return b_packed, b_scale_rowmajor


def _get_b_contract_mfma_fp4_live_m32_direct(
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b_q_u8 = b_q.contiguous().view(torch.uint8)
    torch._assert((b_q_u8.shape[1] % 32) == 0, "B K/2 must be divisible by 32 for direct m32 B packing")
    b_packed = torch.empty((b_q_u8.shape[1] // 32, b_q_u8.shape[0], 32), dtype=torch.uint8, device=b_q_u8.device)
    b_scale_u8 = b_scale_sh.contiguous().view(torch.uint8)
    b_scale_rowmajor = torch.empty(
        (b_q_u8.shape[0], (b_q_u8.shape[1] * 2) // SCALE_GROUP),
        dtype=torch.uint8,
        device=b_q_u8.device,
    )
    _module().mxfp4_pack_b_m32_direct_with_scale(b_q_u8, b_scale_u8, b_packed, b_scale_rowmajor)
    return b_packed, b_scale_rowmajor


def _get_b_contract_mfma_fp4_live_m16_direct(
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b_q_u8 = b_q.contiguous().view(torch.uint8)
    b_scale_u8 = b_scale_sh.contiguous().view(torch.uint8)
    b_scale_rowmajor = torch.empty(
        (b_q_u8.shape[0], (b_q_u8.shape[1] * 2) // SCALE_GROUP),
        dtype=torch.uint8,
        device=b_q_u8.device,
    )
    _module().mxfp4_unshuffle_b_scale(b_scale_u8, b_scale_rowmajor)
    return b_q_u8, b_scale_rowmajor


def _get_b_preshuffled_mfma_fp4(b_q: torch.Tensor) -> torch.Tensor:
    b_q_u8 = b_q.contiguous().view(torch.uint8)
    torch._assert((b_q_u8.shape[0] % 16) == 0, "B preshuffle rows must be a multiple of 16")
    b_preshuffled = shuffle_weight(b_q_u8, layout=(16, 16), use_int4=False).reshape(
        b_q_u8.shape[0] // 16,
        b_q_u8.shape[1] * 16,
    ).contiguous()
    return b_preshuffled


def _get_a_contract_mfma_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    a_q, a_scale = quant(a.contiguous(), shuffle=False)
    a_scale_f32 = _expand_scales(a_scale, rows=a.shape[0], cols=a.shape[1])
    a_ref_vals = fp4_utils.mxfp4_to_f32(a_q.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)
    norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
    a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, FIXED_ADJUSTMENT_RULES).contiguous()
    a_packed = fp4_utils.f32_to_mxfp4(a_corrected_vals).contiguous().view(torch.uint8)
    return a_packed, a_scale.contiguous().view(torch.uint8)


def _get_a_contract_mfma_fp4_compiled(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_in = a.contiguous()
    a_packed = torch.empty((a_in.shape[0], a_in.shape[1] // 2), dtype=torch.uint8, device=a_in.device)
    a_scale = torch.empty((a_in.shape[0], a_in.shape[1] // SCALE_GROUP), dtype=torch.uint8, device=a_in.device)
    _module().mxfp4_pack_a_fixed(a_in, a_packed, a_scale)
    return a_packed, a_scale


def _get_a_contract_mfma_fp4_compiled_exact_m32(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_a_contract_mfma_fp4_compiled(a)


def _get_a_contract_mfma_fp4_compiled_exact_m64(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_a_contract_mfma_fp4_compiled(a)


def _get_a_contract_mfma_fp4_compiled_exact_m256(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_a_contract_mfma_fp4_compiled(a)


def _get_b_contract_mfma_fp4_live_m32_direct_exact_m32(
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_b_contract_mfma_fp4_live_m32_direct(b_q, b_scale_sh)


def _get_b_contract_mfma_fp4_live_m32_direct_exact_m64(
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_b_contract_mfma_fp4_live_m32_direct(b_q, b_scale_sh)


def _get_b_contract_mfma_fp4_live_m32_direct_exact_m256(
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_b_contract_mfma_fp4_live_m32_direct(b_q, b_scale_sh)

def _reference_oracle_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    a_q, a_scale = quant(a.contiguous(), shuffle=False)
    a_scale_f32 = _expand_scales(a_scale, rows=a.shape[0], cols=a.shape[1])
    a_ref_vals = fp4_utils.mxfp4_to_f32(a_q.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)

    rules, b_ref = _get_b_contract(b, b_q, b_scale_sh)

    norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
    a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, rules)
    a_ref = (a_corrected_vals * a_scale_f32).to(torch.float32).contiguous()
    return a_ref, b_ref


def _get_corrected_a_preshuffle(
    a: torch.Tensor,
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    a_q_raw, a_scale_raw = quant(a.contiguous(), shuffle=False)
    a_scale_f32 = _expand_scales(a_scale_raw, rows=a.shape[0], cols=a.shape[1])
    a_ref_vals = fp4_utils.mxfp4_to_f32(a_q_raw.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)
    rules, _ = _get_b_contract(b, b_q, b_scale_sh)
    norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
    a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, rules)
    a_corrected = (a_corrected_vals * a_scale_f32).to(torch.float32).contiguous()
    return quant(a_corrected, shuffle=True)


def _get_corrected_a_preshuffle(
    a: torch.Tensor,
    b: torch.Tensor,
    b_q: torch.Tensor,
    b_scale_sh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = _quant()
    a_q_raw, a_scale_raw = quant(a.contiguous(), shuffle=False)
    a_scale_f32 = _expand_scales(a_scale_raw, rows=a.shape[0], cols=a.shape[1])
    a_ref_vals = fp4_utils.mxfp4_to_f32(a_q_raw.contiguous())[: a.shape[0], : a.shape[1]].to(torch.float32)
    rules, _ = _get_b_contract(b, b_q, b_scale_sh)
    norm_a = (a.to(torch.float32) / a_scale_f32).contiguous()
    a_corrected_vals = _apply_adjustment_rules(norm_a, a_ref_vals, rules)
    a_corrected = (a_corrected_vals * a_scale_f32).to(torch.float32).contiguous()
    return quant(a_corrected, shuffle=True)


def _select_kernel_regime(m: int, k: int) -> str:
    if m <= 16:
        return "tiny_m"
    if m <= 128:
        return "medium_m"
    return "fallback"


def custom_kernel(data: input_t) -> output_t:
    a, b, b_q, b_shuffle, b_scale_sh = data
    with _roctx_range("mxfp4/custom_kernel"):
        _phase("enter_custom_kernel", m=int(a.shape[0]), k=int(a.shape[1]), n=int(b.shape[0]))
        exact_tiny_m = a.shape[0] in (4, 8)
        torch._assert(b_q.shape[0] == b.shape[0], "B_q row count must match logical B")
        torch._assert(b_shuffle.shape[0] == b.shape[0], "B_shuffle row count must match logical B")
        torch._assert(b_scale_sh.numel() > 0, "B_scale_sh must be present for the live contract")
        if a.shape[0] == 256 and (a.shape[1] % 64) == 0 and (b.shape[0] % 32) == 0:
            with _roctx_range("mxfp4/exact_m256"):
                _phase("path_exact_m256")
                mod = _module()
                _phase("post_module_return", path="exact_m256")
                c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
                _phase("post_output_alloc", c_shape=list(c.shape))
                with _roctx_range("mxfp4/exact_m256/kernel_launch"):
                    _phase("pre_direct_kernel_launch", chunked=False)
                    mod.mxfp4_mm_hip_mfma_scale_exact_m256_direct_entry(
                        a,
                        b_q.contiguous().view(torch.uint8),
                        b_scale_sh.contiguous().view(torch.uint8),
                        c,
                    )
                    _phase("post_wrapper_return", chunked=False)
                return c
        if a.shape[0] == 64 and (a.shape[1] % 64) == 0 and (b.shape[0] % 32) == 0:
            with _roctx_range("mxfp4/exact_m64"):
                _phase("path_exact_m64")
                mod = _module()
                _phase("post_module_return", path="exact_m64")
                c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
                _phase("post_output_alloc", c_shape=list(c.shape))
                with _roctx_range("mxfp4/exact_m64/kernel_launch"):
                    _phase("pre_direct_kernel_launch", chunked=False)
                    mod.mxfp4_mm_hip_mfma_scale_exact_m64_direct_entry(
                        a,
                        b_q.contiguous().view(torch.uint8),
                        b_scale_sh.contiguous().view(torch.uint8),
                        c,
                    )
                    _phase("post_wrapper_return", chunked=False)
                return c
        if a.shape[0] == 32 and (a.shape[1] % 64) == 0 and (b.shape[0] % 32) == 0:
            with _roctx_range("mxfp4/exact_m32"):
                _phase("path_exact_m32")
                mod = _module()
                _phase("post_module_return", path="exact_m32")
                c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
                _phase("post_output_alloc", c_shape=list(c.shape))
                with _roctx_range("mxfp4/exact_m32/kernel_launch"):
                    _phase("pre_direct_kernel_launch", chunked=False)
                    mod.mxfp4_mm_hip_mfma_scale_exact_m32_direct_entry(
                        a,
                        b_q.contiguous().view(torch.uint8),
                        b_scale_sh.contiguous().view(torch.uint8),
                        c,
                    )
                    _phase("post_wrapper_return", chunked=False)
                return c
        if a.shape[0] >= 32 and (a.shape[0] % 32) == 0 and (a.shape[1] % 64) == 0 and (b.shape[0] % 32) == 0:
            _phase("path_direct_m32_other_multiples32", rows=int(a.shape[0]))
            mod = _module()
            _phase("post_module_return", path="direct_m32_other_multiples32")
            _phase("pre_b_prep", m=int(a.shape[0]), k=int(a.shape[1]), n=int(b.shape[0]))
            b_packed, b_scale = _get_b_contract_mfma_fp4_live_m32_direct(b_q, b_scale_sh)
            _phase("post_b_prep", b_packed_shape=list(b_packed.shape), b_scale_shape=list(b_scale.shape))
            _phase("pre_a_pack_scale_prep", m=int(a.shape[0]), k=int(a.shape[1]), n=int(b.shape[0]))
            a_packed, a_scale = _get_a_contract_mfma_fp4_compiled(a)
            _phase("post_a_pack_scale_prep", packed_shape=list(a_packed.shape), scale_shape=list(a_scale.shape))
            c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
            _phase("post_output_alloc", c_shape=list(c.shape))
            inflight = globals().setdefault("_MFMA_SCALE_INFLIGHT", [])
            inflight.append((a_packed, a_scale, b_packed, b_scale))
            if len(inflight) > 64:
                del inflight[:-64]
            _phase("pre_direct_kernel_launch", chunked=False)
            mod.mxfp4_mm_hip_mfma_scale_exact_m32(a_packed, b_packed, a_scale, b_scale, c)
            _phase("post_wrapper_return", chunked=False)
            return c
        if a.shape[0] == 16 and (a.shape[1] % 128) == 0 and (b.shape[0] % 16) == 0:
            with _roctx_range("mxfp4/exact_m16"):
                _phase("path_direct_m16")
                mod = _module()
                _phase("post_module_return", path="direct_m16")
                c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
                _phase("post_output_alloc", c_shape=list(c.shape))
                with _roctx_range("mxfp4/exact_m16/kernel_launch"):
                    _phase("pre_direct_kernel_launch", chunked=False)
                    mod.mxfp4_mm_hip_mfma_scale_exact_m16_direct_entry(
                        a.contiguous(),
                        b_q.contiguous().view(torch.uint8),
                        b_scale_sh.contiguous().view(torch.uint8),
                        c,
                    )
                    _phase("post_wrapper_return", chunked=False)
                return c
        if a.shape[0] == 8 and (a.shape[1] % 128) == 0 and (b.shape[0] % 16) == 0:
            with _roctx_range("mxfp4/exact_m8"):
                _phase("path_exact_m8")
                mod = _module()
                _phase("post_module_return", path="exact_m8")
                c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
                _phase("post_output_alloc", c_shape=list(c.shape))
                with _roctx_range("mxfp4/exact_m8/kernel_launch"):
                    _phase("pre_direct_kernel_launch", chunked=False)
                    mod.mxfp4_mm_hip_mfma_scale_exact_m8_direct_entry(
                        a.contiguous(),
                        b_q.contiguous().view(torch.uint8),
                        b_scale_sh.contiguous().view(torch.uint8),
                        c,
                    )
                    _phase("post_wrapper_return", chunked=False)
                return c
        if a.shape[0] == 4 and (a.shape[1] % 128) == 0 and (b.shape[0] % 16) == 0:
            with _roctx_range("mxfp4/exact_m4"):
                _phase("path_exact_m4")
                mod = _module()
                _phase("post_module_return", path="exact_m4")
                c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
                _phase("post_output_alloc", c_shape=list(c.shape))
                with _roctx_range("mxfp4/exact_m4/kernel_launch"):
                    _phase("pre_direct_kernel_launch", chunked=False)
                    if int(a.shape[1]) == 512 and int(b.shape[0]) == 2880:
                        mod.mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry_public_k512(
                            a.contiguous(),
                            b_q.contiguous().view(torch.uint8),
                            b_scale_sh.contiguous().view(torch.uint8),
                            c,
                        )
                    else:
                        mod.mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry(
                            a.contiguous(),
                            b_q.contiguous().view(torch.uint8),
                            b_scale_sh.contiguous().view(torch.uint8),
                            c,
                        )
                    _phase("post_wrapper_return", chunked=False)
                return c
        _phase("pre_reference_oracle_inputs")
        a_in, b_in = _reference_oracle_inputs(a, b, b_q, b_scale_sh)
        _phase("post_reference_oracle_inputs", a_ref_shape=list(a_in.shape), b_ref_shape=list(b_in.shape))
        regime = _select_kernel_regime(a_in.shape[0], a_in.shape[1])
        use_mfma_medium = (
            regime == "medium_m"
            and a_in.shape[0] == 16
            and (a_in.shape[0] % 16) == 0
            and (a_in.shape[1] % 16) == 0
            and (b_in.shape[0] % 16) == 0
        )
        _phase("regime_selected", regime=regime, use_mfma_medium=bool(use_mfma_medium))
        if use_mfma_medium:
            _phase("path_mfma_medium")
            a_mfma = a_in.to(torch.bfloat16).contiguous()
            b_mfma = _get_b_contract_bf16(b, b_q, b_scale_sh)
            c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
            _phase("pre_shared_module_call", path="mfma_medium")
            _module().mxfp4_mm_hip_mfma_medium(a_mfma, b_mfma, c)
            return c
        if regime == "medium_m":
            _phase("path_medium_scalar_hip")
            c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
            _phase("pre_shared_module_call", path="medium_scalar_hip")
            _module().mxfp4_mm_hip(a_in, b_in, c)
            return c
        if regime == "fallback":
            if a_in.shape[0] == 256:
                _phase("path_fallback_hip_256")
                c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
                _phase("pre_shared_module_call", path="fallback_hip_256")
                _module().mxfp4_mm_hip(a_in, b_in, c)
                return c
            _phase("path_fallback_torch_mm")
            return torch.mm(a_in, b_in.t()).to(torch.bfloat16)
        _phase("path_default_scalar_hip")
        c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
        _phase("pre_shared_module_call", path="default_scalar_hip")
        _module().mxfp4_mm_hip(a_in, b_in, c)
        return c
