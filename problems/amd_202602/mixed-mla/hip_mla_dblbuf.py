#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Double-buffered KV loading for MLA decode.

Based on P10 with software pipelining:
- Two KV buffers in LDS (9216 bytes each)
- Prologue loads first block; loop prefetches next while computing current
- One sync per block instead of two
- 2 template instantiations (NS=16, NS=32) to keep compilation fast
- -O2 for compilation speed (kernel is memory-bound, -O3 barely helps)
"""

import os

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import hashlib
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

_cache = {}
_MODULE = None

CPP_WRAPPER = """
void mla_ns16(torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scale, torch::Tensor out, torch::Tensor kv_indptr, torch::Tensor sb, torch::Tensor sl, torch::Tensor sc, int bs, float sm);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int QK_DIM = 576;
constexpr int V_DIM = 512;
constexpr int N_HEADS = 16;
constexpr int WARP = 64;
constexpr int Q_PAD = 576;
constexpr int KV_STRIDE = 576;
constexpr int KV_BLOCK_BYTES = 16 * KV_STRIDE;
constexpr int VEC_BYTES = 16;

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef bit16x4 _B16x4;
typedef struct _B16x8 { _B16x4 xy[2]; } _B16x8;

template<int SEL>
__device__ __forceinline__ float hw_fp8(uint32_t pk) {
    return __builtin_amdgcn_cvt_f32_fp8(pk, SEL);
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    return static_cast<uint16_t>(c.u >> 16);
}

__device__ __forceinline__ float fp8_fast(uint8_t x) {
    if (x == 0) return 0.0f;
    uint32_t bits = (static_cast<uint32_t>(x & 0x80) << 24)
                  | ((static_cast<uint32_t>((x >> 3) & 0xF) + 119u) << 23)
                  | (static_cast<uint32_t>(x & 0x7) << 20);
    return __builtin_bit_cast(float, bits);
}

template<int NUM_SPLITS>
__launch_bounds__(64)
__global__ void mla_kernel(
    const __hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ kv_scale_ptr,
    __hip_bfloat16* __restrict__ out,
    const int* __restrict__ kv_indptr,
    float* __restrict__ split_buf,
    float* __restrict__ split_lse,
    int* __restrict__ split_counter,
    float sm_scale
) {
    const int bid = blockIdx.x / NUM_SPLITS;
    const int sid = blockIdx.x % NUM_SPLITS;
    const int lane = threadIdx.x;
    const int lane16 = lane & 15;
    const int group = lane >> 4;

    const float kv_scale = kv_scale_ptr[0];
    const float qk_scale = kv_scale * sm_scale;

    extern __shared__ char smem[];
    auto* q_lds = reinterpret_cast<uint16_t*>(smem);
    auto* kv_buf0 = reinterpret_cast<uint8_t*>(smem + N_HEADS * Q_PAD * 2);
    auto* kv_buf1 = kv_buf0 + KV_BLOCK_BYTES;
    auto* scores_lds = reinterpret_cast<float*>(smem + N_HEADS * Q_PAD * 2 + 2 * KV_BLOCK_BYTES);

    const uint16_t* q_src = reinterpret_cast<const uint16_t*>(q + bid * N_HEADS * QK_DIM);
    for (int i = lane; i < N_HEADS * QK_DIM; i += WARP)
        q_lds[i] = q_src[i];
    __syncthreads();

    const int kv_start = kv_indptr[bid];
    const int kv_len = kv_indptr[bid + 1] - kv_start;
    const int split_size = (kv_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    floatx4 v_acc[32];
    #pragma unroll
    for (int vt = 0; vt < 32; vt++)
        v_acc[vt] = {0.0f, 0.0f, 0.0f, 0.0f};

    float m_local[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    float l_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int num_blocks = (s_end > s_start) ? ((s_end - s_start + 15) / 16) : 0;

    if (num_blocks > 0) {
        // Prologue: load first block into buf[0]
        {
            const int tc = min(16, s_end - s_start);
            const uint8_t* src = &kv[(kv_start + s_start) * QK_DIM];
            #pragma unroll
            for (int i = 0; i < 9; i++) {
                int off = (i * WARP + lane) * VEC_BYTES;
                if (off + VEC_BYTES <= KV_BLOCK_BYTES) {
                    if (tc == 16 || (off / KV_STRIDE) < tc)
                        *reinterpret_cast<uint4*>(kv_buf0 + off) =
                            *reinterpret_cast<const uint4*>(src + off);
                    else
                        *reinterpret_cast<uint4*>(kv_buf0 + off) = {0, 0, 0, 0};
                }
            }
        }
        __syncthreads();

        int cur_buf = 0;
        for (int blk = 0; blk < num_blocks; blk++) {
            const int t_base = s_start + blk * 16;
            const int t_count = min(16, s_end - t_base);

            uint8_t* cur_kv = (cur_buf == 0) ? kv_buf0 : kv_buf1;
            uint8_t* nxt_kv = (cur_buf == 0) ? kv_buf1 : kv_buf0;

            // Prefetch next block
            if (blk + 1 < num_blocks) {
                const int nt = s_start + (blk + 1) * 16;
                const int ntc = min(16, s_end - nt);
                const uint8_t* src = &kv[(kv_start + nt) * QK_DIM];
                #pragma unroll
                for (int i = 0; i < 9; i++) {
                    int off = (i * WARP + lane) * VEC_BYTES;
                    if (off + VEC_BYTES <= KV_BLOCK_BYTES) {
                        if (ntc == 16 || (off / KV_STRIDE) < ntc)
                            *reinterpret_cast<uint4*>(nxt_kv + off) =
                                *reinterpret_cast<const uint4*>(src + off);
                        else
                            *reinterpret_cast<uint4*>(nxt_kv + off) = {0, 0, 0, 0};
                    }
                }
            }

            const bool valid = (lane16 < t_count);

            // MFMA QK
            floatx4 qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int tk = 0; tk < 18; tk++) {
                const int k_base = tk * 32 + 8 * group;
                const uint16_t* qp = &q_lds[lane16 * Q_PAD + k_base];
                _B16x8 a_reg;
                a_reg.xy[0] = {qp[0], qp[1], qp[2], qp[3]};
                a_reg.xy[1] = {qp[4], qp[5], qp[6], qp[7]};

                _B16x8 b_reg;
                if (valid) {
                    uint32_t pk0 = *reinterpret_cast<const uint32_t*>(&cur_kv[lane16 * KV_STRIDE + k_base]);
                    uint32_t pk1 = *reinterpret_cast<const uint32_t*>(&cur_kv[lane16 * KV_STRIDE + k_base + 4]);
                    b_reg.xy[0] = {f32_to_bf16_bits(hw_fp8<0>(pk0)), f32_to_bf16_bits(hw_fp8<1>(pk0)),
                                   f32_to_bf16_bits(hw_fp8<2>(pk0)), f32_to_bf16_bits(hw_fp8<3>(pk0))};
                    b_reg.xy[1] = {f32_to_bf16_bits(hw_fp8<0>(pk1)), f32_to_bf16_bits(hw_fp8<1>(pk1)),
                                   f32_to_bf16_bits(hw_fp8<2>(pk1)), f32_to_bf16_bits(hw_fp8<3>(pk1))};
                } else {
                    b_reg.xy[0] = {0,0,0,0}; b_reg.xy[1] = {0,0,0,0};
                }

                bit16x8 a = __builtin_shufflevector(a_reg.xy[0], a_reg.xy[1], 0,1,2,3,4,5,6,7);
                bit16x8 b = __builtin_shufflevector(b_reg.xy[0], b_reg.xy[1], 0,1,2,3,4,5,6,7);
                qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, qk_acc, 0, 0, 0);
            }

            // Softmax
            #pragma unroll
            for (int gpr = 0; gpr < 4; gpr++) {
                float qk = valid ? qk_acc[gpr] * qk_scale : -1e30f;
                float m_block = qk;
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1)
                    m_block = fmaxf(m_block, __shfl_xor(m_block, off));

                float m_old = m_local[gpr];
                float m_new = fmaxf(m_old, m_block);
                float alpha = expf(m_old - m_new);
                float p = valid ? expf(qk - m_new) : 0.0f;
                float sum_p = p;
                #pragma unroll
                for (int off = 8; off > 0; off >>= 1)
                    sum_p += __shfl_xor(sum_p, off);

                #pragma unroll
                for (int vt = 0; vt < 32; vt++)
                    v_acc[vt][gpr] *= alpha;

                m_local[gpr] = m_new;
                l_local[gpr] = l_local[gpr] * alpha + sum_p;
                scores_lds[(4 * group + gpr) * 16 + lane16] = p;
            }

            // MFMA V
            #pragma unroll
            for (int vt = 0; vt < 32; vt++) {
                const int v_base = vt * 16;
                _B16x4 a_v;
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    a_v[b] = f32_to_bf16_bits(scores_lds[lane16 * 16 + 4 * group + b]);

                _B16x4 b_v;
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    int tok_off = 4 * group + b;
                    b_v[b] = (tok_off < t_count)
                        ? f32_to_bf16_bits(fp8_fast(cur_kv[tok_off * KV_STRIDE + v_base + lane16]) * kv_scale)
                        : static_cast<uint16_t>(0);
                }
                v_acc[vt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_v, b_v, v_acc[vt], 0, 0, 0);
            }

            __syncthreads();
            cur_buf = 1 - cur_buf;
        }
    }

    // Store partial results
    #pragma unroll
    for (int gpr = 0; gpr < 4; gpr++) {
        int head = 4 * group + gpr;
        int so_base = ((bid * N_HEADS + head) * NUM_SPLITS + sid) * V_DIM;
        float inv_l = (l_local[gpr] > 0.0f) ? 1.0f / l_local[gpr] : 0.0f;
        #pragma unroll
        for (int vt = 0; vt < 32; vt++)
            split_buf[so_base + vt * 16 + lane16] = v_acc[vt][gpr] * inv_l;
        if (lane16 == 0)
            split_lse[(bid * N_HEADS + head) * NUM_SPLITS + sid] =
                (l_local[gpr] > 0.0f) ? m_local[gpr] + logf(l_local[gpr]) : -1e30f;
    }

    // Fused reduce
    __threadfence();
    __shared__ int is_last;
    if (lane == 0) {
        int old = atomicAdd(&split_counter[bid], 1);
        is_last = (old == NUM_SPLITS - 1) ? 1 : 0;
    }
    __syncthreads();

    if (is_last) {
        #pragma unroll
        for (int gpr = 0; gpr < 4; gpr++) {
            int head = 4 * group + gpr;
            int lse_base = (bid * N_HEADS + head) * NUM_SPLITS;
            int out_base = (bid * N_HEADS + head) * V_DIM;

            float max_lse = -1e30f;
            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; s++)
                max_lse = fmaxf(max_lse, split_lse[lse_base + s]);

            float sum_w = 0.0f;
            float acc[32];
            #pragma unroll
            for (int vt = 0; vt < 32; vt++) acc[vt] = 0.0f;

            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; s++) {
                float w = expf(split_lse[lse_base + s] - max_lse);
                sum_w += w;
                const float* p = split_buf + ((bid * N_HEADS + head) * NUM_SPLITS + s) * V_DIM;
                #pragma unroll
                for (int vt = 0; vt < 32; vt++)
                    acc[vt] += w * p[vt * 16 + lane16];
            }

            float inv_w = (sum_w > 0.0f) ? 1.0f / sum_w : 0.0f;
            #pragma unroll
            for (int vt = 0; vt < 32; vt++)
                out[out_base + vt * 16 + lane16] = static_cast<__hip_bfloat16>(acc[vt] * inv_w);
        }
    }
}

#define LAUNCH_MLA(NS) \
void mla_ns##NS(torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scale, \
    torch::Tensor out, torch::Tensor kv_indptr, \
    torch::Tensor sb, torch::Tensor sl, torch::Tensor sc, int bs, float sm) { \
    hipMemsetAsync(sc.data_ptr<int>(), 0, bs * sizeof(int), 0); \
    int smem = N_HEADS * Q_PAD * 2 + 2 * KV_BLOCK_BYTES + N_HEADS * 16 * sizeof(float); \
    hipLaunchKernelGGL(mla_kernel<NS>, dim3(bs * NS), dim3(64), smem, 0, \
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr<at::BFloat16>()), \
        reinterpret_cast<const uint8_t*>(kv.data_ptr()), \
        kv_scale.data_ptr<float>(), \
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), \
        kv_indptr.data_ptr<int>(), sb.data_ptr<float>(), sl.data_ptr<float>(), \
        sc.data_ptr<int>(), sm); \
}

LAUNCH_MLA(16)
"""

EXPORT_FUNCTIONS = ["mla_ns16"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_dblbuf_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_dblbuf_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=EXPORT_FUNCTIONS,
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O2"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _get_num_splits(bs, kvl):
    return 16


def _get_bufs(bs, num_splits, dev):
    key = ("dblbuf", bs, num_splits)
    if key not in _cache:
        _cache[key] = {
            "sb": torch.empty(
                bs * NUM_HEADS * num_splits * V_HEAD_DIM,
                dtype=torch.float32,
                device=dev,
            ),
            "sl": torch.empty(
                bs * NUM_HEADS * num_splits, dtype=torch.float32, device=dev
            ),
            "sc": torch.zeros(bs, dtype=torch.int32, device=dev),
            "out": torch.empty(
                (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
            ),
        }
    return _cache[key]


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)

    ns = _get_num_splits(bs, kvl)
    bufs = _get_bufs(bs, ns, q.device)
    bufs["sc"].zero_()

    mod = _module()
    mod.mla_ns16(
        q_reshaped,
        kv_fp8.view(-1),
        kv_scale,
        bufs["out"],
        kv_indptr,
        bufs["sb"],
        bufs["sl"],
        bufs["sc"],
        bs,
        SM_SCALE,
    )
    return bufs["out"]
