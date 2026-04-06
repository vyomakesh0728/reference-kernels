#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Phase 2: MFMA QK + scalar V, single-launch fused kernel.

Changes from Phase 1:
- MFMA 16x16x32_bf16 for QK (18 calls per 16-token block)
- All 16 heads computed simultaneously in one MFMA
- QK scores stored to LDS, then V done head-sequential
- fp8 K → bf16 conversion in registers for MFMA input
- Q cached in LDS as bf16

Architecture per 16-token block:
1. MFMA QK: Q[16 heads, 32 dims] × K[32 dims, 16 tokens] → scores[16,16]
2. Store scores to LDS[16][16]
3. For each head: online softmax + scalar V accumulation using LDS scores
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
void mla_fused(
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scale,
    torch::Tensor out, torch::Tensor kv_indptr,
    torch::Tensor split_buf, torch::Tensor split_lse_buf,
    torch::Tensor split_counter,
    int batch_size, int num_splits, float sm_scale
);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int QK_DIM = 576;
constexpr int V_DIM = 512;
constexpr int N_HEADS = 16;
constexpr int WARP = 64;
constexpr int Q_PAD = 580;  // padded for LDS bank conflicts

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef bit16x4 _B16x4;
typedef struct _B16x8 { _B16x4 xy[2]; } _B16x8;

__device__ __forceinline__ float fp8_to_f32(uint8_t x) {
    if (x == 0 || x == 0x80) return 0.0f;
    const uint32_t sign = (x >> 7);
    const uint32_t exp = (x >> 3) & 0xF;
    const uint32_t man = x & 0x7;
    float val;
    if (exp == 0) {
        val = ldexpf(static_cast<float>(man), -10);
    } else {
        val = ldexpf(1.0f + static_cast<float>(man) * 0.125f, static_cast<int>(exp) - 8);
    }
    return sign ? -val : val;
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    union { float f; uint32_t u; } conv;
    conv.f = f;
    return static_cast<uint16_t>(conv.u >> 16);
}

// LDS layout:
// q_lds:      [16][Q_PAD] bf16 = 18560 bytes
// scores_lds: [16][16] float = 1024 bytes
// Total: ~19.6KB

__launch_bounds__(64)
__global__ void mla_fused_kernel(
    const __hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ kv_scale_ptr,
    __hip_bfloat16* __restrict__ out,
    const int* __restrict__ kv_indptr,
    float* __restrict__ split_buf,
    float* __restrict__ split_lse,
    int* __restrict__ split_counter,
    int num_splits,
    float sm_scale
) {
    const int bid = blockIdx.x / num_splits;
    const int sid = blockIdx.x % num_splits;
    const int lane = threadIdx.x;
    const int lane16 = lane & 15;
    const int group = lane >> 4;

    const float kv_scale = kv_scale_ptr[0];

    extern __shared__ char smem[];
    auto* q_lds = reinterpret_cast<uint16_t*>(smem);  // bf16 as uint16
    auto* scores_lds = reinterpret_cast<float*>(smem + N_HEADS * Q_PAD * 2);

    // Load Q into LDS
    const __hip_bfloat16* q_base = q + bid * N_HEADS * QK_DIM;
    for (int i = lane; i < N_HEADS * QK_DIM; i += WARP) {
        int h = i / QK_DIM;
        int d = i % QK_DIM;
        q_lds[h * Q_PAD + d] = reinterpret_cast<const uint16_t*>(q_base)[i];
    }
    __syncthreads();

    // KV range
    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;
    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    // Per-head state (sequential V accumulation)
    float m_state[N_HEADS], l_state[N_HEADS];
    float v_acc[N_HEADS][8];
    for (int h = 0; h < N_HEADS; h++) {
        m_state[h] = -1e30f;
        l_state[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) v_acc[h][i] = 0.0f;
    }

    // Process 16 tokens at a time
    for (int t_base = s_start; t_base < s_end; t_base += 16) {
        const int t_count = min(16, s_end - t_base);
        const bool my_token_valid = (lane16 < t_count);
        const int my_token = kv_start + t_base + lane16;

        // ===== MFMA QK: 18 tiles of K=32 =====
        floatx4 qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int tile_k = 0; tile_k < QK_DIM; tile_k += 32) {
            const int k_base = tile_k + 8 * group;

            // Load A (Q from LDS): lane%16 = head index
            _B16x8 a_reg;
            if (k_base + 7 < QK_DIM) {
                const uint16_t* qp = &q_lds[lane16 * Q_PAD + k_base];
                a_reg.xy[0] = {qp[0], qp[1], qp[2], qp[3]};
                a_reg.xy[1] = {qp[4], qp[5], qp[6], qp[7]};
            } else {
                a_reg.xy[0] = {0, 0, 0, 0};
                a_reg.xy[1] = {0, 0, 0, 0};
                for (int i = 0; i < 8; i++) {
                    if (k_base + i < QK_DIM) {
                        uint16_t v = q_lds[lane16 * Q_PAD + k_base + i];
                        if (i < 4) a_reg.xy[0][i] = v; else a_reg.xy[1][i - 4] = v;
                    }
                }
            }

            // Load B (K from global as fp8 → bf16): lane%16 = token index
            _B16x8 b_reg;
            if (my_token_valid && k_base + 7 < QK_DIM) {
                const uint8_t* kp = &kv[my_token * QK_DIM + k_base];
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    b_reg.xy[0][i] = f32_to_bf16_bits(fp8_to_f32(kp[i]));
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    b_reg.xy[1][i] = f32_to_bf16_bits(fp8_to_f32(kp[4 + i]));
            } else if (my_token_valid) {
                b_reg.xy[0] = {0, 0, 0, 0};
                b_reg.xy[1] = {0, 0, 0, 0};
                for (int i = 0; i < 8; i++) {
                    if (k_base + i < QK_DIM) {
                        uint16_t v = f32_to_bf16_bits(fp8_to_f32(kv[my_token * QK_DIM + k_base + i]));
                        if (i < 4) b_reg.xy[0][i] = v; else b_reg.xy[1][i - 4] = v;
                    }
                }
            } else {
                b_reg.xy[0] = {0, 0, 0, 0};
                b_reg.xy[1] = {0, 0, 0, 0};
            }

            bit16x8 a = __builtin_shufflevector(a_reg.xy[0], a_reg.xy[1], 0,1,2,3,4,5,6,7);
            bit16x8 b = __builtin_shufflevector(b_reg.xy[0], b_reg.xy[1], 0,1,2,3,4,5,6,7);
            qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, qk_acc, 0, 0, 0);
        }

        // Store QK scores to LDS: qk_acc[gpr] = score[head=4*group+gpr, token=lane16]
        // Apply kv_scale here (MFMA computed raw Q@K^T, need to scale by kv_scale)
        const float combined_scale = kv_scale * sm_scale;
        #pragma unroll
        for (int gpr = 0; gpr < 4; gpr++) {
            int head = 4 * group + gpr;
            float score = my_token_valid ? qk_acc[gpr] * combined_scale : -1e30f;
            scores_lds[head * 16 + lane16] = score;
        }
        __syncthreads();

        // ===== Scalar V accumulation, head-sequential =====
        for (int h = 0; h < N_HEADS; h++) {
            for (int t_off = 0; t_off < t_count; t_off++) {
                float score = scores_lds[h * 16 + t_off];
                int token = kv_start + t_base + t_off;

                // Online softmax
                float m_new = fmaxf(m_state[h], score);
                float exp_diff = expf(m_state[h] - m_new);
                float exp_score = expf(score - m_new);
                float l_new = l_state[h] * exp_diff + exp_score;

                // V accumulation with rescaling
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = lane + i * WARP;
                    if (d < V_DIM) {
                        float v_val = fp8_to_f32(kv[token * QK_DIM + d]) * kv_scale;
                        v_acc[h][i] = v_acc[h][i] * exp_diff + exp_score * v_val;
                    }
                }

                m_state[h] = m_new;
                l_state[h] = l_new;
            }
        }
        __syncthreads();
    }

    // Store partial results
    for (int h = 0; h < N_HEADS; h++) {
        const int so_base = ((bid * N_HEADS + h) * num_splits + sid) * V_DIM;
        const int sl_idx = (bid * N_HEADS + h) * num_splits + sid;

        if (s_start >= kv_len || l_state[h] == 0.0f) {
            for (int i = 0; i < 8; i++) {
                int d = lane + i * WARP;
                if (d < V_DIM) split_buf[so_base + d] = 0.0f;
            }
            if (lane == 0) split_lse[sl_idx] = -1e30f;
        } else {
            float inv_l = 1.0f / l_state[h];
            for (int i = 0; i < 8; i++) {
                int d = lane + i * WARP;
                if (d < V_DIM) split_buf[so_base + d] = v_acc[h][i] * inv_l;
            }
            if (lane == 0) split_lse[sl_idx] = m_state[h] + logf(l_state[h]);
        }
    }

    // Fused reduce
    __threadfence();
    __shared__ int is_last;
    if (lane == 0) {
        int old = atomicAdd(&split_counter[bid], 1);
        is_last = (old == num_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (is_last) {
        for (int h = 0; h < N_HEADS; h++) {
            const int lse_base = (bid * N_HEADS + h) * num_splits;
            const int out_base = (bid * N_HEADS + h) * V_DIM;

            float max_lse = -1e30f;
            for (int s = 0; s < num_splits; s++)
                max_lse = fmaxf(max_lse, split_lse[lse_base + s]);

            float acc[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = 0.0f;
            float sum_w = 0.0f;

            for (int s = 0; s < num_splits; s++) {
                float w = expf(split_lse[lse_base + s] - max_lse);
                sum_w += w;
                const float* p = split_buf + ((bid * N_HEADS + h) * num_splits + s) * V_DIM;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = lane + i * WARP;
                    if (d < V_DIM) acc[i] += w * p[d];
                }
            }

            float inv_w = (sum_w > 0.0f) ? 1.0f / sum_w : 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * WARP;
                if (d < V_DIM)
                    out[out_base + d] = static_cast<__hip_bfloat16>(acc[i] * inv_w);
            }
        }
    }
}

void mla_fused(
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scale,
    torch::Tensor out, torch::Tensor kv_indptr,
    torch::Tensor split_buf, torch::Tensor split_lse_buf,
    torch::Tensor split_counter,
    int batch_size, int num_splits, float sm_scale
) {
    hipMemsetAsync(split_counter.data_ptr<int>(), 0, batch_size * sizeof(int), 0);

    int smem_bytes = N_HEADS * Q_PAD * 2 + N_HEADS * 16 * sizeof(float);
    dim3 block(64);
    dim3 grid(batch_size * num_splits);
    hipLaunchKernelGGL(
        mla_fused_kernel,
        grid, block, smem_bytes, 0,
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        kv_scale.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(),
        split_buf.data_ptr<float>(),
        split_lse_buf.data_ptr<float>(),
        split_counter.data_ptr<int>(),
        num_splits, sm_scale
    );
}
"""

EXPORT_FUNCTIONS = ["mla_fused"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_p2_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_p2_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=EXPORT_FUNCTIONS,
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _get_num_splits(bs, kvl):
    total = bs * kvl
    if total <= 256:
        return 1
    if total <= 1024:
        return 2
    if total <= 4096:
        return 4
    if total <= 16384:
        return 8
    return 16


def _get_bufs(bs, num_splits, dev):
    key = ("p2_bufs", bs, num_splits)
    if key not in _cache:
        _cache[key] = {
            "split_buf": torch.empty(
                bs * NUM_HEADS * num_splits * V_HEAD_DIM,
                dtype=torch.float32,
                device=dev,
            ),
            "split_lse": torch.empty(
                bs * NUM_HEADS * num_splits, dtype=torch.float32, device=dev
            ),
            "counter": torch.zeros(bs, dtype=torch.int32, device=dev),
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

    num_splits = _get_num_splits(bs, kvl)
    bufs = _get_bufs(bs, num_splits, q.device)
    bufs["counter"].zero_()

    mod = _module()
    mod.mla_fused(
        q_reshaped,
        kv_fp8.view(-1),
        kv_scale,
        bufs["out"],
        kv_indptr,
        bufs["split_buf"],
        bufs["split_lse"],
        bufs["counter"],
        bs,
        num_splits,
        SM_SCALE,
    )
    return bufs["out"]
