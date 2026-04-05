#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""HIP MLA with MFMA for QK, vectorized loads, multi-wavefront latency hiding.

Key optimizations over scalar kernel:
1. MFMA 16x16x32 for QK: 16 heads × 16 tokens × 32 dims per instruction
   - Eliminates warp reduction for QK (implicit in MFMA)
   - Processes 16 tokens in parallel
2. Q cached in LDS (loaded once, reused for all tokens)
3. V accumulation via per-lane accumulator + 16-way shuffle reduce
4. Aggressive split-K for GPU saturation

MFMA register mapping (mfma_f32_16x16x32_bf16):
  A[i][k]: lane%16=i(head), k=8*(lane/16)+4*gpr+byte
  B[k][j]: lane%16=j(token), k=8*(lane/16)+4*gpr+byte
  D[i][j]: lane%16=j(token), i=4*(lane/16)+gpr
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
void mla_stage1(
    torch::Tensor q, torch::Tensor kv,
    torch::Tensor split_out, torch::Tensor split_lse,
    torch::Tensor kv_indptr,
    int batch_size, int num_splits, float sm_scale
);
void mla_reduce(
    torch::Tensor split_out, torch::Tensor split_lse,
    torch::Tensor out, int batch_size, int num_splits
);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int QK_DIM = 576;
constexpr int V_DIM = 512;
constexpr int N_HEADS = 16;

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

typedef bit16x4 _B16x4;
typedef struct _B16x8 { _B16x4 xy[2]; } _B16x8;

// Q stored in LDS: 16 heads × 576 dims × 2 bytes = 18432 bytes
// V accumulator in LDS: 16 heads × 512 dims × 4 bytes = 32768 bytes
// Softmax state in LDS: 16 heads × 2 floats × 4 bytes = 128 bytes
// Total: ~51KB

__launch_bounds__(64)
__global__ void mla_stage1_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ kv,
    float* __restrict__ split_out,
    float* __restrict__ split_lse,
    const int* __restrict__ kv_indptr,
    int num_splits,
    float sm_scale
) {
    const int bid = blockIdx.x / num_splits;
    const int sid = blockIdx.x % num_splits;
    const int lane = threadIdx.x;
    const int lane16 = lane & 15;
    const int group = lane >> 4;

    extern __shared__ char smem[];
    auto* q_lds = reinterpret_cast<__hip_bfloat16*>(smem);
    // q_lds: [16][576] = 18432 bytes, but pad to [16][580] for bank conflicts
    constexpr int Q_STRIDE = 580;
    auto* v_acc = reinterpret_cast<float*>(smem + N_HEADS * Q_STRIDE * sizeof(__hip_bfloat16));
    // v_acc: [16][512] = 32768 bytes
    auto* sm_state = reinterpret_cast<float*>(smem + N_HEADS * Q_STRIDE * sizeof(__hip_bfloat16) + N_HEADS * V_DIM * sizeof(float));
    // sm_state: [16][2] = 128 bytes (m, l)

    // Load Q into LDS
    const __hip_bfloat16* q_base = q + bid * N_HEADS * QK_DIM;
    for (int i = lane; i < N_HEADS * QK_DIM; i += 64) {
        int h = i / QK_DIM;
        int d = i % QK_DIM;
        q_lds[h * Q_STRIDE + d] = q_base[i];
    }

    // Init V accumulator and softmax state
    for (int i = lane; i < N_HEADS * V_DIM; i += 64) {
        v_acc[i] = 0.0f;
    }
    if (lane < N_HEADS) {
        sm_state[lane * 2] = -1e30f;
        sm_state[lane * 2 + 1] = 0.0f;
    }
    __syncthreads();

    // KV range
    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;
    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    if (s_start >= kv_len) {
        // Empty split
        for (int i = lane; i < N_HEADS * V_DIM; i += 64) {
            int h = i / V_DIM;
            int d = i % V_DIM;
            split_out[((bid * N_HEADS + h) * num_splits + sid) * V_DIM + d] = 0.0f;
        }
        if (lane < N_HEADS) {
            split_lse[(bid * N_HEADS + lane) * num_splits + sid] = -1e30f;
        }
        return;
    }

    // Process 16 tokens at a time with MFMA
    for (int t_base = s_start; t_base < s_end; t_base += 16) {
        const int t_count = min(16, s_end - t_base);
        const bool my_token_valid = (lane16 < t_count);
        const int my_token = kv_start + t_base + lane16;

        // ===== QK via MFMA 16x16x32 =====
        floatx4 qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int tile_k = 0; tile_k < QK_DIM; tile_k += 32) {
            // Load A (Q from LDS): lane%16=head, 8 bf16 from Q[head, tile_k+8*group+0..7]
            _B16x8 a_reg;
            const int k_base = tile_k + 8 * group;
            if (k_base + 7 < QK_DIM) {
                const uint16_t* qp = reinterpret_cast<const uint16_t*>(&q_lds[lane16 * Q_STRIDE + k_base]);
                a_reg.xy[0] = {qp[0], qp[1], qp[2], qp[3]};
                a_reg.xy[1] = {qp[4], qp[5], qp[6], qp[7]};
            } else {
                a_reg.xy[0] = {0, 0, 0, 0};
                a_reg.xy[1] = {0, 0, 0, 0};
                for (int i = 0; i < 8 && k_base + i < QK_DIM; i++) {
                    uint16_t v = reinterpret_cast<const uint16_t*>(&q_lds[lane16 * Q_STRIDE + k_base + i])[0];
                    if (i < 4) a_reg.xy[0][i] = v; else a_reg.xy[1][i - 4] = v;
                }
            }

            // Load B (K from global): lane%16=token, 8 bf16 from K[token, tile_k+8*group+0..7]
            _B16x8 b_reg;
            if (my_token_valid && k_base + 7 < QK_DIM) {
                const uint16_t* kp = reinterpret_cast<const uint16_t*>(&kv[my_token * QK_DIM + k_base]);
                b_reg.xy[0] = {kp[0], kp[1], kp[2], kp[3]};
                b_reg.xy[1] = {kp[4], kp[5], kp[6], kp[7]};
            } else if (my_token_valid) {
                b_reg.xy[0] = {0, 0, 0, 0};
                b_reg.xy[1] = {0, 0, 0, 0};
                for (int i = 0; i < 8 && k_base + i < QK_DIM; i++) {
                    uint16_t v = reinterpret_cast<const uint16_t*>(&kv[my_token * QK_DIM + k_base + i])[0];
                    if (i < 4) b_reg.xy[0][i] = v; else b_reg.xy[1][i - 4] = v;
                }
            } else {
                b_reg.xy[0] = {0, 0, 0, 0};
                b_reg.xy[1] = {0, 0, 0, 0};
            }

            bit16x8 a = __builtin_shufflevector(a_reg.xy[0], a_reg.xy[1], 0,1,2,3,4,5,6,7);
            bit16x8 b = __builtin_shufflevector(b_reg.xy[0], b_reg.xy[1], 0,1,2,3,4,5,6,7);
            qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, qk_acc, 0, 0, 0);
        }

        // qk_acc[gpr] at lane l = QK[head=4*(l/16)+gpr, token=l%16]
        // Apply sm_scale and mask invalid tokens
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            qk_acc[g] = my_token_valid ? qk_acc[g] * sm_scale : -1e30f;
        }

        // ===== Online softmax per head + V accumulation =====
        // Process all 4 heads sequentially (to reuse V loads)
        #pragma unroll
        for (int gpr = 0; gpr < 4; gpr++) {
            const int head = 4 * group + gpr;
            float qk_val = qk_acc[gpr];

            // Max across 16 tokens
            float max_qk = qk_val;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                max_qk = fmaxf(max_qk, __shfl_xor(max_qk, off));
            }

            float m_old = sm_state[head * 2];
            float l_old = sm_state[head * 2 + 1];
            float m_new = fmaxf(m_old, max_qk);
            float exp_diff = expf(m_old - m_new);
            float p = my_token_valid ? expf(qk_val - m_new) : 0.0f;

            float sum_p = p;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                sum_p += __shfl_xor(sum_p, off);
            }
            float l_new = l_old * exp_diff + sum_p;

            // Rescale existing V accumulator
            for (int d = lane16; d < V_DIM; d += 16) {
                v_acc[head * V_DIM + d] *= exp_diff;
            }

            // V accumulation via shuffle reduce (no atomics!)
            // Each lane has p for its token. Load V[my_token, d], multiply by p,
            // reduce across 16 lanes to get sum over tokens.
            if (my_token_valid) {
                const __hip_bfloat16* v_ptr = &kv[my_token * QK_DIM];
                for (int d = 0; d < V_DIM; d += 16) {
                    // Each lane loads V[my_token, d + lane16]
                    int vd = d + lane16;
                    float pv = (vd < V_DIM) ? p * static_cast<float>(v_ptr[vd]) : 0.0f;

                    // Reduce across 16 tokens
                    #pragma unroll
                    for (int off = 8; off > 0; off >>= 1) {
                        pv += __shfl_xor(pv, off);
                    }
                    // All lanes in this group now have the sum
                    if (vd < V_DIM) {
                        v_acc[head * V_DIM + vd] += pv;
                    }
                }
            } else {
                // Invalid tokens contribute 0, still need to participate in shuffles
                for (int d = 0; d < V_DIM; d += 16) {
                    float pv = 0.0f;
                    #pragma unroll
                    for (int off = 8; off > 0; off >>= 1) {
                        pv += __shfl_xor(pv, off);
                    }
                    int vd = d + lane16;
                    if (vd < V_DIM) {
                        v_acc[head * V_DIM + vd] += pv;
                    }
                }
            }

            if (lane16 == 0) {
                sm_state[head * 2] = m_new;
                sm_state[head * 2 + 1] = l_new;
            }
        }
        __syncthreads();
    }

    // Normalize and store
    __syncthreads();
    for (int i = lane; i < N_HEADS * V_DIM; i += 64) {
        int h = i / V_DIM;
        int d = i % V_DIM;
        float l = sm_state[h * 2 + 1];
        float val = (l > 0.0f) ? v_acc[i] / l : 0.0f;
        split_out[((bid * N_HEADS + h) * num_splits + sid) * V_DIM + d] = val;
    }
    if (lane < N_HEADS) {
        float m = sm_state[lane * 2];
        float l = sm_state[lane * 2 + 1];
        split_lse[(bid * N_HEADS + lane) * num_splits + sid] = m + logf(fmaxf(l, 1e-30f));
    }
}

__launch_bounds__(64)
__global__ void mla_reduce_kernel(
    const float* __restrict__ split_out,
    const float* __restrict__ split_lse,
    __hip_bfloat16* __restrict__ out,
    int num_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;

    const int lse_base = (bid * N_HEADS + hid) * num_splits;
    const int out_base = (bid * N_HEADS + hid) * V_DIM;

    float max_lse = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        max_lse = fmaxf(max_lse, split_lse[lse_base + s]);
    }

    constexpr int DPT = (V_DIM + 63) / 64;
    float acc[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; i++) acc[i] = 0.0f;
    float sum_w = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float w = expf(split_lse[lse_base + s] - max_lse);
        sum_w += w;
        const float* p = split_out + ((bid * N_HEADS + hid) * num_splits + s) * V_DIM;
        #pragma unroll
        for (int i = 0; i < DPT; i++) {
            int d = tid + i * 64;
            if (d < V_DIM) acc[i] += w * p[d];
        }
    }

    float inv = 1.0f / sum_w;
    #pragma unroll
    for (int i = 0; i < DPT; i++) {
        int d = tid + i * 64;
        if (d < V_DIM) out[out_base + d] = static_cast<__hip_bfloat16>(acc[i] * inv);
    }
}

void mla_stage1(
    torch::Tensor q, torch::Tensor kv,
    torch::Tensor split_out, torch::Tensor split_lse,
    torch::Tensor kv_indptr,
    int batch_size, int num_splits, float sm_scale
) {
    constexpr int Q_STRIDE = 580;
    int smem_bytes = N_HEADS * Q_STRIDE * sizeof(at::BFloat16)
                   + N_HEADS * V_DIM * sizeof(float)
                   + N_HEADS * 2 * sizeof(float);

    dim3 block(64);
    dim3 grid(batch_size * num_splits);
    hipLaunchKernelGGL(
        mla_stage1_kernel,
        grid, block, smem_bytes, 0,
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __hip_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        split_out.data_ptr<float>(),
        split_lse.data_ptr<float>(),
        kv_indptr.data_ptr<int>(),
        num_splits, sm_scale
    );
}

void mla_reduce(
    torch::Tensor split_out, torch::Tensor split_lse,
    torch::Tensor out, int batch_size, int num_splits
) {
    dim3 block(64);
    dim3 grid(batch_size, N_HEADS);
    hipLaunchKernelGGL(
        mla_reduce_kernel,
        grid, block, 0, 0,
        split_out.data_ptr<float>(),
        split_lse.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        num_splits
    );
}
"""

EXPORT_FUNCTIONS = ["mla_stage1", "mla_reduce"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_mfma_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_mfma_{digest}",
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
        return 4
    if total <= 4096:
        return 8
    if total <= 16384:
        return 16
    return 32


def _get_bufs(bs, num_splits, dev):
    key = ("mfma_bufs", bs, num_splits)
    if key not in _cache:
        _cache[key] = (
            torch.empty(
                (bs, NUM_HEADS, num_splits, V_HEAD_DIM), dtype=torch.float32, device=dev
            ),
            torch.empty((bs, NUM_HEADS, num_splits), dtype=torch.float32, device=dev),
            torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev),
        )
    return _cache[key]


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_bf16 = kv_data["bf16"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
    kv_flat = kv_bf16.view(kv_bf16.shape[0], kv_bf16.shape[-1])

    num_splits = _get_num_splits(bs, kvl)
    split_out, split_lse, out = _get_bufs(bs, num_splits, q.device)

    mod = _module()
    mod.mla_stage1(
        q_reshaped,
        kv_flat,
        split_out,
        split_lse,
        kv_indptr,
        bs,
        num_splits,
        SM_SCALE,
    )
    mod.mla_reduce(split_out, split_lse, out, bs, num_splits)
    return out
