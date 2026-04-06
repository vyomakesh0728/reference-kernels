#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Phase 1: Single-launch fused MLA with split-K + inline reduce.

Architecture:
- Single hipLaunchKernelGGL call (zero Python dispatch between stages)
- Grid: (bs * num_splits,) — each block handles ALL 16 heads for one (batch, split)
- Scalar QK + V (Phase 1 = correctness skeleton)
- Fused reduce: atomic counter, last block for each batch does log-sum-exp reduce
- fp8 KV with software dequant (2× bandwidth savings)

Phase 2 will add MFMA QK. Phase 3 will add MFMA V.
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

// Software fp8 E4M3 FNUZ → float conversion
// Bit layout: S(1) E(4) M(3), bias=8, 0x00→0, 0x80→NaN
__device__ __forceinline__ float fp8_to_f32(uint8_t x) {
    if (x == 0 || x == 0x80) return 0.0f;
    const uint32_t sign = (x >> 7);
    const uint32_t exp = (x >> 3) & 0xF;
    const uint32_t man = x & 0x7;
    float val;
    if (exp == 0) {
        val = ldexpf(static_cast<float>(man), -10);  // 2^(-7) * man/8
    } else {
        val = ldexpf(1.0f + static_cast<float>(man) * 0.125f, static_cast<int>(exp) - 8);
    }
    return sign ? -val : val;
}

__launch_bounds__(64)
__global__ void mla_fused_kernel(
    const __hip_bfloat16* __restrict__ q,    // (bs, 16, 576) bf16
    const uint8_t* __restrict__ kv,          // (total_kv, 1, 576) fp8
    const float* __restrict__ kv_scale,      // scalar
    __hip_bfloat16* __restrict__ out,        // (bs, 16, 512) bf16
    const int* __restrict__ kv_indptr,       // (bs+1,)
    float* __restrict__ split_buf,           // (bs, 16, num_splits, 512)
    float* __restrict__ split_lse,           // (bs, 16, num_splits)
    int* __restrict__ split_counter,         // (bs,) atomic
    int num_splits,
    float sm_scale
) {
    const int bid = blockIdx.x / num_splits;
    const int sid = blockIdx.x % num_splits;
    const int tid = threadIdx.x;

    const float scale = kv_scale[0];

    // KV range for this split
    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;
    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    // Process all 16 heads sequentially within this block
    for (int hid = 0; hid < N_HEADS; hid++) {
        const __hip_bfloat16* q_ptr = q + (bid * N_HEADS + hid) * QK_DIM;

        // Load Q into registers (9 elements per thread, stride 64)
        float q_reg[9];
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            int d = tid + i * WARP;
            q_reg[i] = (d < QK_DIM) ? static_cast<float>(q_ptr[d]) : 0.0f;
        }

        float m_prev = -1e30f;
        float l_prev = 0.0f;
        float v_acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) v_acc[i] = 0.0f;

        if (s_start < kv_len) {
            for (int t = s_start; t < s_end; t++) {
                const uint8_t* kv_ptr = kv + (kv_start + t) * QK_DIM;

                // QK dot product
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < 9; i++) {
                    int d = tid + i * WARP;
                    if (d < QK_DIM) {
                        partial += q_reg[i] * fp8_to_f32(kv_ptr[d]) * scale;
                    }
                }

                // Warp reduce
                #pragma unroll
                for (int off = 32; off > 0; off >>= 1) {
                    partial += __shfl_down(partial, off, WARP);
                }
                float qk = __shfl(partial, 0, WARP) * sm_scale;

                // Online softmax
                float m_new = fmaxf(m_prev, qk);
                float exp_diff = expf(m_prev - m_new);
                float exp_qk = expf(qk - m_new);
                float l_new = l_prev * exp_diff + exp_qk;

                // V accumulation
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = tid + i * WARP;
                    if (d < V_DIM) {
                        float v_val = fp8_to_f32(kv_ptr[d]) * scale;
                        v_acc[i] = v_acc[i] * exp_diff + exp_qk * v_val;
                    }
                }

                m_prev = m_new;
                l_prev = l_new;
            }
        }

        // Store partial result
        const int so_base = ((bid * N_HEADS + hid) * num_splits + sid) * V_DIM;
        const int sl_idx = (bid * N_HEADS + hid) * num_splits + sid;

        if (s_start >= kv_len || l_prev == 0.0f) {
            for (int i = 0; i < 8; i++) {
                int d = tid + i * WARP;
                if (d < V_DIM) split_buf[so_base + d] = 0.0f;
            }
            if (tid == 0) split_lse[sl_idx] = -1e30f;
        } else {
            float inv_l = 1.0f / l_prev;
            for (int i = 0; i < 8; i++) {
                int d = tid + i * WARP;
                if (d < V_DIM) split_buf[so_base + d] = v_acc[i] * inv_l;
            }
            if (tid == 0) split_lse[sl_idx] = m_prev + logf(l_prev);
        }
    }

    // Fused reduce: last block for this batch does the final combination
    __threadfence();
    __shared__ int is_last;
    if (tid == 0) {
        int old = atomicAdd(&split_counter[bid], 1);
        is_last = (old == num_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (is_last) {
        // This block reduces all splits for all 16 heads of this batch
        for (int hid = 0; hid < N_HEADS; hid++) {
            const int lse_base = (bid * N_HEADS + hid) * num_splits;
            const int out_base = (bid * N_HEADS + hid) * V_DIM;

            // Find max LSE across splits
            float max_lse = -1e30f;
            for (int s = 0; s < num_splits; s++) {
                max_lse = fmaxf(max_lse, split_lse[lse_base + s]);
            }

            // Weighted sum
            float acc[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = 0.0f;
            float sum_w = 0.0f;

            for (int s = 0; s < num_splits; s++) {
                float w = expf(split_lse[lse_base + s] - max_lse);
                sum_w += w;
                const float* partial = split_buf + ((bid * N_HEADS + hid) * num_splits + s) * V_DIM;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = tid + i * WARP;
                    if (d < V_DIM) acc[i] += w * partial[d];
                }
            }

            float inv_w = (sum_w > 0.0f) ? 1.0f / sum_w : 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = tid + i * WARP;
                if (d < V_DIM) {
                    out[out_base + d] = static_cast<__hip_bfloat16>(acc[i] * inv_w);
                }
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
    // Zero the counter
    hipMemsetAsync(split_counter.data_ptr<int>(), 0, batch_size * sizeof(int), 0);

    dim3 block(64);
    dim3 grid(batch_size * num_splits);
    hipLaunchKernelGGL(
        mla_fused_kernel,
        grid, block, 0, 0,
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
        build_root = Path(tempfile.gettempdir()) / "mla_fused_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_fused_{digest}",
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
    key = ("fused_bufs", bs, num_splits)
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

    # Reset counter to zero each call
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
