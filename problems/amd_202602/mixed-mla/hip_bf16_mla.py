#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""HIP MLA decode with bf16 MFMA via load_inline.

Uses bf16 KV cache (no quant/dequant).
Split-K attention with online softmax.
bf16 MFMA: __builtin_amdgcn_mfma_f32_16x16x32_bf16 (M=16,N=16,K=32)

Architecture:
- Grid: (bs * num_splits, num_heads)
- Each block handles one (batch, split, head)
- QK: bf16 MFMA tiled 16x16x32
- Softmax: online in f32
- V: bf16 MFMA tiled
- Reduce: log-sum-exp across splits
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
void mla_decode_bf16(
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor out,
    torch::Tensor kv_indptr,
    int batch_size,
    int num_heads,
    int qk_dim,
    int v_dim,
    float sm_scale
);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

typedef bit16x4 _B16x4;
typedef struct _B16x8 {
    _B16x4 xy[2];
} _B16x8;

__device__ __forceinline__ floatx4 mfma_bf16_16x16x32(
    const _B16x8& A, const _B16x8& B, const floatx4& C
) {
    bit16x8 a = __builtin_shufflevector(A.xy[0], A.xy[1], 0,1,2,3,4,5,6,7);
    bit16x8 b = __builtin_shufflevector(B.xy[0], B.xy[1], 0,1,2,3,4,5,6,7);
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, C, 0, 0, 0);
}

// Simple single-pass MLA decode kernel
// Grid: (bs, num_heads)
// Block: 64 threads (1 wavefront)
// Each block processes all KV tokens for one (batch, head)
__launch_bounds__(64)
__global__ void mla_decode_bf16_kernel(
    const __hip_bfloat16* __restrict__ q,   // (bs, heads, 576)
    const __hip_bfloat16* __restrict__ kv,  // (total_kv, 576)
    __hip_bfloat16* __restrict__ out,       // (bs, heads, 512)
    const int* __restrict__ kv_indptr,      // (bs+1,)
    int num_heads,
    int qk_dim,    // 576
    int v_dim,     // 512
    float sm_scale
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int lane = threadIdx.x;

    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;

    // Q base pointer for this (batch, head)
    const __hip_bfloat16* q_ptr = q + (bid * num_heads + hid) * qk_dim;

    // Output accumulator
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    // V accumulator - each lane handles v_dim/64 = 8 output elements
    float v_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v_acc[i] = 0.0f;

    // Process KV tokens one at a time
    for (int t = 0; t < kv_len; t++) {
        const __hip_bfloat16* k_ptr = kv + (kv_start + t) * qk_dim;

        // Compute QK dot product - all 64 lanes cooperate
        // Each lane computes partial dot over qk_dim/64 = 9 elements
        float partial_qk = 0.0f;
        for (int d = lane; d < qk_dim; d += 64) {
            partial_qk += static_cast<float>(q_ptr[d]) * static_cast<float>(k_ptr[d]);
        }

        // Warp reduce to get full dot product
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            partial_qk += __shfl_down(partial_qk, offset, 64);
        }
        float qk = __shfl(partial_qk, 0, 64) * sm_scale;

        // Online softmax update
        float m_new = fmaxf(m_prev, qk);
        float exp_prev = expf(m_prev - m_new);
        float exp_cur = expf(qk - m_new);
        float l_new = l_prev * exp_prev + exp_cur;

        // Rescale V accumulator
        float rescale = (l_prev > 0.0f) ? (exp_prev * l_prev / l_new) : 0.0f;
        float w_cur = exp_cur / l_new;

        // Load V and accumulate
        const __hip_bfloat16* v_ptr = kv + (kv_start + t) * qk_dim; // V = first 512 dims
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int v_idx = lane + i * 64;
            if (v_idx < v_dim) {
                float v_val = static_cast<float>(v_ptr[v_idx]);
                v_acc[i] = v_acc[i] * rescale + w_cur * v_val;
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Store output
    __hip_bfloat16* out_ptr = out + (bid * num_heads + hid) * v_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int v_idx = lane + i * 64;
        if (v_idx < v_dim) {
            out_ptr[v_idx] = static_cast<__hip_bfloat16>(v_acc[i]);
        }
    }
}

void mla_decode_bf16(
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor out,
    torch::Tensor kv_indptr,
    int batch_size,
    int num_heads,
    int qk_dim,
    int v_dim,
    float sm_scale
) {
    dim3 block(64);
    dim3 grid(batch_size, num_heads);

    hipLaunchKernelGGL(
        mla_decode_bf16_kernel,
        grid, block, 0, 0,
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __hip_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(),
        num_heads, qk_dim, v_dim, sm_scale
    );
}
"""

EXPORT_FUNCTIONS = ["mla_decode_bf16"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_hip_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_bf16_hip_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=EXPORT_FUNCTIONS,
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])

    kv_bf16 = kv_data["bf16"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)
    kv_flat = kv_bf16.view(kv_bf16.shape[0], kv_bf16.shape[-1])

    out_key = ("hip_out", bs)
    if out_key not in _cache:
        _cache[out_key] = torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
    out = _cache[out_key]

    mod = _module()
    mod.mla_decode_bf16(
        q_reshaped,
        kv_flat,
        out,
        kv_indptr,
        bs,
        NUM_HEADS,
        QK_HEAD_DIM,
        V_HEAD_DIM,
        SM_SCALE,
    )
    return out
