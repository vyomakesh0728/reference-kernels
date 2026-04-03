#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Fused single-pass HIP kernel: online softmax, no split buffers, one kernel launch.

Key optimizations vs hip_bf16_mfma_wip.py:
1. Single kernel (no stage1 + reduce separation)
2. Online softmax in registers (no split buffer allocation)
3. Pre-allocated output buffer
4. Zero Python tensor ops in hot path
"""

from __future__ import annotations
import os
import sys

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

_HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cstdint>

typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v8s __attribute__((ext_vector_type(8)));

#define NH 16
#define QKD 576
#define VD 512
#define TILE 32

__device__ __forceinline__ float bf2f(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)b) << 16;
    return c.f;
}

__device__ __forceinline__ uint16_t f2bf(float v) {
    union { float f; uint32_t u; } c;
    c.f = v;
    c.u += 0x7FFF + ((c.u >> 16) & 1);
    return (uint16_t)(c.u >> 16);
}

__device__ __forceinline__ uint8_t f2fp8(float x) {
    if (x == 0.0f) return 0;
    union { float f; uint32_t u; } c; c.f = x;
    uint32_t s = (c.u >> 31) & 1;
    int e = (int)((c.u >> 23) & 0xFF) - 119;
    uint32_t m = (c.u >> 20) & 0x7;
    if (e <= 0) return 0;
    if (e >= 16) return (s << 7) | 0x7F;
    return (s << 7) | (e << 3) | m;
}

__device__ __forceinline__ float fp82f(uint8_t x) {
    if (x == 0 || x == 0x80) return 0.0f;
    uint32_t s = ((uint32_t)(x & 0x80)) << 24;
    uint32_t e = (x >> 3) & 0xF;
    uint32_t m = x & 0x7;
    if (e == 0) return 0.0f;
    uint32_t ieee = s | ((e + 119) << 23) | (m << 20);
    union { uint32_t u; float f; } c; c.u = ieee;
    return c.f;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int m = 32; m > 0; m >>= 1)
        v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int m = 32; m > 0; m >>= 1)
        v += __shfl_xor(v, m);
    return v;
}

// Fused single-pass MLA decode kernel
// Grid: (bs, 1, 1), Block: (64, 1, 1)
// Each block handles one batch item with all 16 heads
__global__ void mla_fused(
    const uint16_t* __restrict__ Q,
    const uint8_t* __restrict__ KV,
    float kv_scale,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ qo_indptr,
    uint16_t* __restrict__ Out,
    float sm_scale
) {
    // LDS for Q (fp8), KV tile, scores
    __shared__ uint8_t q8[NH * QKD];           // 9216 bytes
    __shared__ uint8_t kv_lds[TILE * (QKD+4)]; // ~18.5 KB (pad for alignment)
    __shared__ float scores[NH * TILE];         // 2048 bytes
    
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;  // 0-63
    const int g = tid >> 4;       // 0-3 (group of 16)
    const int l16 = tid & 15;     // 0-15
    
    // Get KV range for this batch
    const int k0 = kv_indptr[bid];
    const int k1 = kv_indptr[bid + 1];
    const int kv_len = k1 - k0;
    
    if (kv_len == 0) {
        // Empty sequence - zero output
        const int q0 = qo_indptr[bid];
        for (int h = 0; h < NH; h++) {
            for (int d = tid; d < VD; d += 64) {
                Out[(q0 * NH + h) * VD + d] = 0;
            }
        }
        return;
    }
    
    // Load and quantize Q to fp8
    const int q0 = qo_indptr[bid];
    const uint16_t* qp = Q + (int64_t)q0 * NH * QKD;
    
    // Find Q max for quantization
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += 64)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    amax = warp_reduce_max(amax);
    
    float q_scale = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_q_scale = 240.0f / fmaxf(amax, 1e-12f);
    
    // Quantize Q to fp8 in LDS
    for (int i = tid; i < NH * QKD; i += 64)
        q8[i] = f2fp8(bf2f(qp[i]) * inv_q_scale);
    __syncthreads();
    
    // Combined scale for QK scores
    float score_scale = q_scale * kv_scale * sm_scale;
    
    // Per-head online softmax state (4 heads per thread group, g selects which 4)
    float running_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float running_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // V accumulator: each thread accumulates part of V for its 4 heads
    // 512 V dims / 16 threads = 32 dims per thread, × 4 heads = 128 floats
    float v_acc[4][32];
    for (int h = 0; h < 4; h++)
        for (int d = 0; d < 32; d++)
            v_acc[h][d] = 0.0f;
    
    // Process KV in tiles of TILE tokens
    const int KVS = QKD + 4;  // stride with padding
    
    for (int ts = 0; ts < kv_len; ts += TILE) {
        int tile_size = min(TILE, kv_len - ts);
        
        // Load KV tile to LDS
        for (int i = tid; i < tile_size * QKD; i += 64) {
            int tok = i / QKD;
            int dim = i % QKD;
            kv_lds[tok * KVS + dim] = KV[(k0 + ts + tok) * QKD + dim];
        }
        // Zero padding for partial tiles
        if (tile_size < TILE) {
            for (int i = tid; i < (TILE - tile_size) * QKD; i += 64) {
                int tok = tile_size + i / QKD;
                int dim = i % QKD;
                kv_lds[tok * KVS + dim] = 0;
            }
        }
        __syncthreads();
        
        // Compute QK^T via MFMA for this tile
        // 16 heads × 32 tokens = 512 scores
        // Using MFMA 16x16x32 fp8: processes 16 rows × 16 cols × 32 K
        
        v4f qk1 = {0, 0, 0, 0};
        v4f qk2 = {0, 0, 0, 0};
        
        // 576 = 18 × 32, so 18 MFMA calls per head-tile pair
        for (int ch = 0; ch < 18; ch++) {
            int kb = ch * 32;
            // A = Q[head, k], B = KV[token, k]
            // MFMA layout: lane%16 = row for A, col for B
            long a = *(long*)(q8 + l16 * QKD + kb + g * 8);
            long b1 = *(long*)(kv_lds + l16 * KVS + kb + g * 8);
            long b2 = *(long*)(kv_lds + (16 + l16) * KVS + kb + g * 8);
            qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b1, qk1, 0, 0, 0);
            qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b2, qk2, 0, 0, 0);
        }
        
        // Store scores to LDS (need for softmax reduction)
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            scores[head * TILE + l16] = qk1[hi] * score_scale;
            scores[head * TILE + 16 + l16] = qk2[hi] * score_scale;
        }
        __syncthreads();
        
        // Online softmax update per head
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            
            // Find tile max
            float tile_max = -1e20f;
            for (int t = 0; t < tile_size; t++)
                tile_max = fmaxf(tile_max, scores[head * TILE + t]);
            
            // Update running max and compute correction factor
            float new_max = fmaxf(running_max[hi], tile_max);
            float correction = __expf(running_max[hi] - new_max);
            running_max[hi] = new_max;
            
            // Rescale previous V accumulator
            for (int d = 0; d < 32; d++)
                v_acc[hi][d] *= correction;
            running_sum[hi] *= correction;
            
            // Compute softmax weights and accumulate V
            float tile_sum = 0.0f;
            for (int t = 0; t < tile_size; t++) {
                float w = __expf(scores[head * TILE + t] - new_max);
                tile_sum += w;
                
                // Accumulate V: v_acc[hi][d] += w * V[t, dim]
                // V is at offset 0 in KV (first 512 dims)
                for (int d = 0; d < 32; d++) {
                    int vdim = l16 * 32 + d;
                    if (vdim < VD) {
                        float v_val = fp82f(kv_lds[t * KVS + vdim]) * kv_scale;
                        v_acc[hi][d] += w * v_val;
                    }
                }
            }
            running_sum[hi] += tile_sum;
        }
        __syncthreads();
    }
    
    // Final normalization and output
    for (int hi = 0; hi < 4; hi++) {
        int head = g * 4 + hi;
        float inv_sum = (running_sum[hi] > 0.0f) ? (1.0f / running_sum[hi]) : 0.0f;
        
        for (int d = 0; d < 32; d++) {
            int vdim = l16 * 32 + d;
            if (vdim < VD) {
                int out_idx = (q0 * NH + head) * VD + vdim;
                Out[out_idx] = f2bf(v_acc[hi][d] * inv_sum);
            }
        }
    }
}

torch::Tensor mla_fwd(
    torch::Tensor Q,
    torch::Tensor KV,
    double kv_scale,
    torch::Tensor kv_indptr,
    torch::Tensor qo_indptr,
    torch::Tensor Out,
    int64_t bs,
    double sm_scale
) {
    mla_fused<<<bs, 64>>>(
        (const uint16_t*)Q.data_ptr(),
        (const uint8_t*)KV.data_ptr(),
        (float)kv_scale,
        kv_indptr.data_ptr<int32_t>(),
        qo_indptr.data_ptr<int32_t>(),
        (uint16_t*)Out.data_ptr(),
        (float)sm_scale
    );
    return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_fwd", &mla_fwd, "Fused single-pass MLA decode");
}
"""

# Module-level state (survives across calls)
_hip_mod = None
_hip_failed = None
_out_cache = {}


def _get_hip():
    global _hip_mod, _hip_failed
    if _hip_failed:
        return None
    if _hip_mod is not None:
        return _hip_mod
    try:
        _hip_mod = load_inline(
            name="mla_fused_single_pass",
            cpp_sources="",
            cuda_sources=_HIP_SRC,
            extra_cuda_cflags=["-O3", "-ffast-math"],
            verbose=False,
        )
        _hip_failed = False
        return _hip_mod
    except Exception as e:
        _hip_failed = True
        print(f"[mla_fused] HIP JIT failed: {e}", file=sys.stderr)
        return None


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])

    mod = _get_hip()
    if mod is None:
        raise RuntimeError("HIP compilation failed")

    kv_fp8, kv_scale = kv_data["fp8"]
    dev = q.device

    # Pre-allocate output (cached by shape)
    out_key = (bs, dev)
    if out_key not in _out_cache:
        _out_cache[out_key] = torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
        )
    out = _out_cache[out_key]

    # Reshape KV to flat (total_kv, QKD)
    kv_flat = kv_fp8.view(-1, QK_HEAD_DIM)

    # Call fused kernel - single launch, no Python tensor ops
    mod.mla_fwd(
        q.view(-1, NUM_HEADS * QK_HEAD_DIM),  # flatten Q
        kv_flat,
        float(kv_scale.item()),
        kv_indptr.to(torch.int32),
        qo_indptr.to(torch.int32),
        out,
        bs,
        SM_SCALE,
    )

    return out
