#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""HIP kernel: Correct V accumulation with MFMA QK + scalar V.
Focus on correctness first, then optimize.
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

#define NH 16
#define QKD 576
#define VD 512
#define TILE_K 32
#define KVS 580  // Padded stride

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

__device__ __forceinline__ float fp8_to_f32(uint8_t x) {
    if ((x & 0x7F) == 0) return 0.0f;
    union { uint32_t u; float f; } c;
    uint32_t s = ((uint32_t)(x & 0x80)) << 24;
    uint32_t e = (x >> 3) & 0xF;
    uint32_t m = x & 0x7;
    if (e == 0) return 0.0f;
    c.u = s | ((e + 120) << 23) | (m << 20);
    return c.f;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int m = 32; m > 0; m >>= 1) v += __shfl_xor(v, m);
    return v;
}

// Grid: (num_splits, batch_size), Block: 64 threads (1 wave)
__global__ void mla_stage1(
    const uint8_t* __restrict__ Q,
    float q_scale,
    const uint8_t* __restrict__ KV,
    float kv_scale,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ qo_indptr,
    float* __restrict__ splitO,
    float* __restrict__ splitLse,
    int num_splits,
    float sm_scale
) {
    __shared__ uint8_t lds_q[NH * QKD];
    __shared__ uint8_t lds_kv[TILE_K * KVS];
    __shared__ float lds_scores[NH * TILE_K];
    
    const int tid = threadIdx.x;
    const int l16 = tid & 15;
    const int g = tid >> 4;
    
    const int split_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    
    const int kv_start = kv_indptr[batch_id];
    const int kv_end = kv_indptr[batch_id + 1];
    const int kv_len = kv_end - kv_start;
    
    const int split_len = (kv_len + num_splits - 1) / num_splits;
    const int my_start = kv_start + split_id * split_len;
    const int my_end = min(my_start + split_len, kv_end);
    
    // Output base indices
    const int out_base_lse = (batch_id * num_splits + split_id) * NH;
    const int out_base_o = out_base_lse * VD;
    
    if (my_start >= kv_end) {
        for (int h = tid; h < NH; h += 64)
            splitLse[out_base_lse + h] = -1e20f;
        for (int i = tid; i < NH * VD; i += 64)
            splitO[out_base_o + i] = 0.0f;
        return;
    }
    
    // Load Q
    const int q_idx = qo_indptr[batch_id];
    const uint8_t* qp = Q + (int64_t)q_idx * NH * QKD;
    for (int i = tid; i < NH * QKD; i += 64) lds_q[i] = qp[i];
    __syncthreads();
    
    // Each thread handles 4 heads (g*4 to g*4+3)
    float row_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // V accumulators: 4 heads × 8 dims per thread (512/64 = 8)
    float v_acc[4][8];
    for (int h = 0; h < 4; h++)
        for (int d = 0; d < 8; d++)
            v_acc[h][d] = 0.0f;
    
    const float qk_scale = q_scale * kv_scale * sm_scale;
    
    for (int tile_start = my_start; tile_start < my_end; tile_start += TILE_K) {
        int tile_len = min(TILE_K, my_end - tile_start);
        
        // Load KV
        for (int i = tid; i < tile_len * QKD; i += 64) {
            int tok = i / QKD, dim = i % QKD;
            lds_kv[tok * KVS + dim] = KV[(tile_start + tok) * QKD + dim];
        }
        if (tile_len < TILE_K) {
            for (int i = tid; i < (TILE_K - tile_len) * QKD; i += 64) {
                int tok = tile_len + i / QKD, dim = i % QKD;
                lds_kv[tok * KVS + dim] = 0;
            }
        }
        __syncthreads();
        
        // MFMA QK^T (16 heads × 32 tokens)
        v4f qk1 = {0, 0, 0, 0};
        v4f qk2 = {0, 0, 0, 0};
        
        for (int k = 0; k < 18; k++) {
            int kb = k * 32;
            long q_data = *(long*)(lds_q + l16 * QKD + kb + g * 8);
            long kv0 = *(long*)(lds_kv + l16 * KVS + kb + g * 8);
            long kv1 = *(long*)(lds_kv + (l16 + 16) * KVS + kb + g * 8);
            qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(q_data, kv0, qk1, 0, 0, 0);
            qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(q_data, kv1, qk2, 0, 0, 0);
        }
        
        // Write scores: MFMA layout D[i][j] where i=4*g+gpr, j=l16
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            float s0 = qk1[hi] * qk_scale;
            float s1 = qk2[hi] * qk_scale;
            lds_scores[head * TILE_K + l16] = (l16 < tile_len) ? s0 : -1e20f;
            lds_scores[head * TILE_K + l16 + 16] = (l16 + 16 < tile_len) ? s1 : -1e20f;
        }
        __syncthreads();
        
        // Softmax + V accumulation per head
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            
            // Max
            float m = -1e20f;
            for (int t = 0; t < TILE_K; t++) m = fmaxf(m, lds_scores[head * TILE_K + t]);
            
            float new_max = fmaxf(row_max[hi], m);
            float corr = __expf(row_max[hi] - new_max);
            row_max[hi] = new_max;
            
            // Exp and sum
            float tile_sum = 0.0f;
            for (int t = 0; t < TILE_K; t++) {
                float e = __expf(lds_scores[head * TILE_K + t] - new_max);
                lds_scores[head * TILE_K + t] = e;
                tile_sum += e;
            }
            row_sum[hi] = row_sum[hi] * corr + tile_sum;
            
            // Correct V
            for (int d = 0; d < 8; d++) v_acc[hi][d] *= corr;
            
            // Accumulate V: each thread handles 8 dims (tid*8 to tid*8+7)
            for (int d = 0; d < 8; d++) {
                int dim = tid * 8 + d;
                if (dim < VD) {
                    float acc = 0.0f;
                    for (int t = 0; t < tile_len; t++) {
                        float w = lds_scores[head * TILE_K + t];
                        float v = fp8_to_f32(lds_kv[t * KVS + dim]) * kv_scale;
                        acc += w * v;
                    }
                    v_acc[hi][d] += acc;
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    for (int hi = 0; hi < 4; hi++) {
        int head = g * 4 + hi;
        
        // LSE (only one thread writes per head)
        float lse = (row_sum[hi] > 0) ? (__logf(row_sum[hi]) + row_max[hi]) : -1e20f;
        // Reduce LSE across threads (all have same value for this head)
        if (tid % 16 == 0) {
            splitLse[out_base_lse + head] = lse;
        }
        
        // V output: normalize and write
        float inv = (row_sum[hi] > 0) ? (1.0f / row_sum[hi]) : 0.0f;
        for (int d = 0; d < 8; d++) {
            int dim = tid * 8 + d;
            if (dim < VD) {
                splitO[(out_base_lse + head) * VD + dim] = v_acc[hi][d] * inv;
            }
        }
    }
}

// Reduce
__global__ void mla_reduce(
    const float* __restrict__ splitO,
    const float* __restrict__ splitLse,
    uint16_t* __restrict__ Out,
    int bs, int ns
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int head = blockIdx.y;
    
    float max_lse = -1e20f;
    for (int s = 0; s < ns; s++) {
        float l = splitLse[(bid * ns + s) * 16 + head];
        max_lse = fmaxf(max_lse, l);
    }
    
    float acc[8] = {0};
    float tw = 0.0f;
    
    for (int s = 0; s < ns; s++) {
        float l = splitLse[(bid * ns + s) * 16 + head];
        if (l <= -1e19f) continue;
        float w = __expf(l - max_lse);
        tw += w;
        int base = ((bid * ns + s) * 16 + head) * 512;
        for (int i = 0; i < 8; i++) {
            int d = tid * 8 + i;
            if (d < 512) acc[i] += w * splitO[base + d];
        }
    }
    
    float inv = (tw > 0) ? (1.0f / tw) : 0.0f;
    int ob = (bid * 16 + head) * 512;
    for (int i = 0; i < 8; i++) {
        int d = tid * 8 + i;
        if (d < 512) Out[ob + d] = f2bf(acc[i] * inv);
    }
}

torch::Tensor mla_fwd(
    torch::Tensor Q, float qs, torch::Tensor KV, float kvs,
    torch::Tensor kvi, torch::Tensor qo,
    torch::Tensor so, torch::Tensor sl, torch::Tensor out,
    int64_t bs, int64_t ns, double sms
) {
    mla_stage1<<<dim3(ns, bs), 64>>>(
        (const uint8_t*)Q.data_ptr(), qs,
        (const uint8_t*)KV.data_ptr(), kvs,
        kvi.data_ptr<int32_t>(), qo.data_ptr<int32_t>(),
        so.data_ptr<float>(), sl.data_ptr<float>(),
        (int)ns, (float)sms);
    
    mla_reduce<<<dim3(bs, 16), 64>>>(
        so.data_ptr<float>(), sl.data_ptr<float>(),
        (uint16_t*)out.data_ptr(), (int)bs, (int)ns);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mla_fwd", &mla_fwd); }
"""

_hip_mod = None
_hip_failed = None
_buf = {}


def _get_hip():
    global _hip_mod, _hip_failed
    if _hip_failed:
        return None
    if _hip_mod:
        return _hip_mod
    try:
        _hip_mod = load_inline(
            name="mla_correct",
            cpp_sources="",
            cuda_sources=_HIP_SRC,
            extra_cuda_cflags=["-O3", "-ffast-math"],
            verbose=False,
        )
        return _hip_mod
    except Exception as e:
        _hip_failed = True
        print(f"[mla] HIP JIT failed: {e}", file=sys.stderr)
        return None


def _get_ns(bs, kvl):
    if kvl <= 1024:
        return 8 if bs <= 32 else 4
    return 16 if bs <= 64 else 32


from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

FP8 = aiter_dtypes.fp8
_c = {}


def _fb(data):
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs, kvl = int(config["batch_size"]), int(config["kv_seq_len"])
    ns = _get_ns(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, 1, kv_fp8.shape[-1])
    bk = ("dq", q.numel())
    if bk not in _c:
        _c[bk] = (
            torch.empty_like(q, dtype=FP8),
            torch.empty(1, dtype=torch.float32, device=q.device),
        )
    qi, qs = _c[bk]
    dynamic_per_tensor_quant(qi, q, qs)
    qv = qi.view(-1, 16, 576)
    key = (bs, kvl, ns)
    if key not in _c:
        tkv = bs * kvl
        kl = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        ki = torch.arange(tkv, dtype=torch.int32, device=q.device)
        info = get_mla_metadata_info_v1(
            bs,
            1,
            16,
            qv.dtype,
            kv_fp8.dtype,
            is_sparse=False,
            fast_mode=False,
            num_kv_splits=ns,
            intra_batch_mode=True,
        )
        w = [torch.empty(s, dtype=t, device=q.device) for s, t in info]
        wm, wi, ws, ri, rf, rp = w
        get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kl,
            16,
            1,
            True,
            wm,
            ws,
            wi,
            ri,
            rf,
            rp,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=ns,
            intra_batch_mode=True,
            dtype_q=qv.dtype,
            dtype_kv=kv_fp8.dtype,
        )
        _c[key] = {
            "m": {
                "work_meta_data": wm,
                "work_indptr": wi,
                "work_info_set": ws,
                "reduce_indptr": ri,
                "reduce_final_map": rf,
                "reduce_partial_map": rp,
            },
            "kl": kl,
            "ki": ki,
            "o": torch.empty((bs, 16, 512), dtype=torch.bfloat16, device=q.device),
        }
    c = _c[key]
    mla_decode_fwd(
        qv,
        kv_4d,
        c["o"],
        qo_indptr,
        kv_indptr,
        c["ki"],
        c["kl"],
        1,
        page_size=1,
        nhead_kv=1,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **c["m"],
    )
    return c["o"]


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs, kvl = int(config["batch_size"]), int(config["kv_seq_len"])

    mod = _get_hip()
    if mod is None:
        return _fb(data)

    ns = _get_ns(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    dev = q.device

    q_fp8 = torch.empty_like(q, dtype=FP8)
    q_st = torch.empty(1, dtype=torch.float32, device=dev)
    dynamic_per_tensor_quant(q_fp8, q, q_st)

    key = (bs, kvl, ns, dev)
    if key not in _buf:
        _buf[key] = {
            "o": torch.empty((bs, 16, 512), dtype=torch.bfloat16, device=dev),
            "so": torch.empty((bs, ns, 16, 512), dtype=torch.float32, device=dev),
            "sl": torch.empty((bs, ns, 16), dtype=torch.float32, device=dev),
        }
    b = _buf[key]

    mod.mla_fwd(
        q_fp8.view(-1, 16 * 576),
        float(q_st.item()),
        kv_fp8.view(-1, 576),
        float(kv_scale.item()),
        kv_indptr.to(torch.int32),
        qo_indptr.to(torch.int32),
        b["so"],
        b["sl"],
        b["o"],
        bs,
        ns,
        SM_SCALE,
    )

    return b["o"]
