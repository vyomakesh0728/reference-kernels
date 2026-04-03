#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Fused GPU-only fp8 quant: amax+scale+quant all on GPU, zero host sync.
Kernel 1: amax reduction -> writes scale to GPU buffer.
Kernel 2: reads scale from GPU, quantizes Q to fp8."""
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
FP8_DTYPE = aiter_dtypes.fp8
_fp8_finfo = torch.finfo(FP8_DTYPE)

_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

// Kernel 1: amax reduction + compute scale, write both to GPU buffer
// scale = amax / 240, inv_scale = 240 / amax
__global__ void amax_scale_kernel(
    const uint16_t* __restrict__ data,
    float* __restrict__ scale_out,    // [0] = scale, [1] = inv_scale
    int n
) {
    __shared__ float smax[256];
    int tid = threadIdx.x;
    float mx = 0.0f;
    for (int i = tid; i < n; i += 256) {
        union { uint32_t u; float f; } c;
        c.u = ((uint32_t)data[i]) << 16;
        mx = fmaxf(mx, fabsf(c.f));
    }
    smax[tid] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        float amax = fmaxf(smax[0], 1e-12f);
        scale_out[0] = amax / 240.0f;     // scale
        scale_out[1] = 240.0f / amax;     // inv_scale
    }
}

// Kernel 2: quantize bf16 -> fp8 using scale from GPU buffer (no host read!)
__global__ void quant_fp8_kernel(
    const uint16_t* __restrict__ Q_bf16,
    uint8_t* __restrict__ Q_fp8,
    const float* __restrict__ scale_buf,  // reads inv_scale from scale_buf[1]
    int n
) {
    float inv_scale = scale_buf[1];  // all threads read same value (L1 cached)
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        union { uint32_t u; float f; } c;
        c.u = ((uint32_t)Q_bf16[i]) << 16; float v0 = c.f * inv_scale;
        c.u = ((uint32_t)Q_bf16[i+1]) << 16; float v1 = c.f * inv_scale;
        c.u = ((uint32_t)Q_bf16[i+2]) << 16; float v2 = c.f * inv_scale;
        c.u = ((uint32_t)Q_bf16[i+3]) << 16; float v3 = c.f * inv_scale;
        int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, 0, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, pk, true);
        *(int*)(Q_fp8 + i) = pk;
    } else if (i < n) {
        for (int j = i; j < n && j < i + 4; j++) {
            union { uint32_t uu; float ff; } cc;
            cc.uu = ((uint32_t)Q_bf16[j]) << 16;
            float vv = cc.ff * inv_scale;
            if (vv == 0.0f) { Q_fp8[j] = 0; continue; }
            union { float ff2; uint32_t uu2; } cc2; cc2.ff2 = vv;
            uint32_t ss = (cc2.uu2 >> 31) & 1;
            int ee = (int)((cc2.uu2 >> 23) & 0xFF) - 119;
            uint32_t mm = (cc2.uu2 >> 20) & 0x7;
            if (ee <= 0) Q_fp8[j] = 0;
            else if (ee >= 16) Q_fp8[j] = (ss << 7) | 0x7F;
            else Q_fp8[j] = (ss << 7) | (ee << 3) | mm;
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("amax_scale", [](torch::Tensor data, torch::Tensor scale_buf) {
        amax_scale_kernel<<<1, 256>>>(
            (const uint16_t*)data.data_ptr(),
            scale_buf.data_ptr<float>(), data.numel());
    });
    m.def("quant_fp8", [](torch::Tensor Q_bf16, torch::Tensor Q_fp8, torch::Tensor scale_buf) {
        int n = Q_bf16.numel();
        int threads = 256;
        int blocks = (n / 4 + threads - 1) / threads;
        quant_fp8_kernel<<<blocks, threads>>>(
            (const uint16_t*)Q_bf16.data_ptr(),
            (uint8_t*)Q_fp8.data_ptr(),
            scale_buf.data_ptr<float>(), n);
    });
}
"""

_mod = [None]
def _get():
    if _mod[0] is None:
        _mod[0] = load_inline(name='mla_fused_quant_v2', cpp_sources='',
                               cuda_sources=_SRC, extra_cuda_cflags=['-O3'],
                               verbose=True)
    return _mod[0]

_cache = {}

def _pick_splits(bs, kvl):
    if kvl <= 1024: return 8 if bs <= 32 else 4
    if kvl <= 4096: return 16 if bs <= 64 else 8
    return 32

def _get_or_build(bs, ql, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = (bs, kvl, ns, qd, ps, fm)
    if key in _cache: return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(bs, ql, NUM_HEADS, qd, kvd, is_sparse=False, fast_mode=fm, num_kv_splits=ns, intra_batch_mode=True)
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(qo, kvi, kl, NUM_HEADS//NUM_KV_HEADS, NUM_KV_HEADS, True, wm, ws, wi, ri, rf, rp, page_size=ps, kv_granularity=max(ps,16), max_seqlen_qo=ql, uni_seqlen_qo=ql, fast_mode=fm, max_split_per_batch=ns, intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)
    tq = bs * ql
    e = {"meta": {"work_meta_data":wm,"work_indptr":wi,"work_info_set":ws,"reduce_indptr":ri,"reduce_final_map":rf,"reduce_partial_map":rp}, "kl":kl, "ki":ki, "out":torch.empty((tq,NUM_HEADS,V_HEAD_DIM),dtype=torch.bfloat16,device=dev)}
    _cache[key] = e
    return e

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"]); ql = int(config["q_seq_len"]); kvl = int(config["kv_seq_len"])
    ns = _pick_splits(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if bs >= 64 and kvl >= 8192:
        m = _get()
        n = q.numel()
        # Pre-alloc (cached)
        bkey = ("fq_bufs", n)
        if bkey not in _cache:
            _cache[bkey] = (
                torch.empty(n, dtype=torch.uint8, device=q.device),
                torch.empty(2, dtype=torch.float32, device=q.device),  # [scale, inv_scale]
            )
        q_fp8_flat, scale_buf = _cache[bkey]
        # GPU-only: amax+scale (1 block), then quant (many blocks). No host sync!
        m.amax_scale(q, scale_buf)
        m.quant_fp8(q, q_fp8_flat, scale_buf)
        # scale_buf[0] = scale value, pass to aiter as q_scale
        qi = q_fp8_flat.view(FP8_DTYPE).reshape(q.shape)
        qs = scale_buf[0:1]  # slice, stays on GPU (no .item()!)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
        ps, fm = 1, False
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
        qs = None
        ps, fm = 2, True

    c = _get_or_build(bs, ql, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm)
    mla_decode_fwd(qv, kv_4d, c["out"], qo_indptr, kv_indptr, c["ki"], c["kl"], ql, page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE, logit_cap=0.0, num_kv_splits=ns, q_scale=qs, kv_scale=kv_scale, intra_batch_mode=True, **c["meta"])
    return c["out"]
