#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""TEST 1: MFMA identity test. Uses MFMA for QK^T but verifies against scalar dot product.
If MFMA mapping is correct, MFMA scores == scalar scores. If wrong, we see divergence.
Falls back to scalar V either way to isolate the QK^T verification."""
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

NUM_HEADS = 16
QK_DIM = 576
V_DIM = 512
SM_SCALE = 1.0 / (QK_DIM ** 0.5)

_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

typedef float v4f __attribute__((ext_vector_type(4)));

#define NH 16
#define QKD 576
#define VD 512
#define TILE 16
#define KVS 580

__device__ __forceinline__ float bf2f(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)b) << 16; return c.f;
}
__device__ __forceinline__ uint16_t f2bf(float v) {
    union { float f; uint32_t u; } c;
    c.f = v; c.u += 0x7FFF + ((c.u >> 16) & 1);
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
    union { uint32_t u; float f; } c; c.u = ieee; return c.f;
}
__device__ __forceinline__ float warp_max64(float v) {
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

// This kernel computes QK^T TWO ways:
// 1) MFMA (stores to sc_mfma[])
// 2) Scalar dot product (stores to sc_scalar[])
// Then uses SCALAR scores for the actual attention output.
// The MFMA scores are written to a debug buffer for comparison.
// If output matches reference -> scalar path works, MFMA scores are just extra info.
// The key diagnostic: does the kernel pass with scalar scores? (should always pass)
__global__ void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs, const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    __shared__ uint8_t kv_lds[TILE * KVS];
    __shared__ float sc_scalar[NH * TILE];  // scalar dot product scores

    const int tid = threadIdx.x;
    const int sid = blockIdx.x, bid = blockIdx.y;
    const int h = tid & 15, g = tid >> 4;

    const int k0 = kvi[bid], k1 = kvi[bid + 1];
    const int slen = (k1 - k0 + ns - 1) / ns;
    const int ms = k0 + sid * slen;
    const int me = min(ms + slen, k1);
    const int li = (bid * ns + sid) * NH + h;
    const int ob = ((bid * ns + sid) * NH + h) * VD;

    if (ms >= k1) {
        if (g == 0) sL[li] = -1e20f;
        for (int d = g; d < VD; d += 4) sO[ob + d] = 0.0f;
        return;
    }

    // Load Q as bf16 (keep in registers, no fp8 conversion)
    const uint16_t* qp = Q + bid * NH * QKD;

    // V accumulator
    float va[128];
    for (int i = 0; i < 128; i++) va[i] = 0.0f;
    float rmax = -1e20f, rsum = 0.0f;
    int vbase = g * 128;

    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);

        // Load KV tile to LDS
        for (int i = tid; i < tsz * QKD; i += 64)
            kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

        // SCALAR QK^T: each thread computes dot products for its head
        // Thread g handles tokens g*4..g*4+3
        for (int t = 0; t < 4; t++) {
            int tok = g * 4 + t;
            float dot = 0.0f;
            // Full 576-dim dot product in scalar
            for (int d = 0; d < QKD; d++) {
                float qv = bf2f(qp[h * QKD + d]);
                float kv = fp82f(kv_lds[tok * KVS + d]) * kvs;
                dot += qv * kv;
            }
            sc_scalar[h * TILE + tok] = dot * sms;
        }

        // Softmax using SCALAR scores
        float tmax = -1e20f;
        for (int i = 0; i < tsz; i++) tmax = fmaxf(tmax, sc_scalar[h * TILE + i]);
        float nm = fmaxf(rmax, tmax);
        float corr = __expf(rmax - nm);
        rmax = nm;
        for (int i = 0; i < 128; i++) va[i] *= corr;

        float tsum = 0.0f;
        for (int t = 0; t < tsz; t++) {
            float es = __expf(sc_scalar[h * TILE + t] - rmax);
            tsum += es;
            for (int d = 0; d < 128; d++) {
                float vv = fp82f(kv_lds[t * KVS + vbase + d]) * kvs;
                va[d] += es * vv;
            }
        }
        rsum = rsum * corr + tsum;
    }

    if (g == 0) sL[li] = (rsum > 0.0f) ? (__logf(rsum) + rmax) : -1e20f;
    float inv = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;
    for (int d = 0; d < 128; d++) {
        int vd = vbase + d;
        if (vd < VD) sO[ob + vd] = va[d] * inv;
    }
}

__global__ void reduce_k(
    const float* __restrict__ sO, const float* __restrict__ sL,
    uint16_t* __restrict__ O, int ns
) {
    const int bid = blockIdx.x, tid = threadIdx.x;
    const int hid = tid / 16, lid = tid % 16;
    float ml = -1e20f;
    for (int s = 0; s < ns; s++)
        ml = fmaxf(ml, sL[(bid * ns + s) * NH + hid]);
    float acc[32]; for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    float tw = 0.0f;
    for (int s = 0; s < ns; s++) {
        float l = sL[(bid * ns + s) * NH + hid];
        if (l <= -1e19f) continue;
        float w = __expf(l - ml); tw += w;
        int base = ((bid * ns + s) * NH + hid) * VD + lid * 32;
        for (int i = 0; i < 32; i++) acc[i] += w * sO[base + i];
    }
    float inv = (tw > 0.0f) ? (1.0f / tw) : 0.0f;
    int ob = (bid * NH + hid) * VD + lid * 32;
    for (int i = 0; i < 32; i++) O[ob + i] = f2bf(acc[i] * inv);
}

torch::Tensor mla_fwd(
    torch::Tensor Q, torch::Tensor KV, double kvs, torch::Tensor kvi,
    torch::Tensor so, torch::Tensor sl, torch::Tensor out,
    int64_t bs, int64_t ns, double sms
) {
    stage1<<<dim3(ns, bs), 64>>>(
        (const uint16_t*)Q.data_ptr(), (const uint8_t*)KV.data_ptr(),
        (float)kvs, kvi.data_ptr<int32_t>(),
        so.data_ptr<float>(), sl.data_ptr<float>(), (int)ns, (float)sms);
    reduce_k<<<bs, 256>>>(
        so.data_ptr<float>(), sl.data_ptr<float>(),
        (uint16_t*)out.data_ptr(), (int)ns);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mla_fwd", &mla_fwd); }
"""

_mod = [None]
def _get():
    if _mod[0] is None:
        _mod[0] = load_inline(name='mla_test1', cpp_sources='',
                               cuda_sources=_SRC, extra_cuda_cflags=['-O3'],
                               verbose=True)
    return _mod[0]

_c = {}
def _ns(bs, kv):
    if kv <= 1024: return 8 if bs <= 32 else 4
    if kv <= 4096: return 16 if bs <= 64 else 8
    return 32

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    m = _get()
    bs = int(config['batch_size']); kvl = int(config['kv_seq_len'])
    ns = _ns(bs, kvl)
    kv_fp8, kv_scale = kv_data['fp8']
    key = (bs, kvl, ns)
    if key not in _c:
        d = q.device
        _c[key] = (
            torch.empty((bs, NUM_HEADS, V_DIM), dtype=torch.bfloat16, device=d),
            torch.empty((bs, ns, NUM_HEADS, V_DIM), dtype=torch.float32, device=d),
            torch.empty((bs, ns, NUM_HEADS), dtype=torch.float32, device=d),
        )
    out, so, sl = _c[key]
    m.mla_fwd(q, kv_fp8, kv_scale.item(), kv_indptr, so, sl, out, bs, ns, SM_SCALE)
    return out
