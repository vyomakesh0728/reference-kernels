#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""TEST 2: fp8 conversion test. Uses MFMA QK^T with HARDWARE fp8 data (no custom f2fp8).
Q is quantized via torch (aiter_dtypes.fp8), KV is already fp8 from input.
Both go directly to MFMA. Scalar V with hardware fp82f (via LUT verified to work).
If this passes: our f2fp8 conversion was the bug. If fails: MFMA mapping is wrong."""
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes

NUM_HEADS = 16
QK_DIM = 576
V_DIM = 512
SM_SCALE = 1.0 / (QK_DIM ** 0.5)
FP8_DTYPE = aiter_dtypes.fp8
_fp8_finfo = torch.finfo(FP8_DTYPE)

_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

typedef float v4f __attribute__((ext_vector_type(4)));

#define NH 16
#define QKD 576
#define VD 512
#define TILE 16
#define KVS 580

__device__ __forceinline__ uint16_t f2bf(float v) {
    union { float f; uint32_t u; } c;
    c.f = v; c.u += 0x7FFF + ((c.u >> 16) & 1);
    return (uint16_t)(c.u >> 16);
}

// LUT-based fp8 to float (known working from first scalar kernel)
__device__ void build_lut(float* lut, int tid) {
    if (tid < 256) {
        uint8_t x = (uint8_t)tid;
        float v = 0.0f;
        if (x != 0 && x != 0x80) {
            int s = (x >> 7) & 1;
            int e = (x >> 3) & 0xF;
            int m = x & 0x7;
            if (e == 0)
                v = ldexpf((float)m, -10);
            else
                v = ldexpf(1.0f + (float)m * 0.125f, e - 8);
            if (s) v = -v;
        }
        lut[tid] = v;
    }
}

__device__ __forceinline__ float warp_max64(float v) {
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

// MFMA QK^T using pre-quantized fp8 Q (from torch) + fp8 KV (from input)
// Scalar V using LUT fp8 conversion (known working)
__global__ void stage1(
    const uint8_t* __restrict__ Q_fp8,  // (bs, 16, 576) fp8 -- pre-quantized by torch
    const uint8_t* __restrict__ KV,     // (total_kv, 576) fp8
    float q_scale, float kv_scale,
    const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    __shared__ uint8_t kv_lds[TILE * KVS];
    __shared__ float lut[256];
    __shared__ float sc[NH * TILE];

    const int tid = threadIdx.x;
    const int sid = blockIdx.x, bid = blockIdx.y;
    const int h = tid & 15, g = tid >> 4;

    build_lut(lut, tid);
    // Extra entries for tid >= 256: handled by the if (tid < 256) guard

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

    // Q fp8 pointer for this batch item (already quantized in Python)
    const uint8_t* q8 = Q_fp8 + bid * NH * QKD;

    float va[128];
    for (int i = 0; i < 128; i++) va[i] = 0.0f;
    float rmax = -1e20f, rsum = 0.0f;
    float cqk = q_scale * kv_scale * sms;
    int vbase = g * 128;

    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);

        for (int i = tid; i < tsz * QKD; i += 64)
            kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

        // MFMA QK^T: 18 calls
        // A = Q_fp8[head, dim_chunk], B = KV_fp8[token, dim_chunk]
        v4f qk = (v4f){0,0,0,0};
        for (int ch = 0; ch < 18; ch++) {
            int d = ch * 32;
            // A: Q_fp8 direct from global memory (already fp8, no conversion)
            long a = *(long*)(q8 + h * QKD + d + g * 8);
            // B: KV from LDS
            long b = *(long*)(kv_lds + h * KVS + d + g * 8);
            qk = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, qk, 0, 0, 0);
        }

        // Store MFMA scores to LDS
        sc[h * TILE + g * 4 + 0] = qk[0] * cqk;
        sc[h * TILE + g * 4 + 1] = qk[1] * cqk;
        sc[h * TILE + g * 4 + 2] = qk[2] * cqk;
        sc[h * TILE + g * 4 + 3] = qk[3] * cqk;

        // Softmax
        float tmax = -1e20f;
        for (int i = 0; i < tsz; i++) tmax = fmaxf(tmax, sc[h * TILE + i]);
        float nm = fmaxf(rmax, tmax);
        float corr = __expf(rmax - nm);
        rmax = nm;
        for (int i = 0; i < 128; i++) va[i] *= corr;

        float tsum = 0.0f;
        for (int t = 0; t < tsz; t++) {
            float es = __expf(sc[h * TILE + t] - rmax);
            tsum += es;
            for (int d = 0; d < 128; d++) {
                float vv = lut[kv_lds[t * KVS + vbase + d]] * kv_scale;
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
    torch::Tensor Q_fp8, torch::Tensor KV,
    double q_scale, double kv_scale,
    torch::Tensor kvi,
    torch::Tensor so, torch::Tensor sl, torch::Tensor out,
    int64_t bs, int64_t ns, double sms
) {
    stage1<<<dim3(ns, bs), 64>>>(
        (const uint8_t*)Q_fp8.data_ptr(), (const uint8_t*)KV.data_ptr(),
        (float)q_scale, (float)kv_scale,
        kvi.data_ptr<int32_t>(),
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
        _mod[0] = load_inline(name='mla_test2', cpp_sources='',
                               cuda_sources=_SRC, extra_cuda_cflags=['-O3'],
                               verbose=True)
    return _mod[0]

def _quantize_fp8(t):
    finfo = _fp8_finfo
    a = t.abs().amax().clamp(min=1e-12)
    s = a / finfo.max
    return (t / s).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE), s.float().reshape(1)

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

    # Quantize Q using torch (hardware-compatible fp8)
    q_fp8, q_scale = _quantize_fp8(q)

    key = (bs, kvl, ns)
    if key not in _c:
        d = q.device
        _c[key] = (
            torch.empty((bs, NUM_HEADS, V_DIM), dtype=torch.bfloat16, device=d),
            torch.empty((bs, ns, NUM_HEADS, V_DIM), dtype=torch.float32, device=d),
            torch.empty((bs, ns, NUM_HEADS), dtype=torch.float32, device=d),
        )
    out, so, sl = _c[key]
    m.mla_fwd(q_fp8, kv_fp8, q_scale.item(), kv_scale.item(),
              kv_indptr, so, sl, out, bs, ns, SM_SCALE)
    return out
