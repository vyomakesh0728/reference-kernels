#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
V MFMA with bf8 (e5m2fnuz) scores: __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8.

Problem: fp8 e4m3 (4-bit exponent, bias=8) has min positive ~2^(-10). Softmax
scores like 1e-5 get flushed to zero, causing 95% output mismatch.

Fix: quantize softmax scores to bf8 e5m2fnuz (5-bit exponent, bias=16) which has
min positive ~2^(-17) and much better dynamic range for small attention weights.
V values stay as fp8 e4m3 (from input KV data).

Intrinsic: __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8
  A = scores as bf8 e5m2fnuz (better range for small weights)
  B = V as fp8 e4m3fnuz (from input, already quantized)

NOTE on MFMA_SCALE: The scale variants (_mfma_scale_f32_32x32x64_f8f6f4 and
_mfma_scale_f32_16x16x128_f8f6f4) require K=64 or K=128 respectively. Our V
dimension tiles use K=32 (32 tokens per tile), so MFMA_SCALE cannot be used
for the V matmul with 32-token tiles. bf8 scores are the next best option.
"""
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
#define TILE 32
#define HALF 16
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
// fp8 e4m3fnuz: bias=8, 4 exponent bits, 3 mantissa bits, range ~[-240, 240]
__device__ __forceinline__ uint8_t f2fp8(float x) {
    if (x == 0.0f) return 0;
    union { float f; uint32_t u; } c; c.f = x;
    uint32_t s = (c.u >> 31) & 1;
    int e = (int)((c.u >> 23) & 0xFF) - 119;  // 127 - 8 = 119
    uint32_t m = (c.u >> 20) & 0x7;
    if (e <= 0) return 0;
    if (e >= 16) return (s << 7) | 0x7F;
    return (s << 7) | (e << 3) | m;
}
// bf8 e5m2fnuz: bias=16, 5 exponent bits, 2 mantissa bits
// Range: min positive ~2^(-17), max ~2^15 = 57344
// Much better dynamic range for small softmax scores
__device__ __forceinline__ uint8_t f2bf8(float x) {
    if (x == 0.0f) return 0;
    union { float f; uint32_t u; } c; c.f = x;
    uint32_t s = (c.u >> 31) & 1;
    int e = (int)((c.u >> 23) & 0xFF) - 111;  // 127 - 16 = 111 for bf8 bias=16
    uint32_t m = (c.u >> 21) & 0x3;  // 2 mantissa bits
    if (e <= 0) return 0;
    if (e >= 32) return (s << 7) | 0x7F;
    return (s << 7) | (e << 2) | m;
}
__device__ __forceinline__ float warp_max64(float v) {
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

// Grid: (nsplits, bs), Block: (64,) = 1 wavefront
__global__ void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs, const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    __shared__ uint8_t kv_lds[TILE * KVS];     // 32 * 580 = 18560
    __shared__ uint8_t q8[NH * QKD];            // 16 * 576 = 9216
    __shared__ float   sc[NH * TILE];           // 16 * 32 * 4 = 2048
    __shared__ uint8_t sc8[NH * TILE];          // 16 * 32 = 512
    __shared__ uint8_t kvv[VD * TILE];          // 512 * 32 = 16384
    // Total LDS: ~46720 bytes

    const int tid = threadIdx.x;
    const int sid = blockIdx.x, bid = blockIdx.y;
    const int g = tid >> 4, l16 = tid & 15;

    const int k0 = kvi[bid], k1 = kvi[bid + 1];
    const int slen = (k1 - k0 + ns - 1) / ns;
    const int ms = k0 + sid * slen;
    const int me = min(ms + slen, k1);

    if (ms >= k1) {
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            if (l16 == 0) sL[(bid * ns + sid) * NH + head] = -1e20f;
            int ob = ((bid * ns + sid) * NH + head) * VD;
            for (int d = l16; d < VD; d += 16) sO[ob + d] = 0.0f;
        }
        return;
    }

    // Quantize Q -> fp8 in LDS
    const uint16_t* qp = Q + bid * NH * QKD;
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += 64)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    amax = warp_max64(amax);
    float qs = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_qs = 240.0f / fmaxf(amax, 1e-12f);
    for (int i = tid; i < NH * QKD; i += 64)
        q8[i] = f2fp8(bf2f(qp[i]) * inv_qs);

    // V accumulators: 32 chunks x v4f (4 heads per lane group)
    v4f va[32];
    for (int i = 0; i < 32; i++) va[i] = (v4f){0, 0, 0, 0};
    float rmax[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float rsum[4] = {0, 0, 0, 0};
    float cqk = qs * kvs * sms;

    // Main loop: 32-token tiles
    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);

        // Load KV tile to LDS
        for (int i = tid; i < tsz * QKD; i += 64)
            kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

        // QK^T MFMA: tokens 0-15 and 16-31 (fp8_fp8, same as before)
        v4f qk1 = (v4f){0,0,0,0}, qk2 = (v4f){0,0,0,0};
        for (int ch = 0; ch < 18; ch++) {
            int kb = ch * 32;
            long a = *(long*)(q8 + l16 * QKD + kb + g * 8);
            long b1 = *(long*)(kv_lds + l16 * KVS + kb + g * 8);
            long b2 = *(long*)(kv_lds + (HALF + l16) * KVS + kb + g * 8);
            qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b1, qk1, 0, 0, 0);
            qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b2, qk2, 0, 0, 0);
        }

        // Store scores: D[head][token], lane%16=token, GPR=head%4
        for (int hi = 0; hi < 4; hi++) {
            sc[(g*4+hi) * TILE + l16]        = qk1[hi] * cqk;
            sc[(g*4+hi) * TILE + HALF + l16] = qk2[hi] * cqk;
        }

        // Online softmax + score quantization per head
        // Correction vector for V accumulators
        v4f corr_v;
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            float tmax = -1e20f;
            for (int t = 0; t < tsz; t++)
                tmax = fmaxf(tmax, sc[head * TILE + t]);
            float nm = fmaxf(rmax[hi], tmax);
            float c = __expf(rmax[hi] - nm);
            corr_v[hi] = c;
            rmax[hi] = nm;

            float tsum = 0.0f;
            for (int t = 0; t < TILE; t++) {
                float e = (t < tsz) ? __expf(sc[head * TILE + t] - rmax[hi]) : 0.0f;
                sc[head * TILE + t] = e;
                tsum += e;
            }
            rsum[hi] = rsum[hi] * c + tsum;

            // Quantize scores to bf8 e5m2fnuz (NOT fp8 e4m3)
            // Max exp_score = 1.0. bf8 e5m2 max representable ~57344.
            // Scale by 240 to match fp8 convention (could go higher, but 240 keeps
            // the scale factor simple and consistent with KV fp8 dequant).
            // bf8 min positive with bias=16: 2^(-17) ~ 7.6e-6
            // So scores as small as 7.6e-6 / 240 ~ 3.2e-8 survive quantization.
            // Compare fp8 e4m3: min positive 2^(-10) ~ 9.8e-4, scores < 4e-6 die.
            for (int t = 0; t < TILE; t++)
                sc8[head * TILE + t] = f2bf8(sc[head * TILE + t] * 240.0f);
        }

        // Correct V accumulators (per-head correction)
        for (int vc = 0; vc < 32; vc++) va[vc] *= corr_v;

        // Transpose V data: kvv[v_dim * 32 + token] = kv_lds[token * KVS + v_dim]
        for (int i = tid; i < TILE * VD; i += 64) {
            int tok = i / VD;
            int dim = i % VD;
            kvv[dim * TILE + tok] = kv_lds[tok * KVS + dim];
        }

        // V MFMA: 32 calls for 512 V dims (16 per call)
        // A = scores as bf8, B = V as fp8 -> bf8_fp8 intrinsic
        long sa = *(long*)(sc8 + l16 * TILE + g * 8);
        for (int vc = 0; vc < 32; vc++) {
            int vb = vc * 16;
            long sb = *(long*)(kvv + (vb + l16) * TILE + g * 8);
            va[vc] = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(sa, sb, va[vc], 0, 0, 0);
        }
    }

    // Write output
    // V scale: scores were scaled by 240, V values have scale kvs
    // bf8_fp8 MFMA computes: (score_bf8 * V_fp8) in float32
    // score_bf8 = true_score * 240 (from quantization above)
    // V_fp8 already has scale kvs applied externally
    // So result = true_score * 240 * V_fp8_raw
    // Need to divide by 240 and multiply by kvs to get true output
    float vs = kvs / 240.0f;
    for (int hi = 0; hi < 4; hi++) {
        int head = g * 4 + hi;
        int li = (bid * ns + sid) * NH + head;
        int ob = ((bid * ns + sid) * NH + head) * VD;
        if (l16 == 0)
            sL[li] = (rsum[hi] > 0.0f) ? (__logf(rsum[hi]) + rmax[hi]) : -1e20f;
        float inv = (rsum[hi] > 0.0f) ? (vs / rsum[hi]) : 0.0f;
        for (int vc = 0; vc < 32; vc++) {
            int vd = vc * 16 + l16;
            if (vd < VD)
                sO[ob + vd] = va[vc][hi] * inv;
        }
    }
}

// Reduce: combine splits
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
        _mod[0] = load_inline(name='mla_mfma_bf8_v', cpp_sources='',
                               cuda_sources=_SRC, extra_cuda_cflags=['-O1'],
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
