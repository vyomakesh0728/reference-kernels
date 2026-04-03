#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MFMA QK^T (32-tile) + V via mfma_scale_f32_16x16x128_f8f6f4 (K=128, 4 tiles at once)."""
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
typedef int v8i32 __attribute__((ext_vector_type(8)));

#define NH 16
#define QKD 576
#define VD 512
#define TILE 32
#define BIGTILE 128   // 4 x 32-token tiles for K=128 V MFMA
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

// Convert float to E8M0 (biased exponent only, no mantissa/sign)
// E8M0 = floor(log2(|x|)) + 127 bias
__device__ __forceinline__ int f2e8m0(float x) {
    if (x <= 0.0f) return 0;
    union { float f; uint32_t u; } c; c.f = x;
    return (int)((c.u >> 23) & 0xFF);  // IEEE exponent = E8M0 directly
}

// E8M0 to float scale: 2^(e8m0 - 127)
__device__ __forceinline__ float e8m02f(int e) {
    if (e == 0) return 0.0f;
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)e) << 23;
    return c.f;
}

__global__ void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs, const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    // LDS: we need space for 4 KV tiles simultaneously for the K=128 V MFMA
    // But that's 4 * 32 * 580 = 74240 bytes -- too much.
    // Instead: process QK^T for 4 tiles sequentially, store scores in LDS,
    // then do V MFMA once with K=128.
    __shared__ uint8_t kv_lds[TILE * KVS];          // reused per sub-tile
    __shared__ uint8_t q8[NH * QKD];
    __shared__ float   sc[NH * BIGTILE];             // 16 * 128 * 4 = 8192 bytes
    __shared__ uint8_t sc8[NH * BIGTILE];            // 16 * 128 = 2048 bytes (fp8 scores)
    __shared__ uint8_t kvv[VD * BIGTILE];            // 512 * 128 = 65536 bytes
    // Total LDS: ~18560 + 9216 + 8192 + 2048 + 65536 = ~103552 bytes
    // This is under 160KB limit.

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

    // Quantize Q -> fp8
    const uint16_t* qp = Q + bid * NH * QKD;
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += 64)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    amax = warp_max64(amax);
    float qs = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_qs = 240.0f / fmaxf(amax, 1e-12f);
    for (int i = tid; i < NH * QKD; i += 64)
        q8[i] = f2fp8(bf2f(qp[i]) * inv_qs);

    // V accumulators: 32 chunks x v4f
    v4f va[32];
    for (int i = 0; i < 32; i++) va[i] = (v4f){0, 0, 0, 0};
    float rmax[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float rsum[4] = {0, 0, 0, 0};
    float cqk = qs * kvs * sms;

    // Process in BIGTILE (128-token) chunks
    for (int bts = ms; bts < me; bts += BIGTILE) {
        int btsz = min(BIGTILE, me - bts);

        // Phase 1: QK^T for up to 4 sub-tiles of 32 tokens each
        int n_subtiles = (btsz + TILE - 1) / TILE;

        for (int st = 0; st < n_subtiles; st++) {
            int ts = bts + st * TILE;
            int tsz = min(TILE, me - ts);

            // Load KV sub-tile
            for (int i = tid; i < tsz * QKD; i += 64)
                kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
            if (tsz < TILE)
                for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                    kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

            // MFMA QK^T: 2 x 18 calls (proven working)
            v4f qk1 = (v4f){0,0,0,0}, qk2 = (v4f){0,0,0,0};
            for (int ch = 0; ch < 18; ch++) {
                int kb = ch * 32;
                long a = *(long*)(q8 + l16 * QKD + kb + g * 8);
                long b1 = *(long*)(kv_lds + l16 * KVS + kb + g * 8);
                long b2 = *(long*)(kv_lds + (HALF + l16) * KVS + kb + g * 8);
                qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b1, qk1, 0, 0, 0);
                qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b2, qk2, 0, 0, 0);
            }

            // Store scores into BIGTILE-wide buffer at offset st*TILE
            int off = st * TILE;
            for (int hi = 0; hi < 4; hi++) {
                sc[(g*4+hi) * BIGTILE + off + l16]        = qk1[hi] * cqk;
                sc[(g*4+hi) * BIGTILE + off + HALF + l16] = qk2[hi] * cqk;
            }

            // Also transpose V into big buffer: kvv[v_dim * BIGTILE + (off + token)]
            for (int i = tid; i < TILE * VD; i += 64) {
                int tok = i / VD;
                int dim = i % VD;
                kvv[dim * BIGTILE + off + tok] = kv_lds[tok * KVS + dim];
            }
        }

        // Zero-fill remaining sub-tiles if btsz < BIGTILE
        for (int st = n_subtiles; st < 4; st++) {
            int off = st * TILE;
            for (int hi = 0; hi < 4; hi++) {
                for (int t = 0; t < TILE; t++) {
                    sc[(g*4+hi) * BIGTILE + off + t] = 0.0f;
                }
            }
            for (int i = tid; i < TILE * VD; i += 64) {
                int tok = i / VD;
                int dim = i % VD;
                kvv[dim * BIGTILE + off + tok] = 0;
            }
        }

        // Phase 2: Online softmax over all 128 scores + quantize to fp8 with per-32-block E8M0 scale
        // Each thread's group g covers 32 consecutive tokens out of 128
        // So the E8M0 scale is per-group (per 32-token block)

        v4f corr_v;
        float score_scale_a[4]; // per-head score scale for this thread's group
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            float tmax = -1e20f;
            for (int t = 0; t < btsz; t++)
                tmax = fmaxf(tmax, sc[head * BIGTILE + t]);
            float nm = fmaxf(rmax[hi], tmax);
            float c = __expf(rmax[hi] - nm);
            corr_v[hi] = c;
            rmax[hi] = nm;

            // Compute exp scores and find per-group-32 max for E8M0
            float tsum = 0.0f;
            float gmax = 0.0f;  // max in this thread's 32-token group
            for (int t = 0; t < BIGTILE; t++) {
                float e = (t < btsz) ? __expf(sc[head * BIGTILE + t] - rmax[hi]) : 0.0f;
                sc[head * BIGTILE + t] = e;
                if (t < btsz) tsum += e;
                // Track max in our group of 32
                if (t >= g * 32 && t < (g + 1) * 32)
                    gmax = fmaxf(gmax, e);
            }
            rsum[hi] = rsum[hi] * c + tsum;

            // Compute E8M0 scale for this group: covers sc[head * BIGTILE + g*32 .. g*32+31]
            // Scale = max score in group. fp8 = score / scale * 240
            float inv_gmax = (gmax > 1e-12f) ? (240.0f / gmax) : 0.0f;
            score_scale_a[hi] = gmax / 240.0f;  // to reconstruct: fp8_val * scale

            // Quantize scores to fp8 for this group
            for (int t = 0; t < 32; t++) {
                int gt = g * 32 + t;
                sc8[head * BIGTILE + gt] = f2fp8(sc[head * BIGTILE + gt] * inv_gmax);
            }
        }

        // Correct V accumulators
        for (int vc = 0; vc < 32; vc++) va[vc] *= corr_v;

        // Phase 3: V MFMA with mfma_scale_f32_16x16x128_f8f6f4
        // K=128 fp8 values per lane = 128/4 groups = 32 fp8 per lane = 32 bytes = 8 int32 = v8i32
        //
        // A = scores[head][token]: lane%16=head, 32 tokens per group
        // B = V[token][v_dim]: lane%16=v_dim, 32 tokens per group
        //
        // Load A: 32 fp8 bytes from sc8[l16 * BIGTILE + g * 32]
        v8i32 sa;
        {
            const int* src = (const int*)(sc8 + l16 * BIGTILE + g * 32);
            for (int r = 0; r < 8; r++) sa[r] = src[r];
        }

        // E8M0 scale for A: per-thread, encodes the group's score scale
        // We need a single E8M0 that applies to all 4 heads this thread computes.
        // PROBLEM: each head has different score_scale_a[hi]. The hardware only gives
        // one scale_a per thread. Use the max across heads and pre-scale scores.
        // Actually, let's just use scale=0 (neutral, =1.0 = E8M0 bias 127) and
        // bake the scale into the fp8 values themselves (like test_mfma_full.py did).
        // This means scores are already scaled to fit fp8 range via *240.
        int e8m0_a = 127;  // neutral scale (2^0 = 1.0)
        int e8m0_b = 127;  // neutral scale for V

        for (int vc = 0; vc < 32; vc++) {
            int vb = vc * 16;
            // Load B: 32 fp8 V values for v_dim (vb+l16) across 128 tokens
            // kvv[(vb+l16) * BIGTILE + token], tokens [g*32 .. g*32+31]
            v8i32 sb;
            {
                const int* src = (const int*)(kvv + (vb + l16) * BIGTILE + g * 32);
                for (int r = 0; r < 8; r++) sb[r] = src[r];
            }
            va[vc] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                sa, sb, va[vc],
                0, 0,         // Atype=fp8, Btype=fp8
                0, e8m0_a,    // opsel_a=0, scale_a
                0, e8m0_b     // opsel_b=0, scale_b
            );
        }
    }

    // Write output: D[head][v_dim], lane%16=v_dim, head=4*g+gpr
    // Score fp8 was scaled by 240 (then neutral E8M0), V fp8 has kvs scale
    // Combined: result = sum(score_fp8 * v_fp8) where score_fp8 ~ exp_score*240, v_fp8 ~ v_real/kvs
    // So result ~ sum(exp_score * v_real) * 240 / kvs ... wait, no:
    // score_fp8 = exp_score * 240 (quantized), v_fp8 = v_raw (stored as fp8 with kvs scale)
    // MFMA result = sum(score_fp8_dequant * v_fp8_dequant)
    // = sum(exp_score*240 * v_real) approximately  ... not right either.
    // Actually: score quantized as f2fp8(exp_score * 240), so dequant ~ exp_score * 240
    // V stored as raw fp8 with external kvs scale, dequant = fp82f(byte) * kvs
    // But in MFMA, both A and B are treated as fp8 and dequanted by hardware.
    // The V bytes in KV are already in fp8 format (E4M3), so hardware dequant is correct.
    // Result = sum(dequant_A * dequant_B) = sum(~exp_score*240 * ~v_real_scaled)
    // where v_real_scaled = v_raw_fp8 (needs kvs to get true value)
    // So: true_result = MFMA_result * kvs / 240
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

__global__ void reduce_k(
    const float* __restrict__ sO, const float* __restrict__ sL,
    uint16_t* __restrict__ O, int ns
) {
    const int bid = blockIdx.x, tid = threadIdx.x;
    const int hid = tid / 16, lid = tid % 16;
    float ml = -1e20f;
    for (int s = 0; s < ns; s++) ml = fmaxf(ml, sL[(bid * ns + s) * NH + hid]);
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
        _mod[0] = load_inline(name='mla_scale_k128', cpp_sources='',
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
