#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""E2E MFMA opt1: eliminate V transpose, vectorize KV load, unroll loops.
From mfma_e2e_asm_pipeline.py (PASSES 4/4, max error 0.018)."""
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
// KV LDS stride: 576 bytes per token, padded to 584 for 8-byte alignment + bank shift
#define KVS 584

__device__ __forceinline__ float bf2f(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)b) << 16; return c.f;
}
__device__ __forceinline__ uint16_t f2bf(float v) {
    union { float f; uint32_t u; } c;
    c.f = v; c.u += 0x7FFF + ((c.u >> 16) & 1);
    return (uint16_t)(c.u >> 16);
}
__device__ __forceinline__ float warp_max64(float v) {
    #pragma unroll
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}
__device__ __forceinline__ int cvt_pk_fp8_lo(float a, float b, int old) {
    return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, false);
}
__device__ __forceinline__ int cvt_pk_fp8_hi(float a, float b, int old) {
    return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, true);
}

// Grid: (nsplits, bs), Block: (64,)
__global__ void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs, const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    // LDS: KV tile + Q fp8 + score fp8 (no float scores, no V transpose!)
    __shared__ uint8_t kv_lds[TILE * KVS];   // 32 * 584 = 18688
    __shared__ uint8_t q8[NH * QKD];          // 9216
    __shared__ uint8_t sc8[NH * TILE];        // 512
    // Total: ~28K (well within 160K limit)

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

    // === Q quantize with hw fp8 ===
    const uint16_t* qp = Q + bid * NH * QKD;
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += 64)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    amax = warp_max64(amax);
    float qs = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_qs = 1.0f / qs;
    for (int i = tid * 4; i < NH * QKD; i += 256) {
        float v0 = bf2f(qp[i + 0]) * inv_qs;
        float v1 = bf2f(qp[i + 1]) * inv_qs;
        float v2 = bf2f(qp[i + 2]) * inv_qs;
        float v3 = bf2f(qp[i + 3]) * inv_qs;
        int pk = cvt_pk_fp8_lo(v0, v1, 0);
        pk = cvt_pk_fp8_hi(v2, v3, pk);
        *(int*)(q8 + i) = pk;
    }

    float cqk = qs * kvs * sms;
    const float log2e = 1.4426950408889634f;

    v4f va[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) va[i] = (v4f){0, 0, 0, 0};
    float rmax[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float rsum[4] = {0, 0, 0, 0};

    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);

        // === Vectorized KV load (4 bytes at a time) ===
        const int kv_total = tsz * QKD;
        for (int i = tid * 4; i < kv_total; i += 256) {
            int tok = i / QKD;
            int dim = i % QKD;
            // Copy 4 bytes from global to LDS
            *(uint32_t*)(kv_lds + tok * KVS + dim) = *(const uint32_t*)(KV + (ts + tok) * QKD + dim);
        }
        // Handle remaining bytes
        for (int i = (kv_total / 4) * 4 + tid; i < kv_total; i += 64) {
            int tok = i / QKD;
            int dim = i % QKD;
            kv_lds[tok * KVS + dim] = KV[(ts + tok) * QKD + dim];
        }
        // Zero pad
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * KVS; i += 64)
                kv_lds[tsz * KVS + i] = 0;

        // === QK^T MFMA ===
        __builtin_amdgcn_s_setprio(15);
        v4f qk1 = (v4f){0,0,0,0}, qk2 = (v4f){0,0,0,0};
        #pragma unroll
        for (int ch = 0; ch < 18; ch++) {
            int kb = ch * 32;
            long a = *(long*)(q8 + l16 * QKD + kb + g * 8);
            long b1 = *(long*)(kv_lds + l16 * KVS + kb + g * 8);
            long b2 = *(long*)(kv_lds + (HALF + l16) * KVS + kb + g * 8);
            qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b1, qk1, 0, 0, 0);
            qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b2, qk2, 0, 0, 0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(0);

        // === Online softmax + hw fp8 score packing (fused, no float LDS) ===
        // D[head][token]: lane%16=token, GPR=head%4
        // Each lane has scores for 1 token across 4 heads (in GPR 0-3)
        // But we need per-head max across all 32 tokens -> need LDS or shuffle

        // Approach: store scores to registers, do max/sum via shuffle within groups
        // Score layout: qk1[hi] = score for (g*4+hi, l16), qk2[hi] = score for (g*4+hi, l16+16)

        v4f corr_v;
        #pragma unroll
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            float s_lo = qk1[hi] * cqk;  // score for token l16
            float s_hi = qk2[hi] * cqk;  // score for token l16+16

            // Max across 32 tokens: first max lo/hi, then reduce across 16 lanes
            float my_max = fmaxf(s_lo, s_hi);
            // Reduce max across 16 lanes (lanes 0-15 within each group share the same head)
            #pragma unroll
            for (int m = 8; m > 0; m >>= 1) my_max = fmaxf(my_max, __shfl_xor(my_max, m));
            // Now all 16 lanes in the group have the tile max for this head
            // But groups are independent (different k-ranges in MFMA -> same result due to reduction)
            // Actually each group already reduced the same values. my_max is correct.

            // Wait -- MFMA D output: lane%16 gives token index. All 4 groups (g=0..3) hold
            // scores for different heads (head=g*4+hi). Within each group, 16 lanes hold
            // 16 different tokens. So the max for head h is across:
            //   lanes where g = h/4: all 16 lanes (l16 = 0..15 = tokens 0..15 for qk1)
            //   PLUS l16+16 from qk2 (tokens 16..31)
            // So each lane has 2 scores (lo, hi), reduce max within the 16-lane group.
            // But __shfl_xor with mask 1,2,4,8 reduces within the 16 lanes of ONE group.
            // Different groups are at different lane offsets (g*16). __shfl_xor operates
            // within the full 64-lane wavefront, not within 16-lane subgroups!
            // Need to mask to only reduce within the lane group.

            // Actually: all 4 groups hold the SAME head data (A was broadcast across groups).
            // No -- each group processes a DIFFERENT k-range. The MFMA accumulates across all groups.
            // The output D[head][token] in each lane is the FULL dot product result.
            // So all 4 groups have identical scores for the same (head, token) pair.
            // The max reduction across 16 lanes is correct as-is.

            // However, we need max across 32 tokens: lo (lanes 0-15) + hi (from qk2).
            // The hi scores are in the same lanes but from qk2. We already took max(s_lo, s_hi).
            // The 16-lane reduction gives max across all 32 tokens. Correct!

            float nm = fmaxf(rmax[hi], my_max);
            float rescale = __builtin_amdgcn_exp2f((rmax[hi] - nm) * log2e);
            corr_v[hi] = rescale;
            rmax[hi] = nm;

            // Compute exp and local sum
            float e_lo = (l16 < tsz) ? __builtin_amdgcn_exp2f((s_lo - nm) * log2e) : 0.0f;
            float e_hi = (l16 + HALF < tsz) ? __builtin_amdgcn_exp2f((s_hi - nm) * log2e) : 0.0f;

            float my_sum = e_lo + e_hi;
            // Reduce sum across 16 lanes
            #pragma unroll
            for (int m = 8; m > 0; m >>= 1) my_sum += __shfl_xor(my_sum, m);

            rsum[hi] = rsum[hi] * rescale + my_sum;

            // Pack scores to fp8: 2 scores per lane -> 2 fp8 bytes
            // sc8 layout: sc8[head * 32 + token]
            // For V MFMA A operand: A[i=head][k=token], load sc8[l16*TILE + g*8 : +8]
            // So sc8 is indexed as [head][token], head=0..15, token=0..31
            // Each lane writes its 2 tokens:
            float e_lo_s = e_lo * 240.0f;
            float e_hi_s = e_hi * 240.0f;
            int pk = cvt_pk_fp8_lo(e_lo_s, e_hi_s, 0);
            // This packs: byte0 = fp8(e_lo_s), byte1 = fp8(e_hi_s)
            // We need sc8[head*32 + l16] = fp8(e_lo_s), sc8[head*32 + l16+16] = fp8(e_hi_s)
            // But pk has both in 2 bytes. Store byte-by-byte:
            sc8[head * TILE + l16] = (uint8_t)(pk & 0xFF);
            sc8[head * TILE + l16 + HALF] = (uint8_t)((pk >> 8) & 0xFF);
        }

        // Correct V accumulators
        #pragma unroll
        for (int vc = 0; vc < 32; vc++) va[vc] *= corr_v;

        // === V MFMA (no transpose needed!) ===
        // A = scores: A[i=head][k=token], lane%16=head
        //   load sc8[l16 * TILE + g*8 : +8] -> 8 consecutive tokens for this head
        // B = V: B[k=token][j=v_dim], lane%16=v_dim
        //   load kv_lds[token * KVS + v_dim] but need 8 consecutive TOKENS for same v_dim
        //   kv_lds is [token][dim], stride=KVS. 8 tokens = kv_lds[g*8 + t][dim] for t=0..7
        //   These are NOT contiguous (stride KVS between tokens)!
        //   Need to gather 8 bytes from 8 different rows.
        //
        // WAIT: For B[k][j], the register mapping says:
        //   lane%16 = j (v_dim), k = 8*(lane/16) + 4*gpr + byte
        //   8 bytes packed in int64: k_start = g*8, bytes 0-7 = k_start..k_start+7
        //   These 8 k-values (tokens) are at CONSECUTIVE positions.
        //   B register byte b = kv_lds[(g*8 + b) * KVS + (vb + l16)]
        //   = 8 bytes at stride KVS, NOT contiguous!
        //
        // So we DO need to gather for B. Options:
        // (a) Transpose V in LDS (what we had before)
        // (b) Gather 8 bytes manually into int64
        // (c) Use a transposed V buffer
        //
        // Let's try (b): gather 8 bytes into int64 per V chunk

        __builtin_amdgcn_s_setprio(15);
        long sa = *(long*)(sc8 + l16 * TILE + g * 8);

        #pragma unroll
        for (int vc = 0; vc < 32; vc++) {
            int vd = vc * 16 + l16;  // v_dim index
            // Gather 8 bytes: kv_lds[(g*8+0)*KVS + vd], ..., kv_lds[(g*8+7)*KVS + vd]
            uint8_t b0 = kv_lds[(g*8+0)*KVS + vd];
            uint8_t b1 = kv_lds[(g*8+1)*KVS + vd];
            uint8_t b2 = kv_lds[(g*8+2)*KVS + vd];
            uint8_t b3 = kv_lds[(g*8+3)*KVS + vd];
            uint8_t b4 = kv_lds[(g*8+4)*KVS + vd];
            uint8_t b5 = kv_lds[(g*8+5)*KVS + vd];
            uint8_t b6 = kv_lds[(g*8+6)*KVS + vd];
            uint8_t b7 = kv_lds[(g*8+7)*KVS + vd];
            long sb = (long)b0 | ((long)b1 << 8) | ((long)b2 << 16) | ((long)b3 << 24) |
                      ((long)b4 << 32) | ((long)b5 << 40) | ((long)b6 << 48) | ((long)b7 << 56);
            va[vc] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(sa, sb, va[vc], 0, 0, 0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(0);
    }

    // Write output
    for (int hi = 0; hi < 4; hi++) {
        int head = g * 4 + hi;
        int li = (bid * ns + sid) * NH + head;
        int ob = ((bid * ns + sid) * NH + head) * VD;
        if (l16 == 0)
            sL[li] = (rsum[hi] > 0.0f) ? (__logf(rsum[hi]) + rmax[hi]) : -1e20f;
        float inv = (rsum[hi] > 0.0f) ? (kvs / (240.0f * rsum[hi])) : 0.0f;
        #pragma unroll
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
    for (int s = 0; s < ns; s++)
        ml = fmaxf(ml, sL[(bid * ns + s) * NH + hid]);
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    float tw = 0.0f;
    for (int s = 0; s < ns; s++) {
        float l = sL[(bid * ns + s) * NH + hid];
        if (l <= -1e19f) continue;
        float w = __expf(l - ml); tw += w;
        int base = ((bid * ns + s) * NH + hid) * VD + lid * 32;
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[i] += w * sO[base + i];
    }
    float inv = (tw > 0.0f) ? (1.0f / tw) : 0.0f;
    int ob = (bid * NH + hid) * VD + lid * 32;
    #pragma unroll
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
        _mod[0] = load_inline(name='mla_e2e_opt1', cpp_sources='',
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
