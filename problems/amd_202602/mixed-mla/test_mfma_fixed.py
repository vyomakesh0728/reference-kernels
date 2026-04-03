#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MFMA QK^T with CORRECT register mapping + scalar V.
Key fix: D=A*B (not A*B^T). B must be transposed: B[k][j] = KV[dim][token].
A[i][k]: lane%16=i(head), lane/16=k_group. B[k][j]: lane%16=j(token), lane/16=k_group.
D[i][j]: lane%16=j(token), lane/16=i_group. GPR=i%4."""
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

// MFMA register mapping for V_MFMA_F32_16X16X32_FP8_FP8:
//   D = A * B  (NOT A * B^T!)
//   A[i][k]: i=lane%16, k=8*(lane/16) + 4*gpr + byte  (i=row=head)
//   B[k][j]: j=lane%16, k=8*(lane/16) + 4*gpr + byte  (j=col=token)
//   D[i][j]: j=lane%16, i=4*(lane/16) + gpr            (j=col=token)
//
// For QK^T: D[head][token] = sum_k Q[head][k] * K_T[k][token]
//   A = Q[head][k]  -> lane%16 selects head (good, head < 16)
//   B = K_T[k][token] -> lane%16 selects token (good, token < 16 per tile)
//   D = scores[head][token] -> lane%16 = token, GPR = head%4, lane/16 = head/4

__global__ void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs, const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    __shared__ uint8_t kv_lds[TILE * KVS];
    __shared__ uint8_t q8[NH * QKD];
    __shared__ float sc[NH * TILE];
    // Transposed KV tile in LDS: [dim][token] for B operand
    __shared__ uint8_t kvt_lds[QKD * TILE];  // 576 * 16 = 9216 bytes

    const int tid = threadIdx.x;
    const int sid = blockIdx.x, bid = blockIdx.y;
    // For A: lane%16 = head index (i)
    // For B: lane%16 = token index (j)
    // For D: lane%16 = token index (j), GPR = head%4, lane/16 = head/4
    const int g = tid >> 4;     // lane group 0..3 (k_group for A/B, head_group for D)
    const int l16 = tid & 15;  // lane%16

    const int k0 = kvi[bid], k1 = kvi[bid + 1];
    const int slen = (k1 - k0 + ns - 1) / ns;
    const int ms = k0 + sid * slen;
    const int me = min(ms + slen, k1);

    // D output: head = 4*g + gpr, token = l16
    // But we have 16 heads and 16 tokens -> 4 groups * 4 GPR rows = 16 heads
    // For output indexing: head h's data is in lane where g=h/4, GPR=h%4, token=l16

    // For split output/LSE
    // We need to reorganize: for each head, collect all token scores

    if (ms >= k1) {
        // Zero out
        for (int h = 0; h < 4; h++) {
            int head = g * 4 + h;
            int li = (bid * ns + sid) * NH + head;
            int ob = ((bid * ns + sid) * NH + head) * VD;
            if (h == 0) sL[li] = -1e20f;
            for (int d = l16; d < VD; d += 16) sO[ob + d] = 0.0f;
        }
        return;
    }

    // Quantize Q -> fp8
    // A[i=head][k]: lane%16 = head. For each lane, load Q[head=l16][k_range]
    // k_range = 8*g + byte offsets across 2 VGPRs
    // But we need to store Q in a format where loading *(long*) gives the right mapping
    // Q stored as q8[head * QKD + k], load q8[l16 * QKD + 8*g : +8] for A register
    const uint16_t* qp = Q + bid * NH * QKD;
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += 64)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    amax = warp_max64(amax);
    float qs = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_qs = 240.0f / fmaxf(amax, 1e-12f);
    for (int i = tid; i < NH * QKD; i += 64)
        q8[i] = f2fp8(bf2f(qp[i]) * inv_qs);

    // V accumulators: each thread handles 4 heads (g*4..g*4+3) x some V dims
    // Each thread covers token l16's contribution... no, we need per-head V accumulators
    // D output for V MFMA would be similar. But for scalar V, we need different mapping.
    // For scalar V: each of the 4 heads this thread handles, accumulate VD dims.
    // But we only have 16 lanes to cover 512 dims per head.
    // Each lane handles VD/16 = 32 V dims per head, for 4 heads = 128 total.
    float va[4][32];  // [head_within_group][v_dim_chunk]
    for (int h = 0; h < 4; h++)
        for (int d = 0; d < 32; d++) va[h][d] = 0.0f;
    float rmax[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float rsum[4] = {0, 0, 0, 0};
    float cqk = qs * kvs * sms;
    int vbase = l16 * 32;  // each lane handles 32 consecutive V dims

    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);

        // Load KV tile to LDS (row-major: [token][dim])
        for (int i = tid; i < tsz * QKD; i += 64)
            kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

        // Transpose KV for B operand: kvt_lds[k][j] = kv_lds[j][k]
        // 576 * 16 = 9216 bytes, 64 threads: each thread transposes 144 bytes
        for (int i = tid; i < QKD * tsz; i += 64) {
            int k = i / tsz;  // dim index
            int j = i % tsz;  // token index
            kvt_lds[k * TILE + j] = kv_lds[j * KVS + k];
        }
        // Zero-pad transposed buffer for empty token slots
        if (tsz < TILE) {
            for (int i = tid; i < QKD * (TILE - tsz); i += 64) {
                int k = i / (TILE - tsz);
                int j_off = i % (TILE - tsz);
                kvt_lds[k * TILE + tsz + j_off] = 0;
            }
        }

        // MFMA QK^T: 18 chunks of 32 K-dims
        // A[i=head][k]: load from q8[head=l16][k_start + ...]
        //   int64 packing: 8 bytes = k_start..k_start+7 for this lane group
        //   k_start = g * 8 (for the chunk's base) NO WAIT:
        //   Each chunk processes 32 K-dims. The 32 K-dims are split across 4 groups:
        //   group 0: k=0..7, group 1: k=8..15, group 2: k=16..23, group 3: k=24..31
        //   So for chunk ch, k_base = ch*32, lane group g handles k = k_base + g*8 .. + 7
        //
        //   A register: load q8[l16 * QKD + ch*32 + g*8 : +8]  (l16=head, g=k_group)
        //
        // B[k][j=token]: load from kvt_lds[k][j=l16]
        //   int64 packing: 8 bytes = k_start..k_start+7 for column j=l16
        //   k_start = g * 8 (within the 32-dim chunk)
        //   B register: load kvt_lds[(ch*32 + g*8) * TILE + l16 : stride TILE, 8 elements]
        //   BUT int64 load needs 8 CONTIGUOUS bytes. kvt_lds layout: [k][token], stride=TILE=16.
        //   Consecutive k values for same token: kvt_lds[k*16 + l16] are 16 bytes apart!
        //   Need to pack 8 k-values for token l16 into int64.
        //   Can't just *(long*) -- need to gather from stride-16 layout.

        // SOLUTION: pack B data into contiguous layout in LDS
        // Or: rearrange the transpose so consecutive k-values for same token are contiguous
        // kvt2_lds[j * QKD + k] would give contiguous k for each token... but that's [token][dim] = original!
        //
        // The issue: B[k][j] needs contiguous k for each j, but our natural layout is [token][dim].
        // For *(long*) loading, we need 8 consecutive bytes = 8 consecutive k values for one j.
        // Layout: B_packed[j][k] with j-major, k-minor -> exactly [token][dim] = original kv_lds!
        //
        // Wait. B[k][j] in the MFMA means: the matrix B has k as row, j as column.
        // But the register mapping says: lane%16 = j, k = 8*(lane/16) + 4*gpr + byte
        // Each lane holds 8 B values for its column j, at k positions within a k-group.
        //
        // The 8 bytes in the int64 for lane L are B[k_start+0][j], B[k_start+1][j], ..., B[k_start+7][j]
        // where k_start = 8*(L/16) and j = L%16.
        //
        // These 8 values are from 8 DIFFERENT rows of B (same column j).
        // In our kv_lds[token][dim] layout: B[k][j] = KV_T[k][j] = kv_lds[j][k]
        // So the 8 values are: kv_lds[j][k_base+0], kv_lds[j][k_base+1], ..., kv_lds[j][k_base+7]
        // = 8 consecutive bytes at kv_lds[j * KVS + k_base]!
        //
        // THIS IS CONTIGUOUS! We can use *(long*)(kv_lds + j*KVS + ch*32 + g*8) directly!
        // The original kv_lds[token][dim] layout IS the correct B layout for the MFMA!
        // No transpose needed!

        v4f qk = (v4f){0,0,0,0};
        for (int ch = 0; ch < 18; ch++) {
            int k_base = ch * 32;
            // A[i=head][k]: 8 bytes of Q for head l16, dims k_base+g*8..+7
            long a = *(long*)(q8 + l16 * QKD + k_base + g * 8);
            // B[k][j=token]: 8 bytes of KV for token l16, dims k_base+g*8..+7
            // From kv_lds[token=l16][dim=k_base+g*8], 8 consecutive bytes
            long b = *(long*)(kv_lds + l16 * KVS + k_base + g * 8);
            qk = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, qk, 0, 0, 0);
        }

        // D[i=head][j=token]: lane%16=j=token, GPR=head%4, lane/16=head/4
        // So this thread (lane group g) holds scores for:
        //   heads g*4+0, g*4+1, g*4+2, g*4+3 (in GPR 0,1,2,3)
        //   token l16
        // qk[0] = D[g*4+0][l16], qk[1] = D[g*4+1][l16], qk[2] = D[g*4+2][l16], qk[3] = D[g*4+3][l16]

        // Store to LDS: sc[head * TILE + token]
        sc[(g*4+0) * TILE + l16] = qk[0] * cqk;
        sc[(g*4+1) * TILE + l16] = qk[1] * cqk;
        sc[(g*4+2) * TILE + l16] = qk[2] * cqk;
        sc[(g*4+3) * TILE + l16] = qk[3] * cqk;

        // Softmax + scalar V for each of the 4 heads this thread handles
        for (int hi = 0; hi < 4; hi++) {
            int head = g * 4 + hi;
            float tmax = -1e20f;
            for (int t = 0; t < tsz; t++) tmax = fmaxf(tmax, sc[head * TILE + t]);
            float nm = fmaxf(rmax[hi], tmax);
            float corr = __expf(rmax[hi] - nm);
            rmax[hi] = nm;
            for (int d = 0; d < 32; d++) va[hi][d] *= corr;
            float tsum = 0.0f;
            for (int t = 0; t < tsz; t++) {
                float es = __expf(sc[head * TILE + t] - rmax[hi]);
                tsum += es;
                for (int d = 0; d < 32; d++) {
                    float vv = fp82f(kv_lds[t * KVS + vbase + d]) * kvs;
                    va[hi][d] += es * vv;
                }
            }
            rsum[hi] = rsum[hi] * corr + tsum;
        }
    }

    // Write output: each thread writes 4 heads x 32 V dims
    for (int hi = 0; hi < 4; hi++) {
        int head = g * 4 + hi;
        int li = (bid * ns + sid) * NH + head;
        int ob = ((bid * ns + sid) * NH + head) * VD;
        if (l16 == 0)
            sL[li] = (rsum[hi] > 0.0f) ? (__logf(rsum[hi]) + rmax[hi]) : -1e20f;
        float inv = (rsum[hi] > 0.0f) ? (1.0f / rsum[hi]) : 0.0f;
        for (int d = 0; d < 32; d++)
            sO[ob + vbase + d] = va[hi][d] * inv;
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
        _mod[0] = load_inline(name='mla_mfma_fixed', cpp_sources='',
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
