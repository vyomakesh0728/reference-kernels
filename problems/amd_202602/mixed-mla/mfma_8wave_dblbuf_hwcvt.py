#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Full MFMA kernel: 8 waves, double-buffered LDS, hw fp8 cvt, scheduling hints.
QK^T: fp8 MFMA 16x16x32. V: fp8 MFMA 16x16x32 with hw-packed scores.
8 wavefronts x 64 = 512 threads. Each wave handles 2 heads (16 heads / 8 waves).
32-token KV tiles, ping-pong LDS buffers."""
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
#define NWAVES 8
#define NTHREADS (NWAVES * 64)
#define HEADS_PER_WAVE 2
// KV LDS: 2 buffers x 32 tokens x 580 bytes (padded for bank conflicts)
#define KVS 580
#define KV_BUF_SIZE (TILE * KVS)

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
__device__ __forceinline__ float warp_max(float v) {
    for (int m = 32; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
    return v;
}

// Grid: (nsplits, bs), Block: (512,) = 8 wavefronts
// Each wave handles 2 heads. 8 waves x 2 heads = 16 heads.
__global__ __launch_bounds__(512, 1) void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs, const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    __shared__ uint8_t kv_lds[2 * KV_BUF_SIZE];  // double buffer
    __shared__ uint8_t q8[NH * QKD];
    __shared__ float   sc[NH * TILE];

    const int tid = threadIdx.x;         // 0..511
    const int wid = tid / 64;            // wave 0..7
    const int lid = tid % 64;            // lane within wave
    const int g = lid >> 4;              // lane group 0..3
    const int l16 = lid & 15;            // lane % 16

    const int sid = blockIdx.x, bid = blockIdx.y;
    const int k0 = kvi[bid], k1 = kvi[bid + 1];
    const int slen = (k1 - k0 + ns - 1) / ns;
    const int ms = k0 + sid * slen;
    const int me = min(ms + slen, k1);

    // Each wave handles 2 heads: head0 = wid*2, head1 = wid*2+1
    const int head0 = wid * HEADS_PER_WAVE;

    if (ms >= k1) {
        for (int hi = 0; hi < HEADS_PER_WAVE; hi++) {
            int head = head0 + hi;
            if (lid == 0) sL[(bid * ns + sid) * NH + head] = -1e20f;
            int ob = ((bid * ns + sid) * NH + head) * VD;
            for (int d = lid; d < VD; d += 64) sO[ob + d] = 0.0f;
        }
        return;
    }

    // === Cooperative Q quantization (all 512 threads) ===
    const uint16_t* qp = Q + bid * NH * QKD;
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += NTHREADS)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    // Cross-wave reduction via LDS
    __shared__ float wave_max[NWAVES];
    float wm = warp_max(amax);
    if (lid == 0) wave_max[wid] = wm;
    __syncthreads();
    if (tid == 0) {
        float gm = 0;
        for (int w = 0; w < NWAVES; w++) gm = fmaxf(gm, wave_max[w]);
        wave_max[0] = gm;
    }
    __syncthreads();
    amax = wave_max[0];

    float qs = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_qs = 240.0f / fmaxf(amax, 1e-12f);
    for (int i = tid; i < NH * QKD; i += NTHREADS)
        q8[i] = f2fp8(bf2f(qp[i]) * inv_qs);
    __syncthreads();

    // === Per-wave V accumulators ===
    // Each wave processes 2 heads, MFMA output has 4 GPR rows per head group
    // For 2 heads: use 2 separate accumulator sets of 32 v4f each
    // But lane%16 gives v_dim for V MFMA D output, so each lane covers 32 v_dim chunks
    // Actually with 2 heads per wave, we can do 2 separate MFMA calls per V chunk
    // Simpler: accumulate per-head with scalar V (for correctness first)

    float va[2][32];  // [head_idx 0/1][v_dim_chunk]
    for (int h = 0; h < 2; h++)
        for (int d = 0; d < 32; d++) va[h][d] = 0.0f;
    float rmax[2] = {-1e20f, -1e20f};
    float rsum[2] = {0, 0};
    float cqk = qs * kvs * sms;
    int vbase = l16 * 32;  // each lane handles 32 V dims

    int cur = 0;  // current LDS buffer index

    // === Preload first KV tile ===
    int first_tsz = min(TILE, me - ms);
    for (int i = tid; i < first_tsz * QKD; i += NTHREADS)
        (kv_lds + cur * KV_BUF_SIZE)[(i / QKD) * KVS + (i % QKD)] = KV[(ms + i / QKD) * QKD + (i % QKD)];
    if (first_tsz < TILE)
        for (int i = tid; i < (TILE - first_tsz) * QKD; i += NTHREADS)
            (kv_lds + cur * KV_BUF_SIZE)[(first_tsz + i / QKD) * KVS + (i % QKD)] = 0;
    __syncthreads();

    // === Main tile loop with double buffering ===
    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);
        int next_ts = ts + TILE;
        int nxt = 1 - cur;

        // --- Async load next tile (all threads cooperate) ---
        __builtin_amdgcn_s_setprio(0);  // low priority for memory
        if (next_ts < me) {
            int next_tsz = min(TILE, me - next_ts);
            for (int i = tid; i < next_tsz * QKD; i += NTHREADS)
                (kv_lds + nxt * KV_BUF_SIZE)[(i / QKD) * KVS + (i % QKD)] = KV[(next_ts + i / QKD) * QKD + (i % QKD)];
            if (next_tsz < TILE)
                for (int i = tid; i < (TILE - next_tsz) * QKD; i += NTHREADS)
                    (kv_lds + nxt * KV_BUF_SIZE)[(next_tsz + i / QKD) * KVS + (i % QKD)] = 0;
        }

        // --- Compute on current tile ---
        __builtin_amdgcn_s_setprio(15);  // high priority for compute
        int kv_off = cur * KV_BUF_SIZE;

        // QK^T MFMA for this wave's 2 heads
        for (int hi = 0; hi < HEADS_PER_WAVE; hi++) {
            int head = head0 + hi;
            v4f qk1 = (v4f){0,0,0,0}, qk2 = (v4f){0,0,0,0};
            for (int ch = 0; ch < 18; ch++) {
                int kb = ch * 32;
                long a = *(long*)(q8 + head * QKD + kb + g * 8);
                long b1 = *(long*)(kv_lds + kv_off +l16 * KVS + kb + g * 8);
                long b2 = *(long*)(kv_lds + kv_off +(HALF + l16) * KVS + kb + g * 8);
                qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b1, qk1, 0, 0, 0);
                qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b2, qk2, 0, 0, 0);
            }

            // Extract scores: D[i][j] = lane%16 is j(token), GPR is i%4
            // For head: we loaded A with head index. For 16x16 MFMA,
            // A[i=head_within_16][k] -> but we're doing 1 head at a time
            // Actually head is in lane%16 position of A (l16 doesn't change per head)
            // Wait: A operand uses l16 = head. But we iterate head0, head0+1...
            // A is loaded from q8[head * QKD + ...], so lane l16 holds data for 'head',
            // but MFMA A[i][k] has i=l16, meaning all 16 lanes contribute different i values.
            // Since we load q8[head * QKD + ...] (same head for all lanes), all lanes see
            // the same head's Q data but at different k positions.
            // This means MFMA computes: D[i=sameHead?][j=token] -- NO!
            //
            // Actually: A[i][k], lane%16 = i. Each lane loads q8[head * QKD + kb + g*8].
            // This is Q[head][k], same for all i (all lanes load same head).
            // So effectively A is a rank-1 matrix (same row repeated 16 times).
            // D[i][j] = sum_k A[i][k] * B[k][j] = sum_k Q[head][k] * KV[k][j] for all i
            // Result: D[0][j] = D[1][j] = ... = D[15][j] = Q[head] dot KV[j]
            // The scores are replicated across all 16 i values.
            // We just need D[0][j] -> GPR 0, lane l16 = token j.

            // Scores for tokens 0-15 in qk1[0], tokens 16-31 in qk2[0]
            // (GPR 0 = i%4 = 0, which is fine since all rows are identical)
            sc[head * TILE + l16]        = qk1[0] * cqk;
            sc[head * TILE + HALF + l16] = qk2[0] * cqk;
        }

        __builtin_amdgcn_sched_barrier(0);

        // Online softmax + scalar V for this wave's heads
        for (int hi = 0; hi < HEADS_PER_WAVE; hi++) {
            int head = head0 + hi;
            float tmax = -1e20f;
            for (int t = 0; t < tsz; t++) tmax = fmaxf(tmax, sc[head * TILE + t]);
            // Need cross-lane max? No, each lane reads all 32 scores from LDS
            float nm = fmaxf(rmax[hi], tmax);
            float corr = __expf(rmax[hi] - nm);
            rmax[hi] = nm;
            for (int d = 0; d < 32; d++) va[hi][d] *= corr;
            float tsum = 0.0f;
            for (int t = 0; t < tsz; t++) {
                float es = __expf(sc[head * TILE + t] - rmax[hi]);
                tsum += es;
                for (int d = 0; d < 32; d++) {
                    float vv = fp82f(kv_lds[kv_off +t * KVS + vbase + d]) * kvs;
                    va[hi][d] += es * vv;
                }
            }
            rsum[hi] = rsum[hi] * corr + tsum;
        }

        __builtin_amdgcn_s_setprio(0);
        __syncthreads();  // ensure next tile load is complete
        cur = nxt;        // swap buffers
    }

    // Write output
    for (int hi = 0; hi < HEADS_PER_WAVE; hi++) {
        int head = head0 + hi;
        int li = (bid * ns + sid) * NH + head;
        int ob = ((bid * ns + sid) * NH + head) * VD;
        if (lid == 0)
            sL[li] = (rsum[hi] > 0.0f) ? (__logf(rsum[hi]) + rmax[hi]) : -1e20f;
        float inv = (rsum[hi] > 0.0f) ? (1.0f / rsum[hi]) : 0.0f;
        for (int d = 0; d < 32; d++)
            sO[ob + vbase + d] = va[hi][d] * inv;
    }
}

// Reduce: same as before
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
    stage1<<<dim3(ns, bs), NTHREADS>>>(
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
        _mod[0] = load_inline(name='mla_8wave_v2', cpp_sources='',
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
