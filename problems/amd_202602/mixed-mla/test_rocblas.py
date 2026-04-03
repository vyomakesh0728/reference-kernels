#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""TEST 3: torch.matmul for QK^T (uses rocBLAS under the hood) + scalar V.
No MFMA intrinsics, no custom fp8 conversion. Pure PyTorch ops.
This isolates: does the split-K + online softmax + scalar V logic work correctly
when QK^T is computed by a known-correct GEMM?"""
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
__device__ __forceinline__ float bf2f(uint16_t b) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)b) << 16; return c.f;
}

__device__ void build_lut(float* lut, int tid) {
    if (tid < 256) {
        uint8_t x = (uint8_t)tid;
        float v = 0.0f;
        if (x != 0 && x != 0x80) {
            int s = (x >> 7) & 1;
            int e = (x >> 3) & 0xF;
            int m = x & 0x7;
            if (e == 0) v = ldexpf((float)m, -10);
            else v = ldexpf(1.0f + (float)m * 0.125f, e - 8);
            if (s) v = -v;
        }
        lut[tid] = v;
    }
}

// Uses pre-computed QK scores (from torch.matmul in Python) + scalar V
// Grid: (nsplits, bs), Block: (64,)
__global__ void stage1_with_scores(
    const float* __restrict__ QK_scores,  // (bs, nsplits, 16, tile_tokens) pre-computed
    const uint8_t* __restrict__ KV,
    float kv_scale,
    const int32_t* __restrict__ kvi,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, int max_split_len
) {
    __shared__ uint8_t kv_lds[TILE * KVS];
    __shared__ float lut[256];

    const int tid = threadIdx.x;
    const int sid = blockIdx.x, bid = blockIdx.y;
    const int h = tid & 15, g = tid >> 4;

    build_lut(lut, tid);

    const int k0 = kvi[bid], k1 = kvi[bid + 1];
    const int slen = (k1 - k0 + ns - 1) / ns;
    const int ms = k0 + sid * slen;
    const int me = min(ms + slen, k1);
    const int my_len = me - ms;
    const int li = (bid * ns + sid) * NH + h;
    const int ob = ((bid * ns + sid) * NH + h) * VD;

    if (ms >= k1) {
        if (g == 0) sL[li] = -1e20f;
        for (int d = g; d < VD; d += 4) sO[ob + d] = 0.0f;
        return;
    }

    // Pre-computed scores pointer
    const float* my_scores = QK_scores + ((int64_t)bid * ns + sid) * NH * max_split_len + h * max_split_len;

    float va[128];
    for (int i = 0; i < 128; i++) va[i] = 0.0f;
    float rmax = -1e20f, rsum = 0.0f;
    int vbase = g * 128;

    for (int tile_off = 0; tile_off < my_len; tile_off += TILE) {
        int tsz = min(TILE, my_len - tile_off);
        int ts = ms + tile_off;

        for (int i = tid; i < tsz * QKD; i += 64)
            kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

        float tmax = -1e20f;
        for (int i = 0; i < tsz; i++)
            tmax = fmaxf(tmax, my_scores[tile_off + i]);
        float nm = fmaxf(rmax, tmax);
        float corr = __expf(rmax - nm);
        rmax = nm;
        for (int i = 0; i < 128; i++) va[i] *= corr;

        float tsum = 0.0f;
        for (int t = 0; t < tsz; t++) {
            float es = __expf(my_scores[tile_off + t] - rmax);
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
    torch::Tensor QK_scores,  // (bs, ns, 16, max_split_len)
    torch::Tensor KV, double kv_scale,
    torch::Tensor kvi,
    torch::Tensor so, torch::Tensor sl, torch::Tensor out,
    int64_t bs, int64_t ns, int64_t max_split_len
) {
    stage1_with_scores<<<dim3(ns, bs), 64>>>(
        QK_scores.data_ptr<float>(),
        (const uint8_t*)KV.data_ptr(),
        (float)kv_scale, kvi.data_ptr<int32_t>(),
        so.data_ptr<float>(), sl.data_ptr<float>(),
        (int)ns, (int)max_split_len);
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
        _mod[0] = load_inline(name='mla_test3', cpp_sources='',
                               cuda_sources=_SRC, extra_cuda_cflags=['-O3'],
                               verbose=True)
    return _mod[0]

from aiter import dtypes as aiter_dtypes
FP8_DTYPE = aiter_dtypes.fp8
_fp8_finfo = torch.finfo(FP8_DTYPE)

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

    # Compute QK^T using torch (rocBLAS): Q_bf16 @ KV_dequant^T
    # Dequant KV: fp8 -> bf16
    kv_bf16 = kv_data['bf16']  # (total_kv, 1, 576) bf16
    kv_flat = kv_bf16.view(-1, QK_DIM)  # (total_kv, 576)
    q_flat = q.view(bs, NUM_HEADS, QK_DIM)  # (bs, 16, 576)

    # Compute per-split scores
    slen = (kvl + ns - 1) // ns
    max_split_len = slen
    # Pre-allocate score buffer
    qk_scores = torch.zeros((bs, ns, NUM_HEADS, max_split_len), dtype=torch.float32, device=q.device)

    for s in range(ns):
        start = s * slen
        end = min(start + slen, kvl)
        if end <= start:
            continue
        for b in range(bs):
            kv_start = kv_indptr[b].item() + start
            kv_end = kv_indptr[b].item() + end
            kv_chunk = kv_flat[kv_start:kv_end]  # (chunk_len, 576)
            # QK^T = Q @ K^T: (16, 576) @ (576, chunk_len) -> (16, chunk_len)
            scores = torch.matmul(q_flat[b].float(), kv_chunk.float().t()) * SM_SCALE
            qk_scores[b, s, :, :end-start] = scores

    key = (bs, kvl, ns)
    if key not in _c:
        d = q.device
        _c[key] = (
            torch.empty((bs, NUM_HEADS, V_DIM), dtype=torch.bfloat16, device=d),
            torch.empty((bs, ns, NUM_HEADS, V_DIM), dtype=torch.float32, device=d),
            torch.empty((bs, ns, NUM_HEADS), dtype=torch.float32, device=d),
        )
    out, so, sl = _c[key]
    m.mla_fwd(qk_scores, kv_fp8, kv_scale.item(), kv_indptr,
              so, sl, out, bs, ns, max_split_len)
    return out
