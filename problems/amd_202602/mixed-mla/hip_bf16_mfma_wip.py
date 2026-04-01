#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Primary: split-K MLA HIP kernel (fp8 MFMA QK, bf16 softmax weights + bf16 V MFMA).
Falls back to hybrid aiter if JIT fails or batching is non-decode (q_seq_len != 1)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd
from aiter.ops.quant import dynamic_per_tensor_quant

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

FP8_DTYPE = aiter_dtypes.fp8
_cache: dict = {}

_HIP_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

typedef float v4f __attribute__((ext_vector_type(4)));
typedef short v8s __attribute__((ext_vector_type(8)));

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
__device__ __forceinline__ uint16_t f2bf_trunc(float v) {
    union { float f; uint32_t u; } c;
    c.f = v;
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

__global__ void stage1(
    const uint16_t* __restrict__ Q,
    const uint8_t*  __restrict__ KV,
    float kvs,
    const int32_t* __restrict__ kvi,
    const int32_t* __restrict__ qo,
    float* __restrict__ sO, float* __restrict__ sL,
    int ns, float sms
) {
    __shared__ uint8_t kv_lds[TILE * KVS];
    __shared__ uint8_t q8[NH * QKD];
    __shared__ float   sc[NH * TILE];
    __shared__ uint16_t sc_bf16[NH * TILE];
    __shared__ uint16_t kvv_bf16[VD * TILE];

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

    const int q0 = qo[bid];
    const uint16_t* qp = Q + static_cast<int64_t>(q0) * NH * QKD;
    float amax = 0.0f;
    for (int i = tid; i < NH * QKD; i += 64)
        amax = fmaxf(amax, fabsf(bf2f(qp[i])));
    amax = warp_max64(amax);
    float qs = fmaxf(amax, 1e-12f) / 240.0f;
    float inv_qs = 240.0f / fmaxf(amax, 1e-12f);
    for (int i = tid; i < NH * QKD; i += 64)
        q8[i] = f2fp8(bf2f(qp[i]) * inv_qs);

    v4f va[32];
    for (int i = 0; i < 32; i++) va[i] = (v4f){0, 0, 0, 0};
    float rmax[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float rsum[4] = {0, 0, 0, 0};
    float cqk = qs * kvs * sms;

    for (int ts = ms; ts < me; ts += TILE) {
        int tsz = min(TILE, me - ts);

        for (int i = tid; i < tsz * QKD; i += 64)
            kv_lds[(i / QKD) * KVS + (i % QKD)] = KV[(ts + i / QKD) * QKD + (i % QKD)];
        if (tsz < TILE)
            for (int i = tid; i < (TILE - tsz) * QKD; i += 64)
                kv_lds[(tsz + i / QKD) * KVS + (i % QKD)] = 0;

        v4f qk1 = (v4f){0,0,0,0}, qk2 = (v4f){0,0,0,0};
        for (int ch = 0; ch < 18; ch++) {
            int kb = ch * 32;
            long a = *(long*)(q8 + l16 * QKD + kb + g * 8);
            long b1 = *(long*)(kv_lds + l16 * KVS + kb + g * 8);
            long b2 = *(long*)(kv_lds + (HALF + l16) * KVS + kb + g * 8);
            qk1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b1, qk1, 0, 0, 0);
            qk2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b2, qk2, 0, 0, 0);
        }

        for (int hi = 0; hi < 4; hi++) {
            sc[(g*4+hi) * TILE + l16]        = qk1[hi] * cqk;
            sc[(g*4+hi) * TILE + HALF + l16] = qk2[hi] * cqk;
        }

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

            for (int t = 0; t < TILE; t++)
                sc_bf16[head * TILE + t] = f2bf_trunc(sc[head * TILE + t]);
        }

        for (int vc = 0; vc < 32; vc++) va[vc] *= corr_v;

        for (int i = tid; i < TILE * VD; i += 64) {
            int tok = i / VD;
            int dim = i % VD;
            float fv = fp82f(kv_lds[tok * KVS + dim]);
            kvv_bf16[dim * TILE + tok] = f2bf_trunc(fv);
        }

        v8s sa;
        for (int k = 0; k < 8; k++)
            sa[k] = (short)sc_bf16[l16 * TILE + g * 8 + k];

        for (int vc = 0; vc < 32; vc++) {
            int vb = vc * 16;
            v8s sb;
            for (int k = 0; k < 8; k++)
                sb[k] = (short)kvv_bf16[(vb + l16) * TILE + g * 8 + k];
            va[vc] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(sa, sb, va[vc], 0, 0, 0);
        }
    }

    float vs = kvs;
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
    torch::Tensor Q, torch::Tensor KV, double kvs, torch::Tensor kvi, torch::Tensor qo,
    torch::Tensor so, torch::Tensor sl, torch::Tensor out,
    int64_t bs, int64_t ns, double sms
) {
    stage1<<<dim3(ns, bs), 64>>>(
        (const uint16_t*)Q.data_ptr(), (const uint8_t*)KV.data_ptr(),
        (float)kvs, kvi.data_ptr<int32_t>(), qo.data_ptr<int32_t>(),
        so.data_ptr<float>(), sl.data_ptr<float>(),
        (int)ns, (float)sms);
    reduce_k<<<bs, 256>>>(
        so.data_ptr<float>(), sl.data_ptr<float>(),
        (uint16_t*)out.data_ptr(), (int)ns);
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mla_fwd", &mla_fwd); }
"""

_hip_mod = None
_hip_failed: bool | None = None
_path_marks: set[str] = set()


def _diag(msg: str) -> None:
    print(f"[mixed-mla submission] {msg}", file=sys.stderr, flush=True)


def _hip_kv_splits(bs: int, kvl: int) -> int:
    if kvl <= 1024:
        return 8 if bs <= 32 else 4
    if kvl <= 4096:
        return 16 if bs <= 64 else 8
    return 32


def _get_hip():
    global _hip_mod, _hip_failed
    if _hip_failed is True:
        return None
    if _hip_mod is not None:
        return _hip_mod
    verbose_jit = os.environ.get("MIXED_MLA_HIP_VERBOSE", "1") != "0"
    try:
        if not torch.cuda.is_available():
            _hip_failed = True
            _diag("HIP JIT skipped: torch.cuda.is_available() is False")
            return None
        _diag(
            "HIP JIT: compiling extension 'mla_submission_hip_bf16v' "
            f"(set MIXED_MLA_HIP_VERBOSE=0 to silence hipcc output) verbose={verbose_jit}"
        )
        _hip_mod = load_inline(
            name="mla_submission_hip_bf16v",
            cpp_sources="",
            cuda_sources=_HIP_SRC,
            extra_cuda_cflags=["-O3"],
            verbose=verbose_jit,
        )
        _hip_failed = False
        _diag(
            "HIP JIT OK: mla_submission_hip_bf16v loaded; custom_kernel will use HIP when eligible"
        )
        return _hip_mod
    except Exception as e:
        _hip_failed = True
        _hip_mod = None
        _diag(f"HIP JIT FAILED: {type(e).__name__}: {e}")
        return None


def _get_config(bs, kvl):
    if kvl <= 1024:
        if bs <= 32:
            return (8, False, 2, True)
        if bs <= 64:
            return (4, False, 2, True)
        return (4, False, 2, True)
    else:
        if bs <= 4:
            return (32, False, 2, True)
        if bs <= 32:
            return (8, True, 1, False)
        if bs <= 64:
            return (8, True, 1, False)
        return (16, True, 1, False)


def _get_or_build(bs, kvl, qd, kvd, qo, kvi, ns, dev, ps, fm):
    key = (bs, kvl, ns, qd, ps, fm)
    if key in _cache:
        return _cache[key]
    tkv = bs * kvl
    kl = (kvi[1:] - kvi[:-1]).to(torch.int32)
    ki = torch.arange(tkv, dtype=torch.int32, device=dev)
    info = get_mla_metadata_info_v1(
        bs,
        1,
        NUM_HEADS,
        qd,
        kvd,
        is_sparse=False,
        fast_mode=fm,
        num_kv_splits=ns,
        intra_batch_mode=True,
    )
    w = [torch.empty(s, dtype=t, device=dev) for s, t in info]
    wm, wi, ws, ri, rf, rp = w
    get_mla_metadata_v1(
        qo,
        kvi,
        kl,
        NUM_HEADS // NUM_KV_HEADS,
        NUM_KV_HEADS,
        True,
        wm,
        ws,
        wi,
        ri,
        rf,
        rp,
        page_size=ps,
        kv_granularity=max(ps, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=fm,
        max_split_per_batch=ns,
        intra_batch_mode=True,
        dtype_q=qd,
        dtype_kv=kvd,
    )
    e = {
        "meta": {
            "work_meta_data": wm,
            "work_indptr": wi,
            "work_info_set": ws,
            "reduce_indptr": ri,
            "reduce_final_map": rf,
            "reduce_partial_map": rp,
        },
        "kl": kl,
        "ki": ki,
        "out": torch.empty(
            (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
        ),
    }
    _cache[key] = e
    return e


def _run_aiter(data: input_t) -> torch.Tensor:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    ns, use_a8w8, ps, fm = _get_config(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])

    if use_a8w8:
        bkey = ("dq", q.numel())
        if bkey not in _cache:
            _cache[bkey] = (
                torch.empty_like(q, dtype=FP8_DTYPE),
                torch.empty(1, dtype=torch.float32, device=q.device),
            )
        qi, qs = _cache[bkey]
        dynamic_per_tensor_quant(qi, q, qs)
        qv = qi.view(-1, NUM_HEADS, QK_HEAD_DIM)
    else:
        qv = q.view(-1, NUM_HEADS, QK_HEAD_DIM)
        qs = None

    c = _get_or_build(
        bs, kvl, qv.dtype, kv_fp8.dtype, qo_indptr, kv_indptr, ns, q.device, ps, fm
    )
    mla_decode_fwd(
        qv,
        kv_4d,
        c["out"],
        qo_indptr,
        kv_indptr,
        c["ki"],
        c["kl"],
        1,
        page_size=ps,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=ns,
        q_scale=qs,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **c["meta"],
    )
    return c["out"]


_hip_buf: dict[tuple, tuple] = {}


def _run_hip(data: input_t) -> torch.Tensor | None:
    q, kv_data, qo_indptr, kv_indptr, config = data
    mod = _get_hip()
    if mod is None:
        return None

    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])
    ns = _hip_kv_splits(bs, kvl)
    kv_fp8, kv_scale = kv_data["fp8"]
    d = q.device
    key = (bs, kvl, ns, d)
    if key not in _hip_buf:
        _hip_buf[key] = (
            torch.empty((bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=d),
            torch.empty((bs, ns, NUM_HEADS, V_HEAD_DIM), dtype=torch.float32, device=d),
            torch.empty((bs, ns, NUM_HEADS), dtype=torch.float32, device=d),
        )
    out, so, sl = _hip_buf[key]

    kv_f = kv_fp8.reshape(-1, QK_HEAD_DIM).contiguous()
    q_c = q.contiguous()
    qi = qo_indptr.contiguous().to(torch.int32)
    ki = kv_indptr.contiguous().to(torch.int32)

    mod.mla_fwd(
        q_c,
        kv_f,
        float(kv_scale.item()),
        ki,
        qi,
        so,
        sl,
        out,
        bs,
        ns,
        SM_SCALE,
    )
    return out


def custom_kernel(data: input_t) -> output_t:
    _, _, qo_indptr, kv_indptr, config = data
    q = data[0]
    bs = int(config["batch_size"])
    qsl = int(config["q_seq_len"])

    uniform_q = (
        qsl == 1 and q.shape[0] == bs and q.is_cuda and kv_indptr.device.type == "cuda"
    )

    if uniform_q:
        hip_out = _run_hip(data)
        if hip_out is not None:
            if "hip" not in _path_marks:
                _path_marks.add("hip")
                _diag(
                    "custom_kernel path=HIP (split-K fp8 QK MFMA + bf16 softmax + bf16 V MFMA)"
                )
            return hip_out
        if "aiter_nohip" not in _path_marks:
            _path_marks.add("aiter_nohip")
            _diag(
                "custom_kernel path=aiter fallback (decode eligible for HIP but module not loaded — see HIP JIT lines above)"
            )
    else:
        if "aiter_shape" not in _path_marks:
            _path_marks.add("aiter_shape")
            _diag(
                f"custom_kernel path=aiter (not HIP-eligible: q_seq_len={qsl}, q.shape[0]={q.shape[0]}, bs={bs}, "
                f"q.is_cuda={getattr(q, 'is_cuda', None)}, kv_indptr.device={getattr(kv_indptr, 'device', None)})"
            )

    return _run_aiter(data)
