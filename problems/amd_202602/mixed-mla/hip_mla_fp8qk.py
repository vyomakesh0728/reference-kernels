#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Phase 10 fp8-QK: Native fp8×fp8 MFMA for QK computation.

Changes from P10 (bf16 QK MFMA):
1. Q pre-quantized to fp8 in Python via aiter dynamic_per_tensor_quant
2. QK uses __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8 (long operands)
3. Eliminates 16 ALU ops/tile: 8 hw_fp8 conversions + 8 f32_to_bf16_bits
4. Q stored as uint8 in LDS: 9216 bytes (was 18432 for bf16) — halved
5. head_scale = q_scale * kv_scale * sm_scale applied to raw MFMA result
6. V path unchanged (bf16 MFMA with software fp8 dequant)
"""

import os

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import hashlib
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.quant import dynamic_per_tensor_quant
from aiter import dtypes as aiter_dtypes

FP8_DTYPE = aiter_dtypes.fp8

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

_cache = {}
_MODULE = None

CPP_WRAPPER = """
void mla_ns16(torch::Tensor q_fp8, torch::Tensor q_scale, torch::Tensor kv, torch::Tensor kv_scale, torch::Tensor out, torch::Tensor kv_indptr, torch::Tensor sb, torch::Tensor sl, torch::Tensor sc, int bs, float sm);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int QK_DIM = 576;
constexpr int V_DIM = 512;
constexpr int N_HEADS = 16;
constexpr int WARP = 64;
constexpr int Q_FP8_PAD = 576;
constexpr int KV_STRIDE = 576;
constexpr int KV_BLOCK_BYTES = 16 * KV_STRIDE;
constexpr int VEC_BYTES = 16;
constexpr int Q_LDS_BYTES = N_HEADS * Q_FP8_PAD;   // 9216

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef bit16x4 _B16x4;

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    return static_cast<uint16_t>(c.u >> 16);
}

__device__ __forceinline__ float fp8_fast(uint8_t x) {
    if (x == 0) return 0.0f;
    uint32_t bits = (static_cast<uint32_t>(x & 0x80) << 24)
                  | ((static_cast<uint32_t>((x >> 3) & 0xF) + 119u) << 23)
                  | (static_cast<uint32_t>(x & 0x7) << 20);
    return __builtin_bit_cast(float, bits);
}

template<int NUM_SPLITS>
__launch_bounds__(64)
__global__ void mla_kernel(
    const uint8_t* __restrict__ q_fp8,
    const float* __restrict__ q_scale_ptr,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ kv_scale_ptr,
    __hip_bfloat16* __restrict__ out,
    const int* __restrict__ kv_indptr,
    float* __restrict__ split_buf,
    float* __restrict__ split_lse,
    int* __restrict__ split_counter,
    float sm_scale
) {
    const int bid = blockIdx.x / NUM_SPLITS;
    const int sid = blockIdx.x % NUM_SPLITS;
    const int lane = threadIdx.x;
    const int lane16 = lane & 15;
    const int group = lane >> 4;

    const float q_scale_val = q_scale_ptr[0];
    const float kv_scale = kv_scale_ptr[0];
    const float head_scale = q_scale_val * kv_scale * sm_scale;

    extern __shared__ char smem[];
    auto* q_fp8_lds = reinterpret_cast<uint8_t*>(smem);
    auto* kv_lds = reinterpret_cast<uint8_t*>(smem + Q_LDS_BYTES);
    auto* scores_lds = reinterpret_cast<float*>(smem + Q_LDS_BYTES + KV_BLOCK_BYTES);

    // Load Q fp8: 9216 bytes via 576 × uint4 (16B) loads, 9 iterations/lane
    {
        const uint4* q_src = reinterpret_cast<const uint4*>(q_fp8 + bid * N_HEADS * QK_DIM);
        uint4* q_dst = reinterpret_cast<uint4*>(q_fp8_lds);
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            int idx = i * WARP + lane;
            q_dst[idx] = q_src[idx];
        }
    }
    __syncthreads();

    const int kv_start = kv_indptr[bid];
    const int kv_len = kv_indptr[bid + 1] - kv_start;
    const int split_size = (kv_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    floatx4 v_acc[32];
    #pragma unroll
    for (int vt = 0; vt < 32; vt++)
        v_acc[vt] = {0.0f, 0.0f, 0.0f, 0.0f};

    float m_local[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    float l_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t_base = s_start; t_base < s_end; t_base += 16) {
        const int t_count = min(16, s_end - t_base);
        const bool is_full_block = (t_count == 16);

        // Vectorized KV load: global → LDS
        const uint8_t* kv_src = &kv[(kv_start + t_base) * QK_DIM];
        if (is_full_block) {
            #pragma unroll
            for (int i = 0; i < 9; i++) {
                int offset = (i * WARP + lane) * VEC_BYTES;
                if (offset + VEC_BYTES <= KV_BLOCK_BYTES) {
                    *reinterpret_cast<uint4*>(kv_lds + offset) =
                        *reinterpret_cast<const uint4*>(kv_src + offset);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 9; i++) {
                int offset = (i * WARP + lane) * VEC_BYTES;
                if (offset + VEC_BYTES <= KV_BLOCK_BYTES) {
                    int tok = offset / KV_STRIDE;
                    if (tok < t_count) {
                        *reinterpret_cast<uint4*>(kv_lds + offset) =
                            *reinterpret_cast<const uint4*>(kv_src + offset);
                    } else {
                        *reinterpret_cast<uint4*>(kv_lds + offset) = {0, 0, 0, 0};
                    }
                }
            }
        }
        __syncthreads();

        const bool valid = (lane16 < t_count);

        // FP8×FP8 MFMA QK: each lane loads 8 fp8 bytes into a long
        // A[i][k]: lane%16=i(head), k = 8*(lane/16) + byte_within_8
        // B[k][j]: lane%16=j(token), k = 8*(lane/16) + byte_within_8
        floatx4 qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int tk = 0; tk < 18; tk++) {
            const int k_base = tk * 32 + 8 * group;

            long a_val = *reinterpret_cast<const long*>(
                &q_fp8_lds[lane16 * Q_FP8_PAD + k_base]);

            long b_val = valid
                ? *reinterpret_cast<const long*>(
                    &kv_lds[lane16 * KV_STRIDE + k_base])
                : 0L;

            qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                a_val, b_val, qk_acc, 0, 0, 0);
        }

        // Softmax + V rescale (no sync needed — single wavefront)
        #pragma unroll
        for (int gpr = 0; gpr < 4; gpr++) {
            float qk = valid ? qk_acc[gpr] * head_scale : -1e30f;
            float m_block = qk;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                m_block = fmaxf(m_block, __shfl_xor(m_block, off));

            float m_old = m_local[gpr];
            float m_new = fmaxf(m_old, m_block);
            float alpha = expf(m_old - m_new);
            float p = valid ? expf(qk - m_new) : 0.0f;
            float sum_p = p;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                sum_p += __shfl_xor(sum_p, off);

            #pragma unroll
            for (int vt = 0; vt < 32; vt++)
                v_acc[vt][gpr] *= alpha;

            m_local[gpr] = m_new;
            l_local[gpr] = l_local[gpr] * alpha + sum_p;
            scores_lds[(4 * group + gpr) * 16 + lane16] = p;
        }

        // MFMA V (unchanged — bf16 MFMA with software fp8 dequant)
        #pragma unroll
        for (int vt = 0; vt < 32; vt++) {
            const int v_base = vt * 16;
            _B16x4 a_v;
            #pragma unroll
            for (int b = 0; b < 4; b++)
                a_v[b] = f32_to_bf16_bits(scores_lds[lane16 * 16 + 4 * group + b]);

            _B16x4 b_v;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                int tok_off = 4 * group + b;
                b_v[b] = (tok_off < t_count)
                    ? f32_to_bf16_bits(fp8_fast(kv_lds[tok_off * KV_STRIDE + v_base + lane16]) * kv_scale)
                    : static_cast<uint16_t>(0);
            }
            v_acc[vt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_v, b_v, v_acc[vt], 0, 0, 0);
        }
        // Sync before next KV load overwrites kv_lds
        __syncthreads();
    }

    // Store partial results
    #pragma unroll
    for (int gpr = 0; gpr < 4; gpr++) {
        int head = 4 * group + gpr;
        int so_base = ((bid * N_HEADS + head) * NUM_SPLITS + sid) * V_DIM;
        float inv_l = (l_local[gpr] > 0.0f) ? 1.0f / l_local[gpr] : 0.0f;
        #pragma unroll
        for (int vt = 0; vt < 32; vt++)
            split_buf[so_base + vt * 16 + lane16] = v_acc[vt][gpr] * inv_l;
        if (lane16 == 0)
            split_lse[(bid * N_HEADS + head) * NUM_SPLITS + sid] =
                (l_local[gpr] > 0.0f) ? m_local[gpr] + logf(l_local[gpr]) : -1e30f;
    }

    // Fused reduce
    __threadfence();
    __shared__ int is_last;
    if (lane == 0) {
        int old = atomicAdd(&split_counter[bid], 1);
        is_last = (old == NUM_SPLITS - 1) ? 1 : 0;
    }
    __syncthreads();

    if (is_last) {
        #pragma unroll
        for (int gpr = 0; gpr < 4; gpr++) {
            int head = 4 * group + gpr;
            int lse_base = (bid * N_HEADS + head) * NUM_SPLITS;
            int out_base = (bid * N_HEADS + head) * V_DIM;

            float max_lse = -1e30f;
            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; s++)
                max_lse = fmaxf(max_lse, split_lse[lse_base + s]);

            float sum_w = 0.0f;
            float acc[32];
            #pragma unroll
            for (int vt = 0; vt < 32; vt++) acc[vt] = 0.0f;

            #pragma unroll
            for (int s = 0; s < NUM_SPLITS; s++) {
                float w = expf(split_lse[lse_base + s] - max_lse);
                sum_w += w;
                const float* p = split_buf + ((bid * N_HEADS + head) * NUM_SPLITS + s) * V_DIM;
                #pragma unroll
                for (int vt = 0; vt < 32; vt++)
                    acc[vt] += w * p[vt * 16 + lane16];
            }

            float inv_w = (sum_w > 0.0f) ? 1.0f / sum_w : 0.0f;
            #pragma unroll
            for (int vt = 0; vt < 32; vt++)
                out[out_base + vt * 16 + lane16] = static_cast<__hip_bfloat16>(acc[vt] * inv_w);
        }
    }
}

#define LAUNCH_MLA(NS) \
void mla_ns##NS(torch::Tensor q_fp8, torch::Tensor q_scale, torch::Tensor kv, torch::Tensor kv_scale, \
    torch::Tensor out, torch::Tensor kv_indptr, \
    torch::Tensor sb, torch::Tensor sl, torch::Tensor sc, int bs, float sm) { \
    hipMemsetAsync(sc.data_ptr<int>(), 0, bs * sizeof(int), 0); \
    int smem = Q_LDS_BYTES + KV_BLOCK_BYTES + N_HEADS * 16 * sizeof(float); \
    hipLaunchKernelGGL(mla_kernel<NS>, dim3(bs * NS), dim3(64), smem, 0, \
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()), \
        q_scale.data_ptr<float>(), \
        reinterpret_cast<const uint8_t*>(kv.data_ptr()), \
        kv_scale.data_ptr<float>(), \
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()), \
        kv_indptr.data_ptr<int>(), sb.data_ptr<float>(), sl.data_ptr<float>(), \
        sc.data_ptr<int>(), sm); \
}

LAUNCH_MLA(16)
"""

EXPORT_FUNCTIONS = ["mla_ns16"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_fp8qk_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_fp8qk_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=EXPORT_FUNCTIONS,
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O0"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _get_num_splits(bs, kvl):
    return 16


def _get_bufs(bs, num_splits, dev):
    key = ("fp8qk", bs, num_splits)
    if key not in _cache:
        _cache[key] = {
            "sb": torch.empty(
                bs * NUM_HEADS * num_splits * V_HEAD_DIM,
                dtype=torch.float32,
                device=dev,
            ),
            "sl": torch.empty(
                bs * NUM_HEADS * num_splits, dtype=torch.float32, device=dev
            ),
            "sc": torch.zeros(bs, dtype=torch.int32, device=dev),
            "out": torch.empty(
                (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
            ),
        }
    return _cache[key]


_DISPATCH = {16: "mla_ns16"}


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]

    # Quantize Q bf16 → fp8 with per-tensor scale
    q_fp8 = torch.empty(q.numel(), dtype=FP8_DTYPE, device=q.device)
    q_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    dynamic_per_tensor_quant(q_fp8, q.reshape(-1), q_scale)

    ns = _get_num_splits(bs, kvl)
    bufs = _get_bufs(bs, ns, q.device)
    bufs["sc"].zero_()

    mod = _module()
    fn = getattr(mod, _DISPATCH[ns])
    fn(
        q_fp8,
        q_scale,
        kv_fp8.view(-1),
        kv_scale,
        bufs["out"],
        kv_indptr,
        bufs["sb"],
        bufs["sl"],
        bufs["sc"],
        bs,
        SM_SCALE,
    )
    return bufs["out"]
