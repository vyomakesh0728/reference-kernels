#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Phase 8: Native fp8 MFMA for QK — eliminate dequant ALU bottleneck.

Key change: mfma_f32_16x16x32_fp8_fp8 for QK computation.
- K is already fp8 in memory — load raw bytes directly into MFMA
- Q quantized to fp8 once during LDS load (amortized over all token blocks)
- Eliminates 144 fp8→bf16 conversions per 16-token block (74% of ALU work)

Q quantization: per-head dynamic fp8
  1. Find amax across 576 dims (warp reduce)
  2. Compute scale = amax / 240 (fp8 e4m3 max)
  3. Quantize: fp8_val = round(q_val / scale)
  4. Store q_fp8 + q_scale in LDS

QK result scaling: raw_qk * q_scale * kv_scale * sm_scale
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

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

_cache = {}
_MODULE = None

CPP_WRAPPER = """
void mla_fused(
    torch::Tensor q_fp8, torch::Tensor q_scale, torch::Tensor kv, torch::Tensor kv_scale,
    torch::Tensor out, torch::Tensor kv_indptr,
    torch::Tensor split_buf, torch::Tensor split_lse_buf,
    torch::Tensor split_counter,
    int batch_size, int num_splits, float sm_scale
);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int QK_DIM = 576;
constexpr int V_DIM = 512;
constexpr int N_HEADS = 16;
constexpr int WARP = 64;
constexpr int Q_FP8_PAD = 576;  // no padding needed for fp8
constexpr int N_QK_TILES = 18;
constexpr int N_V_TILES = 32;

using i32x8_t = int __attribute__((ext_vector_type(8)));
using i64x4_t = long __attribute__((ext_vector_type(4)));
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

// Quantize a float to fp8 E4M3 FNUZ
__device__ __forceinline__ uint8_t f32_to_fp8(float val, float inv_scale) {
    float scaled = val * inv_scale;
    // Clamp to fp8 e4m3 range: [-240, 240]
    scaled = fminf(fmaxf(scaled, -240.0f), 240.0f);
    // Use hardware conversion if available, otherwise approximate
    uint32_t bits = __builtin_bit_cast(uint32_t, scaled);
    uint32_t sign = (bits >> 31) & 1;
    int exp = static_cast<int>((bits >> 23) & 0xFF) - 127;
    uint32_t man = bits & 0x7FFFFF;

    // fp8 e4m3: bias=8, exp range [-7, 8], mantissa 3 bits
    int fp8_exp = exp + 8;
    if (fp8_exp <= 0) {
        // Denormal or zero
        if (fp8_exp < -3) return 0;
        uint32_t shift = 1 - fp8_exp;
        uint32_t fp8_man = ((0x800000 | man) >> (20 + shift));
        return static_cast<uint8_t>((sign << 7) | (fp8_man & 0x7));
    }
    if (fp8_exp >= 16) fp8_exp = 15; // clamp
    uint32_t fp8_man = (man + 0x80000) >> 20; // round
    if (fp8_man >= 8) { fp8_man = 0; fp8_exp++; }
    if (fp8_exp >= 16) { fp8_exp = 15; fp8_man = 7; }
    return static_cast<uint8_t>((sign << 7) | (fp8_exp << 3) | (fp8_man & 0x7));
}

// LDS layout:
//   q_fp8:     [16][576] uint8     = 9216 bytes  (Q quantized to fp8)
//   q_scales:  [16] float          = 64 bytes    (per-head Q scales)
//   scores:    [16][16] float      = 1024 bytes
//   Total: ~10.3 KB

constexpr int Q_FP8_SIZE = N_HEADS * Q_FP8_PAD;
constexpr int Q_SCALES_SIZE = N_HEADS * sizeof(float);

__launch_bounds__(64)
__global__ void mla_fused_kernel(
    const uint8_t* __restrict__ q_fp8,      // (bs, 16, 576) already fp8
    const float* __restrict__ q_scale_ptr,   // scalar
    const uint8_t* __restrict__ kv,
    const float* __restrict__ kv_scale_ptr,
    __hip_bfloat16* __restrict__ out,
    const int* __restrict__ kv_indptr,
    float* __restrict__ split_buf,
    float* __restrict__ split_lse,
    int* __restrict__ split_counter,
    int num_splits,
    float sm_scale
) {
    const int bid = blockIdx.x / num_splits;
    const int sid = blockIdx.x % num_splits;
    const int lane = threadIdx.x;
    const int lane16 = lane & 15;
    const int group = lane >> 4;

    const float kv_scale = kv_scale_ptr[0];

    extern __shared__ char smem[];
    auto* q_fp8_lds = reinterpret_cast<uint8_t*>(smem);
    auto* q_scales_lds = reinterpret_cast<float*>(smem + Q_FP8_SIZE);
    auto* scores_lds = reinterpret_cast<float*>(smem + Q_FP8_SIZE + Q_SCALES_SIZE);

    // Load pre-quantized Q fp8 into LDS
    const uint8_t* q_base_fp8 = q_fp8 + bid * N_HEADS * QK_DIM;
    for (int i = lane; i < N_HEADS * QK_DIM; i += WARP) {
        int h = i / QK_DIM;
        int d = i % QK_DIM;
        q_fp8_lds[h * Q_FP8_PAD + d] = q_base_fp8[i];
    }
    // All heads share the same q_scale (per-tensor quantization)
    float q_scale_val = q_scale_ptr[0];
    if (lane < N_HEADS) q_scales_lds[lane] = q_scale_val;
    __syncthreads();

    // KV range
    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;
    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    floatx4 v_acc[N_V_TILES];
    #pragma unroll
    for (int vt = 0; vt < N_V_TILES; vt++)
        v_acc[vt] = {0.0f, 0.0f, 0.0f, 0.0f};

    float m_local[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    float l_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Per-head combined scale: q_scale[head] * kv_scale * sm_scale
    // head = 4*group + gpr, but q_scale varies per head
    float head_scale[4];
    #pragma unroll
    for (int gpr = 0; gpr < 4; gpr++)
        head_scale[gpr] = q_scales_lds[4 * group + gpr] * kv_scale * sm_scale;

    if (s_start < kv_len) {
        for (int t_base = s_start; t_base < s_end; t_base += 16) {
            const int t_count = min(16, s_end - t_base);
            const bool valid = (lane16 < t_count);
            const int abs_tok = kv_start + t_base + lane16;

            // ===== FP8 MFMA QK: Q_fp8 @ K_fp8^T =====
            // mfma_f32_16x16x32_fp8_fp8: A(i32x8), B(i32x8), C(floatx4)
            // A: Q[16 heads, 32 dims] fp8 — from q_fp8_lds
            // B: K[32 dims, 16 tokens] fp8 — from global kv
            floatx4 qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int tk = 0; tk < N_QK_TILES; tk++) {
                const int k_base = tk * 32 + 8 * group;

                // A: Q fp8 from LDS — pack 8 bytes into long (64-bit)
                // lane%16 = head, 8 bytes from q_fp8_lds[head][k_base..k_base+7]
                union { long val; uint8_t b[8]; } a_buf;
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    a_buf.b[i] = q_fp8_lds[lane16 * Q_FP8_PAD + k_base + i];

                // B: K fp8 from global — pack 8 bytes into long
                union { long val; uint8_t b[8]; } b_buf;
                if (valid) {
                    const uint8_t* kp = &kv[abs_tok * QK_DIM + k_base];
                    #pragma unroll
                    for (int i = 0; i < 8; i++) b_buf.b[i] = kp[i];
                } else {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) b_buf.b[i] = 0;
                }

                qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                    a_buf.val, b_buf.val, qk_acc, 0, 0, 0
                );
            }

            // ===== Online softmax =====
            // QK result needs scaling: raw_qk * q_scale[head] * kv_scale * sm_scale
            #pragma unroll
            for (int gpr = 0; gpr < 4; gpr++) {
                float qk = valid ? qk_acc[gpr] * head_scale[gpr] : -1e30f;

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
                for (int vt = 0; vt < N_V_TILES; vt++)
                    v_acc[vt][gpr] *= alpha;

                m_local[gpr] = m_new;
                l_local[gpr] = l_local[gpr] * alpha + sum_p;

                scores_lds[(4 * group + gpr) * 16 + lane16] = p;
            }
            __syncthreads();

            // ===== MFMA V (bf16, same as Phase 4) =====
            #pragma unroll
            for (int vt = 0; vt < N_V_TILES; vt++) {
                const int v_base = vt * 16;

                _B16x4 a_v;
                #pragma unroll
                for (int b = 0; b < 4; b++)
                    a_v[b] = f32_to_bf16_bits(scores_lds[lane16 * 16 + 4 * group + b]);

                _B16x4 b_v;
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    int tok_off = 4 * group + b;
                    int vd = v_base + lane16;
                    if (tok_off < t_count && vd < V_DIM) {
                        int at = kv_start + t_base + tok_off;
                        // Branchless fp8 dequant for V
                        uint8_t raw = kv[at * QK_DIM + vd];
                        float f = (raw == 0) ? 0.0f : __builtin_bit_cast(float,
                            (static_cast<uint32_t>(raw & 0x80) << 24)
                            | ((static_cast<uint32_t>((raw >> 3) & 0xF) + 119u) << 23)
                            | (static_cast<uint32_t>(raw & 0x7) << 20));
                        b_v[b] = f32_to_bf16_bits(f * kv_scale);
                    } else {
                        b_v[b] = 0;
                    }
                }

                v_acc[vt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    a_v, b_v, v_acc[vt], 0, 0, 0
                );
            }
            __syncthreads();
        }
    }

    // Store partial results
    #pragma unroll
    for (int gpr = 0; gpr < 4; gpr++) {
        int head = 4 * group + gpr;
        int so_base = ((bid * N_HEADS + head) * num_splits + sid) * V_DIM;
        int sl_idx = (bid * N_HEADS + head) * num_splits + sid;

        float inv_l = (l_local[gpr] > 0.0f) ? 1.0f / l_local[gpr] : 0.0f;
        #pragma unroll
        for (int vt = 0; vt < N_V_TILES; vt++)
            split_buf[so_base + vt * 16 + lane16] = v_acc[vt][gpr] * inv_l;

        if (lane16 == 0)
            split_lse[sl_idx] = (l_local[gpr] > 0.0f)
                ? m_local[gpr] + logf(l_local[gpr]) : -1e30f;
    }

    // Fused reduce
    __threadfence();
    __shared__ int is_last;
    if (lane == 0) {
        int old = atomicAdd(&split_counter[bid], 1);
        is_last = (old == num_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (is_last) {
        #pragma unroll
        for (int gpr = 0; gpr < 4; gpr++) {
            int head = 4 * group + gpr;
            int lse_base = (bid * N_HEADS + head) * num_splits;
            int out_base = (bid * N_HEADS + head) * V_DIM;

            float max_lse = -1e30f;
            for (int s = 0; s < num_splits; s++)
                max_lse = fmaxf(max_lse, split_lse[lse_base + s]);

            float sum_w = 0.0f;
            float acc[N_V_TILES];
            #pragma unroll
            for (int vt = 0; vt < N_V_TILES; vt++) acc[vt] = 0.0f;

            for (int s = 0; s < num_splits; s++) {
                float w = expf(split_lse[lse_base + s] - max_lse);
                sum_w += w;
                const float* p = split_buf + ((bid * N_HEADS + head) * num_splits + s) * V_DIM;
                #pragma unroll
                for (int vt = 0; vt < N_V_TILES; vt++)
                    acc[vt] += w * p[vt * 16 + lane16];
            }

            float inv_w = (sum_w > 0.0f) ? 1.0f / sum_w : 0.0f;
            #pragma unroll
            for (int vt = 0; vt < N_V_TILES; vt++)
                out[out_base + vt * 16 + lane16] = static_cast<__hip_bfloat16>(acc[vt] * inv_w);
        }
    }
}

void mla_fused(
    torch::Tensor q_fp8, torch::Tensor q_scale, torch::Tensor kv, torch::Tensor kv_scale,
    torch::Tensor out, torch::Tensor kv_indptr,
    torch::Tensor split_buf, torch::Tensor split_lse_buf,
    torch::Tensor split_counter,
    int batch_size, int num_splits, float sm_scale
) {
    hipMemsetAsync(split_counter.data_ptr<int>(), 0, batch_size * sizeof(int), 0);

    int smem_bytes = Q_FP8_SIZE + Q_SCALES_SIZE + N_HEADS * 16 * sizeof(float);
    dim3 block(64);
    dim3 grid(batch_size * num_splits);
    hipLaunchKernelGGL(
        mla_fused_kernel,
        grid, block, smem_bytes, 0,
        reinterpret_cast<const uint8_t*>(q_fp8.data_ptr()),
        q_scale.data_ptr<float>(),
        reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        kv_scale.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(),
        split_buf.data_ptr<float>(),
        split_lse_buf.data_ptr<float>(),
        split_counter.data_ptr<int>(),
        num_splits, sm_scale
    );
}
"""

EXPORT_FUNCTIONS = ["mla_fused"]


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mla_p8_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_p8_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=EXPORT_FUNCTIONS,
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _get_num_splits(bs, kvl):
    if kvl <= 1024:
        if bs <= 4:
            return 16
        if bs <= 64:
            return 4
        return 4
    else:
        if bs <= 4:
            return 32
        if bs <= 64:
            return 16
        return 8


def _get_bufs(bs, num_splits, dev):
    key = ("p8_bufs", bs, num_splits)
    if key not in _cache:
        _cache[key] = {
            "split_buf": torch.empty(
                bs * NUM_HEADS * num_splits * V_HEAD_DIM,
                dtype=torch.float32,
                device=dev,
            ),
            "split_lse": torch.empty(
                bs * NUM_HEADS * num_splits, dtype=torch.float32, device=dev
            ),
            "counter": torch.zeros(bs, dtype=torch.int32, device=dev),
            "out": torch.empty(
                (bs, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
            ),
        }
    return _cache[key]


from aiter import dtypes as aiter_dtypes
from aiter.ops.quant import dynamic_per_tensor_quant

FP8_DTYPE = aiter_dtypes.fp8


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]

    # Quantize Q to fp8 using aiter (known correct)
    q_key = ("q_fp8", q.numel())
    if q_key not in _cache:
        _cache[q_key] = (
            torch.empty_like(q, dtype=FP8_DTYPE),
            torch.empty(1, dtype=torch.float32, device=q.device),
        )
    q_fp8, q_scale = _cache[q_key]
    dynamic_per_tensor_quant(q_fp8, q, q_scale)

    num_splits = _get_num_splits(bs, kvl)
    bufs = _get_bufs(bs, num_splits, q.device)
    bufs["counter"].zero_()

    mod = _module()
    mod.mla_fused(
        q_fp8.view(-1),
        q_scale,
        kv_fp8.view(-1),
        kv_scale,
        bufs["out"],
        kv_indptr,
        bufs["split_buf"],
        bufs["split_lse"],
        bufs["counter"],
        bs,
        num_splits,
        SM_SCALE,
    )
    return bufs["out"]
