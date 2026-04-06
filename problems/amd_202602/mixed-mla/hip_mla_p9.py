#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Phase 9: Vectorized LDS loading — fix the memory access pattern.

The actual bottleneck: byte-by-byte global loads waste 64-byte cache lines.
Each byte load fetches 64B but uses 1B = 1.5% utilization.

Fix: vectorized uint4 (128-bit) global→LDS loads, then MFMA reads from LDS.
64 threads × 16 bytes = 1024 bytes/iteration, 9 iterations = 9216 bytes.
100% cache line utilization vs ~12% in Phase 4.

This is Phase 5 done right — vectorized loads instead of byte-by-byte.
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
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scale,
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
constexpr int Q_PAD = 580;
constexpr int KV_STRIDE = 576;
constexpr int KV_BLOCK_BYTES = 16 * KV_STRIDE;  // 9216 bytes
constexpr int N_QK_TILES = 18;
constexpr int N_V_TILES = 32;

// Vectorized load width: 16 bytes per thread per iteration
constexpr int VEC_BYTES = 16;
constexpr int LOAD_ITERS = (KV_BLOCK_BYTES + WARP * VEC_BYTES - 1) / (WARP * VEC_BYTES);  // 9

using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
typedef bit16x4 _B16x4;
typedef struct _B16x8 { _B16x4 xy[2]; } _B16x8;

template<int SEL>
__device__ __forceinline__ float hw_fp8(uint32_t pk) {
    return __builtin_amdgcn_cvt_f32_fp8(pk, SEL);
}

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

// LDS: q_lds[16][Q_PAD] bf16 (18560B) + kv_lds[16][576] u8 (9216B) + scores[16][16] f32 (1024B)
// Total: ~28.8 KB

__launch_bounds__(64)
__global__ void mla_fused_kernel(
    const __hip_bfloat16* __restrict__ q,
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
    const float qk_scale = kv_scale * sm_scale;

    extern __shared__ char smem[];
    auto* q_lds = reinterpret_cast<uint16_t*>(smem);
    auto* kv_lds = reinterpret_cast<uint8_t*>(smem + N_HEADS * Q_PAD * 2);
    auto* scores_lds = reinterpret_cast<float*>(smem + N_HEADS * Q_PAD * 2 + KV_BLOCK_BYTES);

    // Load Q into LDS (one-time, scalar with padding)
    const __hip_bfloat16* q_base = q + bid * N_HEADS * QK_DIM;
    for (int i = lane; i < N_HEADS * QK_DIM; i += WARP) {
        int h = i / QK_DIM;
        int d = i % QK_DIM;
        q_lds[h * Q_PAD + d] = reinterpret_cast<const uint16_t*>(q_base)[i];
    }
    __syncthreads();

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

    if (s_start < kv_len) {
        for (int t_base = s_start; t_base < s_end; t_base += 16) {
            const int t_count = min(16, s_end - t_base);

            // ===== VECTORIZED KV LOAD: global → LDS =====
            // 16 tokens × 576 bytes = 9216 bytes
            // 64 threads × 16 bytes = 1024 bytes/iter → 9 iterations
            const uint8_t* kv_src = &kv[(kv_start + t_base) * QK_DIM];
            #pragma unroll
            for (int i = 0; i < LOAD_ITERS; i++) {
                int offset = (i * WARP + lane) * VEC_BYTES;
                if (offset + VEC_BYTES <= KV_BLOCK_BYTES) {
                    // Check if this data is within valid tokens
                    int tok = offset / KV_STRIDE;
                    if (tok < t_count) {
                        *reinterpret_cast<uint4*>(kv_lds + offset) =
                            *reinterpret_cast<const uint4*>(kv_src + offset);
                    } else {
                        *reinterpret_cast<uint4*>(kv_lds + offset) = {0, 0, 0, 0};
                    }
                }
            }
            __syncthreads();

            const bool valid = (lane16 < t_count);

            // ===== MFMA QK from LDS =====
            floatx4 qk_acc = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int tk = 0; tk < N_QK_TILES; tk++) {
                const int k_base = tk * 32 + 8 * group;

                // A: Q from LDS (bf16)
                _B16x8 a_reg;
                const uint16_t* qp = &q_lds[lane16 * Q_PAD + k_base];
                a_reg.xy[0] = {qp[0], qp[1], qp[2], qp[3]};
                a_reg.xy[1] = {qp[4], qp[5], qp[6], qp[7]};

                // B: K from KV_LDS (fp8→bf16 via hw conversion)
                _B16x8 b_reg;
                if (valid) {
                    // Load 8 bytes as 2 uint32 from LDS (fast!)
                    const uint8_t* kp = &kv_lds[lane16 * KV_STRIDE + k_base];
                    uint32_t pk0 = *reinterpret_cast<const uint32_t*>(kp);
                    uint32_t pk1 = *reinterpret_cast<const uint32_t*>(kp + 4);
                    b_reg.xy[0] = {
                        f32_to_bf16_bits(hw_fp8<0>(pk0)),
                        f32_to_bf16_bits(hw_fp8<1>(pk0)),
                        f32_to_bf16_bits(hw_fp8<2>(pk0)),
                        f32_to_bf16_bits(hw_fp8<3>(pk0))
                    };
                    b_reg.xy[1] = {
                        f32_to_bf16_bits(hw_fp8<0>(pk1)),
                        f32_to_bf16_bits(hw_fp8<1>(pk1)),
                        f32_to_bf16_bits(hw_fp8<2>(pk1)),
                        f32_to_bf16_bits(hw_fp8<3>(pk1))
                    };
                } else {
                    b_reg.xy[0] = {0,0,0,0};
                    b_reg.xy[1] = {0,0,0,0};
                }

                bit16x8 a = __builtin_shufflevector(a_reg.xy[0], a_reg.xy[1], 0,1,2,3,4,5,6,7);
                bit16x8 b = __builtin_shufflevector(b_reg.xy[0], b_reg.xy[1], 0,1,2,3,4,5,6,7);
                qk_acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, qk_acc, 0, 0, 0);
            }

            // ===== Online softmax =====
            #pragma unroll
            for (int gpr = 0; gpr < 4; gpr++) {
                float qk = valid ? qk_acc[gpr] * qk_scale : -1e30f;

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

            // ===== MFMA V from LDS =====
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
                        b_v[b] = f32_to_bf16_bits(fp8_fast(kv_lds[tok_off * KV_STRIDE + vd]) * kv_scale);
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
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scale,
    torch::Tensor out, torch::Tensor kv_indptr,
    torch::Tensor split_buf, torch::Tensor split_lse_buf,
    torch::Tensor split_counter,
    int batch_size, int num_splits, float sm_scale
) {
    hipMemsetAsync(split_counter.data_ptr<int>(), 0, batch_size * sizeof(int), 0);

    int smem_bytes = N_HEADS * Q_PAD * 2 + KV_BLOCK_BYTES + N_HEADS * 16 * sizeof(float);
    dim3 block(64);
    dim3 grid(batch_size * num_splits);
    hipLaunchKernelGGL(
        mla_fused_kernel,
        grid, block, smem_bytes, 0,
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr<at::BFloat16>()),
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
        build_root = Path(tempfile.gettempdir()) / "mla_p9_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_p9_{digest}",
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
    key = ("p9_bufs", bs, num_splits)
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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = int(config["batch_size"])
    kvl = int(config["kv_seq_len"])

    kv_fp8, kv_scale = kv_data["fp8"]
    q_reshaped = q.view(bs, NUM_HEADS, QK_HEAD_DIM)

    num_splits = _get_num_splits(bs, kvl)
    bufs = _get_bufs(bs, num_splits, q.device)
    bufs["counter"].zero_()

    mod = _module()
    mod.mla_fused(
        q_reshaped,
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
