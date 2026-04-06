#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Phase 6: Multi-wavefront for latency hiding — 256 threads, 4 WFs per block.

The actual bottleneck was never compute — it's global memory latency with
only 1 wavefront (zero latency hiding). With 4 wavefronts per block,
the GPU overlaps memory stalls from one WF with compute from another.

Architecture:
- 256 threads = 4 wavefronts per block
- Each wavefront handles 4 heads independently (4 × 4 = 16)
- Scalar QK + V (simpler, lower register pressure → higher occupancy)
- 4-way latency hiding → ~4× improvement over Phase 4

Register budget per wavefront:
  Q: 36 floats (4 heads × 9 dims/thread)
  V_acc: 32 floats (4 heads × 8 dims/thread)
  Softmax: 8 floats (4 heads × m,l)
  Total: ~76 VGPRs → 6+ WFs per SIMD possible!
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
constexpr int N_WF = 4;
constexpr int HEADS_PER_WF = N_HEADS / N_WF;  // 4

// Branchless fp8 E4M3 FNUZ → f32
__device__ __forceinline__ float fp8_fast(uint8_t x) {
    if (x == 0) return 0.0f;
    uint32_t bits = (static_cast<uint32_t>(x & 0x80) << 24)
                  | ((static_cast<uint32_t>((x >> 3) & 0xF) + 119u) << 23)
                  | (static_cast<uint32_t>(x & 0x7) << 20);
    return __builtin_bit_cast(float, bits);
}

// 256 threads = 4 wavefronts. Each WF handles 4 heads.
// No MFMA — pure scalar with vectorized memory and latency hiding.
__launch_bounds__(256)
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
    const int wf_id = threadIdx.x / WARP;   // 0-3
    const int lane = threadIdx.x % WARP;     // 0-63

    const float kv_scale = kv_scale_ptr[0];
    const float combined_scale = kv_scale * sm_scale;

    const int kv_start = kv_indptr[bid];
    const int kv_end = kv_indptr[bid + 1];
    const int kv_len = kv_end - kv_start;
    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int s_start = sid * split_size;
    const int s_end = min(s_start + split_size, kv_len);

    // Each WF handles HEADS_PER_WF=4 heads
    const int head_start = wf_id * HEADS_PER_WF;

    // Load Q into registers: 4 heads × 9 dims per thread
    float q_reg[HEADS_PER_WF][9];
    #pragma unroll
    for (int h = 0; h < HEADS_PER_WF; h++) {
        const __hip_bfloat16* qp = q + (bid * N_HEADS + head_start + h) * QK_DIM;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            int d = lane + i * WARP;
            q_reg[h][i] = (d < QK_DIM) ? static_cast<float>(qp[d]) : 0.0f;
        }
    }

    // V accumulator + softmax state
    float v_acc[HEADS_PER_WF][8];
    float m_state[HEADS_PER_WF], l_state[HEADS_PER_WF];
    #pragma unroll
    for (int h = 0; h < HEADS_PER_WF; h++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) v_acc[h][i] = 0.0f;
        m_state[h] = -1e30f;
        l_state[h] = 0.0f;
    }

    if (s_start < kv_len) {
        for (int t = s_start; t < s_end; t++) {
            const uint8_t* kv_ptr = kv + (kv_start + t) * QK_DIM;

            // Load KV dims for this token (shared across 4 heads)
            // Each thread loads 9 fp8 values at stride 64
            float kv_reg[9];
            #pragma unroll
            for (int i = 0; i < 9; i++) {
                int d = lane + i * WARP;
                kv_reg[i] = (d < QK_DIM) ? fp8_fast(kv_ptr[d]) * kv_scale : 0.0f;
            }

            // QK dot products for 4 heads (reuse kv_reg)
            #pragma unroll
            for (int h = 0; h < HEADS_PER_WF; h++) {
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < 9; i++) {
                    partial += q_reg[h][i] * kv_reg[i];
                }

                // Warp reduce
                #pragma unroll
                for (int off = 32; off > 0; off >>= 1)
                    partial += __shfl_down(partial, off, WARP);
                float qk = __shfl(partial, 0, WARP) * sm_scale;

                // Online softmax
                float m_new = fmaxf(m_state[h], qk);
                float exp_diff = expf(m_state[h] - m_new);
                float exp_qk = expf(qk - m_new);
                float l_new = l_state[h] * exp_diff + exp_qk;

                // V accumulation
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = lane + i * WARP;
                    v_acc[h][i] = v_acc[h][i] * exp_diff + exp_qk * ((d < V_DIM) ? kv_reg[i] : 0.0f);
                }

                m_state[h] = m_new;
                l_state[h] = l_new;
            }
        }
    }

    // Store partial results (each WF stores its 4 heads independently)
    #pragma unroll
    for (int h = 0; h < HEADS_PER_WF; h++) {
        int head = head_start + h;
        int so_base = ((bid * N_HEADS + head) * num_splits + sid) * V_DIM;
        int sl_idx = (bid * N_HEADS + head) * num_splits + sid;

        float inv_l = (l_state[h] > 0.0f) ? 1.0f / l_state[h] : 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane + i * WARP;
            if (d < V_DIM) split_buf[so_base + d] = v_acc[h][i] * inv_l;
        }
        if (lane == 0)
            split_lse[sl_idx] = (l_state[h] > 0.0f)
                ? m_state[h] + logf(l_state[h]) : -1e30f;
    }

    // Fused reduce — all 256 threads sync, then WF 0 reduces
    __threadfence();
    __shared__ int is_last;
    if (threadIdx.x == 0) {
        int old = atomicAdd(&split_counter[bid], 1);
        is_last = (old == num_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (is_last) {
        // Each WF reduces its own 4 heads
        #pragma unroll
        for (int h = 0; h < HEADS_PER_WF; h++) {
            int head = head_start + h;
            int lse_base = (bid * N_HEADS + head) * num_splits;
            int out_base = (bid * N_HEADS + head) * V_DIM;

            float max_lse = -1e30f;
            for (int s = 0; s < num_splits; s++)
                max_lse = fmaxf(max_lse, split_lse[lse_base + s]);

            float sum_w = 0.0f;
            float acc[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) acc[i] = 0.0f;

            for (int s = 0; s < num_splits; s++) {
                float w = expf(split_lse[lse_base + s] - max_lse);
                sum_w += w;
                const float* p = split_buf + ((bid * N_HEADS + head) * num_splits + s) * V_DIM;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int d = lane + i * WARP;
                    if (d < V_DIM) acc[i] += w * p[d];
                }
            }

            float inv_w = (sum_w > 0.0f) ? 1.0f / sum_w : 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int d = lane + i * WARP;
                if (d < V_DIM)
                    out[out_base + d] = static_cast<__hip_bfloat16>(acc[i] * inv_w);
            }
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

    dim3 block(256);
    dim3 grid(batch_size * num_splits);
    hipLaunchKernelGGL(
        mla_fused_kernel,
        grid, block, 0, 0,
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
        build_root = Path(tempfile.gettempdir()) / "mla_p6_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_p6_{digest}",
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
    key = ("p6_bufs", bs, num_splits)
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
