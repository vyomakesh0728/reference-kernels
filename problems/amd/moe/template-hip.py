#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
HIP correctness-first MoE submission.
Dequants raw MXFP4 weights in Python, routes in Python, uses HIP tiled GEMM.
"""
import hashlib
import os
from pathlib import Path
import tempfile

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.utility import fp4_utils

MXFP4_BLOCK_SIZE = 32

CPP_WRAPPER = """
void hip_gemm_nt(torch::Tensor a, torch::Tensor b, torch::Tensor c);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int TILE_K = 64;

// C[M,N] = A[M,K] @ B[N,K].T
__global__ void gemm_nt_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k
) {
    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float a_tile[TILE_M][TILE_K];
    __shared__ float b_tile[TILE_N][TILE_K];

    double acc = 0.0;

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        for (int lk = threadIdx.x; lk < TILE_K; lk += blockDim.x) {
            int gk = tile_k + lk;
            a_tile[threadIdx.y][lk] = (row < m && gk < k) ? a[row * k + gk] : 0.0f;
        }
        for (int lk = threadIdx.y; lk < TILE_K; lk += blockDim.y) {
            int gk = tile_k + lk;
            b_tile[threadIdx.x][lk] = (col < n && gk < k) ? b[col * k + gk] : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll 1
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += (double)a_tile[threadIdx.y][kk] * (double)b_tile[threadIdx.x][kk];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = (float)acc;
    }
}

void hip_gemm_nt(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    dim3 block(TILE_N, TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(gemm_nt_f32_kernel, grid, block, 0, 0,
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "moe_hip_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"moe_hip_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["hip_gemm_nt"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _dequant_mxfp4_2d(weight_fp4, scale_e8m0, rows, cols):
    """Dequant single expert MXFP4 weight -> float32 [rows, cols]."""
    w_f32 = fp4_utils.mxfp4_to_f32(weight_fp4.contiguous())[:rows, :cols]
    s_f32 = fp4_utils.e8m0_to_f32(scale_e8m0.contiguous())[:rows]
    s_f32 = s_f32.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)[:, :cols]
    return (w_f32 * s_f32).to(torch.float32).contiguous()


def _gemm_nt(a, b):
    """C = A @ B.T via HIP. A:[M,K] B:[N,K] -> C:[M,N] float32."""
    a = a.to(torch.float32).contiguous()
    b = b.to(torch.float32).contiguous()
    c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    _module().hip_gemm_nt(a, b, c)
    return c


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        _, _, _, _,
        topk_weights, topk_ids, config,
    ) = data

    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    M = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    E = gate_up_weight.shape[0]

    _module()

    output = torch.zeros((M, d_hidden), dtype=torch.float32, device=hidden_states.device)

    flat_tokens = torch.arange(M, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    flat_experts = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    for e in range(E):
        mask = (flat_experts == e)
        if not mask.any():
            continue

        tids = flat_tokens[mask]
        wts = flat_weights[mask]

        gu_w = _dequant_mxfp4_2d(
            gate_up_weight[e], gate_up_weight_scale[e],
            2 * d_expert, d_hidden,
        )
        dn_w = _dequant_mxfp4_2d(
            down_weight[e], down_weight_scale[e],
            d_hidden, d_expert,
        )

        h = hidden_states[tids].to(torch.float32)

        # Stage 1: gate_up GEMM + SwiGLU
        gu_out = _gemm_nt(h, gu_w)
        gate = F.silu(gu_out[:, :d_expert])
        up = gu_out[:, d_expert:]
        inter = gate * up

        # Stage 2: down GEMM
        out_e = _gemm_nt(inter, dn_w)

        # Weighted accumulate
        output.index_add_(0, tids, wts.unsqueeze(1) * out_e)

    return output.to(torch.bfloat16)
