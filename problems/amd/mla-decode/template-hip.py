#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
HIP correctness-first MLA decode submission.
FP8 quant/dequant in Python (to match reference quantization noise),
attention computed via HIP tiled GEMM.
"""
import hashlib
import os
from pathlib import Path
import tempfile

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

try:
    from aiter import dtypes as aiter_dtypes
    FP8_DTYPE = aiter_dtypes.fp8
except Exception:
    FP8_DTYPE = getattr(torch, "float8_e4m3fnuz", None) or getattr(torch, "float8_e4m3fn", None)

NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

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
        build_root = Path(tempfile.gettempdir()) / "mla_hip_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name=f"mla_hip_{digest}",
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["hip_gemm_nt"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _quantize_fp8(tensor):
    """Dynamic per-tensor FP8 quantization matching reference path."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)


def _gemm_nt(a, b):
    """C = A @ B.T via HIP. A:[M,K] B:[N,K] -> C:[M,N] float32."""
    a = a.to(torch.float32).contiguous()
    b = b.to(torch.float32).contiguous()
    c = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    _module().hip_gemm_nt(a, b, c)
    return c


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = int(config["batch_size"])
    num_heads = int(config.get("num_heads", NUM_HEADS))
    v_head_dim = int(config.get("v_head_dim", V_HEAD_DIM))
    qk_head_dim = int(config.get("qk_head_dim", QK_HEAD_DIM))
    sm_scale = float(config.get("sm_scale", SM_SCALE))

    _module()

    # Match reference: fp8 quantize Q, use pre-quantized fp8 KV, dequant to float32
    q_fp8, q_scale = _quantize_fp8(q.contiguous())
    kv_fp8, kv_scale = kv_data["fp8"]

    q_f32 = q_fp8.to(torch.float32) * q_scale.item()
    kv_f32 = kv_fp8.to(torch.float32) * kv_scale.item()

    out = torch.empty((q.shape[0], num_heads, v_head_dim), dtype=torch.bfloat16, device=q.device)

    for b_idx in range(batch_size):
        q_start = int(qo_indptr[b_idx].item())
        q_end = int(qo_indptr[b_idx + 1].item())
        kv_start = int(kv_indptr[b_idx].item())
        kv_end = int(kv_indptr[b_idx + 1].item())

        if q_start == q_end:
            continue

        # KV for this batch element: [kv_len, qk_head_dim]
        kv_b = kv_f32[kv_start:kv_end, 0, :].contiguous()
        # V is first v_head_dim dims, transposed for NT kernel: [v_head_dim, kv_len]
        v_b_t = kv_b[:, :v_head_dim].T.contiguous()

        # Q for this batch element: [q_len * num_heads, qk_head_dim]
        q_b = q_f32[q_start:q_end].reshape(-1, qk_head_dim)
        q_len = q_end - q_start

        # QK^T: [q_len * num_heads, kv_len]
        scores = _gemm_nt(q_b, kv_b) * sm_scale

        # Softmax per head
        probs = torch.softmax(scores, dim=-1)

        # probs @ V via NT: probs @ (V.T).T = _gemm_nt(probs, V.T)
        # probs: [q_len * num_heads, kv_len], v_b_t: [v_head_dim, kv_len]
        # result: [q_len * num_heads, v_head_dim]
        out_b = _gemm_nt(probs, v_b_t)

        out[q_start:q_end] = out_b.reshape(q_len, num_heads, v_head_dim).to(torch.bfloat16)

    return out
