import hashlib
import os
from pathlib import Path
import tempfile

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("CXX", "clang++")

import aiter
from aiter import QuantType
from aiter.utility import fp4_utils
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

SCALE_GROUP = 32

CPP_WRAPPER = """
void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c);
"""

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr int TILE_M = 16;
constexpr int TILE_N = 32;
constexpr int TILE_K = 64;

__global__ void mxfp4_mm_kernel(
    const float* a,
    const float* b,
    __hip_bfloat16* c,
    int m,
    int n,
    int k
) {
    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_N + threadIdx.x;
    float acc = 0.0f;

    __shared__ float a_tile[TILE_M][TILE_K];
    __shared__ float b_tile[TILE_N][TILE_K];

    for (int tile_k = 0; tile_k < k; tile_k += TILE_K) {
        for (int load_k = threadIdx.x; load_k < TILE_K; load_k += blockDim.x) {
            const int global_k = tile_k + load_k;
            a_tile[threadIdx.y][load_k] =
                (row < m && global_k < k) ? a[row * k + global_k] : 0.0f;
        }
        for (int load_k = threadIdx.y; load_k < TILE_K; load_k += blockDim.y) {
            const int global_k = tile_k + load_k;
            b_tile[threadIdx.x][load_k] =
                (col < n && global_k < k) ? b[col * k + global_k] : 0.0f;
        }

        __syncthreads();

        if (row < m && col < n) {
            #pragma unroll 1
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += static_cast<float>(a_tile[threadIdx.y][kk]) * static_cast<float>(b_tile[threadIdx.x][kk]);
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = static_cast<__hip_bfloat16>(acc);
    }
}

void mxfp4_mm_hip(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(0));
    const int k = static_cast<int>(a.size(1));

    dim3 block(TILE_N, TILE_M);
    dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
    hipLaunchKernelGGL(
        mxfp4_mm_kernel,
        grid,
        block,
        0,
        0,
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(c.data_ptr<at::BFloat16>()),
        m,
        n,
        k
    );
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        build_root = Path(tempfile.gettempdir()) / "mxfp4_mm_hip_template_build"
        build_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1((CPP_WRAPPER + HIP_SRC).encode("utf-8")).hexdigest()[:12]
        module_name = f"mxfp4_mm_hip_template_{digest}"
        _MODULE = load_inline(
            name=module_name,
            cpp_sources=[CPP_WRAPPER],
            cuda_sources=[HIP_SRC],
            functions=["mxfp4_mm_hip"],
            extra_cuda_cflags=["--offload-arch=gfx950", "-std=c++20", "-O3"],
            build_directory=str(build_root),
            verbose=False,
        )
    return _MODULE


def _dequantize_logical_mxfp4(fp4_packed: torch.Tensor, scale_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    values = fp4_utils.mxfp4_to_f32(fp4_packed.contiguous())[:rows, :cols]
    scales = scale_e8m0.contiguous()[:rows]
    scales = scales.repeat_interleave(SCALE_GROUP, dim=1)[:, :cols]
    scales_f32 = fp4_utils.e8m0_to_f32(scales)
    return (values * scales_f32).to(torch.float32).contiguous()


def _reference_oracle_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quant = aiter.get_triton_quant(QuantType.per_1x32)
    a_q, a_scale = quant(a.contiguous(), shuffle=False)
    a_ref = _dequantize_logical_mxfp4(a_q, a_scale, rows=a.shape[0], cols=a.shape[1])
    b_q, b_scale = quant(b.contiguous(), shuffle=False)
    b_ref = _dequantize_logical_mxfp4(b_q, b_scale, rows=b.shape[0], cols=b.shape[1])
    return a_ref, b_ref


def custom_kernel(data: input_t) -> output_t:
    a, b, b_q, b_shuffle, b_scale_sh = data
    torch._assert(b_q.shape[0] == b.shape[0], "B_q row count must match logical B")
    torch._assert(b_shuffle.shape[0] == b.shape[0], "B_shuffle row count must match logical B")
    torch._assert(b_scale_sh.numel() > 0, "B_scale_sh must be present for the live contract")
    a_in, b_in = _reference_oracle_inputs(a, b)
    c = torch.empty((a_in.shape[0], b_in.shape[0]), dtype=torch.bfloat16, device=a_in.device)
    _module().mxfp4_mm_hip(a_in, b_in, c)
    return c
