import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define MMA_M 16
#define MMA_N 8
#define MMA_K 64

__device__ __forceinline__ void unpack_nvfp4_to_fp16(
    const uint8_t* __restrict__ packed,
    half* __restrict__ unpacked,
    const half* __restrict__ scale_fp8,
    const float global_scale,
    int elements
) {
    int idx = threadIdx.x * 4;
    if (idx < elements) {
        uint8_t p = packed[idx >> 1];

        #pragma unroll
        for(int i = 0; i < 4; i += 2) {
            uint8_t v0 = (p >> (i*4)) & 0xF;
            uint8_t v1 = (p >> ((i+1)*4)) & 0xF;

            int exp0 = (v0 >> 1) & 0x3;
            int man0 = v0 & 0x1;
            int sign0 = (v0 >> 3) & 0x1;

            int exp1 = (v1 >> 1) & 0x3;
            int man1 = v1 & 0x1;
            int sign1 = (v1 >> 3) & 0x1;

            float f0 = (sign0 ? -1.0f : 1.0f) * (1.0f + man0) * powf(2.0f, exp0 - 1.0f);
            float f1 = (sign1 ? -1.0f : 1.0f) * (1.0f + man1) * powf(2.0f, exp1 - 1.0f);

            int block_idx = (idx + i) / 16;
            float sf = __half2float(scale_fp8[block_idx]) * global_scale;

            unpacked[idx + i] = __float2half(f0 * sf);
            unpacked[idx + i + 1] = __float2half(f1 * sf);
        }
    }
}

__global__ void __launch_bounds__(256) fused_nvfp4_gemm_batched(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ b_packed,
    const half* __restrict__ sfa_fp8,
    const half* __restrict__ sfb_fp8,
    half* __restrict__ c,
    int M, int K, int L
) {
    extern __shared__ char smem[];
    half* a_smem = (half*)smem;
    half* b_smem = a_smem + 256 * 64;

    int batch = blockIdx.z;
    if(batch >= L) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int m_block = blockIdx.x * 128;

    wmma::fragment<wmma::matrix_a, 16, 8, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 8, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 8, 16, half> acc_frag;

    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    int a_offset = batch * M * (K/2);
    int b_offset = batch * (K/2);
    int sfa_offset = batch * M * (K/16);
    int sfb_offset = batch * (K/16);

    for(int k_tile = 0; k_tile < K; k_tile += 64) {
        __syncthreads();

        if(threadIdx.x < 128) {
            int m_idx = m_block + threadIdx.x;
            if(m_idx < M) {
                unpack_nvfp4_to_fp16(
                    a_packed + a_offset + m_idx * (K/2) + (k_tile/2),
                    a_smem + threadIdx.x * 64,
                    sfa_fp8 + sfa_offset + m_idx * (K/16) + (k_tile/16),
                    1.0f,
                    64
                );
            }
        }

        if(threadIdx.x < 2) {
            unpack_nvfp4_to_fp16(
                b_packed + b_offset + (k_tile/2),
                b_smem + threadIdx.x * 32,
                sfb_fp8 + sfb_offset + (k_tile/16),
                1.0f,
                64
            );
        }

        __syncthreads();

        for(int k = 0; k < 64; k += 16) {
            int m_warp = warp_id * 16;

            wmma::load_matrix_sync(a_frag, a_smem + m_warp * 64 + k, 64);
            wmma::load_matrix_sync(b_frag, b_smem + k, 64);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int m_warp = warp_id * 16;
    if(m_block + m_warp < M) {
        wmma::store_matrix_sync(c + batch * M + m_block + m_warp, acc_frag, 1, wmma::mem_row_major);
    }
}

std::vector<torch::Tensor> nvfp4_gemm_batched(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor sfa,
    torch::Tensor sfb,
    int64_t M,
    int64_t K,
    int64_t L
) {
    auto c = torch::zeros({L, M, 1}, torch::dtype(torch::kFloat16).device(a_packed.device()));

    dim3 grid((M + 127) / 128, 1, L);
    dim3 block(256);
    int smem = (256 * 64 + 64) * sizeof(half);

    fused_nvfp4_gemm_batched<<<grid, block, smem>>>(
        a_packed.data_ptr<uint8_t>(),
        b_packed.data_ptr<uint8_t>(),
        sfa.data_ptr<half>(),
        sfb.data_ptr<half>(),
        c.data_ptr<half>(),
        M, K, L
    );

    return {c};
}
"""

cpp_source = """
std::vector<torch::Tensor> nvfp4_gemm_batched(
    torch::Tensor a_packed,
    torch::Tensor b_packed,
    torch::Tensor sfa,
    torch::Tensor sfb,
    int64_t M,
    int64_t K,
    int64_t L
);
"""

module = None


def get_module():
    global module
    if module is None:
        module = load_inline(
            name="nvfp4_cutlass_opt",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["nvfp4_gemm_batched"],
            verbose=False,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--maxrregcount=128",
                "-arch=sm_100",
                "--ftz=true",
                "--prec-div=false",
                "--prec-sqrt=false",
            ],
        )
    return module


def custom_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    M, K, L = c_ref.shape

    sfa_gpu = sfa_ref_cpu.cuda(non_blocking=True)
    sfb_gpu = sfb_ref_cpu.cuda(non_blocking=True)

    mod = get_module()
    result = mod.nvfp4_gemm_batched(
        a_ref.reshape(M, -1, L), b_ref.reshape(1, -1, L), sfa_gpu, sfb_gpu, M, K, L
    )

    c_ref[:, 0, :] = result[0].permute(1, 0, 2).squeeze(1)
    return c_ref
