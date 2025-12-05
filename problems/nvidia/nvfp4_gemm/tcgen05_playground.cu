// Minimal tcgen05.mma playground for SM100/SM100a.
// Based on CUTLASS cute/tutorial/blackwell/01_mma_sm100.cu
// and PTX ISA tcgen05.mma documentation.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "cutlass/numeric_types.h"
#include <cute/arch/mma_sm100_desc.hpp>

using cutlass::half_t;

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 16;

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr32;
    asm volatile(
        "{\n"
        "  .reg .u64 addr64;\n"
        "  cvta.to.shared.u64 addr64, %1;\n"
        "  cvt.u32.u64 %0, addr64;\n"
        "}\n"
        : "=r"(addr32)
        : "l"(ptr)
    );
    return addr32;
}

__global__ void tcgen05_kernel(const half_t* A, const half_t* B, float* C) {
#if __CUDA_ARCH__ >= 1000
    __shared__ uint32_t tmem_base_ptr;
    extern __shared__ unsigned char smem_raw[];
    half_t* smem_A = reinterpret_cast<half_t*>(smem_raw);
    half_t* smem_B = smem_A + M * K;

    int tid = threadIdx.x;

    // Copy A and B tiles from global memory to shared memory.
    for (int idx = tid; idx < M * K; idx += blockDim.x) {
        smem_A[idx] = A[idx];
    }
    for (int idx = tid; idx < K * N; idx += blockDim.x) {
        smem_B[idx] = B[idx];
    }

    __syncthreads();

    if (tid == 0) {
        // Naive GEMM for verification: C = A * B, shapes MxK, KxN.
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a = static_cast<float>(smem_A[m * K + k]);
                    float b = static_cast<float>(smem_B[k * N + n]);
                    acc += a * b;
                }
                C[m * N + n] = acc;
            }
        }
    }

    __syncthreads();

    // Allocate TMEM - ALL threads must participate (collective operation)
    if (tid == 0) {
        tmem_base_ptr = 0;  // Initialize
    }
    __syncthreads();
    
    // ALL threads execute this (remove the if statement!)
    {
        uint32_t dst_smem = cvta_to_shared_u32(&tmem_base_ptr);
        int num_columns = 512;  // full SM100 TMEM slice
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :
            : "r"(dst_smem), "r"(num_columns));
    }

    __syncthreads();

    uint32_t tmem_c = tmem_base_ptr; 

    // Construct simple SMEM descriptors from the base shared-memory addresses.
    uint64_t a_desc = static_cast<uint64_t>(cvta_to_shared_u32(smem_A));
    uint64_t b_desc = static_cast<uint64_t>(cvta_to_shared_u32(smem_B));

    // Build an instruction descriptor for F16xF16->F32, MxN = 128x128, K-major A/B.
    uint64_t idescE = cute::UMMA::make_runtime_instr_desc<
        half_t, half_t, float,
        M, N,
        cute::UMMA::Major::K,
        cute::UMMA::Major::K>();

    uint32_t scaleC = 1u;  // overwrite accumulator on first use
    uint32_t mask[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};

    // Issue a single tcgen05.mma.cta_group::1.kind::f16 using the CUTLASS UMMA signature.
    if (tid == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n"
            "}\n"
            :
            : "r"(tmem_c),
              "l"(a_desc),
              "l"(b_desc),
              "r"(static_cast<uint32_t>(idescE >> 32)),
              "r"(scaleC),
              "r"(mask[0]),
              "r"(mask[1]),
              "r"(mask[2]),
              "r"(mask[3]));
    }

    // Use tcgen05.ld to read back a small portion of the accumulator tile from TMEM.
    volatile uint32_t acc0 = 0, acc1 = 0;
    if (tid == 0) {
        asm volatile(
            "tcgen05.ld.sync.aligned.16x128b.x1.b32 "
            "{%0, %1}, [%2];\n"
            : "=r"(acc0), "=r"(acc1)
            : "r"(tmem_c));
        printf("TMEM readback: acc0=%u acc1=%u\n", acc0, acc1);
    }
#endif
}

int main() {
    const std::size_t size_A = static_cast<std::size_t>(M) * K;
    const std::size_t size_B = static_cast<std::size_t>(K) * N;
    const std::size_t size_C = static_cast<std::size_t>(M) * N;

    half_t* hA = new half_t[size_A];
    half_t* hB = new half_t[size_B];
    float*  hC = new float[size_C];

    for (std::size_t i = 0; i < size_A; ++i) hA[i] = half_t(1.0f);
    for (std::size_t i = 0; i < size_B; ++i) hB[i] = half_t(1.0f);

    half_t* dA = nullptr;
    half_t* dB = nullptr;
    float*  dC = nullptr;

    cudaMalloc(&dA, size_A * sizeof(half_t));
    cudaMalloc(&dB, size_B * sizeof(half_t));
    cudaMalloc(&dC, size_C * sizeof(float));

    cudaMemcpy(dA, hA, size_A * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size_B * sizeof(half_t), cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(128, 1, 1);
    std::size_t smem_bytes = (static_cast<std::size_t>(M) * K + static_cast<std::size_t>(K) * N) * sizeof(half_t);

    // Use cluster launch for tcgen05 instructions
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;  // cluster size = 1 CTA
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaError_t launch_err = cudaLaunchKernelEx(&config, tcgen05_kernel, dA, dB, dC);

    // CRITICAL: Check launch errors first (PTX assembly issues)
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch error: %s\n",
                    cudaGetErrorString(launch_err));
        return 1;
    }

    // CRITICAL: Check runtime errors (TMEM access, illegal instructions)
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize error: %s\n",
                    cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, dC, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    std::printf("C[0,0]=%f C[0,1]=%f (expected %d)\n", hC[0], hC[1], K);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}
