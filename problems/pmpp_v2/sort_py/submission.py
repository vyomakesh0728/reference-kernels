import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
import os

os.makedirs("./cuda_build_sort", exist_ok=True)

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <c10/cuda/CUDAStream.h>

torch::Tensor cub_sort_kernel(torch::Tensor data, torch::Tensor output) {
    const int n = data.numel();
    float* __restrict__ d_keys_in = data.data_ptr<float>();
    float* __restrict__ d_keys_out = output.data_ptr<float>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int begin_bit = 0;
    int end_bit = 32;

    // Get temp storage size
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        n,
        begin_bit,
        end_bit,
        stream
    );

    // Architecture-specific allocation
    if (prop.major == 9) {
        // H100/B200: Use regular allocation for consistency in ranked benchmarks
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    } else if (prop.major == 8 && prop.minor == 9) {
        // L4: async allocation
        cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
    } else if (prop.major == 8 && prop.minor == 0) {
        // A100: regular allocation
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    } else {
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    // Execute sort
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        n,
        begin_bit,
        end_bit,
        stream
    );

    // Architecture-specific deallocation
    if (prop.major == 8 && prop.minor == 9) {
        cudaFreeAsync(d_temp_storage, stream);
    } else {
        cudaFree(d_temp_storage);
    }

    return output;
}
"""

cpp_source = """
torch::Tensor cub_sort_kernel(torch::Tensor data, torch::Tensor output);
"""

try:
    cuda_home = (
        os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    )

    cub_sort_module = load_inline(
        name="cub_radix_sort_fixed",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["cub_sort_kernel"],
        with_cuda=True,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-I{cuda_home}/include",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        extra_include_paths=[f"{cuda_home}/include"],
        build_directory="./cuda_build_sort",
        verbose=False,
    )

    def _custom_kernel(data: input_t) -> output_t:
        input_tensor, output_tensor = data

        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        if not output_tensor.is_contiguous():
            output_tensor = output_tensor.contiguous()

        cub_sort_module.cub_sort_kernel(input_tensor, output_tensor)

        return output_tensor

    custom_kernel = _custom_kernel

except Exception as e:
    print(f"CUB kernel compilation failed: {e}")

    def _custom_kernel(data: input_t) -> output_t:
        data, output = data
        output[...] = torch.sort(data)[0]
        return output

    custom_kernel = torch.compile(_custom_kernel, mode="max-autotune")
