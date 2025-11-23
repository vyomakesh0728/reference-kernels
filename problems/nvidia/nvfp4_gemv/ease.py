import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass.cute.runtime import make_ptr
from task import input_t, output_t

mma_tiler_mnk = (128, 1, 256)
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
threads_per_cta = 256


def ceil_div(a, b):
    return (a + b - 1) // b


@cute.kernel
def kernel_bandwidth_optimized(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )

    tCgC = gC_mnl[tidx, None, bidx, bidy, bidz]
    tCgC = cute.make_tensor(tCgC.iterator, 1)
    res = cute.zeros_like(tCgC, cutlass.Float32)

    k_tile_cnt = gA_mkl.layout[3].shape

    for k_tile in range(k_tile_cnt):
        tAgA = gA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgB = gB_nkl[0, None, bidy, k_tile, bidz]
        tAgSFA = gSFA_mkl[tidx, None, bidx, k_tile, bidz]
        tBgSFB = gSFB_nkl[0, None, bidy, k_tile, bidz]

        tArA = cute.make_rmem_tensor_like(tAgA, cutlass.Float32)
        tBrB = cute.make_rmem_tensor_like(tBgB, cutlass.Float32)
        tArSFA = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)
        tBrSFB = cute.make_rmem_tensor_like(tBgSFB, cutlass.Float32)

        a_val_nvfp4 = tAgA.load()
        b_val_nvfp4 = tBgB.load()
        sfa_val_fp8 = tAgSFA.load()
        sfb_val_fp8 = tBgSFB.load()

        a_val = a_val_nvfp4.to(cutlass.Float32)
        b_val = b_val_nvfp4.to(cutlass.Float32)
        sfa_val = sfa_val_fp8.to(cutlass.Float32)
        sfb_val = sfb_val_fp8.to(cutlass.Float32)

        tArA.store(a_val)
        tBrB.store(b_val)
        tArSFA.store(sfa_val)
        tBrSFB.store(sfb_val)

        for i in cutlass.range_constexpr(mma_tiler_mnk[2] // 8):
            ii = i * 8
            res += tArA[ii] * tArSFA[ii] * tBrB[ii] * tBrSFB[ii]
            res += tArA[ii + 1] * tArSFA[ii + 1] * tBrB[ii + 1] * tBrSFB[ii + 1]
            res += tArA[ii + 2] * tArSFA[ii + 2] * tBrB[ii + 2] * tBrSFB[ii + 2]
            res += tArA[ii + 3] * tArSFA[ii + 3] * tBrB[ii + 3] * tBrSFB[ii + 3]
            res += tArA[ii + 4] * tArSFA[ii + 4] * tBrB[ii + 4] * tBrSFB[ii + 4]
            res += tArA[ii + 5] * tArSFA[ii + 5] * tBrB[ii + 5] * tBrSFB[ii + 5]
            res += tArA[ii + 6] * tArSFA[ii + 6] * tBrB[ii + 6] * tBrSFB[ii + 6]
            res += tArA[ii + 7] * tArSFA[ii + 7] * tBrB[ii + 7] * tBrSFB[ii + 7]

    tCgC.store(res.to(cutlass.Float16))
    return


@cute.jit
def my_kernel_optimized(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    m, _, k, l = problem_size
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 128), l),
            stride=(cute.assume(k, 128), 1, cute.assume(m * k, 128)),
        ),
    )
    n_padded_128 = 128
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n_padded_128, cute.assume(k, 128), l),
            stride=(cute.assume(k, 128), 1, cute.assume(n_padded_128 * k, 128)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 128), 1, l), stride=(1, 1, m))
    )
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

    grid = (
        cute.ceil_div(c_tensor.shape[0], mma_tiler_mnk[0]),
        1,
        c_tensor.shape[2],
    )

    kernel_bandwidth_optimized(
        a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )


_compiled_kernel_cache = None


def compile_kernel():
    global _compiled_kernel_cache

    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache

    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    _compiled_kernel_cache = cute.compile(
        my_kernel_optimized, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0)
    )

    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    compiled_func = compile_kernel()

    m, k, l = a.shape
    k = k * 2
    n = 1

    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c
