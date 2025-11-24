import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr
from task import input_t, output_t

# Optimization: Increase K tile size to 2048 to maximize memory burst/bandwidth.
# 2048 elements * 4 bits = 1024 bytes.
# 128 rows * 1024 bytes = 128 KB per tile, fitting well in RF/Smem.
mma_tiler_mnk = (128, 1, 2048)
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
threads_per_cta = 256


@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # Shared Memory for Reduction: (128 rows, 32 cols/threads)
    # Used to aggregate partial sums from the K-contiguous threads
    smem = cutlass.utils.SmemAllocator()
    sC = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(shape=(128, 32), stride=(32, 1)),
        byte_alignment=16,
    )

    copy_op = cute.nvgpu.CopyUniversalOp()
    copy_atom_AB = cute.make_copy_atom(copy_op, ab_dtype, num_bits_per_copy=128)

    # Thread/value layouts for A/B
    thr_layout_AB = cute.make_layout((8, 32), stride=(32, 1))  # threads over (M,K)
    val_layout_AB = cute.make_layout((1, 32), stride=(32, 1))  # 32 vals/thread along K

    # Turn (thr, val) layouts into (layout_tv, tiler_mn)
    tiler_AB, layout_tv_AB = cute.make_layout_tv(thr_layout_AB, val_layout_AB)

    # Legacy-style TiledCopy: (layout_tv, tiler_mn)
    loader_AB = cute.make_tiled_copy(
        copy_atom_AB,
        layout_tv_AB,
        tiler_AB,
    )

    # SFA/SFB loader: same pattern, just different dtype + copy_atom
    copy_atom_SF = cute.make_copy_atom(copy_op, sf_dtype, num_bits_per_copy=16)

    thr_layout_SF = cute.make_layout((8, 32), stride=(32, 1))
    val_layout_SF = cute.make_layout((1, 32), stride=(32, 1))
    tiler_SF, layout_tv_SF = cute.make_layout_tv(thr_layout_SF, val_layout_SF)

    loader_SF = cute.make_tiled_copy(
        copy_atom_SF,
        layout_tv_SF,
        tiler_SF,
    )

    # Accumulators: Each thread covers 16 rows (128 / 8)
    accum = cute.full((16,), 0.0, cutlass.Float32)

    # Global Loop over K-dimension
    k_tile_count = cute.ceil_div(mA_mkl.shape[1], mma_tiler_mnk[2])
    for k_iter in range(k_tile_count):
        # Slice Global Tiles for current K-block
        gA_k = cute.local_tile(mA_mkl, mma_tiler_mnk, (bidx, k_iter, bidz))
        gB_k = cute.local_tile(mB_nkl, mma_tiler_mnk, (0, k_iter, bidz))
        gSFA_k = cute.local_tile(mSFA_mkl, mma_tiler_mnk, (bidx, k_iter, bidz))
        gSFB_k = cute.local_tile(mSFB_nkl, mma_tiler_mnk, (0, k_iter, bidz))

        # Partition Data to Threads
        tAgA = loader_AB.get_slice(tidx).partition_S(gA_k)
        tBgB = loader_AB.get_slice(tidx).partition_S(gB_k)
        tAgSFA = loader_SF.get_slice(tidx).partition_S(gSFA_k)
        tBgSFB = loader_SF.get_slice(tidx).partition_S(gSFB_k)

        # RMEM (register) tensors with same shape as the thread tiles, in FP32 for math
        rA = cute.make_rmem_tensor_like(tAgA, cutlass.Float32)
        rB = cute.make_rmem_tensor_like(tBgB, cutlass.Float32)
        rSFA = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)
        rSFB = cute.make_rmem_tensor_like(tBgSFB, cutlass.Float32)

        cute.copy(loader_AB, tAgA, rA)
        cute.copy(loader_AB, tBgB, rB)
        cute.copy(loader_SF, tAgSFA, rSFA)
        cute.copy(loader_SF, tBgSFB, rSFB)

        # Math Loop (Processing loaded registers)
        # rA shape implies we have 16 M-steps per thread.
        for m_i in range(16):
            # Pre-load scales for this vector (2 scales for 32 elements)
            sfa_0 = float(rSFA[((0, 0), m_i, 0)])
            sfa_1 = float(rSFA[((0, 1), m_i, 0)])
            sfb_0 = float(rSFB[((0, 0), m_i, 0)])
            sfb_1 = float(rSFB[((0, 1), m_i, 0)])

            # Vector Dot Product
            sum_val = 0.0
            for v in range(32):
                val_a = float(rA[((0, v), m_i, 0)])
                val_b = float(rB[((0, v), m_i, 0)])
                scale_a = sfa_0 if v < 16 else sfa_1
                scale_b = sfb_0 if v < 16 else sfb_1
                sum_val += val_a * scale_a * val_b * scale_b

            accum[m_i] += sum_val

    # Reduction Phase
    # 1. Store Partial Sums to Shared Memory
    col_id = tidx % 32  # K-slice ID (0..31)

    for m_i in range(16):
        row = (tidx // 32) + m_i * 8
        sC[row, col_id] = accum[m_i]

    cute.arch.sync_threads()

    # 2. Sum across K-slices (cols) and Write to Global
    # Only the first column of threads performs the reduction and write
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, (bidx, 0, bidz))

    if col_id == 0:
        for m_i in range(16):
            row = (tidx // 32) + m_i * 8
            total = 0.0
            # Sum 32 partial results from Shared Memory
            for k_i in range(32):
                total += sC[row, k_i]
            gC[row, 0, 0] = cutlass.Float16(total)


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    m, _, k, l = problem_size

    # Define Tensors
    # A: (M, K, L) - K-major (Stride: K, 1, M*K)
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(shape=(m, k, l), stride=(k, 1, m * k)),
    )

    # B: (1, K, L) - Treated as broadcastable in M
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            shape=(1, k, l), stride=(k, 1, k)
        ),  # M-stride is 0 effectively
    )

    # SFA: (M, K/16, L)
    sfa_tensor = cute.make_tensor(
        sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    )

    # SFB: (1, K/16, L)
    sfb_tensor = cute.make_tensor(
        sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    )

    # C: (M, 1, L)
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout(shape=(m, 1, l), stride=(1, 1, m))
    )

    grid = (cute.ceil_div(m, mma_tiler_mnk[0]), 1, l)
    kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )

    return


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
        my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0)
    )

    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    a, b, _, _, sfa_permuted, sfb_permuted, c = data
    compiled_func = compile_kernel()

    m, k, l = a.shape
    # Note: Python side K is logical elements.
    # Kernel expects logical dimensions.

    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    sfb_ptr = make_ptr(
        sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, 1, k, l))

    return c
