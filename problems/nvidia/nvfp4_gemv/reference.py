import numpy as np
import torch
from task import input_t, output_t
from utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16

# FP4 E2M1 lookup table (same as in CUDA kernel)
fp4_e2m1_lut = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def decode_fp4_e2m1(nibble):
    """Decode a single FP4 E2M1 nibble to float"""
    return fp4_e2m1_lut[nibble & 0x0F]


def decode_fp8_e4m3(byte_val):
    """Decode FP8 E4M3 to float using torch"""
    # Create a torch tensor with the byte value and convert to float8_e4m3fn
    tensor = torch.tensor([byte_val], dtype=torch.uint8).view(torch.float8_e4m3fn)
    return tensor.to(torch.float32).item()


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data

    # Get dimensions from MxNxL layout
    M, K_packed, L = a_ref.shape
    K = K_packed * 2
    K_scales = K // 16

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(L):
        # ====================================================================
        # DEBUG: Print decoded FP4 values and scales for first batch
        # ====================================================================
        if l_idx == 0:
            print("=" * 80)
            print("[REFERENCE] Decoding FP4 values for batch 0")
            print("=" * 80)

            # Convert tensors to CPU numpy for easier indexing
            a_bytes = a_ref[:, :, l_idx].cpu().view(torch.uint8).numpy()
            b_bytes = b_ref[:, :, l_idx].cpu().view(torch.uint8).numpy()
            sfa_bytes = sfa_ref_cpu[:, :, l_idx].view(torch.uint8).numpy()
            sfb_bytes = sfb_ref_cpu[:, :, l_idx].view(torch.uint8).numpy()

            # Print A matrix values (first 3 rows, first 32 elements)
            for m in range(min(3, M)):
                for k_packed in range(min(16, K_packed)):  # 16 packed = 32 unpacked
                    packed_byte = a_bytes[m, k_packed]
                    scale_idx = k_packed // 8
                    scale_byte = sfa_bytes[m, scale_idx] if scale_idx < K_scales else 0
                    scale_val = decode_fp8_e4m3(scale_byte)

                    # Decode two FP4 values from packed byte
                    # High nibble is first element, low nibble is second
                    nibble0 = (packed_byte >> 4) & 0x0F
                    nibble1 = packed_byte & 0x0F
                    fp4_0 = decode_fp4_e2m1(nibble0)
                    fp4_1 = decode_fp4_e2m1(nibble1)
                    scaled_0 = fp4_0 * scale_val
                    scaled_1 = fp4_1 * scale_val

                    k0 = k_packed * 2
                    k1 = k0 + 1

                    print(
                        f"[REF A] batch={l_idx} m={m} k={k0},{k1} | "
                        f"packed=0x{packed_byte:02x} scale_byte=0x{scale_byte:02x} scale={scale_val:.6f} | "
                        f"fp4_raw=({fp4_0:.6f},{fp4_1:.6f}) scaled=({scaled_0:.6f},{scaled_1:.6f})"
                    )

            # Print B vector values (first 32 elements)
            print()
            for k_packed in range(min(16, K_packed)):  # 16 packed = 32 unpacked
                packed_byte = b_bytes[
                    0, k_packed
                ]  # B is [1, K_packed, L] -> [n=0, k_packed]
                scale_idx = k_packed // 8
                scale_byte = sfb_bytes[0, scale_idx] if scale_idx < K_scales else 0
                scale_val = decode_fp8_e4m3(scale_byte)

                # Decode two FP4 values
                nibble0 = (packed_byte >> 4) & 0x0F
                nibble1 = packed_byte & 0x0F
                fp4_0 = decode_fp4_e2m1(nibble0)
                fp4_1 = decode_fp4_e2m1(nibble1)
                scaled_0 = fp4_0 * scale_val
                scaled_1 = fp4_1 * scale_val

                k0 = k_packed * 2
                k1 = k0 + 1

                print(
                    f"[REF B] batch={l_idx} k={k0},{k1} | "
                    f"packed=0x{packed_byte:02x} scale_byte=0x{scale_byte:02x} scale={scale_val:.6f} | "
                    f"fp4_raw=({fp4_0:.6f},{fp4_1:.6f}) scaled=({scaled_0:.6f},{scaled_1:.6f})"
                )

            print("=" * 80)

        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]
    return c_ref


def generate_input(
    m: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled GEMV.

    Args:
        m: Number of rows in matrix A
        k: Number of columns in A (and length of vector b)
        l: Batch size
        seed: Random seed for reproducibility

    Returns:
        Tuple of (a, b, scale_a, scale_b, c) where:
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [1, k, l] - Input vector in torch.float4e2m1fn_x2 data type
            scale_a: [m, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b: [1, k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_a_permuted: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            scale_b_permuted: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, 1, l] - Output vector in torch.float16 data type
    """
    torch.manual_seed(seed)

    # GEMV N dimension is always 1
    n = 1
    # Scaling factor needs to pad the N size to 128
    n_padded_128 = 128

    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    # Pad b tensor's N dimension to 128 to call torch._scaled_mm for nvfp4 dot product computation
    b_ref = torch.randint(
        0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(
            0, 3, ref_shape, dtype=torch.int8, device="cuda"
        )
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        # permute to match ref_permute_order
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,  # batch size
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        # Reorder scale factor tensor to (32, 4, rest_m, 4, rest_k, l) layout
        # Which is needed by the CuTe customized kernel
        mma_permute_order = (3, 4, 1, 5, 2, 0)
        # Generate a random int8 tensor, then convert to float8_e4m3fn
        rand_int_tensor = torch.randint(
            0, 3, mma_shape, dtype=torch.int8, device="cuda"
        )
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(
            *mma_permute_order
        )

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device="cuda")
        j_idx = torch.arange(sf_k, device="cuda")
        b_idx = torch.arange(l, device="cuda")

        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing="ij")

        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = (
            ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]
        )

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

    sfa_ref = sfa_ref_cpu.to("cuda")
    sfb_ref = sfb_ref_cpu.to("cuda")

    return (a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
