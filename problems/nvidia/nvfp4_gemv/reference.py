import torch
from task import input_t, output_t
from utils import make_match_reference

# Scaling factor vector size
sf_vec_size = 16

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
    m, _, l = c_ref.shape
    k = a_ref.shape[1] * 2
    k_scales = k // 16

    # ========== DEBUG PRINTS ==========
    print("\n" + "="*60)
    print("DEBUG: reference.py ref_kernel")
    print("="*60)

    print(f"\nInput tensor shapes:")
    print(f"  a_ref: {a_ref.shape} dtype={a_ref.dtype}")
    print(f"  b_ref: {b_ref.shape} dtype={b_ref.dtype}")
    print(f"  sfa_ref_cpu: {sfa_ref_cpu.shape} dtype={sfa_ref_cpu.dtype}")
    print(f"  sfb_ref_cpu: {sfb_ref_cpu.shape} dtype={sfb_ref_cpu.dtype}")
    print(f"  c_ref: {c_ref.shape} dtype={c_ref.dtype}")
    print(f"  M={m}, K={k}, L={l}, K_scales={k_scales}")

    # Print raw bytes for comparison (view as uint8)
    a_bytes = a_ref.view(torch.uint8)
    b_bytes = b_ref.view(torch.uint8)
    sfa_bytes = sfa_ref_cpu.view(torch.uint8)
    sfb_bytes = sfb_ref_cpu.view(torch.uint8)

    print(f"\nFirst 8 bytes of tensors (batch 0, i.e. [:,:,0]):")
    print(f"  a_bytes[0,:8,0]: {a_bytes[0, :min(8, a_bytes.shape[1]), 0].tolist()}")
    print(f"  b_bytes[0,:8,0]: {b_bytes[0, :min(8, b_bytes.shape[1]), 0].tolist()}")
    print(f"  sfa_bytes[0,:8,0]: {sfa_bytes[0, :min(8, sfa_bytes.shape[1]), 0].tolist()}")
    print(f"  sfb_bytes[0,:8,0]: {sfb_bytes[0, :min(8, sfb_bytes.shape[1]), 0].tolist()}")

    # Decode first FP4 value manually to verify nibble order
    if a_bytes.numel() > 0:
        packed_val = a_bytes[0, 0, 0].item()
        hi_nibble = (packed_val >> 4) & 0x0F
        lo_nibble = packed_val & 0x0F
        fp4_lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        print(f"\nFP4 decoding check (byte={packed_val:#04x}):")
        print(f"  High nibble (bits 7-4) = {hi_nibble:#x} -> FP4 value = {fp4_lut[hi_nibble]} (element 0)")
        print(f"  Low nibble (bits 3-0)  = {lo_nibble:#x} -> FP4 value = {fp4_lut[lo_nibble]} (element 1)")

    # Call torch._scaled_mm to compute the GEMV result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])

        if l_idx == 0:
            print(f"\nBlocked scale factor shapes (batch 0):")
            print(f"  scale_a (blocked): {scale_a.shape}")
            print(f"  scale_b (blocked): {scale_b.shape}")
            print(f"  scale_a[:8]: {scale_a[:8].view(torch.uint8).tolist()}")
            print(f"  scale_b[:8]: {scale_b[:8].view(torch.uint8).tolist()}")
            
            # DEBUG: Match kernel debug output format
            print("\n=== REFERENCE DEBUG (batch 0) ===")
            
            # SFB: row 0, first 4 scale columns (before blocking)
            sfb_row0 = sfb_ref_cpu[0, :4, l_idx].view(torch.uint8)
            print(f"SFB raw bytes [row=0, 0-3] (pre-block): 0x{sfb_row0[0].item():02x} 0x{sfb_row0[1].item():02x} 0x{sfb_row0[2].item():02x} 0x{sfb_row0[3].item():02x}")
            
            # Decode FP8 values
            sfb_fp8 = sfb_ref_cpu[0, :4, l_idx].float()
            print(f"SFB decoded FP8 [row=0, 0-3]: {sfb_fp8[0].item():.6f} {sfb_fp8[1].item():.6f} {sfb_fp8[2].item():.6f} {sfb_fp8[3].item():.6f}")
            
            # B packed bytes: row 0, first 4 bytes
            b_row0 = b_ref[0, :4, l_idx].view(torch.uint8)
            print(f"B packed bytes [row=0, 0-3]: 0x{b_row0[0].item():02x} 0x{b_row0[1].item():02x} 0x{b_row0[2].item():02x} 0x{b_row0[3].item():02x}")
            
            # Decode B vector manually (first 8 elements)
            fp4_lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
            b_decoded = []
            for i in range(4):
                packed = b_ref[0, i, l_idx].view(torch.uint8).item()
                scale_idx = i // 8  # same scale for 16 elements
                scale_val = sfb_ref_cpu[0, scale_idx, l_idx].float().item()
                lo = packed & 0x0F
                hi = (packed >> 4) & 0x0F
                b_decoded.append(fp4_lut[lo] * scale_val)
                b_decoded.append(fp4_lut[hi] * scale_val)
            print(f"B decoded [row=0, 0-7]: {b_decoded[0]:.6f} {b_decoded[1]:.6f} {b_decoded[2]:.6f} {b_decoded[3]:.6f} {b_decoded[4]:.6f} {b_decoded[5]:.6f} {b_decoded[6]:.6f} {b_decoded[7]:.6f}")
            
            # SFA: row 0, first 4 scale columns (before blocking)
            sfa_row0 = sfa_ref_cpu[0, :4, l_idx].view(torch.uint8)
            print(f"SFA raw bytes [row=0, 0-3] (pre-block): 0x{sfa_row0[0].item():02x} 0x{sfa_row0[1].item():02x} 0x{sfa_row0[2].item():02x} 0x{sfa_row0[3].item():02x}")
            
            # Decode SFA FP8
            sfa_fp8 = sfa_ref_cpu[0, :4, l_idx].float()
            print(f"SFA decoded FP8 [row=0, 0-3]: {sfa_fp8[0].item():.6f} {sfa_fp8[1].item():.6f} {sfa_fp8[2].item():.6f} {sfa_fp8[3].item():.6f}")
            
            # A packed bytes: row 0, first 4 bytes
            a_row0 = a_ref[0, :4, l_idx].view(torch.uint8)
            print(f"A packed bytes [row=0, 0-3]: 0x{a_row0[0].item():02x} 0x{a_row0[1].item():02x} 0x{a_row0[2].item():02x} 0x{a_row0[3].item():02x}")
            
            # Decode A[row=0] manually (first 8 elements)
            a_decoded = []
            for i in range(4):
                packed = a_ref[0, i, l_idx].view(torch.uint8).item()
                scale_idx = i // 8  # same scale for 16 elements
                scale_val = sfa_ref_cpu[0, scale_idx, l_idx].float().item()
                lo = packed & 0x0F
                hi = (packed >> 4) & 0x0F
                a_decoded.append(fp4_lut[lo] * scale_val)
                a_decoded.append(fp4_lut[hi] * scale_val)
            print(f"A decoded [row=0, 0-7]: {a_decoded[0]:.6f} {a_decoded[1]:.6f} {a_decoded[2]:.6f} {a_decoded[3]:.6f} {a_decoded[4]:.6f} {a_decoded[5]:.6f} {a_decoded[6]:.6f} {a_decoded[7]:.6f}")
            
            # Compute expected cute_scale_idx values for comparison
            # CuTe MMA layout: [32, 4, n_m_blocks, 4, n_k_blocks, L]
            # For element at (row, scale_col, batch):
            #   mm32 = row % 32
            #   mm4  = (row % 128) // 32
            #   mm   = row // 128
            #   kk4  = scale_col % 4
            #   kk   = scale_col // 4
            n_m_blocks = ceil_div(m, 128)
            n_k_blocks_ref = ceil_div(k_scales, 4)
            
            def compute_cute_idx(row, scale_col, batch_idx, n_m_blk, n_k_blk, L_val):
                mm32 = row % 32
                mm4 = (row % 128) // 32
                mm = row // 128
                kk4 = scale_col % 4
                kk = scale_col // 4
                idx = (mm32 * (4 * n_m_blk * 4 * n_k_blk * L_val)
                     + mm4 * (n_m_blk * 4 * n_k_blk * L_val)
                     + mm * (4 * n_k_blk * L_val)
                     + kk4 * (n_k_blk * L_val)
                     + kk * L_val
                     + batch_idx)
                return idx
            
            # Test indices for rows 1, 32, 128
            idx_r1 = compute_cute_idx(1, 0, l_idx, n_m_blocks, n_k_blocks_ref, l)
            idx_r32 = compute_cute_idx(32, 0, l_idx, n_m_blocks, n_k_blocks_ref, l)
            idx_r128 = compute_cute_idx(128, 0, l_idx, n_m_blocks, n_k_blocks_ref, l)
            print(f"REFERENCE cute_idx row test: r1={idx_r1} r32={idx_r32} r128={idx_r128}")
            print(f"  (n_m_blocks={n_m_blocks}, n_k_blocks={n_k_blocks_ref}, L={l})")
            
            print("=== END REFERENCE DEBUG ===")
            
            # Compute expected dot product for row 0 manually
            row0_sum = 0.0
            for k_idx in range(k // 2):
                packed_a = a_ref[0, k_idx, l_idx].view(torch.uint8).item()
                packed_b = b_ref[0, k_idx, l_idx].view(torch.uint8).item()
                scale_a_idx = k_idx // 8
                scale_b_idx = k_idx // 8
                scale_a_val = sfa_ref_cpu[0, scale_a_idx, l_idx].float().item()
                scale_b_val = sfb_ref_cpu[0, scale_b_idx, l_idx].float().item()
                
                a_lo = packed_a & 0x0F
                a_hi = (packed_a >> 4) & 0x0F
                b_lo = packed_b & 0x0F
                b_hi = (packed_b >> 4) & 0x0F
                
                a0 = fp4_lut[a_lo] * scale_a_val
                a1 = fp4_lut[a_hi] * scale_a_val
                b0 = fp4_lut[b_lo] * scale_b_val
                b1 = fp4_lut[b_hi] * scale_b_val
                
                row0_sum += a0 * b0 + a1 * b1
            print(f"REFERENCE row 0 expected dot product: {row0_sum:.4f}")

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

    print(f"\nOutput c_ref: {c_ref.shape}")
    print(f"  c_ref[:5,0,0] (first 5 elements): {c_ref[:min(5, c_ref.shape[0]), 0, 0].tolist()}")
    print("="*60 + "\n")

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
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )
    
    # Helper function to prepare the scale factor tensors for both reference
    # kernel and customize kernel. The customized data layout can be found in:
    # https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    def create_scale_factor_tensors(l, mn, sf_k):
        # Create the reference scale factor tensor (mn, sf_k, l) on CPU.
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        # Init with uint8 tensor, then convert to float8_e4m3fn
        ref_f8_random_int = torch.randint(0, 3, ref_shape, dtype=torch.int8, device='cuda')
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
        rand_int_tensor = torch.randint(0, 3, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        # Permute according to mma_permute_order
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        # GPU-side vectorized reordering (replaces slow CPU nested loops)
        # Create index grids for all dimensions
        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')
        
        # Create meshgrid for all combinations of (i, j, b)
        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')
        
        # Calculate target indices in vectorized manner
        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k
        
        # Perform the reordering with advanced indexing (all on GPU)
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]
        
        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k)

    sfa_ref = sfa_ref_cpu.to("cuda")
    sfb_ref = sfb_ref_cpu.to("cuda")
    
    return (a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
