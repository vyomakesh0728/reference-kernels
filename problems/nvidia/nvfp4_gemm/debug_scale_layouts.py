#!/usr/bin/env python3
"""
Debug script to compare scale factor byte layouts:
1. to_blocked(sfa_ref_cpu) - what torch._scaled_mm expects
2. sfa_permuted.flatten() - what we're passing to tcgen05 kernel
"""

import torch
from reference import to_blocked, ceil_div, generate_input

def debug_scale_layouts():
    # Test with small dimensions for clarity
    M, N, K, L = 128, 256, 256, 1
    seed = 42
    
    # Generate input data
    data = generate_input(M, N, K, L, seed)
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, sfa_permuted, sfb_permuted, c_ref = data
    
    print(f"Testing with M={M}, N={N}, K={K}, L={L}")
    print(f"sfa_ref_cpu shape: {sfa_ref_cpu.shape}")  # (M, K_scales, L)
    print(f"sfa_permuted shape: {sfa_permuted.shape}")  # (32, 4, rest_m, 4, rest_k, L)
    
    # Get L=0 slice
    sfa_ref_l0 = sfa_ref_cpu[:, :, 0]  # (M, K_scales)
    sfa_perm_l0 = sfa_permuted[..., 0]  # (32, 4, rest_m, 4, rest_k)
    
    # Apply to_blocked to sfa_ref
    blocked = to_blocked(sfa_ref_l0)  # Returns flattened 1D tensor
    
    # Flatten sfa_permuted for comparison
    perm_flat = sfa_perm_l0.flatten()
    
    print(f"\nto_blocked output shape: {blocked.shape}")
    print(f"sfa_permuted flattened shape: {perm_flat.shape}")
    print(f"Total bytes match: {blocked.numel() == perm_flat.numel()}")
    
    # Convert to uint8 for byte comparison
    blocked_bytes = blocked.view(torch.uint8).cpu()
    perm_bytes = perm_flat.view(torch.uint8).cpu()
    
    print(f"\nBlocked bytes shape: {blocked_bytes.shape}")
    print(f"Permuted bytes shape: {perm_bytes.shape}")
    
    # Compare byte by byte
    if blocked_bytes.numel() == perm_bytes.numel():
        matches = (blocked_bytes == perm_bytes).sum().item()
        total = blocked_bytes.numel()
        print(f"\nByte comparison: {matches}/{total} bytes match ({100*matches/total:.2f}%)")
        
        if matches != total:
            # Find first mismatch
            mismatches = (blocked_bytes != perm_bytes).nonzero(as_tuple=True)[0]
            print(f"First 20 mismatch indices: {mismatches[:20].tolist()}")
            
            for i in mismatches[:5].tolist():
                print(f"  Index {i}: blocked={blocked_bytes[i].item()}, permuted={perm_bytes[i].item()}")
            
            # Show the first 64 bytes of each for visual comparison
            print(f"\nFirst 64 bytes of to_blocked:")
            print(blocked_bytes[:64].tolist())
            print(f"\nFirst 64 bytes of sfa_permuted.flatten():")
            print(perm_bytes[:64].tolist())
    else:
        print(f"ERROR: Size mismatch! blocked={blocked_bytes.numel()}, permuted={perm_bytes.numel()}")
    
    # Also analyze the M=128, K_scales=16 tile structure
    K_scales = K // 16
    print(f"\n--- Tile Structure Analysis ---")
    print(f"M = {M}, K_scales = {K_scales}")
    print(f"sfa_ref_cpu[0:4, 0:4] (first 4x4 corner):")
    print(sfa_ref_l0[:4, :4].view(torch.uint8).cpu().tolist())
    
    # Show how to_blocked permutes data
    # to_blocked: view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    #             .reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    n_row_blocks = ceil_div(M, 128)
    n_col_blocks = ceil_div(K_scales, 4)
    print(f"\nto_blocked intermediate shapes:")
    print(f"  view({n_row_blocks}, 128, {n_col_blocks}, 4) -> permute(0,2,1,3)")
    print(f"  -> reshape(-1, 4, 32, 4) -> transpose(1, 2)")
    print(f"  -> reshape(-1, 32, 16) -> flatten()")
    
    # Show sfa_permuted structure
    print(f"\nsfa_permuted 6D layout: (32, 4, rest_m, 4, rest_k)")
    rest_m = ceil_div(M, 128)
    rest_k = ceil_div(K_scales, 4)
    print(f"  rest_m = {rest_m}, rest_k = {rest_k}")
    print(f"  Total elements = 32 * 4 * {rest_m} * 4 * {rest_k} = {32*4*rest_m*4*rest_k}")
    
    # Sample some specific positions
    print(f"\nSampling specific positions:")
    for m in [0, 31, 32, 127]:
        for k in [0, 3, 4, 15]:
            if m < M and k < K_scales:
                ref_val = sfa_ref_l0[m, k].view(torch.uint8).item()
                
                # Calculate 6D index
                mm32 = m % 32
                mm4 = (m % 128) // 32
                mm = m // 128
                kk = k // 4
                kk4 = k % 4
                
                if mm < rest_m and kk < rest_k:
                    perm_val = sfa_perm_l0[mm32, mm4, mm, kk4, kk].view(torch.uint8).item()
                    match_str = "✓" if ref_val == perm_val else "✗"
                    print(f"  m={m:3d}, k={k:2d}: ref={ref_val:3d}, perm[{mm32},{mm4},{mm},{kk4},{kk}]={perm_val:3d} {match_str}")

if __name__ == "__main__":
    debug_scale_layouts()
