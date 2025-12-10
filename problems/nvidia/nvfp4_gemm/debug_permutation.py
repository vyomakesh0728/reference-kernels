
import torch

def ceil_div(a, b):
    return (a + b - 1) // b

def to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    # blocks shape: (n_row, n_col, 128, 4)
    # flatten to use reshape
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()

def simulate_cuda_permutation(M, K):
    # Simulate the formula:
    # dst_idx = kb * 512 + (m % 32) * 16 + (m / 32) * 4 + k_in_block
    # where kb = k // 4, k_in_block = k % 4
    
    # We produce a flat array where `flat[dst_idx] = src[m, k]`
    # We want to check if this matches `to_blocked` which produces `rearranged`
    # to_blocked returns a flat array where `rearranged[i]` corresponds to some `src[m, k]`
    # If to_blocked[i] == src[m, k], then `i` corresponds to `dst_idx` in CUDA formula.
    
    src = torch.arange(M * K).reshape(M, K).float()
    blocked = to_blocked(src)
    
    # Verify mapping
    # Create an empty array of size M*K
    reconstructed = torch.zeros(M*K, dtype=torch.float)
    
    # CUDA Formula
    # M=128, K=16 (One tile)
    for m in range(M):
        for k in range(K):
            val = src[m, k]
            
            kb = k // 4
            k_in_block = k % 4
            
            dst_idx = kb * 512 + (m % 32) * 16 + (m // 32) * 4 + k_in_block
            
            reconstructed[dst_idx] = val
            
    # Compare
    if torch.all(blocked == reconstructed):
        print("MATCH!")
    else:
        print("MISMATCH!")
        print("Blocked:", blocked[:20])
        print("Reconst:", reconstructed[:20])
        
        # Find first mismatch
        diff = (blocked != reconstructed).nonzero(as_tuple=True)[0]
        if len(diff) > 0:
            i = diff[0].item()
            print(f"First mismatch at index {i}: Blocked={blocked[i]}, Reconst={reconstructed[i]}")

print("Testing 128x16...")
simulate_cuda_permutation(128, 16)
