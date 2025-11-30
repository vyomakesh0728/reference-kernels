#!/usr/bin/env python3
"""Quick test to isolate the hang issue."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

import torch
from submission import custom_kernel
from reference import generate_input
from utils import set_seed

# Test the problematic configuration
test_configs = [
    (4096, 7168, 1, "M=4096, K=7168, L=1 (rank-2 CTA)"),
    (4096, 7168, 2, "M=4096, K=7168, L=2 (rank-3 cluster)"),
    (4096, 7168, 4, "M=4096, K=7168, L=4 (rank-3 cluster)"),
    (4096, 7168, 8, "M=4096, K=7168, L=8 (rank-3 cluster - HANGS)"),
]

for m, k, l, desc in test_configs:
    print(f"\n{'='*60}")
    print(f"Testing: {desc}")
    print(f"{'='*60}")

    set_seed(1111)
    data = generate_input(m=m, k=k, l=l, seed=1111)

    try:
        print(f"Launching kernel...")
        output = custom_kernel(data)
        torch.cuda.synchronize()
        print(f"✅ SUCCESS! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        break
