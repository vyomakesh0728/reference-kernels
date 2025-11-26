#!/usr/bin/env python3
"""Test with detailed CUDA error reporting."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

import torch
from submission import custom_kernel
from reference import generate_input
from utils import set_seed

# Enable CUDA error checking
torch.cuda.init()
torch.backends.cudnn.benchmark = False

# Set seed
set_seed(1111)

# Generate test data for the smallest test case first
print("Generating test data: M=128, K=256, L=1")
data = generate_input(m=128, k=256, l=1, seed=1111)

print("Running custom_kernel...")
try:
    output = custom_kernel(data)
    torch.cuda.synchronize()
    print("✓ Kernel executed successfully!")
except RuntimeError as e:
    print(f"✗ Kernel failed: {e}")
    # Get last CUDA error
    try:
        torch.cuda.synchronize()
    except RuntimeError as e2:
        print(f"Additional CUDA error info: {e2}")
    import traceback
    traceback.print_exc()
