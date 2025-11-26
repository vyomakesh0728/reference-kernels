#!/usr/bin/env python3
"""Test small case with correctness check."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

import torch
from submission import custom_kernel
from reference import generate_input, check_implementation
from utils import set_seed

# Set seed
set_seed(1111)

# Generate test data for small case
print("Generating test data: M=128, K=256, L=1")
data = generate_input(m=128, k=256, l=1, seed=1111)

print("Running custom_kernel...")
try:
    output = custom_kernel(data)
    torch.cuda.synchronize()
    print("✓ Kernel executed successfully!")

    # Check correctness
    good, message = check_implementation(data, output)
    if good:
        print("✓ Correctness check PASSED!")
    else:
        print(f"✗ Correctness check failed: {message}")
except Exception as e:
    print(f"✗ Kernel failed with error: {e}")
    import traceback
    traceback.print_exc()
