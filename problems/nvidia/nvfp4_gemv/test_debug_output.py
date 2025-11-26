#!/usr/bin/env python3
"""Debug output values."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

import torch
from submission import custom_kernel
from reference import generate_input, ref_kernel
from utils import set_seed

# Set seed
set_seed(1111)

# Generate test data for medium size (2 blocks: M=256)
print("Generating test data: M=256, K=256, L=1")
data = generate_input(m=256, k=256, l=1, seed=1111)

print("Running custom_kernel...")
output = custom_kernel(data)
torch.cuda.synchronize()
print("✓ Kernel executed successfully!")

print("\nRunning reference implementation...")
expected = ref_kernel(data)

print("\nActual output (first 10 rows):")
print(output[:10, 0, 0])

print("\nExpected output (first 10 rows):")
print(expected[:10, 0, 0])

print("\nActual output (rows 128-138, i.e., second block):")
print(output[128:138, 0, 0])

print("\nExpected output (rows 128-138):")
print(expected[128:138, 0, 0])
