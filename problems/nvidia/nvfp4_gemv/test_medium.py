#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

from submission import custom_kernel
from reference import generate_input, check_implementation
from utils import set_seed

set_seed(1111)

# Test with M=256 (2 blocks), K=512 (2 K-tiles)
print("Testing: M=256, K=512, L=1")
data = generate_input(m=256, k=512, l=1, seed=1111)

print("Running custom_kernel...")
output = custom_kernel(data)
print("✓ Kernel executed successfully!")

good, message = check_implementation(data, output)
if good:
    print("✓ Correctness check PASSED!")
else:
    print(f"✗ Correctness check failed: {message}")
