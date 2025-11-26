#!/usr/bin/env python3
"""Direct test without multiprocessing to get better error messages."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

from submission import custom_kernel
from reference import generate_input, check_implementation
from utils import set_seed

# Set seed
set_seed(1111)

# Generate test data for the first benchmark
print("Generating test data: M=7168, K=16384, L=1")
data = generate_input(m=7168, k=16384, l=1, seed=1111)

print("Running custom_kernel...")
try:
    output = custom_kernel(data)
    print("✓ Kernel executed successfully!")

    # Check correctness
    good, message = check_implementation(data, output)
    if good:
        print("✓ Correctness check passed!")
    else:
        print(f"✗ Correctness check failed: {message}")
except Exception as e:
    print(f"✗ Kernel failed with error: {e}")
    import traceback
    traceback.print_exc()
