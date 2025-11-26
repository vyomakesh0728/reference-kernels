#!/usr/bin/env python3
"""Pre-compile the CUDA module before running benchmarks."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

print("Pre-compiling CUDA module...")
from submission import get_module
mod = get_module()
print("✓ Module compiled successfully!")
print(f"Module: {mod}")
