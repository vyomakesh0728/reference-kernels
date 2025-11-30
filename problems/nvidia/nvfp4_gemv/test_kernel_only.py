#!/usr/bin/env python3
"""Test kernel execution only, no reference check."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

import torch
from submission import custom_kernel
from reference import generate_input
from utils import set_seed

# Test configurations
test_cases = [
    # (M, K, L, description)
    (7168, 16384, 1, "rank-2: CTA + SWIZZLE_NONE + box_k=16"),
    (7168, 16384, 4, "rank-3: Cluster + SWIZZLE_128B + box_k=K_scales_padded"),
    (7168, 16384, 8, "rank-3: Cluster + SWIZZLE_128B + box_k=K_scales_padded"),
]

print("Testing kernel execution for all configurations...")
print("="*60)

for m, k, l, desc in test_cases:
    print(f"\nTest: M={m}, K={k}, L={l}")
    print(f"Config: {desc}")

    # Set seed for reproducibility
    set_seed(1111)

    # Generate test data
    data = generate_input(m=m, k=k, l=l, seed=1111)

    try:
        # Warmup run
        _ = custom_kernel(data)
        torch.cuda.synchronize()

        # Timed run
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_us = elapsed_time_ms * 1000

        print(f"✓ Kernel executed successfully! Output shape: {output.shape}")
        print(f"  Execution time: {elapsed_time_us:.2f} μs ({elapsed_time_ms:.3f} ms)")
    except Exception as e:
        print(f"✗ Kernel FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        break  # Stop on first failure

print("\n" + "="*60)
print("All kernel executions completed!")
print("="*60)
