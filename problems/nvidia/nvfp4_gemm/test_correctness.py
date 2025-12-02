#!/usr/bin/env python3
"""
Correctness validation test for NVFP4 GEMM kernel.
Runs custom kernel against reference implementation for all test cases.
"""
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reference import generate_input, check_implementation
from submission import custom_kernel
from utils import set_seed

# Test cases from task.yml
TEST_CASES = [
    {"m": 128, "n": 256, "k": 256, "l": 1, "seed": 1111},
]

def run_correctness_tests():
    """Run correctness validation for all test cases."""
    print("=" * 80)
    print("NVFP4 GEMM Correctness Validation")
    print("=" * 80)
    print(f"\nRunning {len(TEST_CASES)} test cases...")
    print(f"Tolerance: rtol=1e-03, atol=1e-03\n")
    
    passed = 0
    failed = 0
    
    for idx, test_spec in enumerate(TEST_CASES):
        m, n, k, l, seed = test_spec["m"], test_spec["n"], test_spec["k"], test_spec["l"], test_spec["seed"]
        
        print(f"\n[Test {idx+1}/{len(TEST_CASES)}] M={m}, N={n}, K={k}, L={l}")
        print("-" * 60)
        
        try:
            # Set seed for reproducibility
            set_seed(seed)
            
            # Generate input data
            print("  Generating input data...")
            data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
            
            # Clone data for custom kernel (to avoid modifying reference data)
            data_copy = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
            
            # Run custom kernel
            print("  Running custom kernel...")
            torch.cuda.synchronize()
            output = custom_kernel(data_copy)
            torch.cuda.synchronize()
            
            # Check against reference
            print("  Validating against reference...")
            success, message = check_implementation(data, output)
            
            if success:
                print(f"  ✓ PASSED")
                passed += 1
            else:
                print(f"  ✗ FAILED: {message}")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed:      {passed} ({100*passed//len(TEST_CASES)}%)")
    print(f"Failed:      {failed} ({100*failed//len(TEST_CASES)}%)")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_correctness_tests())
