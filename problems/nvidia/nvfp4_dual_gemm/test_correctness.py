#!/usr/bin/env python3
"""
Correctness validation test for NVFP4 dual GEMM kernel.
Runs custom kernel against reference implementation for all test cases.
"""
import argparse
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
    {"m": 1536, "n": 512, "k": 7168, "l": 1, "seed": 1111},
    {"m": 256, "n": 512, "k": 256, "l": 1, "seed": 1111},
    {"m": 1536, "n": 512, "k": 7168, "l": 1, "seed": 1111},
    {"m": 3072, "n": 1024, "k": 1536, "l": 1, "seed": 1111},
    {"m": 7168, "n": 1024, "k": 256, "l": 1, "seed": 1111},
    {"m": 7168, "n": 2304, "k": 2048, "l": 1, "seed": 1111},
    {"m": 4608, "n": 384, "k": 7168, "l": 1, "seed": 1111},
    {"m": 7168, "n": 384, "k": 2304, "l": 1, "seed": 1111},
    {"m": 512, "n": 768, "k": 7168, "l": 1, "seed": 1111},
    {"m": 4096, "n": 768, "k": 512, "l": 1, "seed": 1111},
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="NVFP4 dual GEMM correctness validation."
    )
    parser.add_argument(
        "--only",
        type=int,
        default=0,
        help="Run only the first N test cases (0 means all).",
    )
    return parser.parse_args()


def run_correctness_tests(only_count: int):
    """Run correctness validation for all test cases."""
    print("=" * 80)
    print("NVFP4 Dual GEMM Correctness Validation")
    print("=" * 80)
    if only_count < 0:
        raise ValueError("--only must be >= 0")
    if only_count == 0:
        selected_cases = TEST_CASES
    else:
        selected_cases = TEST_CASES[:only_count]
    print(f"\nRunning {len(selected_cases)} test cases...")
    print(f"Tolerance: rtol=1e-03, atol=1e-03\n")
    
    passed = 0
    failed = 0
    
    for idx, test_spec in enumerate(selected_cases):
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
    total = len(selected_cases)
    print(f"Total tests: {total}")
    print(f"Passed:      {passed} ({100*passed//total}%)")
    print(f"Failed:      {failed} ({100*failed//total}%)")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_correctness_tests(args.only))
