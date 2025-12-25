#!/usr/bin/env python3
"""
Performance benchmarking test for NVFP4 GEMM kernel.
Measures execution latency without correctness checks and calculates geometric mean.
"""
import sys
import time
import math
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reference import generate_input
from submission import custom_kernel
from dsl import custom_kernel as dsl_custom_kernel

from utils import set_seed, clear_l2_cache

# Benchmark cases from task.yml
BENCHMARK_CASES = [
    {"m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111},
    {"m": 128, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"m": 128, "n": 7168, "k": 2048, "l": 1, "seed": 1111},
]

# Speed of light targets (in microseconds) at 1.5GHz clock
SPEED_OF_LIGHT_TARGETS = [8.994, 2.354, 1.333]

NUM_WARMUP_RUNS = 5
NUM_BENCHMARK_RUNS = 50

def benchmark_kernel(m, n, k, l, seed):
    """
    Benchmark a single kernel configuration.
    
    Returns:
        Tuple of (mean_time_us, std_time_us, min_time_us, max_time_us)
    """
    set_seed(seed)
    
    # Generate input data
    data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
    
    # Warmup runs
    for _ in range(NUM_WARMUP_RUNS):
        _ = dsl_custom_kernel(data)
    torch.cuda.synchronize()
    
    # Benchmark runs
    timings = []
    for _ in range(NUM_BENCHMARK_RUNS):
        clear_l2_cache()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = dsl_custom_kernel(data)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Convert milliseconds to microseconds
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_us = elapsed_ms * 1000.0
        timings.append(elapsed_us)
    
    # Calculate statistics
    mean_time = sum(timings) / len(timings)
    variance = sum((t - mean_time) ** 2 for t in timings) / (len(timings) - 1)
    std_time = math.sqrt(variance)
    min_time = min(timings)
    max_time = max(timings)
    
    return mean_time, std_time, min_time, max_time

def run_benchmarks():
    """Run performance benchmarks for all cases."""
    print("=" * 80)
    print("NVFP4 GEMM Performance Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Warmup runs:    {NUM_WARMUP_RUNS}")
    print(f"  Benchmark runs: {NUM_BENCHMARK_RUNS}")
    print(f"\nRunning {len(BENCHMARK_CASES)} benchmark cases...\n")
    
    results = []
    
    for idx, test_spec in enumerate(BENCHMARK_CASES):
        m, n, k, l, seed = test_spec["m"], test_spec["n"], test_spec["k"], test_spec["l"], test_spec["seed"]
        target_us = SPEED_OF_LIGHT_TARGETS[idx]
        
        print(f"\n[Benchmark {idx+1}/{len(BENCHMARK_CASES)}] M={m}, N={n}, K={k}, L={l}")
        print("-" * 60)
        print(f"  Speed of light target: {target_us:.3f} μs")
        
        try:
            mean_time, std_time, min_time, max_time = benchmark_kernel(m, n, k, l, seed)
            
            # Calculate performance metrics
            speedup = target_us / mean_time
            efficiency_pct = speedup * 100
            
            print(f"  Results:")
            print(f"    Mean:       {mean_time:.3f} μs")
            print(f"    Std Dev:    {std_time:.3f} μs")
            print(f"    Min:        {min_time:.3f} μs")
            print(f"    Max:        {max_time:.3f} μs")
            print(f"    Efficiency: {efficiency_pct:.1f}% of speed of light")
            
            results.append({
                "m": m, "n": n, "k": k, "l": l,
                "mean": mean_time,
                "std": std_time,
                "min": min_time,
                "max": max_time,
                "target": target_us,
                "efficiency": efficiency_pct
            })
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append(None)
    
    # Calculate geometric mean
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == len(BENCHMARK_CASES):
        # Geometric mean of execution times
        geom_mean_time = math.exp(sum(math.log(r["mean"]) for r in valid_results) / len(valid_results))
        
        # Geometric mean of speed of light targets
        geom_mean_target = math.exp(sum(math.log(r["target"]) for r in valid_results) / len(valid_results))
        
        # Overall efficiency
        overall_efficiency = (geom_mean_target / geom_mean_time) * 100
        
        print(f"\nGeometric Mean:")
        print(f"  Kernel time:    {geom_mean_time:.3f} μs")
        print(f"  Target time:    {geom_mean_target:.3f} μs")
        print(f"  Efficiency:     {overall_efficiency:.1f}% of speed of light")
        
        print(f"\nPer-benchmark efficiency:")
        for idx, r in enumerate(valid_results):
            print(f"  [{idx+1}] M={r['m']:4d} N={r['n']:4d} K={r['k']:5d}: {r['efficiency']:5.1f}% ({r['mean']:.3f} μs / {r['target']:.3f} μs)")
        
        print(f"\n{'✓' if overall_efficiency >= 90 else '⚠'} Competition ranking metric (geometric mean): {geom_mean_time:.3f} μs")
        
        return 0
    else:
        print(f"\n✗ {len(BENCHMARK_CASES) - len(valid_results)} benchmark(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_benchmarks())
