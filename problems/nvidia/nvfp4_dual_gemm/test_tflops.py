#!/usr/bin/env python3
"""
TFLOPs benchmarking for NVFP4 dual GEMM kernel.
Reports throughput based on two GEMMs: 4 * M * N * K ops.
"""
import sys
import math
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reference import generate_input
from submission import custom_kernel
from utils import set_seed, clear_l2_cache

# Benchmark cases from task.yml
BENCHMARK_CASES = [
    {"m": 256, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"m": 512, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"m": 256, "n": 3072, "k": 4096, "l": 1, "seed": 1111},
    {"m": 512, "n": 3072, "k": 7168, "l": 1, "seed": 1111},
]

# Speed of light targets (in microseconds) at 1.5GHz clock
SPEED_OF_LIGHT_TARGETS = [4.708, 8.714, 2.125, 6.535]

NUM_WARMUP_RUNS = 5
NUM_BENCHMARK_RUNS = 50


def ops_dual_gemm(m: int, n: int, k: int) -> float:
    # Two GEMMs, each 2 * M * N * K ops (mul+add).
    return 4.0 * m * n * k


def tflops_from_time(ops: float, time_us: float) -> float:
    return ops / (time_us * 1e-6) / 1e12


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
        _ = custom_kernel(data)
    torch.cuda.synchronize()

    # Benchmark runs
    timings = []
    for _ in range(NUM_BENCHMARK_RUNS):
        clear_l2_cache()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = custom_kernel(data)
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
    """Run TFLOPs benchmarks for all cases."""
    print("=" * 80)
    print("NVFP4 Dual GEMM TFLOPs Benchmark")
    print("=" * 80)
    print("\nFormula: TFLOPs = (4 * M * N * K) / time_seconds / 1e12")
    print(f"  Warmup runs:    {NUM_WARMUP_RUNS}")
    print(f"  Benchmark runs: {NUM_BENCHMARK_RUNS}")
    print(f"\nRunning {len(BENCHMARK_CASES)} benchmark cases...\n")

    results = []

    for idx, test_spec in enumerate(BENCHMARK_CASES):
        m, n, k, l, seed = test_spec["m"], test_spec["n"], test_spec["k"], test_spec["l"], test_spec["seed"]
        target_us = SPEED_OF_LIGHT_TARGETS[idx]

        print(f"\n[Benchmark {idx+1}/{len(BENCHMARK_CASES)}] M={m}, N={n}, K={k}, L={l}")
        print("-" * 60)
        print(f"  Speed of light target: {target_us:.3f} us")

        try:
            mean_time, std_time, min_time, max_time = benchmark_kernel(m, n, k, l, seed)
            ops = ops_dual_gemm(m, n, k)
            tflops = tflops_from_time(ops, mean_time)
            target_tflops = tflops_from_time(ops, target_us)
            efficiency_pct = (tflops / target_tflops) * 100.0

            print("  Results:")
            print(f"    Mean:       {mean_time:.3f} us")
            print(f"    Std Dev:    {std_time:.3f} us")
            print(f"    Min:        {min_time:.3f} us")
            print(f"    Max:        {max_time:.3f} us")
            print(f"    TFLOPs:     {tflops:.2f}")
            print(f"    Efficiency:{efficiency_pct:6.1f}% of speed of light")

            results.append({
                "m": m, "n": n, "k": k, "l": l,
                "mean": mean_time,
                "std": std_time,
                "min": min_time,
                "max": max_time,
                "tflops": tflops,
                "target": target_us,
                "efficiency": efficiency_pct
            })
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append(None)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_results = [r for r in results if r is not None]

    if len(valid_results) == len(BENCHMARK_CASES):
        geom_mean_time = math.exp(sum(math.log(r["mean"]) for r in valid_results) / len(valid_results))
        geom_mean_target = math.exp(sum(math.log(r["target"]) for r in valid_results) / len(valid_results))
        overall_efficiency = (geom_mean_target / geom_mean_time) * 100

        print("\nGeometric Mean:")
        print(f"  Kernel time:    {geom_mean_time:.3f} us")
        print(f"  Target time:    {geom_mean_target:.3f} us")
        print(f"  Efficiency:     {overall_efficiency:.1f}% of speed of light")

        print("\nPer-benchmark TFLOPs:")
        for idx, r in enumerate(valid_results):
            print(f"  [{idx+1}] M={r['m']:4d} N={r['n']:4d} K={r['k']:5d}: {r['tflops']:.2f} TFLOPs")

        print(f"\n{'✓' if overall_efficiency >= 90 else '⚠'} Competition ranking metric (geometric mean): {geom_mean_time:.3f} us")
        return 0

    print(f"\n✗ {len(BENCHMARK_CASES) - len(valid_results)} benchmark(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(run_benchmarks())
