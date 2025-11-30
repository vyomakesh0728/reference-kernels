#!/usr/bin/env python3
"""Test kernel execution only, no reference check."""
import sys
sys.path.insert(0, '/root/reference-kernels/problems/nvidia/nvfp4_gemv')

import math
import time
import torch
from submission import custom_kernel
from reference import generate_input
from utils import set_seed, clear_l2_cache


def calculate_stats(durations):
    """Calculate statistical data from a list of durations in nanoseconds."""
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum((x - avg) ** 2 for x in durations)
    std = math.sqrt(variance / (runs - 1)) if runs > 1 else 0
    err = std / math.sqrt(runs) if runs > 0 else 0

    return {
        'runs': runs,
        'mean': avg,
        'std': std,
        'err': err,
        'best': float(best),
        'worst': float(worst)
    }


# Test configurations (matching task.yml benchmarks)
# Speed of Light targets from task.yml:
#   M=7168, K=16384, L=1: 8.622 μs
#   M=4096, K=7168,  L=8: 17.275 μs
#   M=7168, K=2048,  L=4: 4.317 μs
# Ranking: geometric mean of these 3 benchmarks
test_cases = [
    # (M, K, L, description, target_us)
    (7168, 16384, 1, "rank-2: CTA + SWIZZLE_NONE + box_k=16", 8.622),
    (4096, 7168,  8, "rank-3: Cluster + SWIZZLE_128B + box_k=K_scales_padded", 17.275),
    (7168, 2048,  4, "rank-3: Cluster + SWIZZLE_128B + box_k=K_scales_padded", 4.317),
]

print("Testing kernel execution for all configurations...")
print("="*60)

# Store results for geometric mean calculation
all_results = []

# Warmup with first test case (matching eval.py line 306)
print("\nWarmup run...")
set_seed(1111)
warmup_data = generate_input(m=test_cases[0][0], k=test_cases[0][1], l=test_cases[0][2], seed=1111)
try:
    _ = custom_kernel(warmup_data)
    torch.cuda.synchronize()
    print("✓ Warmup completed")
except Exception as e:
    print(f"✗ Warmup FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

for m, k, l, desc, target_us in test_cases:
    print(f"\nTest: M={m}, K={k}, L={l}")
    print(f"Config: {desc}")
    print(f"Speed of Light Target: {target_us:.3f} μs")

    # Set seed for reproducibility
    set_seed(1111)

    # Generate test data once
    data = generate_input(m=m, k=k, l=l, seed=1111)

    try:
        # Multiple timed runs with L2 cache clearing (matching eval.py)
        max_repeats = 200
        max_time_ns = 10e9  # 10 seconds
        durations = []

        bm_start_time = time.perf_counter_ns()

        for i in range(max_repeats):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Clear L2 cache before each run (matching eval.py line 241)
            clear_l2_cache()

            start_event.record()
            output = custom_kernel(data)
            end_event.record()
            torch.cuda.synchronize()

            # Convert ms to nanoseconds (matching eval.py line 247)
            duration_ns = start_event.elapsed_time(end_event) * 1e6
            durations.append(duration_ns)

            # Smart stopping logic (matching eval.py lines 258-269)
            total_bm_duration = time.perf_counter_ns() - bm_start_time
            if i > 1 and total_bm_duration > 1e8:  # at least 2 runs, 100ms total
                stats = calculate_stats(durations)
                # Stop if relative error < 0.1% or time limit exceeded
                if (stats['err'] / stats['mean'] < 0.001 or
                    stats['mean'] * stats['runs'] > max_time_ns or
                    total_bm_duration > 120e9):  # 2 minute wallclock limit
                    break

        # Calculate final statistics
        stats = calculate_stats(durations)
        mean_us = stats['mean'] / 1e3
        best_us = stats['best'] / 1e3

        # Store result for geometric mean
        all_results.append({
            'm': m, 'k': k, 'l': l,
            'mean_us': mean_us,
            'best_us': best_us,
            'target_us': target_us
        })

        print(f"✓ Kernel executed successfully! Output shape: {output.shape}")
        print(f"  Runs: {stats['runs']}")
        print(f"  Mean: {stats['mean']:.2f} ns ({mean_us:.2f} μs, {stats['mean']/1e6:.3f} ms)")
        print(f"  Std:  {stats['std']:.2f} ns ({stats['std']/1e3:.2f} μs)")
        print(f"  Err:  {stats['err']:.2f} ns ({stats['err']/1e3:.2f} μs)")
        print(f"  Best: {stats['best']:.2f} ns ({best_us:.2f} μs, {stats['best']/1e6:.3f} ms)")
        print(f"  Worst: {stats['worst']:.2f} ns ({stats['worst']/1e3:.2f} μs, {stats['worst']/1e6:.3f} ms)")
        print(f"  Relative error: {(stats['err']/stats['mean']*100):.3f}%")
        print(f"  📊 vs Speed of Light: {mean_us:.2f} μs / {target_us:.3f} μs = {mean_us/target_us:.2f}x slower")
        print(f"     Best: {best_us:.2f} μs / {target_us:.3f} μs = {best_us/target_us:.2f}x slower")

    except Exception as e:
        print(f"✗ Kernel FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        break  # Stop on first failure

print("\n" + "="*60)
print("All kernel executions completed!")
print("="*60)

# Calculate geometric mean (matching task.yml ranking_by: "geom")
if all_results:
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY (matching task.yml)")
    print("="*60)

    geom_mean_times = []
    geom_target_times = []

    for result in all_results:
        print(f"\nM={result['m']}, K={result['k']}, L={result['l']}:")
        print(f"  Mean:   {result['mean_us']:.3f} μs")
        print(f"  Best:   {result['best_us']:.3f} μs")
        print(f"  Target: {result['target_us']:.3f} μs")
        print(f"  Ratio:  {result['mean_us']/result['target_us']:.2f}x slower (mean)")
        geom_mean_times.append(result['mean_us'])
        geom_target_times.append(result['target_us'])

    # Calculate geometric mean
    import math
    geom_mean_us = math.exp(sum(math.log(t) for t in geom_mean_times) / len(geom_mean_times))
    geom_target_us = math.exp(sum(math.log(t) for t in geom_target_times) / len(geom_target_times))

    print("\n" + "="*60)
    print(f"GEOMETRIC MEAN (ranking metric):")
    print(f"  Current: {geom_mean_us:.3f} μs")
    print(f"  Target:  {geom_target_us:.3f} μs (Speed of Light)")
    print(f"  Ratio:   {geom_mean_us/geom_target_us:.2f}x slower")
    print("="*60)
