#!/usr/bin/env python3
"""
Benchmark script for theta.py to measure performance of 12-bit counter extraction
and theta calculation.

This benchmark helps identify whether Cython optimization would provide meaningful
benefit by measuring the actual time spent in hot paths:

1. 12-bit counter extraction from bins 18, 19, 20 (bit manipulation)
2. Theta interpolation (run-length distribution)
3. Sorting for index lookups
4. Overall theta calculation

Usage:
    python scripts/benchmark_theta.py [--n-frames N] [--warmup W] [--runs R]

Results are compared against typical frame processing time to determine if
theta calculation is a bottleneck worth optimizing with Cython.
"""

import argparse
import time
from typing import Callable

import numpy as np

# Add src to path for development mode
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wamos_tpw.frame import Frame, FrameMetadata  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402


def create_synthetic_frame(
    n_bearings: int = 360, n_distances: int = 512, seed: int | None = None
) -> Frame:
    """
    Create a synthetic frame with realistic 12-bit counter values.

    The 12-bit counter is encoded in the top nibbles of bins 18, 19, 20:
    - Bin 18 bits 12-15 -> counter bits 8-11
    - Bin 19 bits 12-15 -> counter bits 4-7
    - Bin 20 bits 12-15 -> counter bits 0-3

    Args:
        n_bearings: Number of bearing rows (radials)
        n_distances: Number of distance columns
        seed: Random seed for reproducibility

    Returns:
        Frame with synthetic data containing realistic theta encoding
    """
    if seed is not None:
        np.random.seed(seed)

    # Create base intensity data (bottom 12 bits, values 0-4095)
    raw_data = np.random.randint(0, 4096, (n_bearings, n_distances), dtype=np.uint16)

    # Simulate realistic theta counter progression
    # Radar typically has ~1 degree per radial, with runs of 3-5 radials per degree
    radials_per_degree = n_bearings / 360
    degrees = np.floor(np.arange(n_bearings) / radials_per_degree).astype(np.uint16)

    # Encode degrees into the 12-bit counter spread across bins 18, 19, 20
    # Bin 18: bits 8-11 of counter -> shifted to bits 12-15 of data
    # Bin 19: bits 4-7 of counter -> shifted to bits 12-15 of data
    # Bin 20: bits 0-3 of counter -> shifted to bits 12-15 of data
    counter_bits_8_11 = (degrees >> 8) & 0x0F  # Extract bits 8-11
    counter_bits_4_7 = (degrees >> 4) & 0x0F  # Extract bits 4-7
    counter_bits_0_3 = degrees & 0x0F  # Extract bits 0-3

    # Place in top nibble of respective bins
    raw_data[:, 18] = (raw_data[:, 18] & 0x0FFF) | (counter_bits_8_11 << 12)
    raw_data[:, 19] = (raw_data[:, 19] & 0x0FFF) | (counter_bits_4_7 << 12)
    raw_data[:, 20] = (raw_data[:, 20] & 0x0FFF) | (counter_bits_0_3 << 12)

    metadata = FrameMetadata(
        timestamp=np.datetime64("2024-12-15T10:30:00"),
        filename="synthetic_test.pol",
        latitude=18.57,
        longitude=142.96,
        samples_in_range=n_distances,
        sampling_frequency=20.0,
        sample_delay_range=150.0,
        radar_height=25.0,
    )

    return Frame(raw_data, metadata, config=None, validate=False)


def benchmark_function(
    func: Callable, args: tuple, n_warmup: int = 3, n_runs: int = 20
) -> tuple[float, float, float, float, object]:
    """
    Benchmark a function with warmup and multiple runs.

    Args:
        func: Function to benchmark
        args: Arguments to pass to function
        n_warmup: Number of warmup iterations
        n_runs: Number of timed iterations

    Returns:
        Tuple of (mean_time, std_time, min_time, max_time, result)
    """
    # Warmup
    for _ in range(n_warmup):
        result = func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return times.mean(), times.std(), times.min(), times.max(), result


def benchmark_extract_degrees_isolated(data: np.ndarray, n_runs: int = 1000) -> dict:
    """
    Benchmark just the 12-bit counter extraction (the hot path for Cython).

    This isolates the bit manipulation operations that would benefit most
    from Cython optimization.

    Args:
        data: Raw frame data array (n_bearings, n_distances)
        n_runs: Number of iterations

    Returns:
        Dict with timing statistics
    """
    COUNTER_BINS = (18, 19, 20)
    NIBBLE_MASK = np.uint16(0xF000)

    # Warmup
    for _ in range(10):
        nibbles = data[:, COUNTER_BINS] & NIBBLE_MASK
        nibbles = np.right_shift(nibbles, [4, 8, 12]).astype(np.uint16)
        _ = nibbles[:, 0] | nibbles[:, 1] | nibbles[:, 2]

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        nibbles = data[:, COUNTER_BINS] & NIBBLE_MASK
        nibbles = np.right_shift(nibbles, [4, 8, 12]).astype(np.uint16)
        _ = nibbles[:, 0] | nibbles[:, 1] | nibbles[:, 2]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "std_us": times.std() * 1e6,
        "min_us": times.min() * 1e6,
        "max_us": times.max() * 1e6,
        "n_runs": n_runs,
    }


def benchmark_interpolate_isolated(degrees: np.ndarray, n_runs: int = 1000) -> dict:
    """
    Benchmark just the theta interpolation step.

    Args:
        degrees: Array of whole degree values
        n_runs: Number of iterations

    Returns:
        Dict with timing statistics
    """

    def interpolate_theta(degrees: np.ndarray) -> np.ndarray:
        """Reproduce the interpolation logic from Theta class."""
        unique_degrees, degree_indices, run_counts = np.unique(
            degrees, return_inverse=True, return_counts=True
        )

        mean_run_length = np.mean(run_counts)
        run_counts = run_counts.astype(np.float32)
        first_run_count = run_counts[0]

        run_counts[0] = max(run_counts[0], mean_run_length)
        run_counts[-1] = max(run_counts[-1], mean_run_length)

        degree_diffs = np.diff(unique_degrees, append=unique_degrees[-1] + 1)
        step_sizes = degree_diffs / run_counts
        delta = step_sizes[degree_indices]

        delta[0] += (run_counts[0] - first_run_count) * delta[0]

        return ((unique_degrees[0] + np.cumsum(delta) - (delta / 2)) % 360).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = interpolate_theta(degrees)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = interpolate_theta(degrees)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "std_us": times.std() * 1e6,
        "min_us": times.min() * 1e6,
        "max_us": times.max() * 1e6,
        "n_runs": n_runs,
    }


def benchmark_sort_isolated(theta: np.ndarray, n_runs: int = 1000) -> dict:
    """
    Benchmark just the sorting step (for index lookups).

    Args:
        theta: Array of theta values
        n_runs: Number of iterations

    Returns:
        Dict with timing statistics
    """
    # Warmup
    for _ in range(10):
        _ = np.argsort(theta)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sorted_indices = np.argsort(theta)
        _ = theta[sorted_indices]
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "std_us": times.std() * 1e6,
        "min_us": times.min() * 1e6,
        "max_us": times.max() * 1e6,
        "n_runs": n_runs,
    }


def benchmark_index_lookup(theta_obj: Theta, n_lookups: int = 360, n_runs: int = 500) -> dict:
    """
    Benchmark the index lookup operation.

    Args:
        theta_obj: Theta object with calculated angles
        n_lookups: Number of angles to look up per call
        n_runs: Number of iterations

    Returns:
        Dict with timing statistics
    """
    # Create random lookup angles
    lookup_angles = np.random.rand(n_lookups) * 360

    # Warmup
    for _ in range(10):
        _ = theta_obj.index(lookup_angles)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = theta_obj.index(lookup_angles)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_us": times.mean() * 1e6,
        "std_us": times.std() * 1e6,
        "min_us": times.min() * 1e6,
        "max_us": times.max() * 1e6,
        "n_runs": n_runs,
        "n_lookups": n_lookups,
    }


def print_timing_result(name: str, stats: dict, indent: int = 2) -> None:
    """Pretty print timing results."""
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  Mean: {stats['mean_us']:8.2f} us +/- {stats['std_us']:.2f} us")
    print(f"{prefix}  Range: [{stats['min_us']:.2f}, {stats['max_us']:.2f}] us")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark theta.py performance for Cython optimization analysis"
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=100,
        help="Number of frames for full Theta() benchmark (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1000,
        help="Number of runs for isolated benchmarks (default: 1000)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=str,
        default=["360x512", "360x752", "720x1024"],
        help="Frame sizes to test as NxM (default: 360x512 360x752 720x1024)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Theta Calculation Benchmark")
    print("=" * 70)
    print()
    print("Purpose: Measure time spent in theta calculation to determine")
    print("         if Cython optimization would provide meaningful benefit.")
    print()

    for size_str in args.sizes:
        n_bearings, n_distances = map(int, size_str.split("x"))

        print("-" * 70)
        print(f"Frame size: {n_bearings} bearings x {n_distances} distances")
        print(f"  Total elements: {n_bearings * n_distances:,}")
        print("-" * 70)

        # Create synthetic frame
        frame = create_synthetic_frame(n_bearings, n_distances, seed=42)

        # =====================================================================
        # 1. Full Theta calculation benchmark
        # =====================================================================
        print("\n1. Full Theta() calculation (including all steps):")

        mean_t, std_t, min_t, max_t, theta_obj = benchmark_function(
            lambda f: Theta(f), (frame,), n_warmup=args.warmup, n_runs=args.n_frames
        )
        print(f"   Mean: {mean_t * 1000:8.3f} ms +/- {std_t * 1000:.3f} ms")
        print(f"   Range: [{min_t * 1000:.3f}, {max_t * 1000:.3f}] ms")

        # Get internal timing breakdown from last run
        print("\n   Internal timing breakdown (from Theta.timing):")
        for step, t in theta_obj.timing.items():
            pct = (t / mean_t) * 100 if mean_t > 0 else 0
            print(f"     {step:20s}: {t * 1000:8.3f} ms ({pct:5.1f}%)")

        # =====================================================================
        # 2. Isolated bit manipulation benchmark (hot path for Cython)
        # =====================================================================
        print("\n2. Isolated bit manipulation (12-bit counter extraction):")
        print("   This is the primary candidate for Cython optimization.")

        extract_stats = benchmark_extract_degrees_isolated(frame.raw, n_runs=args.runs)
        print_timing_result("extract_degrees", extract_stats)

        # Calculate operations per second
        ops_per_sec = 1e6 / extract_stats["mean_us"]
        print(f"   Throughput: {ops_per_sec:,.0f} frames/sec")

        # =====================================================================
        # 3. Isolated interpolation benchmark
        # =====================================================================
        print("\n3. Isolated theta interpolation:")

        # Extract degrees for interpolation benchmark
        COUNTER_BINS = (18, 19, 20)
        NIBBLE_MASK = np.uint16(0xF000)
        nibbles = frame.raw[:, COUNTER_BINS] & NIBBLE_MASK
        nibbles = np.right_shift(nibbles, [4, 8, 12]).astype(np.uint16)
        degrees = nibbles[:, 0] | nibbles[:, 1] | nibbles[:, 2]

        interp_stats = benchmark_interpolate_isolated(degrees, n_runs=args.runs)
        print_timing_result("interpolate_theta", interp_stats)

        # =====================================================================
        # 4. Isolated sorting benchmark
        # =====================================================================
        print("\n4. Isolated sorting (for index lookups):")

        sort_stats = benchmark_sort_isolated(theta_obj.theta, n_runs=args.runs)
        print_timing_result("sort", sort_stats)

        # =====================================================================
        # 5. Index lookup benchmark
        # =====================================================================
        print("\n5. Index lookup (typical use case with 360 angles):")

        lookup_stats = benchmark_index_lookup(theta_obj, n_lookups=360, n_runs=args.runs // 2)
        print_timing_result("index", lookup_stats)
        print(f"     Lookups per call: {lookup_stats['n_lookups']}")

        # =====================================================================
        # Summary and Cython recommendation
        # =====================================================================
        print("\n" + "=" * 50)
        print("Summary for Cython optimization analysis:")
        print("=" * 50)

        total_time_us = mean_t * 1e6
        extract_time_us = extract_stats["mean_us"]
        interp_time_us = interp_stats["mean_us"]
        sort_time_us = sort_stats["mean_us"]

        print(f"\n   Total Theta() time:        {total_time_us:8.2f} us")
        print(
            f"   extract_degrees (bit ops): {extract_time_us:8.2f} us ({extract_time_us / total_time_us * 100:.1f}%)"
        )
        print(
            f"   interpolate_theta:         {interp_time_us:8.2f} us ({interp_time_us / total_time_us * 100:.1f}%)"
        )
        print(
            f"   sort:                      {sort_time_us:8.2f} us ({sort_time_us / total_time_us * 100:.1f}%)"
        )

        # Recommendation based on timings
        print("\n   Cython optimization recommendation:")

        if extract_time_us < 10:
            print("   - Bit manipulation is already very fast (<10 us)")
            print("     Cython would provide minimal benefit here.")
        elif extract_time_us < 50:
            print("   - Bit manipulation takes 10-50 us")
            print("     Cython could provide modest speedup (2-5x typical)")
            print("     but overall impact may be limited.")
        else:
            print("   - Bit manipulation takes >50 us")
            print("     Cython optimization could provide meaningful benefit.")

        if sort_time_us > extract_time_us:
            print(
                f"   - Note: Sorting ({sort_time_us:.0f} us) dominates extraction ({extract_time_us:.0f} us)"
            )
            print("     NumPy sort is already optimized; Cython won't help here.")

        print()

    # =========================================================================
    # Final comparison with typical pipeline operations
    # =========================================================================
    print("=" * 70)
    print("Context: Typical pipeline operation times for comparison")
    print("=" * 70)
    print()
    print("Typical times (rough estimates for 360x512 frame):")
    print("  - File I/O (decompressed .pol): ~1-5 ms")
    print("  - Intensity extraction:         ~50-100 us")
    print("  - Deramp:                       ~200-500 us")
    print("  - Destreak:                     ~500-2000 us")
    print("  - Grid projection (NumPy):     ~5-20 ms")
    print("  - Grid projection (Numba):     ~0.5-2 ms")
    print()
    print("If theta calculation is <1% of total pipeline time,")
    print("Cython optimization is unlikely to provide meaningful benefit.")
    print()


if __name__ == "__main__":
    main()
