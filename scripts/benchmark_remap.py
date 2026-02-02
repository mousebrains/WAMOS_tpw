#!/usr/bin/env python3
"""Benchmark remap_to_common_grid implementations."""

import time
import numpy as np

# Original implementation (before optimization)
def remap_original(
    intensity: np.ndarray,
    count: np.ndarray | None,
    src_x_edges: np.ndarray,
    src_y_edges: np.ndarray,
    dst_x_edges: np.ndarray,
    dst_y_edges: np.ndarray,
    dst_n_x: int,
    dst_n_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Original implementation with meshgrid."""
    src_x_centers = (src_x_edges[:-1] + src_x_edges[1:]) / 2
    src_y_centers = (src_y_edges[:-1] + src_y_edges[1:]) / 2

    dst_dx = dst_x_edges[1] - dst_x_edges[0]
    dst_dy = dst_y_edges[1] - dst_y_edges[0]

    # meshgrid creates two full-size arrays
    src_xx, src_yy = np.meshgrid(src_x_centers, src_y_centers, indexing="xy")

    dst_ix = ((src_xx - dst_x_edges[0]) / dst_dx).astype(np.int32)
    dst_iy = ((src_yy - dst_y_edges[0]) / dst_dy).astype(np.int32)

    valid = (
        (dst_ix >= 0)
        & (dst_ix < dst_n_x)
        & (dst_iy >= 0)
        & (dst_iy < dst_n_y)
        & ~np.isnan(intensity)
    )

    if not np.any(valid):
        return (
            np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
            np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
        )

    valid_ix = dst_ix[valid]
    valid_iy = dst_iy[valid]
    valid_intensity = intensity[valid]

    if count is not None:
        valid_count = count[valid]
    else:
        valid_count = np.ones(np.sum(valid), dtype=np.int32)

    linear_idx = valid_iy * dst_n_x + valid_ix
    grid_size = dst_n_x * dst_n_y

    dst_sum = np.bincount(
        linear_idx, weights=valid_intensity * valid_count, minlength=grid_size
    ).reshape((dst_n_y, dst_n_x))

    dst_count = np.bincount(linear_idx, weights=valid_count, minlength=grid_size).reshape(
        (dst_n_y, dst_n_x)
    )

    return dst_sum.astype(np.float64), dst_count.astype(np.int32)


# Import optimized version
from wamos_tpw.grid import remap_to_common_grid as remap_optimized


# Try to import numba for JIT version
try:
    import numba
    HAS_NUMBA = True

    @numba.jit(nopython=True, parallel=True)
    def _remap_numba_core(
        intensity: np.ndarray,
        count: np.ndarray,
        has_count: bool,
        dst_ix_1d: np.ndarray,
        dst_iy_1d: np.ndarray,
        dst_n_x: int,
        dst_n_y: int,
        y_start: int,
        x_start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated core remap loop."""
        grid_size = dst_n_x * dst_n_y
        dst_sum = np.zeros(grid_size, dtype=np.float64)
        dst_count_out = np.zeros(grid_size, dtype=np.float64)

        sub_n_y, sub_n_x = intensity.shape

        for iy in numba.prange(sub_n_y):
            dst_iy = dst_iy_1d[y_start + iy]
            for ix in range(sub_n_x):
                val = intensity[iy, ix]
                if np.isnan(val):
                    continue
                dst_ix = dst_ix_1d[x_start + ix]
                linear_idx = dst_iy * dst_n_x + dst_ix

                if has_count:
                    c = count[iy, ix]
                    dst_sum[linear_idx] += val * c
                    dst_count_out[linear_idx] += c
                else:
                    dst_sum[linear_idx] += val
                    dst_count_out[linear_idx] += 1

        return dst_sum.reshape((dst_n_y, dst_n_x)), dst_count_out.reshape((dst_n_y, dst_n_x))

    def remap_numba(
        intensity: np.ndarray,
        count: np.ndarray | None,
        src_x_edges: np.ndarray,
        src_y_edges: np.ndarray,
        dst_x_edges: np.ndarray,
        dst_y_edges: np.ndarray,
        dst_n_x: int,
        dst_n_y: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated remap implementation."""
        # Early exit if no overlap
        if (
            src_x_edges[-1] < dst_x_edges[0]
            or src_x_edges[0] > dst_x_edges[-1]
            or src_y_edges[-1] < dst_y_edges[0]
            or src_y_edges[0] > dst_y_edges[-1]
        ):
            return (
                np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
                np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
            )

        dst_dx = dst_x_edges[1] - dst_x_edges[0]
        dst_dy = dst_y_edges[1] - dst_y_edges[0]
        dst_x0 = dst_x_edges[0]
        dst_y0 = dst_y_edges[0]

        src_x_centers = (src_x_edges[:-1] + src_x_edges[1:]) * 0.5
        src_y_centers = (src_y_edges[:-1] + src_y_edges[1:]) * 0.5

        dst_ix_1d = ((src_x_centers - dst_x0) / dst_dx).astype(np.int32)
        dst_iy_1d = ((src_y_centers - dst_y0) / dst_dy).astype(np.int32)

        valid_x = (dst_ix_1d >= 0) & (dst_ix_1d < dst_n_x)
        valid_y = (dst_iy_1d >= 0) & (dst_iy_1d < dst_n_y)

        valid_x_idx = np.where(valid_x)[0]
        valid_y_idx = np.where(valid_y)[0]

        if len(valid_x_idx) == 0 or len(valid_y_idx) == 0:
            return (
                np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
                np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
            )

        y_start, y_end = valid_y_idx[0], valid_y_idx[-1] + 1
        x_start, x_end = valid_x_idx[0], valid_x_idx[-1] + 1

        sub_intensity = intensity[y_start:y_end, x_start:x_end]

        if count is not None:
            sub_count = count[y_start:y_end, x_start:x_end].astype(np.float64)
            has_count = True
        else:
            sub_count = np.empty((0, 0), dtype=np.float64)
            has_count = False

        dst_sum, dst_count = _remap_numba_core(
            sub_intensity.astype(np.float64),
            sub_count,
            has_count,
            dst_ix_1d,
            dst_iy_1d,
            dst_n_x,
            dst_n_y,
            y_start,
            x_start,
        )

        return dst_sum, dst_count.astype(np.int32)

except ImportError:
    HAS_NUMBA = False
    remap_numba = None


def create_test_data(src_size: int, dst_size: int, overlap: float = 0.8):
    """Create test data for benchmarking."""
    # Source grid
    src_x_edges = np.linspace(0, 1000, src_size + 1)
    src_y_edges = np.linspace(0, 1000, src_size + 1)

    # Destination grid (partially overlapping)
    offset = (1 - overlap) * 500
    dst_x_edges = np.linspace(offset, 1000 + offset, dst_size + 1)
    dst_y_edges = np.linspace(offset, 1000 + offset, dst_size + 1)

    # Random intensity with some NaNs
    intensity = np.random.rand(src_size, src_size).astype(np.float64) * 100
    intensity[np.random.rand(src_size, src_size) < 0.1] = np.nan

    # Random counts
    count = np.random.randint(1, 10, (src_size, src_size)).astype(np.int32)

    return intensity, count, src_x_edges, src_y_edges, dst_x_edges, dst_y_edges, dst_size, dst_size


def benchmark_function(func, args, n_warmup=3, n_runs=20):
    """Benchmark a function."""
    # Warmup
    for _ in range(n_warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.mean(times), np.std(times), result


def main():
    print("=" * 70)
    print("Benchmark: remap_to_common_grid implementations")
    print("=" * 70)

    # Test different grid sizes
    sizes = [(200, 200), (400, 400), (600, 600)]

    for src_size, dst_size in sizes:
        print(f"\nSource: {src_size}x{src_size}, Destination: {dst_size}x{dst_size}")
        print("-" * 50)

        args = create_test_data(src_size, dst_size)

        # Benchmark original
        mean_orig, std_orig, result_orig = benchmark_function(remap_original, args)
        print(f"Original (meshgrid):    {mean_orig*1000:8.3f} ms ± {std_orig*1000:.3f} ms")

        # Benchmark optimized
        mean_opt, std_opt, result_opt = benchmark_function(remap_optimized, args)
        speedup = mean_orig / mean_opt
        print(f"Optimized (no meshgrid): {mean_opt*1000:8.3f} ms ± {std_opt*1000:.3f} ms  ({speedup:.2f}x speedup)")

        # Verify results match
        sum_match = np.allclose(result_orig[0], result_opt[0], rtol=1e-10, equal_nan=True)
        count_match = np.allclose(result_orig[1], result_opt[1], rtol=1e-10)
        print(f"  Results match: sum={sum_match}, count={count_match}")

        # Benchmark numba if available
        if HAS_NUMBA:
            # Extra warmup for JIT compilation
            mean_numba, std_numba, result_numba = benchmark_function(
                remap_numba, args, n_warmup=5, n_runs=20
            )
            speedup_numba = mean_orig / mean_numba
            print(f"Numba JIT:              {mean_numba*1000:8.3f} ms ± {std_numba*1000:.3f} ms  ({speedup_numba:.2f}x speedup)")

            sum_match = np.allclose(result_orig[0], result_numba[0], rtol=1e-10, equal_nan=True)
            count_match = np.allclose(result_orig[1], result_numba[1], rtol=1e-10)
            print(f"  Results match: sum={sum_match}, count={count_match}")
        else:
            print("Numba not available, skipping JIT benchmark")

    # Test early exit (no overlap)
    print(f"\nEarly exit test (no overlap):")
    print("-" * 50)
    intensity, count, src_x, src_y, _, _, _, _ = create_test_data(400, 400)
    # Create non-overlapping destination
    dst_x = np.linspace(2000, 3000, 401)
    dst_y = np.linspace(2000, 3000, 401)
    args_no_overlap = (intensity, count, src_x, src_y, dst_x, dst_y, 400, 400)

    mean_orig, _, _ = benchmark_function(remap_original, args_no_overlap)
    mean_opt, _, _ = benchmark_function(remap_optimized, args_no_overlap)
    print(f"Original:  {mean_orig*1000:8.3f} ms")
    print(f"Optimized: {mean_opt*1000:8.3f} ms  ({mean_orig/mean_opt:.1f}x speedup with early exit)")


if __name__ == "__main__":
    main()
