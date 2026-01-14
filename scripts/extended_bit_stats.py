#!/usr/bin/env python3
"""
Generate bit statistics for a day's worth of WAMOS polar files.

This script loads all polar files for a given date and calculates
bit transition statistics across the entire day.

Usage:
    python extended_bit_stats.py 2022-04-05 2022-04-06 /path/to/POLAR
    python extended_bit_stats.py 2022-04-05 2022-04-06 /path/to/POLAR --distance-bins 0,18,19,20
    python extended_bit_stats.py 2022-04-05 2022-04-06 /path/to/POLAR --correlations

Cross-correlation options:
    --correlations          Calculate Pearson correlations between bit/bin pairs
    --corr-threshold 0.1    Minimum |r| to display (default: 0.1)
    --p-threshold 0.05      Maximum p-value for significance (default: 0.05)
    --freq-tolerance 0.25   Only correlate pairs within 25% frequency of each other
    --max-lag 1000          Maximum lag to search for cross-correlation (default: 1000)

For best performance with free-threaded Python 3.13+:
    python3.13t extended_bit_stats.py ... --workers 8

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.stats import pearsonr

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFrame  # noqa: E402
from wamos_tpw.args import add_time_range_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.plotting import parse_distance_bins  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame_bits(args: tuple) -> tuple:
    """
    Worker function to load a single frame and extract top nibble bits.

    Args:
        args: Tuple of (filename, distance_bins)

    Returns:
        Tuple of (timestamp, nibble_array) or (None, None) on error
    """
    fn, distance_bins = args
    try:
        frame = PolarFrame(fn)
        a = np.bitwise_and(frame.raw[:, distance_bins], 0xF000)
        a = np.right_shift(a, 12).astype(np.uint8)
        ts = np.datetime64(frame.metadata.get("frame_datetime") or frame.metadata.get("datetime"))
        return ts, a
    except Exception as e:
        logger.warning(f"Failed to load {fn}: {e}")
        return None, None


def calc_stats(bits, times):
    """
    Calculate transition statistics for bit data.

    Args:
        bits: 2D array of bit values (n_samples, n_bins)
        times: 1D array of timestamps

    Returns:
        Tuple of (hz_array, std_array, pct_zero_array)
    """
    dBits = np.diff(bits, axis=0)

    nZeros = np.sum(bits == 0, axis=0).astype(float)
    n = float(bits.shape[0])
    pct_zero = np.divide(nZeros, n, where=n != 0) * 100

    hz = np.empty(dBits.shape[1])
    std = np.empty(dBits.shape[1])

    for index in range(dBits.shape[1]):
        iTransition = np.where(dBits[:, index] != 0)[0]
        if len(iTransition) < 3:
            hz[index] = np.nan
            std[index] = np.nan
        else:
            dt = np.diff(times[iTransition]).astype("timedelta64[ms]").astype(float) / 1000.0
            hz[index] = 1 / dt.mean()
            std[index] = dt.std()

    return hz, std, pct_zero


def print_progress(current: int, total: int, width: int = 40, prefix: str = "Progress") -> None:
    """
    Print a progress bar that updates in place.

    Args:
        current: Current progress count
        total: Total count
        width: Width of the progress bar in characters
        prefix: Prefix text
    """
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    cnt_width = len(str(total))
    msg = f"\r{prefix}: [{bar}] {current:>{cnt_width}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()  # Newline when complete


def print_stats_table(distance_bins, stats_dict):
    """
    Print a formatted table of bit statistics.

    Args:
        distance_bins: List of distance bin indices
        stats_dict: Dict mapping bit name to (hz, std, pct_zero) tuples
    """
    # Column widths
    bin_w = 5
    hz_w = 8
    std_w = 8
    pct_w = 7

    # Build header
    bit_names = list(stats_dict.keys())
    header1 = f"{'bin':>{bin_w}} |"
    header2 = f"{'':>{bin_w}} |"
    sep = "-" * bin_w + "-+"

    for name in bit_names:
        col_w = hz_w + std_w + pct_w + 2
        header1 += f" {name:^{col_w}} |"
        header2 += f" {'Hz':>{hz_w}} {'std_s':>{std_w}} {'%zero':>{pct_w}} |"
        sep += "-" * (col_w + 2) + "+"

    print(sep)
    print(header1)
    print(header2)
    print(sep)

    # Print rows
    for i, bin_idx in enumerate(distance_bins):
        row = f"{bin_idx:>{bin_w}} |"
        for name in bit_names:
            hz, std, pct = stats_dict[name]
            hz_val = f"{hz[i]:>{hz_w}.1f}" if np.isfinite(hz[i]) else f"{'N/A':>{hz_w}}"
            std_val = f"{std[i]:>{std_w}.3f}" if np.isfinite(std[i]) else f"{'N/A':>{std_w}}"
            pct_val = f"{pct[i]:>{pct_w}.1f}"
            row += f" {hz_val} {std_val} {pct_val} |"
        print(row)

    print(sep)


def print_stats_by_frequency(distance_bins, stats_dict, min_sigma=5.0):
    """
    Print a flat table of bit/distance combinations sorted by frequency.

    Only shows signals where mean_interval / std_s >= min_sigma (i.e., significant frequencies).

    Args:
        distance_bins: List of distance bin indices
        stats_dict: Dict mapping bit name to (hz, std, pct_zero) tuples
        min_sigma: Minimum significance level (mean_interval / std >= min_sigma)
    """
    # Build flat list of (hz, label, std, pct, sigma)
    rows = []
    for bit_name, (hz_arr, std_arr, pct_arr) in stats_dict.items():
        for i, bin_idx in enumerate(distance_bins):
            hz = hz_arr[i]
            std = std_arr[i]
            pct = pct_arr[i]
            # Create label like b13_d00
            label = f"{bit_name}_d{bin_idx:02d}"

            # Calculate significance: mean_interval / std_s
            # mean_interval = 1/Hz (in seconds), std is already in seconds
            if np.isfinite(hz) and np.isfinite(std) and hz > 0 and std > 0:
                mean_interval = 1.0 / hz
                sigma = mean_interval / std
                rows.append((hz, label, std, pct, sigma))

    # Filter by minimum sigma and sort by Hz descending
    rows = [r for r in rows if r[4] >= min_sigma]
    rows.sort(key=lambda x: -x[0])

    # Print table
    print("-" * 57)
    print(f"{'Signal':<12} {'Hz':>10} {'std_s':>10} {'%zero':>10} {'sigma':>10}")
    print("-" * 57)

    for hz, label, std, pct, sigma in rows:
        print(f"{label:<12} {hz:>10.1f} {std:>10.3f} {pct:>10.1f} {sigma:>10.1f}")

    print("-" * 57)
    print(f"Showing signals with significance >= {min_sigma} sigma")


def _calc_single_correlation(args):
    """
    Worker function to calculate Pearson correlation and lag cross-correlation.

    Args:
        args: Tuple of (i, j, col_i, col_j, std_i, std_j, max_lag)

    Returns:
        Tuple of (i, j, r, p, best_lag, lag_corr)
        - r, p: Pearson correlation and p-value at lag 0
        - best_lag: Lag with maximum correlation (negative = i leads j)
        - lag_corr: Correlation value at best lag
    """
    i, j, col_i, col_j, std_i, std_j, max_lag = args

    # Check for constant columns (no variance)
    if std_i == 0 or std_j == 0:
        return i, j, np.nan, np.nan, 0, np.nan

    # Pearson correlation at lag 0
    r, p = pearsonr(col_i, col_j)

    # Cross-correlation to find best lag
    # Normalize to zero-mean
    a = col_i - col_i.mean()
    b = col_j - col_j.mean()

    # Full cross-correlation
    xcorr = correlate(a, b, mode="full")
    mid = len(xcorr) // 2

    # Search within max_lag
    start = max(0, mid - max_lag)
    end = min(len(xcorr), mid + max_lag + 1)
    search_slice = xcorr[start:end]

    # Find best lag
    best_idx = np.argmax(np.abs(search_slice))
    best_lag = best_idx - (mid - start)

    # Normalize correlation at best lag
    n = len(a)
    lag_corr = xcorr[start + best_idx] / (n * std_i * std_j)

    return i, j, r, p, best_lag, lag_corr


def calc_correlations(
    nibble, distance_bins, stats_dict, workers=None, freq_tolerance=0.25, max_lag=1000
):
    """
    Calculate Pearson correlation and lag cross-correlation between bit/bin combinations
    with similar frequencies.

    Uses ThreadPoolExecutor for GIL-free parallel computation with free-threaded Python.
    Only computes correlations for pairs where frequencies are within freq_tolerance of each other.

    Args:
        nibble: 2D array (n_samples, n_bins) of 4-bit values
        distance_bins: List of bin indices
        stats_dict: Dict mapping bit name to (hz, std, pct_zero) tuples
        workers: Number of worker threads (default: CPU count)
        freq_tolerance: Maximum relative frequency difference (default: 0.25 = 25%)
        max_lag: Maximum lag to search for cross-correlation (default: 1000 samples)

    Returns:
        Tuple of (corr_matrix, p_matrix, lag_matrix, lag_corr_matrix, labels)
        - corr_matrix: Correlation coefficients at lag 0 (n_cols, n_cols)
        - p_matrix: P-values for significance (n_cols, n_cols)
        - lag_matrix: Best lag values (n_cols, n_cols)
        - lag_corr_matrix: Correlation at best lag (n_cols, n_cols)
        - labels: List of column labels
    """
    n_samples = nibble.shape[0]
    n_bins = len(distance_bins)
    n_cols = n_bins * 4

    # Build matrix: columns are (bin0_b12, bin0_b13, bin0_b14, bin0_b15, bin1_b12, ...)
    data = np.empty((n_samples, n_cols), dtype=np.float32)
    labels = []
    frequencies = []  # Hz for each column

    bit_names = ["b12", "b13", "b14", "b15"]
    for i, bin_idx in enumerate(distance_bins):
        data[:, i * 4 + 0] = (nibble[:, i] & 0b0001).astype(np.float32)  # b12
        data[:, i * 4 + 1] = (nibble[:, i] & 0b0010).astype(np.float32)  # b13
        data[:, i * 4 + 2] = (nibble[:, i] & 0b0100).astype(np.float32)  # b14
        data[:, i * 4 + 3] = (nibble[:, i] & 0b1000).astype(np.float32)  # b15
        labels.extend([f"b{b}_d{bin_idx:02d}" for b in [12, 13, 14, 15]])
        # Get frequencies for each bit at this distance bin
        for bit_name in bit_names:
            hz_arr = stats_dict[bit_name][0]
            frequencies.append(hz_arr[i])

    frequencies = np.array(frequencies)

    # Pre-calculate standard deviations for variance check
    stds = np.std(data, axis=0)

    # Initialize matrices with diagonal
    corr_matrix = np.eye(n_cols)
    p_matrix = np.zeros((n_cols, n_cols))
    lag_matrix = np.zeros((n_cols, n_cols), dtype=np.int32)
    lag_corr_matrix = np.eye(n_cols)

    # Build work items for upper triangle, only for pairs with similar frequencies
    work_items = []
    skipped = 0
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            hz_i = frequencies[i]
            hz_j = frequencies[j]

            # Skip if either frequency is NaN or zero
            if not np.isfinite(hz_i) or not np.isfinite(hz_j) or hz_i == 0 or hz_j == 0:
                skipped += 1
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                p_matrix[i, j] = np.nan
                p_matrix[j, i] = np.nan
                lag_corr_matrix[i, j] = np.nan
                lag_corr_matrix[j, i] = np.nan
                continue

            # Check if frequencies are within tolerance of each other
            # Use relative difference: |hz_i - hz_j| / max(hz_i, hz_j) <= tolerance
            max_hz = max(hz_i, hz_j)
            rel_diff = abs(hz_i - hz_j) / max_hz
            if rel_diff > freq_tolerance:
                skipped += 1
                corr_matrix[i, j] = np.nan
                corr_matrix[j, i] = np.nan
                p_matrix[i, j] = np.nan
                p_matrix[j, i] = np.nan
                lag_corr_matrix[i, j] = np.nan
                lag_corr_matrix[j, i] = np.nan
                continue

            work_items.append((i, j, data[:, i], data[:, j], stds[i], stds[j], max_lag))

    n_pairs = len(work_items)
    n_workers = workers or os.cpu_count()
    logger.info(
        "Calculating %d correlation pairs (%d skipped due to frequency mismatch) with %d workers",
        n_pairs,
        skipped,
        n_workers,
    )

    # Calculate correlations in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_calc_single_correlation, item): item[:2] for item in work_items}

        for k, future in enumerate(as_completed(futures)):
            print_progress(k + 1, n_pairs, prefix="Correlating")

            i, j, r, p, best_lag, lag_corr = future.result()
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
            p_matrix[i, j] = p
            p_matrix[j, i] = p
            lag_matrix[i, j] = best_lag
            lag_matrix[j, i] = -best_lag  # Reversed lag for symmetric access
            lag_corr_matrix[i, j] = lag_corr
            lag_corr_matrix[j, i] = lag_corr

    return corr_matrix, p_matrix, lag_matrix, lag_corr_matrix, labels


def print_correlation_table(
    corr_matrix, p_matrix, lag_matrix, lag_corr_matrix, labels, threshold=0.1, p_threshold=0.05
):
    """
    Print significant correlations in a table format with lag information.

    Args:
        corr_matrix: Correlation coefficient matrix (at lag 0)
        p_matrix: P-value matrix
        lag_matrix: Best lag values matrix
        lag_corr_matrix: Correlation at best lag matrix
        labels: Column labels
        threshold: Minimum absolute correlation to display
        p_threshold: Maximum p-value to consider significant
    """
    print(f"\nSignificant Correlations (|r| >= {threshold}, p < {p_threshold}):")
    print("-" * 78)
    print(f"{'Pair':<25} {'r(lag=0)':>10} {'P-value':>12} {'Best Lag':>10} {'r(best)':>10}")
    print("-" * 78)

    n = len(labels)
    significant = []

    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            p = p_matrix[i, j]
            lag = lag_matrix[i, j]
            lag_r = lag_corr_matrix[i, j]

            if np.isnan(r) or np.isnan(p):
                continue

            # Use best of lag=0 or best lag correlation for threshold
            best_r = max(abs(r), abs(lag_r)) if np.isfinite(lag_r) else abs(r)
            if best_r >= threshold and p < p_threshold:
                significant.append((best_r, labels[i], labels[j], r, p, lag, lag_r))

    # Sort by best absolute correlation (descending)
    significant.sort(key=lambda x: -x[0])

    for _, label_i, label_j, r, p, lag, lag_r in significant:
        pair = f"{label_i} vs {label_j}"
        lag_r_str = f"{lag_r:>10.4f}" if np.isfinite(lag_r) else f"{'N/A':>10}"
        print(f"{pair:<25} {r:>10.4f} {p:>12.2e} {lag:>10d} {lag_r_str}")

    print("-" * 78)
    print(f"Total: {len(significant)} significant pairs")
    print("Best Lag: positive = second signal leads, negative = first signal leads")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bit statistics for many frames worth of polar files"
    )

    add_time_range_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "--distance-bins",
        type=str,
        default="0:21",
        help="Distance bins to analyze (e.g., '0,18,19,20' or '0:21')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=5.0,
        help="Minimum significance (mean_interval/std) for frequency table (default: 5.0)",
    )
    parser.add_argument(
        "--correlations",
        action="store_true",
        help="Calculate and display cross-correlations between bits and distance bins",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.1,
        help="Minimum absolute correlation to display (default: 0.1)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Maximum p-value for significance (default: 0.05)",
    )
    parser.add_argument(
        "--freq-tolerance",
        type=float,
        default=0.25,
        help="Maximum relative frequency difference for correlation pairs (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=1000,
        help="Maximum lag to search for cross-correlation (default: 1000 samples)",
    )
    args = parser.parse_args()
    setup_logging(args)

    # Parse distance bins
    # Use a dummy n_ranges for initial parsing, will be validated later
    distance_bins = parse_distance_bins(args.distance_bins, 2000)

    # Run analysis
    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)
        logger.info(f"Loading {n_files} files with {args.workers or os.cpu_count()} workers")

        # Load frames in parallel using ThreadPoolExecutor
        bits = {}
        work_items = [(fn, distance_bins) for fn in files]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame_bits, item): item[0] for item in work_items}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")

                ts, a = future.result()
                if ts is not None:
                    bits[ts] = a

        print(f"Successfully loaded {len(bits)} unique frames")

        times = []  # Time sorted timestamps
        nibble = []  # Time sorted bits
        for ts in sorted(bits):
            val = bits[ts]
            nibble.append(val)
            a = np.empty(val.shape[0])
            a[:] = np.nan
            a[0] = ts
            times.append(a)

        nibble = np.concatenate(nibble, axis=0)
        times = np.concatenate(times, axis=0)

        # Interpolate missing timestamps
        q = np.where(np.isfinite(times))[0]  # Valid timestamps
        indices = np.arange(times.shape[0])  # All indices
        f = interp1d(indices[q], times[q], kind="nearest", fill_value="extrapolate")
        times = f(indices).astype("datetime64[us]")  # Fill in missing values

        # Calculate statistics for each bit
        stats = {
            "b12": calc_stats(np.bitwise_and(nibble, 0b0001), times),
            "b13": calc_stats(np.bitwise_and(nibble, 0b0010), times),
            "b14": calc_stats(np.bitwise_and(nibble, 0b0100), times),
            "b15": calc_stats(np.bitwise_and(nibble, 0b1000), times),
        }

        # Print results
        print(f"\nBit Statistics: {args.stime} to {args.etime}")
        print(f"Frames: {len(bits)}, Total samples: {nibble.shape[0]}")
        print()
        print_stats_table(distance_bins, stats)
        print(f"\nSorted by frequency (>= {args.min_sigma} sigma):")
        print_stats_by_frequency(distance_bins, stats, args.min_sigma)

        # Calculate and print correlations if requested
        if args.correlations:
            print("\nCalculating cross-correlations...")
            corr_matrix, p_matrix, lag_matrix, lag_corr_matrix, labels = calc_correlations(
                nibble, distance_bins, stats, args.workers, args.freq_tolerance, args.max_lag
            )
            print_correlation_table(
                corr_matrix,
                p_matrix,
                lag_matrix,
                lag_corr_matrix,
                labels,
                threshold=args.corr_threshold,
                p_threshold=args.p_threshold,
            )

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.exception(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
