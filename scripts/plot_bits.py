#!/usr/bin/env python3
"""
Plot top 4 bits for multiple WAMOS polar frames.

This script loads polar files and displays bit 12-15 values across all bearings
for specified distance bins, with vertical lines marking frame boundaries.

Usage:
    python plot_bits.py 20220405T1400 20220405T1401 /path/to/POLAR
    python plot_bits.py 20220405T1400 20220405T1401 /path/to/POLAR --distance-bins 0,18,19,20
    python plot_bits.py 20220405T1400 20220405T1401 /path/to/POLAR --output bits.png
    python plot_bits.py 20220405T1400 20220405T1401 /path/to/POLAR --summary
    python plot_bits.py 20220405T1400 20220405T1401 /path/to/POLAR --summary --min-sigma 3.0
    python plot_bits.py 20220405T1400 20220405T1401 /path/to/POLAR --group-by-distance
    python plot_bits.py ... --group-by-distance --reverse-bits

For best performance with free-threaded Python 3.13+:
    python3.13t plot_bits.py ... --workers 8

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_distance_bins(spec: str, n_distances: int) -> list[int]:
    """
    Parse a distance bin specification string.

    Supports:
    - Comma-separated values: "0,18,19,20"
    - Slice notation: "0:21" (exclusive end)
    - Mixed: "0,5,10:15,20"

    Args:
        spec: Distance bin specification string
        n_distances: Maximum number of range bins (for validation)

    Returns:
        List of distance bin indices
    """
    bins = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":", 1)
            start = int(start) if start else 0
            end = int(end) if end else n_distances
            bins.extend(range(start, min(end, n_distances)))
        else:
            idx = int(part)
            if 0 <= idx < n_distances:
                bins.append(idx)
    return sorted(set(bins))


def load_frame(fn: str) -> tuple:
    """
    Worker function to load a single frame.

    Args:
        fn: Filename to load

    Returns:
        Tuple of (timestamp, Frame) or (None, None) on error
    """
    try:
        frame = PolarFile(fn).frame()
        ts = frame.timestamp
        return ts, frame
    except Exception as e:
        logger.warning(f"Failed to load {fn}: {e}")
        return None, None


def print_progress(current: int, total: int, width: int = 40, prefix: str = "Progress") -> None:
    """Print a progress bar that updates in place."""
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    cnt_width = len(str(total))
    msg = f"\r{prefix}: [{bar}] {current:>{cnt_width}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()


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


def plot_frames_bits(
    frames,
    distance_bins,
    figsize=(14, 10),
    title=None,
    transition_bit=None,
    transition_bin=0,
):
    """
    Plot bits 12-15 for multiple frames in a 2x2 grid.

    Args:
        frames: List of Frame objects
        distance_bins: List of distance bin indices
        figsize: Figure size as (width, height)
        title: Plot title
        transition_bit: Bit to mark transitions for (12-15)
        transition_bin: Distance bin for transition detection

    Returns:
        Tuple of (fig, axes)
    """
    import matplotlib.pyplot as plt

    # Concatenate data from all frames
    frame_boundaries = [0]
    all_data = {12: [], 13: [], 14: [], 15: []}

    for frame in frames:
        n = frame.n_bearings
        for bit in [12, 13, 14, 15]:
            bit_data = getattr(frame, f"bit{bit}")[:, distance_bins]
            all_data[bit].append(bit_data)
        frame_boundaries.append(frame_boundaries[-1] + n)

    # Concatenate
    for bit in [12, 13, 14, 15]:
        all_data[bit] = np.concatenate(all_data[bit], axis=0)

    total_samples = all_data[12].shape[0]
    x = np.arange(total_samples)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for ax, bit in zip(axes, [12, 13, 14, 15]):
        data = all_data[bit]
        for i, d_bin in enumerate(distance_bins):
            ax.plot(x, data[:, i] + i * 1.2, linewidth=0.5, label=f"d{d_bin:02d}")
        ax.set_ylabel(f"Bit {bit}")
        ax.set_ylim(-0.2, len(distance_bins) * 1.2)
        ax.set_yticks([])

        # Frame boundaries
        for boundary in frame_boundaries[1:-1]:
            ax.axvline(boundary, color="gray", linewidth=0.3, alpha=0.5)

        # Transition lines
        if transition_bit == bit and transition_bin in distance_bins:
            d_idx = distance_bins.index(transition_bin)
            trans = np.where(data[:-1, d_idx] != data[1:, d_idx])[0] + 1
            for t in trans:
                ax.axvline(t, color="red", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Sample index")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_frames_bits_by_distance(
    frames,
    distance_bins,
    figsize=(14, 10),
    title=None,
    reverse_bits=False,
    transition_bit=None,
    transition_bin=0,
):
    """
    Plot bits grouped by distance bin.

    Args:
        frames: List of Frame objects
        distance_bins: List of distance bin indices
        figsize: Figure size as (width, height)
        title: Plot title
        reverse_bits: If True, display bits 15-12 instead of 12-15
        transition_bit: Bit to mark transitions for (12-15)
        transition_bin: Distance bin for transition detection

    Returns:
        Tuple of (fig, axes)
    """
    import matplotlib.pyplot as plt

    # Concatenate data from all frames
    frame_boundaries = [0]
    all_data = {}

    for d_bin in distance_bins:
        all_data[d_bin] = {12: [], 13: [], 14: [], 15: []}

    for frame in frames:
        n = frame.n_bearings
        for d_bin in distance_bins:
            for bit in [12, 13, 14, 15]:
                bit_data = getattr(frame, f"bit{bit}")[:, d_bin]
                all_data[d_bin][bit].append(bit_data)
        frame_boundaries.append(frame_boundaries[-1] + n)

    # Concatenate
    for d_bin in distance_bins:
        for bit in [12, 13, 14, 15]:
            all_data[d_bin][bit] = np.concatenate(all_data[d_bin][bit], axis=0)

    total_samples = len(all_data[distance_bins[0]][12])
    x = np.arange(total_samples)

    n_bins = len(distance_bins)
    fig, axes = plt.subplots(n_bins, 1, figsize=figsize, sharex=True)
    if n_bins == 1:
        axes = [axes]

    bits_order = [15, 14, 13, 12] if reverse_bits else [12, 13, 14, 15]

    for ax, d_bin in zip(axes, distance_bins):
        for i, bit in enumerate(bits_order):
            data = all_data[d_bin][bit]
            ax.plot(x, data + i * 1.2, linewidth=0.5, label=f"b{bit}")
        ax.set_ylabel(f"d{d_bin:02d}")
        ax.set_ylim(-0.2, 4 * 1.2)
        ax.set_yticks([])

        # Frame boundaries
        for boundary in frame_boundaries[1:-1]:
            ax.axvline(boundary, color="gray", linewidth=0.3, alpha=0.5)

        # Transition lines
        if transition_bit is not None and d_bin == transition_bin:
            data = all_data[d_bin][transition_bit]
            trans = np.where(data[:-1] != data[1:])[0] + 1
            for t in trans:
                ax.axvline(t, color="red", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Sample index")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Plot top 4 bits for multiple WAMOS polar frames")

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "--distance-bins",
        type=str,
        default="0:21",
        help="Distance bins to display (e.g., '0:21' or '0,18,19,50')",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save plot to file instead of displaying"
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved plots (default: 150)")
    parser.add_argument(
        "--summary", action="store_true", help="Print bit statistics summary instead of plotting"
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=5.0,
        help="Minimum significance (mean_interval/std) for frequency table (default: 5.0)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="14,10",
        help="Figure size as 'width,height' in inches (default: 14,10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--group-by-distance",
        action="store_true",
        help="Group bits by distance bin (b12-15 for d00, then b12-15 for d01, etc.)",
    )
    parser.add_argument(
        "--reverse-bits",
        action="store_true",
        help="Display bits in reverse order (b15 to b12) when using --group-by-distance",
    )
    parser.add_argument(
        "--transition",
        type=str,
        default=None,
        metavar="dNNbNN",
        help="Show transition lines for specified bit/bin (e.g., d00b12 for bit 12 at bin 0)",
    )

    args = parser.parse_args()
    setup_logging(args)

    # Parse transition argument (e.g., d00b12 -> bin=0, bit=12)
    transition_bit = None
    transition_bin = 0
    if args.transition:
        import re

        match = re.match(r"d(\d+)b(\d+)", args.transition.lower())
        if match:
            transition_bin = int(match.group(1))
            transition_bit = int(match.group(2))
            if transition_bit not in (12, 13, 14, 15):
                logger.error(f"Invalid bit in --transition: {transition_bit}. Must be 12-15.")
                return 1
        else:
            logger.error(
                f"Invalid --transition format: {args.transition}. Use dNNbNN (e.g., d00b12)"
            )
            return 1

    # Parse figure size
    try:
        figsize = tuple(float(x) for x in args.figsize.split(","))
    except ValueError:
        logger.error(f"Invalid figsize: {args.figsize}")
        return 1

    # Load frames in parallel
    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in time range")
            return 1

        logger.info(f"Loading {n_files} files with {args.workers or os.cpu_count()} workers")

        # Load frames in parallel
        frames_dict = {}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame, fn): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")

                ts, frame = future.result()
                if ts is not None:
                    frames_dict[ts] = frame
        elapsed = time.perf_counter() - t0

        # Sort frames by timestamp
        frames = [frames_dict[ts] for ts in sorted(frames_dict)]
        n_loaded = len(frames)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(f"Successfully loaded {n_loaded} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)")

        if not frames:
            logger.error("No valid frames loaded")
            return 1

        distance_bins = parse_distance_bins(args.distance_bins, frames[0].n_distances)

        if args.summary:
            # Extract nibble data and timestamps from frames
            times_list = []
            nibble_list = []
            sorted_timestamps = sorted(frames_dict.keys())

            for ts in sorted_timestamps:
                frame = frames_dict[ts]
                # Extract top nibble for selected distance bins
                raw_selected = frame.raw[:, distance_bins]
                nibble = np.right_shift(np.bitwise_and(raw_selected, 0xF000), 12).astype(np.uint8)
                nibble_list.append(nibble)

                # Build timestamp array (only first sample has the timestamp)
                t_arr = np.empty(frame.n_bearings)
                t_arr[:] = np.nan
                t_arr[0] = ts
                times_list.append(t_arr)

            nibble = np.concatenate(nibble_list, axis=0)
            times = np.concatenate(times_list, axis=0)

            # Interpolate missing timestamps
            q = np.where(np.isfinite(times))[0]
            indices = np.arange(times.shape[0])
            f = interp1d(indices[q], times[q], kind="nearest", fill_value="extrapolate")
            times = f(indices).astype("datetime64[us]")

            # Calculate statistics for each bit
            stats = {
                "b12": calc_stats(np.bitwise_and(nibble, 0b0001), times),
                "b13": calc_stats(np.bitwise_and(nibble, 0b0010), times),
                "b14": calc_stats(np.bitwise_and(nibble, 0b0100), times),
                "b15": calc_stats(np.bitwise_and(nibble, 0b1000), times),
            }

            # Print results
            print(f"\nBit Statistics: {args.stime} to {args.etime}")
            print(f"Frames: {n_loaded}, Total samples: {nibble.shape[0]}")
            print()
            print_stats_table(distance_bins, stats)
            print(f"\nSorted by frequency (>= {args.min_sigma} sigma):")
            print_stats_by_frequency(distance_bins, stats, args.min_sigma)
        else:
            # Plot bits
            import matplotlib.pyplot as plt

            title = f"Top 4 bits: {args.stime} to {args.etime}"

            if args.group_by_distance:
                # Group by distance bin: b12-15 for d00, then b12-15 for d01, etc.
                fig, axes = plot_frames_bits_by_distance(
                    frames,
                    distance_bins=distance_bins,
                    figsize=figsize,
                    title=title,
                    reverse_bits=args.reverse_bits,
                    transition_bit=transition_bit,
                    transition_bin=transition_bin,
                )
            else:
                # Default: 2x2 grid with one subplot per bit
                fig, axes = plot_frames_bits(
                    frames,
                    distance_bins=distance_bins,
                    figsize=figsize,
                    title=title,
                    transition_bit=transition_bit,
                    transition_bin=transition_bin,
                )

            if args.output:
                fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved: {args.output}")
            else:
                plt.show()

        # Report peak memory usage
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == "darwin":
            peak_mem_mb = peak_mem / (1024 * 1024)
        else:
            peak_mem_mb = peak_mem / 1024
        print(f"Peak memory: {peak_mem_mb:.1f} MB")

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.exception(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
