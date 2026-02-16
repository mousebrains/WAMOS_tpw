#!/usr/bin/env python3
"""
Generate bit statistics for the 1Hz bit overlaps.

This script loads all polar files for a given time range and calculates
statistics on bit transitions relative to the 1Hz signal at bin 18.

Usage:
    python 1Hz_stats.py 20220405 20220406 /path/to/POLAR
    python 1Hz_stats.py 20220405 20220406 /path/to/POLAR --distance-bins 0:21

For best performance with free-threaded Python 3.13+:
    python3.13t 1Hz_stats.py ... --workers 8

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

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_distance_bins(spec: str, n_ranges: int) -> list[int]:
    """
    Parse a distance bin specification string.

    Supports:
    - Comma-separated values: "0,18,19,20"
    - Slice notation: "0:21" (exclusive end)
    - Mixed: "0,5,10:15,20"

    Args:
        spec: Distance bin specification string
        n_ranges: Maximum number of range bins (for validation)

    Returns:
        List of distance bin indices
    """
    bins = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":", 1)
            start = int(start) if start else 0
            end = int(end) if end else n_ranges
            bins.extend(range(start, min(end, n_ranges)))
        else:
            idx = int(part)
            if 0 <= idx < n_ranges:
                bins.append(idx)
    return sorted(set(bins))


def load_frame_bits(args: tuple) -> list | None:
    """
    Worker function to load a single frame and extract top nibble bits.

    Args:
        args: Tuple of (filename, distance_bins)

    Returns:
        List of [RPT, first_values..., pre_counts..., post_counts..., norm_counts...]
        where norm_counts = post_counts / (RPT - 1), or None on error.
    """
    fn, distance_bins = args

    d18 = distance_bins.index(18)

    try:
        frame = PolarFile(fn).frame()
        a = np.bitwise_and(frame.raw[:, distance_bins], 0xF000)
        a = np.right_shift(a, 12).astype(np.uint8)

        qPre = np.bitwise_and(a[:, d18], 0b0001) == 0  # 1Hz is zero
        qPost = np.logical_not(qPre)

        first = []
        nPre = []
        nPost = []

        for bit in [1, 2, 4, 8]:
            b = np.bitwise_and(a, bit) != 0
            pulse = np.diff(b, axis=0, append=0) != 0
            first.extend((b[0, :] != 0).astype("uint8").tolist())
            nPre.extend(pulse[qPre, :].sum(axis=0).tolist())
            nPost.extend(pulse[qPost, :].sum(axis=0).tolist())

        rpt = frame.metadata.repeat_time
        if rpt is None or rpt <= 0:
            logger.warning("Invalid RPT %s in %s, skipping", rpt, fn)
            return None

        row = [rpt]
        row.extend(first)  # First bit in frame
        row.extend(nPre)  # Number of transitions pre 1Hz transition
        row.extend(nPost)  # Number of transitions post 1Hz transition
        row.extend((np.array(nPost) / (rpt - 1)).tolist())  # Normalized post counts

        return row
    except Exception as e:
        logger.warning("Failed to load %s: %s", fn, e)
        return None


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1Hz bit statistics for many frames worth of polar files"
    )

    add_common_arguments(parser)
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
    args = parser.parse_args()
    setup_logging(args)

    # Parse distance bins
    # Use a dummy n_ranges for initial parsing, will be validated later
    distance_bins = parse_distance_bins(args.distance_bins, 2000)

    if 18 not in distance_bins:
        distance_bins.append(18)
        distance_bins = sorted(distance_bins)

    # Run analysis
    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)
        logger.info("Loading %d files with %s workers", n_files, args.workers or os.cpu_count())

        # Load frames in parallel using ThreadPoolExecutor
        work_items = [(fn, distance_bins) for fn in files]
        items = []

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame_bits, item): item[0] for item in work_items}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    items.append(result)
        elapsed = time.perf_counter() - t0

        n_loaded = len(items)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(f"Successfully loaded {n_loaded} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        items = np.array(items)
        rpt = items[:, 0]
        items = items[:, 1:]
        mu = items.mean(axis=0)
        std = items.std(axis=0)

        mu = mu.reshape(4, 4, len(distance_bins))
        std = std.reshape(4, 4, len(distance_bins))

        # Print formatted table
        n_wide = 80
        print()
        print("1Hz Bit Statistics:")
        print("-" * n_wide)
        print(f"{'Signal':^10} | {'FirstVal':^13} | {'Pre':^15} | {'Post':^15} | {'Norm':^15}")
        print("-" * n_wide)

        for index in range(4):  # Walk over bits
            bit = index + 12
            for j in range(len(distance_bins)):
                signal = f"b{bit:02d}_d{distance_bins[j]:02d}"
                first_val = f"{mu[0, index, j]:.2f} +/- {std[0, index, j]:.2f}"
                pre_val = f"{mu[1, index, j]:.2f} +/- {std[1, index, j]:.2f}"
                post_val = f"{mu[2, index, j]:.2f} +/- {std[2, index, j]:.2f}"
                norm_val = f"{mu[3, index, j]:.2f} +/- {std[3, index, j]:.2f}"
                row = (
                    f"{signal:^10} | {first_val:>12} | {pre_val:>15} | "
                    f"{post_val:>15} | {norm_val:>15}"
                )
                print(row)

        print("-" * n_wide)
        print(f"Frames: {n_loaded}, Mean RPT: {rpt.mean():.3f}s")

        # Report peak memory usage
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == "darwin":
            peak_mem_mb = peak_mem / (1024 * 1024)
        else:
            peak_mem_mb = peak_mem / 1024
        print(f"Peak memory: {peak_mem_mb:.1f} MB")

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.exception("Error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
