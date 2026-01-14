#!/usr/bin/env python3
"""
Plot the 12-bit counter constructed from top nibbles of 3 distance bins.

The WAMOS radar encodes a 12-bit binary counter across 3 distance bins (18, 19, 20):
  - Bin 18, bits 12-15 → bits 8-11 of the counter
  - Bin 19, bits 12-15 → bits 4-7 of the counter
  - Bin 20, bits 12-15 → bits 0-3 of the counter

This script plots this 12-bit value (0-4095) across multiple frames, with optional
transition lines for a specified bit/bin.

Usage:
    python plot_nibble.py 20220405T1400 20220405T1401 /path/to/POLAR
    python plot_nibble.py 20220405T1400 20220405T1401 /path/to/POLAR --transition=d00b12
    python plot_nibble.py 20220405T1400 20220405T1401 /path/to/POLAR --bins 18,19,20
    python plot_nibble.py 20220405T1400 20220405T1401 /path/to/POLAR --output counter.png

Dec-2025, Pat Welch, pat@mousebrains.com
    in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFrame  # noqa: E402
from wamos_tpw.args import add_time_range_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame_data(fn: str, bins: list[int]) -> tuple | None:
    """
    Load a single frame and extract nibble data from specified bins.

    Args:
        fn: Filename to load
        bins: List of 3 distance bin indices

    Returns:
        Tuple of (timestamp, counter_values, bit_arrays, n_bearings) or None on error
        where bit_arrays is a dict with keys 12, 13, 14, 15
    """
    try:
        frame = PolarFrame(fn)
        ts = np.datetime64(frame.metadata.get("frame_datetime") or frame.metadata.get("datetime"))

        # Extract 12-bit counter from 3 bins
        # Bin 0 (of the 3): bits 8-11
        # Bin 1 (of the 3): bits 4-7
        # Bin 2 (of the 3): bits 0-3
        raw = frame.raw

        # Get top nibble (bits 12-15) from each bin, shifted appropriately
        counter = np.right_shift(np.bitwise_and(raw[:, bins[0]], 0xF000), 4).astype(
            np.uint16
        )  # bits 8-11
        counter += np.right_shift(np.bitwise_and(raw[:, bins[1]], 0xF000), 8).astype(
            np.uint16
        )  # bits 4-7
        counter += np.right_shift(np.bitwise_and(raw[:, bins[2]], 0xF000), 12).astype(
            np.uint16
        )  # bits 0-3

        # Extract all bits for transition detection
        bit_arrays = {
            12: frame.bit12,
            13: frame.bit13,
            14: frame.bit14,
            15: frame.bit15,
        }

        return ts, counter, bit_arrays, frame.n_bearings

    except Exception as e:
        logger.warning("Failed to load %s: %s", fn, e)
        return None


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot 12-bit counter from top nibbles of 3 distance bins"
    )

    add_time_range_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "--bins",
        type=str,
        default="18,19,20",
        help="Three distance bins to use for 12-bit counter (default: 18,19,20)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save plot to file instead of displaying",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved plots (default: 150)")
    parser.add_argument(
        "--figsize",
        type=str,
        default="16,8",
        help="Figure size as 'width,height' in inches (default: 16,8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--transition",
        type=str,
        default=None,
        metavar="dNNbNN",
        help="Show transition lines for specified bit/bin (e.g., d00b12)",
    )
    parser.add_argument(
        "--frame-lines",
        action="store_true",
        help="Show frame boundary lines (blue)",
    )

    args = parser.parse_args()
    setup_logging(args)

    # Parse bins
    try:
        bins = [int(x.strip()) for x in args.bins.split(",")]
        if len(bins) != 3:
            logger.error("Must specify exactly 3 bins")
            return 1
    except ValueError:
        logger.error(f"Invalid bins format: {args.bins}")
        return 1

    # Parse transition argument
    transition_bit = None
    transition_bin = 0
    if args.transition:
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

    # Load frames
    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in time range")
            return 1

        logger.info("Loading %d files with %d workers", n_files, args.workers or os.cpu_count())

        # Load frames in parallel
        results: dict = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame_data, fn, bins): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    ts, counter, bit_arrays, n_bearings = result
                    results[ts] = (counter, bit_arrays, n_bearings)

        n_loaded = len(results)
        print(f"Successfully loaded {n_loaded} of {n_files} frames")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort by timestamp and concatenate
        sorted_times = sorted(results.keys())
        counter_list = []
        bit_lists: dict[int, list] = {12: [], 13: [], 14: [], 15: []}
        frame_boundaries = [0]

        for ts in sorted_times:
            counter, bit_arrays, n_bearings = results[ts]
            counter_list.append(counter)
            for bit_num in (12, 13, 14, 15):
                bit_lists[bit_num].append(bit_arrays[bit_num])
            frame_boundaries.append(frame_boundaries[-1] + n_bearings)

        combined_counter = np.concatenate(counter_list, axis=0)
        combined_bits: dict[int, np.ndarray] = {
            bit_num: np.concatenate(bit_lists[bit_num], axis=0) for bit_num in (12, 13, 14, 15)
        }
        total_samples = len(combined_counter)

        # Calculate transitions if requested
        transitions = None
        if transition_bit is not None:
            trans_data = combined_bits[transition_bit][:, transition_bin]
            transitions = np.where(trans_data[:-1] != trans_data[1:])[0] + 1

        # Plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(total_samples)
        ax.plot(x, combined_counter, linewidth=0.5, color="#1f77b4")

        # Frame boundaries
        if args.frame_lines:
            for boundary in frame_boundaries[1:-1]:
                ax.axvline(boundary, color="blue", linewidth=0.5, alpha=0.3)

        # Transition lines
        if transitions is not None:
            for t in transitions:
                ax.axvline(t, color="red", linewidth=0.5, alpha=0.5)

        ax.set_xlabel("Sample index")
        ax.set_ylabel("12-bit counter value")
        ax.set_xlim(0, total_samples)
        ax.set_ylim(0, 4096)

        title = f"12-bit counter from bins {bins}: {args.stime} to {args.etime}"
        title += f"\n{n_loaded} frames, {total_samples} samples"
        if transition_bit is not None:
            title += f" | Red lines: bit {transition_bit} @ bin {transition_bin}"
        ax.set_title(title)

        fig.tight_layout()

        if args.output:
            fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved: {args.output}")
        else:
            plt.show()

    except (FileNotFoundError, ValueError, OSError) as e:
        logger.exception(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
