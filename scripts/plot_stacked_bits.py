#!/usr/bin/env python3
"""
Plot multiple bit signals stacked vertically for visual correlation analysis.

This script loads polar files and displays specified bit/distance bin combinations
stacked for easy visual comparison.

Usage:
    python plot_stacked_bits.py 20220405T1400 20220405T1401 /path/to/POLAR b13_d00 b12_d20
    python plot_stacked_bits.py 20220405T1400 20220405T1401 /path/to/POLAR b13_d00 b12_d20 b14_d19
    python plot_stacked_bits.py ... b13_d00 b12_d20 --output plot.png

Signal format: b{bit}_d{distance_bin}
    e.g., b13_d00 = bit 13 at distance bin 0
          b12_d20 = bit 12 at distance bin 20

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import re
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


def parse_signal_spec(spec: str) -> tuple[int, int]:
    """
    Parse a signal specification like 'b13_d00' into (bit, distance_bin).

    Args:
        spec: Signal specification string

    Returns:
        Tuple of (bit_number, distance_bin)

    Raises:
        ValueError: If spec format is invalid
    """
    match = re.match(r"^b(\d+)_d(\d+)$", spec)
    if not match:
        raise ValueError(
            f"Invalid signal spec '{spec}'. Expected format: b{{bit}}_d{{bin}} (e.g., b13_d00)"
        )

    bit = int(match.group(1))
    dist_bin = int(match.group(2))

    if bit not in (12, 13, 14, 15):
        raise ValueError(f"Invalid bit number {bit}. Must be 12, 13, 14, or 15.")

    return bit, dist_bin


def load_frame_bits(args: tuple) -> tuple:
    """
    Worker function to load a single frame and extract specified bits.

    Args:
        args: Tuple of (filename, signals) where signals is list of (bit, dist_bin)

    Returns:
        Tuple of (timestamp, dict of signal_name -> bit_array) or (None, None) on error
    """
    fn, signals = args
    try:
        frame = PolarFile(fn).frame()
        ts = frame.timestamp

        # Extract each requested signal
        result = {}
        for bit, dist_bin in signals:
            if dist_bin >= frame.n_distances:
                logger.warning(
                    f"Distance bin {dist_bin} >= n_distances {frame.n_distances} in {fn}"
                )
                continue

            if bit == 12:
                data = frame.bit12[:, dist_bin]
            elif bit == 13:
                data = frame.bit13[:, dist_bin]
            elif bit == 14:
                data = frame.bit14[:, dist_bin]
            elif bit == 15:
                data = frame.bit15[:, dist_bin]

            key = f"b{bit}_d{dist_bin:02d}"
            result[key] = data.astype(np.uint8)

        return ts, result, frame.n_bearings
    except Exception as e:
        logger.warning(f"Failed to load {fn}: {e}")
        return None, None, None


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


def main():
    parser = argparse.ArgumentParser(
        description="Plot multiple bit signals stacked for visual correlation analysis"
    )

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument("signals", nargs="+", help="Signal specifications (e.g., b13_d00 b12_d20)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (if not specified, displays interactively)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="16,10",
        help="Figure size as 'width,height' in inches (default: 16,10)",
    )
    parser.add_argument(
        "--no-frame-lines",
        action="store_true",
        help="Don't show vertical lines at frame boundaries",
    )

    args = parser.parse_args()
    setup_logging(args)

    # Parse signal specifications
    try:
        signals = [parse_signal_spec(s) for s in args.signals]
    except ValueError as e:
        logger.error(str(e))
        return 1

    signal_names = [f"b{bit}_d{dist:02d}" for bit, dist in signals]
    logger.info(f"Plotting signals: {', '.join(signal_names)}")

    # Parse figure size
    try:
        figsize = tuple(float(x) for x in args.figsize.split(","))
    except ValueError:
        logger.error(f"Invalid figsize: {args.figsize}")
        return 1

    # Load files
    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in time range")
            return 1

        logger.info(f"Loading {n_files} files with {args.workers or os.cpu_count()} workers")

        # Load frames in parallel
        frames_data = {}
        work_items = [(fn, signals) for fn in files]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame_bits, item): item[0] for item in work_items}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")

                ts, data, n_bearings = future.result()
                if ts is not None and data:
                    frames_data[ts] = (data, n_bearings)
        elapsed = time.perf_counter() - t0

        n_loaded = len(frames_data)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(f"Successfully loaded {n_loaded} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)")

        if not frames_data:
            logger.error("No valid frames loaded")
            return 1

        # Concatenate data in time order
        combined = {name: [] for name in signal_names}
        frame_boundaries = [0]

        for ts in sorted(frames_data):
            data, n_bearings = frames_data[ts]
            for name in signal_names:
                if name in data:
                    combined[name].append(data[name])
                else:
                    # Fill with zeros if signal not available
                    combined[name].append(np.zeros(n_bearings, dtype=np.uint8))
            frame_boundaries.append(frame_boundaries[-1] + n_bearings)

        # Concatenate arrays
        for name in signal_names:
            combined[name] = np.concatenate(combined[name])

        total_samples = len(combined[signal_names[0]])
        print(f"Total samples: {total_samples}")

        # Create plot
        import matplotlib.pyplot as plt

        n_signals = len(signal_names)
        fig, axes = plt.subplots(n_signals, 1, figsize=figsize, sharex=True)

        if n_signals == 1:
            axes = [axes]

        x = np.arange(total_samples)

        for ax, name in zip(axes, signal_names):
            data = combined[name]

            # Plot as filled area
            ax.fill_between(x, data, alpha=0.7)
            ax.set_ylim(-0.1, 1.1)
            ax.set_ylabel(name)
            ax.set_yticks([0, 1])

            # Add frame boundary lines
            if not args.no_frame_lines:
                for boundary in frame_boundaries[1:-1]:
                    ax.axvline(boundary, color="red", linewidth=0.3, alpha=0.5)

        axes[-1].set_xlabel("Sample index")
        axes[0].set_title(f"Bit signals: {args.stime} to {args.etime} ({n_loaded} frames)")

        fig.tight_layout()

        if args.output:
            plt.savefig(args.output, dpi=150)
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
