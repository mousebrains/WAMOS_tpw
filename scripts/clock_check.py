#!/usr/bin/env python3
"""
Check that the clocking bits make sense frame to frame

This script loads all polar files for a given time range and calculates
statistics on clocking

Usage:
    python clock_check.py 20220405 20220406 /path/to/POLAR

For best performance with free-threaded Python 3.13+:
    python3.13t clock_check.py ... --workers 8

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

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFrame  # noqa: E402
from wamos_tpw.args import add_time_range_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.pps import WamosPPS  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame_bits(fn: str) -> WamosPPS | None:
    """
    Worker function to load a single frame and create WamosPPS object.

    Args:
        fn: Filename to load

    Returns:
        WamosPPS object or None on error.
    """
    try:
        frame = PolarFrame(fn)
        return WamosPPS(frame)
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
    msg = f"\r{prefix}: [{bar}] {current:>{cnt_width}}/{total} ({pct*100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()  # Newline when complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check clocking consistency across polar files"
    )

    add_time_range_arguments(parser)
    add_logging_arguments(parser)

    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--progress",
        action="store_true",
        dest="progress_flag",
        help="Enable progress bar",
    )
    grp.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress_flag",
        help="Disable progress bar",
    )
    parser.set_defaults(progress_flag=True)

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})"
    )
    args = parser.parse_args()
    setup_logging(args)

    # Run analysis
    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)
        logger.info("Loading %d files with %d workers", n_files, args.workers or os.cpu_count())

        # Load frames in parallel using ThreadPoolExecutor
        pps_dict: dict[np.datetime64, WamosPPS] = {}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame_bits, fn): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    pps_dict[result.timestamp] = result

        n_loaded = len(pps_dict)
        print(f"Successfully loaded {n_loaded} frames")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        times = sorted(pps_dict)
        n_frames = len(times)

        # Collect statistics
        gaps = []
        durations = []
        pulse_counts = []
        bearing_counts = []

        prev_time = None
        for index, t0 in enumerate(times):
            LHS = pps_dict[times[index - 1]] if index > 0 else None
            pps = pps_dict[t0]
            RHS = pps_dict[times[index + 1]] if index < (n_frames - 1) else None

            pps.update(LHS, RHS)

            t = pps.time
            duration = (t.max() - t.min()).astype(float) / 1e9
            gap: float | None = None
            if prev_time is not None:
                gap = (t.min() - prev_time).astype(float) / 1e9

            durations.append(duration)
            pulse_counts.append(len(pps.pps_indices))
            bearing_counts.append(pps.n_bearings)
            if gap is not None:
                gaps.append(gap)

            logger.info(
                "# %d at %s n %s has %d PPS pulses, times %s to %s dt %.4fs gap %s",
                index,
                t0.astype("datetime64[ms]"),
                pps.n_bearings,
                len(pps.pps_indices),
                t.min(),
                t.max(),
                duration,
                f"{gap:.6f}s" if gap is not None else None,
            )
            prev_time = t.max()

        # Print summary statistics
        print("\n" + "=" * 60)
        print("CLOCK CHECK SUMMARY")
        print("=" * 60)

        print("\nTime Range:")
        print(f"  First frame: {times[0]}")
        print(f"  Last frame:  {times[-1]}")

        print("\nFrames:")
        print(f"  Total loaded:   {n_frames}")
        print(f"  Bearings/frame: {np.mean(bearing_counts):.1f} mean, "
              f"{np.min(bearing_counts)}-{np.max(bearing_counts)} range")

        print("\nPPS Pulses:")
        print(f"  Total pulses:   {sum(pulse_counts)}")
        print(f"  Pulses/frame:   {np.mean(pulse_counts):.2f} mean, "
              f"{np.min(pulse_counts)}-{np.max(pulse_counts)} range")

        print("\nFrame Durations:")
        print(f"  Mean:  {np.mean(durations):.4f}s")
        print(f"  Std:   {np.std(durations):.6f}s")
        print(f"  Range: {np.min(durations):.4f}s to {np.max(durations):.4f}s")

        if gaps:
            print("\nFrame-to-Frame Gaps:")
            print(f"  Mean:  {np.mean(gaps)*1000:.3f} ms")
            print(f"  Std:   {np.std(gaps)*1000:.3f} ms")
            print(f"  Range: {np.min(gaps)*1000:.3f} to {np.max(gaps)*1000:.3f} ms")

            # Count anomalous gaps
            large_gaps = [g for g in gaps if abs(g) > 0.1]  # > 100ms
            if large_gaps:
                print(f"  Large gaps (>100ms): {len(large_gaps)}")

        print()

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
