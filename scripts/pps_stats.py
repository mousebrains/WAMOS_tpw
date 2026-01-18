#!/usr/bin/env python3
"""
Analyze PPS (Pulse-Per-Second) pulse statistics across multiple polar frames.

This script loads polar files for a given time range and calculates statistics
on the PPS signal (bit 12 at distance bin 0), including:
  - Number of PPS pulses per frame
  - Pulse frequency (pulses per second)
  - Timing gaps between frames
  - Total estimated duration from metadata

Usage:
    python pps_stats.py 20220405 20220406 /path/to/POLAR

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


def load_frame_pps(fn: str) -> dict | None:
    """
    Load a single frame and extract PPS pulse information.

    Args:
        fn: Filename to load

    Returns:
        Dictionary with frame info, or None on error.
    """
    try:
        frame = PolarFile(fn).frame()
        t0 = frame.timestamp.astype("datetime64[ns]")

        rpt = frame.metadata.repeat_time or 0.0
        n_bearings = frame.n_bearings

        # Extract PPS signal from bit 12 at distance bin 0
        pps = frame.bit12[:, 0]
        pps_indices = np.where(pps)[0]
        n_pulses = len(pps_indices)

        # Calculate gaps between consecutive pulses (in radial indices)
        if n_pulses > 1:
            pulse_gaps = np.diff(pps_indices)
            mean_gap = float(np.mean(pulse_gaps))
            std_gap = float(np.std(pulse_gaps))
            min_gap = int(np.min(pulse_gaps))
            max_gap = int(np.max(pulse_gaps))
        else:
            mean_gap = np.nan
            std_gap = np.nan
            min_gap = 0
            max_gap = 0

        return {
            "filename": fn,
            "timestamp": t0,
            "rpt": float(rpt),
            "n_bearings": n_bearings,
            "n_pulses": n_pulses,
            "pps_indices": pps_indices,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
            "min_gap": min_gap,
            "max_gap": max_gap,
        }
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
    parser = argparse.ArgumentParser(description="Analyze PPS pulse statistics across polar files")

    add_common_arguments(parser)
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
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument("--per-frame", action="store_true", help="Show per-frame statistics")

    args = parser.parse_args()
    setup_logging(args)

    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        logger.info("Loading %d files with %d workers", n_files, args.workers or os.cpu_count())

        # Load frames in parallel
        results: list[dict] = []

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame_pps, fn): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)
        elapsed = time.perf_counter() - t0

        n_loaded = len(results)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(f"\nSuccessfully loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort by timestamp
        def sort_key(x: dict) -> np.datetime64:
            ts = x["timestamp"]
            return ts if ts is not None else np.datetime64(0, "ns")

        results.sort(key=sort_key)

        # Calculate aggregate statistics
        all_pulses = [r["n_pulses"] for r in results]
        all_rpts = [r["rpt"] for r in results if r["rpt"] > 0]
        all_bearings = [r["n_bearings"] for r in results]
        all_mean_gaps = [r["mean_gap"] for r in results if not np.isnan(r["mean_gap"])]

        # Calculate timing statistics
        first_ts = results[0]["timestamp"]
        last_ts = results[-1]["timestamp"]
        last_rpt = results[-1]["rpt"]

        if first_ts is not None and last_ts is not None:
            # Total span from first frame start to last frame end
            total_span_ns = (last_ts - first_ts).astype(float)
            total_span_s = total_span_ns / 1e9 + last_rpt
            span_str = f"{total_span_s:.3f}s"
        else:
            total_span_s = np.nan
            span_str = "N/A"

        # Estimated total time from RPT values
        total_rpt = sum(all_rpts)

        # Calculate pulse frequency (pulses per second)
        total_pulses = sum(all_pulses)
        if total_span_s > 0 and not np.isnan(total_span_s):
            pulse_freq = total_pulses / total_span_s
        else:
            pulse_freq = total_pulses / total_rpt if total_rpt > 0 else np.nan

        # Calculate frame gaps (time between consecutive frames)
        frame_gaps = []
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            if prev["timestamp"] is not None and curr["timestamp"] is not None:
                expected_end = prev["timestamp"] + np.timedelta64(int(prev["rpt"] * 1e9), "ns")
                gap_ns = (curr["timestamp"] - expected_end).astype(float)
                frame_gaps.append(gap_ns / 1e9)

        # Print summary
        print("\n" + "=" * 60)
        print("PPS PULSE STATISTICS SUMMARY")
        print("=" * 60)

        print("\nTime Range:")
        print(f"  First frame: {first_ts}")
        print(f"  Last frame:  {last_ts}")
        print(f"  Total span:  {span_str}")
        print(f"  Sum of RPTs: {total_rpt:.3f}s")

        print("\nFrames:")
        print(f"  Total loaded:    {n_loaded}")
        print(
            f"  Bearings/frame:  {np.mean(all_bearings):.1f} mean, "
            f"{np.min(all_bearings)}-{np.max(all_bearings)} range"
        )
        if all_rpts:
            print(
                f"  RPT (seconds):   {np.mean(all_rpts):.4f} mean, "
                f"{np.min(all_rpts):.4f}-{np.max(all_rpts):.4f} range"
            )

        print("\nPPS Pulses:")
        print(f"  Total pulses:    {total_pulses}")
        print(f"  Pulses/frame:    {np.mean(all_pulses):.2f} mean, {np.std(all_pulses):.2f} std")
        print(f"  Pulses range:    {np.min(all_pulses)}-{np.max(all_pulses)}")
        print(f"  Pulse frequency: {pulse_freq:.4f} Hz (expected: 1.0 Hz)")

        if all_mean_gaps:
            print("\nPulse Gaps (radials between consecutive pulses):")
            print(f"  Mean gap:        {np.mean(all_mean_gaps):.2f} radials")
            print(f"  Std gap:         {np.std(all_mean_gaps):.2f} radials")

            # Expected gap based on RPT and bearings
            if all_rpts and all_bearings:
                expected_gap = np.mean(all_bearings) / np.mean(all_rpts)
                print(f"  Expected gap:    {expected_gap:.2f} radials (for 1 Hz PPS)")

        if frame_gaps:
            print("\nFrame Gaps (time between expected end and actual start):")
            print(f"  Mean gap:        {np.mean(frame_gaps) * 1000:.3f} ms")
            print(f"  Std gap:         {np.std(frame_gaps) * 1000:.3f} ms")
            min_gap = np.min(frame_gaps) * 1000
            max_gap = np.max(frame_gaps) * 1000
            print(f"  Range:           {min_gap:.3f} to {max_gap:.3f} ms")

            # Count large gaps (potential missing frames)
            large_gaps = [g for g in frame_gaps if abs(g) > 0.5]  # > 500ms
            if large_gaps:
                print(f"  Large gaps (>500ms): {len(large_gaps)}")

        # Per-frame output if requested
        if args.per_frame:
            print("\n" + "=" * 60)
            print("PER-FRAME DETAILS")
            print("=" * 60)
            print(
                f"{'#':>4} {'Timestamp':>26} {'RPT':>7} {'Bearings':>8} "
                f"{'Pulses':>7} {'MeanGap':>8}"
            )
            print("-" * 65)

            for i, r in enumerate(results):
                ts_str = str(r["timestamp"])[:26] if r["timestamp"] is not None else "N/A"
                gap_str = f"{r['mean_gap']:.1f}" if not np.isnan(r["mean_gap"]) else "N/A"
                print(
                    f"{i:4d} {ts_str:>26} {r['rpt']:7.4f} {r['n_bearings']:8d} "
                    f"{r['n_pulses']:7d} {gap_str:>8}"
                )

        # Report peak memory usage
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == "darwin":
            peak_mem_mb = peak_mem / (1024 * 1024)
        else:
            peak_mem_mb = peak_mem / 1024
        print(f"\nPeak memory: {peak_mem_mb:.1f} MB")

        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
