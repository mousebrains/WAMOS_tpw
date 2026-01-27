#!/usr/bin/env python3
"""
Check theta (beam angle) statistics across multiple frames.

This script loads all polar files for a given time range and calculates
statistics on theta values including min, max, step size, and range.

Features:
  - Extracts theta angles from 12-bit counter in radar data
  - Reports min, max, step size, initial and final theta for each frame
  - Generates plots of theta statistics vs frame number
  - Reports aggregate statistics across all frames

Usage:
    python theta_check.py 20220405 20220406 /path/to/POLAR
    python theta_check.py 20220405 20220406 /path/to/POLAR --plot
    python theta_check.py 20220405 20220406 /path/to/POLAR --plot -o theta_stats.png

For best performance with free-threaded Python 3.13+:
    python3.13t theta_check.py ... --workers 8

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402

logger = logging.getLogger(__name__)

# Global timing accumulators (thread-safe)
_timing_lock = Lock()
_timing_stats = {
    "polarfile": [],
    "frame": [],
    "theta": [],
}
# Sub-step timing accumulators
_substep_stats: dict[str, dict[str, list[float]]] = {
    "polarfile": {},
    "theta": {},
}


def load_frame(
    fn: str,
    config: Config,
    use_processes: bool = False,
) -> dict | None:
    """
    Worker function to load a single frame and extract theta information.

    Args:
        fn: Filename to load
        config: Configuration object
        use_processes: If True, return timing info (for ProcessPoolExecutor)

    Returns:
        Dictionary with frame info, or None on error.
    """
    try:
        timings = {}
        substeps: dict[str, dict[str, float]] = {}

        # PolarFile loading
        t0 = time.perf_counter()
        pf = PolarFile(fn, config=config)
        timings["polarfile"] = time.perf_counter() - t0
        substeps["polarfile"] = pf.timing

        # Frame extraction
        t0 = time.perf_counter()
        frame = pf.frame()
        timings["frame"] = time.perf_counter() - t0

        # Theta calculation
        t0 = time.perf_counter()
        theta = Theta(frame)
        timings["theta"] = time.perf_counter() - t0
        substeps["theta"] = theta.timing

        # Extract theta statistics
        theta_arr = theta.theta
        theta_min = float(theta_arr.min())
        theta_max = float(theta_arr.max())
        theta_initial = float(theta_arr[0])
        theta_final = float(theta_arr[-1])
        # Estimate step size from differences (median to be robust)
        theta_diffs = np.diff(theta_arr)
        # Handle wrap-around at 360/0
        theta_diffs = np.where(theta_diffs < -180, theta_diffs + 360, theta_diffs)
        theta_diffs = np.where(theta_diffs > 180, theta_diffs - 360, theta_diffs)
        theta_step = (
            float(np.median(theta_diffs[theta_diffs > 0])) if np.any(theta_diffs > 0) else 0.0
        )

        theta_stats = {
            "min": theta_min,
            "max": theta_max,
            "initial": theta_initial,
            "final": theta_final,
            "step": theta_step,
            "n_radials": len(theta_arr),
        }

        if use_processes:
            # Return timing info for aggregation in main process
            return {
                "filename": fn,
                "theta_stats": theta_stats,
                "timings": timings,
                "substeps": substeps,
            }
        else:
            # Accumulate timing stats (thread-safe)
            with _timing_lock:
                for step, t in timings.items():
                    _timing_stats[step].append(t)
                # Accumulate sub-step timings
                for step, sub_timings in substeps.items():
                    for sub_step, sub_time in sub_timings.items():
                        if sub_step not in _substep_stats[step]:
                            _substep_stats[step][sub_step] = []
                        _substep_stats[step][sub_step].append(sub_time)

            return {
                "filename": fn,
                "theta_stats": theta_stats,
            }
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def wrap_to_180(diff: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap angle differences to [-180, 180] range.

    Args:
        diff: Array of theta differences in degrees

    Returns:
        Wrapped differences
    """
    return ((diff + 180) % 360) - 180


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
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()  # Newline when complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check theta (beam theta) continuity across polar files"
    )

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument("--config", type=str, default=None, help="Path to config file")

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
        help=f"Number of workers (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor (avoids GIL)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Tolerance for continuity check in degrees (default: 1.0)",
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="Show per-frame statistics",
    )
    parser.add_argument(
        "--show-gaps",
        action="store_true",
        help="Only show frames with continuity gaps (requires --per-frame)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plot of theta statistics vs frame",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="14,8",
        help="Figure size as 'width,height' in inches (default: 14,8)",
    )

    args = parser.parse_args()
    setup_logging(args)

    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        executor_type = "processes" if args.processes else "threads"
        logger.info(
            "Loading %d files with %d %s",
            n_files,
            args.workers or os.cpu_count(),
            executor_type,
        )

        # Load frames in parallel
        results: list[dict] = []

        config = Config(args.config)
        t0 = time.perf_counter()
        ExecutorClass = ProcessPoolExecutor if args.processes else ThreadPoolExecutor
        with ExecutorClass(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame, fn, config, args.processes): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)

                    # Aggregate timing stats from process results
                    if args.processes and "timings" in result:
                        for step, t in result["timings"].items():
                            _timing_stats[step].append(t)
                        for step, sub_timings in result.get("substeps", {}).items():
                            for sub_step, sub_time in sub_timings.items():
                                if sub_step not in _substep_stats[step]:
                                    _substep_stats[step][sub_step] = []
                                _substep_stats[step][sub_step].append(sub_time)
        elapsed = time.perf_counter() - t0

        n_loaded = len(results)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(
            f"Successfully loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)"
        )

        # Print per-step timing statistics
        if n_loaded > 0:
            print("\n=== Per-Step Processing Statistics ===")
            total_time = sum(sum(times) for times in _timing_stats.values())

            print(f"{'Step':<12} {'Time (ms)':<12} {'Time %':<10}")
            print("-" * 34)
            for step in ["polarfile", "frame", "theta"]:
                times = _timing_stats[step]
                if times:
                    avg_time_ms = np.mean(times) * 1000
                    time_pct = (sum(times) / total_time * 100) if total_time > 0 else 0
                    print(f"{step:<12} {avg_time_ms:<12.3f} {time_pct:<10.1f}")

            # Print sub-step timing breakdown
            print("\n=== Sub-Step Timing Breakdown ===")
            for step in ["polarfile", "theta"]:
                substeps = _substep_stats.get(step, {})
                if substeps:
                    step_total = sum(_timing_stats[step])
                    print(f"\n{step}:")
                    for sub_name, sub_times in sorted(substeps.items()):
                        if sub_times:
                            avg_sub_ms = np.mean(sub_times) * 1000
                            sub_total = sum(sub_times)
                            sub_pct = (sub_total / step_total * 100) if step_total > 0 else 0
                            print(f"  {sub_name:<20} {avg_sub_ms:>8.3f} ms  ({sub_pct:>5.1f}%)")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort results by filename for consistent ordering
        results.sort(key=lambda x: x["filename"])

        # Extract theta statistics arrays
        theta_mins = np.array([r["theta_stats"]["min"] for r in results])
        theta_maxs = np.array([r["theta_stats"]["max"] for r in results])
        theta_initials = np.array([r["theta_stats"]["initial"] for r in results])
        theta_finals = np.array([r["theta_stats"]["final"] for r in results])
        theta_steps = np.array([r["theta_stats"]["step"] for r in results])
        n_radials = np.array([r["theta_stats"]["n_radials"] for r in results])

        # Print theta statistics
        print("\n=== Theta Statistics ===")
        print(f"{'Statistic':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 55)
        print(
            f"{'Min theta':<15} {theta_mins.mean():>10.2f} {theta_mins.std():>10.3f} {theta_mins.min():>10.2f} {theta_mins.max():>10.2f}"
        )
        print(
            f"{'Max theta':<15} {theta_maxs.mean():>10.2f} {theta_maxs.std():>10.3f} {theta_maxs.min():>10.2f} {theta_maxs.max():>10.2f}"
        )
        print(
            f"{'Initial theta':<15} {theta_initials.mean():>10.2f} {theta_initials.std():>10.3f} {theta_initials.min():>10.2f} {theta_initials.max():>10.2f}"
        )
        print(
            f"{'Final theta':<15} {theta_finals.mean():>10.2f} {theta_finals.std():>10.3f} {theta_finals.min():>10.2f} {theta_finals.max():>10.2f}"
        )
        print(
            f"{'Step size':<15} {theta_steps.mean():>10.4f} {theta_steps.std():>10.4f} {theta_steps.min():>10.4f} {theta_steps.max():>10.4f}"
        )
        print(
            f"{'N radials':<15} {n_radials.mean():>10.1f} {n_radials.std():>10.1f} {n_radials.min():>10} {n_radials.max():>10}"
        )

        # Calculate theta range (max - min) per frame
        theta_ranges = theta_maxs - theta_mins
        print(
            f"{'Theta range':<15} {theta_ranges.mean():>10.2f} {theta_ranges.std():>10.3f} {theta_ranges.min():>10.2f} {theta_ranges.max():>10.2f}"
        )

        # Generate plot if requested
        if args.plot or args.output:
            import matplotlib.pyplot as plt

            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                figsize = (14, 8)

            fig, axes = plt.subplots(2, 2, figsize=figsize)
            frame_nums = np.arange(n_loaded)

            # Plot 1: Min and Max theta vs frame
            ax = axes[0, 0]
            ax.plot(frame_nums, theta_mins, "b-", linewidth=0.5, label="Min", alpha=0.7)
            ax.plot(frame_nums, theta_maxs, "r-", linewidth=0.5, label="Max", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Theta (°)")
            ax.set_title("Min/Max Theta vs Frame")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Initial and Final theta vs frame
            ax = axes[0, 1]
            ax.plot(frame_nums, theta_initials, "b-", linewidth=0.5, label="Initial", alpha=0.7)
            ax.plot(frame_nums, theta_finals, "r-", linewidth=0.5, label="Final", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Theta (°)")
            ax.set_title("Initial/Final Theta vs Frame")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Step size vs frame
            ax = axes[1, 0]
            ax.plot(frame_nums, theta_steps, "g-", linewidth=0.5, alpha=0.7)
            ax.axhline(
                theta_steps.mean(),
                color="k",
                linestyle="--",
                linewidth=1,
                label=f"Mean: {theta_steps.mean():.4f}",
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("Step size (°)")
            ax.set_title("Theta Step Size vs Frame")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 4: Theta range vs frame
            ax = axes[1, 1]
            ax.plot(frame_nums, theta_ranges, "m-", linewidth=0.5, alpha=0.7)
            ax.axhline(
                theta_ranges.mean(),
                color="k",
                linestyle="--",
                linewidth=1,
                label=f"Mean: {theta_ranges.mean():.2f}",
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("Range (°)")
            ax.set_title("Theta Range (Max - Min) vs Frame")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig.suptitle(f"Theta Statistics: {args.stime} to {args.etime} ({n_loaded} frames)")
            plt.tight_layout()

            if args.output:
                plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
                print(f"\nSaved plot: {args.output}")
            else:
                plt.show()

        # Report peak memory usage
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
