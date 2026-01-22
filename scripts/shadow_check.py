#!/usr/bin/env python3
"""
Check shadow region statistics across multiple frames.

This script loads all polar files for a given time range and calculates
statistics on shadow region detection, including variance in shadow indices
and theta angles.

Features:
  - Detects shadow regions using intensity-based edge detection
  - Calculates variance, skewness, and kurtosis for shadow start/end indices and thetas
  - Generates 4-panel plots: histograms of distributions relative to means, scatter plots
  - Reports mean, std, variance, skewness, kurtosis, and range for all shadow parameters

Usage:
    python shadow_check.py 20220405 20220406 /path/to/POLAR
    python shadow_check.py 20220405 20220406 /path/to/POLAR --plot
    python shadow_check.py 20220405 20220406 /path/to/POLAR --plot -o shadow_stats.png

For best performance with free-threaded Python 3.13+:
    python3.13t shadow_check.py ... --workers 8

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.shadow import Shadow  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402 - Single-frame theta calculator

logger = logging.getLogger(__name__)


def load_frame(
    fn: str,
    config: Config,
    stats_only: bool = False,
) -> dict | None:
    """
    Worker function to load a single frame and extract shadow information.

    Args:
        fn: Filename to load
        config: Configuration object
        stats_only: If True, return only timing and shadow stats (reduces memory)

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

        # Destreak processing
        t0 = time.perf_counter()
        destreaked = Destreak(frame)
        timings["destreak"] = time.perf_counter() - t0
        substeps["destreak"] = destreaked.timing

        # Shadow detection
        t0 = time.perf_counter()
        shadow = Shadow(destreaked.intensity, theta)
        timings["shadow"] = time.perf_counter() - t0
        substeps["shadow"] = shadow.timing

        result = {
            "filename": fn,
            "timing": timings,
            "substeps": substeps,
            # Always include shadow stats (small data)
            "shadow_indices": shadow.indices.copy() if len(shadow.indices) > 0 else None,
            "shadow_thetas": shadow.thetas.copy() if len(shadow.thetas) > 0 else None,
            "theta_bias": shadow.theta_bias,
        }

        # Only store large objects if needed for plotting
        if not stats_only:
            result["shadow"] = shadow

        return result
    except Exception:
        logger.exception("Failed to load %s", fn)
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
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()  # Newline when complete


@dataclass
class Stats:
    """Statistical summary of an array."""

    mean: float
    median: float
    std: float
    var: float
    skew: float
    kurtosis: float
    min: float
    max: float


def compute_stats(arr: np.ndarray) -> Stats:
    """Compute statistical summary of an array."""
    return Stats(
        mean=arr.mean(),
        median=np.median(arr),
        std=arr.std(),
        var=arr.var(),
        skew=stats.skew(arr),
        kurtosis=stats.kurtosis(arr),
        min=arr.min(),
        max=arr.max(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check shadow region statistics across polar files"
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
        "--plot",
        action="store_true",
        help="Show shadow statistics plots",
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
        stats_only = not args.plot

        t0 = time.perf_counter()
        ExecutorClass = ProcessPoolExecutor if args.processes else ThreadPoolExecutor
        with ExecutorClass(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame, fn, config, stats_only): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)
        elapsed = time.perf_counter() - t0

        n_loaded = len(results)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(
            f"Successfully loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)"
        )

        # Aggregate timing statistics from results
        timing_stats: dict[str, list[float]] = {
            "polarfile": [],
            "frame": [],
            "theta": [],
            "destreak": [],
            "shadow": [],
        }
        substep_stats: dict[str, dict[str, list[float]]] = {
            "polarfile": {},
            "theta": {},
            "destreak": {},
            "shadow": {},
        }

        for r in results:
            for step, t in r["timing"].items():
                timing_stats[step].append(t)
            for step, sub_timings in r["substeps"].items():
                for sub_step, sub_time in sub_timings.items():
                    if sub_step not in substep_stats[step]:
                        substep_stats[step][sub_step] = []
                    substep_stats[step][sub_step].append(sub_time)

        # Print per-step timing statistics
        if n_loaded > 0:
            print("\n=== Per-Step Processing Statistics ===")
            total_time = sum(sum(times) for times in timing_stats.values())

            print(f"{'Step':<12} {'Time (ms)':<12} {'Time %':<10}")
            print("-" * 34)
            for step in ["polarfile", "frame", "theta", "destreak", "shadow"]:
                times = timing_stats[step]
                if times:
                    avg_time_ms = np.mean(times) * 1000
                    time_pct = (sum(times) / total_time * 100) if total_time > 0 else 0
                    print(f"{step:<12} {avg_time_ms:<12.3f} {time_pct:<10.1f}")

            # Print sub-step timing breakdown
            print("\n=== Sub-Step Timing Breakdown ===")
            for step in ["polarfile", "theta", "destreak", "shadow"]:
                substeps = substep_stats.get(step, {})
                if substeps:
                    step_total = sum(timing_stats[step])
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

        results.sort(key=lambda x: x["filename"])

        # Extract shadow data for statistics
        shadow_indices_start = []
        shadow_indices_end = []
        shadow_thetas_start = []
        shadow_thetas_end = []
        theta_biases = []
        frame_numbers = []

        for i, r in enumerate(results):
            indices = r["shadow_indices"]
            thetas = r["shadow_thetas"]
            theta_biases.append(r["theta_bias"])
            if indices is not None and len(indices) > 0:
                # Take first shadow region (typically the main aft shadow)
                shadow_indices_start.append(indices[0, 0])
                shadow_indices_end.append(indices[0, 1])
                shadow_thetas_start.append(thetas[0, 0])
                shadow_thetas_end.append(thetas[0, 1])
                frame_numbers.append(i)

        shadow_indices_start = np.array(shadow_indices_start)
        shadow_indices_end = np.array(shadow_indices_end)
        shadow_thetas_start = np.array(shadow_thetas_start)
        shadow_thetas_end = np.array(shadow_thetas_end)
        theta_biases = np.array(theta_biases)
        frame_numbers = np.array(frame_numbers)

        # Calculate statistics
        n_with_shadow = len(shadow_indices_start)
        print(f"\nFrames with shadow detected: {n_with_shadow} of {n_loaded}")

        if n_with_shadow > 0:
            # Compute statistics using helper function
            idx_start = compute_stats(shadow_indices_start)
            idx_end = compute_stats(shadow_indices_end)
            idx_width = compute_stats(shadow_indices_end - shadow_indices_start)
            theta_start = compute_stats(shadow_thetas_start)
            theta_end = compute_stats(shadow_thetas_end)
            theta_width = compute_stats(shadow_thetas_end - shadow_thetas_start)

            print("\n=== Shadow Index Statistics ===")
            print(
                f"Start index:  mean={idx_start.mean:.1f}, median={idx_start.median:.1f}, "
                f"std={idx_start.std:.2f}, var={idx_start.var:.2f}, "
                f"skew={idx_start.skew:.3f}, kurt={idx_start.kurtosis:.3f}"
            )
            print(
                f"End index:    mean={idx_end.mean:.1f}, median={idx_end.median:.1f}, "
                f"std={idx_end.std:.2f}, var={idx_end.var:.2f}, "
                f"skew={idx_end.skew:.3f}, kurt={idx_end.kurtosis:.3f}"
            )
            print(
                f"Width:        mean={idx_width.mean:.1f}, median={idx_width.median:.1f}, "
                f"std={idx_width.std:.2f}"
            )
            print(
                f"Range:        start=[{idx_start.min:.0f}, {idx_start.max:.0f}], "
                f"end=[{idx_end.min:.0f}, {idx_end.max:.0f}]"
            )

            print("\n=== Shadow Theta Statistics ===")
            print(
                f"Start theta:  mean={theta_start.mean:.2f}°, median={theta_start.median:.2f}°, "
                f"std={theta_start.std:.3f}°, var={theta_start.var:.4f}, "
                f"skew={theta_start.skew:.3f}, kurt={theta_start.kurtosis:.3f}"
            )
            print(
                f"End theta:    mean={theta_end.mean:.2f}°, median={theta_end.median:.2f}°, "
                f"std={theta_end.std:.3f}°, var={theta_end.var:.4f}, "
                f"skew={theta_end.skew:.3f}, kurt={theta_end.kurtosis:.3f}"
            )
            print(
                f"Width:        mean={theta_width.mean:.2f}°, median={theta_width.median:.2f}°, "
                f"std={theta_width.std:.3f}°"
            )
            print(
                f"Range:        start=[{theta_start.min:.2f}°, {theta_start.max:.2f}°], "
                f"end=[{theta_end.min:.2f}°, {theta_end.max:.2f}°]"
            )

        # Theta bias statistics (collected from all frames, not just those with shadow)
        if len(theta_biases) > 0:
            bias_stats = compute_stats(theta_biases)
            print("\n=== Theta Bias Statistics ===")
            print(
                f"Bias:         mean={bias_stats.mean:.3f}°, median={bias_stats.median:.3f}°, "
                f"std={bias_stats.std:.4f}°, range=[{bias_stats.min:.3f}°, {bias_stats.max:.3f}°]"
            )

        # Generate scatter plot if requested
        if args.plot and n_with_shadow > 0:
            import matplotlib.pyplot as plt

            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                figsize = (14, 8)

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # Plot 1: Histogram of shadow indices relative to their means (combined start/end)
            ax = axes[0, 0]
            idx_start_rel = shadow_indices_start - idx_start.mean
            idx_end_rel = shadow_indices_end - idx_end.mean
            bins_idx = np.linspace(
                min(idx_start_rel.min(), idx_end_rel.min()),
                max(idx_start_rel.max(), idx_end_rel.max()),
                50,
            )
            ax.hist(
                idx_start_rel,
                bins=bins_idx,
                histtype="step",
                label="Start",
                color="blue",
                linewidth=1.5,
            )
            ax.hist(
                idx_end_rel, bins=bins_idx, histtype="step", label="End", color="red", linewidth=1.5
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Index deviation from mean")
            ax.set_ylabel("Count")
            ax.set_title("Shadow Index Distributions (relative to mean)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Histogram of shadow thetas relative to their means (combined start/end)
            ax = axes[0, 1]
            theta_start_rel = shadow_thetas_start - theta_start.mean
            theta_end_rel = shadow_thetas_end - theta_end.mean
            bins_theta = np.linspace(
                min(theta_start_rel.min(), theta_end_rel.min()),
                max(theta_start_rel.max(), theta_end_rel.max()),
                50,
            )
            ax.hist(
                theta_start_rel,
                bins=bins_theta,
                histtype="step",
                label="Start",
                color="blue",
                linewidth=1.5,
            )
            ax.hist(
                theta_end_rel,
                bins=bins_theta,
                histtype="step",
                label="End",
                color="red",
                linewidth=1.5,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Theta deviation from mean (°)")
            ax.set_ylabel("Count")
            ax.set_title("Shadow Theta Distributions (relative to mean)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Start index vs Start theta (scatter)
            ax = axes[1, 0]
            sc = ax.scatter(
                shadow_thetas_start,
                shadow_indices_start,
                s=5,
                alpha=0.5,
                c=frame_numbers,
                cmap="viridis",
            )
            ax.set_xlabel("Start theta (°)")
            ax.set_ylabel("Start index")
            ax.set_title("Start Index vs Start Theta")
            ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label="Frame")

            # Plot 4: End index vs End theta (scatter)
            ax = axes[1, 1]
            sc = ax.scatter(
                shadow_thetas_end,
                shadow_indices_end,
                s=5,
                alpha=0.5,
                c=frame_numbers,
                cmap="viridis",
            )
            ax.set_xlabel("End theta (°)")
            ax.set_ylabel("End index")
            ax.set_title("End Index vs End Theta")
            ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label="Frame")

            fig.suptitle(
                f"Shadow Statistics: {args.stime} to {args.etime} ({n_with_shadow} frames)"
            )
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
