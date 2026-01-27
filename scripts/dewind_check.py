#!/usr/bin/env python3
"""
Check dewind sinusoidal fit statistics across multiple frames.

This script loads all polar files for a given time range and calculates
statistics on dewind fit parameters (amplitude and phase), providing insight
into the look-angle-dependent intensity correction stability.

Features:
  - Runs full pipeline: PolarFile -> Theta -> Destreak -> Shadow -> Deramp -> Dewind
  - Collects amplitude and phase from each frame's sinusoidal fit
  - Reports mean, std, min, max for fit parameters
  - Generates plots of parameter distributions and theta profiles

Usage:
    python dewind_check.py 20220405 20220406 /path/to/POLAR
    python dewind_check.py 20220405 20220406 /path/to/POLAR --plot
    python dewind_check.py 20220405 20220406 /path/to/POLAR --plot -o dewind_stats.png

Jan-2026, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

# Suppress "Mean of empty slice" warnings from shadow-masked regions
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.deramp import Deramp  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.dewind import Dewind  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.range import Range  # noqa: E402
from wamos_tpw.shadow import Shadow  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402

logger = logging.getLogger(__name__)


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


def load_frame(
    fn: str,
    config: Config,
) -> dict | None:
    """
    Worker function to load a single frame and extract dewind information.

    Args:
        fn: Filename to load
        config: Configuration object

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
        if shadow.theta_bias:  # Only update if there is a bias to apply
            theta.set_bias(shadow.theta_bias)  # Apply shadow theta bias
        timings["shadow"] = time.perf_counter() - t0
        substeps["shadow"] = shadow.timing

        # Range calculation
        t0 = time.perf_counter()
        rng = Range(frame)
        timings["range"] = time.perf_counter() - t0

        # Deramp
        t0 = time.perf_counter()
        masked_intensity = shadow.mask(destreaked.intensity)
        deramp = Deramp(masked_intensity, rng)
        timings["deramp"] = time.perf_counter() - t0

        # Dewind - need to copy intensity since Dewind modifies in-place
        t0 = time.perf_counter()
        # Use Theta's pre-sorted arrays
        sort_idx = theta._sorted_indices
        theta_sorted = theta._sorted_theta
        pre_dewind_raw = np.nanmean(deramp.intensity[sort_idx, :], axis=1)
        dewind = Dewind(deramp.intensity, theta)  # theta already has bias applied via set_bias()
        post_dewind_raw = np.nanmean(dewind.intensity[sort_idx, :], axis=1)
        timings["dewind"] = time.perf_counter() - t0

        # Interpolate to common theta grid (0-360 degrees, 1-degree resolution)
        # Filter out NaN values before interpolation
        theta_grid = np.arange(0, 360, 1.0, dtype=np.float32)
        valid_pre = ~np.isnan(pre_dewind_raw)
        valid_post = ~np.isnan(post_dewind_raw)
        pre_dewind_mean = np.interp(
            theta_grid, theta_sorted[valid_pre], pre_dewind_raw[valid_pre], period=360
        ).astype(np.float32)
        post_dewind_mean = np.interp(
            theta_grid, theta_sorted[valid_post], post_dewind_raw[valid_post], period=360
        ).astype(np.float32)

        # Extract actual wind data from frame metadata (header WINDS and WINDR)
        wind_speed = frame.metadata.wind_speed  # From WINDS header (m/s)
        wind_direction = frame.metadata.wind_direction  # From WINDR header (degrees)

        result = {
            "filename": fn,
            "timing": timings,
            "substeps": substeps,
            "amplitude": dewind.amplitude,
            "phi_degrees": dewind.phi_degrees,
            "pre_dewind_mean": pre_dewind_mean,
            "post_dewind_mean": post_dewind_mean,
            "theta_bias": shadow.theta_bias,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
        }

        return result
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def print_progress(current: int, total: int, width: int = 40, prefix: str = "Progress") -> None:
    """Print a progress bar that updates in place."""
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check dewind sinusoidal fit statistics across polar files"
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
        help="Show dewind statistics plots",
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
            futures = {executor.submit(load_frame, fn, config): fn for fn in files}

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
            "range": [],
            "deramp": [],
            "dewind": [],
        }
        substep_stats: dict[str, dict[str, list[float]]] = {
            "polarfile": {},
            "theta": {},
            "destreak": {},
            "shadow": {},
        }

        for r in results:
            for step, t in r["timing"].items():
                if step in timing_stats:
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
            for step in [
                "polarfile",
                "frame",
                "theta",
                "destreak",
                "shadow",
                "range",
                "deramp",
                "dewind",
            ]:
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

        # Extract dewind parameters for statistics
        amplitudes = np.array([r["amplitude"] for r in results])
        phis = np.array([r["phi_degrees"] for r in results])
        theta_biases = np.array([r["theta_bias"] for r in results])

        print("\n=== Dewind Fit Statistics ===")
        print(f"{'Param':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 60)
        print(
            f"{'Amplitude':<12} {np.mean(amplitudes):>12.4g} {np.std(amplitudes):>12.4g} "
            f"{np.min(amplitudes):>12.4g} {np.max(amplitudes):>12.4g}"
        )
        print(
            f"{'Phi (deg)':<12} {np.mean(phis):>12.4g} {np.std(phis):>12.4g} "
            f"{np.min(phis):>12.4g} {np.max(phis):>12.4g}"
        )

        # Theta bias statistics
        if len(theta_biases) > 0:
            bias_stats = compute_stats(theta_biases)
            print("\n=== Theta Bias Statistics ===")
            print(
                f"Bias:         mean={bias_stats.mean:.3f}\u00b0, median={bias_stats.median:.3f}\u00b0, "
                f"std={bias_stats.std:.4f}\u00b0, range=[{bias_stats.min:.3f}\u00b0, {bias_stats.max:.3f}\u00b0]"
            )

        # Actual wind statistics from header (WINDS and WINDR)
        wind_speeds = np.array([r["wind_speed"] for r in results if r["wind_speed"] is not None])
        wind_dirs = np.array(
            [r["wind_direction"] for r in results if r["wind_direction"] is not None]
        )

        if len(wind_speeds) > 0 or len(wind_dirs) > 0:
            print("\n=== Measured Wind Statistics (from header) ===")
            if len(wind_speeds) > 0:
                ws_stats = compute_stats(wind_speeds)
                print(
                    f"Speed (m/s):  mean={ws_stats.mean:.2f}, median={ws_stats.median:.2f}, "
                    f"std={ws_stats.std:.2f}, range=[{ws_stats.min:.2f}, {ws_stats.max:.2f}]"
                )
            else:
                print("Speed:        No data available")
            if len(wind_dirs) > 0:
                wd_stats = compute_stats(wind_dirs)
                print(
                    f"Direction:    mean={wd_stats.mean:.1f}\u00b0, median={wd_stats.median:.1f}\u00b0, "
                    f"std={wd_stats.std:.2f}\u00b0, range=[{wd_stats.min:.1f}\u00b0, {wd_stats.max:.1f}\u00b0]"
                )
            else:
                print("Direction:    No data available")

        # Generate plots if requested
        if args.plot and n_loaded > 0:
            import matplotlib.pyplot as plt

            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                figsize = (14, 8)

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

            # Top row: Theta profiles (spanning both columns)
            ax_theta = fig.add_subplot(gs[0, :])

            # Common theta grid (same as used in load_frame)
            theta_grid = np.arange(0, 360, 1.0)

            # Average pre/post dewind profiles across all frames
            pre_means = np.array([r["pre_dewind_mean"] for r in results])
            post_means = np.array([r["post_dewind_mean"] for r in results])

            pre_avg = np.nanmean(pre_means, axis=0)
            post_avg = np.nanmean(post_means, axis=0)
            pre_std = np.nanstd(pre_means, axis=0)
            post_std = np.nanstd(post_means, axis=0)

            ax_theta.plot(theta_grid, pre_avg, label="Pre-dewind (mean)", alpha=0.8)
            ax_theta.fill_between(
                theta_grid,
                pre_avg - pre_std,
                pre_avg + pre_std,
                alpha=0.2,
                label="Pre-dewind (\u00b11\u03c3)",
            )
            ax_theta.plot(theta_grid, post_avg, label="Post-dewind (mean)", alpha=0.8)
            ax_theta.fill_between(
                theta_grid,
                post_avg - post_std,
                post_avg + post_std,
                alpha=0.2,
                label="Post-dewind (\u00b11\u03c3)",
            )
            ax_theta.set_xlabel("Theta (degrees)")
            ax_theta.set_ylabel("Mean intensity")
            ax_theta.set_title("Look Angle Profiles (averaged over all frames)")
            ax_theta.legend(loc="upper right", fontsize=8)
            ax_theta.grid(True, alpha=0.3)

            # Add measured wind statistics text box to theta profile plot
            stats_lines = ["Measured Wind (from header):"]
            if len(wind_speeds) > 0:
                ws_mean = np.mean(wind_speeds)
                ws_median = np.median(wind_speeds)
                ws_std = np.std(wind_speeds)
                stats_lines.append(
                    f"  Speed: mean={ws_mean:.1f} m/s, median={ws_median:.1f} m/s, std={ws_std:.2f} m/s"
                )
            else:
                stats_lines.append("  Speed: No data")
            if len(wind_dirs) > 0:
                wd_mean = np.mean(wind_dirs)
                wd_median = np.median(wind_dirs)
                wd_std = np.std(wind_dirs)
                stats_lines.append(
                    f"  Dir: mean={wd_mean:.1f}\u00b0, median={wd_median:.1f}\u00b0, std={wd_std:.2f}\u00b0"
                )
            else:
                stats_lines.append("  Dir: No data")
            stats_text = "\n".join(stats_lines)

            ax_theta.text(
                0.02,
                0.98,
                stats_text,
                transform=ax_theta.transAxes,
                verticalalignment="top",
                fontsize=9,
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Bottom left: Fit Amplitude histogram
            amp_mean = np.mean(amplitudes)
            amp_std = np.std(amplitudes)
            ax_amp = fig.add_subplot(gs[1, 0])
            ax_amp.hist(amplitudes, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax_amp.axvline(
                amp_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {amp_mean:.4g}"
            )
            ax_amp.set_xlabel("Fit Amplitude")
            ax_amp.set_ylabel("Count")
            ax_amp.set_title(f"Dewind Fit Amplitude: \u03bc={amp_mean:.4g}, \u03c3={amp_std:.4g}")
            ax_amp.grid(True, alpha=0.3)

            # Bottom right: Fit Phi histogram
            phi_mean = np.mean(phis)
            phi_std = np.std(phis)
            ax_phi = fig.add_subplot(gs[1, 1])
            ax_phi.hist(phis, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax_phi.axvline(
                phi_mean,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {phi_mean:.4g}\u00b0",
            )
            ax_phi.set_xlabel("Fit Phi (degrees)")
            ax_phi.set_ylabel("Count")
            ax_phi.set_title(
                f"Dewind Fit Phase: \u03bc={phi_mean:.4g}\u00b0, \u03c3={phi_std:.4g}\u00b0"
            )
            ax_phi.grid(True, alpha=0.3)

            fig.suptitle(f"Dewind Statistics: {args.stime} to {args.etime} ({n_loaded} frames)")
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
