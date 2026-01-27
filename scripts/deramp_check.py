#!/usr/bin/env python3
"""
Check deramp polynomial statistics across multiple frames.

This script loads all polar files for a given time range and calculates
statistics on deramp polynomial coefficients, providing insight into
the range-dependent intensity correction stability.

Features:
  - Runs full pipeline: PolarFile -> Theta -> Destreak -> Shadow -> Deramp
  - Collects polynomial coefficients from each frame
  - Reports mean, std, min, max for each coefficient
  - Generates plots of coefficient distributions and range profiles

Usage:
    python deramp_check.py 20220405 20220406 /path/to/POLAR
    python deramp_check.py 20220405 20220406 /path/to/POLAR --plot
    python deramp_check.py 20220405 20220406 /path/to/POLAR --plot -o deramp_stats.png

Jan-2026, Pat Welch, pat@mousebrains.com
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
from wamos_tpw.deramp import Deramp  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
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
    Worker function to load a single frame and extract deramp information.

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
        timings["shadow"] = time.perf_counter() - t0
        substeps["shadow"] = shadow.timing

        # Range calculation
        t0 = time.perf_counter()
        rng = Range(frame)
        timings["range"] = time.perf_counter() - t0

        # Deramp - need to copy intensity since Deramp modifies in-place
        t0 = time.perf_counter()
        masked_intensity = shadow.mask(destreaked.intensity)
        pre_deramp_mean = np.nanmean(masked_intensity, axis=0).copy()
        deramp = Deramp(masked_intensity, rng)
        timings["deramp"] = time.perf_counter() - t0

        # Extract polynomial coefficients
        poly_coefs = deramp.polynomial.convert().coef

        result = {
            "filename": fn,
            "timing": timings,
            "substeps": substeps,
            "poly_order": deramp.order,
            "poly_coefs": poly_coefs.copy(),
            "slant_range": rng.slant_range.copy(),
            "pre_deramp_mean": pre_deramp_mean,
            "post_deramp_mean": np.nanmean(deramp.intensity, axis=0).copy(),
            "shadow_thetas": shadow.thetas.copy() if len(shadow.thetas) > 0 else None,
            "theta_bias": shadow.theta_bias,
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
        description="Check deramp polynomial statistics across polar files"
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
        help="Show deramp statistics plots",
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
        default="14,10",
        help="Figure size as 'width,height' in inches (default: 14,10)",
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
            for step in ["polarfile", "frame", "theta", "destreak", "shadow", "range", "deramp"]:
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

        # Extract polynomial coefficients for statistics
        poly_order = results[0]["poly_order"]
        n_coefs = poly_order + 1

        # Collect coefficients - each row is one frame, each column is one coefficient
        all_coefs = np.array([r["poly_coefs"][:n_coefs] for r in results])

        # Collect theta bias
        theta_biases = np.array([r["theta_bias"] for r in results])

        print(f"\n=== Deramp Polynomial Statistics (order={poly_order}) ===")
        print(f"{'Coef':<8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 60)
        for i in range(n_coefs):
            coef_vals = all_coefs[:, i]
            print(
                f"c[{i}]     {np.mean(coef_vals):>12.4g} {np.std(coef_vals):>12.4g} "
                f"{np.min(coef_vals):>12.4g} {np.max(coef_vals):>12.4g}"
            )

        # Theta bias statistics
        if len(theta_biases) > 0:
            bias_stats = compute_stats(theta_biases)
            print("\n=== Theta Bias Statistics ===")
            print(
                f"Bias:         mean={bias_stats.mean:.3f}\u00b0, median={bias_stats.median:.3f}\u00b0, "
                f"std={bias_stats.std:.4f}\u00b0, range=[{bias_stats.min:.3f}\u00b0, {bias_stats.max:.3f}\u00b0]"
            )

        # Generate plots if requested
        if args.plot and n_loaded > 0:
            import matplotlib.pyplot as plt

            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                figsize = (14, 10)

            # Layout: range profiles on top, coefficient histograms below
            # Determine grid size for coefficients (aim for ~3 columns)
            n_hist_cols = min(3, n_coefs)
            n_hist_rows = (n_coefs + n_hist_cols - 1) // n_hist_cols

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(
                1 + n_hist_rows, n_hist_cols, height_ratios=[1.5] + [1] * n_hist_rows
            )

            # Top row: Range profiles (spanning all columns)
            ax_range = fig.add_subplot(gs[0, :])
            slant_range = results[0]["slant_range"]

            # Average pre/post deramp profiles across all frames
            pre_means = np.array([r["pre_deramp_mean"] for r in results])
            post_means = np.array([r["post_deramp_mean"] for r in results])

            pre_avg = np.nanmean(pre_means, axis=0)
            post_avg = np.nanmean(post_means, axis=0)
            pre_std = np.nanstd(pre_means, axis=0)
            post_std = np.nanstd(post_means, axis=0)

            ax_range.plot(slant_range, pre_avg, label="Pre-deramp (mean)", alpha=0.8)
            ax_range.fill_between(
                slant_range,
                pre_avg - pre_std,
                pre_avg + pre_std,
                alpha=0.2,
                label="Pre-deramp (\u00b11\u03c3)",
            )
            ax_range.plot(slant_range, post_avg, label="Post-deramp (mean)", alpha=0.8)
            ax_range.fill_between(
                slant_range,
                post_avg - post_std,
                post_avg + post_std,
                alpha=0.2,
                label="Post-deramp (\u00b11\u03c3)",
            )
            ax_range.set_xlabel("Slant range (m)")
            ax_range.set_ylabel("Mean intensity")
            ax_range.set_title("Range Profiles (averaged over all frames)")
            ax_range.legend(loc="upper right", fontsize=8)
            ax_range.grid(True, alpha=0.3)

            # Coefficient histograms (one per subplot)
            for i in range(n_coefs):
                row = 1 + i // n_hist_cols
                col = i % n_hist_cols
                ax = fig.add_subplot(gs[row, col])

                coef_vals = all_coefs[:, i]
                coef_mean = np.mean(coef_vals)
                coef_std = np.std(coef_vals)

                ax.hist(coef_vals, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.axvline(
                    coef_mean,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Mean: {coef_mean:.4g}",
                )
                ax.set_xlabel(f"c[{i}]")
                ax.set_ylabel("Count")
                ax.set_title(f"c[{i}]: \u03bc={coef_mean:.4g}, \u03c3={coef_std:.4g}")
                ax.grid(True, alpha=0.3)

            fig.suptitle(
                f"Deramp Statistics: {args.stime} to {args.etime} ({n_loaded} frames, order={poly_order})"
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
