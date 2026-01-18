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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
) -> dict | None:
    """
    Worker function to load a single frame and extract theta information.

    Args:
        fn: Filename to load
        detect_shadow: Whether to detect shadow regions

    Returns:
        Dictionary with frame info, or None on error.
    """
    try:
        pf = PolarFile(fn, config=config)
        frame = pf.frame()
        theta = Theta(frame)
        destreaked = Destreak(frame)
        shadow = Shadow(destreaked.intensity, theta)
        return {
            "filename": fn,
            "shadow": shadow,
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
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
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
        help="Show plot of theta vs time",
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
    parser.add_argument(
        "--pcolor",
        action="store_true",
        help="Show pcolor plot of intensity vs theta and range",
    )
    parser.add_argument(
        "--n-ranges",
        type=int,
        default=100,
        help="Number of range bins to plot in pcolor (default: 100)",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=0,
        help="Starting range bin for pcolor plot (default: 0)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum intensity value for pcolor colormap",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum intensity value for pcolor colormap",
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

        logger.info(
            "Loading %d files with %d workers",
            n_files,
            args.workers or os.cpu_count(),
        )

        # Load frames in parallel
        results: list[dict] = []

        config = Config(args.config)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
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

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        results.sort(key=lambda x: x["filename"])

        # Extract shadow data for statistics
        shadow_indices_start = []
        shadow_indices_end = []
        shadow_thetas_start = []
        shadow_thetas_end = []
        frame_numbers = []

        for i, r in enumerate(results):
            shadow = r["shadow"]
            if len(shadow.indices) > 0:
                # Take first shadow region (typically the main aft shadow)
                shadow_indices_start.append(shadow.indices[0, 0])
                shadow_indices_end.append(shadow.indices[0, 1])
                shadow_thetas_start.append(shadow.thetas[0, 0])
                shadow_thetas_end.append(shadow.thetas[0, 1])
                frame_numbers.append(i)

        shadow_indices_start = np.array(shadow_indices_start)
        shadow_indices_end = np.array(shadow_indices_end)
        shadow_thetas_start = np.array(shadow_thetas_start)
        shadow_thetas_end = np.array(shadow_thetas_end)
        frame_numbers = np.array(frame_numbers)

        # Calculate statistics
        n_with_shadow = len(shadow_indices_start)
        print(f"\nFrames with shadow detected: {n_with_shadow} of {n_loaded}")

        if n_with_shadow > 0:
            # Index statistics
            idx_start_mean = shadow_indices_start.mean()
            idx_start_std = shadow_indices_start.std()
            idx_start_var = shadow_indices_start.var()
            idx_start_skew = stats.skew(shadow_indices_start)
            idx_start_kurt = stats.kurtosis(shadow_indices_start)
            idx_end_mean = shadow_indices_end.mean()
            idx_end_std = shadow_indices_end.std()
            idx_end_var = shadow_indices_end.var()
            idx_end_skew = stats.skew(shadow_indices_end)
            idx_end_kurt = stats.kurtosis(shadow_indices_end)
            idx_width = shadow_indices_end - shadow_indices_start
            idx_width_mean = idx_width.mean()
            idx_width_std = idx_width.std()

            # Theta statistics
            theta_start_mean = shadow_thetas_start.mean()
            theta_start_std = shadow_thetas_start.std()
            theta_start_var = shadow_thetas_start.var()
            theta_start_skew = stats.skew(shadow_thetas_start)
            theta_start_kurt = stats.kurtosis(shadow_thetas_start)
            theta_end_mean = shadow_thetas_end.mean()
            theta_end_std = shadow_thetas_end.std()
            theta_end_var = shadow_thetas_end.var()
            theta_end_skew = stats.skew(shadow_thetas_end)
            theta_end_kurt = stats.kurtosis(shadow_thetas_end)
            theta_width = shadow_thetas_end - shadow_thetas_start
            theta_width_mean = theta_width.mean()
            theta_width_std = theta_width.std()

            print("\n=== Shadow Index Statistics ===")
            print(
                f"Start index:  mean={idx_start_mean:.1f}, std={idx_start_std:.2f}, "
                f"var={idx_start_var:.2f}, skew={idx_start_skew:.3f}, kurt={idx_start_kurt:.3f}"
            )
            print(
                f"End index:    mean={idx_end_mean:.1f}, std={idx_end_std:.2f}, "
                f"var={idx_end_var:.2f}, skew={idx_end_skew:.3f}, kurt={idx_end_kurt:.3f}"
            )
            print(f"Width:        mean={idx_width_mean:.1f}, std={idx_width_std:.2f}")
            print(
                f"Range:        start=[{shadow_indices_start.min()}, {shadow_indices_start.max()}], "
                f"end=[{shadow_indices_end.min()}, {shadow_indices_end.max()}]"
            )

            print("\n=== Shadow Theta Statistics ===")
            print(
                f"Start theta:  mean={theta_start_mean:.2f}°, std={theta_start_std:.3f}°, "
                f"var={theta_start_var:.4f}, skew={theta_start_skew:.3f}, kurt={theta_start_kurt:.3f}"
            )
            print(
                f"End theta:    mean={theta_end_mean:.2f}°, std={theta_end_std:.3f}°, "
                f"var={theta_end_var:.4f}, skew={theta_end_skew:.3f}, kurt={theta_end_kurt:.3f}"
            )
            print(f"Width:        mean={theta_width_mean:.2f}°, std={theta_width_std:.3f}°")
            print(
                f"Range:        start=[{shadow_thetas_start.min():.2f}°, {shadow_thetas_start.max():.2f}°], "
                f"end=[{shadow_thetas_end.min():.2f}°, {shadow_thetas_end.max():.2f}°]"
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
            idx_start_rel = shadow_indices_start - idx_start_mean
            idx_end_rel = shadow_indices_end - idx_end_mean
            bins_idx = np.linspace(
                min(idx_start_rel.min(), idx_end_rel.min()),
                max(idx_start_rel.max(), idx_end_rel.max()),
                50,
            )
            ax.hist(idx_start_rel, bins=bins_idx, alpha=0.6, label="Start", color="blue")
            ax.hist(idx_end_rel, bins=bins_idx, alpha=0.6, label="End", color="red")
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Index deviation from mean")
            ax.set_ylabel("Count")
            ax.set_title("Shadow Index Distributions (relative to mean)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Histogram of shadow thetas relative to their means (combined start/end)
            ax = axes[0, 1]
            theta_start_rel = shadow_thetas_start - theta_start_mean
            theta_end_rel = shadow_thetas_end - theta_end_mean
            bins_theta = np.linspace(
                min(theta_start_rel.min(), theta_end_rel.min()),
                max(theta_start_rel.max(), theta_end_rel.max()),
                50,
            )
            ax.hist(theta_start_rel, bins=bins_theta, alpha=0.6, label="Start", color="blue")
            ax.hist(theta_end_rel, bins=bins_theta, alpha=0.6, label="End", color="red")
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
