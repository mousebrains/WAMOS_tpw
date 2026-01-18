#!/usr/bin/env python3
"""
Check that the theta (beam angle) word makes sense frame-to-frame.

This script loads all polar files for a given time range and calculates
statistics on the 12-bit theta value extracted from timing bits (bins 18-20).
Uses run-length encoding to evenly distribute angles within each theta bin.

Features:
  - Extracts 12-bit theta from nibbles at bins 18, 19, 20
  - Uses run-length encoding to find constant-theta segments
  - Evenly distributes angles within each segment
  - Checks frame-to-frame angle continuity
  - Multi-frame context for better edge accuracy

Usage:
    python theta_check.py 20220405 20220406 /path/to/POLAR
    python theta_check.py 20220405 20220406 /path/to/POLAR --tolerance 2.0
    python theta_check.py 20220405 20220406 /path/to/POLAR --per-frame

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
            futures = {
                executor.submit(load_frame, fn, config): fn
                for fn in files
            }

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)
        elapsed = time.perf_counter() - t0

        n_loaded = len(results)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(f"Successfully loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        results.sort( key=lambda x: x["filename"] )
        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
