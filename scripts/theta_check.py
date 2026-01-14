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
from wamos_tpw.destreak import DestreakFrame  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.pps import WamosPPS  # noqa: E402
from wamos_tpw.theta_calc import WamosTheta  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame_theta(
    fn: str,
    destreak_config: dict | None = None,
    shadow_config: dict | None = None,
) -> dict | None:
    """
    Worker function to load a single frame and extract theta information.

    Args:
        fn: Filename to load
        destreak_config: Optional destreak configuration dict
        shadow_config: Optional shadow detection configuration dict

    Returns:
        Dictionary with frame info, or None on error.
    """
    try:
        polar_frame = PolarFrame(fn)

        # Apply destreaking
        frame = DestreakFrame(polar_frame, destreak_config)

        # Create WamosPPS from frame
        pps_obj = WamosPPS(frame)

        # Create WamosTheta object from frame (with optional shadow detection)
        wtheta = WamosTheta(frame, shadow_config=shadow_config)

        return {
            "filename": fn,
            "timestamp": pps_obj.timestamp,
            "rpt": pps_obj.rpt,
            "n_bearings": frame.n_bearings,
            "n_ranges": frame.n_ranges,
            "intensity": frame.intensity,  # Destreaked intensity
            "original_intensity": frame.original_intensity,  # Original intensity
            "n_streak_pixels": frame.n_streak_pixels,
            "streak_fraction": frame.streak_fraction,
            "wtheta": wtheta,
            "wpps": pps_obj,
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
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct*100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()  # Newline when complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check theta (beam theta) continuity across polar files"
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
        "--output", "-o",
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
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Detect and plot radar shadow region (leading and trailing edges)",
    )
    parser.add_argument(
        "--shadow-search",
        type=str,
        default="140-220",
        help="Theta range to search for shadow region (default: 140-220)",
    )
    parser.add_argument(
        "--shadow-bin-size",
        type=float,
        default=0.5,
        help="Theta bin size in degrees for shadow detection (default: 0.5)",
    )
    parser.add_argument(
        "--shadow-min-width",
        type=float,
        default=50.0,
        help="Minimum shadow width in degrees to accept (default: 50.0)",
    )
    parser.add_argument(
        "--destreak-min-length",
        type=int,
        default=4,
        help="Minimum consecutive streak length along radial (default: 4)",
    )
    parser.add_argument(
        "--destreak-k-sigma",
        type=float,
        default=5.0,
        help="Number of MAD units above median to flag as streak (default: 5.0)",
    )
    parser.add_argument(
        "--destreak-neighbor-size",
        type=int,
        default=5,
        help="Number of neighbors for local statistics (default: 5)",
    )
    parser.add_argument(
        "--adjust",
        type=str,
        choices=["none", "shift", "shift_scale"],
        default="none",
        help="Theta adjustment mode: none, shift, or shift_scale (default: none)",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Show histogram of shadow edges with Gaussian fits (requires --shadow)",
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

        # Build destreak configuration
        destreak_config = {
            "min_length": args.destreak_min_length,
            "k_sigma": args.destreak_k_sigma,
            "neighbor_size": args.destreak_neighbor_size,
        }

        # Build shadow configuration if shadow detection is enabled
        shadow_config: dict | None = None
        if args.shadow or args.adjust != "none":
            # Parse shadow search range
            try:
                lo, hi = args.shadow_search.strip().split("-")
                search_min = float(lo)
                search_max = float(hi)
            except ValueError:
                logger.error("Invalid shadow-search format: %s", args.shadow_search)
                return 1

            shadow_config = {
                "search_min": search_min,
                "search_max": search_max,
                "bin_size": args.shadow_bin_size,
                "min_width": args.shadow_min_width,
            }

        # Load frames in parallel
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(load_frame_theta, fn, destreak_config, shadow_config): fn
                for fn in files
            }

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)

        n_loaded = len(results)
        print(f"Successfully loaded {n_loaded} of {n_files} frames")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort by timestamp
        def sort_key(x: dict) -> np.datetime64:
            ts = x["timestamp"]
            return ts if ts is not None else np.datetime64(0, "ns")

        results.sort(key=sort_key)

        # Update theta and PPS objects with adjacent frame context
        print("Updating theta and PPS calculations with multi-frame context...")
        for i, curr in enumerate(results):
            prev = results[i - 1] if i > 0 else None
            nxt = results[i + 1] if i < len(results) - 1 else None

            # Update WamosPPS with adjacent PPS objects
            curr["wpps"].update(
                prev["wpps"] if prev else None,
                nxt["wpps"] if nxt else None,
            )

        # Check continuity between consecutive frames
        gaps = []
        discontinuities = []

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            # Calculate gap between frames
            prev_last = prev["wtheta"].last_theta
            curr_first = curr["wtheta"].first_theta
            gap = curr_first - prev_last
            # Wrap to [-180, 180]
            if gap > 180:
                gap -= 360
            elif gap < -180:
                gap += 360

            gaps.append(gap)

            if abs(gap) > args.tolerance:
                discontinuities.append({
                    "index": i,
                    "gap": gap,
                    "prev_ts": prev["timestamp"],
                    "curr_ts": curr["timestamp"],
                    "prev_last": prev_last,
                    "curr_first": curr_first,
                })

        # Calculate aggregate statistics
        all_spans = [r["wtheta"].theta_span for r in results]
        all_first = [r["wtheta"].first_theta for r in results]
        all_last = [r["wtheta"].last_theta for r in results]

        # Calculate angle steps (theta difference between consecutive radials)
        all_angle_steps: list[np.ndarray] = []
        for r in results:
            theta = r["wtheta"].theta
            # Compute differences, handling wraparound
            dtheta = np.diff(theta)
            # Wrap to [-180, 180] to handle 360->0 transition
            dtheta = ((dtheta + 180) % 360) - 180
            all_angle_steps.append(dtheta)

        # Concatenate all angle steps for global statistics
        all_steps_concat = np.concatenate(all_angle_steps)

        # Per-frame angle step statistics
        mean_steps = [float(np.mean(s)) for s in all_angle_steps]
        std_steps = [float(np.std(s)) for s in all_angle_steps]
        min_steps = [float(np.min(s)) for s in all_angle_steps]
        max_steps = [float(np.max(s)) for s in all_angle_steps]

        # Print summary
        print("\n" + "=" * 70)
        print("THETA CONTINUITY CHECK SUMMARY")
        print("=" * 70)

        first_ts = results[0]["timestamp"]
        last_ts = results[-1]["timestamp"]

        print("\nTime Range:")
        print(f"  First frame: {first_ts}")
        print(f"  Last frame:  {last_ts}")

        print("\nFrames:")
        print(f"  Total loaded:    {n_loaded}")

        # Destreak statistics
        all_streak_pixels = [r["n_streak_pixels"] for r in results]
        all_streak_fractions = [r["streak_fraction"] * 100 for r in results]
        total_streak_pixels = sum(all_streak_pixels)
        mean_streak_pct = np.mean(all_streak_fractions)
        print("\nDestreaking:")
        print(f"  Total streak pixels: {total_streak_pixels}")
        print(f"  Mean streak %:       {mean_streak_pct:.3f}%")

        # Statistics table
        print("\n" + "-" * 82)
        print(
            f"{'Metric':^12} | {'Min':>10} | {'Max':>10} | "
            f"{'Mean':>10} | {'Median':>10} | {'Std':>10}"
        )
        print("-" * 82)

        def fmt_row(label: str, arr: list) -> str:
            a = np.array(arr)
            return (
                f"{label:^12} | {np.min(a):>10.2f} | {np.max(a):>10.2f} | "
                f"{np.mean(a):>10.2f} | {np.median(a):>10.2f} | {np.std(a):>10.2f}"
            )

        print(fmt_row("First theta", all_first))
        print(fmt_row("Last theta", all_last))
        print(fmt_row("Angle span", all_spans))
        print(fmt_row("Step mean", mean_steps))
        print(fmt_row("Step std", std_steps))
        print(fmt_row("Step min", min_steps))
        print(fmt_row("Step max", max_steps))
        if gaps:
            print(fmt_row("Frame gaps", gaps))
        print("-" * 82)

        # Global angle step statistics
        print("\nAngle Steps (per-radial theta differences):")
        print(f"  Total steps:     {len(all_steps_concat)}")
        print(f"  Mean:            {np.mean(all_steps_concat):.4f}°")
        print(f"  Median:          {np.median(all_steps_concat):.4f}°")
        print(f"  Std:             {np.std(all_steps_concat):.4f}°")
        print(f"  Min:             {np.min(all_steps_concat):.4f}°")
        print(f"  Max:             {np.max(all_steps_concat):.4f}°")

        # Count anomalous steps (significantly different from median)
        median_step = np.median(all_steps_concat)
        step_threshold = 3 * np.std(all_steps_concat)
        anomalous_steps = np.abs(all_steps_concat - median_step) > step_threshold
        n_anomalous = int(np.sum(anomalous_steps))
        if n_anomalous > 0:
            pct = 100 * n_anomalous / len(all_steps_concat)
            print(f"  Anomalous (>3σ): {n_anomalous} ({pct:.2f}%)")

        print("\nContinuity:")
        print(f"  Tolerance:       {args.tolerance}°")
        print(f"  Total gaps:      {len(gaps)}")
        if gaps:
            gaps_arr = np.array(gaps)
            print(f"  Gap mean:        {np.mean(gaps_arr):.4f}°")
            print(f"  Gap std:         {np.std(gaps_arr):.4f}°")
            print(f"  Gap range:       {np.min(gaps_arr):.4f}° to "
                  f"{np.max(gaps_arr):.4f}°")
        print(f"  Discontinuities: {len(discontinuities)} "
              f"(|gap| > {args.tolerance}°)")

        if discontinuities:
            print("\nDiscontinuities:")
            print("-" * 70)
            hdr = f"{'#':>4} {'Timestamp':>26} {'Gap':>10} {'Prev':>10} {'Curr':>10}"
            print(hdr)
            print("-" * 70)
            for d in discontinuities[:20]:
                ts_str = (
                    str(d["curr_ts"])[:26]
                    if d["curr_ts"] is not None
                    else "N/A"
                )
                print(
                    f"{d['index']:4d} {ts_str:>26} {d['gap']:>10.4f}° "
                    f"{d['prev_last']:>10.4f}° {d['curr_first']:>10.4f}°"
                )
            if len(discontinuities) > 20:
                print(f"  ... and {len(discontinuities) - 20} more")

        # Calculate shadow region if requested (per-frame and global)
        shadow_region: tuple[float | None, float | None] = (None, None)
        if args.shadow:
            from wamos_tpw.shadow import ShadowDetector

            # Shadow detection was already done during WamosTheta initialization
            # Extract per-frame shadow regions from wtheta
            for r in results:
                wtheta = r["wtheta"]
                sr = wtheta.shadow_region
                r["shadow_region"] = (sr.leading, sr.trailing)

            # Calculate global shadow region (all frames combined)
            all_thetas_shadow = []
            all_intensities_shadow = []

            for r in results:
                wtheta = r["wtheta"]
                intensity = r["intensity"]

                thetas = wtheta.theta
                all_thetas_shadow.append(thetas)
                all_intensities_shadow.append(intensity)

            if all_thetas_shadow:
                thetas_concat = np.concatenate(all_thetas_shadow)
                intens_concat = np.concatenate(all_intensities_shadow, axis=0)

                # Use ShadowDetector for global calculation
                assert shadow_config is not None
                detector = ShadowDetector(
                    search_min=shadow_config["search_min"],
                    search_max=shadow_config["search_max"],
                    bin_size=shadow_config["bin_size"],
                    min_width=shadow_config["min_width"],
                )
                global_shadow = detector.detect(thetas_concat, intens_concat)
                shadow_region = (global_shadow.leading, global_shadow.trailing)

                leading, trailing = shadow_region
                print("\nShadow Region (global):")
                if leading is not None:
                    print(f"  Leading edge:  {leading:.1f}°")
                else:
                    print("  Leading edge:  not detected")
                if trailing is not None:
                    print(f"  Trailing edge: {trailing:.1f}°")
                else:
                    print("  Trailing edge: not detected")
                if leading is not None and trailing is not None:
                    print(f"  Shadow span:   {trailing - leading:.1f}°")

            # Calculate per-frame shadow region statistics
            leading_edges: list[float] = []
            trailing_edges: list[float] = []

            for r in results:
                region = r.get("shadow_region", (None, None))
                if region[0] is not None:
                    leading_edges.append(region[0])
                if region[1] is not None:
                    trailing_edges.append(region[1])

            # Print statistics
            from scipy import stats as sp_stats

            print("\nShadow Region Statistics (per-frame):")
            print("-" * 106)
            hdr = (
                f"{'Edge':^15} | {'Count':>6} | {'Min':>8} | {'Max':>8} | "
                f"{'Mean':>8} | {'Median':>8} | {'Std':>8} | {'Skew':>7} | {'Kurt':>7}"
            )
            print(hdr)
            print("-" * 106)

            for name, edges in [("Leading", leading_edges), ("Trailing", trailing_edges)]:
                if edges:
                    arr = np.array(edges)
                    skew = float(sp_stats.skew(arr))
                    kurt = float(sp_stats.kurtosis(arr))
                    print(
                        f"{name:^15} | {len(edges):>6} | {np.min(arr):>8.2f} | "
                        f"{np.max(arr):>8.2f} | {np.mean(arr):>8.2f} | "
                        f"{np.median(arr):>8.2f} | {np.std(arr):>8.2f} | "
                        f"{skew:>7.3f} | {kurt:>7.3f}"
                    )
                else:
                    print(
                        f"{name:^15} | {0:>6} |      N/A |      N/A |"
                        f"      N/A |      N/A |      N/A |     N/A |     N/A"
                    )
            print("-" * 106)

            # Apply theta adjustment if requested
            if args.adjust != "none" and leading_edges and trailing_edges:
                # Compute median shadow edges as targets
                median_leading = float(np.median(leading_edges))
                median_trailing = float(np.median(trailing_edges))

                print(f"\nTheta Adjustment Mode: {args.adjust}")
                print(f"  Target leading edge:  {median_leading:.2f}°")
                print(f"  Target trailing edge: {median_trailing:.2f}°")

                # Apply adjustment to each frame with valid shadow detection
                shifts: list[float] = []
                scales: list[float] = []
                n_adjusted = 0

                for r in results:
                    wtheta = r["wtheta"]

                    if wtheta.has_shadow:
                        wtheta.adjust_to_shadow(
                            target_leading=median_leading,
                            target_trailing=median_trailing,
                            mode=args.adjust,
                        )

                        shifts.append(wtheta.shift)
                        scales.append(wtheta.scale)
                        n_adjusted += 1

                # Print adjustment statistics
                print(f"\n  Frames adjusted: {n_adjusted} of {len(results)}")

                if shifts:
                    from scipy import stats as sp_stats

                    shifts_arr = np.array(shifts)
                    shift_skew = float(sp_stats.skew(shifts_arr))
                    shift_kurt = float(sp_stats.kurtosis(shifts_arr))
                    print("\n  Shift Statistics (degrees):")
                    print(f"    Min:      {np.min(shifts_arr):>10.4f}°")
                    print(f"    Max:      {np.max(shifts_arr):>10.4f}°")
                    print(f"    Mean:     {np.mean(shifts_arr):>10.4f}°")
                    print(f"    Median:   {np.median(shifts_arr):>10.4f}°")
                    print(f"    Std:      {np.std(shifts_arr):>10.4f}°")
                    print(f"    Skewness: {shift_skew:>10.4f}")
                    print(f"    Kurtosis: {shift_kurt:>10.4f}")

                if scales and args.adjust == "shift_scale":
                    scales_arr = np.array(scales)
                    scale_skew = float(sp_stats.skew(scales_arr))
                    scale_kurt = float(sp_stats.kurtosis(scales_arr))
                    print("\n  Scale Statistics (factor):")
                    print(f"    Min:      {np.min(scales_arr):>10.6f}")
                    print(f"    Max:      {np.max(scales_arr):>10.6f}")
                    print(f"    Mean:     {np.mean(scales_arr):>10.6f}")
                    print(f"    Median:   {np.median(scales_arr):>10.6f}")
                    print(f"    Std:      {np.std(scales_arr):>10.6f}")
                    print(f"    Skewness: {scale_skew:>10.4f}")
                    print(f"    Kurtosis: {scale_kurt:>10.4f}")

                # Recalculate angle statistics after adjustment
                print("\nAdjusted Theta Statistics:")
                adj_spans = [r["wtheta"].theta_span for r in results if r["wtheta"].is_adjusted]
                adj_first = [r["wtheta"].first_theta for r in results if r["wtheta"].is_adjusted]
                adj_last = [r["wtheta"].last_theta for r in results if r["wtheta"].is_adjusted]

                if adj_spans:
                    print("-" * 82)
                    print(
                        f"{'Metric':^12} | {'Min':>10} | {'Max':>10} | "
                        f"{'Mean':>10} | {'Median':>10} | {'Std':>10}"
                    )
                    print("-" * 82)
                    print(fmt_row("First theta", adj_first))
                    print(fmt_row("Last theta", adj_last))
                    print(fmt_row("Angle span", adj_spans))
                    print("-" * 82)

                    # Recalculate angle steps for adjusted frames
                    adj_angle_steps: list[np.ndarray] = []
                    for r in results:
                        if r["wtheta"].is_adjusted:
                            theta = r["wtheta"].theta
                            dtheta = np.diff(theta)
                            dtheta = ((dtheta + 180) % 360) - 180
                            adj_angle_steps.append(dtheta)

                    if adj_angle_steps:
                        adj_steps_concat = np.concatenate(adj_angle_steps)
                        print("\nAdjusted Angle Steps:")
                        print(f"  Total steps: {len(adj_steps_concat)}")
                        print(f"  Mean:        {np.mean(adj_steps_concat):.4f}°")
                        print(f"  Median:      {np.median(adj_steps_concat):.4f}°")
                        print(f"  Std:         {np.std(adj_steps_concat):.4f}°")
                        print(f"  Min:         {np.min(adj_steps_concat):.4f}°")
                        print(f"  Max:         {np.max(adj_steps_concat):.4f}°")

            # Plot histogram of shadow edges with Gaussian fits
            if args.hist and leading_edges and trailing_edges:
                import matplotlib.pyplot as plt
                from scipy import stats

                # Parse figure size
                try:
                    figsize = tuple(float(x) for x in args.figsize.split(","))
                except ValueError:
                    logger.error("Invalid figsize: %s", args.figsize)
                    return 1

                fig, ax = plt.subplots(figsize=figsize)

                # Calculate adjusted edges if adjustment was applied
                adj_leading_edges: list[float] = []
                adj_trailing_edges: list[float] = []

                if args.adjust != "none":
                    for r in results:
                        region = r.get("shadow_region", (None, None))
                        wtheta = r["wtheta"]

                        if region[0] is not None and region[1] is not None:
                            orig_leading = region[0]
                            orig_trailing = region[1]

                            if args.adjust == "shift":
                                # Adjusted = original + shift
                                adj_leading_edges.append(orig_leading + wtheta.shift)
                                adj_trailing_edges.append(orig_trailing + wtheta.shift)
                            elif args.adjust == "shift_scale":
                                # Adjusted = (original - frame_leading) * scale + target_leading
                                adj_lead = (
                                    (orig_leading - orig_leading) * wtheta.scale
                                    + median_leading
                                )
                                adj_trail = (
                                    (orig_trailing - orig_leading) * wtheta.scale
                                    + median_leading
                                )
                                adj_leading_edges.append(adj_lead)
                                adj_trailing_edges.append(adj_trail)

                # Determine bin edges to cover all data
                all_edges = leading_edges + trailing_edges
                if adj_leading_edges:
                    all_edges = all_edges + adj_leading_edges + adj_trailing_edges
                bin_min = min(all_edges) - 2
                bin_max = max(all_edges) + 2
                bins: list[float] = list(np.linspace(bin_min, bin_max, 50))

                # Function to plot histogram with Gaussian fit
                def plot_hist_with_gaussian(
                    data: list[float],
                    color: str,
                    label: str,
                    linestyle: str = "-",
                    alpha: float = 0.5,
                ) -> tuple[float, float, float, float]:
                    """Plot histogram with Gaussian fit, return (mu, sigma, skew, kurt)."""
                    arr = np.array(data)
                    # Plot histogram
                    counts, bin_edges, _ = ax.hist(
                        arr,
                        bins=bins,
                        alpha=alpha,
                        color=color,
                        label=label,
                        edgecolor="black",
                        linewidth=0.5,
                    )

                    # Fit Gaussian and compute statistics
                    mu, sigma = stats.norm.fit(arr)
                    skew = float(stats.skew(arr))
                    kurt = float(stats.kurtosis(arr))

                    # Plot Gaussian fit
                    x = np.linspace(bin_min, bin_max, 200)
                    bin_width = bins[1] - bins[0]
                    scale_factor = len(arr) * bin_width
                    pdf = stats.norm.pdf(x, mu, sigma) * scale_factor
                    ax.plot(
                        x,
                        pdf,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                        label=(
                            f"{label} fit: μ={mu:.2f}°, σ={sigma:.2f}°, "
                            f"skew={skew:.3f}, kurt={kurt:.3f}"
                        ),
                    )
                    return mu, sigma, skew, kurt

                # Plot original edges
                _ = plot_hist_with_gaussian(
                    leading_edges, "blue", "Original Leading", alpha=0.4
                )
                _ = plot_hist_with_gaussian(
                    trailing_edges, "red", "Original Trailing", alpha=0.4
                )

                # Plot adjusted edges if available
                if adj_leading_edges and adj_trailing_edges:
                    _ = plot_hist_with_gaussian(
                        adj_leading_edges,
                        "cyan",
                        "Adjusted Leading",
                        linestyle="--",
                        alpha=0.3,
                    )
                    _ = plot_hist_with_gaussian(
                        adj_trailing_edges,
                        "orange",
                        "Adjusted Trailing",
                        linestyle="--",
                        alpha=0.3,
                    )

                ax.set_xlabel("Theta (degrees)")
                ax.set_ylabel("Count")
                ax.legend(loc="upper right", fontsize=7)

                # Add shift/scale statistics as text box if adjustment was applied
                if args.adjust != "none" and shifts and scales:
                    shifts_arr = np.array(shifts)
                    scales_arr = np.array(scales)

                    shift_skew = float(stats.skew(shifts_arr))
                    shift_kurt = float(stats.kurtosis(shifts_arr))
                    scale_skew = float(stats.skew(scales_arr))
                    scale_kurt = float(stats.kurtosis(scales_arr))

                    stats_text = (
                        f"Shift Statistics (n={len(shifts)}):\n"
                        f"  μ={np.mean(shifts_arr):.3f}°, "
                        f"σ={np.std(shifts_arr):.3f}°\n"
                        f"  skew={shift_skew:.3f}, kurt={shift_kurt:.3f}\n"
                        f"  range=[{np.min(shifts_arr):.3f}°, "
                        f"{np.max(shifts_arr):.3f}°]"
                    )

                    if args.adjust == "shift_scale":
                        stats_text += (
                            f"\n\nScale Statistics (n={len(scales)}):\n"
                            f"  μ={np.mean(scales_arr):.6f}, "
                            f"σ={np.std(scales_arr):.6f}\n"
                            f"  skew={scale_skew:.3f}, kurt={scale_kurt:.3f}\n"
                            f"  range=[{np.min(scales_arr):.6f}, "
                            f"{np.max(scales_arr):.6f}]"
                        )

                    # Add text box in upper left
                    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
                    ax.text(
                        0.02,
                        0.98,
                        stats_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        fontfamily="monospace",
                        bbox=props,
                    )

                title = f"Shadow Edge Distribution: {args.stime} to {args.etime}"
                title += f"\n{len(leading_edges)} frames"
                if args.adjust != "none":
                    title += f" (adjustment: {args.adjust})"
                ax.set_title(title)

                fig.tight_layout()

                if args.output:
                    out_path = Path(args.output)
                    hist_output = out_path.parent / f"{out_path.stem}_hist{out_path.suffix}"
                    fig.savefig(hist_output, dpi=args.dpi, bbox_inches="tight")
                    print(f"Saved histogram: {hist_output}")
                else:
                    plt.show()

        # Per-frame output if requested
        if args.per_frame:
            print("\n" + "=" * 70)
            print("PER-FRAME DETAILS")
            leading, trailing = shadow_region
            if leading is not None or trailing is not None:
                lead_str = f"{leading:.1f}°" if leading else "N/A"
                trail_str = f"{trailing:.1f}°" if trailing else "N/A"
                print(f"Global shadow region: {lead_str} - {trail_str}")
            print("=" * 70)

            # Build header based on whether shadow is calculated
            if args.shadow:
                hdr = (
                    f"{'#':>4} {'Timestamp':>26} {'nBear':>6} {'First':>10} {'Last':>10} "
                    f"{'Span':>8} {'Gap':>10} {'Leading':>8} {'Trailing':>8}"
                )
                print(hdr)
                print("-" * 115)
            else:
                hdr = (
                    f"{'#':>4} {'Timestamp':>26} {'nBear':>6} {'First':>10} {'Last':>10} "
                    f"{'Span':>8} {'Gap':>10}"
                )
                print(hdr)
                print("-" * 90)

            for i, r in enumerate(results):
                ts_str = (
                    str(r["timestamp"])[:26]
                    if r["timestamp"] is not None
                    else "N/A"
                )
                gap_str = f"{gaps[i-1]:.4f}°" if i > 0 else "N/A"
                wt = r["wtheta"]
                n_bear = r["n_bearings"]

                if args.show_gaps and i > 0:
                    if abs(gaps[i - 1]) <= args.tolerance:
                        continue

                if args.shadow:
                    region = r.get("shadow_region", (None, None))
                    lead_str = f"{region[0]:.1f}" if region[0] else "-"
                    trail_str = f"{region[1]:.1f}" if region[1] else "-"
                    print(
                        f"{i:4d} {ts_str:>26} {n_bear:>6} {wt.first_theta:>10.4f}° "
                        f"{wt.last_theta:>10.4f}° {wt.theta_span:>8.2f}° "
                        f"{gap_str:>10} {lead_str:>8} {trail_str:>8}"
                    )
                else:
                    print(
                        f"{i:4d} {ts_str:>26} {n_bear:>6} {wt.first_theta:>10.4f}° "
                        f"{wt.last_theta:>10.4f}° {wt.theta_span:>8.2f}° "
                        f"{gap_str:>10}"
                    )

        # Plot theta vs time if requested
        if args.plot or args.output:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            # Parse figure size
            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                logger.error("Invalid figsize: %s", args.figsize)
                return 1

            fig, ax = plt.subplots(figsize=figsize)

            # Concatenate times and angles from all frames
            all_times = []
            all_thetas = []
            frame_boundaries = []
            current_idx = 0

            for r in results:
                wpps = r["wpps"]
                wtheta = r["wtheta"]

                # Get times from WamosPPS (already updated with context)
                times = wpps.time

                # Get angles from WamosTheta
                thetas = wtheta.theta

                all_times.append(times)
                all_thetas.append(thetas)

                current_idx += len(times)
                frame_boundaries.append(current_idx)

            # Concatenate all data
            times_concat = np.concatenate(all_times)
            thetas_concat = np.concatenate(all_thetas)

            # Plot using numpy datetime64 directly (matplotlib handles it)
            ax.plot(times_concat, thetas_concat, linewidth=0.5, color="#1f77b4")

            # Add frame boundary lines (vertical, blue)
            for boundary in frame_boundaries[:-1]:
                if boundary < len(times_concat):
                    ax.axvline(
                        times_concat[boundary],
                        color="blue",
                        linewidth=0.5,
                        alpha=0.7
                    )

            # Format x-axis as datetime
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()

            ax.set_xlabel("Time")
            ax.set_ylabel("Theta (degrees)")
            ax.set_ylim(0, 360)

            title = f"Theta vs Time: {args.stime} to {args.etime}"
            title += f"\n{n_loaded} frames, {len(thetas_concat)} samples"
            ax.set_title(title)

            fig.tight_layout()

            if args.output:
                fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved: {args.output}")
            else:
                plt.show()

        # Pcolor plot of intensity vs theta and range
        if args.pcolor:
            import matplotlib.pyplot as plt

            # Parse figure size
            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                logger.error("Invalid figsize: %s", args.figsize)
                return 1

            fig, ax = plt.subplots(figsize=figsize)

            # Determine range bin slice
            range_start = args.range_start
            range_end = range_start + args.n_ranges

            # Collect theta and intensity data from all frames
            all_thetas = []
            all_intensities = []

            for r in results:
                wtheta = r["wtheta"]
                intensity = r["intensity"]

                # Clip range bins to available data
                actual_end = min(range_end, intensity.shape[1])
                if actual_end <= range_start:
                    continue

                # Get theta for each radial and intensity for selected range bins
                thetas = wtheta.theta  # (n_bearings,)
                intens = intensity[:, range_start:actual_end]  # (n_bearings, n_ranges)

                all_thetas.append(thetas)
                all_intensities.append(intens)

            if not all_thetas:
                logger.error("No data available for pcolor plot")
                return 1

            # Concatenate all data
            thetas_concat = np.concatenate(all_thetas)  # (total_radials,)
            intens_concat = np.concatenate(all_intensities, axis=0)  # (total_radials, n_ranges)

            # Sort by theta for better visualization
            sort_idx = np.argsort(thetas_concat)
            thetas_sorted = thetas_concat[sort_idx]
            intens_sorted = intens_concat[sort_idx, :]

            # Create range bin array
            n_ranges_actual = intens_sorted.shape[1]
            range_bins = np.arange(range_start, range_start + n_ranges_actual + 1)

            # Create theta edges for pcolormesh (need n+1 edges for n cells)
            # Use midpoints between consecutive theta values as edges
            theta_edges = np.empty(len(thetas_sorted) + 1)
            theta_edges[1:-1] = (thetas_sorted[:-1] + thetas_sorted[1:]) / 2
            theta_edges[0] = thetas_sorted[0] - (thetas_sorted[1] - thetas_sorted[0]) / 2
            theta_edges[-1] = thetas_sorted[-1] + (thetas_sorted[-1] - thetas_sorted[-2]) / 2

            # Create meshgrid for pcolormesh
            # X = theta edges, Y = range bin edges
            pcm = ax.pcolormesh(
                theta_edges,
                range_bins,
                intens_sorted.T,  # Transpose so range is on y-axis
                shading="flat",
                cmap="viridis",
                vmin=args.vmin,
                vmax=args.vmax,
            )

            ax.set_xlabel("Theta (degrees)")
            ax.set_ylabel("Range Bin")
            ax.set_xlim(0, 360)

            # Plot shadow region edges (already calculated earlier)
            leading, trailing = shadow_region
            if leading is not None:
                ax.axvline(leading, color="red", linewidth=2, linestyle="--")
            if trailing is not None:
                ax.axvline(trailing, color="red", linewidth=2, linestyle="--")

            # Add colorbar
            fig.colorbar(pcm, ax=ax, label="Intensity (12-bit)")

            range_end_actual = range_start + n_ranges_actual - 1
            title = f"Intensity vs Theta: {args.stime} to {args.etime}"
            title += f"\n{n_loaded} frames, range bins {range_start}-{range_end_actual}"
            if leading is not None or trailing is not None:
                lead_str = f"{leading:.1f}°" if leading else "N/A"
                trail_str = f"{trailing:.1f}°" if trailing else "N/A"
                title += f"\nShadow region: {lead_str} - {trail_str}"
            ax.set_title(title)

            fig.tight_layout()

            if args.output:
                # Append "_pcolor" to output filename
                out_path = Path(args.output)
                pcolor_output = out_path.parent / f"{out_path.stem}_pcolor{out_path.suffix}"
                fig.savefig(pcolor_output, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved pcolor: {pcolor_output}")
            else:
                plt.show()

        print()
        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
