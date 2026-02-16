#!/usr/bin/env python3
"""
Check timestamp and position calculations across multiple frames.

This script loads polar files, extracts PPS (Pulse Per Second) timing signals,
and calculates per-radial timestamps, headings, and positions by interpolating
between consecutive frames.

Features:
  - Extracts PPS indices from each frame (bit 12, bin 18)
  - Sorts frames by timestamp
  - Calculates per-radial timestamps using previous, center, and next frames
  - Interpolates ship heading between frames
  - Interpolates latitude/longitude between frames
  - Reports timing and position statistics
  - Optional plots of timing distributions and position tracks

Usage:
    python timestamp_check.py 20220405 20220406 /path/to/POLAR
    python timestamp_check.py 20220405 20220406 /path/to/POLAR --plot
    python timestamp_check.py 20220405 20220406 /path/to/POLAR --plot -o timestamp_stats.png

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# PPS timing signal location
PPS_BIT = 12
PPS_BIN = 18
PPS_MASK = np.uint16(1 << PPS_BIT)

# Earth radius in meters (WGS84 mean radius)
EARTH_RADIUS = 6_371_000.0


@dataclass
class FrameInfo:
    """Information extracted from a single frame for timestamp calculation."""

    filename: str
    timestamp: np.datetime64
    n_radials: int
    repeat_time: float  # Frame duration in seconds
    pps_indices: np.ndarray  # Indices where PPS transitions occur
    theta: np.ndarray  # Radar-relative angles (degrees)
    latitude: float | None
    longitude: float | None
    heading: float | None
    ship_speed: float | None  # m/s
    ship_course: float | None


def find_pps_transitions(data: np.ndarray, bin_idx: int = PPS_BIN) -> np.ndarray:
    """
    Find indices where PPS signal transitions (0->1 or 1->0).

    Args:
        data: Raw uint16 frame data (n_bearings, n_distances)
        bin_idx: Distance bin containing PPS signal

    Returns:
        Array of radial indices where PPS transitions occur
    """
    if bin_idx >= data.shape[1]:
        return np.array([], dtype=int)

    pps_signal = (data[:, bin_idx] & PPS_MASK) != 0
    changes = np.diff(pps_signal.astype(np.int8)) != 0
    return np.where(changes)[0] + 1


def load_frame(fn: str, config: Config) -> FrameInfo | None:
    """
    Load a single frame and extract timing information.

    Args:
        fn: Filename to load
        config: Configuration object

    Returns:
        FrameInfo with extracted data, or None on error.
    """
    from wamos_tpw.theta import Theta

    try:
        pf = PolarFile(fn, config=config)
        frame = pf.frame()
        meta = frame.metadata

        # Find PPS transitions
        pps_indices = find_pps_transitions(frame.raw)

        # Calculate theta (radar-relative angles)
        theta_obj = Theta(frame)
        theta = theta_obj.theta.copy()

        return FrameInfo(
            filename=fn,
            timestamp=frame.timestamp,
            n_radials=frame.n_bearings,
            repeat_time=meta.repeat_time if meta.repeat_time > 0 else 1.5,
            pps_indices=pps_indices,
            theta=theta,
            latitude=meta.latitude,
            longitude=meta.longitude,
            heading=meta.heading,
            ship_speed=meta.ship_speed,
            ship_course=meta.ship_course,
        )
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def calculate_radial_times(
    prev_info: FrameInfo | None,
    curr_info: FrameInfo,
    next_info: FrameInfo | None,
) -> np.ndarray:
    """
    Calculate timestamps for each radial in the current frame.

    Uses PPS transitions from previous, current, and next frames to establish
    timing reference points, then interpolates radial times.

    Args:
        prev_info: Previous frame info (None if first frame)
        curr_info: Current frame info
        next_info: Next frame info (None if last frame)

    Returns:
        Array of float64 timestamps (seconds since epoch) for each radial
    """
    n_radials = curr_info.n_radials
    rpt = curr_info.repeat_time

    # Convert frame timestamp to seconds since epoch
    frame_time = curr_info.timestamp.astype("datetime64[ns]").astype(np.float64) / 1e9

    # Time per radial (uniform assumption)
    dt = rpt / n_radials

    # Find PPS reference in current frame
    pps = curr_info.pps_indices
    if len(pps) > 0:
        # Use first PPS transition as 1-second boundary reference
        pps_radial = pps[0]
        # The PPS transition marks where time crosses a 1-second boundary
        # Radial times are distributed uniformly around this reference
        # Assume frame timestamp corresponds to start of frame
        time_at_pps = frame_time + pps_radial * dt
        # Round to nearest second boundary
        pps_second = np.floor(time_at_pps)
        # Adjust frame start time to align with PPS
        frame_start = pps_second - pps_radial * dt
    else:
        # No PPS - use frame timestamp as start
        frame_start = frame_time

    # Calculate radial times
    radial_times = frame_start + np.arange(n_radials) * dt

    return radial_times


def interpolate_heading(
    radial_times: np.ndarray,
    curr_info: FrameInfo,
    next_info: FrameInfo | None,
) -> np.ndarray:
    """
    Interpolate ship heading for each radial between current and next frame.

    Args:
        radial_times: Timestamps for each radial
        curr_info: Current frame info
        next_info: Next frame info (None if last frame)

    Returns:
        Array of headings in degrees for each radial
    """
    n_radials = len(radial_times)
    curr_heading = curr_info.heading if curr_info.heading is not None else 0.0

    if next_info is None or next_info.heading is None:
        return np.full(n_radials, curr_heading, dtype=np.float32)

    next_heading = next_info.heading

    # Handle wraparound (e.g., 359 -> 1 degrees)
    delta = next_heading - curr_heading
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360

    # Interpolate based on time
    curr_time = curr_info.timestamp.astype("datetime64[ns]").astype(np.float64) / 1e9
    next_time = next_info.timestamp.astype("datetime64[ns]").astype(np.float64) / 1e9
    duration = next_time - curr_time

    if duration <= 0:
        return np.full(n_radials, curr_heading, dtype=np.float32)

    t_frac = (radial_times - curr_time) / duration
    t_frac = np.clip(t_frac, 0, 1)

    headings = (curr_heading + delta * t_frac) % 360
    return headings.astype(np.float32)


def calculate_earth_bearing(
    theta: np.ndarray,
    heading_interpolated: np.ndarray,
) -> np.ndarray:
    """
    Calculate earth-referenced bearing for each radial.

    Earth bearing = (theta + heading) % 360

    Args:
        theta: Radar-relative angles in degrees
        heading_interpolated: Interpolated heading per radial in degrees

    Returns:
        Array of earth-referenced bearings in degrees [0, 360)
    """
    return (theta + heading_interpolated) % 360


def interpolate_position(
    radial_times: np.ndarray,
    curr_info: FrameInfo,
    next_info: FrameInfo | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate latitude/longitude for each radial between current and next frame.

    Args:
        radial_times: Timestamps for each radial
        curr_info: Current frame info
        next_info: Next frame info (None if last frame)

    Returns:
        Tuple of (latitude, longitude) arrays in degrees
    """
    n_radials = len(radial_times)
    curr_lat = curr_info.latitude if curr_info.latitude is not None else 0.0
    curr_lon = curr_info.longitude if curr_info.longitude is not None else 0.0

    if next_info is None or next_info.latitude is None or next_info.longitude is None:
        # No next frame - use ship motion to extrapolate
        speed = curr_info.ship_speed if curr_info.ship_speed is not None else 0.0
        course = (
            curr_info.ship_course
            if curr_info.ship_course is not None
            else (curr_info.heading if curr_info.heading is not None else 0.0)
        )

        if speed <= 0:
            return (
                np.full(n_radials, curr_lat, dtype=np.float64),
                np.full(n_radials, curr_lon, dtype=np.float64),
            )

        curr_time = curr_info.timestamp.astype("datetime64[ns]").astype(np.float64) / 1e9
        time_deltas = radial_times - curr_time

        # Calculate displacement
        displacement = speed * time_deltas
        course_rad = np.deg2rad(course)

        delta_north = displacement * np.cos(course_rad)
        delta_east = displacement * np.sin(course_rad)

        # Convert to lat/lon
        meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(curr_lat))

        lat = curr_lat + delta_north / meters_per_deg_lat
        lon = curr_lon + delta_east / meters_per_deg_lon if meters_per_deg_lon > 0 else curr_lon

        return lat.astype(np.float64), lon.astype(np.float64)

    # Interpolate between frames
    next_lat = next_info.latitude
    next_lon = next_info.longitude

    curr_time = curr_info.timestamp.astype("datetime64[ns]").astype(np.float64) / 1e9
    next_time = next_info.timestamp.astype("datetime64[ns]").astype(np.float64) / 1e9
    duration = next_time - curr_time

    if duration <= 0:
        return (
            np.full(n_radials, curr_lat, dtype=np.float64),
            np.full(n_radials, curr_lon, dtype=np.float64),
        )

    t_frac = (radial_times - curr_time) / duration
    t_frac = np.clip(t_frac, 0, 1)

    lat = curr_lat + (next_lat - curr_lat) * t_frac
    lon = curr_lon + (next_lon - curr_lon) * t_frac

    return lat.astype(np.float64), lon.astype(np.float64)


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
        description="Check timestamp and position calculations across polar files"
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
        "--plot",
        action="store_true",
        help="Show timestamp and position plots",
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
        default="14,12",
        help="Figure size as 'width,height' in inches (default: 14,12)",
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

        logger.info("Loading %d files with %d threads", n_files, args.workers or os.cpu_count())

        # Load frames in parallel
        frame_infos: list[FrameInfo] = []
        config = Config(args.config)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame, fn, config): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    frame_infos.append(result)
        elapsed = time.perf_counter() - t0

        n_loaded = len(frame_infos)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(
            f"Successfully loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)"
        )

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort frames by timestamp
        frame_infos.sort(key=lambda x: x.timestamp)
        print(f"Frames span: {frame_infos[0].timestamp} to {frame_infos[-1].timestamp}")

        # Calculate per-radial data for each frame
        print("\n=== Processing Frames ===")
        all_radial_times: list[np.ndarray] = []
        all_headings: list[np.ndarray] = []
        all_thetas: list[np.ndarray] = []
        all_earth_bearings: list[np.ndarray] = []
        all_lats: list[np.ndarray] = []
        all_lons: list[np.ndarray] = []
        pps_counts: list[int] = []

        for i, curr_info in enumerate(frame_infos):
            prev_info = frame_infos[i - 1] if i > 0 else None
            next_info = frame_infos[i + 1] if i < n_loaded - 1 else None

            radial_times = calculate_radial_times(prev_info, curr_info, next_info)
            headings = interpolate_heading(radial_times, curr_info, next_info)
            earth_bearings = calculate_earth_bearing(curr_info.theta, headings)
            lats, lons = interpolate_position(radial_times, curr_info, next_info)

            all_radial_times.append(radial_times)
            all_headings.append(headings)
            all_thetas.append(curr_info.theta)
            all_earth_bearings.append(earth_bearings)
            all_lats.append(lats)
            all_lons.append(lons)
            pps_counts.append(len(curr_info.pps_indices))

        # Statistics
        total_radials = sum(len(t) for t in all_radial_times)
        pps_counts_arr = np.array(pps_counts)

        print("\n=== PPS Statistics ===")
        print(f"Total radials: {total_radials}")
        print(
            f"PPS transitions per frame: mean={pps_counts_arr.mean():.1f}, "
            f"min={pps_counts_arr.min()}, max={pps_counts_arr.max()}"
        )
        print(f"Frames with PPS: {np.sum(pps_counts_arr > 0)} / {n_loaded}")

        # Timing statistics
        print("\n=== Timing Statistics ===")
        time_steps = []
        for times in all_radial_times:
            if len(times) > 1:
                time_steps.extend(np.diff(times))
        time_steps = np.array(time_steps)

        if len(time_steps) > 0:
            print(
                f"Radial time step: mean={time_steps.mean() * 1000:.4f} ms, "
                f"std={time_steps.std() * 1000:.4f} ms"
            )
            print(
                f"Time step range: [{time_steps.min() * 1000:.4f}, {time_steps.max() * 1000:.4f}] ms"
            )

        # Heading statistics
        print("\n=== Heading Statistics ===")
        all_hdg = np.concatenate(all_headings)
        print(f"Heading range: [{all_hdg.min():.2f}, {all_hdg.max():.2f}] degrees")
        print(f"Heading: mean={np.mean(all_hdg):.2f}, std={np.std(all_hdg):.2f} degrees")

        # Earth bearing statistics
        print("\n=== Earth Bearing Statistics ===")
        all_theta = np.concatenate(all_thetas)
        all_earth = np.concatenate(all_earth_bearings)
        print(
            f"Theta (radar-relative) range: [{all_theta.min():.2f}, {all_theta.max():.2f}] degrees"
        )
        print(f"Earth bearing range: [{all_earth.min():.2f}, {all_earth.max():.2f}] degrees")
        print(f"Earth bearing: mean={np.mean(all_earth):.2f}, std={np.std(all_earth):.2f} degrees")

        # Position statistics
        print("\n=== Position Statistics ===")
        all_lat = np.concatenate(all_lats)
        all_lon = np.concatenate(all_lons)
        print(f"Latitude range: [{all_lat.min():.6f}, {all_lat.max():.6f}] degrees")
        print(f"Longitude range: [{all_lon.min():.6f}, {all_lon.max():.6f}] degrees")

        # Calculate total distance traveled
        if len(all_lat) > 1:
            dlat = np.diff(all_lat)
            dlon = np.diff(all_lon)
            meters_per_deg = np.pi * EARTH_RADIUS / 180.0
            dx = dlon * meters_per_deg * np.cos(np.deg2rad(np.mean(all_lat)))
            dy = dlat * meters_per_deg
            distances = np.sqrt(dx**2 + dy**2)
            total_distance = np.sum(distances)
            print(
                f"Total distance traveled: {total_distance:.1f} m ({total_distance / 1852:.2f} nm)"
            )

        # Generate plots if requested
        if args.plot and n_loaded > 0:
            import matplotlib.pyplot as plt

            try:
                figsize = tuple(float(x) for x in args.figsize.split(","))
            except ValueError:
                figsize = (14, 12)

            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

            # Top left: PPS counts per frame
            ax_pps = fig.add_subplot(gs[0, 0])
            frame_nums = np.arange(n_loaded)
            ax_pps.bar(frame_nums, pps_counts_arr, alpha=0.7)
            ax_pps.axhline(
                pps_counts_arr.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {pps_counts_arr.mean():.1f}",
            )
            ax_pps.set_xlabel("Frame number")
            ax_pps.set_ylabel("PPS transition count")
            ax_pps.set_title("PPS Transitions per Frame")
            ax_pps.legend()
            ax_pps.grid(True, alpha=0.3)

            # Top right: Time step histogram
            ax_dt = fig.add_subplot(gs[0, 1])
            if len(time_steps) > 0:
                ax_dt.hist(time_steps * 1000, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
                ax_dt.axvline(
                    time_steps.mean() * 1000,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {time_steps.mean() * 1000:.4f} ms",
                )
                ax_dt.set_xlabel("Time step (ms)")
                ax_dt.set_ylabel("Count")
                ax_dt.set_title("Radial Time Step Distribution")
                ax_dt.legend()
                ax_dt.grid(True, alpha=0.3)

            # Middle left: Heading vs time
            ax_hdg = fig.add_subplot(gs[1, 0])
            # Plot heading for each frame
            for i, (times, hdg) in enumerate(zip(all_radial_times, all_headings, strict=False)):
                if i % max(1, n_loaded // 20) == 0:  # Plot every Nth frame
                    ax_hdg.plot(times - all_radial_times[0][0], hdg, alpha=0.5, linewidth=0.5)
            ax_hdg.set_xlabel("Time (seconds from start)")
            ax_hdg.set_ylabel("Heading (degrees)")
            ax_hdg.set_title("Ship Heading vs Time")
            ax_hdg.set_ylim(0, 360)
            ax_hdg.grid(True, alpha=0.3)

            # Middle right: Earth bearing distribution
            ax_earth = fig.add_subplot(gs[1, 1])
            ax_earth.hist(all_earth, bins=72, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax_earth.set_xlabel("Earth bearing (degrees)")
            ax_earth.set_ylabel("Count")
            ax_earth.set_title("Earth Bearing Distribution (all radials)")
            ax_earth.set_xlim(0, 360)
            ax_earth.grid(True, alpha=0.3)

            # Bottom left: Theta vs Earth bearing for sample frames
            ax_compare = fig.add_subplot(gs[2, 0])
            for i in range(min(5, n_loaded)):
                idx = i * (n_loaded // min(5, n_loaded))
                ax_compare.scatter(
                    all_thetas[idx], all_earth_bearings[idx], alpha=0.3, s=1, label=f"Frame {idx}"
                )
            ax_compare.plot([0, 360], [0, 360], "k--", alpha=0.3, label="1:1 line")
            ax_compare.set_xlabel("Theta (radar-relative, degrees)")
            ax_compare.set_ylabel("Earth bearing (degrees)")
            ax_compare.set_title("Theta vs Earth Bearing (sample frames)")
            ax_compare.set_xlim(0, 360)
            ax_compare.set_ylim(0, 360)
            ax_compare.grid(True, alpha=0.3)

            # Bottom right: Ship track (lat/lon)
            ax_track = fig.add_subplot(gs[2, 1])
            # Subsample for clarity
            subsample = max(1, total_radials // 5000)
            lat_sub = all_lat[::subsample]
            lon_sub = all_lon[::subsample]
            ax_track.plot(lon_sub, lat_sub, "b-", alpha=0.5, linewidth=0.5)
            ax_track.plot(lon_sub[0], lat_sub[0], "go", markersize=8, label="Start")
            ax_track.plot(lon_sub[-1], lat_sub[-1], "ro", markersize=8, label="End")
            ax_track.set_xlabel("Longitude (degrees)")
            ax_track.set_ylabel("Latitude (degrees)")
            ax_track.set_title("Ship Track")
            ax_track.legend()
            ax_track.grid(True, alpha=0.3)
            ax_track.set_aspect("equal")

            fig.suptitle(
                f"Timestamp Analysis: {args.stime} to {args.etime} ({n_loaded} frames, {total_radials} radials)"
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
