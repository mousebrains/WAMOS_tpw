#!/usr/bin/env python3
"""
Combine multiple radar frames into a single earth-referenced image.

This script loads polar files, processes them through the full pipeline,
and combines them into a single composite image in earth coordinates
with ship motion compensation.

Features:
  - Full pipeline: PolarFile -> Theta -> Destreak -> Shadow -> Deramp -> Dewind
  - Earth-referenced coordinates using heading interpolation
  - Ship motion compensation (each radial from its own position)
  - Gridding onto regular Cartesian grid
  - Display combined image with ship track overlay

Usage:
    python combine_check.py 20220405 20220406 /path/to/POLAR
    python combine_check.py 20220405 20220406 /path/to/POLAR --grid-size 800
    python combine_check.py 20220405 20220406 /path/to/POLAR -o combined.png

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
from wamos_tpw.deramp import Deramp  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.dewind import Dewind  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.range import Range  # noqa: E402
from wamos_tpw.shadow import Shadow  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402

logger = logging.getLogger(__name__)

# Earth radius in meters (WGS84 mean radius)
EARTH_RADIUS = 6_371_000.0


@dataclass
class FrameData:
    """Processed frame data for combining."""

    filename: str
    timestamp: np.datetime64
    theta: np.ndarray  # Radar-relative angles (degrees)
    heading: float  # Ship heading at frame time
    range_vals: np.ndarray  # Slant range (meters)
    intensity: np.ndarray  # Processed intensity (n_radials, n_ranges)
    latitude: float
    longitude: float
    repeat_time: float
    ship_speed: float
    ship_course: float
    wind_speed: float | None
    wind_direction: float | None


def load_and_process_frame(fn: str, config: Config) -> FrameData | None:
    """
    Load and process a single frame through the full pipeline.

    Args:
        fn: Filename to load
        config: Configuration object

    Returns:
        FrameData with processed intensity and coordinates, or None on error.
    """
    try:
        pf = PolarFile(fn, config=config)
        frame = pf.frame()
        meta = frame.metadata

        # Calculate theta
        theta_obj = Theta(frame)

        # Destreak
        destreaked = Destreak(frame)

        # Shadow detection and theta bias
        shadow = Shadow(destreaked.intensity, theta_obj)
        if shadow.theta_bias:
            theta_obj.set_bias(shadow.theta_bias)

        # Range calculation
        rng = Range(frame)

        # Deramp
        masked_intensity = shadow.mask(destreaked.intensity)
        deramp = Deramp(masked_intensity, rng)

        # Dewind
        dewind = Dewind(deramp.intensity, theta_obj)

        return FrameData(
            filename=fn,
            timestamp=frame.timestamp,
            theta=theta_obj.theta.copy(),
            heading=meta.heading if meta.heading is not None else 0.0,
            range_vals=rng.slant_range.copy(),
            intensity=dewind.intensity.astype(np.float32),
            latitude=meta.latitude if meta.latitude is not None else 0.0,
            longitude=meta.longitude if meta.longitude is not None else 0.0,
            repeat_time=meta.repeat_time if meta.repeat_time > 0 else 1.5,
            ship_speed=meta.ship_speed if meta.ship_speed is not None else 0.0,
            ship_course=meta.ship_course
            if meta.ship_course is not None
            else (meta.heading if meta.heading is not None else 0.0),
            wind_speed=meta.wind_speed,
            wind_direction=meta.wind_direction,
        )
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def interpolate_heading(
    curr_heading: float, next_heading: float | None, n_radials: int
) -> np.ndarray:
    """Interpolate heading across radials within a frame."""
    if next_heading is None:
        return np.full(n_radials, curr_heading, dtype=np.float32)

    # Handle wraparound
    delta = next_heading - curr_heading
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360

    t_frac = np.linspace(0, 1, n_radials, endpoint=False)
    return ((curr_heading + delta * t_frac) % 360).astype(np.float32)


def interpolate_position(
    curr_lat: float,
    curr_lon: float,
    next_lat: float | None,
    next_lon: float | None,
    n_radials: int,
    ship_speed: float,
    ship_course: float,
    repeat_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate lat/lon for each radial."""
    if next_lat is not None and next_lon is not None:
        # Interpolate between frames
        t_frac = np.linspace(0, 1, n_radials, endpoint=False)
        lat = curr_lat + (next_lat - curr_lat) * t_frac
        lon = curr_lon + (next_lon - curr_lon) * t_frac
    else:
        # Extrapolate using ship motion
        if ship_speed <= 0:
            return (
                np.full(n_radials, curr_lat, dtype=np.float64),
                np.full(n_radials, curr_lon, dtype=np.float64),
            )

        dt = np.linspace(0, repeat_time, n_radials, endpoint=False)
        displacement = ship_speed * dt
        course_rad = np.deg2rad(ship_course)

        delta_north = displacement * np.cos(course_rad)
        delta_east = displacement * np.sin(course_rad)

        meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(curr_lat))

        lat = curr_lat + delta_north / meters_per_deg_lat
        lon = curr_lon + delta_east / meters_per_deg_lon if meters_per_deg_lon > 0 else curr_lon

    return lat.astype(np.float64), lon.astype(np.float64)


def compute_earth_coordinates(
    frame_data: FrameData,
    heading_interp: np.ndarray,
    ship_lat: np.ndarray,
    ship_lon: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute earth-referenced x/y coordinates for all pixels in a frame.

    Args:
        frame_data: Processed frame data
        heading_interp: Interpolated heading per radial
        ship_lat, ship_lon: Ship position per radial
        ref_lat, ref_lon: Reference position for coordinate origin

    Returns:
        Tuple of (x_earth, y_earth) arrays in meters, shape (n_radials, n_ranges)
    """
    theta = frame_data.theta
    range_vals = frame_data.range_vals

    # Earth bearing for each radial
    earth_bearing = (theta + heading_interp) % 360
    earth_bearing_rad = np.deg2rad(earth_bearing)

    # Create meshgrid for vectorized computation
    bearing_mesh, range_mesh = np.meshgrid(earth_bearing_rad, range_vals, indexing="ij")

    # Radar-relative x/y (relative to ship)
    x_radar = range_mesh * np.sin(bearing_mesh)  # East from ship
    y_radar = range_mesh * np.cos(bearing_mesh)  # North from ship

    # Ship position offset from reference (in meters)
    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    ship_x = (ship_lon - ref_lon) * meters_per_deg_lon  # East offset
    ship_y = (ship_lat - ref_lat) * meters_per_deg_lat  # North offset

    # Expand ship position to 2D
    ship_x_2d = ship_x[:, np.newaxis]
    ship_y_2d = ship_y[:, np.newaxis]

    # Earth coordinates (radar position + ship offset)
    x_earth = x_radar + ship_x_2d
    y_earth = y_radar + ship_y_2d

    return x_earth.astype(np.float32), y_earth.astype(np.float32)


def _compute_bounds_from_metadata(
    frames: list[FrameData],
    ref_lat: float,
    ref_lon: float,
    meters_per_deg_lat: float,
    meters_per_deg_lon: float,
    padding: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Compute grid bounds from frame metadata (ship position ± max range).

    This is faster than computing full earth coordinates for all frames.
    """
    # Ship positions in meters
    ship_x = np.array([(f.longitude - ref_lon) * meters_per_deg_lon for f in frames])
    ship_y = np.array([(f.latitude - ref_lat) * meters_per_deg_lat for f in frames])

    # Max range across all frames
    max_range = max(f.range_vals.max() for f in frames)

    # Bounds: envelope of ship positions ± max range
    all_x_min = ship_x.min() - max_range
    all_x_max = ship_x.max() + max_range
    all_y_min = ship_y.min() - max_range
    all_y_max = ship_y.max() + max_range

    # Add padding
    x_range = all_x_max - all_x_min
    y_range = all_y_max - all_y_min
    all_x_min -= x_range * padding
    all_x_max += x_range * padding
    all_y_min -= y_range * padding
    all_y_max += y_range * padding

    return all_x_min, all_x_max, all_y_min, all_y_max


@dataclass
class ChunkResult:
    """Result from processing a chunk of frames."""

    sum_grid: np.ndarray
    count_grid: np.ndarray
    ship_track_x: list
    ship_track_y: list
    ship_speeds: list
    ship_headings: list
    wind_speeds: list
    wind_dirs: list


def _grid_frame_chunk(
    chunk_indices: list[int],
    frames: list[FrameData],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    grid_size: int,
    ref_lat: float,
    ref_lon: float,
    meters_per_deg_lat: float,
    meters_per_deg_lon: float,
) -> ChunkResult:
    """
    Grid a chunk of frames, returning partial sum/count grids.

    Each worker has its own accumulation grids to avoid thread contention.
    """
    sum_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    ship_track_x = []
    ship_track_y = []
    ship_speeds = []
    ship_headings = []
    wind_speeds = []
    wind_dirs = []

    for i in chunk_indices:
        frame = frames[i]
        next_frame = frames[i + 1] if i < len(frames) - 1 else None

        n_radials = len(frame.theta)
        next_heading = next_frame.heading if next_frame else None
        heading_interp = interpolate_heading(frame.heading, next_heading, n_radials)

        next_lat = next_frame.latitude if next_frame else None
        next_lon = next_frame.longitude if next_frame else None
        ship_lat, ship_lon = interpolate_position(
            frame.latitude,
            frame.longitude,
            next_lat,
            next_lon,
            n_radials,
            frame.ship_speed,
            frame.ship_course,
            frame.repeat_time,
        )

        x_earth, y_earth = compute_earth_coordinates(
            frame, heading_interp, ship_lat, ship_lon, ref_lat, ref_lon
        )

        # Ship track (subsampled)
        ship_x = (ship_lon - ref_lon) * meters_per_deg_lon
        ship_y = (ship_lat - ref_lat) * meters_per_deg_lat
        ship_track_x.extend(ship_x[::10])
        ship_track_y.extend(ship_y[::10])

        # Collect ship and wind data for polar plots
        if frame.ship_speed > 0:
            ship_speeds.append(frame.ship_speed)
            ship_headings.append(frame.heading)
        if frame.wind_speed is not None and frame.wind_direction is not None:
            wind_speeds.append(frame.wind_speed)
            wind_dirs.append(frame.wind_direction)

        # Flatten for binning
        x_flat = x_earth.ravel()
        y_flat = y_earth.ravel()
        values_flat = frame.intensity.ravel()

        # Compute bin indices
        x_idx = np.searchsorted(x_edges, x_flat) - 1
        y_idx = np.searchsorted(y_edges, y_flat) - 1

        # Valid indices (in range and not NaN)
        valid = (
            (x_idx >= 0)
            & (x_idx < grid_size)
            & (y_idx >= 0)
            & (y_idx < grid_size)
            & ~np.isnan(values_flat)
        )

        x_idx_valid = x_idx[valid]
        y_idx_valid = y_idx[valid]
        values_valid = values_flat[valid].astype(np.float64)

        # Accumulate into local grids
        np.add.at(sum_grid, (y_idx_valid, x_idx_valid), values_valid)
        np.add.at(count_grid, (y_idx_valid, x_idx_valid), 1)

    return ChunkResult(
        sum_grid=sum_grid,
        count_grid=count_grid,
        ship_track_x=ship_track_x,
        ship_track_y=ship_track_y,
        ship_speeds=ship_speeds,
        ship_headings=ship_headings,
        wind_speeds=wind_speeds,
        wind_dirs=wind_dirs,
    )


def grid_frames(
    frames: list[FrameData],
    grid_size: int = 800,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Grid all frames onto a common earth-referenced grid.

    Uses parallel workers to process frame chunks independently, then merges
    the partial grids at the end (map-reduce pattern).

    Args:
        frames: List of processed frame data
        grid_size: Number of bins in each dimension
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        Tuple of (x_edges, y_edges, gridded_intensity, metadata)
    """
    if not frames:
        raise ValueError("No frames to grid")

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # Reference position (first frame)
    ref_lat = frames[0].latitude
    ref_lon = frames[0].longitude

    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    # Compute bounds from metadata (fast, single pass)
    all_x_min, all_x_max, all_y_min, all_y_max = _compute_bounds_from_metadata(
        frames, ref_lat, ref_lon, meters_per_deg_lat, meters_per_deg_lon
    )

    # Create grid edges
    x_edges = np.linspace(all_x_min, all_x_max, grid_size + 1)
    y_edges = np.linspace(all_y_min, all_y_max, grid_size + 1)

    # Split frame indices into chunks for parallel processing
    n_frames = len(frames)
    chunk_size = max(1, (n_frames + n_workers - 1) // n_workers)
    chunks = [list(range(i, min(i + chunk_size, n_frames))) for i in range(0, n_frames, chunk_size)]

    # Process chunks in parallel
    sum_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    ship_track_x = []
    ship_track_y = []
    ship_speeds = []
    ship_headings = []
    wind_speeds = []
    wind_dirs = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _grid_frame_chunk,
                chunk,
                frames,
                x_edges,
                y_edges,
                grid_size,
                ref_lat,
                ref_lon,
                meters_per_deg_lat,
                meters_per_deg_lon,
            )
            for chunk in chunks
        ]

        for future in as_completed(futures):
            result = future.result()
            sum_grid += result.sum_grid
            count_grid += result.count_grid
            ship_track_x.extend(result.ship_track_x)
            ship_track_y.extend(result.ship_track_y)
            ship_speeds.extend(result.ship_speeds)
            ship_headings.extend(result.ship_headings)
            wind_speeds.extend(result.wind_speeds)
            wind_dirs.extend(result.wind_dirs)

    # Compute mean
    with np.errstate(invalid="ignore"):
        gridded = sum_grid / count_grid
    gridded[count_grid == 0] = np.nan

    metadata = {
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "n_frames": len(frames),
        "ship_track_x": np.array(ship_track_x),
        "ship_track_y": np.array(ship_track_y),
        "ship_speeds": ship_speeds,
        "ship_headings": ship_headings,
        "wind_speeds": wind_speeds,
        "wind_dirs": wind_dirs,
        "start_time": frames[0].timestamp,
        "end_time": frames[-1].timestamp,
    }

    return x_edges, y_edges, gridded.astype(np.float32), metadata


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
    parser = argparse.ArgumentParser(description="Combine radar frames into earth-referenced image")

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
        "--grid-size",
        type=int,
        default=800,
        help="Grid size in pixels (default: 800)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
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
        default="12,10",
        help="Figure size as 'width,height' in inches (default: 12,10)",
    )
    parser.add_argument(
        "--show-track",
        action="store_true",
        help="Overlay ship track on the image",
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

        if args.max_frames:
            files = files[: args.max_frames]
            n_files = len(files)

        logger.info("Loading %d files with %d threads", n_files, args.workers or os.cpu_count())

        # Load and process frames in parallel
        frame_data_list: list[FrameData] = []
        config = Config(args.config)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_and_process_frame, fn, config): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    frame_data_list.append(result)
        elapsed = time.perf_counter() - t0

        n_loaded = len(frame_data_list)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(f"Loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort frames by timestamp
        frame_data_list.sort(key=lambda x: x.timestamp)
        print(f"Frames span: {frame_data_list[0].timestamp} to {frame_data_list[-1].timestamp}")

        # Grid frames
        n_grid_workers = args.workers or os.cpu_count()
        print(
            f"\nGridding {n_loaded} frames onto {args.grid_size}x{args.grid_size} grid with {n_grid_workers} workers..."
        )
        t0 = time.perf_counter()
        x_edges, y_edges, gridded, metadata = grid_frames(
            frame_data_list, args.grid_size, args.workers
        )
        elapsed = time.perf_counter() - t0
        print(f"Gridding completed in {elapsed:.2f}s")

        # Statistics
        valid_pixels = ~np.isnan(gridded)
        print("\n=== Combined Image Statistics ===")
        print(f"Grid size: {args.grid_size} x {args.grid_size}")
        print(f"Valid pixels: {np.sum(valid_pixels)} ({100 * np.mean(valid_pixels):.1f}%)")
        print(f"X range: [{x_edges[0]:.1f}, {x_edges[-1]:.1f}] m")
        print(f"Y range: [{y_edges[0]:.1f}, {y_edges[-1]:.1f}] m")
        print(f"Intensity range: [{np.nanmin(gridded):.1f}, {np.nanmax(gridded):.1f}]")
        print(f"Reference position: ({metadata['ref_lat']:.6f}, {metadata['ref_lon']:.6f})")

        # Plot
        import matplotlib.pyplot as plt

        try:
            figsize = tuple(float(x) for x in args.figsize.split(","))
        except ValueError:
            figsize = (12, 10)

        fig, ax = plt.subplots(figsize=figsize)

        # Compute extent
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        # Display image
        vmin = np.nanpercentile(gridded, 1)
        vmax = np.nanpercentile(gridded, 99)
        im = ax.imshow(
            gridded,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Ship track overlay
        if args.show_track:
            track_x = metadata["ship_track_x"]
            track_y = metadata["ship_track_y"]
            ax.plot(track_x, track_y, "ro", markersize=2, fillstyle="none", alpha=0.7)
            ax.plot(track_x[0], track_y[0], "go", markersize=8)
            ax.plot(track_x[-1], track_y[-1], "ro", markersize=8)

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title(
            f"Combined Radar Image ({n_loaded} frames)\n"
            f"{metadata['start_time']} to {metadata['end_time']}"
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Intensity")

        # Ship polar inset (upper right, label upper left)
        ship_speeds = metadata["ship_speeds"]
        ship_headings = metadata["ship_headings"]
        if ship_speeds and ship_headings:
            ax_ship = ax.inset_axes([0.88, 0.85, 0.11, 0.14], projection="polar")
            ax_ship.set_theta_zero_location("N")
            ax_ship.set_theta_direction(-1)
            ages = np.linspace(0, 1, len(ship_speeds))
            ax_ship.scatter(
                np.deg2rad(ship_headings), ship_speeds, c=ages, cmap="viridis", s=6, alpha=0.8
            )
            ax_ship.set_xticklabels([])
            ax_ship.tick_params(labelsize=4)
            ax_ship.set_facecolor("white")
            ax_ship.patch.set_alpha(0.8)
            ax_ship.grid(True, linewidth=0.3, alpha=0.5)
            ax.text(0.88, 0.995, "Ship", transform=ax.transAxes, fontsize=7, ha="left", va="top")

        # Wind polar inset (lower right, label lower left)
        wind_speeds = metadata["wind_speeds"]
        wind_dirs = metadata["wind_dirs"]
        if wind_speeds and wind_dirs:
            ax_wind = ax.inset_axes([0.88, 0.01, 0.11, 0.14], projection="polar")
            ax_wind.set_theta_zero_location("N")
            ax_wind.set_theta_direction(-1)
            ages = np.linspace(0, 1, len(wind_speeds))
            ax_wind.scatter(
                np.deg2rad(wind_dirs), wind_speeds, c=ages, cmap="viridis", s=6, alpha=0.8
            )
            ax_wind.set_xticklabels([])
            ax_wind.tick_params(labelsize=4)
            ax_wind.set_facecolor("white")
            ax_wind.patch.set_alpha(0.8)
            ax_wind.grid(True, linewidth=0.3, alpha=0.5)
            ax.text(0.88, 0.01, "Wind", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")

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
