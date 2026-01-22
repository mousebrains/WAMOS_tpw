#! /usr/bin/env python3
#
# Sliding window combine for smooth movie generation
# Uses incremental add/remove for efficient computation
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import tempfile
import contextlib
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from wamos_tpw.config import Config


class SlidingWindowCombine:
    """
    Incremental sliding window combiner for radar frames.

    Maintains running sum/count grids and a buffer of per-frame contributions.
    When sliding, removes oldest frame's contribution and adds newest.

    This is O(1) per slide instead of O(window_size) for recomputation.

    Memory usage: ~15 MB per frame in buffer (sum float64 + count int32)
    For 42-frame window: ~630 MB
    """

    def __init__(
        self,
        window_seconds: float,
        grid_shape: tuple[int, int] = (1200, 1600),
        output_every: int = 5,
    ):
        """
        Initialize sliding window combiner.

        Args:
            window_seconds: Window duration in seconds
            grid_shape: (n_along, n_cross) grid dimensions
            output_every: Generate output every N frames
        """
        self.window_seconds = window_seconds
        self.grid_shape = grid_shape
        self.output_every = output_every

        # Running totals
        self.sum_grid = np.zeros(grid_shape, dtype=np.float64)
        self.count_grid = np.zeros(grid_shape, dtype=np.int32)

        # Buffer of (timestamp, frame_sum, frame_count) tuples
        self.frame_buffer: deque = deque()

        # Frame counter for output timing
        self.frame_counter = 0

        # Grid bounds (set on first frame)
        self.x_edges = None
        self.y_edges = None
        self.ref_lat = None
        self.ref_lon = None

        # Ship track accumulator
        self.ship_x_all = []
        self.ship_y_all = []
        self.ship_speeds_all = []
        self.ship_headings_all = []
        self.wind_speeds_all = []
        self.wind_dirs_all = []

    def set_grid(
        self,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        ref_lat: float,
        ref_lon: float,
    ) -> None:
        """Set grid parameters (called once at start)."""
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon

    def add_frame(
        self,
        timestamp: pd.Timestamp,
        frame_sum: np.ndarray,
        frame_count: np.ndarray,
        ship_x: float = None,
        ship_y: float = None,
        ship_speed: float = None,
        ship_heading: float = None,
        wind_speed: float = None,
        wind_dir: float = None,
    ) -> None:
        """Add a new frame's contribution to the running total."""
        self.sum_grid += frame_sum
        self.count_grid += frame_count
        self.frame_buffer.append((timestamp, frame_sum.copy(), frame_count.copy()))

        # Track ship/wind data
        if ship_x is not None:
            self.ship_x_all.append(ship_x)
        if ship_y is not None:
            self.ship_y_all.append(ship_y)
        if ship_speed is not None:
            self.ship_speeds_all.append(ship_speed)
        if ship_heading is not None:
            self.ship_headings_all.append(ship_heading)
        if wind_speed is not None:
            self.wind_speeds_all.append(wind_speed)
        if wind_dir is not None:
            self.wind_dirs_all.append(wind_dir)

        self.frame_counter += 1

    def _remove_expired(self, current_time: pd.Timestamp) -> None:
        """Remove frames older than window_seconds from current_time."""
        cutoff = current_time - pd.Timedelta(seconds=self.window_seconds)

        while self.frame_buffer:
            oldest_time, oldest_sum, oldest_count = self.frame_buffer[0]
            if oldest_time < cutoff:
                # Remove oldest frame's contribution
                self.sum_grid -= oldest_sum
                self.count_grid -= oldest_count
                self.frame_buffer.popleft()
            else:
                break

    def get_combined(self) -> np.ndarray:
        """Get current combined image from running totals."""
        with np.errstate(invalid="ignore"):
            result = self.sum_grid / self.count_grid
        result[self.count_grid == 0] = np.nan
        return result.astype(np.float32)

    def slide(
        self,
        timestamp: pd.Timestamp,
        frame_sum: np.ndarray,
        frame_count: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray | None, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Slide window: remove expired frames, add new frame.

        Args:
            timestamp: New frame's timestamp
            frame_sum: New frame's gridded sum contribution
            frame_count: New frame's gridded count contribution
            **kwargs: Ship/wind data passed to add_frame

        Returns:
            (combined_image, start_time, end_time) if output_every reached,
            (None, None, None) otherwise
        """
        # Remove frames outside window
        self._remove_expired(timestamp)

        # Add new frame
        self.add_frame(timestamp, frame_sum, frame_count, **kwargs)

        # Check if we should output
        if self.frame_counter % self.output_every == 0 and len(self.frame_buffer) > 0:
            start_time = self.frame_buffer[0][0]
            end_time = self.frame_buffer[-1][0]
            return self.get_combined(), start_time, end_time

        return None, None, None

    def get_window_times(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Get current window start and end times."""
        if not self.frame_buffer:
            return None, None
        return self.frame_buffer[0][0], self.frame_buffer[-1][0]

    def get_ship_track(self) -> tuple[np.ndarray, np.ndarray]:
        """Get accumulated ship track (x, y) in meters."""
        return np.array(self.ship_x_all), np.array(self.ship_y_all)

    def get_recent_ship_track(self, n_recent: int = None) -> tuple[np.ndarray, np.ndarray]:
        """Get ship track for frames currently in window."""
        n_frames = len(self.frame_buffer)
        if n_recent is None:
            n_recent = n_frames
        n_recent = min(n_recent, len(self.ship_x_all))
        return (
            np.array(self.ship_x_all[-n_recent:]),
            np.array(self.ship_y_all[-n_recent:]),
        )

    @property
    def n_frames_in_window(self) -> int:
        """Number of frames currently in the sliding window."""
        return len(self.frame_buffer)

    @property
    def total_frames_processed(self) -> int:
        """Total number of frames processed so far."""
        return self.frame_counter

    def clear(self) -> None:
        """Clear all buffers and reset state."""
        self.sum_grid.fill(0)
        self.count_grid.fill(0)
        self.frame_buffer.clear()
        self.frame_counter = 0
        self.ship_x_all.clear()
        self.ship_y_all.clear()
        self.ship_speeds_all.clear()
        self.ship_headings_all.clear()
        self.wind_speeds_all.clear()
        self.wind_dirs_all.clear()


def _grid_single_frame(
    frame,
    frame_idx: int,
    theta,
    bearing_obj,
    config,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Grid a single frame and return its contribution arrays.

    Returns:
        (frame_sum, frame_count, ship_x, ship_y)
    """
    # Get earth coordinates
    x_earth, y_earth = bearing_obj.xy_earth(frame_idx)

    # Get intensity (prefer corrected > deramped > raw)
    if frame.corrected_intensity is not None:
        intensity = frame.corrected_intensity.ravel()
    elif frame.deramped_intensity is not None:
        intensity = frame.deramped_intensity.ravel()
    else:
        intensity = frame.intensity.ravel().astype(np.float32)

    x_flat = x_earth.ravel()
    y_flat = y_earth.ravel()

    # Create frame-specific sum/count grids
    n_along = len(y_edges) - 1
    n_cross = len(x_edges) - 1
    frame_sum = np.zeros((n_along, n_cross), dtype=np.float64)
    frame_count = np.zeros((n_along, n_cross), dtype=np.int32)

    # Bin the data
    valid = ~np.isnan(intensity)
    x_valid = x_flat[valid]
    y_valid = y_flat[valid]
    i_valid = intensity[valid]

    # Find bin indices
    x_idx = np.searchsorted(x_edges, x_valid) - 1
    y_idx = np.searchsorted(y_edges, y_valid) - 1

    # Clip to valid range
    in_bounds = (x_idx >= 0) & (x_idx < n_cross) & (y_idx >= 0) & (y_idx < n_along)

    x_idx = x_idx[in_bounds]
    y_idx = y_idx[in_bounds]
    i_valid = i_valid[in_bounds]

    # Accumulate
    np.add.at(frame_sum, (y_idx, x_idx), i_valid)
    np.add.at(frame_count, (y_idx, x_idx), 1)

    # Get ship position for this frame
    ship_x, ship_y = bearing_obj.ship_xy(frame_idx)

    return frame_sum, frame_count, float(np.mean(ship_x)), float(np.mean(ship_y))


def _save_sliding_frame(
    gridded: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    output_path: str,
    ref_lat: float,
    ref_lon: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    n_frames: int,
    frame_number: int,
    ship_x: np.ndarray,
    ship_y: np.ndarray,
    ship_speeds: list,
    ship_headings: list,
    wind_speeds: list,
    wind_dirs: list,
) -> None:
    """Save a sliding window combined frame as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    EARTH_RADIUS = 6371000.0

    # Convert to lat/lon
    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    # Ship track
    ship_lon = ref_lon + ship_x / meters_per_deg_lon
    ship_lat = ref_lat + ship_y / meters_per_deg_lat

    # Grid meshgrid
    xx, yy = np.meshgrid(x_edges, y_edges)
    lon_grid = ref_lon + xx / meters_per_deg_lon
    lat_grid = ref_lat + yy / meters_per_deg_lat
    del xx, yy

    # Intensity limits
    valid_data = gridded[~np.isnan(gridded)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [1, 99])
    else:
        vmin, vmax = 0, 1
    del valid_data

    # Figure
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 1], hspace=0.02)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0])
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)

    # Title
    start_ts = pd.Timestamp(start_time).floor("s")
    end_ts = pd.Timestamp(end_time).ceil("s")
    title_str = f"{start_ts} to {end_ts}"

    # Main plot
    im = ax_main.pcolormesh(
        lon_grid, lat_grid, gridded, cmap="viridis", vmin=vmin, vmax=vmax, shading="flat"
    )
    plt.colorbar(im, ax=ax_main, label="Intensity")

    # Ship track (subsample if needed)
    n_track = len(ship_lon)
    if n_track > 5000:
        step = n_track // 5000
        track_lon = ship_lon[::step]
        track_lat = ship_lat[::step]
    else:
        track_lon = ship_lon
        track_lat = ship_lat

    if len(track_lon) > 0:
        ax_main.plot(track_lon, track_lat, "ro", markersize=2, fillstyle="none", alpha=0.7)

    # Axis limits from valid data
    valid_mask = np.logical_and(~np.isnan(gridded), gridded != 0)
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) > 0:
        lon_vals = lon_grid[valid_rows, valid_cols]
        lat_vals = lat_grid[valid_rows, valid_cols]
        lon_min, lon_max = lon_vals.min(), lon_vals.max()
        lat_min, lat_max = lat_vals.min(), lat_vals.max()
    else:
        lon_min, lon_max = lon_grid.min(), lon_grid.max()
        lat_min, lat_max = lat_grid.min(), lat_grid.max()

    mean_lat = (lat_min + lat_max) / 2
    aspect_ratio = 1.0 / np.cos(np.deg2rad(mean_lat))

    ax_main.set_xlim(lon_min, lon_max)
    ax_main.set_ylim(lat_min, lat_max)

    def format_coord(val, pos):
        return f"{val:.4f}"

    ax_main.xaxis.set_major_formatter(FuncFormatter(format_coord))
    ax_main.yaxis.set_major_formatter(FuncFormatter(format_coord))
    ax_main.set_xlabel("Longitude (°)")
    ax_main.set_ylabel("Latitude (°)")
    ax_main.set_title(title_str, fontsize=11)
    ax_main.set_aspect(aspect_ratio)
    ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Ship polar inset
    if ship_speeds and ship_headings:
        ax_ship = ax_main.inset_axes([0.88, 0.85, 0.11, 0.14], projection="polar")
        ax_ship.set_theta_zero_location("N")
        ax_ship.set_theta_direction(-1)
        headings_rad = np.deg2rad(ship_headings[-n_frames:])
        speeds = ship_speeds[-n_frames:]
        ages = np.linspace(0, 1, len(speeds))
        ax_ship.scatter(headings_rad, speeds, c=ages, cmap="viridis", s=6, alpha=0.8)
        ax_ship.set_xticklabels([])
        ax_ship.tick_params(labelsize=4)
        ax_ship.set_facecolor("white")
        ax_ship.patch.set_alpha(0.8)
        ax_ship.grid(True, linewidth=0.3, alpha=0.5)
        ax_main.text(
            0.87, 0.99, "Ship", transform=ax_main.transAxes, fontsize=7, ha="right", va="top"
        )

    # Wind polar inset
    if wind_speeds and wind_dirs:
        ax_wind = ax_main.inset_axes([0.88, 0.01, 0.11, 0.14], projection="polar")
        ax_wind.set_theta_zero_location("N")
        ax_wind.set_theta_direction(-1)
        wind_rad = np.deg2rad(wind_dirs[-n_frames:])
        speeds = wind_speeds[-n_frames:]
        ages = np.linspace(0, 1, len(speeds))
        ax_wind.scatter(wind_rad, speeds, c=ages, cmap="viridis", s=6, alpha=0.8)
        ax_wind.set_xticklabels([])
        ax_wind.tick_params(labelsize=4)
        ax_wind.set_facecolor("white")
        ax_wind.patch.set_alpha(0.8)
        ax_wind.grid(True, linewidth=0.3, alpha=0.5)
        ax_main.text(
            0.87, 0.01, "Wind", transform=ax_main.transAxes, fontsize=7, ha="right", va="bottom"
        )

    # Info panel
    ax_info.axis("off")
    start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_ts.strftime("%H:%M:%S")
    duration_s = (end_ts - start_ts).total_seconds()
    duration_str = f"{duration_s:.0f}s" if duration_s < 60 else f"{duration_s / 60:.1f}m"

    info_text = f"{start_str} - {end_str}  |  {n_frames} frames  |  Duration: {duration_str}  |  Frame {frame_number}"
    ax_info.text(
        0.5,
        0.5,
        info_text,
        transform=ax_info.transAxes,
        ha="center",
        va="center",
        fontfamily="monospace",
        fontsize=9,
    )

    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def generate_sliding_movie(args, config: "Config") -> None:
    """
    Generate a smooth movie using sliding window combine.

    Each output frame is a combination of frames within a time window.
    As time advances, old frames drop out and new frames are added,
    creating smooth transitions.

    Args:
        args: Parsed arguments with:
            - movie: Output path
            - window: Window duration in seconds (default 42)
            - output_every: Output every N frames (default 5)
            - fps: Movie frame rate
            - frames_dir: Optional directory for PNG frames
            - stime, etime, polar_path: Data selection
            - radar_height, max_frames, process: Processing options
        config: Config object
    """
    from wamos_tpw.multi_theta import MultiTheta as BearingTheta, MultiBearing as Bearing
    from wamos_tpw.deramp import Deramp
    from wamos_tpw.destreak import Destreak
    from wamos_tpw.dewind import Dewind
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.range import Range
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.theta import Theta  # Single-frame theta
    from wamos_tpw.combine_streaming import compute_grid_bounds_from_metadata
    from wamos_tpw.filenames import Filenames

    window_seconds = getattr(args, "window", 42)
    output_every = getattr(args, "output_every", 5)

    logging.info(f"Sliding window: {window_seconds}s, output every {output_every} frames")

    # Get all files in time range
    filenames = Filenames(
        stime=args.stime,
        etime=args.etime,
        polar_path=args.polar_path,
    )
    file_list = list(filenames)

    if not file_list:
        logging.warning("No files found in time range")
        return

    logging.info(f"Found {len(file_list)} files")

    # First pass: compute grid bounds
    logging.info("Computing grid bounds from metadata...")
    bounds_result = compute_grid_bounds_from_metadata(
        file_list, config, args.radar_height, args.max_frames
    )
    if bounds_result is None:
        logging.error("Failed to compute grid bounds")
        return

    (
        all_metadata,
        x_min,
        x_max,
        y_min,
        y_max,
        max_range,
        ship_x_meta,
        ship_y_meta,
        ship_speeds_meta,
        ship_headings_meta,
        wind_speeds_meta,
        wind_dirs_meta,
    ) = bounds_result

    logging.info(f"Grid bounds: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]")

    # Set up grid
    n_along, n_cross = 1200, 1600
    x_edges = np.linspace(x_min, x_max, n_cross + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, n_along + 1, dtype=np.float32)

    # Reference position from first file
    ref_lat = all_metadata[0].latitude or 0.0
    ref_lon = all_metadata[0].longitude or 0.0
    del all_metadata

    # Initialize sliding window combiner
    slider = SlidingWindowCombine(
        window_seconds=window_seconds,
        grid_shape=(n_along, n_cross),
        output_every=output_every,
    )
    slider.set_grid(x_edges, y_edges, ref_lat, ref_lon)

    # Set up output directory
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        frames_dir = args.frames_dir
        dir_context = contextlib.nullcontext(frames_dir)
    else:
        dir_context = tempfile.TemporaryDirectory()

    frame_paths = []
    output_frame_num = 0

    with dir_context as frames_dir:
        logging.info("Processing frames with sliding window...")

        # Process each file
        total_frames = 0
        chunk_size = 50

        # Progress bar for files
        file_pbar = tqdm(file_list, desc="Files", unit="file")
        for file_idx, fpath in enumerate(file_pbar):
            try:
                pf = PolarFile(fpath)
                file_frames = pf.frames
            except Exception as e:
                logging.warning(f"Failed to load {fpath}: {e}")
                continue

            # Process frames in chunks for theta calculation
            i = 0
            while i < len(file_frames):
                chunk_end = min(i + chunk_size, len(file_frames))
                if args.max_frames and total_frames + (chunk_end - i) > args.max_frames:
                    chunk_end = i + (args.max_frames - total_frames)

                chunk_frames = file_frames[i:chunk_end]
                if not chunk_frames:
                    break

                # Compute theta and bearing for chunk (for earth coordinate mapping)
                chunk_theta = BearingTheta(chunk_frames, config, refine=False)
                bearing_obj = Bearing(
                    chunk_theta, radar_height=args.radar_height, cache_coordinates=False
                )

                # Process and grid each frame
                for j, frame in enumerate(chunk_frames):
                    # Apply processing if requested (Destreak → Shadow → Deramp → Dewind)
                    if args.process:
                        frame_theta = Theta(frame)
                        destreaked = Destreak(frame)
                        shadow = Shadow(destreaked.intensity, frame_theta)
                        if shadow.theta_bias:
                            frame_theta.set_bias(shadow.theta_bias)
                        masked = shadow.mask(destreaked.intensity)
                        deramp = Deramp(masked, Range(frame))
                        dewind = Dewind(deramp.intensity, frame_theta)
                        frame.corrected_intensity = dewind.intensity

                    # Grid this frame
                    frame_sum, frame_count, ship_x, ship_y = _grid_single_frame(
                        frame,
                        j,
                        chunk_theta,
                        bearing_obj,
                        config,
                        x_edges,
                        y_edges,
                        ref_lat,
                        ref_lon,
                    )

                    # Get ship/wind data
                    meta = frame.metadata
                    ship_speed = meta.speed if meta.speed else None
                    ship_heading = meta.heading if meta.heading else None
                    wind_speed = (
                        meta.wind_speed if hasattr(meta, "wind_speed") and meta.wind_speed else None
                    )
                    wind_dir = (
                        meta.wind_direction
                        if hasattr(meta, "wind_direction") and meta.wind_direction
                        else None
                    )

                    # Slide window
                    timestamp = pd.Timestamp(frame.timestamp)
                    combined, start_time, end_time = slider.slide(
                        timestamp,
                        frame_sum,
                        frame_count,
                        ship_x=ship_x,
                        ship_y=ship_y,
                        ship_speed=ship_speed,
                        ship_heading=ship_heading,
                        wind_speed=wind_speed,
                        wind_dir=wind_dir,
                    )

                    # Output frame if ready
                    if combined is not None:
                        output_frame_num += 1
                        ts_str = start_time.strftime("%Y%m%dT%H%M%S")
                        output_path = f"{frames_dir}/frame_{output_frame_num:06d}_{ts_str}.png"

                        ship_x_arr, ship_y_arr = slider.get_recent_ship_track()

                        _save_sliding_frame(
                            combined,
                            x_edges,
                            y_edges,
                            output_path,
                            ref_lat,
                            ref_lon,
                            start_time,
                            end_time,
                            slider.n_frames_in_window,
                            output_frame_num,
                            ship_x_arr,
                            ship_y_arr,
                            slider.ship_speeds_all,
                            slider.ship_headings_all,
                            slider.wind_speeds_all,
                            slider.wind_dirs_all,
                        )
                        frame_paths.append(output_path)

                        # Update progress bar with output info
                        file_pbar.set_postfix(
                            frames=total_frames,
                            outputs=output_frame_num,
                            window=slider.n_frames_in_window,
                        )

                    # Clear frame cache
                    frame.clear_cache()
                    frame.deramped_intensity = None
                    frame.corrected_intensity = None

                    total_frames += 1
                    if args.max_frames and total_frames >= args.max_frames:
                        break

                # Clear chunk memory
                bearing_obj.clear_cache()
                del chunk_frames, chunk_theta, bearing_obj
                gc.collect()

                i = chunk_end
                if args.max_frames and total_frames >= args.max_frames:
                    break

            if args.max_frames and total_frames >= args.max_frames:
                break

        file_pbar.close()
        logging.info(f"Processed {total_frames} frames, generated {len(frame_paths)} output frames")

        if not frame_paths:
            logging.warning("No frames were rendered")
            return

        # Create movie with ffmpeg
        if not args.movie:
            logging.info(f"Frames saved to: {frames_dir}/")
        else:
            logging.info(f"Creating MP4 with ffmpeg ({args.fps} fps)...")

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                logging.error("ffmpeg not found. Install with: brew install ffmpeg")
                logging.info(f"Frames saved to: {frames_dir}/")
            else:
                ffmpeg_cmd = [
                    ffmpeg_path,
                    "-y",
                    "-framerate",
                    str(args.fps),
                    "-pattern_type",
                    "glob",
                    "-i",
                    f"{frames_dir}/frame_*.png",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    args.movie,
                ]

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"ffmpeg error: {result.stderr}")
                else:
                    logging.info(f"Movie saved to: {args.movie}")
                    if args.frames_dir:
                        logging.info(f"Frames preserved in: {frames_dir}/")


def add_subparser(subparsers) -> None:
    """Add sliding movie subparser to CLI."""
    from wamos_tpw.filenames import _timestamp_type, _directory_type

    parser = subparsers.add_parser(
        "sliding-movie",
        help="Generate smooth movie with sliding window combine",
        epilog="Timestamp formats: YYYY, YYYYMM, YYYYMMDD, YYYYMMDDHH, YYYYMMDDHHmm, "
        "YYYYMMDDTHHmm, YYYY-MM-DD, YYYY-MM-DDTHH:mm:ss",
    )
    parser.add_argument("stime", type=_timestamp_type, help="Start time")
    parser.add_argument("etime", type=_timestamp_type, help="End time")
    parser.add_argument("polar_path", type=_directory_type, help="Path to polar data directory")

    parser.add_argument("--movie", "-m", help="Output movie path (e.g., output.mp4)")
    parser.add_argument(
        "--window", "-w", type=float, default=42, help="Window duration in seconds (default: 42)"
    )
    parser.add_argument(
        "--output-every",
        "-e",
        type=int,
        default=5,
        help="Output frame every N input frames (default: 5)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Movie frames per second (default: 10)")
    parser.add_argument("--frames-dir", "-f", help="Directory to save frame images")
    parser.add_argument("--config", "-c", help="Path to YAML config file")
    parser.add_argument("--radar-height", type=float, help="Radar height above water (m)")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument(
        "--process", action="store_true", default=True, help="Apply processing (deramp + destreak)"
    )
    parser.add_argument(
        "--no-process", dest="process", action="store_false", help="Skip processing"
    )

    parser.set_defaults(func=_run_sliding_movie)


def _run_sliding_movie(args) -> None:
    """CLI entry point for sliding movie generation."""
    from wamos_tpw.config import Config

    config = Config(args.config) if args.config else Config()
    generate_sliding_movie(args, config)
