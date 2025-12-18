#! /usr/bin/env python3
#
# Movie generation for WAMOS combined data
# Creates MP4 movies from radar frame sequences
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import contextlib
import gc
import logging
import os
import shutil
import subprocess
import tempfile
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from wamos_tpw.config import WamosConfig

# Re-export for backward compatibility (noqa: F401)
from wamos_tpw.combine_shadow import (  # noqa: F401
    compute_chunk_shadow_offset as _compute_chunk_shadow_offset,
    detect_shadow_edges as _detect_shadow_edges,
)
from wamos_tpw.combine_streaming import (  # noqa: F401
    compute_grid_bounds_from_metadata as _compute_grid_bounds_from_metadata,
    grid_frame_streaming as _grid_frame_streaming,
    load_file_metadata as _load_file_metadata,
    normalize_frames as _normalize_frames,
    process_single_frame as _process_single_frame,
)


def _save_gridded_frame(
    gridded,
    x_edges,
    y_edges,
    output_path: str,
    ref_lat: float,
    ref_lon: float,
    first_timestamp,
    last_timestamp,
    n_frames: int,
    ship_x,
    ship_y,
    ship_speeds,
    ship_headings,
    wind_speeds,
    wind_dirs,
) -> None:
    """Save the gridded data as a PNG image with ship/wind polar insets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EARTH_RADIUS = 6371000.0

    # Convert grid edges to lat/lon
    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    # Convert ship track from meters to lat/lon
    ship_lon = ref_lon + ship_x / meters_per_deg_lon
    ship_lat = ref_lat + ship_y / meters_per_deg_lat

    # Create meshgrid for pcolormesh
    xx, yy = np.meshgrid(x_edges, y_edges)
    lon_grid = ref_lon + xx / meters_per_deg_lon
    lat_grid = ref_lat + yy / meters_per_deg_lat
    del xx, yy

    # Calculate intensity limits
    valid_data = gridded[~np.isnan(gridded)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [1, 99])
    else:
        vmin, vmax = 0, 1
    del valid_data

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 1], hspace=0.02)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0])
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)

    # Title
    start_ts = pd.Timestamp(first_timestamp).floor("s")
    end_ts = pd.Timestamp(last_timestamp).ceil("s")
    title_str = f"{start_ts} to {end_ts}"

    # Main plot
    im = ax_main.pcolormesh(
        lon_grid, lat_grid, gridded, cmap="viridis", vmin=vmin, vmax=vmax, shading="flat"
    )
    plt.colorbar(im, ax=ax_main, label="Intensity")

    # Overlay ship track (subsample if too many points)
    n_track = len(ship_lon)
    if n_track > 5000:
        step = n_track // 5000
        track_lon = ship_lon[::step]
        track_lat = ship_lat[::step]
    else:
        track_lon = ship_lon
        track_lat = ship_lat
    ax_main.plot(track_lon, track_lat, "r-", linewidth=1.5)

    # Find valid data extent for axis limits
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
    ax_main.set_xlabel("Longitude (°)")
    ax_main.set_ylabel("Latitude (°)")
    ax_main.set_title(title_str, fontsize=11)
    ax_main.set_aspect(aspect_ratio)
    ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Ship polar inset (upper right corner)
    if ship_speeds and ship_headings:
        ax_ship = ax_main.inset_axes([0.88, 0.85, 0.11, 0.14], projection="polar")
        ax_ship.set_theta_zero_location("N")
        ax_ship.set_theta_direction(-1)
        headings_rad = np.deg2rad(ship_headings)
        ages = np.linspace(0, 1, len(ship_speeds))
        ax_ship.scatter(headings_rad, ship_speeds, c=ages, cmap="viridis", s=6, alpha=0.8)
        ax_ship.set_xticklabels([])
        ax_ship.tick_params(labelsize=4)
        ax_ship.set_facecolor("white")
        ax_ship.patch.set_alpha(0.8)
        ax_ship.grid(True, linewidth=0.3, alpha=0.5)
        ax_main.text(
            0.87, 0.99, "Ship", transform=ax_main.transAxes, fontsize=7, ha="right", va="top"
        )

    # Wind polar inset (lower right corner)
    if wind_speeds and wind_dirs:
        ax_wind = ax_main.inset_axes([0.88, 0.01, 0.11, 0.14], projection="polar")
        ax_wind.set_theta_zero_location("N")
        ax_wind.set_theta_direction(-1)
        wind_rad = np.deg2rad(wind_dirs)
        ages = np.linspace(0, 1, len(wind_speeds))
        ax_wind.scatter(wind_rad, wind_speeds, c=ages, cmap="viridis", s=6, alpha=0.8)
        ax_wind.set_xticklabels([])
        ax_wind.tick_params(labelsize=4)
        ax_wind.set_facecolor("white")
        ax_wind.patch.set_alpha(0.8)
        ax_wind.grid(True, linewidth=0.3, alpha=0.5)
        ax_main.text(
            0.87, 0.01, "Wind", transform=ax_main.transAxes, fontsize=7, ha="right", va="bottom"
        )

    # Info panel with timestamp overlay
    ax_info.axis("off")

    # Format timestamp range for display
    start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_ts.strftime("%H:%M:%S")
    duration_s = (end_ts - start_ts).total_seconds()
    duration_str = f"{duration_s:.0f}s" if duration_s < 60 else f"{duration_s / 60:.1f}m"

    info_text = f"{start_str} - {end_str}  |  {n_frames} frames  |  Duration: {duration_str}"
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

    # Save
    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def _process_group(
    period_str: str,
    file_list: list[str],
    output_path: str,
    config_path: str | None,
    radar_height: float | None,
    max_frames: int | None,
    do_process: bool,
    chunk_size: int = 50,
) -> tuple[str, str | None, float]:
    """
    Process a single group and save the frame.

    Uses streaming/chunked processing to minimize memory usage:
    1. First pass: Load metadata only to compute grid bounds
    2. Second pass: Process frames in chunks, accumulate into grid, discard

    Args:
        period_str: Period timestamp string for logging
        file_list: List of file paths to load
        output_path: Path to save the output frame image
        config_path: Path to YAML config file (or None for defaults)
        radar_height: Radar height above water (m)
        max_frames: Maximum frames to process
        do_process: Whether to apply processing (deramp + destreak)
        chunk_size: Number of frames to process at a time (default 50)

    Returns:
        Tuple of (period_str, output_path, elapsed_seconds) on success,
        (period_str, None, elapsed_seconds) on failure
    """
    from wamos_tpw.bearing import Theta, Bearing
    from wamos_tpw.config import WamosConfig
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.combine_shadow import compute_chunk_shadow_offset
    from wamos_tpw.combine_streaming import (
        compute_grid_bounds_from_metadata,
        process_single_frame,
        grid_frame_streaming,
    )

    start_time = time.perf_counter()

    try:
        # Load config
        config = WamosConfig(config_path) if config_path else WamosConfig()

        # Step 1: Compute grid bounds from metadata only (parallel loading)
        bounds_result = compute_grid_bounds_from_metadata(
            file_list, config, radar_height, max_frames
        )
        if bounds_result is None:
            return (period_str, None, time.perf_counter() - start_time)

        (
            all_metadata,
            x_min,
            x_max,
            y_min,
            y_max,
            max_range,
            ship_x,
            ship_y,
            ship_speeds,
            ship_headings,
            wind_speeds,
            wind_dirs,
        ) = bounds_result
        n_total_frames = len(all_metadata)
        del all_metadata  # Free metadata list

        # Set up grid
        n_along, n_cross = 1200, 1600
        x_edges = np.linspace(x_min, x_max, n_cross + 1, dtype=np.float32)
        y_edges = np.linspace(y_min, y_max, n_along + 1, dtype=np.float32)

        # For rotated grid, we'd need to compute track angle - use non-rotated for simplicity
        sum_total = np.zeros((n_along, n_cross), dtype=np.float64)
        count_total = np.zeros((n_along, n_cross), dtype=np.int32)

        # Reference position for coordinate conversion
        ref_lat = None
        ref_lon = None
        first_timestamp = None
        last_timestamp = None

        # Step 2: Process frames in chunks
        frame_idx_global = 0

        for fpath in file_list:
            try:
                pf = PolarFile(fpath)
                file_frames = pf.frames
            except Exception as e:
                logging.warning(f"Failed to load {fpath}: {e}")
                continue

            # Process this file's frames in chunks
            i = 0
            while i < len(file_frames):
                # Determine chunk boundaries
                chunk_end = min(i + chunk_size, len(file_frames))
                if max_frames and frame_idx_global + (chunk_end - i) > max_frames:
                    chunk_end = i + (max_frames - frame_idx_global)

                chunk_frames = file_frames[i:chunk_end]
                if not chunk_frames:
                    break

                # Track timestamps
                if first_timestamp is None:
                    first_timestamp = chunk_frames[0].timestamp
                    ref_lat = chunk_frames[0].metadata.latitude or 0.0
                    ref_lon = chunk_frames[0].metadata.longitude or 0.0
                last_timestamp = chunk_frames[-1].timestamp

                # Compute theta for this chunk (no cross-chunk refinement)
                theta = Theta(chunk_frames, config, refine=False)

                # Per-chunk theta refinement using shadow edge detection
                # ALWAYS compute offset - it's needed for correct coordinate mapping
                offset, shadow_start, shadow_end = compute_chunk_shadow_offset(
                    chunk_frames, theta, config
                )

                # Apply offset to theta bearing arrays for correct coordinate calculation
                # This corrects the bearing used by Bearing.xy_earth() and in_shadow()
                for j in range(len(chunk_frames)):
                    theta._bearing_per_frame[j] = (theta._bearing_per_frame[j] - offset) % 360
                theta._bearing = np.concatenate(theta._bearing_per_frame)

                # Create Bearing for coordinate calculation (now uses corrected bearings)
                bearing_obj = Bearing(theta, radar_height=radar_height, cache_coordinates=False)

                if do_process:
                    # Sequential deramp + destreak for all frames in chunk
                    # Note: offset already applied to theta, so pass 0 for deramp offset
                    for j, frame in enumerate(chunk_frames):
                        process_single_frame(
                            j,
                            frame,
                            theta,
                            config,
                            0.0,  # offset already applied to theta
                            shadow_start,
                            shadow_end,
                        )

                # Sequential gridding
                for j, frame in enumerate(chunk_frames):
                    grid_frame_streaming(
                        frame,
                        j,
                        theta,
                        bearing_obj,
                        config,
                        x_edges,
                        y_edges,
                        sum_total,
                        count_total,
                        ref_lat,
                        ref_lon,
                    )

                # Clear chunk memory
                theta.clear_shadow_data()
                bearing_obj.clear_cache()
                for frame in chunk_frames:
                    frame.clear_cache()
                    frame.deramped_intensity = None
                    frame.corrected_intensity = None
                del chunk_frames, theta, bearing_obj
                gc.collect()

                frame_idx_global += chunk_end - i
                i = chunk_end

                if max_frames and frame_idx_global >= max_frames:
                    break

            if max_frames and frame_idx_global >= max_frames:
                break

        # Compute final gridded values
        with np.errstate(invalid="ignore"):
            gridded = (sum_total / count_total).astype(np.float32)
        gridded[count_total == 0] = np.nan
        del sum_total, count_total

        # Save the frame image
        _save_gridded_frame(
            gridded,
            x_edges,
            y_edges,
            output_path,
            ref_lat,
            ref_lon,
            first_timestamp,
            last_timestamp,
            n_total_frames,
            ship_x,
            ship_y,
            ship_speeds,
            ship_headings,
            wind_speeds,
            wind_dirs,
        )

        del gridded
        gc.collect()

        return (period_str, output_path, time.perf_counter() - start_time)

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logging.error(f"Error processing group {period_str}: {e}")
        import traceback

        traceback.print_exc()
        return (period_str, None, elapsed)


def _report_memory_profile() -> None:
    """Report memory profiling results if tracemalloc is active."""
    if not tracemalloc.is_tracing():
        return

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert to human-readable format
    def format_bytes(size: float) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(size) < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    logging.info("=== Memory Profile ===")
    logging.info(f"  Current: {format_bytes(current)}")
    logging.info(f"  Peak:    {format_bytes(peak)}")
    logging.info("======================")


def generate_movie(args, config: "WamosConfig") -> None:
    """
    Generate an MP4 movie from radar frame sequences.

    Processes groups in parallel using ProcessPoolExecutor.
    Supports checkpoint/resume and optional memory profiling.

    Args:
        args: Parsed command line arguments with:
            - movie: Output movie path (e.g., output.mp4)
            - frames_dir: Optional directory to save frame images
            - resume: Whether to resume from checkpoint
            - fps: Frames per second
            - workers: Number of parallel workers
            - stime, etime, polar_path: Time range and data path
            - groupby: Grouping frequency
            - radar_height: Radar height override
            - max_frames: Max frames per group
            - process: Whether to apply processing
            - profile_memory: Whether to enable memory profiling
        config: WamosConfig object
    """
    from wamos_tpw.processed import ProcessedFrames

    # Start memory profiling if requested
    profile_memory = getattr(args, "profile_memory", False)
    if profile_memory:
        tracemalloc.start()
        logging.info("Memory profiling enabled")

    if args.movie:
        logging.info(f"Generating movie: {args.movie}")
    else:
        logging.info("Generating frames (no movie)")

    # Checkpoint/resume validation
    if args.resume and not args.frames_dir:
        logging.warning("--resume requires --frames-dir; ignoring --resume")
        args.resume = False

    # Use persistent directory if specified, otherwise temp directory
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        frames_dir = args.frames_dir
        dir_context = contextlib.nullcontext(frames_dir)
    else:
        dir_context = tempfile.TemporaryDirectory()

    with dir_context as frames_dir:
        if args.frames_dir:
            logging.info(f"Saving frames to: {frames_dir}")

        logging.debug("Opening ProcessedFrames...")
        with ProcessedFrames(
            stime=args.stime,
            etime=args.etime,
            polar_path=args.polar_path,
            groupby=args.groupby,
            config=config,
            radar_height=args.radar_height,
        ) as pframes:
            logging.info(f"Discovered {len(pframes)} files")

            # Get grouped file lists (without loading frames yet)
            groups = pframes.groups()

            # Calculate expected files per group (median) for sparse group detection
            group_sizes = [len(files) for files in groups.values()]
            if group_sizes:
                expected_files = sorted(group_sizes)[len(group_sizes) // 2]  # median
                min_files_threshold = max(1, expected_files // 3)  # at least 1/3 of expected
            else:
                expected_files = 0
                min_files_threshold = 1

            # Prepare work items
            work_items = []
            frame_paths = []
            skipped_count = 0
            sparse_count = 0

            for period, file_list in groups.items():
                # Skip sparse groups (less than 1/3 of expected files)
                if len(file_list) < min_files_threshold:
                    sparse_count += 1
                    logging.debug(
                        f"Group {period}: skipped (sparse: {len(file_list)}/{expected_files} files)"
                    )
                    continue

                # Format timestamp for filename: YYYYmmddTHHMMSS (sorts chronologically)
                ts = pd.Timestamp(period)
                ts_str = ts.strftime("%Y%m%dT%H%M%S")
                output_path = f"{frames_dir}/frame_{ts_str}.png"

                # Checkpoint: skip if frame already exists and --resume is set
                if args.resume and os.path.exists(output_path):
                    frame_paths.append(output_path)
                    skipped_count += 1
                    logging.debug(f"Group {period}: skipped (checkpoint)")
                    continue

                work_items.append(
                    {
                        "period_str": str(period),
                        "file_list": file_list,
                        "output_path": output_path,
                        "config_path": args.config,
                        "radar_height": args.radar_height,
                        "max_frames": args.max_frames,
                        "do_process": args.process,
                    }
                )

        # Log sparse group skipping
        if sparse_count > 0:
            logging.info(
                f"Skipped {sparse_count} sparse groups (<{min_files_threshold} files, "
                f"expected ~{expected_files})"
            )

        # Process groups in parallel
        n_work = len(work_items)
        if n_work == 0:
            if not frame_paths:
                logging.warning("No frames were rendered successfully")
                _report_memory_profile()
                return
        else:
            # Determine number of workers
            n_workers = args.workers if args.workers else min(n_work, os.cpu_count() or 4)
            n_workers = max(1, min(n_workers, n_work))

            logging.info(f"Processing {n_work} groups with {n_workers} parallel workers...")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for item in work_items:
                    future = executor.submit(_process_group, **item)
                    futures[future] = item["period_str"]

                completed = 0
                for future in as_completed(futures):
                    period_str = futures[future]
                    completed += 1

                    # Format timestamp floored to second
                    ts = pd.Timestamp(period_str).floor("s")
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

                    try:
                        _, result_path, elapsed = future.result()
                        if result_path:
                            frame_paths.append(result_path)
                            logging.info(
                                f"Group {completed}/{n_work}: {ts_str} completed in {elapsed:.1f}s"
                            )
                        else:
                            logging.warning(
                                f"Group {completed}/{n_work}: {ts_str} failed after {elapsed:.1f}s"
                            )
                    except Exception as e:
                        logging.error(f"Group {completed}/{n_work}: {ts_str} error: {e}")

        if not frame_paths:
            logging.warning("No frames were rendered successfully")
            _report_memory_profile()
            return

        # Sort frame paths to ensure chronological order
        frame_paths.sort()

        rendered_count = len(frame_paths) - skipped_count
        if skipped_count > 0:
            logging.info(
                f"Resumed from checkpoint: {skipped_count} existing, {rendered_count} rendered"
            )
        else:
            logging.info(f"Rendered {len(frame_paths)} frames")

        # Create movie with ffmpeg (if requested)
        if not args.movie:
            logging.info(f"Frames saved to: {frames_dir}/")
        else:
            logging.info(f"Creating MP4 with ffmpeg ({args.fps} fps)...")

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                logging.error("ffmpeg not found. Install with: brew install ffmpeg")
                if not args.frames_dir:
                    # Copy frames to output directory as fallback
                    fallback_dir = args.movie.replace(".mp4", "_frames")
                    os.makedirs(fallback_dir, exist_ok=True)
                    for i, path in enumerate(frame_paths):
                        if path:
                            shutil.copy(path, f"{fallback_dir}/frame_{i:06d}.png")
                    logging.info(f"Frames saved to: {fallback_dir}/")
                else:
                    logging.info(f"Frames saved to: {frames_dir}/")
                logging.info("To create movie manually:")
                logging.info(
                    f"  ffmpeg -framerate {args.fps} -pattern_type glob -i '{frames_dir}/frame_*.png' "
                    f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
                    f"-c:v libx264 -pix_fmt yuv420p {args.movie}"
                )
            else:
                # Use ffmpeg with H.264 codec for web compatibility
                # Use glob pattern since filenames include timestamps
                ffmpeg_cmd = [
                    ffmpeg_path,
                    "-y",  # Overwrite output
                    "-framerate",
                    str(args.fps),
                    "-pattern_type",
                    "glob",
                    "-i",
                    f"{frames_dir}/frame_*.png",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # H.264 needs even dimensions
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",  # Quality (lower = better, 18-28 typical)
                    "-pix_fmt",
                    "yuv420p",  # Web compatibility
                    "-movflags",
                    "+faststart",  # Web streaming
                    args.movie,
                ]

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"ffmpeg error: {result.stderr}")
                else:
                    logging.info(f"Movie saved to: {args.movie}")
                    if args.frames_dir:
                        logging.info(f"Frames preserved in: {frames_dir}/")

    # Report memory profile at end of successful run
    _report_memory_profile()
