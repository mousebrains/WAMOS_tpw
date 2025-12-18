#! /usr/bin/env python3
#
# Movie generation for WAMOS combined data
# Creates MP4 movies from radar frame sequences
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wamos_tpw.config import WamosConfig


def _process_group(
    period_str: str,
    file_list: list[str],
    output_path: str,
    config_path: str | None,
    radar_height: float | None,
    max_frames: int | None,
    do_process: bool,
) -> tuple[str, str | None]:
    """
    Worker function to process a single group and save the frame.

    Runs in a separate process. Creates all necessary objects locally
    to avoid pickling issues.

    Args:
        period_str: Period timestamp string for logging
        file_list: List of file paths to load
        output_path: Path to save the output frame image
        config_path: Path to YAML config file (or None for defaults)
        radar_height: Radar height above water (m)
        max_frames: Maximum frames to process
        do_process: Whether to apply processing (deramp + destreak)

    Returns:
        Tuple of (period_str, output_path) on success, (period_str, None) on failure
    """
    import gc

    from wamos_tpw.bearing import Theta
    from wamos_tpw.combine import Combine
    from wamos_tpw.combine_plot import save_frame
    from wamos_tpw.config import WamosConfig
    from wamos_tpw.deramp import Deramp
    from wamos_tpw.destreak import Destreak
    from wamos_tpw.polarfile import PolarFile

    try:
        # Load config
        config = WamosConfig(config_path) if config_path else WamosConfig()

        # Load frames from file paths
        frames = []
        for fpath in file_list:
            try:
                pf = PolarFile(fpath)
                frames.extend(pf.frames)
            except Exception as e:
                logging.warning(f"Failed to load {fpath}: {e}")

        if max_frames:
            frames = frames[:max_frames]

        if not frames:
            return (period_str, None)

        # Process frames if requested
        if do_process:
            # Refine theta (shadow detection)
            theta = Theta(frames, config, refine=True)

            # Get shadow bounds from theta refinement
            shadow_start = theta.shadow_left_mean
            shadow_end = theta.shadow_right_mean

            # Deramp each frame
            for i, frame in enumerate(frames):
                bearing = theta.bearing_for_frame(i)
                deramp = Deramp(
                    frame,
                    config,
                    bearing=bearing,
                    shadow_start=shadow_start,
                    shadow_end=shadow_end,
                )
                frame.deramped_intensity = deramp.corrected_intensity

            # Destreak each frame (uses temporal neighbors)
            n_frames_count = len(frames)
            corrected = []
            for i, center in enumerate(frames):
                prev_frame = frames[i - 1] if i > 0 else None
                next_frame = frames[i + 1] if i < n_frames_count - 1 else None
                ds = Destreak(prev_frame, center, next_frame, config)
                corrected.append(ds.corrected_intensity)

            # Normalize
            normalized = _normalize_frames(corrected)
            for frame, corr in zip(frames, normalized):
                frame.corrected_intensity = corr

        # Create Combine and save frame
        combine = Combine(frames, config, radar_height=radar_height)
        save_frame(combine, output_path)

        # Clear memory
        combine.bearing._theta.clear_shadow_data()
        combine.bearing.clear_cache()
        for frame in frames:
            frame.clear_cache()
        del combine, frames
        gc.collect()

        return (period_str, output_path)

    except Exception as e:
        logging.error(f"Error processing group {period_str}: {e}")
        return (period_str, None)


def _normalize_frames(corrected: list) -> list:
    """
    Normalize corrected frames to [0, 1] range.

    Args:
        corrected: List of corrected intensity arrays

    Returns:
        List of normalized arrays
    """
    import numpy as np

    # Find global min/max across all frames
    all_values = np.concatenate([c.ravel() for c in corrected])
    valid = np.isfinite(all_values)
    if not valid.any():
        return corrected

    vmin = np.percentile(all_values[valid], 1)
    vmax = np.percentile(all_values[valid], 99)

    if vmax <= vmin:
        return corrected

    # Normalize each frame
    normalized = []
    for c in corrected:
        norm = (c - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
        normalized.append(norm)

    return normalized


def generate_movie(args, config: "WamosConfig") -> None:
    """
    Generate an MP4 movie from radar frame sequences.

    Processes groups in parallel using ProcessPoolExecutor.
    Supports checkpoint/resume.

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
        config: WamosConfig object
    """
    from wamos_tpw.processed import ProcessedFrames

    logging.info(f"Generating movie: {args.movie}")

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

            # Prepare work items
            work_items = []
            frame_paths = []
            skipped_count = 0

            for period, file_list in groups.items():
                # Format timestamp for filename: YYYYMMDD_HHMMSS (sorts chronologically)
                ts_str = str(period).replace("-", "").replace(":", "").replace(" ", "_")
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

        # Process groups in parallel
        n_work = len(work_items)
        if n_work == 0:
            if not frame_paths:
                logging.warning("No frames were rendered successfully")
                return
        else:
            # Determine number of workers
            n_workers = args.workers if args.workers else min(n_work, os.cpu_count() or 4)
            n_workers = max(1, min(n_workers, n_work))

            logging.info(f"Processing {n_work} groups with {n_workers} workers...")

            if n_workers == 1:
                # Sequential processing (simpler, better for debugging)
                for i, item in enumerate(work_items):
                    period_str, result_path = _process_group(**item)
                    if result_path:
                        frame_paths.append(result_path)
                        logging.info(f"Group {i + 1}/{n_work}: {period_str} completed")
                    else:
                        logging.warning(f"Group {i + 1}/{n_work}: {period_str} failed")
            else:
                # Parallel processing
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(_process_group, **item): item["period_str"]
                        for item in work_items
                    }

                    completed = 0
                    for future in as_completed(futures):
                        period_str = futures[future]
                        completed += 1
                        try:
                            _, result_path = future.result()
                            if result_path:
                                frame_paths.append(result_path)
                                logging.info(f"Group {completed}/{n_work}: {period_str} completed")
                            else:
                                logging.warning(f"Group {completed}/{n_work}: {period_str} failed")
                        except Exception as e:
                            logging.error(f"Group {completed}/{n_work}: {period_str} error: {e}")

        if not frame_paths:
            logging.warning("No frames were rendered successfully")
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

        # Create movie with ffmpeg
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
