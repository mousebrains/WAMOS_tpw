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
from argparse import Namespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wamos_tpw.config import WamosConfig


def generate_movie(args: Namespace, config: "WamosConfig") -> None:
    """
    Generate an MP4 movie from radar frame sequences.

    Streams through groups: load → process → render → save → discard
    to minimize memory usage. Supports checkpoint/resume.

    Args:
        args: Parsed command line arguments with:
            - movie: Output movie path (e.g., output.mp4)
            - frames_dir: Optional directory to save frame images
            - resume: Whether to resume from checkpoint
            - fps: Frames per second
            - stime, etime, polar_path: Time range and data path
            - groupby: Grouping frequency
            - radar_height: Radar height override
            - max_frames: Max frames per group
            - process: Whether to apply processing
        config: WamosConfig object
    """
    from wamos_tpw.combine import Combine
    from wamos_tpw.combine_plot import save_frame
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

            # Stream through groups: load → process → render → save → discard
            frame_paths = []
            group_idx = 0
            skipped_count = 0

            # Get grouped file lists (without loading frames yet)
            groups = pframes.groups()

            logging.debug("Starting group iteration...")
            for period, file_list in groups.items():
                # Format timestamp for filename: YYYYMMDD_HHMMSS (sorts chronologically)
                ts_str = str(period).replace("-", "").replace(":", "").replace(" ", "_")
                output_path = f"{frames_dir}/frame_{ts_str}.png"

                # Checkpoint: skip if frame already exists and --resume is set
                if args.resume and os.path.exists(output_path):
                    frame_paths.append(output_path)
                    skipped_count += 1
                    group_idx += 1
                    logging.debug(f"Group {group_idx}: {period} - skipped (checkpoint)")
                    continue

                # Now load the frames (only if we need to render)
                logging.debug(f"Group {group_idx}: {period} - loading...")
                frames = pframes.load_files(file_list)
                if args.max_frames:
                    frames = frames[: args.max_frames]
                if not frames:
                    logging.debug(f"Group {group_idx}: empty, skipping")
                    continue

                logging.debug(f"Group {group_idx}: {len(frames)} frames")

                # Process frames if requested
                if args.process:
                    logging.debug(f"Group {group_idx}: processing...")
                    pframes.refine_theta(frames)
                    pframes.deramp_frames(frames, show_progress=False)
                    corrected = pframes.destreak_frames(frames, show_progress=False)
                    normalized = pframes.normalize_frames(corrected)
                    for frame, corr in zip(frames, normalized):
                        frame.corrected_intensity = corr

                # Create Combine and save frame
                logging.debug(f"Group {group_idx}: combining...")
                combine = Combine(frames, config, radar_height=args.radar_height)
                logging.debug(f"Group {group_idx}: saving to {output_path}")
                save_frame(combine, output_path)
                frame_paths.append(output_path)

                # Clear memory before next group
                combine.bearing._theta.clear_shadow_data()
                combine.bearing.clear_cache()
                for frame in frames:
                    frame.clear_cache()
                del combine, frames
                gc.collect()

                group_idx += 1
                logging.info(f"Group {group_idx}: {period} completed")

        if not frame_paths:
            logging.warning("No frames were rendered successfully")
            return

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
