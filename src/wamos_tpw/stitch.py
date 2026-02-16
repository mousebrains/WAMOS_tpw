#! /usr/bin/env python3
#
# Stitch command for combining outputs into movies and KMZ files
#
# Supports:
# - Combining NetCDF files into MP4 movies
# - Combining NetCDF files into KML/KMZ
# - Stitching multiple MP4 files into a larger movie
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.merged_image import MergedImage

logger = logging.getLogger(__name__)


# ============================================================
# NetCDF Loading
# ============================================================


def load_merged_from_netcdf(filepath: str | Path) -> MergedImage:
    """
    Load a MergedImage from a NetCDF file.

    Args:
        filepath: Path to NetCDF file

    Returns:
        MergedImage object reconstructed from file
    """
    import xarray as xr

    from wamos_tpw.merged_image import MergedImage

    ds = xr.open_dataset(filepath)

    # Extract coordinates
    x_centers = ds["x"].values
    y_centers = ds["y"].values

    # Reconstruct edges from centers (assuming uniform spacing)
    dx = x_centers[1] - x_centers[0] if len(x_centers) > 1 else ds.attrs.get("grid_spacing_m", 10.0)
    dy = y_centers[1] - y_centers[0] if len(y_centers) > 1 else ds.attrs.get("grid_spacing_m", 10.0)

    x_edges = np.concatenate([x_centers - dx / 2, [x_centers[-1] + dx / 2]])
    y_edges = np.concatenate([y_centers - dy / 2, [y_centers[-1] + dy / 2]])

    merged = MergedImage(
        intensity=ds["intensity"].values,
        x_edges=x_edges,
        y_edges=y_edges,
        start_time=np.datetime64(ds["time_start"].values),
        end_time=np.datetime64(ds["time_end"].values),
        n_frames=ds.attrs.get("n_frames", 1),
        utm_zone=ds.attrs.get("utm_zone", 10),
        hemisphere=ds.attrs.get("hemisphere", "north"),
        center_lat=ds.attrs.get("center_latitude", 0.0),
        center_lon=ds.attrs.get("center_longitude", 0.0),
        grid_spacing=ds.attrs.get("grid_spacing_m", 10.0),
        mean_heading=ds.attrs.get("mean_ship_heading_deg", 0.0),
        mean_ship_speed=ds.attrs.get("mean_ship_speed_m_s"),
        mean_wind_speed=ds.attrs.get("mean_wind_speed_m_s"),
        mean_wind_direction=ds.attrs.get("mean_wind_direction_deg"),
        window_index=ds.attrs.get("window_index", 0),
    )

    ds.close()
    return merged


def iter_netcdf_files(
    input_dir: str | Path, pattern: str = "merged_*.nc"
) -> Iterator[tuple[Path, np.datetime64]]:
    """
    Iterate over NetCDF files in a directory, sorted by timestamp.

    Args:
        input_dir: Directory containing NetCDF files
        pattern: Glob pattern for matching files

    Yields:
        Tuples of (filepath, start_time) sorted by start_time
    """
    import xarray as xr

    input_dir = Path(input_dir)
    files_with_times = []

    for filepath in input_dir.glob(pattern):
        try:
            ds = xr.open_dataset(filepath)
            start_time = np.datetime64(ds["time_start"].values)
            ds.close()
            files_with_times.append((filepath, start_time))
        except Exception as e:
            logger.warning("Could not read %s: %s", filepath, e)
            continue

    # Sort by start time
    files_with_times.sort(key=lambda x: x[1])

    for filepath, start_time in files_with_times:
        yield filepath, start_time


def load_netcdf_files(
    input_dir: str | Path,
    pattern: str = "merged_*.nc",
    max_files: int | None = None,
) -> list[MergedImage]:
    """
    Load all NetCDF files from a directory into MergedImage objects.

    Args:
        input_dir: Directory containing NetCDF files
        pattern: Glob pattern for matching files
        max_files: Maximum number of files to load (None = all)

    Returns:
        List of MergedImage objects sorted by start time
    """
    merged_images = []

    for i, (filepath, _) in enumerate(iter_netcdf_files(input_dir, pattern)):
        if max_files is not None and i >= max_files:
            break

        try:
            merged = load_merged_from_netcdf(filepath)
            merged_images.append(merged)
        except Exception as e:
            logger.warning("Could not load %s: %s", filepath, e)
            continue

    logger.info("Loaded %d merged images from %s", len(merged_images), input_dir)
    return merged_images


# ============================================================
# MP4 Stitching
# ============================================================


def stitch_mp4_files(
    input_files: list[str | Path],
    output_path: str | Path,
    reencode: bool = False,
) -> str:
    """
    Stitch multiple MP4 files into a single movie using ffmpeg.

    Args:
        input_files: List of input MP4 file paths (in order)
        output_path: Output MP4 file path
        reencode: If True, re-encode the video (slower but more compatible)
                  If False, use stream copy (fast, but files must be compatible)

    Returns:
        Path to created file
    """
    if not input_files:
        logger.warning("No input files for stitching")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary file list for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = Path(f.name)
        for input_file in input_files:
            # ffmpeg concat format requires 'file' prefix and escaped paths
            escaped_path = str(Path(input_file).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    try:
        if reencode:
            # Re-encode for maximum compatibility
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
        else:
            # Stream copy (fast, lossless)
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(output_path),
            ]

        logger.info("Stitching %d MP4 files into %s", len(input_files), output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr)
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        logger.info("Created stitched movie: %s", output_path)
        return str(output_path)

    finally:
        list_file.unlink(missing_ok=True)


def find_mp4_files(
    input_dir: str | Path,
    pattern: str = "*.mp4",
    sort_by_name: bool = True,
) -> list[Path]:
    """
    Find MP4 files in a directory.

    Args:
        input_dir: Directory to search
        pattern: Glob pattern for matching files
        sort_by_name: Sort files by name (default True)

    Returns:
        List of MP4 file paths
    """
    input_dir = Path(input_dir)
    files = list(input_dir.glob(pattern))

    if sort_by_name:
        files.sort()

    return files


# ============================================================
# CLI Interface
# ============================================================


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    subparsers = parser.add_subparsers(
        dest="stitch_command",
        title="stitch commands",
        description="Available stitching operations",
    )

    # Subcommand: images-to-movie
    images_movie = subparsers.add_parser(
        "images-to-movie",
        help="Create MP4 movie from NetCDF files",
        description="Load merged images from NetCDF files and create an MP4 movie.",
    )
    images_movie.add_argument("input_dir", type=str, help="Directory containing NetCDF files")
    images_movie.add_argument("output", type=str, help="Output MP4 file path")
    images_movie.add_argument(
        "--pattern", default="merged_*.nc", help="Glob pattern for NetCDF files"
    )
    images_movie.add_argument("--fps", type=float, default=2.0, help="Frames per second")
    images_movie.add_argument("--cmap", default="viridis", help="Colormap name")
    images_movie.add_argument("--dpi", type=int, default=150, help="Output resolution")
    images_movie.add_argument("--no-range-rings", action="store_true", help="Disable range rings")
    images_movie.add_argument("--no-inset", action="store_true", help="Disable ship/wind inset")
    images_movie.add_argument("--max-files", type=int, help="Maximum number of files to process")
    images_movie.set_defaults(func=_run_images_to_movie)

    # Subcommand: images-to-kml
    images_kml = subparsers.add_parser(
        "images-to-kml",
        help="Create KML file from NetCDF files",
        description="Load merged images from NetCDF files and create a KML file.",
    )
    images_kml.add_argument("input_dir", type=str, help="Directory containing NetCDF files")
    images_kml.add_argument("output", type=str, help="Output KML file path")
    images_kml.add_argument(
        "--pattern", default="merged_*.nc", help="Glob pattern for NetCDF files"
    )
    images_kml.add_argument("--max-files", type=int, help="Maximum number of files to process")
    images_kml.set_defaults(func=_run_images_to_kml)

    # Subcommand: images-to-kmz
    images_kmz = subparsers.add_parser(
        "images-to-kmz",
        help="Create KMZ file from NetCDF files",
        description="Load merged images from NetCDF files and create a self-contained KMZ file.",
    )
    images_kmz.add_argument("input_dir", type=str, help="Directory containing NetCDF files")
    images_kmz.add_argument("output", type=str, help="Output KMZ file path")
    images_kmz.add_argument(
        "--pattern", default="merged_*.nc", help="Glob pattern for NetCDF files"
    )
    images_kmz.add_argument("--max-files", type=int, help="Maximum number of files to process")
    images_kmz.set_defaults(func=_run_images_to_kmz)

    # Subcommand: movies
    movies = subparsers.add_parser(
        "movies",
        help="Stitch MP4 files into a larger movie",
        description="Concatenate multiple MP4 files into a single movie using ffmpeg.",
    )
    movies.add_argument("output", type=str, help="Output MP4 file path")
    movies.add_argument(
        "input_files",
        nargs="*",
        help="Input MP4 files (or use --input-dir)",
    )
    movies.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing MP4 files to stitch",
    )
    movies.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern for MP4 files when using --input-dir",
    )
    movies.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode video (slower but more compatible)",
    )
    movies.set_defaults(func=_run_stitch_movies)


def _run_images_to_movie(args) -> None:
    """Run images-to-movie subcommand."""
    from wamos_tpw.output_writers import write_mp4_movie

    merged_images = load_netcdf_files(
        args.input_dir,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    if not merged_images:
        logger.error("No merged images found")
        return

    write_mp4_movie(
        merged_images,
        args.output,
        fps=args.fps,
        cmap=args.cmap,
        dpi=args.dpi,
        range_rings=not args.no_range_rings,
        show_inset=not args.no_inset,
        release=True,  # Free memory as we go
    )


def _run_images_to_kml(args) -> None:
    """Run images-to-kml subcommand."""
    from wamos_tpw.output_writers import write_kml

    merged_images = load_netcdf_files(
        args.input_dir,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    if not merged_images:
        logger.error("No merged images found")
        return

    write_kml(merged_images, args.output)


def _run_images_to_kmz(args) -> None:
    """Run images-to-kmz subcommand."""
    from wamos_tpw.output_writers import write_kmz

    merged_images = load_netcdf_files(
        args.input_dir,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    if not merged_images:
        logger.error("No merged images found")
        return

    write_kmz(merged_images, args.output)


def _run_stitch_movies(args) -> None:
    """Run stitch movies subcommand."""
    # Collect input files
    if args.input_files:
        input_files = [Path(f) for f in args.input_files]
    elif args.input_dir:
        input_files = find_mp4_files(args.input_dir, pattern=args.pattern)
    else:
        logger.error("Must specify input files or --input-dir")
        return

    if not input_files:
        logger.error("No input MP4 files found")
        return

    logger.info("Found %d MP4 files to stitch", len(input_files))

    stitch_mp4_files(
        input_files,
        args.output,
        reencode=args.reencode,
    )


def add_subparser(subparsers) -> None:
    """Register the 'stitch' subcommand."""
    p = subparsers.add_parser(
        "stitch",
        help="Combine images/movies into larger outputs",
        description="Stitch NetCDF files into movies/KMZ, or combine MP4 files.",
    )
    _add_arguments(p)
    p.set_defaults(func=_run_stitch)


def _run_stitch(args) -> None:
    """Run the stitch command."""
    if not hasattr(args, "stitch_command") or args.stitch_command is None:
        # Print help if no subcommand given
        import argparse

        parser = argparse.ArgumentParser()
        _add_arguments(parser)
        parser.print_help()
        return

    # Call the appropriate subcommand function
    args.func(args)


if __name__ == "__main__":
    from wamos_tpw.cli_utils import create_standalone_main

    def _add_args_standalone(parser):
        _add_arguments(parser)

    main = create_standalone_main(
        _add_args_standalone,
        _run_stitch,
        "Stitch images and movies together",
    )
    main()
