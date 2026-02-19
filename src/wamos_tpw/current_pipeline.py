#! /usr/bin/env python3
#
# Surface current extraction pipeline for WAMOS radar data
#
# Processes sequential radar frames through the existing frame interpolation
# pipeline, then feeds blocks of individual projected frames into the 3D FFT
# current extraction algorithm.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Surface current extraction pipeline and CLI for WAMOS radar data.

This module provides:

1. **CurrentPipeline** — Wraps the existing :class:`FilesMergePipeline`
   processing infrastructure to collect individual projected frames (rather
   than averaging them) and feed blocks of N frames into the 3D FFT current
   extraction algorithm.

2. **CLI interface** — ``wamos current`` subcommand with depth, block size,
   sub-region size, and other tuning parameters.

3. **NetCDF output** — CF-1.8 NetCDF files with ``ux``, ``uy``, ``speed``,
   ``direction``, and ``snr`` variables.

Example::

    wamos current "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR \\
        --ship-data ./output/ -o ./currents/
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wamos_tpw.current import CurrentMap, FrameCube
from wamos_tpw.grid import compute_common_grid_from_stats
from wamos_tpw.merged_image import TimeWindowConfig
from wamos_tpw.window import create_time_windows

if TYPE_CHECKING:
    from wamos_tpw.config import Config

logger = logging.getLogger(__name__)

# Default config values (same as current.py)
_BLOCK_FRAMES_DEFAULT = 32
_BLOCK_OVERLAP_DEFAULT = 0.5


# ============================================================
# Pipeline Class
# ============================================================


class CurrentPipeline:
    """Extract surface currents from sequential radar frames.

    Processes frames in parallel using the same
    :class:`PriorityProcessExecutor` infrastructure as
    :class:`FilesMergePipeline`, but collects individual projected frames
    into blocks of N frames and runs :class:`CurrentExtractor` on each block.

    Args:
        filenames: List of polar file paths (time-sorted).
        config: YAML configuration.
        block_frames: Number of frames per analysis block.
        block_overlap: Overlap fraction between consecutive blocks.
        n_workers: Number of parallel workers.
        tolerance: Time tolerance multiplier for frame pairing.
        qTiming: Enable timing statistics.
        qProgress: Show progress bars.
        ship_data_dir: Directory with instrument NetCDF files.
        grid_spacing: Grid cell size in meters.
    """

    def __init__(
        self,
        filenames: list[str],
        config: Config | None = None,
        block_frames: int = _BLOCK_FRAMES_DEFAULT,
        block_overlap: float = _BLOCK_OVERLAP_DEFAULT,
        n_workers: int | None = None,
        tolerance: float = 1.2,
        qTiming: bool = False,
        qProgress: bool = True,
        ship_data_dir: str | None = None,
        grid_spacing: float | None = None,
    ):
        self._files = filenames
        self._config = config
        self._block_frames = block_frames
        self._block_overlap = block_overlap
        self._n_workers = n_workers or min(len(filenames), os.cpu_count() or 1)
        self._tolerance = tolerance
        self._qTiming = qTiming
        self._qProgress = qProgress
        self._ship_data_dir = ship_data_dir
        self._grid_spacing = grid_spacing

        # Estimate window duration from file count and expected rotation period
        # We'll use ~1.5s per frame as a rough estimate for time windowing
        estimated_dt = 1.5
        window_seconds = block_frames * estimated_dt

        self._window_config = TimeWindowConfig(
            window_seconds=window_seconds,
            overlap_fraction=block_overlap,
            min_frames_per_window=max(4, block_frames // 4),
        )

        # Create time windows
        self._windows = create_time_windows(filenames, self._window_config)

        # Statistics
        self._n_processed = 0
        self._timings: dict[str, float] = {}

    @property
    def n_windows(self) -> int:
        """Number of analysis blocks."""
        return len(self._windows)

    def _build_cube(self, frames: list[dict]) -> FrameCube | None:
        """Build a FrameCube from a list of interpolated frame dicts.

        Args:
            frames: Interpolated frame data dicts.

        Returns:
            FrameCube or None if insufficient data.
        """
        if len(frames) < self._window_config.min_frames_per_window:
            return None

        frames.sort(key=lambda x: x["timestamp"])

        # Collect grid computation inputs
        max_ranges = []
        range_resolutions = []
        position_stats = []

        for f in frames:
            max_ranges.append(f.get("ground_range_max", 3000.0))
            frame_gp = f.get("grid_params") or {}
            range_resolutions.append(frame_gp.get("grid_spacing", 7.5))
            if "position_stats" in f:
                position_stats.append(f["position_stats"])

        if not position_stats or len(position_stats) != len(frames):
            return None

        grid_params = compute_common_grid_from_stats(
            position_stats=position_stats,
            max_ranges=max_ranges,
            range_resolutions=range_resolutions,
        )

        return FrameCube.from_frame_dicts(frames, grid_params)

    def iter_current_maps(self) -> Iterator[CurrentMap]:
        """Yield CurrentMap objects as analysis blocks complete.

        Uses the same streaming architecture as :class:`FilesMergePipeline`
        but instead of merging frames, builds FrameCubes and extracts currents.
        """
        from tqdm import tqdm

        from wamos_tpw.interpolator_tasks import TASK_HANDLERS
        from wamos_tpw.priority_executor import (
            Priority,
            PriorityProcessExecutor,
            SharedMemoryManager,
            TripletCollector,
        )

        if not self._windows:
            logger.warning("No time windows to process")
            return

        # Pre-compute window membership
        window_needs: dict[int, set[int]] = {}
        file_to_windows: dict[int, list[int]] = defaultdict(list)
        window_time_ranges: dict[int, tuple[np.datetime64, np.datetime64]] = {}

        for window_idx, (start_time, end_time, file_indices) in enumerate(self._windows):
            window_needs[window_idx] = set(file_indices)
            window_time_ranges[window_idx] = (start_time, end_time)
            for file_idx in file_indices:
                file_to_windows[file_idx].append(window_idx)

        window_frames: dict[int, list[dict]] = defaultdict(list)
        completed_windows: set[int] = set()

        all_file_indices = set()
        for indices in window_needs.values():
            all_file_indices.update(indices)
        file_indices_to_process = sorted(all_file_indices)

        config_dict = self._config._config if self._config else None

        executor = PriorityProcessExecutor(
            max_workers=self._n_workers,
            task_handlers=TASK_HANDLERS,
        )
        executor.start()
        shm_manager = SharedMemoryManager()

        t0_total = time.perf_counter()

        file_iter = iter(file_indices_to_process)
        total_files = len(file_indices_to_process)
        files_submitted = 0
        pending_files = 0
        pending_interp = 0
        max_pending = int(self._n_workers * 3.0)

        # Track file/frame progress
        files_with_all_frames_interpolated: set[int] = set()
        file_frame_counts: dict[int, int] = {}
        file_frames_interpolated: dict[int, int] = defaultdict(int)
        file_shm_names: dict[int, list[str]] = defaultdict(list)

        triplet_collector = TripletCollector(total_files=len(self._files))

        def submit_batch(n: int) -> int:
            nonlocal files_submitted, pending_files
            loaded_not_interp = len(file_frame_counts) - len(files_with_all_frames_interpolated)
            max_loaded_ahead = max_pending * 2
            room = min(n, max_pending - pending_files, max_loaded_ahead - loaded_not_interp)
            if room <= 0:
                return 0
            submitted = 0
            for _ in range(room):
                try:
                    file_idx = next(file_iter)
                    filepath = self._files[file_idx]
                    task_data = (filepath, file_idx, config_dict, self._qTiming)
                    executor.submit(Priority.LOW, "process_file", task_data)
                    files_submitted += 1
                    pending_files += 1
                    submitted += 1
                except StopIteration:
                    break
            return submitted

        pbar_files = tqdm(
            total=total_files, desc="Loading files", unit="file", disable=not self._qProgress
        )
        pbar_interp = tqdm(
            total=total_files, desc="Interpolating", unit="frame", disable=not self._qProgress
        )
        pbar_current = tqdm(
            total=len(self._windows),
            desc="Extracting currents",
            unit="block",
            disable=not self._qProgress,
        )

        submit_batch(max_pending)

        while pending_files > 0 or pending_interp > 0 or files_submitted < total_files:
            result = executor.get_result(timeout=0.1)
            if result is None:
                if files_submitted < total_files and pending_files < max_pending:
                    submit_batch(max_pending - pending_files)
                continue

            if result.error:
                logger.error("Error: %s", result.error)
                if result.task_type == "process_file":
                    pending_files -= 1
                    pbar_files.update(1)
                else:
                    pending_interp -= 1
                    pbar_interp.update(1)
                continue

            if result.task_type == "process_file":
                pending_files -= 1
                pbar_files.update(1)
                submit_batch(1)

                data = result.data
                file_idx = data["file_index"]
                file_frame_counts[file_idx] = len(data["frames"])

                for frame_data in data["frames"]:
                    triplet_collector.add(frame_data)
                    if frame_data.theta_shm:
                        shm_manager.register(frame_data.theta_shm[0], refcount=1)
                        file_shm_names[file_idx].append(frame_data.theta_shm[0])
                    if frame_data.ground_range_shm:
                        shm_manager.register(frame_data.ground_range_shm[0], refcount=1)
                        file_shm_names[file_idx].append(frame_data.ground_range_shm[0])
                    if frame_data.intensity_shm:
                        shm_manager.register(frame_data.intensity_shm[0], refcount=1)
                        file_shm_names[file_idx].append(frame_data.intensity_shm[0])

                triplet_collector.file_complete(file_idx, len(data["frames"]))

                for prev, current, next_frame in triplet_collector.ready_triplets():
                    task_data = (
                        prev,
                        current,
                        next_frame,
                        self._tolerance,
                        True,
                        None,
                        self._ship_data_dir,
                        self._grid_spacing,
                    )
                    executor.submit(Priority.MEDIUM, "interpolate", task_data)
                    pending_interp += 1

            elif result.task_type == "interpolate":
                pending_interp -= 1
                pbar_interp.update(1)
                data = result.data

                if data["success"]:
                    file_idx = data["file_index"]
                    timestamp = data["timestamp"]

                    file_frames_interpolated[file_idx] += 1
                    if file_frames_interpolated[file_idx] >= file_frame_counts.get(file_idx, 0):
                        files_with_all_frames_interpolated.add(file_idx)
                        if file_idx in file_shm_names:
                            shm_manager.release_many(file_shm_names[file_idx])
                            del file_shm_names[file_idx]
                        triplet_collector.prune_emitted()

                    for window_idx in file_to_windows[file_idx]:
                        if window_idx in completed_windows:
                            continue
                        start_time, end_time = window_time_ranges[window_idx]
                        if start_time <= timestamp < end_time:
                            window_frames[window_idx].append(data)

                    for window_idx in file_to_windows[file_idx]:
                        if window_idx in completed_windows:
                            continue
                        needed = window_needs[window_idx]
                        if needed.issubset(files_with_all_frames_interpolated):
                            frames = window_frames.pop(window_idx, [])
                            completed_windows.add(window_idx)

                            cube = self._build_cube(frames)
                            if cube is not None:
                                try:
                                    current_map = CurrentMap.from_cube(cube, self._config)
                                    self._n_processed += 1
                                    pbar_current.update(1)
                                    yield current_map
                                except Exception:
                                    logger.exception(
                                        "Current extraction failed for window %d", window_idx
                                    )

                for prev, current, next_frame in triplet_collector.ready_triplets():
                    task_data = (
                        prev,
                        current,
                        next_frame,
                        self._tolerance,
                        True,
                        None,
                        self._ship_data_dir,
                        self._grid_spacing,
                    )
                    executor.submit(Priority.MEDIUM, "interpolate", task_data)
                    pending_interp += 1

        pbar_files.close()
        pbar_interp.close()
        pbar_current.close()

        triplet_collector.prune_emitted()
        shm_manager.cleanup()
        executor.shutdown()

        self._timings["total"] = time.perf_counter() - t0_total
        logger.info(
            "Extracted currents from %d blocks in %.2fs",
            self._n_processed,
            self._timings["total"],
        )


# ============================================================
# NetCDF Output
# ============================================================


def write_current_netcdf(current_map: CurrentMap, output_dir: str) -> str:
    """Write a CurrentMap to a CF-1.8 NetCDF file.

    Args:
        current_map: CurrentMap to write.
        output_dir: Output directory.

    Returns:
        Path to created file.
    """
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not installed, skipping NetCDF output")
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = (
        np.datetime_as_string(current_map.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = (
        np.datetime_as_string(current_map.end_time, unit="s").replace(":", "-").replace("T", "_")
    )
    filename = f"current_{start_str}_to_{end_str}.nc"
    filepath = output_dir / filename

    ds = xr.Dataset(
        data_vars={
            "ux": (
                ["y", "x"],
                current_map.ux.astype(np.float32),
                {
                    "long_name": "Eastward sea water velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "uy": (
                ["y", "x"],
                current_map.uy.astype(np.float32),
                {
                    "long_name": "Northward sea water velocity",
                    "standard_name": "northward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "speed": (
                ["y", "x"],
                current_map.speed.astype(np.float32),
                {
                    "long_name": "Sea water speed",
                    "standard_name": "sea_water_speed",
                    "units": "m s-1",
                },
            ),
            "direction": (
                ["y", "x"],
                current_map.direction.astype(np.float32),
                {
                    "long_name": "Direction of sea water velocity",
                    "standard_name": "direction_of_sea_water_velocity",
                    "units": "degree",
                    "comment": "Direction current flows TO, clockwise from north",
                },
            ),
            "snr": (
                ["y", "x"],
                current_map.snr.astype(np.float32),
                {
                    "long_name": "Signal to noise ratio of current estimate",
                    "units": "1",
                },
            ),
        },
        coords={
            "x": (
                ["x"],
                current_map.tile_x_centers.astype(np.float32),
                {
                    "long_name": "Distance east from center",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                ["y"],
                current_map.tile_y_centers.astype(np.float32),
                {
                    "long_name": "Distance north from center",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "time_start": current_map.start_time,
            "time_end": current_map.end_time,
        },
        attrs={
            "title": "WAMOS surface current estimates",
            "institution": "WAMOS TPW",
            "source": "wamos current",
            "history": f"Created {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            "center_latitude": current_map.center_lat,
            "center_longitude": current_map.center_lon,
            "water_depth_m": current_map.depth if np.isfinite(current_map.depth) else -1.0,
        },
    )

    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds.data_vars}
    ds.to_netcdf(filepath, encoding=encoding)

    logger.debug("Wrote current estimates to %s", filepath)
    return str(filepath)


# ============================================================
# CLI Interface
# ============================================================


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")

    # Current extraction parameters
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help="Water depth in meters (default: inf = deep water)",
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=_BLOCK_FRAMES_DEFAULT,
        help=f"Frames per analysis block (default: {_BLOCK_FRAMES_DEFAULT})",
    )
    parser.add_argument(
        "--block-overlap",
        type=float,
        default=_BLOCK_OVERLAP_DEFAULT,
        help=f"Overlap between blocks (default: {_BLOCK_OVERLAP_DEFAULT})",
    )
    parser.add_argument(
        "--sub-region-size",
        type=float,
        default=None,
        help="Sub-region side length in meters (default: 2000)",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=None,
        help="Max current speed to search in m/s (default: 3.0)",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        help="Minimum SNR to accept estimate (default: 1.5)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for current estimates",
    )
    parser.add_argument(
        "--format",
        choices=["netcdf", "png", "both"],
        default="netcdf",
        help="Output format (default: netcdf)",
    )

    # Processing options
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.2,
        help="Time tolerance multiplier (default: 1.2)",
    )
    parser.add_argument("--timing", "-t", action="store_true", help="Show timing statistics")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots")
    parser.add_argument(
        "--ship-data",
        type=str,
        default=None,
        help="Directory with instrument NetCDF files for high-frequency interpolation",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=None,
        help="Grid cell size in meters for earth projection (default: auto)",
    )

    # Progress bar options
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress", dest="progress", action="store_true", default=True, help="Show progress bars"
    )
    progress_group.add_argument(
        "--no-progress", dest="progress", action="store_false", help="Hide progress bars"
    )


def add_subparser(subparsers) -> None:
    """Register the 'current' subcommand."""
    p = subparsers.add_parser(
        "current",
        help="Extract surface currents via 3D FFT dispersion fitting",
        description="Extract surface current vectors from sequential radar images "
        "using 3D FFT and gravity wave dispersion relation fitting.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'current' command."""
    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames

    config = Config(args.config) if args.config else Config()

    # Apply CLI overrides to config
    if args.depth is not None:
        config["current.depth"] = args.depth
    if args.sub_region_size is not None:
        config["current.sub_region_size"] = args.sub_region_size
    if args.search_radius is not None:
        config["current.search_radius"] = args.search_radius
    if args.min_snr is not None:
        config["current.min_snr"] = args.min_snr

    filenames = Filenames(args.stime, args.etime, args.polar_path)
    files = list(filenames)

    if not files:
        logger.warning(
            "No files found in %s for time range %s to %s",
            args.polar_path,
            args.stime,
            args.etime,
        )
        return

    logger.info("Found %d files", len(files))

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", args.output_dir)

    # Pre-build ship data cache
    ship_data_dir = getattr(args, "ship_data", None)
    if ship_data_dir:
        from wamos_tpw.instruments.ship_data import ShipData

        sd = ShipData(Path(ship_data_dir))
        logger.info("Ship data: %s", sd)

    pipeline = CurrentPipeline(
        filenames=files,
        config=config,
        block_frames=args.block_frames,
        block_overlap=args.block_overlap,
        n_workers=args.workers,
        tolerance=args.tolerance,
        qTiming=args.timing,
        qProgress=args.progress,
        ship_data_dir=ship_data_dir,
        grid_spacing=getattr(args, "grid_spacing", None),
    )

    logger.info("Created %d analysis blocks", pipeline.n_windows)

    current_maps: list[CurrentMap] = []

    for current_map in pipeline.iter_current_maps():
        current_maps.append(current_map)

        if args.output_dir:
            if args.format in ("netcdf", "both"):
                write_current_netcdf(current_map, args.output_dir)
            if args.format in ("png", "both"):
                _write_current_png(current_map, args.output_dir)

    logger.info("Extracted %d current maps", len(current_maps))

    if args.plot and current_maps:
        _show_current_viewer(current_maps)


def _write_current_png(current_map: CurrentMap, output_dir: str) -> str:
    """Write a CurrentMap as a PNG quiver plot."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = (
        np.datetime_as_string(current_map.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = (
        np.datetime_as_string(current_map.end_time, unit="s").replace(":", "-").replace("T", "_")
    )
    filename = f"current_{start_str}_to_{end_str}.png"
    filepath = output_dir / filename

    from wamos_tpw.current_diagnostics import CurrentMapDiag

    fig = plt.figure(figsize=(18, 5))
    diag = CurrentMapDiag(current_map)
    diag.plot(fig=fig)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Wrote current plot to %s", filepath)
    return str(filepath)


def _show_current_viewer(current_maps: list[CurrentMap]) -> None:
    """Show interactive viewer for current maps."""
    import matplotlib.pyplot as plt

    from wamos_tpw.current_diagnostics import CurrentMapDiag

    for i, cm in enumerate(current_maps):
        fig = plt.figure(figsize=(18, 5))
        fig.suptitle(f"Current Map {i + 1}/{len(current_maps)}")
        diag = CurrentMapDiag(cm)
        diag.plot(fig=fig)

    plt.show()


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(
    _add_arguments, run, "Extract surface currents via 3D FFT dispersion fitting"
)

if __name__ == "__main__":
    main()
