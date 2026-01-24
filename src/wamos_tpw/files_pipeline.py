#! /usr/bin/env python3
#
# Frame merge pipeline for WAMOS polar data
#
# Merges multiple frames into motion-corrected composite images for movie creation.
# Uses overlapping time windows to produce smooth transitions between merged images.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

from wamos_tpw.grid import compute_common_grid, remap_to_common_grid
from wamos_tpw.merged_image import MergedImage, TimeWindowConfig
from wamos_tpw.window import WindowAccumulator, create_time_windows

if TYPE_CHECKING:
    from wamos_tpw.config import Config

logger = logging.getLogger(__name__)


# ============================================================
# Main Pipeline Class
# ============================================================


class FilesMergePipeline:
    """
    Merge radar frames into motion-corrected composite images.

    Processes frames in parallel, projects them onto a common UTM grid,
    and accumulates them into time-windowed composites.

    Example::

        pipeline = FilesMergePipeline(files, config, window_config)
        for merged in pipeline.iter_merged():
            # Process merged image (display, save, etc.)
    """

    def __init__(
        self,
        filenames: list[str],
        config: "Config | None" = None,
        window_config: TimeWindowConfig | None = None,
        n_workers: int | None = None,
        tolerance: float = 1.2,
        qTiming: bool = False,
        qProgress: bool = True,
    ):
        """
        Initialize the merge pipeline.

        Args:
            filenames: List of polar file paths to process (should be time-sorted)
            config: YAML configuration information
            window_config: Time window configuration
            n_workers: Number of parallel workers (default: auto)
            tolerance: Multiplier for repeat_time to accept frame pair
            qTiming: Enable timing statistics
            qProgress: Show progress bars
        """
        self._files = filenames
        self._config = config
        self._window_config = window_config or TimeWindowConfig()
        self._n_workers = n_workers or min(len(filenames), os.cpu_count() or 1)
        self._tolerance = tolerance
        self._qTiming = qTiming
        self._qProgress = qProgress

        # Create time windows
        self._windows = create_time_windows(filenames, self._window_config)

        # Statistics
        self._n_processed = 0
        self._n_merged = 0
        self._timings: dict[str, float] = {}

    @property
    def n_windows(self) -> int:
        """Number of time windows."""
        return len(self._windows)

    @property
    def window_config(self) -> TimeWindowConfig:
        """Return the window configuration."""
        return self._window_config

    def _merge_window(self, window_idx: int, frames: list[dict]) -> MergedImage | None:
        """
        Merge frames into a single MergedImage for a window.

        Args:
            window_idx: Index of this window
            frames: List of interpolated frame data dicts

        Returns:
            MergedImage or None if not enough valid frames
        """
        if len(frames) < self._window_config.min_frames_per_window:
            return None

        # Sort by timestamp
        frames.sort(key=lambda x: x["timestamp"])

        # Compute common grid for this window
        latitudes = [f["latitudes"] for f in frames if "latitudes" in f]
        longitudes = [f["longitudes"] for f in frames if "longitudes" in f]

        if not latitudes:
            return None

        max_ranges = []
        range_resolutions = []

        for f in frames:
            if "grid_params" in f and f["grid_params"]:
                max_ranges.append(f["grid_params"].get("n_x", 1000) * 10)
                range_resolutions.append(f["grid_params"].get("grid_spacing", 10))
            else:
                max_ranges.append(3000.0)
                range_resolutions.append(7.5)

        grid_params = compute_common_grid(
            latitudes=latitudes,
            longitudes=longitudes,
            max_ranges=max_ranges,
            range_resolutions=range_resolutions,
            resolution_scale=self._window_config.resolution_scale,
        )

        # Create accumulator
        accumulator = WindowAccumulator(
            x_edges=grid_params["x_edges"],
            y_edges=grid_params["y_edges"],
            grid_spacing=grid_params["grid_spacing"],
            utm_zone=grid_params["utm_zone"],
            hemisphere=grid_params["hemisphere"],
            center_lat=grid_params["center_lat"],
            center_lon=grid_params["center_lon"],
        )

        # Project each frame onto the common grid
        for frame_data in frames:
            if "latitudes" not in frame_data:
                continue

            headings = frame_data.get("headings")
            mean_heading = float(np.mean(headings)) if headings is not None else 0.0

            if frame_data.get("projected_intensity") is not None:
                proj_intensity = frame_data["projected_intensity"]
                proj_count = frame_data.get("projected_count")
                frame_grid_params = frame_data.get("grid_params", {})

                frame_x_edges_centered = frame_grid_params.get("x_edges")
                frame_y_edges_centered = frame_grid_params.get("y_edges")
                frame_center_lat = frame_grid_params.get("center_lat")
                frame_center_lon = frame_grid_params.get("center_lon")

                if (
                    frame_x_edges_centered is not None
                    and frame_y_edges_centered is not None
                    and frame_center_lat is not None
                    and frame_center_lon is not None
                ):
                    frame_center_x, frame_center_y = grid_params["transformer"].transform(
                        frame_center_lon, frame_center_lat
                    )
                    frame_x_edges_utm = frame_x_edges_centered + frame_center_x
                    frame_y_edges_utm = frame_y_edges_centered + frame_center_y

                    frame_sum, frame_count = remap_to_common_grid(
                        proj_intensity,
                        proj_count,
                        frame_x_edges_utm,
                        frame_y_edges_utm,
                        grid_params["x_edges_utm"],
                        grid_params["y_edges_utm"],
                        grid_params["n_x"],
                        grid_params["n_y"],
                    )
                else:
                    frame_sum = np.zeros((grid_params["n_y"], grid_params["n_x"]), dtype=np.float64)
                    frame_count = np.zeros((grid_params["n_y"], grid_params["n_x"]), dtype=np.int32)
                    if proj_intensity.shape == frame_sum.shape:
                        valid = ~np.isnan(proj_intensity)
                        frame_sum[valid] = proj_intensity[valid]
                        if proj_count is not None:
                            frame_count[valid] = proj_count[valid]
                        else:
                            frame_count[valid] = 1
            else:
                logger.debug(
                    "Frame (%d, %d) has no projected data",
                    frame_data.get("file_index", -1),
                    frame_data.get("frame_index", -1),
                )
                frame_sum = np.zeros((grid_params["n_y"], grid_params["n_x"]), dtype=np.float64)
                frame_count = np.zeros((grid_params["n_y"], grid_params["n_x"]), dtype=np.int32)

            accumulator.add_projected(
                projected_intensity=frame_sum,
                projected_count=frame_count,
                timestamp=frame_data["timestamp"],
                heading=mean_heading,
                ship_speed=frame_data.get("ship_speed"),
                wind_speed=frame_data.get("wind_speed"),
                wind_direction=frame_data.get("wind_direction"),
            )

        if accumulator.n_frames >= self._window_config.min_frames_per_window:
            return accumulator.finalize(
                window_index=window_idx,
                interpolate_gaps=self._window_config.interpolate_gaps,
            )
        return None

    def iter_merged(self) -> Iterator[MergedImage]:
        """
        Yield merged images as windows complete.

        Uses streaming architecture: windows are merged and yielded as soon as
        all their constituent files have been interpolated, rather than waiting
        for all files to complete first.
        """
        from tqdm import tqdm

        from wamos_tpw.interpolator import TASK_HANDLERS
        from wamos_tpw.priority_executor import (
            Priority,
            PriorityProcessExecutor,
            SharedMemoryManager,
            TripletCollector,
        )

        if not self._windows:
            logger.warning("No time windows to process")
            return

        # Pre-compute window membership: which windows need which files
        # window_needs[window_idx] = set of file indices still needed
        # file_to_windows[file_idx] = list of window indices that need this file
        window_needs: dict[int, set[int]] = {}
        file_to_windows: dict[int, list[int]] = defaultdict(list)
        window_time_ranges: dict[int, tuple[np.datetime64, np.datetime64]] = {}

        for window_idx, (start_time, end_time, file_indices) in enumerate(self._windows):
            window_needs[window_idx] = set(file_indices)
            window_time_ranges[window_idx] = (start_time, end_time)
            for file_idx in file_indices:
                file_to_windows[file_idx].append(window_idx)

        # Frames collected per window: window_idx -> list of frame_data
        window_frames: dict[int, list[dict]] = defaultdict(list)

        # Track completed windows to avoid re-processing
        completed_windows: set[int] = set()

        # Get all unique file indices
        all_file_indices = set()
        for indices in window_needs.values():
            all_file_indices.update(indices)
        file_indices_to_process = sorted(all_file_indices)

        # Serialize config
        config_dict = self._config._config if self._config else None

        # Create executor
        executor = PriorityProcessExecutor(
            max_workers=self._n_workers,
            task_handlers=TASK_HANDLERS,
        )
        executor.start()
        shm_manager = SharedMemoryManager()

        t0_total = time.perf_counter()

        # Lazy/batched submission to avoid queue buffer overflow
        # Submit tasks incrementally as workers consume them
        file_iter = iter(file_indices_to_process)
        total_files = len(file_indices_to_process)
        files_submitted = 0
        pending_files = 0
        pending_interp = 0

        # Initial batch size: enough to keep workers busy without overwhelming queue
        # Queue buffer on macOS is ~64KB; with ~500 bytes per task, ~100 tasks fit
        # Use 2x workers as target in-flight to maintain throughput
        max_pending = max(self._n_workers * 4, 100)

        def submit_batch(n: int) -> int:
            """Submit up to n file tasks. Returns number submitted."""
            nonlocal files_submitted, pending_files
            submitted = 0
            for _ in range(n):
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

        # Triplet collection
        triplet_collector = TripletCollector(total_files=len(self._files))

        # Track files that have completed interpolation (all frames done)
        files_with_all_frames_interpolated: set[int] = set()
        # Track how many frames each file has and how many are interpolated
        file_frame_counts: dict[int, int] = {}
        file_frames_interpolated: dict[int, int] = defaultdict(int)
        # Track shared memory names per file for cleanup
        file_shm_names: dict[int, list[str]] = defaultdict(list)

        # Progress bars
        pbar_files = tqdm(
            total=total_files,
            desc="Loading files",
            unit="file",
            disable=not self._qProgress,
        )
        pbar_interp = tqdm(
            total=total_files,
            desc="Interpolating",
            unit="frame",
            disable=not self._qProgress,
        )
        pbar_merged = tqdm(
            total=len(self._windows), desc="Merging", unit="window", disable=not self._qProgress
        )

        # Track interpolation count for timing statistics
        n_interp_completed = 0

        # Memory tracking
        try:
            import resource

            def get_max_memory_mb():
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        except ImportError:

            def get_max_memory_mb():
                return 0.0

        # Phase timing accumulators
        time_loading = 0.0
        time_interpolating = 0.0
        time_merging = 0.0

        # Submit initial batch to get workers started
        submit_batch(max_pending)

        # Main processing loop - submit more tasks as results arrive
        while pending_files > 0 or pending_interp > 0 or files_submitted < total_files:
            result = executor.get_result(timeout=0.1)
            if result is None:
                # No result yet - submit more if queue has room
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
                t_phase = time.perf_counter()
                pending_files -= 1
                pbar_files.update(1)

                # Submit replacement task to maintain queue pressure
                submit_batch(1)

                data = result.data
                file_idx = data["file_index"]

                # Track frame count for this file
                file_frame_counts[file_idx] = len(data["frames"])

                # Add frames to triplet collector
                for frame_data in data["frames"]:
                    triplet_collector.add(frame_data)

                    # Register shared memory and track for later cleanup
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

                # Check for ready triplets
                for prev, current, next_frame in triplet_collector.ready_triplets():
                    task_data = (
                        prev,
                        current,
                        next_frame,
                        self._tolerance,
                        True,
                        None,
                    )
                    executor.submit(Priority.MEDIUM, "interpolate", task_data)
                    pending_interp += 1

                time_loading += time.perf_counter() - t_phase

            elif result.task_type == "interpolate":
                t_phase = time.perf_counter()
                pending_interp -= 1
                pbar_interp.update(1)
                n_interp_completed += 1
                data = result.data

                if data["success"]:
                    file_idx = data["file_index"]
                    timestamp = data["timestamp"]

                    # Track interpolation progress for this file
                    file_frames_interpolated[file_idx] += 1
                    if file_frames_interpolated[file_idx] >= file_frame_counts.get(file_idx, 0):
                        files_with_all_frames_interpolated.add(file_idx)

                        # Release shared memory for this file's frames
                        if file_idx in file_shm_names:
                            shm_manager.release_many(file_shm_names[file_idx])
                            del file_shm_names[file_idx]

                        # Prune triplet collector to free memory
                        triplet_collector.prune_emitted()

                    # Route this frame to all windows that need it
                    for window_idx in file_to_windows[file_idx]:
                        if window_idx in completed_windows:
                            continue

                        start_time, end_time = window_time_ranges[window_idx]
                        if start_time <= timestamp < end_time:
                            window_frames[window_idx].append(data)

                    # Check if any windows are now complete
                    # A window is complete when all its files have finished interpolation
                    for window_idx in file_to_windows[file_idx]:
                        if window_idx in completed_windows:
                            continue

                        # Check if all files for this window are done
                        needed = window_needs[window_idx]
                        if needed.issubset(files_with_all_frames_interpolated):
                            # Window is ready to merge
                            t_merge = time.perf_counter()
                            frames = window_frames.get(window_idx, [])
                            merged = self._merge_window(window_idx, frames)
                            time_merging += time.perf_counter() - t_merge

                            if merged is not None:
                                self._n_merged += 1
                                pbar_merged.update(1)
                                yield merged

                            # Mark as completed and free memory
                            completed_windows.add(window_idx)
                            if window_idx in window_frames:
                                del window_frames[window_idx]

                # Check for more ready triplets
                for prev, current, next_frame in triplet_collector.ready_triplets():
                    task_data = (
                        prev,
                        current,
                        next_frame,
                        self._tolerance,
                        True,
                        None,
                    )
                    executor.submit(Priority.MEDIUM, "interpolate", task_data)
                    pending_interp += 1

                time_interpolating += time.perf_counter() - t_phase

        pbar_files.close()
        pbar_interp.close()
        pbar_merged.close()

        # Get memory stats before cleanup
        shm_stats = shm_manager.stats
        triplet_stats = {
            "items": triplet_collector.item_count,
            "emitted": triplet_collector.emitted_count,
            "pending": triplet_collector.pending_count,
        }

        # Clean up
        remaining_shm = shm_manager.cleanup()
        executor.shutdown()

        self._timings["total"] = time.perf_counter() - t0_total
        max_mem = get_max_memory_mb()

        logger.info(
            "Merged %d windows in %.2fs, max memory: %.1f MB",
            self._n_merged,
            self._timings["total"],
            max_mem,
        )

        if self._qTiming:
            print("\nTiming Statistics:")
            print(f"  Total time:    {self._timings['total']:.2f}s")
            print(f"  Loading:       {time_loading:.2f}s")
            print(f"  Interpolating: {time_interpolating:.2f}s")
            print(f"  Merging:       {time_merging:.2f}s")
            print(f"  Files processed: {len(file_indices_to_process)}")
            print(f"  Frames interpolated: {n_interp_completed}")
            print(f"  Windows merged: {self._n_merged}")
            print(f"  Max memory: {max_mem:.1f} MB")
            print("\nMemory Management:")
            print(f"  SharedMem registered: {shm_stats['registered']}")
            print(f"  SharedMem released:   {shm_stats['released']}")
            print(f"  SharedMem orphaned:   {remaining_shm}")
            print(f"  Triplet items remaining: {triplet_stats['items']}")
            print(f"  Triplet pending:      {triplet_stats['pending']}")


# ============================================================
# CLI Interface
# ============================================================


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")

    # Window configuration
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Window duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction between windows (default: 0.5 = 50%%)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Minimum frames per window (default: 5)",
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=1.0,
        help="Grid resolution multiplier (2.0 = 2x finer grid, default: 1.0)",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Fill NaN gaps using nearest neighbor interpolation",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for merged images",
    )
    parser.add_argument(
        "--format",
        choices=["netcdf", "png", "both"],
        default="netcdf",
        help="Output format (default: netcdf)",
    )

    # Movie and georeferenced output
    parser.add_argument(
        "--mp4",
        type=str,
        default=None,
        metavar="FILE",
        help="Generate MP4 movie file (requires ffmpeg)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for MP4 movie (default: 2.0)",
    )
    parser.add_argument(
        "--geotiff",
        action="store_true",
        help="Write georeferenced GeoTIFF files (requires rasterio)",
    )
    parser.add_argument(
        "--kml",
        type=str,
        default=None,
        metavar="FILE",
        help="Generate KML file with ground overlays for Google Earth",
    )
    parser.add_argument(
        "--kmz",
        type=str,
        default=None,
        metavar="FILE",
        help="Generate self-contained KMZ file (KML + images in ZIP archive)",
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
    parser.add_argument("--plot", action="store_true", help="Show interactive viewer")

    # Progress bar options (mutually exclusive)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show progress bars (default)",
    )
    progress_group.add_argument(
        "--no-progress", dest="progress", action="store_false", help="Hide progress bars"
    )


def add_subparser(subparsers) -> None:
    """Register the 'files-pipeline' subcommand."""
    p = subparsers.add_parser(
        "files-pipeline",
        help="Merge frames into motion-corrected composite images",
        description="Process radar files and merge into time-windowed composite images",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'files-pipeline' command."""
    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.merged_viewer import show_merged_viewer, show_single_merged_image
    from wamos_tpw.output_writers import (
        write_geotiff,
        write_kml,
        write_kmz,
        write_merged_netcdf,
        write_merged_png,
        write_mp4_movie,
    )

    config = Config(args.config) if args.config else Config()

    filenames = Filenames(args.stime, args.etime, args.polar_path)
    files = list(filenames)

    if not files:
        logging.warning(
            "No files found in %s for time range %s to %s",
            args.polar_path,
            args.stime,
            args.etime,
        )
        return

    logging.info("Found %d files", len(files))

    # Create window config
    window_config = TimeWindowConfig(
        window_seconds=args.window,
        overlap_fraction=args.overlap,
        min_frames_per_window=args.min_frames,
        resolution_scale=args.resolution_scale,
        interpolate_gaps=args.interpolate,
    )

    logging.info(
        "Window config: %.1fs duration, %.0f%% overlap, min %d frames, %.1fx resolution%s",
        window_config.window_seconds,
        window_config.overlap_fraction * 100,
        window_config.min_frames_per_window,
        window_config.resolution_scale,
        ", interpolate gaps" if window_config.interpolate_gaps else "",
    )

    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Output directory: %s", args.output_dir)

    # Create pipeline
    pipeline = FilesMergePipeline(
        filenames=files,
        config=config,
        window_config=window_config,
        n_workers=args.workers,
        tolerance=args.tolerance,
        qTiming=args.timing,
        qProgress=args.progress,
    )

    logging.info("Created %d time windows", pipeline.n_windows)

    # Collect merged images
    merged_images = []
    first_shown = False

    for merged in pipeline.iter_merged():
        merged_images.append(merged)

        # Show first image immediately if --plot is requested
        if args.plot and not first_shown:
            first_shown = True
            show_single_merged_image(merged)

        # Write output files
        if args.output_dir:
            if args.format in ("netcdf", "both"):
                write_merged_netcdf(merged, args.output_dir)
            if args.format in ("png", "both"):
                write_merged_png(merged, args.output_dir)
            if args.geotiff:
                write_geotiff(merged, args.output_dir)

    logging.info("Created %d merged images", len(merged_images))

    # Generate MP4 movie if requested
    if args.mp4 and merged_images:
        write_mp4_movie(merged_images, args.mp4, fps=args.fps)

    # Generate KML file with ground overlays if requested
    if args.kml and merged_images:
        write_kml(merged_images, args.kml)

    # Generate KMZ file (self-contained package) if requested
    if args.kmz and merged_images:
        write_kmz(merged_images, args.kmz)

    # Show full viewer if requested (after all images collected)
    if args.plot and merged_images:
        show_merged_viewer(merged_images)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Merge frames into composite images")

if __name__ == "__main__":
    main()
