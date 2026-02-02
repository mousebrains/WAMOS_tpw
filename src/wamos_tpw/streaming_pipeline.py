#! /usr/bin/env python3
#
# Streaming merge pipeline for WAMOS polar data
#
# Starts processing files immediately as they're discovered, rather than
# waiting for all files to be found first.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import queue as _queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Iterator

import numpy as np

from wamos_tpw.merged_image import MergedImage, TimeWindowConfig
from wamos_tpw.streaming_filenames import (
    DiscoveredFile,
    StreamingFilenames,
    WindowTracker,
    create_time_windows_from_bounds,
)
from wamos_tpw.window import WindowAccumulator

if TYPE_CHECKING:
    from wamos_tpw.config import Config

logger = logging.getLogger(__name__)


class StreamingMergePipeline:
    """
    Streaming merge pipeline that processes files as they're discovered.

    Unlike FilesMergePipeline which requires all files upfront, this pipeline:
    - Starts file discovery in background
    - Begins processing as soon as first files are found
    - Completes windows as discovery progresses

    This dramatically reduces time-to-first-output for large datasets.

    Example::

        pipeline = StreamingMergePipeline(
            stime='2022-03-01',
            etime='2022-03-31',
            polar_path='/data/POLAR',
            config=config,
        )
        for merged in pipeline.iter_merged():
            # Process merged image
            save_netcdf(merged)
    """

    def __init__(
        self,
        stime: np.datetime64 | str,
        etime: np.datetime64 | str,
        polar_path: str,
        config: "Config | None" = None,
        window_config: TimeWindowConfig | None = None,
        n_workers: int | None = None,
        tolerance: float = 1.2,
        qTiming: bool = False,
        qProgress: bool = True,
        max_windows: int | None = None,
    ):
        """
        Initialize the streaming merge pipeline.

        Args:
            stime: Start time for processing
            etime: End time for processing
            polar_path: Root directory containing polar files
            config: YAML configuration
            window_config: Time window configuration
            n_workers: Number of parallel workers
            tolerance: Multiplier for repeat_time to accept frame pair
            qTiming: Enable timing statistics
            qProgress: Show progress bars
            max_windows: Maximum number of windows to process (None = all)
        """
        self._stime = np.datetime64(stime, "ns")
        self._etime = np.datetime64(etime, "ns")
        self._polar_path = polar_path
        self._config = config
        self._window_config = window_config or TimeWindowConfig()
        self._n_workers = n_workers
        self._tolerance = tolerance
        self._qTiming = qTiming
        self._qProgress = qProgress

        # Pre-create windows based on time bounds
        self._windows = create_time_windows_from_bounds(
            self._stime,
            self._etime,
            window_seconds=self._window_config.window_seconds,
            overlap=self._window_config.overlap_fraction,
        )

        # Limit windows if max_windows is specified
        if max_windows is not None and max_windows > 0:
            total_windows = len(self._windows)
            if max_windows < total_windows:
                self._windows = self._windows[:max_windows]
                logger.info(
                    "Limiting to first %d of %d windows (--max-windows)",
                    max_windows,
                    total_windows,
                )

        # Statistics
        self._n_files_discovered = 0
        self._n_files_processed = 0
        self._n_merged = 0

    @property
    def n_windows(self) -> int:
        """Number of time windows."""
        return len(self._windows)

    def iter_merged(self) -> Iterator[MergedImage]:
        """
        Yield merged images as windows complete.

        Starts file discovery in background and processes files as they arrive.
        Windows are merged and yielded as soon as all their files are processed.
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

        # Create window tracker
        window_tracker = WindowTracker(
            windows=self._windows,
            min_frames_per_window=self._window_config.min_frames_per_window,
        )

        # Map file_id -> list of window indices
        file_to_windows: dict[int, list[int]] = defaultdict(list)

        # Frames collected per window: window_idx -> list of frame_data
        window_frames: dict[int, list[dict]] = defaultdict(list)
        window_time_ranges: dict[int, tuple[np.datetime64, np.datetime64]] = {}
        for start, end, idx in self._windows:
            window_time_ranges[idx] = (start, end)

        # Track completed windows
        completed_windows: set[int] = set()

        # Serialize config
        config_dict = self._config._config if self._config else None

        # Create executor
        executor = PriorityProcessExecutor(
            max_workers=self._n_workers,
            task_handlers=TASK_HANDLERS,
        )
        executor.start()
        shm_manager = SharedMemoryManager()

        # Start streaming file discovery
        streaming = StreamingFilenames(
            self._stime,
            self._etime,
            self._polar_path,
            workers=max(2, (self._n_workers or 4) // 2),  # Use some workers for discovery
        )
        streaming.start()

        t0_total = time.perf_counter()

        # Backpressure control
        max_pending = (self._n_workers or 4) * 3
        pending_files = 0
        pending_interp = 0

        # Triplet collection
        triplet_collector = TripletCollector(total_files=1_000_000)  # Upper bound

        # Track files that have completed interpolation
        files_with_all_frames_interpolated: set[int] = set()
        file_frame_counts: dict[int, int] = {}
        file_frames_interpolated: dict[int, int] = defaultdict(int)
        file_shm_names: dict[int, list[str]] = defaultdict(list)

        # File submission queue (discovered files waiting to be submitted)
        files_to_submit: list[DiscoveredFile] = []
        discovery_complete = False

        # Progress bars
        pbar_discovery = tqdm(
            desc="Discovering",
            unit="file",
            disable=not self._qProgress,
        )
        pbar_files = tqdm(
            desc="Loading",
            unit="file",
            disable=not self._qProgress,
        )
        pbar_interp = tqdm(
            desc="Interpolating",
            unit="frame",
            disable=not self._qProgress,
        )
        pbar_merged = tqdm(
            total=len(self._windows),
            desc="Merging",
            unit="window",
            disable=not self._qProgress,
        )

        # Merge thread
        merge_input: _queue.Queue[tuple[int, list[dict]]] = _queue.Queue()
        merge_output: _queue.Queue[tuple[int, MergedImage | None, float]] = _queue.Queue()
        merge_shutdown = threading.Event()
        max_queued_merges = 4
        pending_merges = 0

        def _merge_window(window_idx: int, frames: list[dict]) -> MergedImage | None:
            """Merge frames into a single image."""
            if not frames:
                return None

            # Get grid parameters from first frame
            first = frames[0]
            if "x_edges" not in first:
                return None

            # Create accumulator
            accumulator = WindowAccumulator(
                x_edges=first["x_edges"],
                y_edges=first["y_edges"],
                grid_spacing=first.get("grid_spacing", 10.0),
                utm_zone=first.get("utm_zone", 0),
                hemisphere=first.get("hemisphere", "north"),
                center_lat=first.get("center_lat", 0.0),
                center_lon=first.get("center_lon", 0.0),
            )

            # Add all frames
            for frame_data in frames:
                if "projected_intensity" not in frame_data:
                    continue
                accumulator.add_projected(
                    projected_intensity=frame_data["projected_intensity"],
                    projected_count=frame_data["projected_count"],
                    timestamp=frame_data["timestamp"],
                    heading=frame_data.get("heading", 0.0),
                    ship_speed=frame_data.get("ship_speed"),
                    wind_speed=frame_data.get("wind_speed"),
                    wind_direction=frame_data.get("wind_direction"),
                )

            if accumulator.n_frames == 0:
                return None

            return accumulator.finalize(window_index=window_idx)

        def _merge_thread_fn():
            while not merge_shutdown.is_set():
                try:
                    window_idx, frames = merge_input.get(timeout=0.1)
                except _queue.Empty:
                    continue
                t0 = time.perf_counter()
                merged = _merge_window(window_idx, frames)
                elapsed = time.perf_counter() - t0
                merge_output.put((window_idx, merged, elapsed))

        merge_thread = threading.Thread(target=_merge_thread_fn, daemon=True)
        merge_thread.start()

        _merged_results: list[MergedImage] = []

        def _collect_merges():
            nonlocal pending_merges
            while True:
                try:
                    window_idx, merged, elapsed = merge_output.get_nowait()
                    pending_merges -= 1
                    if merged is not None:
                        self._n_merged += 1
                        pbar_merged.update(1)
                        _merged_results.append(merged)
                    completed_windows.add(window_idx)
                except _queue.Empty:
                    break

        def _wait_for_merge_room():
            nonlocal pending_merges
            while pending_merges >= max_queued_merges:
                _collect_merges()
                if pending_merges >= max_queued_merges:
                    time.sleep(0.01)

        def _submit_file(file: DiscoveredFile) -> None:
            nonlocal pending_files
            task_data = (file.filepath, file.file_id, config_dict, self._qTiming)
            executor.submit(Priority.LOW, "process_file", task_data)
            pending_files += 1

        def _submit_from_queue() -> int:
            """Submit files from queue up to backpressure limit."""
            submitted = 0
            while files_to_submit and pending_files < max_pending:
                file = files_to_submit.pop(0)
                _submit_file(file)
                submitted += 1
            return submitted

        # Iterator over discovery batches (non-blocking)
        batch_iter = streaming.iter_batches(timeout=0.1)
        batch_exhausted = False

        # Main processing loop
        while True:
            # Check for completed merges
            _collect_merges()
            while _merged_results:
                yield _merged_results.pop(0)

            # Check termination condition
            if (
                discovery_complete
                and not files_to_submit
                and pending_files == 0
                and pending_interp == 0
                and pending_merges == 0
            ):
                break

            # Try to get more discovered files (non-blocking)
            if not batch_exhausted:
                try:
                    batch = next(batch_iter)
                    for file in batch.files:
                        # Assign file to windows
                        windows = window_tracker.assign_file(file)
                        file_to_windows[file.file_id] = windows

                        # Queue for submission
                        files_to_submit.append(file)
                        self._n_files_discovered += 1
                        pbar_discovery.update(1)

                    # Mark windows as discovery-complete based on time progress
                    window_tracker.mark_discovery_complete_before(batch.hour_end_ns)

                    if batch.is_final:
                        window_tracker.mark_all_discovery_complete()
                        discovery_complete = True
                        batch_exhausted = True
                        pbar_discovery.close()

                except StopIteration:
                    window_tracker.mark_all_discovery_complete()
                    discovery_complete = True
                    batch_exhausted = True
                    pbar_discovery.close()

            # Submit files from queue
            _submit_from_queue()

            # Process executor results
            result = executor.get_result(timeout=0.05)
            if result is None:
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
                self._n_files_processed += 1

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

                # Submit interpolation tasks
                for prev, current, next_frame in triplet_collector.ready_triplets():
                    task_data = (prev, current, next_frame, self._tolerance, True, None)
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
                        window_tracker.mark_file_processed(file_idx)

                        if file_idx in file_shm_names:
                            shm_manager.release_many(file_shm_names[file_idx])
                            del file_shm_names[file_idx]

                        triplet_collector.prune_emitted()

                    # Route frame to windows
                    for window_idx in file_to_windows.get(file_idx, []):
                        if window_idx in completed_windows:
                            continue
                        start_time, end_time = window_time_ranges[window_idx]
                        if start_time <= timestamp < end_time:
                            window_frames[window_idx].append(data)

                    # Check for windows ready to merge
                    for window_idx in window_tracker.get_ready_windows():
                        if window_idx in completed_windows:
                            continue

                        # Verify all files for this window are interpolated
                        needed = window_tracker.get_window_files(window_idx)
                        if not needed.issubset(files_with_all_frames_interpolated):
                            continue

                        _wait_for_merge_room()
                        frames = window_frames.pop(window_idx, [])
                        merge_input.put((window_idx, frames))
                        pending_merges += 1

                # Check for more ready triplets
                for prev, current, next_frame in triplet_collector.ready_triplets():
                    task_data = (prev, current, next_frame, self._tolerance, True, None)
                    executor.submit(Priority.MEDIUM, "interpolate", task_data)
                    pending_interp += 1

        # Cleanup
        merge_shutdown.set()
        merge_thread.join(timeout=2.0)
        streaming.stop()
        executor.shutdown()
        shm_manager.cleanup()

        pbar_files.close()
        pbar_interp.close()
        pbar_merged.close()

        elapsed = time.perf_counter() - t0_total
        logger.info(
            f"Streaming pipeline complete: {self._n_files_discovered} files discovered, "
            f"{self._n_files_processed} processed, {self._n_merged} windows merged "
            f"in {elapsed:.1f}s"
        )

        # Yield any remaining results
        _collect_merges()
        while _merged_results:
            yield _merged_results.pop(0)


# CLI support
def _add_arguments(parser) -> None:
    """Add command arguments."""
    from wamos_tpw.filenames import _directory_type, _timestamp_type
    from wamos_tpw.files_pipeline import parse_duration

    parser.add_argument("stime", type=_timestamp_type, help="Start time")
    parser.add_argument("etime", type=_timestamp_type, help="End time")
    parser.add_argument("polar_path", type=_directory_type, help="Polar file directory")
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")

    # Window configuration
    parser.add_argument(
        "--window",
        type=parse_duration,
        default=60.0,
        metavar="DURATION",
        help="Window duration (default: 60s). Accepts: 60, 60s, 1.5m, 1h, 0.5d",
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


def run(args) -> None:
    """Run streaming pipeline with full output support."""
    import logging
    from pathlib import Path

    from wamos_tpw.config import Config
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

    pipeline = StreamingMergePipeline(
        stime=args.stime,
        etime=args.etime,
        polar_path=str(args.polar_path),
        config=config,
        window_config=window_config,
        n_workers=args.workers,
        tolerance=args.tolerance,
        qTiming=args.timing,
        qProgress=args.progress,
    )

    logging.info("Created streaming pipeline with %d time windows", pipeline.n_windows)

    # Only accumulate merged images in memory when bulk output is requested
    needs_bulk = bool(args.mp4 or args.kml or args.kmz or args.plot)
    merged_images = [] if needs_bulk else None
    n_merged = 0
    first_shown = False

    for merged in pipeline.iter_merged():
        n_merged += 1
        if merged_images is not None:
            merged_images.append(merged)

        # Show first image immediately if --plot is requested
        if args.plot and not first_shown:
            first_shown = True
            show_single_merged_image(merged)

        # Write per-window output files
        if args.output_dir:
            if args.format in ("netcdf", "both"):
                write_merged_netcdf(merged, args.output_dir)
            if args.format in ("png", "both"):
                write_merged_png(merged, args.output_dir)
            if args.geotiff:
                write_geotiff(merged, args.output_dir)

    logging.info("Created %d merged images (streaming mode)", n_merged)

    # Bulk outputs (require all images in memory)
    if merged_images:
        if args.kml:
            write_kml(merged_images, args.kml)
        if args.kmz:
            write_kmz(merged_images, args.kmz)
        if args.plot:
            show_merged_viewer(merged_images)
        if args.mp4:
            write_mp4_movie(merged_images, args.mp4, fps=args.fps, release=True)
        del merged_images


def add_subparser(subparsers) -> None:
    """Register the 'stream-pipeline' subcommand."""
    p = subparsers.add_parser(
        "stream-pipeline",
        help="Streaming merge pipeline (processes files as discovered)",
        description="Merge radar frames using streaming file discovery. "
        "Starts processing immediately as files are discovered, ideal for large datasets.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


if __name__ == "__main__":
    from wamos_tpw.cli_utils import create_standalone_main

    main = create_standalone_main(
        _add_arguments,
        run,
        "Streaming merge pipeline for WAMOS radar data",
    )
    main()
