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

from wamos_tpw.grid import (
    compute_common_grid,
    compute_common_grid_from_stats,
    remap_to_common_grid,
)
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
        max_windows: int | None = None,
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
            max_windows: Maximum number of windows to process (None = all)
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
        self._n_processed = 0
        self._n_merged = 0
        self._timings: dict[str, float] = {}
        self._merge_stats: list[dict] = []

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
        t0_merge = time.perf_counter()

        if len(frames) < self._window_config.min_frames_per_window:
            return None

        # Sort by timestamp
        frames.sort(key=lambda x: x["timestamp"])

        # Collect grid computation inputs
        max_ranges = []
        range_resolutions = []
        position_stats = []

        for f in frames:
            max_ranges.append(f.get("ground_range_max", 3000.0))
            # Use per-frame projected grid_spacing (accounts for angular width)
            # rather than raw range_resolution (radial sample spacing only)
            frame_gp = f.get("grid_params") or {}
            range_resolutions.append(frame_gp.get("grid_spacing", 7.5))
            # Collect position stats if available
            if "position_stats" in f:
                position_stats.append(f["position_stats"])

        t0 = time.perf_counter()

        # Use optimized stats-based grid computation if position_stats available
        if position_stats and len(position_stats) == len(frames):
            grid_params = compute_common_grid_from_stats(
                position_stats=position_stats,
                max_ranges=max_ranges,
                range_resolutions=range_resolutions,
                resolution_scale=self._window_config.resolution_scale,
            )
        else:
            # Fallback to original method (for backwards compatibility)
            latitudes = [f["latitudes"] for f in frames if "latitudes" in f]
            longitudes = [f["longitudes"] for f in frames if "longitudes" in f]
            if not latitudes:
                return None
            grid_params = compute_common_grid(
                latitudes=latitudes,
                longitudes=longitudes,
                max_ranges=max_ranges,
                range_resolutions=range_resolutions,
                resolution_scale=self._window_config.resolution_scale,
            )
        t_grid = time.perf_counter() - t0

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
        t_remap_total = 0.0
        t_accum_total = 0.0
        n_remapped = 0
        n_fallback = 0
        n_no_proj = 0

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
                    ref_lat = grid_params["ref_lat"]
                    ref_lon = grid_params["ref_lon"]
                    m_per_deg_lon = grid_params["m_per_deg_lon"]

                    frame_center_x = (frame_center_lon - ref_lon) * m_per_deg_lon
                    frame_center_y = (frame_center_lat - ref_lat) * 111_319.5
                    frame_x_edges_abs = frame_x_edges_centered + frame_center_x
                    frame_y_edges_abs = frame_y_edges_centered + frame_center_y

                    t0 = time.perf_counter()
                    frame_sum, frame_count = remap_to_common_grid(
                        proj_intensity,
                        proj_count,
                        frame_x_edges_abs,
                        frame_y_edges_abs,
                        grid_params["x_edges_abs"],
                        grid_params["y_edges_abs"],
                        grid_params["n_x"],
                        grid_params["n_y"],
                    )
                    t_remap_total += time.perf_counter() - t0
                    n_remapped += 1
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
                    n_fallback += 1
            else:
                logger.debug(
                    "Frame (%d, %d) has no projected data",
                    frame_data.get("file_index", -1),
                    frame_data.get("frame_index", -1),
                )
                frame_sum = np.zeros((grid_params["n_y"], grid_params["n_x"]), dtype=np.float64)
                frame_count = np.zeros((grid_params["n_y"], grid_params["n_x"]), dtype=np.int32)
                n_no_proj += 1

            t0 = time.perf_counter()
            accumulator.add_projected(
                projected_intensity=frame_sum,
                projected_count=frame_count,
                timestamp=frame_data["timestamp"],
                heading=mean_heading,
                ship_speed=frame_data.get("ship_speed"),
                wind_speed=frame_data.get("wind_speed"),
                wind_direction=frame_data.get("wind_direction"),
            )
            t_accum_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        result = None
        if accumulator.n_frames >= self._window_config.min_frames_per_window:
            result = accumulator.finalize(
                window_index=window_idx,
                interpolate_gaps=self._window_config.interpolate_gaps,
            )
        t_finalize = time.perf_counter() - t0

        t_total = time.perf_counter() - t0_merge

        self._merge_stats.append(
            {
                "total": t_total,
                "grid": t_grid,
                "remap": t_remap_total,
                "accumulate": t_accum_total,
                "finalize": t_finalize,
                "n_frames": len(frames),
                "n_remapped": n_remapped,
                "n_fallback": n_fallback,
                "n_no_proj": n_no_proj,
                "n_y": grid_params["n_y"],
                "n_x": grid_params["n_x"],
                "grid_spacing": grid_params["grid_spacing"],
            }
        )

        return result

    def iter_merged(self) -> Iterator[MergedImage]:
        """
        Yield merged images as windows complete.

        Uses streaming architecture: windows are merged and yielded as soon as
        all their constituent files have been interpolated, rather than waiting
        for all files to complete first.
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

        # Limit in-flight file loads to keep workers busy without excessive
        # shared memory accumulation.  Each loaded-but-not-interpolated file
        # holds ~376 KB of shared memory (intensity + theta + ground_range).
        max_pending = self._n_workers * 3

        def submit_batch(n: int) -> int:
            """Submit up to n file tasks. Returns number submitted.

            Respects two limits:
            - max_pending: caps tasks submitted but not yet returned
            - max_loaded_ahead: caps files loaded but not yet interpolated,
              preventing unbounded shared memory growth on large runs
            """
            nonlocal files_submitted, pending_files
            # Files loaded but not yet interpolated (holding shared memory)
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

        # ---- Merge thread: runs _merge_window off the main event loop ----
        import queue as _queue
        import threading

        merge_input: _queue.Queue[tuple[int, list[dict]]] = _queue.Queue()
        merge_output: _queue.Queue[tuple[int, MergedImage | None, float]] = _queue.Queue()
        merge_shutdown = threading.Event()

        def _merge_thread_fn():
            while not merge_shutdown.is_set():
                try:
                    window_idx, frames = merge_input.get(timeout=0.1)
                except _queue.Empty:
                    continue
                t0 = time.perf_counter()
                merged = self._merge_window(window_idx, frames)
                elapsed = time.perf_counter() - t0
                merge_output.put((window_idx, merged, elapsed))

        merge_thread = threading.Thread(target=_merge_thread_fn, daemon=True)
        merge_thread.start()
        pending_merges = 0
        # Cap queued merges to bound memory: each window holds ~42 frame dicts
        # with projected arrays. 4 windows ≈ 25 MB — keeps merge thread fed
        # without unbounded growth on 100K+ file runs.
        max_queued_merges = 4

        # Submit initial batch to get workers started
        submit_batch(max_pending)

        # Helper: drain completed merges from the merge thread (non-blocking)
        def _collect_merges():
            nonlocal time_merging, pending_merges
            while True:
                try:
                    window_idx, merged, elapsed = merge_output.get_nowait()
                    time_merging += elapsed
                    pending_merges -= 1
                    if merged is not None:
                        self._n_merged += 1
                        pbar_merged.update(1)
                        _merged_results.append(merged)
                    completed_windows.add(window_idx)
                except _queue.Empty:
                    break

        def _wait_for_merge_room():
            """Block until pending_merges drops below max_queued_merges."""
            nonlocal time_merging, pending_merges
            while pending_merges >= max_queued_merges:
                try:
                    window_idx, merged, elapsed = merge_output.get(timeout=0.1)
                    time_merging += elapsed
                    pending_merges -= 1
                    if merged is not None:
                        self._n_merged += 1
                        pbar_merged.update(1)
                        _merged_results.append(merged)
                    completed_windows.add(window_idx)
                except _queue.Empty:
                    pass

        _merged_results: list[MergedImage] = []

        # Main processing loop - submit more tasks as results arrive
        while (
            pending_files > 0
            or pending_interp > 0
            or files_submitted < total_files
            or pending_merges > 0
        ):
            # Yield any completed merges without blocking
            _collect_merges()
            while _merged_results:
                yield _merged_results.pop(0)

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
                    for window_idx in file_to_windows[file_idx]:
                        if window_idx in completed_windows:
                            continue

                        needed = window_needs[window_idx]
                        if needed.issubset(files_with_all_frames_interpolated):
                            # Apply backpressure: wait if merge queue is full
                            _wait_for_merge_room()
                            # Submit to merge thread
                            frames = window_frames.pop(window_idx, [])
                            merge_input.put((window_idx, frames))
                            pending_merges += 1

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

        # Drain any remaining merge results
        _collect_merges()
        while _merged_results:
            yield _merged_results.pop(0)

        # Shut down merge thread
        merge_shutdown.set()
        merge_thread.join(timeout=2.0)

        pbar_files.close()
        pbar_interp.close()
        pbar_merged.close()

        # Final prune to free any remaining triplet items
        triplet_collector.prune_emitted()

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
            print(f"  Merging:       {time_merging:.2f}s (background thread)")
            print(f"  Files processed: {len(file_indices_to_process)}")
            print(f"  Frames interpolated: {n_interp_completed}")
            print(f"  Windows merged: {self._n_merged}")
            print(f"  Max memory: {max_mem:.1f} MB")

            if self._merge_stats:
                n_win = len(self._merge_stats)
                tot = [s["total"] for s in self._merge_stats]
                t_grid = [s["grid"] for s in self._merge_stats]
                t_remap = [s["remap"] for s in self._merge_stats]
                t_accum = [s["accumulate"] for s in self._merge_stats]
                t_final = [s["finalize"] for s in self._merge_stats]
                n_frames = [s["n_frames"] for s in self._merge_stats]
                n_remap = [s["n_remapped"] for s in self._merge_stats]
                grids = [f"{s['n_y']}x{s['n_x']}" for s in self._merge_stats]
                spacing = self._merge_stats[0]["grid_spacing"]

                # Per-frame remap across all windows
                total_remapped = sum(n_remap)
                total_remap_time = sum(t_remap)
                avg_remap_per_frame = (
                    (total_remap_time / total_remapped * 1000) if total_remapped else 0
                )

                print(f"\nMerge Breakdown ({n_win} windows, grid ~{grids[0]} @ {spacing:.1f}m):")
                print(
                    f"  Per window (mean):  {np.mean(tot) * 1000:.0f}ms total, {np.mean(n_frames):.0f} frames"
                )
                print(f"    compute_grid:  {np.mean(t_grid) * 1000:.1f}ms")
                print(
                    f"    remap:         {np.mean(t_remap) * 1000:.1f}ms ({avg_remap_per_frame:.1f}ms/frame, {total_remapped} total)"
                )
                print(f"    accumulate:    {np.mean(t_accum) * 1000:.1f}ms")
                print(f"    finalize:      {np.mean(t_final) * 1000:.1f}ms")
                print(f"  Per window (range): {min(tot) * 1000:.0f}–{max(tot) * 1000:.0f}ms")

            print("\nMemory Management:")
            print(f"  SharedMem registered: {shm_stats['registered']}")
            print(f"  SharedMem released:   {shm_stats['released']}")
            print(f"  SharedMem orphaned:   {remaining_shm}")
            print(f"  Triplet items remaining: {triplet_stats['items']}")
            print(f"  Triplet pending:      {triplet_stats['pending']}")


# ============================================================
# CLI Interface
# ============================================================


def parse_duration(value: str) -> float:
    """Parse a duration string into seconds.

    Accepts bare numbers (interpreted as seconds) or numbers with a suffix:
      s = seconds, m = minutes, h = hours, d = days.

    Examples:
        "60"   -> 60.0
        "60s"  -> 60.0
        "1.5m" -> 90.0
        "1h"   -> 3600.0
        "0.5d" -> 43200.0
    """
    import argparse
    import re

    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([smhd]?)", value.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid duration: {value!r}  (use e.g. 60, 60s, 1.5m, 1h, 0.5d)"
        )
    number = float(m.group(1))
    unit = m.group(2) or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return number * multiplier


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
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

    # Streaming mode
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming file discovery (starts processing sooner for large datasets)",
    )

    # Memory monitoring
    parser.add_argument(
        "--memory-stats",
        action="store_true",
        help="Show detailed memory usage statistics during and after processing",
    )

    # Batch processing
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of windows to process (default: all). "
        "Useful for testing or batch processing large datasets.",
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


def _run_streaming(args, config) -> None:
    """Execute the 'files-pipeline' command with streaming file discovery.

    This mode starts processing files as they are discovered, rather than
    waiting for all files to be found first. Useful for large datasets.
    """
    from pathlib import Path

    from wamos_tpw.memory_monitor import MemoryMonitor, log_memory_stats
    from wamos_tpw.merged_viewer import show_merged_viewer, show_single_merged_image
    from wamos_tpw.output_writers import (
        write_geotiff,
        write_kml,
        write_kmz,
        write_merged_netcdf,
        write_merged_png,
        write_mp4_movie,
    )
    from wamos_tpw.streaming_pipeline import StreamingMergePipeline

    logging.info("Using streaming file discovery mode")

    # Set up memory monitoring if requested
    memory_monitor = None
    if getattr(args, "memory_stats", False):
        sample_interval = 5.0  # Sample memory every 5 seconds
        memory_monitor = MemoryMonitor(sample_interval=sample_interval)
        memory_monitor.__enter__()

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

    # Create streaming pipeline
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
        max_windows=getattr(args, "max_windows", None),
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

    if memory_monitor:
        memory_monitor.checkpoint("After pipeline iteration")

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

    # Log final memory statistics
    if memory_monitor:
        memory_monitor.__exit__(None, None, None)
        log_memory_stats(memory_monitor.stats, "Streaming pipeline")


def run(args) -> None:
    """Execute the 'files-pipeline' command."""
    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.memory_monitor import MemoryMonitor, log_memory_stats
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

    # Use streaming mode if requested
    if getattr(args, "streaming", False):
        _run_streaming(args, config)
        return

    # Set up memory monitoring if requested
    memory_monitor = None
    if getattr(args, "memory_stats", False):
        sample_interval = 5.0  # Sample memory every 5 seconds
        memory_monitor = MemoryMonitor(sample_interval=sample_interval)
        memory_monitor.__enter__()

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
        max_windows=getattr(args, "max_windows", None),
    )

    logging.info("Created %d time windows", pipeline.n_windows)

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

    logging.info("Created %d merged images", n_merged)

    if memory_monitor:
        memory_monitor.checkpoint("After pipeline iteration")

    # Bulk outputs (require all images in memory)
    if merged_images:
        # KML/KMZ/plot first, then MP4 last so it can release each image
        if args.kml:
            write_kml(merged_images, args.kml)
        if args.kmz:
            write_kmz(merged_images, args.kmz)
        if args.plot:
            show_merged_viewer(merged_images)
        if args.mp4:
            write_mp4_movie(merged_images, args.mp4, fps=args.fps, release=True)
        del merged_images

    # Log final memory statistics
    if memory_monitor:
        memory_monitor.__exit__(None, None, None)
        log_memory_stats(memory_monitor.stats, "Files pipeline")


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Merge frames into composite images")

if __name__ == "__main__":
    main()
