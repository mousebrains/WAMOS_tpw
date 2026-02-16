#! /usr/bin/env python3
#
# Frame interpolation/extrapolation for per-radial metadata
#
# Uses PPS pulses from adjacent frames to calculate accurate timestamps,
# falling back to linear interpolation when PPS is unavailable.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.frame_pipeline import FramePipeline


logger = logging.getLogger(__name__)


class FrameInterpolator:
    """
    Compute interpolated/extrapolated per-radial metadata for a frame.

    Uses a triplet approach (previous, current, next) for:
    - Timestamps: Uses PPS pulses as anchors when available, otherwise linear
    - Position: Interpolates/extrapolates latitude and longitude

    PPS-based timing:
    - PPS pulses occur at whole seconds (GPS-synchronized)
    - Uses PPS from prev, current, and next frames to build timing model
    - Falls back to linear model (start_time + repeat_time) when no PPS

    Position interpolation:
    - If current and next are within time tolerance: forward interpolation
    - Else if previous and current are within tolerance: backward extrapolation
    """

    # Constants
    _DEFAULT_TOLERANCE = 1.2  # 20% margin on repeat_time
    _DEFAULT_REPEAT_TIME = 1.43  # seconds from R/V Revelle should be specified in RPT field
    _MIN_DIDT = 1  # Minimum indices/second for PPS extrapolation

    def __init__(
        self,
        prev: FramePipeline | None,
        current: FramePipeline,
        next_frame: FramePipeline | None,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> None:
        """
        Initialize frame interpolator.

        Args:
            prev: Previous frame (can be None for first frame)
            current: Current frame to compute per-radial values for
            next_frame: Next frame (can be None for last frame)
            tolerance: Multiplier for repeat_time to accept pair (1.2 = 20% margin)
        """

        self._current = current
        self._tolerance = tolerance
        self._timing_method: str = "linear"  # or "PPS(n)"

        repeat_time = current.metadata.repeat_time
        if not repeat_time:
            repeat_time = self._DEFAULT_REPEAT_TIME
            logger.warning(
                "Frame %s has no repeat_time in metadata; "
                "falling back to default %.2fs (R/V Revelle). "
                "Set RPT in polar file header or config for other ships.",
                current.metadata.filename,
                repeat_time,
            )
        max_dt = repeat_time * tolerance

        self._repeat_time = repeat_time
        self._n_radials = current.n_bearings

        # Check if prev is close enough to current, if not then drop it
        if prev is not None:
            dt = (current.metadata.timestamp - prev.metadata.timestamp) / np.timedelta64(1, "s")
            if dt > max_dt or dt <= 0:
                prev = None

        # Check if next is close enough to current, if not then drop it
        if next_frame is not None:
            dt = (next_frame.metadata.timestamp - current.metadata.timestamp) / np.timedelta64(
                1, "s"
            )
            if dt > max_dt or dt <= 0:
                next_frame = None

        # Compute timestamps using PPS or linear fallback
        self._compute_timestamps(prev, current, next_frame)

        # Compute lat/lon/headings/speeds
        self._compute_positions(prev, current, next_frame)

        self._dt = (self._times[-1] - self._times[0]) / np.timedelta64(1, "s")  # Time span
        self._method = (
            "interpolate"
            if next_frame is not None
            else "extrapolate"
            if prev is not None
            else "none"
        )

    def _extract_PPS(self, frame: FramePipeline) -> tuple[np.ndarray, np.datetime64, int]:
        """Extract PPS pulse indices from a frame."""

        if frame is not None:
            pps = frame.pps.indices if frame.pps is not None else None
            t0 = frame.metadata.timestamp
            n = frame.n_bearings
        else:
            pps = None
            n = 0

        if pps is None:
            pps = np.empty(0, dtype=int)
            t0 = np.datetime64(0, "s")

        return (pps, t0, n)

    def _compute_timestamps(
        self,
        prev: FramePipeline | None,
        curr: FramePipeline,
        nxt: FramePipeline | None,
    ) -> None:
        """Compute per-radial timestamps using PPS anchors or linear fallback.

        If PPS pulses are found, then use all of the pulses from the previous, current, and next
        frames to anchor the second transitions.
        The reference time to round to a whole second corresponds to the first index
        closest to the middle. The frame's timestamp can be ~+-0.25 seconds from the actual
        frame's starttime.

        If no PPS pulses are found, then fall back to linear timestamps
        based on start_time and repeat_time.
        """

        # Get PPS locations from previous frame
        (prev_PPS, prev_t0, prev_n) = self._extract_PPS(prev)
        (curr_PPS, curr_t0, curr_n) = self._extract_PPS(curr)
        (next_PPS, next_t0, next_n) = self._extract_PPS(nxt)

        indices = np.concatenate(
            (
                prev_PPS - prev_n,
                curr_PPS,
                next_PPS + curr_n,
            )
        )

        t0 = np.concatenate(
            (
                prev_t0 + np.arange(prev_PPS.size, dtype="timedelta64[s]"),
                curr_t0 + np.arange(curr_PPS.size, dtype="timedelta64[s]"),
                next_t0 + np.arange(next_PPS.size, dtype="timedelta64[s]"),
            )
        )

        if indices.size > 0:
            # Find index closest to the 1/2 second
            # The metadata timestamp may be off by ~+-0.25 seconds,
            self._timing_method = f"PPS({indices.size})"
            #
            # Find the PPS pulse with t0 closest to 1/2 second transition to avoid rounding slop
            #
            t1 = t0.astype(
                "datetime64[s]"
            )  # Floored to whole second, so tt is the offset from whole second
            dt = np.abs(t0 - t1 - np.timedelta64(500, "ms"))  # Distance from 1/2 second
            i_ref = np.argmin(dt)  # index of the PPS closest to middle
            t_ref = t1[i_ref]  # Whole second at the PPS pulse closest to 1/2 second
            t_pps = (
                t_ref - np.timedelta64(i_ref, "s") + np.arange(indices.size, dtype="timedelta64[s]")
            )  # Whole second for each PPS pulse

            # np.interp does not extrapolate, so extend if needed
            if indices[0] > 0:  # first PPS is after first index, so extrapolate back
                if indices.size == 1:
                    dIdt = np.maximum(self._MIN_DIDT, curr_n / self._repeat_time)
                else:
                    dIdt = np.maximum(self._MIN_DIDT, np.abs(np.mean(np.diff(indices))))

                cnt = int(np.ceil(indices[0] / dIdt))
                indices = np.insert(indices, 0, indices[0] - dIdt * cnt)
                t_pps = np.insert(t_pps, 0, t_pps[0] - np.timedelta64(cnt, "s"))

            if indices[-1] < curr_n - 1:  # last PPS is before last index, so extrapolate forward
                if indices.size == 1:
                    dIdt = np.maximum(self._MIN_DIDT, curr_n / self._repeat_time)
                else:
                    dIdt = np.maximum(self._MIN_DIDT, np.abs(np.mean(np.diff(indices))))

                cnt = int(np.ceil((curr_n - 1 - indices[-1]) / dIdt))
                indices = np.insert(indices, indices.size, indices[-1] + dIdt * cnt)
                t_pps = np.insert(t_pps, t_pps.size, t_pps[-1] + np.timedelta64(cnt, "s"))

        else:  # No PPS at all
            self._timing_method = "linear"
            indices = np.array([0, curr_n])  # Assume repeat is total time, curr_n instead of -1
            meta = self._current.metadata
            # Metadata timestamp is end-of-frame, so radial 0 starts at
            # timestamp - repeat_time and radial curr_n ends at timestamp.
            t_pps = meta.timestamp + np.array([-meta.repeat_time * 1e9, 0]).astype(int).astype(
                "timedelta64[ns]"
            )

        # Now interpolate to get per-radial timestamps, with known PPS anchors
        dt = np.interp(
            np.arange(curr_n),
            indices,
            (t_pps - t_pps[0]).astype("timedelta64[us]").astype(float),
        )

        self._times = t_pps[0] + dt.astype("timedelta64[us]")

    # Compute lat/lon/headings/speeds for each radial
    def _compute_positions(
        self,
        prev: FramePipeline | None,
        curr: FramePipeline,
        nxt: FramePipeline | None,
    ) -> None:
        """Compute per-radial information using interpolation if nxt is available,
        else extrapolation if prev is available.
        If neither is available, do nothing.

        One might think of using the ship speed and heading for the lat/lon, but
        one has to consider the ship can be crabbing due to currents and winds.
        """
        n = curr.n_bearings

        # Guard against missing navigation data
        if (
            curr.metadata.latitude is None
            or curr.metadata.longitude is None
            or curr.metadata.heading is None
        ):
            logger.warning("Missing lat/lon/heading in frame metadata; using constant fill values")
            self._latitudes = np.full(n, curr.metadata.latitude or 0.0)
            self._longitudes = np.full(n, curr.metadata.longitude or 0.0)
            self._headings = np.full(n, curr.metadata.heading or 0.0)
            return

        # Drop neighbor if it has missing navigation data
        if nxt is not None and (
            nxt.metadata.latitude is None
            or nxt.metadata.longitude is None
            or nxt.metadata.heading is None
        ):
            nxt = None
        if prev is not None and (
            prev.metadata.latitude is None
            or prev.metadata.longitude is None
            or prev.metadata.heading is None
        ):
            prev = None

        # The starting point of the frame at the first radial
        indices = [0]
        lat = [curr.metadata.latitude]
        lon = [curr.metadata.longitude]
        hdg = [curr.metadata.heading]

        indices.append(n)  # Used for interpolation/extrapolation/nothing

        if nxt is not None:  # Interpolation from current starting values to next starting values
            lat.append(nxt.metadata.latitude)
            lon.append(nxt.metadata.longitude)
            hdg.append(nxt.metadata.heading)
        elif prev is not None:  # Extrapolate if prev is available
            lat.append(
                curr.metadata.latitude
                + (curr.metadata.latitude - prev.metadata.latitude) / prev.n_bearings * n
            )
            lon.append(
                curr.metadata.longitude
                + (curr.metadata.longitude - prev.metadata.longitude) / prev.n_bearings * n
            )
            hdg.append(
                curr.metadata.heading
                + (curr.metadata.heading - prev.metadata.heading) / prev.n_bearings * n
            )
        else:  # No prev or next, so just duplicate current values
            lat.append(curr.metadata.latitude)
            lon.append(curr.metadata.longitude)
            hdg.append(curr.metadata.heading)

        # Radials are equally spaced in time, so interpolate over indices
        self._latitudes = np.interp(np.arange(n), indices, lat)
        self._longitudes = self._interp_angle(np.arange(n), indices, lon, True)
        self._headings = self._interp_angle(np.arange(n), indices, hdg)

    def _interp_angle(
        self,
        j: np.ndarray,  # Target indices
        indices: np.ndarray,  # Source indices
        theta: np.ndarray,  # Source angles
        q180: bool = False,  # Should result be wrapped to [-180,180]
    ) -> np.ndarray:
        # Interpolate angles (0-360) handling wrap-around
        result = np.interp(j, indices, np.asarray(theta) % 360, period=360)  # [0,360)
        return (result + 180) % 360 - 180 if q180 else result

    @property
    def method(self) -> str:
        """Return the position method used: 'interpolate' or 'extrapolate'."""
        return self._method

    @property
    def timing_method(self) -> str:
        """Return the timing method used: 'PPS(n)' or 'linear'."""
        return self._timing_method

    @property
    def frame(self) -> FramePipeline:
        """Return the current frame."""
        return self._current

    @property
    def times(self) -> np.ndarray:
        """Return per-radial timestamps."""
        return self._times

    @property
    def latitudes(self) -> np.ndarray:
        """Return interpolated/extrapolated latitudes for each radial."""
        return self._latitudes

    @property
    def longitudes(self) -> np.ndarray:
        """Return interpolated/extrapolated longitudes for each radial."""
        return self._longitudes

    @property
    def headings(self) -> np.ndarray:
        """Return interpolated/extrapolated ship headings for each radial (degrees, 0-360)."""
        return self._headings

    @property
    def time_delta(self) -> float:
        """Return time delta used for position interpolation/extrapolation in seconds."""
        return self._dt

    @property
    def repeat_time(self) -> float:
        """Return the current frame's repeat time in seconds."""
        return self._repeat_time

    def __repr__(self) -> str:
        return (
            f"<FrameInterpolator position={self._method} timing={self._timing_method} "
            f"dt={self._dt:.2f}s n_radials={len(self._times)}>"
        )


# ============================================================
# Data structures for parallel processing
# ============================================================


@dataclass
class FrameData:
    """
    Serializable frame data for multiprocessing.

    Contains the essential metadata and shared memory references
    needed for triplet collection and interpolation.
    """

    filepath: str
    file_index: int  # Index of this file in the processing order
    frame_index: int  # Frame index within the file
    timestamp: np.datetime64
    repeat_time: float
    latitude: float | None
    longitude: float | None
    heading: float | None
    ship_speed: float | None
    wind_speed: float | None
    wind_direction: float | None
    n_bearings: int
    n_distances: int
    pps_indices: np.ndarray | None  # PPS pulse indices
    # Shared memory references for arrays (name, shape, dtype)
    theta_shm: tuple[str, tuple, np.dtype] | None = None
    ground_range_shm: tuple[str, tuple, np.dtype] | None = None
    intensity_shm: tuple[str, tuple, np.dtype] | None = None
    timings: dict[str, float] = field(default_factory=dict)


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.2,
        help="Time tolerance multiplier (default: 1.2)",
    )
    parser.add_argument("--timing", "-t", action="store_true", help="Show timing statistics")
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--project", "-p", action="store_true", help="Project dewinded intensity onto UTM grid"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot the projected intensity (requires --project)"
    )
    parser.add_argument(
        "--netcdf-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for per-frame NetCDF files (requires --project)",
    )
    parser.add_argument(
        "--ship-data",
        type=str,
        default=None,
        help="Directory with instrument NetCDF files (from revelle CLI) for high-frequency interpolation",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=None,
        help="Grid cell size in meters for earth projection (default: auto from range/angular resolution)",
    )

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
    """Register the 'interpolator' subcommand."""
    p = subparsers.add_parser(
        "interpolator",
        help="Test frame interpolation/extrapolation",
        description="Test per-radial metadata interpolation between frames with priority processing",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'interpolator' command with priority processing."""
    import os

    from tqdm import tqdm

    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.interpolator_tasks import TASK_HANDLERS
    from wamos_tpw.priority_executor import (
        Priority,
        PriorityProcessExecutor,
        SharedMemoryManager,
        TripletCollector,
    )

    # --plot and --netcdf-dir imply --project
    if args.plot or args.netcdf_dir:
        args.project = True

    # Create output directory if specified
    netcdf_dir = args.netcdf_dir
    if netcdf_dir:
        os.makedirs(netcdf_dir, exist_ok=True)
        logging.info("NetCDF output directory: %s", netcdf_dir)

    config = Config(args.config) if args.config else Config()

    # Pre-build ship data cache in main process so workers just memmap
    ship_data_dir = getattr(args, "ship_data", None)
    if ship_data_dir:
        from pathlib import Path

        from wamos_tpw.instruments.ship_data import ShipData

        sd = ShipData(Path(ship_data_dir))
        logging.info("Ship data: %s", sd)

    filenames = Filenames(args.stime, args.etime, args.polar_path)
    files = list(filenames)

    if not files:
        logging.warning(
            "No files found in %s for time range %s to %s", args.polar_path, args.stime, args.etime
        )
        return

    logging.info("Found %d files to process", len(files))

    n_workers = args.workers
    if n_workers is None:
        n_workers = min(len(files), os.cpu_count() or 1)

    # Serialize config for passing to workers
    config_dict = config._config if config else None

    # Create executor and shared memory manager
    executor = PriorityProcessExecutor(
        max_workers=n_workers,
        task_handlers=TASK_HANDLERS,
    )
    executor.start()
    shm_manager = SharedMemoryManager()

    logging.info("Processing with %d workers", n_workers)

    t0_total = time.perf_counter()

    # Submit file processing tasks (low priority)
    pending_files = len(files)
    pending_interp = 0
    total_frames_expected = 0

    for file_idx, filepath in enumerate(files):
        task_data = (filepath, file_idx, config_dict, args.timing)
        executor.submit(Priority.LOW, "process_file", task_data)

    # Triplet collection and interpolation results
    # Pass total_files to enable streaming triplet emission
    triplet_collector = TripletCollector(total_files=len(files))
    interp_results = []
    max_rss = 0

    # Statistics
    n_interpolated = 0
    n_extrapolated = 0
    n_pps = 0
    n_linear = 0

    # Progress bars
    pbar_files = tqdm(
        total=len(files), desc="Loading files", unit="file", disable=not args.progress
    )
    pbar_interp = tqdm(
        total=len(files), desc="Interpolating", unit="frame", disable=not args.progress
    )

    # Create streaming viewer if plotting requested (shows first frame immediately)
    viewer = None
    if args.plot:
        from wamos_tpw.projection import InterpolatorViewer

        viewer = InterpolatorViewer()
        viewer.show()

    # Process results as they come in
    while pending_files > 0 or pending_interp > 0:
        result = executor.get_result(timeout=0.1)
        if result is None:
            continue

        if result.error:
            logging.error("Error: %s", result.error)
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
            data = result.data
            max_rss = max(max_rss, data["peak_rss"])
            file_idx = data["file_index"]

            # Add frames to triplet collector
            for frame_data in data["frames"]:
                triplet_collector.add(frame_data)
                total_frames_expected += 1

                # Register shared memory with refcount of 1 (will be consumed by interpolation)
                # Note: each frame is used in up to 3 triplets (as prev, current, next)
                # But we simplify by releasing after interpolation completes
                if frame_data.theta_shm:
                    shm_manager.register(frame_data.theta_shm[0], refcount=1)
                if frame_data.ground_range_shm:
                    shm_manager.register(frame_data.ground_range_shm[0], refcount=1)
                if frame_data.intensity_shm:
                    shm_manager.register(frame_data.intensity_shm[0], refcount=1)

            # Mark file as complete with frame count (enables neighbor detection)
            triplet_collector.file_complete(file_idx, len(data["frames"]))

            # Check for ready triplets - now safe because TripletCollector
            # only emits when true consecutive neighbors are confirmed
            for prev, current, next_frame in triplet_collector.ready_triplets():
                task_data = (
                    prev,
                    current,
                    next_frame,
                    args.tolerance,
                    args.project,
                    netcdf_dir,
                    getattr(args, "ship_data", None),
                    getattr(args, "grid_spacing", None),
                )
                executor.submit(Priority.MEDIUM, "interpolate", task_data)
                pending_interp += 1

        elif result.task_type == "interpolate":
            pending_interp -= 1
            pbar_interp.update(1)
            data = result.data
            max_rss = max(max_rss, data.get("peak_rss", 0))

            if data["success"]:
                interp_results.append(data)

                if data["method"] == "interpolate":
                    n_interpolated += 1
                else:
                    n_extrapolated += 1

                if data["timing_method"].startswith("PPS"):
                    n_pps += 1
                else:
                    n_linear += 1

            # Check for more ready triplets (in case file results arrived while processing)
            for prev, current, next_frame in triplet_collector.ready_triplets():
                task_data = (
                    prev,
                    current,
                    next_frame,
                    args.tolerance,
                    args.project,
                    netcdf_dir,
                    getattr(args, "ship_data", None),
                    getattr(args, "grid_spacing", None),
                )
                executor.submit(Priority.MEDIUM, "interpolate", task_data)
                pending_interp += 1

            # Feed result to viewer if plotting (streams as they arrive)
            if viewer and data["success"] and data.get("projected_intensity") is not None:
                viewer.add_result(data)

    pbar_files.close()
    pbar_interp.close()

    elapsed = time.perf_counter() - t0_total
    executor.shutdown()

    # Clean up any remaining shared memory
    orphaned = shm_manager.cleanup()
    if orphaned > 0:
        logging.debug("Cleaned up %d orphaned shared memory blocks", orphaned)

    # Sort results by (file_index, frame_index)
    interp_results.sort(key=lambda r: (r["file_index"], r["frame_index"]))

    # Display results
    logging.info("=" * 60)
    logging.info(
        "Processed %d files, %d frames in %.2fs (%.1f frames/sec)",
        len(files),
        total_frames_expected,
        elapsed,
        total_frames_expected / elapsed if elapsed > 0 else 0,
    )
    logging.info("Workers: %d", n_workers)
    logging.info(
        "Summary: position: %d interpolated, %d extrapolated; timing: %d pps, %d linear",
        n_interpolated,
        n_extrapolated,
        n_pps,
        n_linear,
    )

    # Timing statistics
    if args.timing and interp_results:
        # Collect timing data from all successful results
        timing_keys = [
            "read_shm",
            "build_proxies",
            "interpolate",
            "project",
            "proj_ship_pos",
            "proj_grid_setup",
            "proj_bearings",
            "proj_bincount",
            "proj_finalize",
            "netcdf",
            "total",
        ]
        timing_sums = {k: 0.0 for k in timing_keys}
        timing_counts = {k: 0 for k in timing_keys}

        for data in interp_results:
            if "timings" in data:
                for k in timing_keys:
                    if k in data["timings"]:
                        timing_sums[k] += data["timings"][k]
                        timing_counts[k] += 1

        logging.info("=" * 60)
        logging.info("Per-frame timing statistics (averages over %d frames):", len(interp_results))
        for k in timing_keys:
            if timing_counts[k] > 0:
                avg_ms = timing_sums[k] / timing_counts[k] * 1000
                total_s = timing_sums[k]
                logging.info("  %-18s: %7.2f ms avg, %7.2f s total", k, avg_ms, total_s)

    # Finalize viewer and wait for user to close
    if viewer:
        viewer.mark_loading_complete()
        logging.info("Loading complete. Use viewer controls to navigate frames.")
        viewer.wait()

    # Memory stats
    if max_rss > 0:
        # macOS returns bytes, Linux returns KB
        import sys

        if sys.platform == "darwin":
            max_rss_mb = max_rss / (1024 * 1024)
        else:
            max_rss_mb = max_rss / 1024
        logging.info("Peak worker RSS: %.1f MB", max_rss_mb)

    return interp_results


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Test frame interpolation/extrapolation")

if __name__ == "__main__":
    main()
