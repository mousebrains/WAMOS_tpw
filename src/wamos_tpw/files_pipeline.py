#! /usr/bin/env python3
#
# Multiple files processing pipeline for WAMOS polar data
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import Config

from wamos_tpw.file_pipeline import FilePipeline
from wamos_tpw.frame_pipeline import FramePipeline
from wamos_tpw.interpolator import FrameInterpolator


@dataclass
class EarthGrid:
    """Earth-referenced Cartesian coordinate system for projected radar data."""

    # Grid definition
    x_edges: np.ndarray  # East-west bin edges in meters
    y_edges: np.ndarray  # North-south bin edges in meters
    grid_spacing: float  # Grid cell size in meters

    # Projected data (sum and count for averaging)
    intensity_sum: np.ndarray = field(default=None)
    intensity_count: np.ndarray = field(default=None)

    # Reference position (origin)
    ref_latitude: float = 0.0
    ref_longitude: float = 0.0

    @property
    def x_centers(self) -> np.ndarray:
        """Return x (east) bin centers."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        """Return y (north) bin centers."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def intensity(self) -> np.ndarray:
        """Return averaged projected intensity (NaN where no data)."""
        with np.errstate(invalid="ignore"):
            result = self.intensity_sum / self.intensity_count
        result[self.intensity_count == 0] = np.nan
        return result

    @property
    def n_x(self) -> int:
        """Return number of x bins."""
        return len(self.x_edges) - 1

    @property
    def n_y(self) -> int:
        """Return number of y bins."""
        return len(self.y_edges) - 1

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in meters."""
        return (self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1])


@dataclass
class ProjectionResult:
    """
    Retained data after earth projection and memory cleanup.

    Contains only the essential data for analysis:
    - Earth coordinate system and projected intensities
    - Ship navigation data (speeds, headings)
    - Wind data (speeds, directions)
    - Frame timing information
    """

    # Earth grid with projected data
    earth_grid: EarthGrid

    # Ship navigation per frame
    ship_speeds: np.ndarray  # m/s per frame
    ship_headings: np.ndarray  # degrees per frame (mean heading during frame)

    # Wind data per frame
    wind_speeds: np.ndarray  # m/s per frame
    wind_directions: np.ndarray  # degrees per frame

    # Timing
    frame_start_times: np.ndarray  # datetime64 per frame
    frame_end_times: np.ndarray  # datetime64 per frame

    # Statistics
    n_frames: int = 0
    n_radials_total: int = 0


# Earth radius in meters (WGS84 mean radius)
EARTH_RADIUS = 6371000.0


logger = logging.getLogger(__name__)


def create_overlapping_groups(
    filenames,
    groupby: str,
    overlap: int = 1,
) -> list[tuple[np.datetime64, list[str], int, int]]:
    """
    Create overlapping groups from a Filenames object.

    Each group includes `overlap` files from the previous group (at start)
    and `overlap` files from the next group (at end), enabling frame pairing
    and interpolation across group boundaries.

    Args:
        filenames: Filenames object with itergroups() method
        groupby: Time frequency for grouping (e.g., 'h', '30m')
        overlap: Number of files to overlap from adjacent groups (default: 1)

    Returns:
        List of (period, files, n_overlap_before, n_overlap_after) tuples where:
        - period: Group start time
        - files: List of files including overlap from both directions
        - n_overlap_before: Number of overlap files at start (from previous group)
        - n_overlap_after: Number of overlap files at end (from next group)
    """
    groups = list(filenames.itergroups(groupby))
    result = []

    for i, (period, files) in enumerate(groups):
        # Add overlap from previous group at start
        if i > 0:
            prev_files = groups[i - 1][1][-overlap:]
        else:
            prev_files = []

        # Add overlap from next group at end
        if i < len(groups) - 1:
            next_files = groups[i + 1][1][:overlap]
        else:
            next_files = []

        combined = prev_files + files + next_files
        result.append((period, combined, len(prev_files), len(next_files)))

    return result


class FilesPipeline:
    """
    Process a list of polar files in a single process/thread.

    Aggregates results from multiple FilePipeline instances, collecting
    frame timings and metadata for batch processing.

    Files are processed sequentially in the order provided (typically time-sorted
    from Filenames), preserving time ordering for frame pair iteration.

    Supports overlap files at both ends for frame pairing across group boundaries.
    Creates FrameInterpolator objects for each primary frame during construction.
    """

    def __init__(
        self,
        filenames: list[str],
        config: Config | None = None,
        qSave: bool = False,
        qTiming: bool = False,
        n_overlap_files_before: int = 0,
        n_overlap_files_after: int = 0,
        tolerance: float = 1.2,
        qProject: bool = False,
    ) -> None:
        """
        Process a list of polar files.

        Args:
            filenames: List of polar file paths to process (should be time-sorted)
            config: YAML configuration information
            qSave: Save intermediate results for debugging
            qTiming: Time each processing step
            n_overlap_files_before: Number of files at the start that are overlap
                                   (from previous group, used for pairing only)
            n_overlap_files_after: Number of files at the end that are overlap
                                  (from next group, used for pairing only)
            tolerance: Multiplier for repeat_time to accept frame pair (1.2 = 20% margin)
            qProject: Perform earth projection after processing frames
        """
        self._filenames = filenames
        self._config = config
        self._tolerance = tolerance
        self._file_pipelines: list[FilePipeline] = []
        self._frame_pipelines: list[FramePipeline] = []
        self._frame_timings: list[dict[str, float]] = []
        self._interpolators: list[FrameInterpolator] = []
        self._total_frames = 0
        self._n_overlap_files_before = n_overlap_files_before
        self._n_overlap_files_after = n_overlap_files_after
        self._n_overlap_frames_before = 0
        self._n_overlap_frames_after = 0
        self._n_interpolated = 0
        self._n_extrapolated = 0
        self._n_skipped = 0

        # Spatial data for earth projection (collected during second loop)
        self._all_latitudes: list[np.ndarray] = []  # Per-radial latitudes per frame
        self._all_longitudes: list[np.ndarray] = []  # Per-radial longitudes per frame
        self._all_headings: list[np.ndarray] = []  # Per-radial headings per frame
        self._range_resolutions: list[float] = []  # Range resolution per frame
        self._max_ground_ranges: list[float] = []  # Max ground range per frame
        self._earth_grid: EarthGrid | None = None
        self._projection_result: ProjectionResult | None = None
        self._projection_timings: dict[str, float] = {}  # Timing for projection steps

        n_files = len(filenames)
        primary_start = n_overlap_files_before
        primary_end = n_files - n_overlap_files_after

        # First pass: collect all frames
        for i, fn in enumerate(filenames):
            try:
                pf = FilePipeline(fn, config=config, qSave=qSave, qTiming=qTiming)
                self._file_pipelines.append(pf)
                self._total_frames += len(pf)

                for frm in pf.frames:
                    self._frame_pipelines.append(frm)
                    if i < primary_start:
                        self._n_overlap_frames_before += 1
                    elif i >= primary_end:
                        self._n_overlap_frames_after += 1
                    if qTiming:
                        self._frame_timings.append(frm.timings)
            except Exception:
                logger.exception("Error processing %s", fn)
                raise

        self._n_primary_frames = (
            self._total_frames - self._n_overlap_frames_before - self._n_overlap_frames_after
        )

        # Second pass: create interpolators for primary frames and collect spatial data
        import time

        frames = self._frame_pipelines
        start = self._n_overlap_frames_before
        end = start + self._n_primary_frames

        for i in range(start, end):
            prev_frame = frames[i - 1] if i > 0 else None
            current_frame = frames[i]
            next_frame = frames[i + 1] if i + 1 < len(frames) else None

            try:
                t0 = time.perf_counter() if qTiming else None
                interp = FrameInterpolator(
                    prev_frame,
                    current_frame,
                    next_frame,
                    tolerance=tolerance,
                )
                if t0 is not None and i < len(self._frame_timings):
                    # Add interpolation timing to the corresponding frame's timings
                    self._frame_timings[i]["Interpolate"] = time.perf_counter() - t0
                self._interpolators.append(interp)
                if interp.method == "interpolate":
                    self._n_interpolated += 1
                else:
                    self._n_extrapolated += 1

                # Collect spatial data for earth projection
                self._all_latitudes.append(interp.latitudes)
                self._all_longitudes.append(interp.longitudes)
                self._all_headings.append(interp.headings)
                self._range_resolutions.append(current_frame.range_resolution)
                self._max_ground_ranges.append(current_frame.ground_range[-1])

            except ValueError as e:
                logger.warning("Skipping frame %d: %s", i, e)
                self._n_skipped += 1

        # Third pass: earth projection (if requested)
        if qProject and self._n_primary_frames > 0:
            self.create_earth_grid()
            self.project_to_earth()
            self.finalize_projection()

    def __repr__(self) -> str:
        return f"<FilesPipeline files={len(self._filenames)} frames={self._total_frames}>"

    def __bool__(self) -> bool:
        """Return True if any frames were processed."""
        return self._total_frames > 0

    def __len__(self) -> int:
        """Return the total number of processed frames."""
        return self._total_frames

    def __iter__(self) -> Iterator[FramePipeline]:
        """Iterate over all FramePipeline objects across all files."""
        for pf in self._file_pipelines:
            yield from pf.frames

    @property
    def filenames(self) -> list[str]:
        """Return the list of filenames."""
        return self._filenames

    @property
    def file_pipelines(self) -> list[FilePipeline]:
        """Return the list of FilePipeline objects."""
        return self._file_pipelines

    @property
    def frame_timings(self) -> list[dict[str, float]]:
        """Return the list of frame-level timings."""
        return self._frame_timings

    @property
    def total_frames(self) -> int:
        """Return the total number of frames processed."""
        return self._total_frames

    @property
    def config(self) -> Config | None:
        """Return the configuration object."""
        return self._config

    @property
    def n_primary_frames(self) -> int:
        """Return the number of primary (non-overlap) frames."""
        return self._n_primary_frames

    @property
    def n_overlap_frames(self) -> int:
        """Return the total number of overlap frames (before + after)."""
        return self._n_overlap_frames_before + self._n_overlap_frames_after

    @property
    def n_overlap_frames_before(self) -> int:
        """Return the number of overlap frames at the start."""
        return self._n_overlap_frames_before

    @property
    def n_overlap_frames_after(self) -> int:
        """Return the number of overlap frames at the end."""
        return self._n_overlap_frames_after

    @property
    def primary_frames(self) -> list[FramePipeline]:
        """Return only the primary (non-overlap) frames."""
        start = self._n_overlap_frames_before
        end = self._total_frames - self._n_overlap_frames_after
        return self._frame_pipelines[start:end]

    @property
    def interpolators(self) -> list[FrameInterpolator]:
        """Return the list of FrameInterpolator objects for primary frames."""
        return self._interpolators

    @property
    def n_interpolated(self) -> int:
        """Return the number of frames that were interpolated (forward)."""
        return self._n_interpolated

    @property
    def n_extrapolated(self) -> int:
        """Return the number of frames that were extrapolated (backward)."""
        return self._n_extrapolated

    @property
    def n_skipped(self) -> int:
        """Return the number of frames that were skipped (no valid pair)."""
        return self._n_skipped

    @property
    def earth_grid(self) -> EarthGrid | None:
        """Return the earth grid (available after create_earth_grid is called)."""
        return self._earth_grid

    @property
    def projection_result(self) -> ProjectionResult | None:
        """Return the projection result (available after project_to_earth is called)."""
        return self._projection_result

    @property
    def projection_timings(self) -> dict[str, float]:
        """Return timing statistics for projection steps."""
        return self._projection_timings

    @property
    def average_range_resolution(self) -> float:
        """Return the average range resolution across all frames in meters."""
        if not self._range_resolutions:
            return 0.0
        return float(np.mean(self._range_resolutions))

    @property
    def max_horizontal_range(self) -> float:
        """Return the maximum horizontal (ground) range across all frames in meters."""
        if not self._max_ground_ranges:
            return 0.0
        return float(np.max(self._max_ground_ranges))

    def create_earth_grid(self, padding: float = 1.1) -> EarthGrid:
        """
        Create an earth-referenced Cartesian coordinate system.

        The grid spacing is set to the average range resolution across all frames.
        The extent is determined by the ship track plus the maximum radar range.

        Args:
            padding: Multiplier for radar range to add margin (default 1.1 = 10% margin)

        Returns:
            EarthGrid object with x/y edges defined
        """
        import time as _time

        t0 = _time.perf_counter()

        if not self._all_latitudes:
            raise ValueError("No spatial data available - was interpolation performed?")

        # Calculate grid spacing from average range resolution
        grid_spacing = self.average_range_resolution
        if grid_spacing <= 0:
            raise ValueError("Invalid range resolution")

        # Get reference position (first radial of first frame)
        ref_lat = self._all_latitudes[0][0]
        ref_lon = self._all_longitudes[0][0]

        # Convert lat/lon to meters (flat earth approximation for local area)
        meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

        # Calculate ship track extent in meters
        all_lats = np.concatenate(self._all_latitudes)
        all_lons = np.concatenate(self._all_longitudes)

        ship_x = (all_lons - ref_lon) * meters_per_deg_lon
        ship_y = (all_lats - ref_lat) * meters_per_deg_lat

        # Grid extent: ship track + max radar range + padding
        max_range = self.max_horizontal_range * padding

        x_min = ship_x.min() - max_range
        x_max = ship_x.max() + max_range
        y_min = ship_y.min() - max_range
        y_max = ship_y.max() + max_range

        # Create bin edges aligned to grid spacing
        n_x = int(np.ceil((x_max - x_min) / grid_spacing))
        n_y = int(np.ceil((y_max - y_min) / grid_spacing))

        x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
        y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)

        # Initialize accumulation arrays
        intensity_sum = np.zeros((n_y, n_x), dtype=np.float64)
        intensity_count = np.zeros((n_y, n_x), dtype=np.int32)

        self._earth_grid = EarthGrid(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=grid_spacing,
            intensity_sum=intensity_sum,
            intensity_count=intensity_count,
            ref_latitude=ref_lat,
            ref_longitude=ref_lon,
        )

        elapsed = _time.perf_counter() - t0
        self._projection_timings["CreateGrid"] = elapsed

        logger.debug(
            "Created earth grid: %dx%d cells, %.2fm spacing, "
            "extent: [%.1f, %.1f] x [%.1f, %.1f] m (%.3fs)",
            n_x,
            n_y,
            grid_spacing,
            x_edges[0],
            x_edges[-1],
            y_edges[0],
            y_edges[-1],
            elapsed,
        )

        return self._earth_grid

    def project_to_earth(self) -> None:
        """
        Project all frames onto the earth grid (third loop).

        For each frame:
        - Compute earth bearing = theta (beam angle) + ship heading
        - Compute (x, y) for all radials and range bins using vectorized operations
        - Accumulate intensity into grid cells using bincount (faster than add.at)
        """
        import time as _time

        if self._earth_grid is None:
            self.create_earth_grid()

        grid = self._earth_grid
        primary_frames = self.primary_frames

        # Meters per degree for coordinate conversion
        meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(grid.ref_latitude))

        # Pre-compute grid origin and spacing for direct index calculation
        x_origin = grid.x_edges[0]
        y_origin = grid.y_edges[0]
        inv_spacing = 1.0 / grid.grid_spacing

        # Grid dimensions for linear indexing
        n_x = grid.n_x
        n_y = grid.n_y
        grid_size = n_x * n_y

        # Get flat views of accumulation arrays for efficient in-place addition
        sum_flat = grid.intensity_sum.ravel()
        count_flat = grid.intensity_count.ravel()

        # Track buffer dimensions for reuse (frames may have different sizes)
        buf_n_radials = 0
        buf_n_ranges = 0
        x_buf = None
        y_buf = None
        x_idx_buf = None
        y_idx_buf = None

        t0 = _time.perf_counter()
        n_projected = 0

        # Batch size for accumulating frames before bincount
        batch_size = 10
        batch_indices: list[np.ndarray] = []
        batch_values: list[np.ndarray] = []

        def flush_batch():
            """Process accumulated batch with single bincount call."""
            nonlocal n_projected
            if not batch_indices:
                return
            # Concatenate all indices and values from batch
            all_idx = np.concatenate(batch_indices)
            all_vals = np.concatenate(batch_values)

            # Single bincount for entire batch
            batch_sum = np.bincount(all_idx, weights=all_vals, minlength=grid_size)
            batch_count = np.bincount(all_idx, minlength=grid_size)

            # Accumulate into grid (use np.add with out= to avoid scoping issues with +=)
            np.add(sum_flat, batch_sum, out=sum_flat)
            np.add(count_flat, batch_count, out=count_flat)
            n_projected += len(all_idx)

            # Clear batch
            batch_indices.clear()
            batch_values.clear()

        for frame_idx, frame in enumerate(primary_frames):
            # Get frame data
            theta_ship = frame.theta_array  # Beam angles relative to ship (degrees)
            ground_range = frame.ground_range  # Distance per range bin (meters)
            intensity = frame.final_intensity  # Dewinded intensity

            # Get interpolated navigation for this frame
            lats = self._all_latitudes[frame_idx]
            lons = self._all_longitudes[frame_idx]
            headings = self._all_headings[frame_idx]

            # Get frame dimensions
            n_radials = len(theta_ship)
            n_ranges = len(ground_range)

            # Allocate/reallocate buffers if dimensions changed
            if n_radials != buf_n_radials or n_ranges != buf_n_ranges:
                buf_n_radials = n_radials
                buf_n_ranges = n_ranges
                x_buf = np.empty((n_radials, n_ranges), dtype=np.float64)
                y_buf = np.empty((n_radials, n_ranges), dtype=np.float64)
                x_idx_buf = np.empty((n_radials, n_ranges), dtype=np.int32)
                y_idx_buf = np.empty((n_radials, n_ranges), dtype=np.int32)

            # Convert ship positions to meters from reference (n_radials,)
            ship_x = (lons - grid.ref_longitude) * meters_per_deg_lon
            ship_y = (lats - grid.ref_latitude) * meters_per_deg_lat

            # Compute earth bearing for each radial (n_radials,)
            earth_bearing_rad = np.deg2rad((theta_ship + headings) % 360)

            # Vectorized projection using pre-allocated buffers
            sin_bearing = np.sin(earth_bearing_rad)
            cos_bearing = np.cos(earth_bearing_rad)

            # Compute x, y for all points into pre-allocated buffers
            # x = ship_x + ground_range * sin(bearing)  (East)
            # y = ship_y + ground_range * cos(bearing)  (North)
            np.outer(sin_bearing, ground_range, out=x_buf)
            x_buf += ship_x[:, np.newaxis]
            np.outer(cos_bearing, ground_range, out=y_buf)
            y_buf += ship_y[:, np.newaxis]

            # Direct index calculation (in-place to avoid temporaries)
            x_buf -= x_origin
            x_buf *= inv_spacing
            y_buf -= y_origin
            y_buf *= inv_spacing

            # Convert to int32 indices
            np.copyto(x_idx_buf, x_buf, casting="unsafe")
            np.copyto(y_idx_buf, y_buf, casting="unsafe")

            # Flatten arrays (views, no allocation)
            x_flat = x_idx_buf.ravel()
            y_flat = y_idx_buf.ravel()
            values_flat = intensity.ravel()

            # Filter valid indices (within grid bounds and non-NaN intensity)
            valid = (
                (x_flat >= 0)
                & (x_flat < n_x)
                & (y_flat >= 0)
                & (y_flat < n_y)
                & ~np.isnan(values_flat)
            )

            if np.any(valid):
                # Convert 2D indices to 1D linear indices and add to batch
                linear_idx = y_flat[valid] * n_x + x_flat[valid]
                batch_indices.append(linear_idx.copy())
                batch_values.append(values_flat[valid].copy())

                # Flush batch when it reaches batch_size
                if len(batch_indices) >= batch_size:
                    flush_batch()

        # Flush any remaining frames
        flush_batch()

        elapsed = _time.perf_counter() - t0
        self._projection_timings["Project"] = elapsed
        logger.info(
            "Projected %d frames, %d values in %.2fs (%.0f values/s)",
            len(primary_frames),
            n_projected,
            elapsed,
            n_projected / elapsed,
        )

    def finalize_projection(self) -> ProjectionResult:
        """
        Finalize the projection and clean up memory.

        Collects metadata, creates ProjectionResult, and deletes
        frame data that is no longer needed.

        Returns:
            ProjectionResult with earth grid and metadata
        """
        import time as _time

        t0 = _time.perf_counter()

        if self._earth_grid is None:
            raise ValueError("Earth grid not created - call project_to_earth first")

        primary_frames = self.primary_frames
        n_frames = len(primary_frames)

        # Collect ship navigation data per frame
        ship_speeds = np.array([f.metadata.ship_speed or 0.0 for f in primary_frames])
        ship_headings = np.array([np.mean(self._all_headings[i]) for i in range(n_frames)])

        # Collect wind data per frame
        wind_speeds = np.array([f.metadata.wind_speed or 0.0 for f in primary_frames])
        wind_directions = np.array([f.metadata.wind_direction or 0.0 for f in primary_frames])

        # Collect timing data
        frame_start_times = np.array(
            [interp.times[0] for interp in self._interpolators], dtype="datetime64[ns]"
        )
        frame_end_times = np.array(
            [interp.times[-1] for interp in self._interpolators], dtype="datetime64[ns]"
        )

        # Count total radials
        n_radials_total = sum(len(self._all_latitudes[i]) for i in range(n_frames))

        self._projection_result = ProjectionResult(
            earth_grid=self._earth_grid,
            ship_speeds=ship_speeds,
            ship_headings=ship_headings,
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            frame_start_times=frame_start_times,
            frame_end_times=frame_end_times,
            n_frames=n_frames,
            n_radials_total=n_radials_total,
        )

        self._projection_timings["Finalize"] = _time.perf_counter() - t0

        # Clean up memory - delete frame data, interpolators, spatial arrays
        t_cleanup = _time.perf_counter()
        self._cleanup_memory()
        self._projection_timings["Cleanup"] = _time.perf_counter() - t_cleanup

        # Log total projection timing
        total_time = sum(self._projection_timings.values())
        logger.info(
            "Projection complete: %.3fs (grid=%.3fs, project=%.3fs, finalize=%.3fs, cleanup=%.3fs)",
            total_time,
            self._projection_timings.get("CreateGrid", 0),
            self._projection_timings.get("Project", 0),
            self._projection_timings.get("Finalize", 0),
            self._projection_timings.get("Cleanup", 0),
        )

        return self._projection_result

    def _cleanup_memory(self) -> None:
        """Delete frame data and intermediate results to free memory."""
        import gc

        # Clear frame pipelines (contains intensity arrays)
        for fp in self._file_pipelines:
            for frm in fp.frames:
                frm._final_intensity = None
                frm._theta_array = None
                frm._ground_range = None

        # Clear interpolators
        self._interpolators.clear()

        # Clear spatial arrays (already copied to result)
        self._all_latitudes.clear()
        self._all_longitudes.clear()
        self._all_headings.clear()
        self._range_resolutions.clear()
        self._max_ground_ranges.clear()

        # Clear frame pipeline references
        self._frame_pipelines.clear()
        self._file_pipelines.clear()

        # Force garbage collection
        gc.collect()

        logger.info("Memory cleanup complete")


class ProjectionDiagnostics:
    """
    Diagnostic visualization for earth projection results.

    Provides plotting for:
    - Projected intensity on earth grid
    - Ship speed and heading polar plots
    - Wind speed and direction polar plots
    - Time series of navigation and environmental data
    """

    def __init__(self, result: ProjectionResult) -> None:
        """
        Initialize diagnostics viewer.

        Args:
            result: ProjectionResult from FilesPipeline.finalize_projection()
        """
        self._result = result
        self._grid = result.earth_grid

    @property
    def result(self) -> ProjectionResult:
        """Return the projection result."""
        return self._result

    def plot(
        self,
        figsize: tuple[float, float] = (16, 12),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Create comprehensive diagnostic plot.

        Shows:
        - Main: Projected intensity on earth grid
        - Top-right: Ship speed/heading polar plot
        - Bottom-right: Wind speed/direction polar plot

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plot
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        result = self._result
        grid = self._grid

        # Get intensity data
        intensity = grid.intensity

        # Auto-scale if not specified
        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

        # Main intensity plot (spans left 2 columns)
        ax_main = fig.add_subplot(gs[:, 0])
        self._plot_intensity(ax_main, intensity, cmap, vmin, vmax)

        # Ship speed/heading polar plot (top-right)
        ax_ship = fig.add_subplot(gs[0, 1], projection="polar")
        self._plot_ship_polar(ax_ship)

        # Wind speed/direction polar plot (bottom-right)
        ax_wind = fig.add_subplot(gs[1, 1], projection="polar")
        self._plot_wind_polar(ax_wind)

        # Time series plot (right column, spans both rows)
        ax_time = fig.add_subplot(gs[:, 2])
        self._plot_time_series(ax_time)

        # Title
        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        fig.suptitle(
            f"Earth Projection: {result.n_frames} frames, {result.n_radials_total} radials\n"
            f"{start_time} to {end_time}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()

    def _plot_intensity(
        self,
        ax,
        intensity: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
    ) -> None:
        """Plot projected intensity on earth grid."""
        grid = self._grid

        im = ax.pcolormesh(
            grid.x_edges, grid.y_edges, intensity, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Projected Intensity")
        ax.set_aspect("equal")

        # Add colorbar
        ax.figure.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        # Add grid info
        ax.text(
            0.02,
            0.98,
            f"Grid: {grid.n_x}x{grid.n_y}\nSpacing: {grid.grid_spacing:.1f}m",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def _plot_ship_polar(self, ax) -> None:
        """
        Plot ship speed and heading as a polar scatter plot.

        Heading is the angle (0=North, clockwise), speed is the radius.
        """
        result = self._result

        # Convert headings to radians (matplotlib polar: 0=East, counter-clockwise)
        # We want 0=North, clockwise, so: theta = 90 - heading (in degrees) then to radians
        headings_rad = np.deg2rad(90 - result.ship_headings)

        # Speed is the radius
        speeds = result.ship_speeds

        # Color by frame index for temporal information
        colors = np.arange(len(speeds))

        ax.scatter(headings_rad, speeds, c=colors, cmap="viridis", s=20, alpha=0.7)

        # Configure polar plot
        ax.set_theta_zero_location("N")  # 0 degrees at top
        ax.set_theta_direction(-1)  # Clockwise

        ax.set_title("Ship Speed/Heading", pad=10)
        ax.set_xlabel("")

        # Add statistics
        mean_speed = np.mean(speeds)
        mean_heading = self._circular_mean(result.ship_headings)
        ax.text(
            0.5,
            -0.15,
            f"Mean: {mean_speed:.1f} m/s @ {mean_heading:.0f}°",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    def _plot_wind_polar(self, ax) -> None:
        """
        Plot wind speed and direction as a polar scatter plot.

        Direction is the angle (0=North, clockwise - direction wind comes FROM),
        speed is the radius.
        """
        result = self._result

        # Convert directions to radians
        directions_rad = np.deg2rad(90 - result.wind_directions)

        # Speed is the radius
        speeds = result.wind_speeds

        # Color by frame index
        colors = np.arange(len(speeds))

        ax.scatter(directions_rad, speeds, c=colors, cmap="coolwarm", s=20, alpha=0.7)

        # Configure polar plot
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.set_title("Wind Speed/Direction", pad=10)

        # Add statistics
        mean_speed = np.mean(speeds)
        mean_dir = self._circular_mean(result.wind_directions)
        ax.text(
            0.5,
            -0.15,
            f"Mean: {mean_speed:.1f} m/s from {mean_dir:.0f}°",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    def _plot_time_series(self, ax) -> None:
        """Plot time series of ship and wind data."""
        result = self._result

        # Convert times to relative seconds from start
        t0 = result.frame_start_times[0]
        times = (result.frame_start_times - t0) / np.timedelta64(1, "s")

        # Create twin axis for speeds
        ax2 = ax.twinx()

        # Plot headings on primary axis
        ax.plot(times, result.ship_headings, "b-", label="Ship heading", alpha=0.7)
        ax.plot(times, result.wind_directions, "r-", label="Wind direction", alpha=0.7)
        ax.set_ylabel("Direction (degrees)")
        ax.set_ylim(0, 360)
        ax.legend(loc="upper left", fontsize=8)

        # Plot speeds on secondary axis
        ax2.plot(times, result.ship_speeds, "b--", label="Ship speed", alpha=0.7)
        ax2.plot(times, result.wind_speeds, "r--", label="Wind speed", alpha=0.7)
        ax2.set_ylabel("Speed (m/s)")
        ax2.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("Time (s)")
        ax.set_title("Time Series")
        ax.grid(True, alpha=0.3)

    def _circular_mean(self, angles: np.ndarray) -> float:
        """Calculate circular mean of angles in degrees."""
        angles_rad = np.deg2rad(angles)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360

    def plot_intensity_only(
        self,
        figsize: tuple[float, float] = (12, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot only the projected intensity.

        Args:
            figsize: Figure size
            cmap: Colormap name
            vmin: Minimum intensity value
            vmax: Maximum intensity value
        """
        import matplotlib.pyplot as plt

        grid = self._grid
        result = self._result
        intensity = grid.intensity

        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.pcolormesh(
            grid.x_edges, grid.y_edges, intensity, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")

        fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        ax.set_title(f"Projected Intensity\n{result.n_frames} frames, {start_time} to {end_time}")

        # Add statistics
        stats_text = (
            f"Grid: {grid.n_x}x{grid.n_y} ({grid.grid_spacing:.1f}m)\n"
            f"Extent: [{grid.x_edges[0]:.0f}, {grid.x_edges[-1]:.0f}] x "
            f"[{grid.y_edges[0]:.0f}, {grid.y_edges[-1]:.0f}] m"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

    def save_intensity(
        self,
        output_path: str,
        figsize: tuple[float, float] = (12, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        dpi: int = 150,
    ) -> None:
        """
        Save projected intensity plot to file.

        Args:
            output_path: Path to save image (e.g., 'projection.png')
            figsize: Figure size
            cmap: Colormap name
            vmin: Minimum intensity value
            vmax: Maximum intensity value
            dpi: Image resolution
        """
        import matplotlib.pyplot as plt

        grid = self._grid
        result = self._result
        intensity = grid.intensity

        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.pcolormesh(
            grid.x_edges, grid.y_edges, intensity, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")

        fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        ax.set_title(f"Projected Intensity: {result.n_frames} frames\n{start_time} to {end_time}")

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved intensity plot to %s", output_path)

    def __repr__(self) -> str:
        return (
            f"ProjectionDiagnostics(frames={self._result.n_frames}, "
            f"grid={self._grid.n_x}x{self._grid.n_y}, "
            f"spacing={self._grid.grid_spacing:.1f}m)"
        )


class ProjectionViewer:
    """
    Interactive viewer for multiple earth projection results.

    Provides navigation buttons (Back, Play/Pause, Next) and keyboard shortcuts
    to walk through a sequence of ProjectionResult objects.

    Keyboard shortcuts:
    - Left arrow: Previous result
    - Right arrow: Next result
    - Space: Play/Pause
    - q/Escape: Close viewer
    """

    def __init__(
        self,
        results: list[ProjectionResult],
        labels: list[str] | None = None,
    ) -> None:
        """
        Initialize the projection viewer.

        Args:
            results: List of ProjectionResult objects to display
            labels: Optional labels for each result (e.g., group periods)
        """
        if not results:
            raise ValueError("No results to display")

        self._results = results
        self._labels = labels or [f"Result {i + 1}" for i in range(len(results))]
        self._current_idx = 0
        self._playing = False
        self._play_interval = 1000  # milliseconds
        self._timer = None

        # Plot settings (set during show())
        self._cmap = "viridis"
        self._vmin: float | None = None
        self._vmax: float | None = None

        # Figure and axes (created during show())
        self._fig = None
        self._ax_main = None
        self._ax_ship = None
        self._ax_wind = None
        self._ax_time = None
        self._colorbar = None
        self._im = None

    @property
    def n_results(self) -> int:
        """Return the number of results."""
        return len(self._results)

    @property
    def current_index(self) -> int:
        """Return the current result index."""
        return self._current_idx

    @property
    def current_result(self) -> ProjectionResult:
        """Return the current result."""
        return self._results[self._current_idx]

    @property
    def current_label(self) -> str:
        """Return the current label."""
        return self._labels[self._current_idx]

    def show(
        self,
        figsize: tuple[float, float] = (16, 12),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        play_interval: int = 1000,
    ) -> None:
        """
        Display the interactive viewer.

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plot
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
            play_interval: Interval between frames in play mode (milliseconds)
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.gridspec import GridSpec

        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax
        self._play_interval = play_interval

        # Compute global vmin/vmax across all results if not specified
        if self._vmin is None or self._vmax is None:
            all_intensities = []
            for r in self._results:
                intensity = r.earth_grid.intensity
                all_intensities.append(intensity[~np.isnan(intensity)])
            combined = np.concatenate(all_intensities)
            if self._vmin is None:
                self._vmin = np.percentile(combined, 2)
            if self._vmax is None:
                self._vmax = np.percentile(combined, 98)

        # Create figure with custom layout
        self._fig = plt.figure(figsize=figsize)

        # Plot area grid
        gs_plots = GridSpec(
            2,
            3,
            width_ratios=[2, 1, 1],
            height_ratios=[1, 1],
            wspace=0.3,
            hspace=0.3,
            top=0.92,
            bottom=0.12,
            left=0.06,
            right=0.98,
        )

        # Create plot axes
        self._ax_main = self._fig.add_subplot(gs_plots[:, 0])
        self._ax_ship = self._fig.add_subplot(gs_plots[0, 1], projection="polar")
        self._ax_wind = self._fig.add_subplot(gs_plots[1, 1], projection="polar")
        self._ax_time = self._fig.add_subplot(gs_plots[:, 2])

        # Create navigation buttons
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        center_x = 0.5

        ax_back = self._fig.add_axes([center_x - btn_width * 1.6, btn_y, btn_width, btn_height])
        ax_play = self._fig.add_axes([center_x - btn_width * 0.5, btn_y, btn_width, btn_height])
        ax_next = self._fig.add_axes([center_x + btn_width * 0.6, btn_y, btn_width, btn_height])

        self._btn_back = Button(ax_back, "< Back")
        self._btn_play = Button(ax_play, "Play")
        self._btn_next = Button(ax_next, "Next >")

        self._btn_back.on_clicked(self._on_back)
        self._btn_play.on_clicked(self._on_play)
        self._btn_next.on_clicked(self._on_next)

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("close_event", self._on_close)

        # Initial plot
        self._update_plot()

        plt.show()

    def _update_plot(self) -> None:
        """Update all plots for the current result."""
        result = self.current_result
        grid = result.earth_grid
        intensity = grid.intensity

        # Clear axes
        self._ax_main.clear()
        self._ax_ship.clear()
        self._ax_wind.clear()
        self._ax_time.clear()

        # Main intensity plot
        self._im = self._ax_main.pcolormesh(
            grid.x_edges,
            grid.y_edges,
            intensity,
            cmap=self._cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            shading="flat",
        )

        self._ax_main.set_xlabel("East (m)")
        self._ax_main.set_ylabel("North (m)")
        self._ax_main.set_title("Dewinded Intensity")
        self._ax_main.set_aspect("equal")

        # Add colorbar (only on first update, then reuse)
        if self._colorbar is None:
            self._colorbar = self._fig.colorbar(
                self._im, ax=self._ax_main, label="Intensity", shrink=0.8
            )
        else:
            self._colorbar.update_normal(self._im)

        # Grid info
        self._ax_main.text(
            0.02,
            0.98,
            f"Grid: {grid.n_x}x{grid.n_y}\nSpacing: {grid.grid_spacing:.1f}m",
            transform=self._ax_main.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Ship polar plot
        self._plot_ship_polar(self._ax_ship, result)

        # Wind polar plot
        self._plot_wind_polar(self._ax_wind, result)

        # Time series
        self._plot_time_series(self._ax_time, result)

        # Title with navigation info
        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        play_status = " [Playing]" if self._playing else ""
        self._fig.suptitle(
            f"{self.current_label} ({self._current_idx + 1}/{self.n_results}){play_status}\n"
            f"{result.n_frames} frames, {result.n_radials_total} radials | "
            f"{start_time} to {end_time}",
            fontsize=12,
        )

        self._fig.canvas.draw_idle()

    def _plot_ship_polar(self, ax, result: ProjectionResult) -> None:
        """Plot ship speed and heading as a polar scatter plot."""
        headings_rad = np.deg2rad(90 - result.ship_headings)
        speeds = result.ship_speeds
        colors = np.arange(len(speeds))

        ax.scatter(headings_rad, speeds, c=colors, cmap="viridis", s=20, alpha=0.7)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title("Ship Speed/Heading", pad=10)

        mean_speed = np.mean(speeds)
        mean_heading = self._circular_mean(result.ship_headings)
        ax.text(
            0.5,
            -0.15,
            f"Mean: {mean_speed:.1f} m/s @ {mean_heading:.0f}\u00b0",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    def _plot_wind_polar(self, ax, result: ProjectionResult) -> None:
        """Plot wind speed and direction as a polar scatter plot."""
        directions_rad = np.deg2rad(90 - result.wind_directions)
        speeds = result.wind_speeds
        colors = np.arange(len(speeds))

        ax.scatter(directions_rad, speeds, c=colors, cmap="coolwarm", s=20, alpha=0.7)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title("Wind Speed/Direction", pad=10)

        mean_speed = np.mean(speeds)
        mean_dir = self._circular_mean(result.wind_directions)
        ax.text(
            0.5,
            -0.15,
            f"Mean: {mean_speed:.1f} m/s from {mean_dir:.0f}\u00b0",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    def _plot_time_series(self, ax, result: ProjectionResult) -> None:
        """Plot time series of ship and wind data."""
        t0 = result.frame_start_times[0]
        times = (result.frame_start_times - t0) / np.timedelta64(1, "s")

        ax2 = ax.twinx()

        ax.plot(times, result.ship_headings, "b-", label="Ship heading", alpha=0.7)
        ax.plot(times, result.wind_directions, "r-", label="Wind direction", alpha=0.7)
        ax.set_ylabel("Direction (degrees)")
        ax.set_ylim(0, 360)
        ax.legend(loc="upper left", fontsize=8)

        ax2.plot(times, result.ship_speeds, "b--", label="Ship speed", alpha=0.7)
        ax2.plot(times, result.wind_speeds, "r--", label="Wind speed", alpha=0.7)
        ax2.set_ylabel("Speed (m/s)")
        ax2.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("Time (s)")
        ax.set_title("Time Series")
        ax.grid(True, alpha=0.3)

    def _circular_mean(self, angles: np.ndarray) -> float:
        """Calculate circular mean of angles in degrees."""
        angles_rad = np.deg2rad(angles)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360

    def _on_back(self, event) -> None:
        """Handle back button click."""
        self._go_back()

    def _on_next(self, event) -> None:
        """Handle next button click."""
        self._go_next()

    def _on_play(self, event) -> None:
        """Handle play/pause button click."""
        self._toggle_play()

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == "left":
            self._go_back()
        elif event.key == "right":
            self._go_next()
        elif event.key == " ":
            self._toggle_play()
        elif event.key in ("q", "escape"):
            self._stop_play()
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _on_close(self, event) -> None:
        """Handle window close event."""
        self._stop_play()

    def _go_back(self) -> None:
        """Go to the previous result."""
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_plot()

    def _go_next(self) -> None:
        """Go to the next result."""
        if self._current_idx < self.n_results - 1:
            self._current_idx += 1
            self._update_plot()

    def _toggle_play(self) -> None:
        """Toggle play/pause mode."""
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        """Start auto-play mode."""
        if self._playing:
            return

        self._playing = True
        self._btn_play.label.set_text("Pause")

        # Use matplotlib timer for animation
        self._timer = self._fig.canvas.new_timer(interval=self._play_interval)
        self._timer.add_callback(self._play_step)
        self._timer.start()

        self._update_plot()

    def _stop_play(self) -> None:
        """Stop auto-play mode."""
        if not self._playing:
            return

        self._playing = False
        self._btn_play.label.set_text("Play")

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        self._update_plot()

    def _play_step(self) -> None:
        """Advance to next frame during play mode."""
        if not self._playing:
            return

        if self._current_idx < self.n_results - 1:
            self._current_idx += 1
            self._update_plot()
        else:
            # Loop back to start
            self._current_idx = 0
            self._update_plot()

    def __repr__(self) -> str:
        return (
            f"ProjectionViewer(n_results={self.n_results}, "
            f"current={self._current_idx + 1}/{self.n_results})"
        )


def _process_group(
    group_data: tuple[np.datetime64, list[str], int, int],
    config: "Config",
    qTiming: bool,
    tolerance: float,
    qProject: bool = False,
) -> dict:
    """Process a single group and return results with memory usage."""
    import resource

    period, group_files, n_overlap_before, n_overlap_after = group_data

    fp = FilesPipeline(
        group_files,
        config=config,
        qTiming=qTiming,
        n_overlap_files_before=n_overlap_before,
        n_overlap_files_after=n_overlap_after,
        tolerance=tolerance,
        qProject=qProject,
    )

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result = {
        "period": str(period),
        "n_files": len(group_files),
        "n_overlap_before": n_overlap_before,
        "n_overlap_after": n_overlap_after,
        "n_primary_frames": fp.n_primary_frames,
        "n_interpolated": fp.n_interpolated,
        "n_extrapolated": fp.n_extrapolated,
        "frame_timings": fp.frame_timings if qTiming else [],
        "projection_timings": fp.projection_timings if qTiming else {},
        "projection_result": fp.projection_result,
        "peak_rss": peak_rss,
    }
    return result


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument("--timing", "-t", action="store_true", help="Show timing statistics")
    parser.add_argument(
        "--groupby",
        "-g",
        type=str,
        default=None,
        help="Group files by time frequency (e.g., 'h', '30m')",
    )
    parser.add_argument(
        "--overlap", type=int, default=1, help="Number of overlap files between groups (default: 1)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.2,
        help="Time tolerance multiplier for interpolation (default: 1.2)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Number of parallel workers (default: auto)"
    )

    # Pool type selection (mutually exclusive)
    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument("--threadpool", action="store_true", help="Use only ThreadPoolExecutor")
    pool_group.add_argument(
        "--processpool", action="store_true", help="Use only ProcessPoolExecutor"
    )
    pool_group.add_argument(
        "--bothpools", action="store_true", help="Run both ThreadPool and ProcessPool (default)"
    )

    parser.add_argument("--project", "-p", action="store_true", help="Project frames to earth grid")
    parser.add_argument(
        "--plot", action="store_true", help="Show diagnostic plots (implies --project)"
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        metavar="FILE",
        help="Save intensity plot to file (implies --project)",
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis", help="Colormap for intensity plot (default: viridis)"
    )
    parser.add_argument(
        "--vmin", type=float, default=None, help="Minimum intensity value for colormap"
    )
    parser.add_argument(
        "--vmax", type=float, default=None, help="Maximum intensity value for colormap"
    )
    parser.add_argument(
        "--play-interval",
        type=int,
        default=1000,
        help="Interval between frames in play mode (milliseconds, default: 1000)",
    )


def add_subparser(subparsers) -> None:
    """Register the 'files-pipeline' subcommand."""
    p = subparsers.add_parser(
        "files-pipeline",
        help="Process multiple polar files with interpolation",
        description="Test files processing pipeline with frame interpolation",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def _display_projection_stats(
    result: ProjectionResult,
    label: str = "",
) -> None:
    """Display projection statistics for a single result (grid and frame info only)."""
    grid = result.earth_grid

    if label:
        logging.info("")
        logging.info("  %s:", label)

    logging.info("    Grid: %dx%d cells (%.1fm spacing)", grid.n_x, grid.n_y, grid.grid_spacing)
    logging.info(
        "    Extent: [%.0f, %.0f] x [%.0f, %.0f] m",
        grid.x_edges[0],
        grid.x_edges[-1],
        grid.y_edges[0],
        grid.y_edges[-1],
    )
    logging.info("    Frames: %d, Radials: %d", result.n_frames, result.n_radials_total)


def _display_projection_timing_stats(all_timings: dict[str, list[float]]) -> None:
    """Display aggregated projection timing statistics."""
    if not all_timings:
        return

    total_time = sum(sum(vals) for vals in all_timings.values())

    logging.info("")
    logging.info("  Projection timing statistics:")
    header = f"    {'Step':<12} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'%':>6}"
    logging.info(header)
    logging.info("    " + "-" * (len(header) - 4))
    for key, vals in all_timings.items():
        arr = np.array(vals) * 1000  # Convert to ms
        pct = (sum(vals) / total_time * 100) if total_time > 0 else 0
        logging.info(
            f"    {key:<12} {arr.mean():>10.2f} {arr.std():>10.2f} "
            f"{arr.min():>10.2f} {arr.max():>10.2f} {pct:>6.1f}"
        )


def run(args) -> None:
    """Execute the 'files-pipeline' command with parallel processing."""
    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from functools import partial

    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.parallel_runner import (
        aggregate_timings,
        display_benchmark_header,
        display_memory_stats,
        display_timing_stats,
        run_with_executor,
    )

    config = Config(args.config) if args.config else Config()

    filenames = Filenames(args.stime, args.etime, args.polar_path)
    files = list(filenames)

    if not files:
        logging.warning(
            "No files found in %s for time range %s to %s", args.polar_path, args.stime, args.etime
        )
        return

    logging.info("Found %d files", len(files))

    # Check if projection/plotting requested
    do_project = args.project or args.plot or args.save_plot

    if not args.groupby:
        # Non-grouped mode: process all files sequentially
        fp = FilesPipeline(
            files,
            config=config,
            qTiming=args.timing,
            tolerance=args.tolerance,
            qProject=do_project,
        )

        logging.info("Processed %d files, %d frames", len(files), fp.total_frames)
        logging.info(
            "Interpolation: %d interpolated, %d extrapolated, %d skipped",
            fp.n_interpolated,
            fp.n_extrapolated,
            fp.n_skipped,
        )

        if args.timing and fp.frame_timings:
            all_timings = aggregate_timings(
                [{"frame_timings": fp.frame_timings}],
                get_timings=lambda r: r["frame_timings"],
            )
            display_timing_stats(all_timings)

        # Display projection results and plots
        if do_project and fp.projection_result is not None:
            logging.info("")
            logging.info("=" * 60)
            logging.info("Earth Projection")
            logging.info("=" * 60)

            _display_projection_stats(fp.projection_result)

            # Display projection timing statistics
            if args.timing and fp.projection_timings:
                # Convert single timing dict to aggregated format
                proj_timings = {k: [v] for k, v in fp.projection_timings.items()}
                _display_projection_timing_stats(proj_timings)

            # Create diagnostics and plot
            diag = ProjectionDiagnostics(fp.projection_result)

            if args.save_plot:
                diag.save_intensity(
                    args.save_plot,
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )

            if args.plot:
                diag.plot(cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)

        return

    # Grouped mode with parallel processing
    groups = create_overlapping_groups(filenames, args.groupby, overlap=args.overlap)
    logging.info(
        "Created %d groups with %d overlap file(s) on each side", len(groups), args.overlap
    )

    process_func = partial(
        _process_group,
        config=config,
        qTiming=args.timing,
        tolerance=args.tolerance,
        qProject=do_project,
    )

    # Determine number of workers
    n_workers = args.workers
    if n_workers is None:
        n_workers = min(len(groups), os.cpu_count() or 1)

    # Select executor(s) based on arguments
    executors: list[tuple[str, type]] = []
    if args.threadpool:
        executors = [("ThreadPool", ThreadPoolExecutor)]
    elif args.processpool:
        executors = [("ProcessPool", ProcessPoolExecutor)]
    else:
        # Default: run both (--bothpools or no option specified)
        executors = [
            ("ThreadPool", ThreadPoolExecutor),
            ("ProcessPool", ProcessPoolExecutor),
        ]

    for executor_name, Executor in executors:
        logging.info("")
        logging.info("=" * 60)
        logging.info("Running with %s (%d workers)", executor_name, n_workers)
        logging.info("=" * 60)

        bench = run_with_executor(
            executor_name=executor_name,
            Executor=Executor,
            items=groups,
            process_func=process_func,
            n_workers=n_workers,
            item_desc="group",
            get_rss=lambda r: r["peak_rss"],
        )
        # Aggregate statistics
        total_frames = sum(r["n_primary_frames"] for r in bench.results)
        total_interpolated = sum(r["n_interpolated"] for r in bench.results)
        total_extrapolated = sum(r["n_extrapolated"] for r in bench.results)
        total_skipped = total_frames - total_interpolated - total_extrapolated

        display_benchmark_header(
            executor_name=bench.executor_name,
            n_items=len(groups),
            item_label="Groups",
            total_count=total_frames,
            count_label="Frames",
            elapsed=bench.elapsed,
            n_workers=bench.n_workers,
            extra_lines=[
                f"Interpolated: {total_interpolated}, "
                f"Extrapolated: {total_extrapolated}, "
                f"Skipped: {total_skipped}"
            ],
        )

        if args.timing:
            all_timings = aggregate_timings(
                bench.results,
                get_timings=lambda r: r["frame_timings"],
            )
            display_timing_stats(all_timings)

        display_memory_stats(bench.max_worker_rss)

        # Display projection results and plots for each group
        if do_project:
            # Collect valid projection results, labels, and timings
            proj_results = []
            proj_labels = []
            all_proj_timings: dict[str, list[float]] = {}

            for r in bench.results:
                proj_result = r.get("projection_result")
                if proj_result is not None:
                    proj_results.append(proj_result)
                    proj_labels.append(f"Group {r['period']}")

                    # Collect projection timings for aggregation
                    proj_timings = r.get("projection_timings", {})
                    for key, val in proj_timings.items():
                        if key not in all_proj_timings:
                            all_proj_timings[key] = []
                        all_proj_timings[key].append(val)

            # Display aggregated projection timing statistics
            if args.timing and all_proj_timings:
                _display_projection_timing_stats(all_proj_timings)

            # Save plots for each group
            if args.save_plot:
                for i, (proj_result, r) in enumerate(zip(proj_results, bench.results)):
                    diag = ProjectionDiagnostics(proj_result)
                    # Generate unique filename for each group
                    base, ext = (
                        args.save_plot.rsplit(".", 1)
                        if "." in args.save_plot
                        else (args.save_plot, "png")
                    )
                    group_file = f"{base}_{r['period'].replace(':', '-')}.{ext}"
                    diag.save_intensity(
                        group_file,
                        cmap=args.cmap,
                        vmin=args.vmin,
                        vmax=args.vmax,
                    )

            # Show interactive viewer with all groups
            if args.plot and proj_results:
                viewer = ProjectionViewer(proj_results, labels=proj_labels)
                viewer.show(
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    play_interval=args.play_interval,
                )


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test files processing pipeline")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
