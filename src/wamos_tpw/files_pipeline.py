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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import Config

logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================


@dataclass
class TimeWindowConfig:
    """Configuration for time-based windowing of frames."""

    window_seconds: float = 60.0  # Window duration in seconds
    overlap_fraction: float = 0.5  # Overlap between consecutive windows (0.0-1.0)
    min_frames_per_window: int = 5  # Minimum frames required to produce output

    @property
    def stride_seconds(self) -> float:
        """Compute stride (time between window starts) from overlap."""
        return self.window_seconds * (1.0 - self.overlap_fraction)

    def __post_init__(self):
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("overlap_fraction must be in [0, 1)")
        if self.min_frames_per_window < 1:
            raise ValueError("min_frames_per_window must be at least 1")


@dataclass
class MergedImage:
    """A motion-corrected composite image from multiple frames."""

    intensity: np.ndarray  # 2D averaged intensity (n_y, n_x)
    x_edges: np.ndarray  # Grid x edges in meters (from center)
    y_edges: np.ndarray  # Grid y edges in meters (from center)
    start_time: np.datetime64  # Window start time
    end_time: np.datetime64  # Window end time
    n_frames: int  # Number of frames merged
    utm_zone: int  # UTM zone number
    hemisphere: str  # 'north' or 'south'
    center_lat: float  # Grid center latitude
    center_lon: float  # Grid center longitude
    grid_spacing: float  # Grid cell size in meters
    mean_heading: float  # Mean ship heading during window
    mean_ship_speed: float | None = None
    mean_wind_speed: float | None = None
    mean_wind_direction: float | None = None
    window_index: int = 0  # Index of this window in the sequence

    @property
    def n_x(self) -> int:
        """Number of x bins."""
        return len(self.x_edges) - 1

    @property
    def n_y(self) -> int:
        """Number of y bins."""
        return len(self.y_edges) - 1

    @property
    def x_centers(self) -> np.ndarray:
        """Grid x centers."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        """Grid y centers."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def duration_seconds(self) -> float:
        """Duration of the time window in seconds."""
        return (self.end_time - self.start_time) / np.timedelta64(1, "s")


# ============================================================
# Time Window Creation
# ============================================================


def extract_file_timestamp(filepath: str) -> np.datetime64 | None:
    """
    Extract timestamp from a polar filename.

    Expects format: YYYYMMDDHHmmss*.pol*
    """
    name = os.path.basename(filepath)
    if len(name) < 14:
        return None
    timestamp_str = name[:14]
    if not timestamp_str.isdigit():
        return None
    try:
        return np.datetime64(
            f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}"
            f"T{timestamp_str[8:10]}:{timestamp_str[10:12]}:{timestamp_str[12:14]}"
        )
    except ValueError:
        return None


def create_time_windows(
    files: list[str],
    window_config: TimeWindowConfig,
) -> list[tuple[np.datetime64, np.datetime64, list[int]]]:
    """
    Create overlapping time windows from a file list.

    Args:
        files: List of polar file paths (should be time-sorted)
        window_config: Time window configuration

    Returns:
        List of (start_time, end_time, file_indices) tuples

    Example: window=60s, overlap=50% (stride=30s)
      Window 0: [0s, 60s]   - files with timestamps in this range
      Window 1: [30s, 90s]  - overlaps with Window 0
      Window 2: [60s, 120s] - overlaps with Window 1
      ...
    """
    if not files:
        return []

    # Extract timestamps from all files
    timestamps = []
    valid_indices = []
    for i, f in enumerate(files):
        ts = extract_file_timestamp(f)
        if ts is not None:
            timestamps.append(ts)
            valid_indices.append(i)

    if not timestamps:
        return []

    timestamps = np.array(timestamps, dtype="datetime64[ns]")
    valid_indices = np.array(valid_indices)

    # Get time range
    t_min = timestamps.min()
    t_max = timestamps.max()

    window_ns = np.timedelta64(int(window_config.window_seconds * 1e9), "ns")
    stride_ns = np.timedelta64(int(window_config.stride_seconds * 1e9), "ns")

    windows = []
    window_start = t_min

    while window_start <= t_max:
        window_end = window_start + window_ns

        # Find files within this window
        mask = (timestamps >= window_start) & (timestamps < window_end)
        file_indices = valid_indices[mask].tolist()

        # Only include windows with enough frames
        if len(file_indices) >= window_config.min_frames_per_window:
            windows.append((window_start, window_end, file_indices))

        window_start = window_start + stride_ns

    return windows


# ============================================================
# Window Accumulator
# ============================================================


class WindowAccumulator:
    """
    Accumulate multiple frames onto a common UTM grid.

    Handles the projection and averaging of radar frames within a time window.
    """

    def __init__(
        self,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        grid_spacing: float,
        utm_zone: int,
        hemisphere: str,
        center_lat: float,
        center_lon: float,
    ):
        """
        Initialize accumulator with pre-computed grid.

        Args:
            x_edges: Grid x edges in meters (centered on reference point)
            y_edges: Grid y edges in meters
            grid_spacing: Grid cell size in meters
            utm_zone: UTM zone number
            hemisphere: 'north' or 'south'
            center_lat: Grid center latitude
            center_lon: Grid center longitude
        """
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.grid_spacing = grid_spacing
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere
        self.center_lat = center_lat
        self.center_lon = center_lon

        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1
        self.intensity_sum = np.zeros((n_y, n_x), dtype=np.float64)
        self.intensity_count = np.zeros((n_y, n_x), dtype=np.int32)

        self.timestamps: list[np.datetime64] = []
        self.headings: list[float] = []
        self.ship_speeds: list[float] = []
        self.wind_speeds: list[float] = []
        self.wind_directions: list[float] = []

    @property
    def n_frames(self) -> int:
        """Number of frames accumulated."""
        return len(self.timestamps)

    @property
    def n_x(self) -> int:
        """Number of x bins."""
        return len(self.x_edges) - 1

    @property
    def n_y(self) -> int:
        """Number of y bins."""
        return len(self.y_edges) - 1

    def add_projected(
        self,
        projected_intensity: np.ndarray,
        projected_count: np.ndarray,
        timestamp: np.datetime64,
        heading: float,
        ship_speed: float | None = None,
        wind_speed: float | None = None,
        wind_direction: float | None = None,
    ) -> None:
        """
        Add a pre-projected frame to the accumulator.

        Args:
            projected_intensity: 2D projected intensity sum
            projected_count: 2D count of points per cell
            timestamp: Frame timestamp
            heading: Mean ship heading during frame
            ship_speed: Ship speed (m/s) or None
            wind_speed: Wind speed (m/s) or None
            wind_direction: Wind direction (degrees) or None
        """
        self.intensity_sum += projected_intensity
        self.intensity_count += projected_count
        self.timestamps.append(timestamp)
        self.headings.append(heading)

        if ship_speed is not None:
            self.ship_speeds.append(ship_speed)
        if wind_speed is not None:
            self.wind_speeds.append(wind_speed)
        if wind_direction is not None:
            self.wind_directions.append(wind_direction)

    def finalize(self, window_index: int = 0) -> MergedImage:
        """
        Compute averaged intensity and return MergedImage.

        Args:
            window_index: Index of this window in the sequence

        Returns:
            MergedImage with averaged data and metadata
        """
        with np.errstate(invalid="ignore"):
            intensity = self.intensity_sum / self.intensity_count
        intensity[self.intensity_count == 0] = np.nan

        # Compute circular mean for heading
        headings_rad = np.deg2rad(self.headings)
        mean_sin = np.mean(np.sin(headings_rad))
        mean_cos = np.mean(np.cos(headings_rad))
        mean_heading = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360

        # Compute optional means
        mean_ship_speed = float(np.mean(self.ship_speeds)) if self.ship_speeds else None
        mean_wind_speed = float(np.mean(self.wind_speeds)) if self.wind_speeds else None
        mean_wind_dir = None
        if self.wind_directions:
            wd_rad = np.deg2rad(self.wind_directions)
            wd_sin = np.mean(np.sin(wd_rad))
            wd_cos = np.mean(np.cos(wd_rad))
            mean_wind_dir = np.rad2deg(np.arctan2(wd_sin, wd_cos)) % 360

        return MergedImage(
            intensity=intensity,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            n_frames=len(self.timestamps),
            utm_zone=self.utm_zone,
            hemisphere=self.hemisphere,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            grid_spacing=self.grid_spacing,
            mean_heading=float(mean_heading),
            mean_ship_speed=mean_ship_speed,
            mean_wind_speed=mean_wind_speed,
            mean_wind_direction=mean_wind_dir,
            window_index=window_index,
        )


# ============================================================
# Grid Computation
# ============================================================


def compute_common_grid(
    latitudes: list[np.ndarray],
    longitudes: list[np.ndarray],
    max_ranges: list[float],
    range_resolutions: list[float],
    padding: float = 1.1,
) -> dict:
    """
    Compute a common UTM grid that covers all frames.

    Args:
        latitudes: List of per-radial latitude arrays
        longitudes: List of per-radial longitude arrays
        max_ranges: Maximum ground range per frame in meters
        range_resolutions: Range resolution per frame in meters
        padding: Multiplier for max range to add margin

    Returns:
        Dictionary with grid parameters:
        - x_edges, y_edges: Grid edges in meters (centered)
        - grid_spacing: Cell size in meters
        - utm_zone, hemisphere: Coordinate system info
        - center_lat, center_lon: Grid center in degrees
        - transformer: pyproj Transformer from WGS84 to UTM
    """
    from pyproj import CRS, Transformer

    # Get reference position (center of all data)
    all_lats = np.concatenate(latitudes)
    all_lons = np.concatenate(longitudes)
    ref_lat = float(np.mean(all_lats))
    ref_lon = float(np.mean(all_lons))

    # Determine UTM zone
    utm_zone = int((ref_lon + 180) / 6) % 60 + 1
    hemisphere = "north" if ref_lat >= 0 else "south"

    utm_crs = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    # Transform all positions to UTM
    all_x, all_y = transformer.transform(all_lons, all_lats)

    # Grid spacing from average range resolution
    grid_spacing = float(np.mean(range_resolutions))

    # Grid extent: data extent + max radar range + padding
    max_range = float(np.max(max_ranges)) * padding

    x_min = all_x.min() - max_range
    x_max = all_x.max() + max_range
    y_min = all_y.min() - max_range
    y_max = all_y.max() + max_range

    # Create bin edges aligned to grid spacing
    n_x = int(np.ceil((x_max - x_min) / grid_spacing))
    n_y = int(np.ceil((y_max - y_min) / grid_spacing))

    x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
    y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)

    # Compute grid center
    x_center = (x_edges[0] + x_edges[-1]) / 2
    y_center = (y_edges[0] + y_edges[-1]) / 2

    # Convert center to lat/lon
    transformer_inv = Transformer.from_crs(utm_crs, crs_wgs84, always_xy=True)
    center_lon, center_lat = transformer_inv.transform(x_center, y_center)

    # Center the edges for output
    x_edges_centered = x_edges - x_center
    y_edges_centered = y_edges - y_center

    return {
        "x_edges": x_edges_centered,
        "y_edges": y_edges_centered,
        "x_edges_utm": x_edges,
        "y_edges_utm": y_edges,
        "grid_spacing": grid_spacing,
        "utm_zone": utm_zone,
        "hemisphere": hemisphere,
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "x_center_utm": x_center,
        "y_center_utm": y_center,
        "transformer": transformer,
        "n_x": n_x,
        "n_y": n_y,
    }


def project_frame_to_common_grid(
    intensity: np.ndarray,
    theta: np.ndarray,
    ground_range: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings: np.ndarray,
    grid_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a single frame onto a common UTM grid.

    Args:
        intensity: 2D intensity array (n_bearings, n_distances)
        theta: Beam angles relative to ship (degrees)
        ground_range: Ground range (meters) for each distance bin
        latitudes: Per-radial latitudes
        longitudes: Per-radial longitudes
        headings: Per-radial ship headings (degrees)
        grid_params: Common grid parameters from compute_common_grid()

    Returns:
        Tuple of (intensity_sum, intensity_count) arrays for this frame
    """
    transformer = grid_params["transformer"]
    x_edges_utm = grid_params["x_edges_utm"]
    y_edges_utm = grid_params["y_edges_utm"]
    grid_spacing = grid_params["grid_spacing"]
    n_x = grid_params["n_x"]
    n_y = grid_params["n_y"]

    # Convert ship positions to UTM
    ship_x, ship_y = transformer.transform(longitudes, latitudes)

    # Initialize accumulation arrays
    frame_sum = np.zeros((n_y, n_x), dtype=np.float64)
    frame_count = np.zeros((n_y, n_x), dtype=np.int32)

    # Compute earth bearing for each radial
    earth_bearing_rad = np.deg2rad((theta + headings) % 360)
    sin_bearing = np.sin(earth_bearing_rad)
    cos_bearing = np.cos(earth_bearing_rad)

    # Compute x, y for all points (n_bearings, n_distances)
    x_coords = np.outer(sin_bearing, ground_range) + ship_x[:, np.newaxis]
    y_coords = np.outer(cos_bearing, ground_range) + ship_y[:, np.newaxis]

    # Convert to grid indices
    x_origin = x_edges_utm[0]
    y_origin = y_edges_utm[0]
    inv_spacing = 1.0 / grid_spacing

    x_idx = ((x_coords - x_origin) * inv_spacing).astype(np.int32)
    y_idx = ((y_coords - y_origin) * inv_spacing).astype(np.int32)

    # Flatten arrays
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    values_flat = intensity.ravel()

    # Filter valid indices
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~np.isnan(values_flat)

    if np.sum(valid) > 0:
        linear_idx = y_flat[valid] * n_x + x_flat[valid]
        grid_size = n_x * n_y

        batch_sum = np.bincount(linear_idx, weights=values_flat[valid], minlength=grid_size)
        batch_count = np.bincount(linear_idx, minlength=grid_size)

        frame_sum.ravel()[:] = batch_sum
        frame_count.ravel()[:] = batch_count

    return frame_sum, frame_count


# ============================================================
# Grid Remapping
# ============================================================


def _remap_to_common_grid(
    intensity: np.ndarray,
    count: np.ndarray | None,
    src_x_edges: np.ndarray,
    src_y_edges: np.ndarray,
    dst_x_edges: np.ndarray,
    dst_y_edges: np.ndarray,
    dst_n_x: int,
    dst_n_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remap projected intensity from source grid to destination grid.

    Args:
        intensity: Source intensity (averaged) array
        count: Source count array (may be None)
        src_x_edges: Source grid x edges (in absolute UTM coordinates)
        src_y_edges: Source grid y edges (in absolute UTM coordinates)
        dst_x_edges: Destination grid x edges (in absolute UTM coordinates)
        dst_y_edges: Destination grid y edges (in absolute UTM coordinates)
        dst_n_x: Destination grid x dimension
        dst_n_y: Destination grid y dimension

    Returns:
        Tuple of (intensity_sum, count) arrays in destination grid
    """
    # Compute source grid centers
    src_x_centers = (src_x_edges[:-1] + src_x_edges[1:]) / 2
    src_y_centers = (src_y_edges[:-1] + src_y_edges[1:]) / 2

    # Get destination grid spacing
    dst_dx = dst_x_edges[1] - dst_x_edges[0]
    dst_dy = dst_y_edges[1] - dst_y_edges[0]

    # Create meshgrid of source coordinates
    src_xx, src_yy = np.meshgrid(src_x_centers, src_y_centers, indexing="xy")

    # Compute destination indices for all source cells
    dst_ix = ((src_xx - dst_x_edges[0]) / dst_dx).astype(np.int32)
    dst_iy = ((src_yy - dst_y_edges[0]) / dst_dy).astype(np.int32)

    # Find valid cells (within destination grid and not NaN)
    valid = (
        (dst_ix >= 0)
        & (dst_ix < dst_n_x)
        & (dst_iy >= 0)
        & (dst_iy < dst_n_y)
        & ~np.isnan(intensity)
    )

    if not np.any(valid):
        return (
            np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
            np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
        )

    # Get valid values
    valid_ix = dst_ix[valid]
    valid_iy = dst_iy[valid]
    valid_intensity = intensity[valid]

    if count is not None:
        valid_count = count[valid]
    else:
        valid_count = np.ones(np.sum(valid), dtype=np.int32)

    # Use linear indices for bincount
    linear_idx = valid_iy * dst_n_x + valid_ix
    grid_size = dst_n_x * dst_n_y

    # Accumulate weighted intensity and counts
    dst_sum = np.bincount(
        linear_idx, weights=valid_intensity * valid_count, minlength=grid_size
    ).reshape((dst_n_y, dst_n_x))

    dst_count = np.bincount(linear_idx, weights=valid_count, minlength=grid_size).reshape(
        (dst_n_y, dst_n_x)
    )

    return dst_sum.astype(np.float64), dst_count.astype(np.int32)


# ============================================================
# Task Handlers for Priority Executor
# ============================================================


def _do_project_frame(task):  # -> Result:
    """
    Project a single frame onto a common grid.

    Task data: (frame_data, grid_params)
    """
    import resource
    from wamos_tpw.priority_executor import Result, read_shared_array

    frame_data, grid_params_serialized = task.data

    t0 = time.perf_counter()

    # Read arrays from shared memory
    intensity = read_shared_array(*frame_data.intensity_shm) if frame_data.intensity_shm else None
    theta = read_shared_array(*frame_data.theta_shm) if frame_data.theta_shm else None
    ground_range = (
        read_shared_array(*frame_data.ground_range_shm) if frame_data.ground_range_shm else None
    )

    if intensity is None or theta is None or ground_range is None:
        return Result(
            task_type="project_frame",
            task_id=task.task_id,
            data={"success": False, "error": "Missing arrays"},
            shm_to_release=[],
        )

    # Reconstruct grid params with transformer
    from pyproj import CRS, Transformer

    utm_crs = CRS.from_proj4(
        f"+proj=utm +zone={grid_params_serialized['utm_zone']} "
        f"+{grid_params_serialized['hemisphere']} +datum=WGS84"
    )
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    grid_params = {**grid_params_serialized, "transformer": transformer}

    # We need latitudes, longitudes, headings from interpolation
    # These should be passed in frame_data
    latitudes = frame_data.latitudes
    longitudes = frame_data.longitudes
    headings = frame_data.headings

    # Project frame
    frame_sum, frame_count = project_frame_to_common_grid(
        intensity=intensity,
        theta=theta,
        ground_range=ground_range,
        latitudes=latitudes,
        longitudes=longitudes,
        headings=headings,
        grid_params=grid_params,
    )

    elapsed = time.perf_counter() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return Result(
        task_type="project_frame",
        task_id=task.task_id,
        data={
            "success": True,
            "file_index": frame_data.file_index,
            "frame_index": frame_data.frame_index,
            "timestamp": frame_data.timestamp,
            "frame_sum": frame_sum,
            "frame_count": frame_count,
            "heading": float(np.mean(headings)),
            "ship_speed": frame_data.ship_speed,
            "wind_speed": frame_data.wind_speed,
            "wind_direction": frame_data.wind_direction,
            "elapsed": elapsed,
            "peak_rss": peak_rss,
        },
        shm_to_release=[],
    )


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

                    frame_sum, frame_count = _remap_to_common_grid(
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
            return accumulator.finalize(window_index=window_idx)
        return None

    def iter_merged(self) -> Iterator[MergedImage]:
        """
        Yield merged images as windows complete.

        Uses streaming architecture: windows are merged and yielded as soon as
        all their constituent files have been interpolated, rather than waiting
        for all files to complete first.
        """
        from collections import defaultdict

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

        # Submit file processing tasks
        pending_files = len(file_indices_to_process)
        pending_interp = 0

        for file_idx in file_indices_to_process:
            filepath = self._files[file_idx]
            task_data = (filepath, file_idx, config_dict, self._qTiming)
            executor.submit(Priority.LOW, "process_file", task_data)

        # Triplet collection
        triplet_collector = TripletCollector(total_files=len(self._files))

        # Track files that have completed interpolation (all frames done)
        files_with_all_frames_interpolated: set[int] = set()
        # Track how many frames each file has and how many are interpolated
        file_frame_counts: dict[int, int] = {}
        file_frames_interpolated: dict[int, int] = defaultdict(int)

        # Progress bars
        # Interpolation count approximates file count (one frame per file typically)
        pbar_files = tqdm(
            total=len(file_indices_to_process),
            desc="Loading files",
            unit="file",
            disable=not self._qProgress,
        )
        pbar_interp = tqdm(
            total=len(file_indices_to_process),
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

        # Main processing loop
        while pending_files > 0 or pending_interp > 0:
            result = executor.get_result(timeout=0.1)
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
                t_phase = time.perf_counter()
                pending_files -= 1
                pbar_files.update(1)
                data = result.data
                file_idx = data["file_index"]

                # Track frame count for this file
                file_frame_counts[file_idx] = len(data["frames"])

                # Add frames to triplet collector
                for frame_data in data["frames"]:
                    triplet_collector.add(frame_data)

                    if frame_data.theta_shm:
                        shm_manager.register(frame_data.theta_shm[0], refcount=1)
                    if frame_data.ground_range_shm:
                        shm_manager.register(frame_data.ground_range_shm[0], refcount=1)
                    if frame_data.intensity_shm:
                        shm_manager.register(frame_data.intensity_shm[0], refcount=1)

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

        # Clean up
        shm_manager.cleanup()
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


# ============================================================
# Output Functions
# ============================================================


def write_merged_netcdf(merged: MergedImage, output_dir: str) -> str:
    """
    Write merged image to NetCDF file with CF-1.8 conventions.

    Args:
        merged: MergedImage to write
        output_dir: Output directory

    Returns:
        Path to created file
    """
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not installed, skipping NetCDF output")
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from time range
    start_str = (
        np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = np.datetime_as_string(merged.end_time, unit="s").replace(":", "-").replace("T", "_")
    filename = f"merged_{start_str}_to_{end_str}.nc"
    filepath = output_dir / filename

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "intensity": (
                ["y", "x"],
                merged.intensity.astype(np.float32),
                {
                    "long_name": "Merged radar intensity",
                    "units": "counts",
                    "coordinates": "x y",
                },
            ),
        },
        coords={
            "x": (
                ["x"],
                merged.x_centers,
                {
                    "long_name": "Distance east from center",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                ["y"],
                merged.y_centers,
                {
                    "long_name": "Distance north from center",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "time_start": merged.start_time,
            "time_end": merged.end_time,
        },
        attrs={
            "title": "WAMOS merged radar image",
            "institution": "WAMOS TPW",
            "source": "wamos files-pipeline",
            "history": f"Created {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            # Grid metadata
            "grid_spacing_m": merged.grid_spacing,
            "utm_zone": merged.utm_zone,
            "hemisphere": merged.hemisphere,
            "center_latitude": merged.center_lat,
            "center_longitude": merged.center_lon,
            "crs": f"EPSG:326{merged.utm_zone:02d}"
            if merged.hemisphere == "north"
            else f"EPSG:327{merged.utm_zone:02d}",
            # Window metadata
            "n_frames": merged.n_frames,
            "window_index": merged.window_index,
            "mean_ship_heading_deg": merged.mean_heading,
        },
    )

    # Add optional metadata
    if merged.mean_ship_speed is not None:
        ds.attrs["mean_ship_speed_m_s"] = merged.mean_ship_speed
    if merged.mean_wind_speed is not None:
        ds.attrs["mean_wind_speed_m_s"] = merged.mean_wind_speed
    if merged.mean_wind_direction is not None:
        ds.attrs["mean_wind_direction_deg"] = merged.mean_wind_direction

    # Write with compression
    encoding = {"intensity": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    ds.to_netcdf(filepath, encoding=encoding)

    logger.debug("Wrote merged image to %s", filepath)
    return str(filepath)


def write_merged_png(
    merged: MergedImage,
    output_dir: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Write merged image to PNG file.

    Args:
        merged: MergedImage to write
        output_dir: Output directory
        cmap: Colormap name
        vmin: Minimum intensity value (auto if None)
        vmax: Maximum intensity value (auto if None)

    Returns:
        Path to created file
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    start_str = (
        np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = np.datetime_as_string(merged.end_time, unit="s").replace(":", "-").replace("T", "_")
    filename = f"merged_{start_str}_to_{end_str}.png"
    filepath = output_dir / filename

    # Auto-scale
    intensity = merged.intensity
    if vmin is None:
        vmin = float(np.nanpercentile(intensity, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(intensity, 98))

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.pcolormesh(
        merged.x_edges,
        merged.y_edges,
        intensity,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    ax.set_xlabel("Distance East (m)")
    ax.set_ylabel("Distance North (m)")
    ax.set_aspect("equal")

    fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

    # Title
    start_time = np.datetime_as_string(merged.start_time, unit="s")
    end_time = np.datetime_as_string(merged.end_time, unit="s")
    ax.set_title(
        f"Merged Image: {merged.n_frames} frames\n"
        f"{start_time} to {end_time}\n"
        f"Center: {abs(merged.center_lat):.4f}°{'N' if merged.center_lat >= 0 else 'S'}, "
        f"{abs(merged.center_lon):.4f}°{'E' if merged.center_lon >= 0 else 'W'}"
    )

    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Wrote merged image to %s", filepath)
    return str(filepath)


def write_mp4_movie(
    merged_images: list[MergedImage],
    output_path: str,
    fps: float = 2.0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (10, 8),
    range_rings: bool = True,
) -> str:
    """
    Generate an MP4 movie from merged images.

    Args:
        merged_images: List of MergedImage objects (in time order)
        output_path: Output MP4 file path
        fps: Frames per second (default 2.0)
        cmap: Colormap name
        vmin: Minimum intensity value (auto if None)
        vmax: Maximum intensity value (auto if None)
        dpi: Output resolution
        figsize: Figure size in inches
        range_rings: Draw range rings on frames

    Returns:
        Path to created file
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    if not merged_images:
        logger.warning("No merged images for movie")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute global intensity range
    if vmin is None or vmax is None:
        all_valid = []
        for merged in merged_images:
            valid_data = merged.intensity[~np.isnan(merged.intensity)]
            if len(valid_data) > 0:
                all_valid.extend(valid_data.ravel())
        if all_valid:
            if vmin is None:
                vmin = float(np.percentile(all_valid, 2))
            if vmax is None:
                vmax = float(np.percentile(all_valid, 98))
        else:
            vmin, vmax = 0, 1

    # Find global data bounds for consistent framing
    global_bounds = {"xmin": np.inf, "xmax": -np.inf, "ymin": np.inf, "ymax": -np.inf}
    for merged in merged_images:
        valid_mask = ~np.isnan(merged.intensity)
        valid_rows = np.any(valid_mask, axis=1)
        valid_cols = np.any(valid_mask, axis=0)
        if np.any(valid_rows) and np.any(valid_cols):
            row_min, row_max = np.where(valid_rows)[0][[0, -1]]
            col_min, col_max = np.where(valid_cols)[0][[0, -1]]
            global_bounds["xmin"] = min(global_bounds["xmin"], merged.x_edges[col_min])
            global_bounds["xmax"] = max(global_bounds["xmax"], merged.x_edges[col_max + 1])
            global_bounds["ymin"] = min(global_bounds["ymin"], merged.y_edges[row_min])
            global_bounds["ymax"] = max(global_bounds["ymax"], merged.y_edges[row_max + 1])

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.tight_layout()

    # Create writer
    writer = FFMpegWriter(fps=fps, metadata={"title": "WAMOS Radar Movie"})

    logger.info("Generating MP4 movie with %d frames at %.1f fps", len(merged_images), fps)

    with writer.saving(fig, str(output_path), dpi=dpi):
        for i, merged in enumerate(merged_images):
            ax.clear()

            # Find valid data bounds
            valid_mask = ~np.isnan(merged.intensity)
            valid_rows = np.any(valid_mask, axis=1)
            valid_cols = np.any(valid_mask, axis=0)

            if np.any(valid_rows) and np.any(valid_cols):
                row_min, row_max = np.where(valid_rows)[0][[0, -1]]
                col_min, col_max = np.where(valid_cols)[0][[0, -1]]
                cropped = merged.intensity[row_min : row_max + 1, col_min : col_max + 1]
                extent = [
                    merged.x_edges[col_min],
                    merged.x_edges[col_max + 1],
                    merged.y_edges[row_min],
                    merged.y_edges[row_max + 1],
                ]
            else:
                cropped = merged.intensity
                extent = [
                    merged.x_edges[0],
                    merged.x_edges[-1],
                    merged.y_edges[0],
                    merged.y_edges[-1],
                ]

            ax.imshow(
                cropped,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="lower",
                aspect="equal",
            )

            if range_rings:
                _draw_range_rings(ax, extent)

            # Use global bounds for consistent framing
            ax.set_xlim(global_bounds["xmin"], global_bounds["xmax"])
            ax.set_ylim(global_bounds["ymin"], global_bounds["ymax"])

            ax.set_xlabel("Distance East (m)")
            ax.set_ylabel("Distance North (m)")

            start_time = np.datetime_as_string(merged.start_time, unit="s")
            end_time = np.datetime_as_string(merged.end_time, unit="s")
            ax.set_title(
                f"Frame {i + 1}/{len(merged_images)}: {merged.n_frames} frames\n"
                f"{start_time} to {end_time}"
            )

            writer.grab_frame()

    plt.close(fig)
    logger.info("Wrote MP4 movie to %s", output_path)
    return str(output_path)


def write_geotiff(
    merged: MergedImage,
    output_dir: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Write merged image as a georeferenced GeoTIFF file.

    Args:
        merged: MergedImage to write
        output_dir: Output directory
        cmap: Colormap name for RGB conversion
        vmin: Minimum intensity value (auto if None)
        vmax: Maximum intensity value (auto if None)

    Returns:
        Path to created file
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
    except ImportError:
        logger.warning("rasterio not installed, skipping GeoTIFF output")
        return ""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    start_str = (
        np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
    )
    end_str = np.datetime_as_string(merged.end_time, unit="s").replace(":", "-").replace("T", "_")
    filename = f"merged_{start_str}_to_{end_str}.tif"
    filepath = output_dir / filename

    # Get intensity data and compute scaling
    intensity = merged.intensity.copy()
    if vmin is None:
        vmin = float(np.nanpercentile(intensity, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(intensity, 98))

    # Normalize to 0-1 range
    normalized = (intensity - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap to get RGBA
    colormap = plt.get_cmap(cmap)
    rgba = colormap(normalized)  # Shape: (h, w, 4)

    # Convert to uint8
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Handle NaN values - make them transparent
    mask = np.isnan(intensity)
    alpha = np.where(mask, 0, 255).astype(np.uint8)

    # Stack RGB + Alpha
    rgba_uint8 = np.dstack([rgb, alpha])

    # Compute bounds in UTM coordinates
    # The grid is centered, so we need to convert to absolute UTM
    from pyproj import CRS as ProjCRS
    from pyproj import Transformer

    utm_crs = ProjCRS.from_proj4(
        f"+proj=utm +zone={merged.utm_zone} +{merged.hemisphere} +datum=WGS84"
    )
    crs_wgs84 = ProjCRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    # Get center in UTM
    center_x, center_y = transformer.transform(merged.center_lon, merged.center_lat)

    # Compute absolute bounds
    x_min = center_x + merged.x_edges[0]
    x_max = center_x + merged.x_edges[-1]
    y_min = center_y + merged.y_edges[0]
    y_max = center_y + merged.y_edges[-1]

    # Create transform (note: rasterio uses top-left origin, so y is flipped)
    height, width = intensity.shape
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Determine EPSG code
    if merged.hemisphere == "north":
        epsg = 32600 + merged.utm_zone
    else:
        epsg = 32700 + merged.utm_zone

    # Write GeoTIFF
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,  # RGBA
        dtype=np.uint8,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        compress="lzw",
    ) as dst:
        # Write each band (rasterio expects bands first)
        for i in range(4):
            # Flip vertically since rasterio uses top-left origin
            dst.write(np.flipud(rgba_uint8[:, :, i]), i + 1)

        # Add metadata
        dst.update_tags(
            title="WAMOS Merged Radar Image",
            start_time=str(merged.start_time),
            end_time=str(merged.end_time),
            n_frames=str(merged.n_frames),
            center_lat=str(merged.center_lat),
            center_lon=str(merged.center_lon),
        )

    logger.debug("Wrote GeoTIFF to %s", filepath)
    return str(filepath)


def write_kml(
    merged_images: list[MergedImage],
    output_path: str,
    image_dir: str | None = None,
    image_format: str = "png",
) -> str:
    """
    Write a KML file with ground overlays for merged images.

    If image_dir is provided, generates PNG images for each frame.
    The KML file references these images as ground overlays with proper
    geographic positioning.

    Args:
        merged_images: List of MergedImage objects
        output_path: Output KML file path
        image_dir: Directory for overlay images (if None, uses output_path directory)
        image_format: Image format for overlays ("png" or "tiff")

    Returns:
        Path to created KML file
    """
    from xml.etree.ElementTree import Element, SubElement, ElementTree  # nosec B405

    if not merged_images:
        logger.warning("No merged images for KML")
        return ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        image_dir = output_path.parent / "images"
    else:
        image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Compute global intensity range for consistent coloring
    all_valid = []
    for merged in merged_images:
        valid_data = merged.intensity[~np.isnan(merged.intensity)]
        if len(valid_data) > 0:
            all_valid.extend(valid_data.ravel())
    if all_valid:
        vmin = float(np.percentile(all_valid, 2))
        vmax = float(np.percentile(all_valid, 98))
    else:
        vmin, vmax = 0, 1

    # Create KML structure
    kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(kml, "Document")

    # Add document name and description
    name = SubElement(document, "name")
    name.text = "WAMOS Radar Images"

    description = SubElement(document, "description")
    description.text = f"Merged radar images: {len(merged_images)} frames"

    # Create folder for overlays
    folder = SubElement(document, "Folder")
    folder_name = SubElement(folder, "name")
    folder_name.text = "Radar Overlays"

    # Generate images and add overlays
    for i, merged in enumerate(merged_images):
        # Generate image filename
        start_str = (
            np.datetime_as_string(merged.start_time, unit="s").replace(":", "-").replace("T", "_")
        )
        image_filename = f"overlay_{i:04d}_{start_str}.png"
        image_path = image_dir / image_filename

        # Write overlay image (PNG with transparency)
        _write_overlay_png(merged, image_path, vmin=vmin, vmax=vmax)

        # Compute geographic bounds
        bounds = _compute_latlon_bounds(merged)

        # Create GroundOverlay element
        overlay = SubElement(folder, "GroundOverlay")

        overlay_name = SubElement(overlay, "name")
        overlay_name.text = f"Frame {i + 1}: {start_str}"

        # Time span (for time-enabled KML viewers)
        timespan = SubElement(overlay, "TimeSpan")
        begin = SubElement(timespan, "begin")
        begin.text = np.datetime_as_string(merged.start_time, unit="s")
        end = SubElement(timespan, "end")
        end.text = np.datetime_as_string(merged.end_time, unit="s")

        # Icon (image reference)
        icon = SubElement(overlay, "Icon")
        href = SubElement(icon, "href")
        # Use relative path
        href.text = f"images/{image_filename}"

        # LatLonBox for positioning
        latlonbox = SubElement(overlay, "LatLonBox")
        north = SubElement(latlonbox, "north")
        north.text = str(bounds["north"])
        south = SubElement(latlonbox, "south")
        south.text = str(bounds["south"])
        east = SubElement(latlonbox, "east")
        east.text = str(bounds["east"])
        west = SubElement(latlonbox, "west")
        west.text = str(bounds["west"])

    # Write KML file
    tree = ElementTree(kml)
    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    logger.info("Wrote KML file to %s with %d overlays", output_path, len(merged_images))
    return str(output_path)


def write_kmz(
    merged_images: list[MergedImage],
    output_path: str,
) -> str:
    """
    Write a KMZ file (compressed KML with embedded images).

    A KMZ file is a ZIP archive containing:
    - doc.kml: The main KML file
    - images/: Directory with overlay PNG images

    This creates a self-contained package that can be opened directly
    in Google Earth without external file dependencies.

    Args:
        merged_images: List of MergedImage objects
        output_path: Output KMZ file path

    Returns:
        Path to created KMZ file
    """
    import shutil
    import tempfile
    import zipfile

    if not merged_images:
        logger.warning("No merged images for KMZ")
        return ""

    output_path = Path(output_path)
    if not output_path.suffix.lower() == ".kmz":
        output_path = output_path.with_suffix(".kmz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for KML and images
    temp_dir = Path(tempfile.mkdtemp(prefix="wamos_kmz_"))
    try:
        kml_path = temp_dir / "doc.kml"
        image_dir = temp_dir / "images"

        # Generate KML and images using existing function
        write_kml(merged_images, str(kml_path), image_dir=str(image_dir))

        # Package into KMZ (ZIP file)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as kmz:
            # Add doc.kml at root
            kmz.write(kml_path, "doc.kml")

            # Add all images in images/ folder
            for image_file in image_dir.iterdir():
                if image_file.is_file():
                    kmz.write(image_file, f"images/{image_file.name}")

        logger.info(
            "Wrote KMZ file to %s (%d overlays, %.1f MB)",
            output_path,
            len(merged_images),
            output_path.stat().st_size / (1024 * 1024),
        )
        return str(output_path)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_overlay_png(
    merged: MergedImage,
    output_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Write a PNG image suitable for KML overlay (with transparency).

    Args:
        merged: MergedImage to write
        output_path: Output PNG file path
        cmap: Colormap name
        vmin: Minimum intensity value
        vmax: Maximum intensity value
    """
    import matplotlib.pyplot as plt

    intensity = merged.intensity.copy()

    if vmin is None:
        vmin = float(np.nanpercentile(intensity, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(intensity, 98))

    # Normalize to 0-1 range
    normalized = (intensity - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    rgba = colormap(normalized)

    # Set NaN pixels to transparent
    mask = np.isnan(intensity)
    rgba[mask, 3] = 0

    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Flip vertically for correct orientation (image origin is top-left)
    rgba_uint8 = np.flipud(rgba_uint8)

    # Write PNG
    try:
        from PIL import Image

        img = Image.fromarray(rgba_uint8, mode="RGBA")
        img.save(output_path)
    except ImportError:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgba_uint8, origin="upper")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)


def _compute_latlon_bounds(merged: MergedImage) -> dict:
    """
    Compute lat/lon bounds for a merged image.

    Args:
        merged: MergedImage with UTM grid

    Returns:
        Dictionary with north, south, east, west bounds in degrees
    """
    from pyproj import CRS, Transformer

    # Create transformers
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={merged.utm_zone} +{merged.hemisphere} +datum=WGS84")
    crs_wgs84 = CRS.from_epsg(4326)
    transformer_to_ll = Transformer.from_crs(utm_crs, crs_wgs84, always_xy=True)
    transformer_to_utm = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    # Get center in UTM
    center_x, center_y = transformer_to_utm.transform(merged.center_lon, merged.center_lat)

    # Compute corners in UTM
    x_min = center_x + merged.x_edges[0]
    x_max = center_x + merged.x_edges[-1]
    y_min = center_y + merged.y_edges[0]
    y_max = center_y + merged.y_edges[-1]

    # Transform corners to lat/lon
    corners_x = [x_min, x_max, x_min, x_max]
    corners_y = [y_min, y_min, y_max, y_max]
    lons, lats = transformer_to_ll.transform(corners_x, corners_y)

    return {
        "north": max(lats),
        "south": min(lats),
        "east": max(lons),
        "west": min(lons),
    }


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


def _draw_range_rings(ax, extent: list, ring_interval: float = 1000.0) -> None:
    """
    Draw range rings centered at origin.

    Args:
        ax: Matplotlib axes
        extent: [xmin, xmax, ymin, ymax] of the plot
        ring_interval: Distance between rings in meters (default 1000m)
    """
    from matplotlib.patches import Circle

    xmin, xmax, ymin, ymax = extent

    # Compute max range needed to cover the plot
    max_range = max(
        abs(xmin),
        abs(xmax),
        abs(ymin),
        abs(ymax),
        np.sqrt(xmin**2 + ymin**2),
        np.sqrt(xmax**2 + ymin**2),
        np.sqrt(xmin**2 + ymax**2),
        np.sqrt(xmax**2 + ymax**2),
    )

    # Draw rings at regular intervals
    n_rings = int(max_range / ring_interval) + 1
    for i in range(1, n_rings + 1):
        radius = i * ring_interval
        circle = Circle(
            (0, 0),
            radius,
            fill=False,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.5,
            linestyle="--",
        )
        ax.add_patch(circle)

        # Add range label at top of circle (if visible)
        if -radius <= xmax and radius >= xmin and radius <= ymax:
            ax.text(
                0,
                radius,
                f"{radius / 1000:.0f}km",
                ha="center",
                va="bottom",
                fontsize=7,
                color="white",
                alpha=0.7,
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


def _show_single_image(merged: MergedImage) -> None:
    """
    Display a single merged image in a non-blocking window.

    Used to show the first image while processing continues.
    """
    import matplotlib.pyplot as plt

    # Find bounds of actual data (non-NaN values)
    valid_mask = ~np.isnan(merged.intensity)
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    if np.any(valid_rows) and np.any(valid_cols):
        row_min, row_max = np.where(valid_rows)[0][[0, -1]]
        col_min, col_max = np.where(valid_cols)[0][[0, -1]]

        cropped = merged.intensity[row_min : row_max + 1, col_min : col_max + 1]
        extent = [
            merged.x_edges[col_min],
            merged.x_edges[col_max + 1],
            merged.y_edges[row_min],
            merged.y_edges[row_max + 1],
        ]
    else:
        cropped = merged.intensity
        extent = [merged.x_edges[0], merged.x_edges[-1], merged.y_edges[0], merged.y_edges[-1]]

    # Compute intensity range
    valid_data = cropped[~np.isnan(cropped)]
    if len(valid_data) > 0:
        vmin = float(np.percentile(valid_data, 2))
        vmax = float(np.percentile(valid_data, 98))
    else:
        vmin, vmax = 0, 1

    # Enable interactive mode
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        cropped,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        aspect="equal",
    )

    # Add range rings
    _draw_range_rings(ax, extent)

    ax.set_xlabel("Distance East (m)")
    ax.set_ylabel("Distance North (m)")

    start_time = np.datetime_as_string(merged.start_time, unit="s")
    end_time = np.datetime_as_string(merged.end_time, unit="s")

    ax.set_title(f"First Merged Image: {merged.n_frames} frames\n{start_time} to {end_time}")

    fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

    # Force draw and show non-blocking
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    plt.pause(0.5)  # Give time to render


def _show_merged_viewer(merged_images: list[MergedImage], interval_ms: int = 500) -> None:
    """
    Show an interactive viewer for merged images.

    Simple matplotlib-based viewer with navigation between windows.
    Includes play/stop button for automatic playback.

    Args:
        merged_images: List of merged images to display
        interval_ms: Playback interval in milliseconds (default 500ms = 2 fps)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button

    # Turn off interactive mode for blocking display
    plt.ioff()
    # Close any existing figures from single image preview
    plt.close("all")

    if not merged_images:
        logger.warning("No merged images to display")
        return

    # Compute global intensity range across all images
    all_valid = []
    for merged in merged_images:
        valid_data = merged.intensity[~np.isnan(merged.intensity)]
        if len(valid_data) > 0:
            all_valid.extend(valid_data.ravel())

    if not all_valid:
        logger.warning("All merged images have no valid data")
        # Still show the empty images for debugging
        vmin, vmax = 0, 1
    else:
        vmin = float(np.percentile(all_valid, 2))
        vmax = float(np.percentile(all_valid, 98))

    # Compute max speeds for consistent polar plot scaling
    max_ship_speed = 0.0
    max_wind_speed = 0.0
    for merged in merged_images:
        if merged.mean_ship_speed is not None:
            max_ship_speed = max(max_ship_speed, merged.mean_ship_speed)
        if merged.mean_wind_speed is not None:
            max_wind_speed = max(max_wind_speed, merged.mean_wind_speed)

    # Use reasonable defaults if no data
    if max_ship_speed == 0:
        max_ship_speed = 10.0  # m/s
    if max_wind_speed == 0:
        max_wind_speed = 20.0  # m/s

    # Create figure with explicit axes positioning
    # Use square figure for equal aspect ratio
    fig = plt.figure(figsize=(12, 11))

    # Main axes: make it square in figure coordinates
    # Height: 0.78 of figure (11 inches) = 8.58 inches
    # Width: 8.58/12 = 0.715 of figure width
    ax = fig.add_axes((0.08, 0.12, 0.715, 0.78))

    # Colorbar axes (fixed position, won't affect main axes)
    cax = fig.add_axes((0.82, 0.12, 0.03, 0.78))

    # Mutable containers for inset axes (recreated on each update since ax.clear() removes them)
    inset_axes = {"ship": None, "wind": None}

    current_idx = [0]  # Mutable container for callback
    is_playing = [False]  # Mutable container for play state
    animation = [None]  # Mutable container for animation object

    def create_polar_insets():
        """Create polar inset axes inside the main plot."""
        # Position: [left, bottom, width, height] relative to parent axes (0-1)
        # 70% of original size (0.14 vs 0.20), positioned closer to corners
        ax_ship = ax.inset_axes((0.85, 0.85, 0.14, 0.14), projection="polar")
        ax_wind = ax.inset_axes((0.85, 0.01, 0.14, 0.14), projection="polar")
        inset_axes["ship"] = ax_ship
        inset_axes["wind"] = ax_wind
        return ax_ship, ax_wind

    def update_polar_plots(merged):
        """Update the ship and wind polar inset plots."""
        ax_ship = inset_axes["ship"]
        ax_wind = inset_axes["wind"]

        # Ship plot (NE corner of main plot, label in NW corner of polar plot)
        ax_ship.clear()
        ax_ship.set_facecolor((1, 1, 1, 0.85))  # White with transparency
        ax_ship.set_theta_zero_location("N")
        ax_ship.set_theta_direction(-1)
        ax_ship.set_yticklabels([])
        ax_ship.set_xticklabels([])  # No N/E/S/W labels
        ax_ship.set_ylim(0, max_ship_speed * 1.1)

        if merged.mean_ship_speed is not None and merged.mean_ship_speed > 0:
            heading_rad = np.radians(merged.mean_heading)
            ax_ship.annotate(
                "",
                xy=(heading_rad, merged.mean_ship_speed),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": "blue", "lw": 2},
            )
            label_text = f"Ship\n{merged.mean_ship_speed:.1f} m/s"
        else:
            label_text = "Ship\nN/A"
        # Label in NW corner of polar plot
        ax_ship.text(
            0.02,
            0.98,
            label_text,
            transform=ax_ship.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            color="blue",
        )

        # Wind plot (SE corner of main plot, label in SW corner of polar plot)
        ax_wind.clear()
        ax_wind.set_facecolor((1, 1, 1, 0.85))  # White with transparency
        ax_wind.set_theta_zero_location("N")
        ax_wind.set_theta_direction(-1)
        ax_wind.set_yticklabels([])
        ax_wind.set_xticklabels([])  # No N/E/S/W labels
        ax_wind.set_ylim(0, max_wind_speed * 1.1)

        if merged.mean_wind_speed is not None and merged.mean_wind_speed > 0:
            # Wind direction is where wind comes FROM, arrow points in that direction
            wind_dir_rad = np.radians(merged.mean_wind_direction or 0)
            ax_wind.annotate(
                "",
                xy=(wind_dir_rad, merged.mean_wind_speed),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": "green", "lw": 2},
            )
            label_text = f"Wind\n{merged.mean_wind_speed:.1f} m/s"
        else:
            label_text = "Wind\nN/A"
        # Label in SW corner of polar plot
        ax_wind.text(
            0.02,
            0.02,
            label_text,
            transform=ax_wind.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            color="green",
        )

    def update_plot():
        merged = merged_images[current_idx[0]]
        ax.clear()

        # Recreate inset axes (cleared by ax.clear())
        create_polar_insets()

        # Find bounds of actual data (non-NaN values)
        valid_mask = ~np.isnan(merged.intensity)
        valid_rows = np.any(valid_mask, axis=1)
        valid_cols = np.any(valid_mask, axis=0)

        if np.any(valid_rows) and np.any(valid_cols):
            row_min, row_max = np.where(valid_rows)[0][[0, -1]]
            col_min, col_max = np.where(valid_cols)[0][[0, -1]]

            # Crop to valid data region
            cropped = merged.intensity[row_min : row_max + 1, col_min : col_max + 1]
            extent = [
                merged.x_edges[col_min],
                merged.x_edges[col_max + 1],
                merged.y_edges[row_min],
                merged.y_edges[row_max + 1],
            ]
        else:
            # No valid data, show full extent
            cropped = merged.intensity
            extent = [merged.x_edges[0], merged.x_edges[-1], merged.y_edges[0], merged.y_edges[-1]]

        im = ax.imshow(
            cropped,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower",
            aspect="equal",  # Fill the axes
        )

        # Add range rings
        _draw_range_rings(ax, extent)

        ax.set_xlabel("Distance East (m)")
        ax.set_ylabel("Distance North (m)")

        start_time = np.datetime_as_string(merged.start_time, unit="s")
        end_time = np.datetime_as_string(merged.end_time, unit="s")

        ax.set_title(
            f"Window {current_idx[0] + 1}/{len(merged_images)}: {merged.n_frames} frames\n"
            f"{start_time} to {end_time}"
        )

        # Update colorbar (using dedicated axes so it doesn't resize main plot)
        if not hasattr(fig, "_colorbar"):
            fig._colorbar = fig.colorbar(im, cax=cax, label="Intensity")
        else:
            fig._colorbar.update_normal(im)

        # Update polar inset plots
        update_polar_plots(merged)

        fig.canvas.draw_idle()

    def on_prev(event):
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_plot()

    def on_next(event):
        if current_idx[0] < len(merged_images) - 1:
            current_idx[0] += 1
            update_plot()

    def animate_frame(frame_num):
        """Animation callback - advance to next frame."""
        if is_playing[0]:
            if current_idx[0] < len(merged_images) - 1:
                current_idx[0] += 1
            else:
                # Loop back to start
                current_idx[0] = 0
            update_plot()

    def on_play(event):
        """Toggle play/stop state."""
        is_playing[0] = not is_playing[0]

        if is_playing[0]:
            btn_play.label.set_text("Stop")
            # Start animation
            animation[0] = FuncAnimation(
                fig,
                animate_frame,
                interval=interval_ms,
                cache_frame_data=False,
            )
        else:
            btn_play.label.set_text("Play")
            # Stop animation
            if animation[0] is not None:
                animation[0].event_source.stop()
                animation[0] = None

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "left":
            on_prev(event)
        elif event.key == "right":
            on_next(event)
        elif event.key == " ":  # Spacebar toggles play
            on_play(event)

    # Add navigation buttons using explicit figure reference
    ax_prev = fig.add_axes((0.2, 0.02, 0.12, 0.05))
    ax_play = fig.add_axes((0.37, 0.02, 0.12, 0.05))
    ax_next = fig.add_axes((0.54, 0.02, 0.12, 0.05))

    btn_prev = Button(ax_prev, "Previous")
    btn_play = Button(ax_play, "Play")
    btn_next = Button(ax_next, "Next")

    btn_prev.on_clicked(on_prev)
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(on_next)

    # Key bindings
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial plot
    update_plot()

    # Show with blocking
    plt.show(block=True)


def run(args) -> None:
    """Execute the 'files-pipeline' command."""
    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames

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
    )

    logging.info(
        "Window config: %.1fs duration, %.0f%% overlap, min %d frames",
        window_config.window_seconds,
        window_config.overlap_fraction * 100,
        window_config.min_frames_per_window,
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
            # logging.info("Displaying first merged image (processing continues)...")
            _show_single_image(merged)

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
        _show_merged_viewer(merged_images)


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser

    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Merge frames into composite images")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
