#! /usr/bin/env python3
#
# Time window creation and frame accumulation for WAMOS radar data
#
# Jan-2026, Pat Welch, pat@mousebrains.com

"""Time window creation and frame accumulation utilities for merging radar frames."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from wamos_tpw.filenames import extract_file_timestamp

if TYPE_CHECKING:
    from wamos_tpw.grid import GridParams
    from wamos_tpw.merged_image import MergedImage, TimeWindowConfig

logger = logging.getLogger(__name__)


def _interpolate_nan_gaps(intensity: np.ndarray, max_distance: int = 3) -> np.ndarray:
    """
    Fill small NaN gaps in intensity array using nearest neighbor interpolation.

    Only fills NaN pixels that are within max_distance of a valid pixel.
    Large NaN regions (shadows, outside radar) are preserved.

    Args:
        intensity: 2D array with NaN values for missing data
        max_distance: Maximum distance (in pixels) to interpolate across

    Returns:
        Array with small NaN gaps filled by interpolation
    """
    from scipy.ndimage import distance_transform_edt

    # Find NaN mask
    nan_mask = np.isnan(intensity)
    if not np.any(nan_mask):
        return intensity  # No gaps to fill

    valid_mask = ~nan_mask
    if not np.any(valid_mask):
        return intensity  # No valid data to interpolate from

    # Get distance to nearest valid pixel and indices
    distances, indices = distance_transform_edt(nan_mask, return_indices=True)

    # Only fill NaN pixels within max_distance of valid data
    fill_mask = nan_mask & (distances <= max_distance)

    # Copy values from nearest valid pixels (only for small gaps)
    result = intensity.copy()
    result[fill_mask] = intensity[indices[0, fill_mask], indices[1, fill_mask]]

    return result


def create_time_windows(
    files: list[str],
    window_config: TimeWindowConfig,
) -> list[tuple[np.datetime64, np.datetime64, list[int]]]:
    """
    Create overlapping time windows from a file list.

    Uses binary search for O(N log N + W log N) complexity instead of O(W * N),
    where N = number of files and W = number of windows.

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

    # Sort by timestamp for binary search (preserves original file indices)
    sort_order = np.argsort(timestamps)
    sorted_timestamps = timestamps[sort_order]
    sorted_indices = valid_indices[sort_order]

    # Get time range
    t_min = sorted_timestamps[0]
    t_max = sorted_timestamps[-1]

    window_ns = np.timedelta64(int(window_config.window_seconds * 1e9), "ns")
    stride_ns = np.timedelta64(int(window_config.stride_seconds * 1e9), "ns")

    windows = []
    window_start = t_min

    while window_start <= t_max:
        window_end = window_start + window_ns

        # Use binary search to find file range: O(log N) instead of O(N)
        # searchsorted('left') finds first index where value could be inserted
        # searchsorted('right') finds last index where value could be inserted
        start_idx = np.searchsorted(sorted_timestamps, window_start, side="left")
        end_idx = np.searchsorted(sorted_timestamps, window_end, side="left")

        # Extract file indices for this window
        n_files = end_idx - start_idx
        if n_files >= window_config.min_frames_per_window:
            file_indices = sorted_indices[start_idx:end_idx].tolist()
            windows.append((window_start, window_end, file_indices))

        window_start = window_start + stride_ns

    return windows


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

    def finalize(self, window_index: int = 0, interpolate_gaps: bool = False) -> MergedImage:
        """
        Compute averaged intensity and return MergedImage.

        Args:
            window_index: Index of this window in the sequence
            interpolate_gaps: Fill NaN gaps using nearest neighbor interpolation

        Returns:
            MergedImage with averaged data and metadata
        """
        from wamos_tpw.merged_image import MergedImage

        with np.errstate(invalid="ignore"):
            intensity = self.intensity_sum / self.intensity_count
        intensity[self.intensity_count == 0] = np.nan

        # Interpolate gaps if requested
        if interpolate_gaps:
            intensity = _interpolate_nan_gaps(intensity)

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


_DEG2M = 111_319.5


def merge_frames(
    frames: list[dict],
    grid_params: GridParams | dict,
    window_idx: int = 0,
    interpolate_gaps: bool = False,
) -> MergedImage | None:
    """
    Remap interpolated frames onto a common grid and merge.

    Each frame's per-frame projected data is remapped onto ``grid_params``
    and accumulated into a single :class:`MergedImage`.

    Args:
        frames: Interpolated frame data dicts (from ``_do_interpolate``).
        grid_params: Common grid (from ``compute_common_grid*``).
        window_idx: Window index for the output MergedImage.
        interpolate_gaps: Fill NaN gaps via nearest-neighbor interpolation.

    Returns:
        MergedImage or None if no frames contributed valid data.
    """
    from wamos_tpw.grid import remap_to_common_grid

    accumulator = WindowAccumulator(
        x_edges=grid_params["x_edges"],
        y_edges=grid_params["y_edges"],
        grid_spacing=grid_params["grid_spacing"],
        utm_zone=grid_params["utm_zone"],
        hemisphere=grid_params["hemisphere"],
        center_lat=grid_params["center_lat"],
        center_lon=grid_params["center_lon"],
    )

    for frame_data in frames:
        proj_intensity = frame_data.get("projected_intensity")
        if proj_intensity is None:
            logger.debug(
                "Frame (%d, %d) has no projected data",
                frame_data.get("file_index", -1),
                frame_data.get("frame_index", -1),
            )
            continue

        headings = frame_data.get("headings")
        mean_heading = float(np.mean(headings)) if headings is not None else 0.0

        proj_count = frame_data.get("projected_count")
        frame_gp = frame_data.get("grid_params") or {}

        frame_x_edges = frame_gp.get("x_edges")
        frame_y_edges = frame_gp.get("y_edges")
        frame_center_lat = frame_gp.get("center_lat")
        frame_center_lon = frame_gp.get("center_lon")

        if (
            frame_x_edges is not None
            and frame_y_edges is not None
            and frame_center_lat is not None
            and frame_center_lon is not None
        ):
            ref_lat = grid_params["ref_lat"]
            ref_lon = grid_params["ref_lon"]
            m_per_deg_lon = grid_params["m_per_deg_lon"]

            frame_center_x = (frame_center_lon - ref_lon) * m_per_deg_lon
            frame_center_y = (frame_center_lat - ref_lat) * _DEG2M
            frame_x_abs = frame_x_edges + frame_center_x
            frame_y_abs = frame_y_edges + frame_center_y

            frame_sum, frame_count = remap_to_common_grid(
                proj_intensity,
                proj_count,
                frame_x_abs,
                frame_y_abs,
                grid_params["x_edges_abs"],
                grid_params["y_edges_abs"],
                grid_params["n_x"],
                grid_params["n_y"],
            )
        else:
            frame_sum = np.zeros(
                (grid_params["n_y"], grid_params["n_x"]), dtype=np.float64
            )
            frame_count = np.zeros(
                (grid_params["n_y"], grid_params["n_x"]), dtype=np.int32
            )
            if proj_intensity.shape == frame_sum.shape:
                valid = ~np.isnan(proj_intensity)
                frame_sum[valid] = proj_intensity[valid]
                if proj_count is not None:
                    frame_count[valid] = proj_count[valid]
                else:
                    frame_count[valid] = 1

        accumulator.add_projected(
            projected_intensity=frame_sum,
            projected_count=frame_count,
            timestamp=frame_data["timestamp"],
            heading=mean_heading,
            ship_speed=frame_data.get("ship_speed"),
            wind_speed=frame_data.get("wind_speed"),
            wind_direction=frame_data.get("wind_direction"),
        )

    if accumulator.n_frames == 0:
        return None

    return accumulator.finalize(
        window_index=window_idx,
        interpolate_gaps=interpolate_gaps,
    )
