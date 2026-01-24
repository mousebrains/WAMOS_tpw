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
    from wamos_tpw.priority_executor import Result


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

    def __init__(
        self,
        prev: FramePipeline | None,
        current: FramePipeline,
        next_frame: FramePipeline | None,
        tolerance: float = 1.2,
    ) -> None:
        """
        Initialize frame interpolator.

        Args:
            prev: Previous frame (can be None for first frame)
            current: Current frame to compute per-radial values for
            next_frame: Next frame (can be None for last frame)
            tolerance: Multiplier for repeat_time to accept pair (1.2 = 20% margin)

        Raises:
            ValueError: If neither interpolation nor extrapolation is possible
        """
        self._prev = prev
        self._current = current
        self._next = next_frame
        self._tolerance = tolerance
        self._method: str = "none"
        self._timing_method: str = "linear"  # or "pps"

        meta_curr = current.metadata
        repeat_time = meta_curr.repeat_time or 1.5
        max_dt = repeat_time * tolerance

        self._repeat_time = repeat_time
        self._n_radials = current.n_bearings

        # Try forward interpolation first (current + next)
        can_interpolate = False
        if next_frame is not None:
            dt_forward = self._time_delta_seconds(
                meta_curr.timestamp, next_frame.metadata.timestamp
            )
            if dt_forward > 0 and dt_forward <= max_dt:
                can_interpolate = True
                self._dt = dt_forward
                self._method = "interpolate"

        # Fall back to backward extrapolation (prev + current)
        can_extrapolate = False
        if not can_interpolate and prev is not None:
            dt_backward = self._time_delta_seconds(prev.metadata.timestamp, meta_curr.timestamp)
            if dt_backward > 0 and dt_backward <= max_dt:
                can_extrapolate = True
                self._dt = dt_backward
                self._method = "extrapolate"

        if not can_interpolate and not can_extrapolate:
            raise ValueError(
                f"Cannot interpolate or extrapolate for frame at {meta_curr.timestamp}: "
                f"no valid adjacent frame within tolerance {max_dt:.2f}s"
            )

        # Compute timestamps using PPS or linear fallback
        self._compute_timestamps()

        # Compute positions based on method
        if self._method == "interpolate":
            self._compute_interpolated_positions()
        else:
            self._compute_extrapolated_positions()

    def _time_delta_seconds(self, t0: np.datetime64, t1: np.datetime64) -> float:
        """Return time difference in seconds."""
        return (t1 - t0) / np.timedelta64(1, "s")

    def _get_pps_anchors(self, frame: FramePipeline) -> list[tuple[int, np.datetime64]]:
        """
        Get PPS timing anchors from a frame.

        Returns list of (radial_index, whole_second_timestamp) tuples.
        """
        if frame is None or frame.pps is None:
            return []

        pps = frame.pps
        if not pps:
            return []

        pps_indices = pps.indices
        if len(pps_indices) == 0:
            return []

        meta = frame.metadata
        start_time = meta.timestamp
        repeat_time = meta.repeat_time or 1.5
        n_radials = frame.n_bearings

        anchors = []
        for idx in pps_indices:
            # Estimate time at this radial using linear model
            fraction = idx / n_radials
            estimated_ns = start_time + np.timedelta64(int(fraction * repeat_time * 1e9), "ns")

            # Round to nearest whole second (PPS occurs at whole seconds)
            # Convert to seconds since epoch, round, convert back
            estimated_s = estimated_ns.astype("datetime64[s]")
            # Check if we should round up or down
            remainder_ns = (estimated_ns - estimated_s) / np.timedelta64(1, "ns")
            if remainder_ns >= 0.5e9:
                whole_second = estimated_s + np.timedelta64(1, "s")
            else:
                whole_second = estimated_s

            anchors.append((idx, whole_second))

        return anchors

    def _compute_timestamps(self) -> None:
        """Compute per-radial timestamps using PPS anchors or linear fallback."""
        n_radials = self._n_radials

        # Collect PPS anchors from all available frames
        all_anchors = []

        # Get anchors from previous frame (offset indices to be relative to current)
        if self._prev is not None:
            prev_anchors = self._get_pps_anchors(self._prev)
            # Previous frame's radials come before current frame
            # Offset by -n_radials_prev
            n_prev = self._prev.n_bearings
            for idx, ts in prev_anchors:
                all_anchors.append((idx - n_prev, ts))

        # Get anchors from current frame
        curr_anchors = self._get_pps_anchors(self._current)
        all_anchors.extend(curr_anchors)

        # Get anchors from next frame (offset indices)
        if self._next is not None:
            next_anchors = self._get_pps_anchors(self._next)
            for idx, ts in next_anchors:
                all_anchors.append((idx + n_radials, ts))

        if len(all_anchors) >= 2:
            # Use PPS anchors to build timing model
            self._timing_method = "pps"
            self._times = self._interpolate_from_pps(all_anchors, n_radials)
        elif len(all_anchors) == 1:
            # Single PPS anchor - use it with repeat_time for rate
            self._timing_method = "pps"
            idx, ts = all_anchors[0]
            # Rate: radials per nanosecond
            rate_ns = self._repeat_time * 1e9 / n_radials
            radial_indices = np.arange(n_radials)
            offsets_ns = (radial_indices - idx) * rate_ns
            self._times = ts + offsets_ns.astype("timedelta64[ns]")
        else:
            # No PPS anchors - fall back to linear model
            self._timing_method = "linear"
            self._times = self._compute_linear_timestamps()

    def _interpolate_from_pps(
        self, anchors: list[tuple[int, np.datetime64]], n_radials: int
    ) -> np.ndarray:
        """
        Interpolate timestamps from PPS anchors.

        Uses linear interpolation between anchor points.
        """
        # Sort anchors by index
        anchors = sorted(anchors, key=lambda x: x[0])

        # Convert to arrays for interpolation
        anchor_indices = np.array([a[0] for a in anchors])
        anchor_times_ns = np.array(
            [(a[1] - np.datetime64(0, "ns")) / np.timedelta64(1, "ns") for a in anchors]
        )

        # Interpolate for all radial indices
        radial_indices = np.arange(n_radials)
        interpolated_ns = np.interp(radial_indices, anchor_indices, anchor_times_ns)

        # Convert back to datetime64
        return np.datetime64(0, "ns") + interpolated_ns.astype("timedelta64[ns]")

    def _compute_linear_timestamps(self) -> np.ndarray:
        """Compute timestamps using linear model from start_time and repeat_time."""
        meta = self._current.metadata
        n_radials = self._n_radials
        radial_fractions = np.linspace(0, 1, n_radials, endpoint=False)
        return meta.timestamp + (radial_fractions * self._repeat_time * 1e9).astype(
            "timedelta64[ns]"
        )

    def _compute_interpolated_positions(self) -> None:
        """Compute interpolated positions and headings using current and next frames."""
        meta_curr = self._current.metadata
        meta_next = self._next.metadata
        n_radials = self._n_radials

        radial_fractions = np.linspace(0, 1, n_radials, endpoint=False)

        # Position scale: how much of the inter-frame motion applies to one frame
        position_scale = self._repeat_time / self._dt if self._dt > 0 else 1.0

        # Interpolate position
        lat0 = meta_curr.latitude or 0.0
        lon0 = meta_curr.longitude or 0.0
        lat1 = meta_next.latitude or 0.0
        lon1 = meta_next.longitude or 0.0

        self._latitudes = lat0 + radial_fractions * (lat1 - lat0) * position_scale
        self._longitudes = self._interpolate_longitude(lon0, lon1, radial_fractions, position_scale)

        # Interpolate heading (circular)
        hdg0 = meta_curr.heading or 0.0
        hdg1 = meta_next.heading or 0.0
        self._headings = self._interpolate_angle(hdg0, hdg1, radial_fractions, position_scale)

    def _compute_extrapolated_positions(self) -> None:
        """Compute extrapolated positions and headings using previous and current frames."""
        meta_prev = self._prev.metadata
        meta_curr = self._current.metadata
        n_radials = self._n_radials

        radial_fractions = np.linspace(0, 1, n_radials, endpoint=False)

        # Rate of change from prev to current, then project forward
        rate_scale = self._repeat_time / self._dt if self._dt > 0 else 1.0

        # Extrapolate position using rate from prev->current
        lat_prev = meta_prev.latitude or 0.0
        lon_prev = meta_prev.longitude or 0.0
        lat_curr = meta_curr.latitude or 0.0
        lon_curr = meta_curr.longitude or 0.0

        lat_rate = lat_curr - lat_prev
        self._latitudes = lat_curr + radial_fractions * lat_rate * rate_scale

        # Handle longitude carefully for rate calculation
        lon_diff = lon_curr - lon_prev
        if abs(lon_diff) > 180:
            lon_diff = lon_diff - 360 if lon_diff > 0 else lon_diff + 360
        lon_rate = lon_diff
        self._longitudes = lon_curr + radial_fractions * lon_rate * rate_scale
        self._longitudes = ((self._longitudes + 180) % 360) - 180

        # Extrapolate heading (circular)
        hdg_prev = meta_prev.heading or 0.0
        hdg_curr = meta_curr.heading or 0.0
        hdg_diff = hdg_curr - hdg_prev
        if abs(hdg_diff) > 180:
            hdg_diff = hdg_diff - 360 if hdg_diff > 0 else hdg_diff + 360
        self._headings = (hdg_curr + radial_fractions * hdg_diff * rate_scale) % 360

    def _interpolate_longitude(
        self, lon0: float, lon1: float, fractions: np.ndarray, scale: float
    ) -> np.ndarray:
        """Interpolate longitude handling date line wrap-around."""
        lon_diff = lon1 - lon0
        if abs(lon_diff) > 180:
            lon_diff = lon_diff - 360 if lon_diff > 0 else lon_diff + 360
        result = lon0 + fractions * lon_diff * scale
        return ((result + 180) % 360) - 180

    def _interpolate_angle(
        self, angle0: float, angle1: float, fractions: np.ndarray, scale: float
    ) -> np.ndarray:
        """Interpolate angles (0-360) handling wrap-around."""
        angle_diff = angle1 - angle0
        if abs(angle_diff) > 180:
            angle_diff = angle_diff - 360 if angle_diff > 0 else angle_diff + 360
        result = angle0 + fractions * angle_diff * scale
        return result % 360

    @property
    def method(self) -> str:
        """Return the position method used: 'interpolate' or 'extrapolate'."""
        return self._method

    @property
    def timing_method(self) -> str:
        """Return the timing method used: 'pps' or 'linear'."""
        return self._timing_method

    @property
    def frame(self) -> FramePipeline:
        """Return the current frame."""
        return self._current

    @property
    def prev_frame(self) -> FramePipeline | None:
        """Return the previous frame (may be None)."""
        return self._prev

    @property
    def next_frame(self) -> FramePipeline | None:
        """Return the next frame (may be None)."""
        return self._next

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


# ============================================================
# Task handlers for priority executor (must be at module level)
# ============================================================


def _do_process_file(task) -> "Result":
    """
    Process a single polar file and return FrameData for each frame.

    This task loads a file, processes each frame through the pipeline,
    and returns serializable FrameData with shared memory for arrays.
    """
    import resource
    from wamos_tpw.priority_executor import Result, create_shared_array
    from wamos_tpw.config import Config
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.frame_pipeline import FramePipeline

    filepath, file_index, config_dict, qTiming = task.data

    # Reconstruct config
    config = Config()
    if config_dict:
        config._config = config_dict

    t0 = time.perf_counter() if qTiming else None
    pf = PolarFile(filepath, config=config)
    frames_data = []
    shm_names = []

    for frame_idx, frame in enumerate(pf):
        fp = FramePipeline(frame, config=config, qTiming=qTiming)

        # Create shared memory for arrays
        theta_shm = create_shared_array(fp.theta_array)
        ground_range_shm = create_shared_array(fp.ground_range)
        intensity_shm = create_shared_array(fp.final_intensity)

        shm_names.extend([theta_shm[0], ground_range_shm[0], intensity_shm[0]])

        frame_data = FrameData(
            filepath=filepath,
            file_index=file_index,
            frame_index=frame_idx,
            timestamp=fp.metadata.timestamp,
            repeat_time=fp.metadata.repeat_time or 1.5,
            latitude=fp.metadata.latitude,
            longitude=fp.metadata.longitude,
            heading=fp.metadata.heading,
            ship_speed=fp.metadata.ship_speed,
            wind_speed=fp.metadata.wind_speed,
            wind_direction=fp.metadata.wind_direction,
            n_bearings=fp.n_bearings,
            n_distances=fp.n_distances,
            pps_indices=fp.pps.indices if fp.pps else None,
            theta_shm=theta_shm,
            ground_range_shm=ground_range_shm,
            intensity_shm=intensity_shm,
            timings=fp.timings if qTiming else {},
        )
        frames_data.append(frame_data)

    elapsed = time.perf_counter() - t0 if t0 else 0.0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return Result(
        task_type="process_file",
        task_id=task.task_id,
        data={
            "filepath": filepath,
            "file_index": file_index,
            "frames": frames_data,
            "elapsed": elapsed,
            "peak_rss": peak_rss,
        },
        shm_to_release=[],  # Don't release yet - needed for interpolation
    )


def _write_frame_netcdf(
    netcdf_dir: str,
    timestamp: np.datetime64,
    projected_intensity: np.ndarray,
    grid_params: dict,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings: np.ndarray,
    ship_speed: float | None,
    wind_speed: float | None,
    wind_direction: float | None,
    file_index: int,
    frame_index: int,
) -> str:
    """
    Write a single frame's projected data to a NetCDF file.

    Args:
        netcdf_dir: Output directory
        timestamp: Frame timestamp
        projected_intensity: 2D projected intensity array
        grid_params: Grid parameters (x_edges, y_edges, center_lat, etc.)
        latitudes: Per-radial latitudes
        longitudes: Per-radial longitudes
        headings: Per-radial ship headings
        ship_speed: Ship speed (m/s) or None
        wind_speed: Wind speed (m/s) or None
        wind_direction: Wind direction (degrees) or None
        file_index: Source file index
        frame_index: Frame index within file

    Returns:
        Path to the written NetCDF file
    """
    import os

    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not installed, skipping NetCDF output")
        return ""

    # Generate filename from timestamp: YYYYMMDD_HHMMSS_fff.nc
    ts_str = np.datetime_as_string(timestamp, unit="ms")
    # Format: 2022-04-05T14:00:00.123 -> 20220405_140000_123
    filename = ts_str.replace("-", "").replace(":", "").replace("T", "_").replace(".", "_") + ".nc"
    filepath = os.path.join(netcdf_dir, filename)

    # Compute grid centers from edges
    x_centers = (grid_params["x_edges"][:-1] + grid_params["x_edges"][1:]) / 2
    y_centers = (grid_params["y_edges"][:-1] + grid_params["y_edges"][1:]) / 2

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "intensity": (
                ["y", "x"],
                projected_intensity,
                {
                    "long_name": "Projected radar intensity",
                    "units": "counts",
                    "coordinates": "x y",
                },
            ),
        },
        coords={
            "x": (
                ["x"],
                x_centers,
                {
                    "long_name": "Distance east from center",
                    "units": "m",
                    "axis": "X",
                },
            ),
            "y": (
                ["y"],
                y_centers,
                {
                    "long_name": "Distance north from center",
                    "units": "m",
                    "axis": "Y",
                },
            ),
            "time": timestamp,
        },
        attrs={
            "title": "WAMOS radar frame projection",
            "institution": "WAMOS TPW",
            "source": "wamos interpolator",
            "history": f"Created {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            # Grid metadata
            "grid_spacing_m": grid_params["grid_spacing"],
            "utm_zone": grid_params["utm_zone"],
            "hemisphere": grid_params["hemisphere"],
            "center_latitude": grid_params["center_lat"],
            "center_longitude": grid_params["center_lon"],
            "crs": f"EPSG:326{grid_params['utm_zone']:02d}"
            if grid_params["hemisphere"] == "north"
            else f"EPSG:327{grid_params['utm_zone']:02d}",
            # Frame indices
            "file_index": file_index,
            "frame_index": frame_index,
        },
    )

    # Add ship/wind metadata as scalar variables
    if ship_speed is not None:
        ds["ship_speed"] = xr.DataArray(
            ship_speed,
            attrs={"long_name": "Ship speed", "units": "m/s"},
        )
    if wind_speed is not None:
        ds["wind_speed"] = xr.DataArray(
            wind_speed,
            attrs={"long_name": "Wind speed", "units": "m/s"},
        )
    if wind_direction is not None:
        ds["wind_direction"] = xr.DataArray(
            wind_direction,
            attrs={"long_name": "Wind direction (from)", "units": "degrees"},
        )

    # Add mean heading
    ds["ship_heading"] = xr.DataArray(
        float(np.mean(headings)),
        attrs={"long_name": "Mean ship heading", "units": "degrees"},
    )

    # Add per-radial data as 1D arrays
    ds["radial_latitude"] = xr.DataArray(
        latitudes,
        dims=["radial"],
        attrs={"long_name": "Per-radial latitude", "units": "degrees_north"},
    )
    ds["radial_longitude"] = xr.DataArray(
        longitudes,
        dims=["radial"],
        attrs={"long_name": "Per-radial longitude", "units": "degrees_east"},
    )
    ds["radial_heading"] = xr.DataArray(
        headings,
        dims=["radial"],
        attrs={"long_name": "Per-radial ship heading", "units": "degrees"},
    )

    # Write to file with compression
    encoding = {
        "intensity": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "radial_latitude": {"zlib": True, "complevel": 4},
        "radial_longitude": {"zlib": True, "complevel": 4},
        "radial_heading": {"zlib": True, "complevel": 4},
    }

    ds.to_netcdf(filepath, encoding=encoding)

    return filepath


def _do_interpolate(task) -> "Result":
    """
    Perform interpolation on a triplet, then project to UTM grid.

    This task runs the FrameInterpolator logic on serialized frame data,
    then projects the intensity onto a per-frame UTM grid.
    """
    import resource
    from wamos_tpw.priority_executor import Result, read_shared_array

    prev_data, current_data, next_data, tolerance, do_projection, netcdf_dir = task.data

    t0_total = time.perf_counter()
    timings = {}

    # Read intensity, theta, ground_range from shared memory for projection.
    # Only the CURRENT frame needs these arrays - prev/next are only used
    # for their metadata (timestamp, lat, lon, heading, pps) to compute
    # interpolated per-radial values.
    t0 = time.perf_counter()
    intensity = None
    theta = None
    ground_range = None

    if current_data.intensity_shm:
        intensity = read_shared_array(*current_data.intensity_shm)
    if current_data.theta_shm:
        theta = read_shared_array(*current_data.theta_shm)
    if current_data.ground_range_shm:
        ground_range = read_shared_array(*current_data.ground_range_shm)
    timings["read_shm"] = time.perf_counter() - t0

    # Build lightweight wrappers that provide the interface FrameInterpolator expects
    class _MetadataProxy:
        def __init__(self, frame_data: FrameData):
            self.timestamp = frame_data.timestamp
            self.repeat_time = frame_data.repeat_time
            self.latitude = frame_data.latitude
            self.longitude = frame_data.longitude
            self.heading = frame_data.heading

    class _PPSProxy:
        def __init__(self, indices: np.ndarray | None):
            self.indices = indices if indices is not None else np.array([], dtype=np.int32)

        def __bool__(self):
            return len(self.indices) > 0

    class FrameProxy:
        """Proxy for FramePipeline that uses serialized FrameData."""

        def __init__(self, frame_data: FrameData):
            self._data = frame_data
            self._metadata = _MetadataProxy(frame_data)
            self._pps = (
                _PPSProxy(frame_data.pps_indices) if frame_data.pps_indices is not None else None
            )

        @property
        def metadata(self):
            return self._metadata

        @property
        def pps(self):
            return self._pps

        @property
        def n_bearings(self):
            return self._data.n_bearings

    # Create proxies
    t0 = time.perf_counter()
    prev_proxy = FrameProxy(prev_data) if prev_data else None
    current_proxy = FrameProxy(current_data)
    next_proxy = FrameProxy(next_data) if next_data else None
    timings["build_proxies"] = time.perf_counter() - t0

    try:
        # Run interpolation
        t0 = time.perf_counter()
        interp = FrameInterpolator(
            prev_proxy,
            current_proxy,
            next_proxy,
            tolerance=tolerance,
        )
        timings["interpolate"] = time.perf_counter() - t0

        # UTM projection (if requested)
        projected_intensity = None
        grid_params = None

        if do_projection and intensity is not None and ground_range is not None:
            t0 = time.perf_counter()
            from pyproj import CRS, Transformer

            latitudes = interp.latitudes
            longitudes = interp.longitudes
            headings = interp.headings

            # Determine UTM zone from center of frame
            ref_lat = float(np.mean(latitudes))
            ref_lon = float(np.mean(longitudes))
            utm_zone = int((ref_lon + 180) / 6) % 60 + 1
            hemisphere = "north" if ref_lat >= 0 else "south"

            utm_crs = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")
            crs_wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

            # Convert ship positions to UTM
            ship_x, ship_y = transformer.transform(longitudes, latitudes)

            # Compute grid extent for this frame
            max_range = float(ground_range[-1]) * 1.1
            # Grid spacing = max of range resolution and angular width at outermost range
            # Angular width = arc length between adjacent radials = range * (2π / n_bearings)
            range_res = float(ground_range[1] - ground_range[0]) if len(ground_range) > 1 else 10.0
            n_bearings = intensity.shape[0]
            angular_width = float(ground_range[-1]) * 2 * np.pi / n_bearings
            grid_spacing = max(range_res, angular_width)

            x_min = ship_x.min() - max_range
            x_max = ship_x.max() + max_range
            y_min = ship_y.min() - max_range
            y_max = ship_y.max() + max_range

            n_x = int(np.ceil((x_max - x_min) / grid_spacing))
            n_y = int(np.ceil((y_max - y_min) / grid_spacing))

            x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
            y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)

            # Initialize accumulation arrays
            intensity_sum = np.zeros((n_y, n_x), dtype=np.float64)
            intensity_count = np.zeros((n_y, n_x), dtype=np.int32)

            # Project: compute earth bearing and positions
            earth_bearing_rad = np.deg2rad((theta + headings) % 360)
            sin_bearing = np.sin(earth_bearing_rad)
            cos_bearing = np.cos(earth_bearing_rad)

            x_coords = np.outer(sin_bearing, ground_range) + ship_x[:, np.newaxis]
            y_coords = np.outer(cos_bearing, ground_range) + ship_y[:, np.newaxis]

            # Convert to grid indices
            inv_spacing = 1.0 / grid_spacing
            x_idx = ((x_coords - x_min) * inv_spacing).astype(np.int32)
            y_idx = ((y_coords - y_min) * inv_spacing).astype(np.int32)

            # Flatten and filter valid
            x_flat = x_idx.ravel()
            y_flat = y_idx.ravel()
            values_flat = intensity.ravel()

            valid = (
                (x_flat >= 0)
                & (x_flat < n_x)
                & (y_flat >= 0)
                & (y_flat < n_y)
                & ~np.isnan(values_flat)
            )

            if np.sum(valid) > 0:
                linear_idx = y_flat[valid] * n_x + x_flat[valid]
                grid_size = n_x * n_y
                batch_sum = np.bincount(linear_idx, weights=values_flat[valid], minlength=grid_size)
                batch_count = np.bincount(linear_idx, minlength=grid_size)
                intensity_sum.ravel()[:] += batch_sum
                intensity_count.ravel()[:] += batch_count

            # Compute averaged intensity
            with np.errstate(invalid="ignore"):
                projected_intensity = intensity_sum / intensity_count
            projected_intensity[intensity_count == 0] = np.nan

            # Grid center for display coordinates
            x_center = (x_edges[0] + x_edges[-1]) / 2
            y_center = (y_edges[0] + y_edges[-1]) / 2

            # Convert center to lat/lon
            transformer_inv = Transformer.from_crs(utm_crs, crs_wgs84, always_xy=True)
            center_lon, center_lat = transformer_inv.transform(x_center, y_center)

            grid_params = {
                "x_edges": x_edges - x_center,  # Centered coordinates
                "y_edges": y_edges - y_center,
                "grid_spacing": grid_spacing,
                "utm_zone": utm_zone,
                "hemisphere": hemisphere,
                "center_lat": float(center_lat),
                "center_lon": float(center_lon),
                "n_x": n_x,
                "n_y": n_y,
            }
            timings["project"] = time.perf_counter() - t0

            # Write NetCDF file if output directory specified
            if netcdf_dir and projected_intensity is not None:
                t0 = time.perf_counter()
                _write_frame_netcdf(
                    netcdf_dir=netcdf_dir,
                    timestamp=current_data.timestamp,
                    projected_intensity=projected_intensity,
                    grid_params=grid_params,
                    latitudes=latitudes,
                    longitudes=longitudes,
                    headings=headings,
                    ship_speed=current_data.ship_speed,
                    wind_speed=current_data.wind_speed,
                    wind_direction=current_data.wind_direction,
                    file_index=current_data.file_index,
                    frame_index=current_data.frame_index,
                )
                timings["netcdf"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t0_total

        result_data = {
            "file_index": current_data.file_index,
            "frame_index": current_data.frame_index,
            "filepath": current_data.filepath,
            "timestamp": current_data.timestamp,
            "method": interp.method,
            "timing_method": interp.timing_method,
            "time_delta": interp.time_delta,
            "times": interp.times,
            "latitudes": interp.latitudes,
            "longitudes": interp.longitudes,
            "headings": interp.headings,
            # Ship and wind metadata
            "ship_speed": current_data.ship_speed,
            "wind_speed": current_data.wind_speed,
            "wind_direction": current_data.wind_direction,
            # Projected data (if do_projection=True)
            "projected_intensity": projected_intensity,
            "grid_params": grid_params,
            # Timings
            "timings": timings,
            "success": True,
            "error": None,
        }
    except ValueError as e:
        timings["total"] = time.perf_counter() - t0_total
        result_data = {
            "file_index": current_data.file_index,
            "frame_index": current_data.frame_index,
            "filepath": current_data.filepath,
            "timestamp": current_data.timestamp,
            "prev_timestamp": prev_data.timestamp if prev_data else None,
            "next_timestamp": next_data.timestamp if next_data else None,
            "timings": timings,
            "success": False,
            "error": str(e),
        }

    # Track shared memory to release (the triplet's arrays)
    shm_to_release = []
    for frame_data in [prev_data, current_data, next_data]:
        if frame_data:
            if frame_data.theta_shm:
                shm_to_release.append(frame_data.theta_shm[0])
            if frame_data.ground_range_shm:
                shm_to_release.append(frame_data.ground_range_shm[0])
            if frame_data.intensity_shm:
                shm_to_release.append(frame_data.intensity_shm[0])

    elapsed = time.perf_counter() - t0
    result_data["elapsed"] = elapsed
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_data["peak_rss"] = peak_rss

    return Result(
        task_type="interpolate",
        task_id=task.task_id,
        data=result_data,
        shm_to_release=[],  # Don't release - main process handles lifecycle
    )


# Task handlers registry for priority executor
TASK_HANDLERS = {
    "process_file": _do_process_file,
    "interpolate": _do_interpolate,
}


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
    n_skipped = 0

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
                task_data = (prev, current, next_frame, args.tolerance, args.project, netcdf_dir)
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

                if data["timing_method"] == "pps":
                    n_pps += 1
                else:
                    n_linear += 1
            else:
                n_skipped += 1
                prev_ts = data.get("prev_timestamp")
                next_ts = data.get("next_timestamp")
                curr_ts = data["timestamp"]
                logging.warning(
                    "Frame (%d, %d): skipped - %s | prev=%s, curr=%s, next=%s",
                    data["file_index"],
                    data["frame_index"],
                    data["error"],
                    np.datetime_as_string(prev_ts, unit="ms") if prev_ts is not None else "None",
                    np.datetime_as_string(curr_ts, unit="ms"),
                    np.datetime_as_string(next_ts, unit="ms") if next_ts is not None else "None",
                )

            # Check for more ready triplets (in case file results arrived while processing)
            for prev, current, next_frame in triplet_collector.ready_triplets():
                task_data = (prev, current, next_frame, args.tolerance, args.project, netcdf_dir)
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
        "Summary: position: %d interpolated, %d extrapolated; "
        "timing: %d pps, %d linear; %d skipped",
        n_interpolated,
        n_extrapolated,
        n_pps,
        n_linear,
        n_skipped,
    )

    # Timing statistics
    if args.timing and interp_results:
        # Collect timing data from all successful results
        timing_keys = ["read_shm", "build_proxies", "interpolate", "project", "netcdf", "total"]
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
                logging.info("  %-15s: %7.2f ms avg, %7.2f s total", k, avg_ms, total_s)

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


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test frame interpolation/extrapolation")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
