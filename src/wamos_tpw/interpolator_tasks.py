#! /usr/bin/env python3
#
# Task handlers for interpolator priority executor
#
# Extracted from interpolator.py — contains parallel task handlers,
# proxy classes for FrameInterpolator, and NetCDF output helper.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.priority_executor import Result

from wamos_tpw.interpolator import FrameData, FrameInterpolator

logger = logging.getLogger(__name__)

# Numba JIT projection: ~4.5× faster than pure numpy.
# To disable numba and fall back to numpy, either:
#   - uninstall numba: pip3 uninstall numba
#   - or set _HAS_NUMBA = False after this block
try:
    import numba

    @numba.njit(cache=True)
    def _project_numba(sin_bearing, cos_bearing, ground_range, ship_x, ship_y,
                       intensity, x_min, y_min, inv_spacing, n_x, n_y):
        """Single-pass projection: no intermediate arrays, all work in compiled loop."""
        grid_size = n_x * n_y
        out_sum = np.zeros(grid_size, dtype=np.float64)
        out_cnt = np.zeros(grid_size, dtype=np.int32)
        n_bearings = sin_bearing.shape[0]
        n_distances = ground_range.shape[0]
        for i in range(n_bearings):
            sx = ship_x[i] - x_min
            sy = ship_y[i] - y_min
            for j in range(n_distances):
                v = intensity[i, j]
                if np.isnan(v):
                    continue
                xi = int((sin_bearing[i] * ground_range[j] + sx) * inv_spacing)
                yi = int((cos_bearing[i] * ground_range[j] + sy) * inv_spacing)
                if 0 <= xi < n_x and 0 <= yi < n_y:
                    idx = yi * n_x + xi
                    out_sum[idx] += v
                    out_cnt[idx] += 1
        return out_sum, out_cnt

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ============================================================
# Proxy classes for FrameInterpolator in worker processes
# ============================================================


class _MetadataProxy:
    """Lightweight proxy providing the metadata interface FrameInterpolator expects."""

    def __init__(self, frame_data: FrameData):
        self.timestamp = frame_data.timestamp
        self.repeat_time = frame_data.repeat_time
        self.latitude = frame_data.latitude
        self.longitude = frame_data.longitude
        self.heading = frame_data.heading


class _PPSProxy:
    """Lightweight proxy providing the PPS interface FrameInterpolator expects."""

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
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    @property
    def pps(self):
        return self._pps

    @property
    def n_bearings(self):
        return self._data.n_bearings


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
            repeat_time=fp.metadata.repeat_time or FrameInterpolator._DEFAULT_REPEAT_TIME,
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
            "center_latitude": grid_params["center_lat"],
            "center_longitude": grid_params["center_lon"],
            "projection": "equirectangular",
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

    # Create proxies
    t0 = time.perf_counter()
    prev_proxy = FrameProxy(prev_data) if prev_data else None
    current_proxy = FrameProxy(current_data)
    next_proxy = FrameProxy(next_data) if next_data else None
    timings["build_proxies"] = time.perf_counter() - t0

    # Run interpolation
    t0 = time.perf_counter()
    interp = FrameInterpolator(
        prev_proxy,
        current_proxy,
        next_proxy,
        tolerance=tolerance,
    )
    timings["interpolate"] = time.perf_counter() - t0

    # Earth-referenced projection (if requested)
    projected_intensity = None
    grid_params = None

    if do_projection and intensity is not None and ground_range is not None:
        t0_proj = time.perf_counter()

        latitudes = interp.latitudes
        longitudes = interp.longitudes
        headings = interp.headings

        # Reference point for equirectangular projection
        ref_lat = float(np.mean(latitudes))
        ref_lon = float(np.mean(longitudes))

        # Equirectangular: convert ship lat/lon to meters relative to reference
        _DEG2M = 111_319.5  # meters per degree of latitude
        meters_per_deg_lon = _DEG2M * np.cos(np.deg2rad(ref_lat))
        ship_x = (longitudes - ref_lon) * meters_per_deg_lon
        ship_y = (latitudes - ref_lat) * _DEG2M

        t1 = time.perf_counter()
        timings["proj_ship_pos"] = t1 - t0_proj

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

        t2 = time.perf_counter()
        timings["proj_grid_setup"] = t2 - t1

        # Project: compute earth bearing trig
        earth_bearing_rad = np.deg2rad((theta + headings) % 360)
        sin_bearing = np.sin(earth_bearing_rad)
        cos_bearing = np.cos(earth_bearing_rad)

        t3 = time.perf_counter()
        timings["proj_bearings"] = t3 - t2

        # Project all pixels onto the grid
        inv_spacing = 1.0 / grid_spacing

        if _HAS_NUMBA:
            # Single compiled loop — no intermediate arrays
            out_sum, out_cnt = _project_numba(
                sin_bearing, cos_bearing, ground_range, ship_x, ship_y,
                intensity, x_min, y_min, inv_spacing, n_x, n_y)
            intensity_sum.ravel()[:] = out_sum
            intensity_count.ravel()[:] = out_cnt
        else:
            # Numpy fallback: fused coordinate → grid index computation.
            # Eliminates intermediate float64 x_coords/y_coords arrays.
            x_idx = ((np.outer(sin_bearing, ground_range)
                      + (ship_x[:, np.newaxis] - x_min)) * inv_spacing).astype(np.int32).ravel()
            y_idx = ((np.outer(cos_bearing, ground_range)
                      + (ship_y[:, np.newaxis] - y_min)) * inv_spacing).astype(np.int32).ravel()

            values_flat = intensity.ravel()
            valid = (
                (x_idx >= 0)
                & (x_idx < n_x)
                & (y_idx >= 0)
                & (y_idx < n_y)
                & ~np.isnan(values_flat)
            )

            if np.any(valid):
                linear_idx = y_idx[valid] * n_x + x_idx[valid]
                grid_size = n_x * n_y
                intensity_sum.ravel()[:] += np.bincount(
                    linear_idx, weights=values_flat[valid], minlength=grid_size)
                intensity_count.ravel()[:] += np.bincount(
                    linear_idx, minlength=grid_size)

        t6 = time.perf_counter()
        timings["proj_bincount"] = t6 - t3  # projection loop total

        # Compute averaged intensity
        with np.errstate(invalid="ignore"):
            projected_intensity = intensity_sum / intensity_count
        projected_intensity[intensity_count == 0] = np.nan

        # Grid center for display coordinates
        x_center = (x_edges[0] + x_edges[-1]) / 2
        y_center = (y_edges[0] + y_edges[-1]) / 2

        # Convert center back to lat/lon via equirectangular inverse
        center_lon = ref_lon + x_center / meters_per_deg_lon
        center_lat = ref_lat + y_center / _DEG2M

        # UTM zone/hemisphere as informational metadata
        utm_zone = int((center_lon + 180) / 6) % 60 + 1
        hemisphere = "north" if center_lat >= 0 else "south"

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

        t7 = time.perf_counter()
        timings["proj_finalize"] = t7 - t6
        timings["project"] = t7 - t0_proj

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
        # Ground range metadata for merge pipeline
        "ground_range_max": float(ground_range[-1]) if ground_range is not None and len(ground_range) > 0 else 3000.0,
        "range_resolution": float(ground_range[1] - ground_range[0]) if ground_range is not None and len(ground_range) > 1 else 7.5,
        # Projected data (if do_projection=True)
        "projected_intensity": projected_intensity,
        "grid_params": grid_params,
        # Timings
        "timings": timings,
        "success": True,
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

    elapsed = time.perf_counter() - t0_total
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
