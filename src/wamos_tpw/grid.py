#! /usr/bin/env python3
#
# Equirectangular grid computation and projection utilities for WAMOS radar data
#
# Jan-2026, Pat Welch, pat@mousebrains.com

"""
Equirectangular grid computation and projection utilities for merging radar frames.

This module provides functions for:

1. **Grid Computation** - Computing common grids that cover multiple radar frames
   - `compute_common_grid()` - From full per-radial lat/lon arrays
   - `compute_common_grid_from_stats()` - From pre-computed position statistics (faster)

2. **Frame Projection** - Projecting radar frames onto equirectangular grids
   - `project_frame_to_common_grid()` - Project a single frame with motion compensation

3. **Grid Remapping** - Remapping between different grid resolutions
   - `remap_to_common_grid()` - Remap from source grid to destination grid

The module automatically uses Numba JIT acceleration for `remap_to_common_grid`
when Numba is available, providing 3-5x speedup for larger grids.

Example::

    # Compute a common grid for multiple frames
    grid_params = compute_common_grid(latitudes, longitudes, max_ranges, resolutions)

    # Project frames onto the grid
    for frame in frames:
        frame_sum, frame_count = project_frame_to_common_grid(
            intensity, theta, ground_range, lats, lons, headings, grid_params
        )

    # Remap to a different grid resolution
    dst_sum, dst_count = remap_to_common_grid(
        intensity, count, src_x_edges, src_y_edges, dst_x_edges, dst_y_edges, n_x, n_y
    )
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_common_grid",
    "compute_common_grid_from_stats",
    "project_frame_to_common_grid",
    "remap_to_common_grid",
]

_DEG2M = 111_319.5  # meters per degree of latitude


# =============================================================================
# Grid Computation Functions
# =============================================================================


def compute_common_grid(
    latitudes: list[np.ndarray],
    longitudes: list[np.ndarray],
    max_ranges: list[float],
    range_resolutions: list[float],
    padding: float = 1.1,
    resolution_scale: float = 1.0,
) -> dict:
    """
    Compute a common equirectangular grid that covers all frames.

    Args:
        latitudes: List of per-radial latitude arrays
        longitudes: List of per-radial longitude arrays
        max_ranges: Maximum ground range per frame in meters
        range_resolutions: Range resolution per frame in meters
        padding: Multiplier for max range to add margin
        resolution_scale: Grid resolution multiplier (2.0 = 2x finer grid)

    Returns:
        Dictionary with grid parameters:
        - x_edges, y_edges: Grid edges in meters (centered)
        - grid_spacing: Cell size in meters
        - utm_zone, hemisphere: Coordinate system info (informational)
        - center_lat, center_lon: Grid center in degrees
        - ref_lat, ref_lon: Reference point for equirectangular projection
        - m_per_deg_lon: Meters per degree of longitude at ref_lat
    """
    # Get reference position (center of all data)
    all_lats = np.concatenate(latitudes)
    all_lons = np.concatenate(longitudes)
    ref_lat = float(np.mean(all_lats))
    ref_lon = float(np.mean(all_lons))

    # Equirectangular projection: convert lat/lon to meters
    m_per_deg_lon = _DEG2M * np.cos(np.deg2rad(ref_lat))
    all_x = (all_lons - ref_lon) * m_per_deg_lon
    all_y = (all_lats - ref_lat) * _DEG2M

    # Grid spacing from average range resolution, scaled by resolution_scale
    grid_spacing = float(np.mean(range_resolutions)) / resolution_scale

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

    # Convert center to lat/lon via equirectangular inverse
    center_lon = ref_lon + x_center / m_per_deg_lon
    center_lat = ref_lat + y_center / _DEG2M

    # UTM zone/hemisphere as informational metadata
    utm_zone = int((center_lon + 180) / 6) % 60 + 1
    hemisphere = "north" if center_lat >= 0 else "south"

    # Center the edges for output
    x_edges_centered = x_edges - x_center
    y_edges_centered = y_edges - y_center

    return {
        "x_edges": x_edges_centered,
        "y_edges": y_edges_centered,
        "x_edges_abs": x_edges,
        "y_edges_abs": y_edges,
        "grid_spacing": grid_spacing,
        "utm_zone": utm_zone,
        "hemisphere": hemisphere,
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "m_per_deg_lon": m_per_deg_lon,
        "n_x": n_x,
        "n_y": n_y,
    }


def compute_common_grid_from_stats(
    position_stats: list[dict],
    max_ranges: list[float],
    range_resolutions: list[float],
    padding: float = 1.1,
    resolution_scale: float = 1.0,
) -> dict:
    """
    Compute a common equirectangular grid from per-frame position statistics.

    Optimized version of compute_common_grid that uses pre-computed summary
    statistics instead of full per-radial arrays, reducing data transfer
    by ~120x (from ~14,400 values to ~120 for a typical 40-frame window).

    Args:
        position_stats: List of dicts with per-frame position statistics:
            - lat_min, lat_max, lat_mean: Latitude summary stats
            - lon_min, lon_max, lon_mean: Longitude summary stats
        max_ranges: Maximum ground range per frame in meters
        range_resolutions: Range resolution per frame in meters
        padding: Multiplier for max range to add margin
        resolution_scale: Grid resolution multiplier (2.0 = 2x finer grid)

    Returns:
        Dictionary with grid parameters (same as compute_common_grid)
    """
    if not position_stats:
        raise ValueError("position_stats cannot be empty")

    # Compute reference position from per-frame means
    # Mean of means equals true mean when all frames have equal counts
    ref_lat = float(np.mean([s["lat_mean"] for s in position_stats]))
    ref_lon = float(np.mean([s["lon_mean"] for s in position_stats]))

    # Equirectangular projection parameters
    m_per_deg_lon = _DEG2M * np.cos(np.deg2rad(ref_lat))

    # Get lat/lon extremes from per-frame min/max
    lat_min = min(s["lat_min"] for s in position_stats)
    lat_max = max(s["lat_max"] for s in position_stats)
    lon_min = min(s["lon_min"] for s in position_stats)
    lon_max = max(s["lon_max"] for s in position_stats)

    # Convert extremes to meters relative to reference
    x_data_min = (lon_min - ref_lon) * m_per_deg_lon
    x_data_max = (lon_max - ref_lon) * m_per_deg_lon
    y_data_min = (lat_min - ref_lat) * _DEG2M
    y_data_max = (lat_max - ref_lat) * _DEG2M

    # Grid spacing from average range resolution, scaled by resolution_scale
    grid_spacing = float(np.mean(range_resolutions)) / resolution_scale

    # Grid extent: data extent + max radar range + padding
    max_range = float(np.max(max_ranges)) * padding

    x_min = x_data_min - max_range
    x_max = x_data_max + max_range
    y_min = y_data_min - max_range
    y_max = y_data_max + max_range

    # Create bin edges aligned to grid spacing
    n_x = int(np.ceil((x_max - x_min) / grid_spacing))
    n_y = int(np.ceil((y_max - y_min) / grid_spacing))

    x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
    y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)

    # Compute grid center
    x_center = (x_edges[0] + x_edges[-1]) / 2
    y_center = (y_edges[0] + y_edges[-1]) / 2

    # Convert center to lat/lon via equirectangular inverse
    center_lon = ref_lon + x_center / m_per_deg_lon
    center_lat = ref_lat + y_center / _DEG2M

    # UTM zone/hemisphere as informational metadata
    utm_zone = int((center_lon + 180) / 6) % 60 + 1
    hemisphere = "north" if center_lat >= 0 else "south"

    # Center the edges for output
    x_edges_centered = x_edges - x_center
    y_edges_centered = y_edges - y_center

    return {
        "x_edges": x_edges_centered,
        "y_edges": y_edges_centered,
        "x_edges_abs": x_edges,
        "y_edges_abs": y_edges,
        "grid_spacing": grid_spacing,
        "utm_zone": utm_zone,
        "hemisphere": hemisphere,
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "m_per_deg_lon": m_per_deg_lon,
        "n_x": n_x,
        "n_y": n_y,
    }


# =============================================================================
# Frame Projection Functions
# =============================================================================


def _project_frame_numpy(
    intensity: np.ndarray,
    theta: np.ndarray,
    ground_range: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings: np.ndarray,
    grid_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy implementation of frame projection.

    Project a single frame onto a common equirectangular grid.

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
    ref_lat = grid_params["ref_lat"]
    ref_lon = grid_params["ref_lon"]
    m_per_deg_lon = grid_params["m_per_deg_lon"]
    x_edges_abs = grid_params["x_edges_abs"]
    y_edges_abs = grid_params["y_edges_abs"]
    grid_spacing = grid_params["grid_spacing"]
    n_x = grid_params["n_x"]
    n_y = grid_params["n_y"]

    # Convert ship positions to equirectangular meters
    ship_x = (longitudes - ref_lon) * m_per_deg_lon
    ship_y = (latitudes - ref_lat) * _DEG2M

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
    x_origin = x_edges_abs[0]
    y_origin = y_edges_abs[0]
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


# Default to numpy implementation
# Numba version available below but benchmarks show numpy is faster for typical frame sizes
project_frame_to_common_grid = _project_frame_numpy


# =============================================================================
# Grid Remapping Functions
# =============================================================================


def remap_to_common_grid(
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
        src_x_edges: Source grid x edges (in absolute equirectangular meters)
        src_y_edges: Source grid y edges (in absolute equirectangular meters)
        dst_x_edges: Destination grid x edges (in absolute equirectangular meters)
        dst_y_edges: Destination grid y edges (in absolute equirectangular meters)
        dst_n_x: Destination grid x dimension
        dst_n_y: Destination grid y dimension

    Returns:
        Tuple of (intensity_sum, count) arrays in destination grid
    """
    # Early exit if source grid doesn't overlap destination grid
    if (
        src_x_edges[-1] < dst_x_edges[0]
        or src_x_edges[0] > dst_x_edges[-1]
        or src_y_edges[-1] < dst_y_edges[0]
        or src_y_edges[0] > dst_y_edges[-1]
    ):
        return (
            np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
            np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
        )

    # Get destination grid spacing and origin
    dst_dx = dst_x_edges[1] - dst_x_edges[0]
    dst_dy = dst_y_edges[1] - dst_y_edges[0]
    dst_x0 = dst_x_edges[0]
    dst_y0 = dst_y_edges[0]
    inv_dst_dx = 1.0 / dst_dx
    inv_dst_dy = 1.0 / dst_dy

    # Compute source grid centers as 1D arrays (avoid meshgrid)
    src_x_centers = (src_x_edges[:-1] + src_x_edges[1:]) * 0.5
    src_y_centers = (src_y_edges[:-1] + src_y_edges[1:]) * 0.5

    # Compute destination indices for 1D arrays
    dst_ix_1d = ((src_x_centers - dst_x0) * inv_dst_dx).astype(np.int32)
    dst_iy_1d = ((src_y_centers - dst_y0) * inv_dst_dy).astype(np.int32)

    # Find valid x and y ranges (cells that map into destination grid)
    valid_x = (dst_ix_1d >= 0) & (dst_ix_1d < dst_n_x)
    valid_y = (dst_iy_1d >= 0) & (dst_iy_1d < dst_n_y)

    # Get indices of valid rows/columns
    valid_x_idx = np.where(valid_x)[0]
    valid_y_idx = np.where(valid_y)[0]

    if len(valid_x_idx) == 0 or len(valid_y_idx) == 0:
        return (
            np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
            np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
        )

    # Extract only the overlapping subregion of intensity/count
    y_start, y_end = valid_y_idx[0], valid_y_idx[-1] + 1
    x_start, x_end = valid_x_idx[0], valid_x_idx[-1] + 1

    sub_intensity = intensity[y_start:y_end, x_start:x_end]
    sub_dst_ix = dst_ix_1d[x_start:x_end]
    sub_dst_iy = dst_iy_1d[y_start:y_end]

    # Handle count array
    if count is not None:
        sub_count = count[y_start:y_end, x_start:x_end]
    else:
        sub_count = None

    # Find valid (non-NaN) cells in subregion
    valid_mask = ~np.isnan(sub_intensity)

    if not np.any(valid_mask):
        return (
            np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
            np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
        )

    # Build 2D destination indices using broadcasting (no meshgrid needed)
    # sub_dst_iy is (sub_n_y,), sub_dst_ix is (sub_n_x,)
    # We need linear index = iy * dst_n_x + ix for each cell
    sub_n_y, sub_n_x = sub_intensity.shape
    linear_base = sub_dst_iy[:, np.newaxis] * dst_n_x  # (sub_n_y, 1)
    # Broadcasting: (sub_n_y, 1) + (sub_n_x,) -> (sub_n_y, sub_n_x)
    linear_idx_2d = linear_base + sub_dst_ix

    # Extract valid values
    valid_linear_idx = linear_idx_2d[valid_mask]
    valid_intensity = sub_intensity[valid_mask]

    if sub_count is not None:
        valid_count = sub_count[valid_mask]
        weights = valid_intensity * valid_count
    else:
        valid_count = np.int32(1)
        weights = valid_intensity

    # Accumulate using bincount
    grid_size = dst_n_x * dst_n_y
    dst_sum = np.bincount(valid_linear_idx, weights=weights, minlength=grid_size).reshape(
        (dst_n_y, dst_n_x)
    )

    if sub_count is not None:
        dst_count = np.bincount(valid_linear_idx, weights=valid_count, minlength=grid_size).reshape(
            (dst_n_y, dst_n_x)
        )
    else:
        dst_count = np.bincount(valid_linear_idx, minlength=grid_size).reshape((dst_n_y, dst_n_x))

    return dst_sum.astype(np.float64), dst_count.astype(np.int32)


# Optional Numba acceleration for grid operations
# When numba is available, provides significant speedup over pure-numpy
try:
    import numba

    # =========================================================================
    # Numba-accelerated project_frame_to_common_grid
    # =========================================================================

    @numba.jit(nopython=True, cache=True)
    def _project_frame_numba_core(
        intensity: np.ndarray,
        sin_bearing: np.ndarray,
        cos_bearing: np.ndarray,
        ground_range: np.ndarray,
        ship_x: np.ndarray,
        ship_y: np.ndarray,
        x_origin: float,
        y_origin: float,
        inv_spacing: float,
        n_x: int,
        n_y: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated core projection loop.

        Avoids large intermediate arrays by computing coordinates inline.
        Sequential loop to handle accumulation safely.
        """
        grid_size = n_x * n_y
        frame_sum = np.zeros(grid_size, dtype=np.float64)
        frame_count = np.zeros(grid_size, dtype=np.int64)

        n_bearings, n_distances = intensity.shape

        for ib in range(n_bearings):
            sin_b = sin_bearing[ib]
            cos_b = cos_bearing[ib]
            sx = ship_x[ib]
            sy = ship_y[ib]

            for ir in range(n_distances):
                val = intensity[ib, ir]
                if np.isnan(val):
                    continue

                # Compute coordinates inline (no intermediate array)
                r = ground_range[ir]
                x = sin_b * r + sx
                y = cos_b * r + sy

                # Convert to grid indices
                x_idx = int((x - x_origin) * inv_spacing)
                y_idx = int((y - y_origin) * inv_spacing)

                # Check bounds and accumulate
                if 0 <= x_idx < n_x and 0 <= y_idx < n_y:
                    linear_idx = y_idx * n_x + x_idx
                    frame_sum[linear_idx] += val
                    frame_count[linear_idx] += 1

        return frame_sum.reshape((n_y, n_x)), frame_count.reshape((n_y, n_x))

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _project_frame_numba_core_parallel(
        intensity: np.ndarray,
        sin_bearing: np.ndarray,
        cos_bearing: np.ndarray,
        ground_range: np.ndarray,
        ship_x: np.ndarray,
        ship_y: np.ndarray,
        x_origin: float,
        y_origin: float,
        inv_spacing: float,
        n_x: int,
        n_y: int,
        n_threads: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parallel Numba projection using thread-local accumulators.

        Each thread accumulates into its own grid, then merged at the end.
        """
        grid_size = n_x * n_y
        n_bearings, n_distances = intensity.shape

        # Thread-local accumulators (n_threads x grid_size)
        local_sums = np.zeros((n_threads, grid_size), dtype=np.float64)
        local_counts = np.zeros((n_threads, grid_size), dtype=np.int64)

        # Process bearings in parallel
        for ib in numba.prange(n_bearings):
            # Get thread ID (bearings are distributed across threads)
            tid = ib % n_threads

            sin_b = sin_bearing[ib]
            cos_b = cos_bearing[ib]
            sx = ship_x[ib]
            sy = ship_y[ib]

            for ir in range(n_distances):
                val = intensity[ib, ir]
                if np.isnan(val):
                    continue

                r = ground_range[ir]
                x = sin_b * r + sx
                y = cos_b * r + sy

                x_idx = int((x - x_origin) * inv_spacing)
                y_idx = int((y - y_origin) * inv_spacing)

                if 0 <= x_idx < n_x and 0 <= y_idx < n_y:
                    linear_idx = y_idx * n_x + x_idx
                    local_sums[tid, linear_idx] += val
                    local_counts[tid, linear_idx] += 1

        # Merge thread-local results
        frame_sum = np.zeros(grid_size, dtype=np.float64)
        frame_count = np.zeros(grid_size, dtype=np.int64)
        for tid in range(n_threads):
            for i in range(grid_size):
                frame_sum[i] += local_sums[tid, i]
                frame_count[i] += local_counts[tid, i]

        return frame_sum.reshape((n_y, n_x)), frame_count.reshape((n_y, n_x))

    def _project_frame_numba(
        intensity: np.ndarray,
        theta: np.ndarray,
        ground_range: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        headings: np.ndarray,
        grid_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated frame projection."""
        ref_lat = grid_params["ref_lat"]
        ref_lon = grid_params["ref_lon"]
        m_per_deg_lon = grid_params["m_per_deg_lon"]
        x_edges_abs = grid_params["x_edges_abs"]
        y_edges_abs = grid_params["y_edges_abs"]
        grid_spacing = grid_params["grid_spacing"]
        n_x = grid_params["n_x"]
        n_y = grid_params["n_y"]

        # Convert ship positions to equirectangular meters
        ship_x = (longitudes - ref_lon) * m_per_deg_lon
        ship_y = (latitudes - ref_lat) * _DEG2M

        # Compute earth bearing for each radial
        earth_bearing_rad = np.deg2rad((theta + headings) % 360)
        sin_bearing = np.sin(earth_bearing_rad)
        cos_bearing = np.cos(earth_bearing_rad)

        x_origin = x_edges_abs[0]
        y_origin = y_edges_abs[0]
        inv_spacing = 1.0 / grid_spacing

        # Ensure arrays are contiguous float64 for numba
        intensity_f64 = np.ascontiguousarray(intensity, dtype=np.float64)
        ground_range_f64 = np.ascontiguousarray(ground_range, dtype=np.float64)
        ship_x_f64 = np.ascontiguousarray(ship_x, dtype=np.float64)
        ship_y_f64 = np.ascontiguousarray(ship_y, dtype=np.float64)
        sin_bearing_f64 = np.ascontiguousarray(sin_bearing, dtype=np.float64)
        cos_bearing_f64 = np.ascontiguousarray(cos_bearing, dtype=np.float64)

        n_bearings = intensity.shape[0]

        # Use parallel version for larger frames, sequential for smaller
        if n_bearings >= 180:
            n_threads = min(8, n_bearings // 45)  # ~45 bearings per thread minimum
            frame_sum, frame_count = _project_frame_numba_core_parallel(
                intensity_f64,
                sin_bearing_f64,
                cos_bearing_f64,
                ground_range_f64,
                ship_x_f64,
                ship_y_f64,
                x_origin,
                y_origin,
                inv_spacing,
                n_x,
                n_y,
                n_threads,
            )
        else:
            frame_sum, frame_count = _project_frame_numba_core(
                intensity_f64,
                sin_bearing_f64,
                cos_bearing_f64,
                ground_range_f64,
                ship_x_f64,
                ship_y_f64,
                x_origin,
                y_origin,
                inv_spacing,
                n_x,
                n_y,
            )

        return frame_sum, frame_count.astype(np.int32)

    # Note: Numba projection available as _project_frame_numba but benchmarks show
    # numpy is faster for typical frame sizes (360x512). Keep numpy as default.
    # For very large frames (>360x1024), numba may provide slight benefit.

    # =========================================================================
    # Numba-accelerated remap_to_common_grid
    # =========================================================================

    @numba.jit(nopython=True, parallel=True, cache=True)
    def _remap_numba_core(
        intensity: np.ndarray,
        count: np.ndarray,
        has_count: bool,
        dst_ix_1d: np.ndarray,
        dst_iy_1d: np.ndarray,
        dst_n_x: int,
        dst_n_y: int,
        y_start: int,
        x_start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated core remap loop."""
        grid_size = dst_n_x * dst_n_y
        dst_sum = np.zeros(grid_size, dtype=np.float64)
        dst_count_out = np.zeros(grid_size, dtype=np.float64)

        sub_n_y, sub_n_x = intensity.shape

        for iy in numba.prange(sub_n_y):
            dst_iy = dst_iy_1d[y_start + iy]
            for ix in range(sub_n_x):
                val = intensity[iy, ix]
                if np.isnan(val):
                    continue
                dst_ix = dst_ix_1d[x_start + ix]
                linear_idx = dst_iy * dst_n_x + dst_ix

                if has_count:
                    c = count[iy, ix]
                    dst_sum[linear_idx] += val * c
                    dst_count_out[linear_idx] += c
                else:
                    dst_sum[linear_idx] += val
                    dst_count_out[linear_idx] += 1

        return dst_sum.reshape((dst_n_y, dst_n_x)), dst_count_out.reshape((dst_n_y, dst_n_x))

    def _remap_numba(
        intensity: np.ndarray,
        count: np.ndarray | None,
        src_x_edges: np.ndarray,
        src_y_edges: np.ndarray,
        dst_x_edges: np.ndarray,
        dst_y_edges: np.ndarray,
        dst_n_x: int,
        dst_n_y: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated remap implementation."""
        # Early exit if no overlap
        if (
            src_x_edges[-1] < dst_x_edges[0]
            or src_x_edges[0] > dst_x_edges[-1]
            or src_y_edges[-1] < dst_y_edges[0]
            or src_y_edges[0] > dst_y_edges[-1]
        ):
            return (
                np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
                np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
            )

        dst_dx = dst_x_edges[1] - dst_x_edges[0]
        dst_dy = dst_y_edges[1] - dst_y_edges[0]
        dst_x0 = dst_x_edges[0]
        dst_y0 = dst_y_edges[0]

        src_x_centers = (src_x_edges[:-1] + src_x_edges[1:]) * 0.5
        src_y_centers = (src_y_edges[:-1] + src_y_edges[1:]) * 0.5

        dst_ix_1d = ((src_x_centers - dst_x0) / dst_dx).astype(np.int32)
        dst_iy_1d = ((src_y_centers - dst_y0) / dst_dy).astype(np.int32)

        valid_x = (dst_ix_1d >= 0) & (dst_ix_1d < dst_n_x)
        valid_y = (dst_iy_1d >= 0) & (dst_iy_1d < dst_n_y)

        valid_x_idx = np.where(valid_x)[0]
        valid_y_idx = np.where(valid_y)[0]

        if len(valid_x_idx) == 0 or len(valid_y_idx) == 0:
            return (
                np.zeros((dst_n_y, dst_n_x), dtype=np.float64),
                np.zeros((dst_n_y, dst_n_x), dtype=np.int32),
            )

        y_start, y_end = valid_y_idx[0], valid_y_idx[-1] + 1
        x_start, x_end = valid_x_idx[0], valid_x_idx[-1] + 1

        sub_intensity = intensity[y_start:y_end, x_start:x_end]

        if count is not None:
            sub_count = count[y_start:y_end, x_start:x_end].astype(np.float64)
            has_count = True
        else:
            sub_count = np.empty((0, 0), dtype=np.float64)
            has_count = False

        dst_sum, dst_count = _remap_numba_core(
            sub_intensity.astype(np.float64),
            sub_count,
            has_count,
            dst_ix_1d,
            dst_iy_1d,
            dst_n_x,
            dst_n_y,
            y_start,
            x_start,
        )

        return dst_sum, dst_count.astype(np.int32)

    # Replace the pure-numpy implementation with the numba version
    remap_to_common_grid = _remap_numba
    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False
