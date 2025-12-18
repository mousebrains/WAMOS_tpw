#! /usr/bin/env python3
#
# Streaming processing for WAMOS combined data
# Memory-efficient frame processing and gridding
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import WamosConfig


def load_file_metadata(fpath: str) -> tuple[dict | None, list]:
    """
    Load metadata from a single polar file (for parallel execution).

    Args:
        fpath: Path to polar file

    Returns:
        Tuple of (header_dict, list_of_frame_metadata) or (None, []) on error
    """
    from wamos_tpw.polarfile import PolarFile

    try:
        pf = PolarFile(fpath, metadata_only=True)
        return pf.header, list(pf.frame_metadata)
    except Exception as e:
        logging.warning(f"Failed to load metadata from {fpath}: {e}")
        return None, []


def compute_grid_bounds_from_metadata(
    file_list: list[str],
    config: "WamosConfig",
    radar_height: float | None,
    max_frames: int | None,
    workers: int = 8,
) -> tuple[list, float, float, float, float, float, np.ndarray, np.ndarray, list, list, list, list] | None:
    """
    Compute grid bounds from file metadata without loading frame data.

    This enables memory-efficient processing by determining the grid
    dimensions before loading actual frame intensity data.

    Args:
        file_list: List of file paths
        config: WamosConfig
        radar_height: Radar height override (meters)
        max_frames: Maximum frames to process
        workers: Number of threads for parallel metadata loading

    Returns:
        Tuple of (frame_metadata_list, x_min, x_max, y_min, y_max, max_range,
                  ship_x, ship_y, ship_speeds, ship_headings, wind_speeds, wind_dirs)
        or None if no valid metadata found.
    """
    # Earth radius for lat/lon to meters conversion
    EARTH_RADIUS = 6371000.0

    if not file_list:
        return None

    # Parallel metadata loading
    all_metadata = []
    first_header = None

    n_workers = min(workers, len(file_list))
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(load_file_metadata, file_list))

    # Combine results (maintain file order for consistent first_header)
    for header, metadata_list in results:
        if header is not None and first_header is None:
            first_header = header
        all_metadata.extend(metadata_list)

    if not all_metadata:
        return None

    if max_frames:
        all_metadata = all_metadata[:max_frames]

    # Get reference position (first frame)
    ref_lat = all_metadata[0].latitude or 0.0
    ref_lon = all_metadata[0].longitude or 0.0

    # Compute ship track in meters
    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    ship_x = []
    ship_y = []
    for meta in all_metadata:
        lat = meta.latitude or ref_lat
        lon = meta.longitude or ref_lon
        ship_x.append((lon - ref_lon) * meters_per_deg_lon)
        ship_y.append((lat - ref_lat) * meters_per_deg_lat)

    ship_x = np.array(ship_x)
    ship_y = np.array(ship_y)

    # Collect ship and wind data for polar plots
    ship_speeds = []
    ship_headings = []
    wind_speeds = []
    wind_dirs = []
    for meta in all_metadata:
        if meta.ship_speed is not None and meta.heading is not None:
            ship_speeds.append(meta.ship_speed)
            ship_headings.append(meta.heading)
        if meta.wind_speed is not None and meta.wind_direction is not None:
            wind_speeds.append(meta.wind_speed)
            wind_dirs.append(meta.wind_direction)

    # Estimate max radar range from header
    if first_header:
        fifo = first_header.get("FIFO", 752)  # samples in range
        sfreq_mhz = first_header.get("SFREQ", 20.0)  # sampling frequency in MHz
        sfreq_hz = sfreq_mhz * 1e6  # convert to Hz
        sdrng = first_header.get("SDRNG", 0)  # sample delay range
        c_air = 299_792_458.0 / 1.000273  # speed of light in air

        # Slant range calculation
        slant_max = (fifo / sfreq_hz) * c_air / 2 + sdrng

        # Convert to ground range if radar height available
        height = radar_height
        if height is None:
            height = config.radar.height
        if height is None and all_metadata:
            height = all_metadata[0].radar_height or all_metadata[0].wind_sensor_height

        if height and slant_max > height:
            max_range = np.sqrt(slant_max**2 - height**2)
        else:
            max_range = slant_max
    else:
        max_range = 4000.0  # Default fallback

    # Compute bounds with padding
    padding = 1.1
    max_range *= padding
    x_min = ship_x.min() - max_range
    x_max = ship_x.max() + max_range
    y_min = ship_y.min() - max_range
    y_max = ship_y.max() + max_range

    return (all_metadata, x_min, x_max, y_min, y_max, max_range, ship_x, ship_y,
            ship_speeds, ship_headings, wind_speeds, wind_dirs)


def process_single_frame(
    frame_idx: int,
    frame,
    theta,
    config: "WamosConfig",
    offset: float,
    shadow_start: float,
    shadow_end: float,
) -> None:
    """
    Process a single frame: deramp + destreak.

    Modifies frame in-place (sets corrected_intensity).
    Thread-safe as each frame is independent.

    Args:
        frame_idx: Index of frame within the theta object
        frame: Frame object
        theta: Theta object for bearing calculation
        config: WamosConfig
        offset: Bearing offset to apply (degrees)
        shadow_start: Shadow start angle (degrees)
        shadow_end: Shadow end angle (degrees)
    """
    from wamos_tpw.deramp import Deramp
    from wamos_tpw.destreak import Destreak

    # Deramp with refined bearing
    bearing_arr = theta.bearing_for_frame(frame_idx)
    bearing_arr_refined = (bearing_arr - offset) % 360
    deramp = Deramp(
        frame, config, bearing=bearing_arr_refined,
        shadow_start=shadow_start, shadow_end=shadow_end,
    )
    frame.deramped_intensity = deramp.corrected_intensity

    # Destreak (circular theta, no neighbors needed)
    ds = Destreak(None, frame, None, config)
    frame.corrected_intensity = ds.corrected_intensity
    frame.deramped_intensity = None  # Free memory


def grid_frame_streaming(
    frame, frame_idx: int, theta, bearing_obj, config: "WamosConfig",
    x_edges: np.ndarray, y_edges: np.ndarray,
    sum_total: np.ndarray, count_total: np.ndarray,
    ref_lat: float, ref_lon: float,
) -> None:
    """
    Grid a single frame into accumulator arrays (in-place).

    Uses memory-efficient processing by computing coordinates and
    accumulating into pre-allocated arrays.

    Args:
        frame: Frame object
        frame_idx: Index of frame within theta/bearing objects
        theta: Theta object for shadow mask
        bearing_obj: Bearing object for coordinate calculation
        config: WamosConfig
        x_edges: Grid x bin edges
        y_edges: Grid y bin edges
        sum_total: Accumulator for intensity sum (modified in-place)
        count_total: Accumulator for point counts (modified in-place)
        ref_lat: Reference latitude
        ref_lon: Reference longitude
    """
    EARTH_RADIUS = 6371000.0

    # Get frame coordinates in earth frame
    x_rel, y_rel = bearing_obj.xy_earth(frame_idx)

    # Get ship position offset
    meta = frame.metadata
    frame_lat = meta.latitude or ref_lat
    frame_lon = meta.longitude or ref_lon

    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    ship_y = (frame_lat - ref_lat) * meters_per_deg_lat
    ship_x = (frame_lon - ref_lon) * meters_per_deg_lon

    # Add ship offset (broadcast to 2D)
    x_earth = x_rel + ship_x
    y_earth = y_rel + ship_y
    del x_rel, y_rel

    # Get intensity
    intensity = (
        frame.corrected_intensity
        if frame.corrected_intensity is not None
        else frame.deramped_intensity
        if frame.deramped_intensity is not None
        else frame.intensity
    )

    # Get shadow mask
    shadow_mask_1d = theta.in_shadow(frame_idx)
    n_bearings, n_distances = intensity.shape

    # Flatten and compute bin indices
    x_flat = x_earth.ravel()
    del x_earth
    y_flat = y_earth.ravel()
    del y_earth

    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    x_idx = np.searchsorted(x_edges, x_flat) - 1
    del x_flat
    y_idx = np.searchsorted(y_edges, y_flat) - 1
    del y_flat

    # Shadow mask (broadcast)
    shadow_flat = np.broadcast_to(shadow_mask_1d[:, np.newaxis], (n_bearings, n_distances)).ravel()

    # Valid mask
    valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y) & ~shadow_flat
    del shadow_flat

    x_idx_valid = x_idx[valid]
    del x_idx
    y_idx_valid = y_idx[valid]
    del y_idx
    values_valid = intensity.ravel()[valid].astype(np.float64)
    del valid

    # Accumulate
    np.add.at(sum_total, (y_idx_valid, x_idx_valid), values_valid)
    np.add.at(count_total, (y_idx_valid, x_idx_valid), 1)


def normalize_frames(corrected: list) -> list:
    """
    Normalize corrected frames to [0, 1] range.

    Uses memory-efficient reservoir sampling to estimate percentiles
    instead of concatenating all values into a single array.

    Args:
        corrected: List of corrected intensity arrays

    Returns:
        List of normalized arrays
    """
    if not corrected:
        return corrected

    # Memory-efficient percentile estimation using reservoir sampling
    # Sample ~100k values total across all frames to estimate percentiles
    max_samples = 100_000
    total_values = sum(c.size for c in corrected)
    sample_rate = min(1.0, max_samples / total_values) if total_values > 0 else 1.0

    # Collect samples from each frame
    samples = []
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    for c in corrected:
        flat = c.ravel()
        valid_mask = np.isfinite(flat)
        valid_values = flat[valid_mask]
        if len(valid_values) > 0:
            n_samples = max(1, int(len(valid_values) * sample_rate))
            if n_samples < len(valid_values):
                indices = rng.choice(len(valid_values), n_samples, replace=False)
                samples.append(valid_values[indices])
            else:
                samples.append(valid_values)

    if not samples:
        return corrected

    all_samples = np.concatenate(samples)
    if len(all_samples) == 0:
        return corrected

    vmin = np.percentile(all_samples, 1)
    vmax = np.percentile(all_samples, 99)

    # Free samples memory
    del samples, all_samples

    if vmax <= vmin:
        return corrected

    # Normalize each frame in-place where possible
    normalized = []
    for c in corrected:
        norm = (c - vmin) / (vmax - vmin)
        np.clip(norm, 0, 1, out=norm)
        normalized.append(norm)

    return normalized
