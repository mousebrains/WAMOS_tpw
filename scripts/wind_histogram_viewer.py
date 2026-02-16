#!/usr/bin/env python3
"""
Wind-relative intensity viewer.

Plots average intensity vs wind-relative angle, aggregated across all frames
for specified range bins. Uses adjusted theta values and time-varying
interpolated wind information.

The x-axis shows angle relative to wind direction:
    - 0° = upwind (looking into the wind)
    - ±90° = crosswind
    - ±180° = downwind (looking away from wind)

The y-axis shows average intensity across all frames and specified range bins.

Features:
    - Configurable range bins for averaging
    - Per-radial wind direction interpolation
    - Shadow detection and theta adjustment
    - Multiple range bin selections can be plotted together

Usage:
    python wind_histogram_viewer.py 20220405 20220406 /path/to/POLAR --shadow --adjust shift_scale
    python wind_histogram_viewer.py 20220405 20220406 /path/to/POLAR --range-bins 200-400 500-700

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw.combine import Combine  # noqa: E402
from wamos_tpw.multi_theta import MultiTheta as Theta  # noqa: E402

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame(fn: str) -> tuple | None:
    """
    Load a single frame.

    Args:
        fn: Filename to load

    Returns:
        Tuple of (frame, filename) or None on error
    """
    try:
        frame = PolarFile(fn).frame()
        return frame, fn
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def print_progress(current: int, total: int, width: int = 40, prefix: str = "Progress") -> None:
    """Print a progress bar that updates in place."""
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - 1)
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()


def wrap_angle_180(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-180, 180) range."""
    return ((angle + 180) % 360) - 180


def parse_range_bins(range_str: str) -> tuple[int, int]:
    """Parse a range bin specification like '200-400' into (start, end)."""
    parts = range_str.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}")
    return int(parts[0]), int(parts[1])


def is_in_shadow(
    theta: float,
    shadow_leading: float | None,
    shadow_trailing: float | None,
) -> bool:
    """Check if a theta value is within the shadow region."""
    if shadow_leading is None or shadow_trailing is None:
        return False
    # Shadow region is from leading to trailing (clockwise)
    if shadow_leading <= shadow_trailing:
        return shadow_leading <= theta <= shadow_trailing
    else:
        # Wraparound case
        return theta >= shadow_leading or theta <= shadow_trailing


def smooth_array(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average smoothing to an array.

    Uses uniform weights and handles edges by using smaller windows.

    Args:
        arr: Input array
        window: Window size for smoothing

    Returns:
        Smoothed array of same length
    """
    if window < 2:
        return arr.copy()

    result = np.zeros_like(arr)
    n = len(arr)
    half = window // 2

    for i in range(n):
        # Determine window bounds (handle edges)
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        # Only average over valid (non-zero count) values
        valid_mask = np.isfinite(arr[lo:hi]) & (arr[lo:hi] != 0)
        if np.any(valid_mask):
            result[i] = np.mean(arr[lo:hi][valid_mask])
        else:
            result[i] = arr[i]

    return result


def compute_wind_relative_intensity_new(
    frames: list,
    theta: Theta,
    destreaked_intensities: list[np.ndarray],
    range_bins: list[tuple[int, int]],
    angle_bin_size: float = 2.0,
    range_bin_size: float = 50.0,
    smooth_window: int = 10,
) -> dict:
    """
    Compute average intensity vs wind-relative angle and vs range.

    Uses the new API with Frame objects and multi-frame Theta.

    Args:
        frames: List of Frame objects
        theta: Multi-frame Theta object for bearing calculation
        destreaked_intensities: List of destreaked intensity arrays
        range_bins: List of (start_bin, end_bin) tuples for angle plot
        angle_bin_size: Size of angle bins in degrees
        range_bin_size: Size of range bins in meters for range plot
        smooth_window: Window size for smoothing median (in range bins)

    Returns:
        Dictionary with angle and range data.
    """
    # Create angle bins from -180 to 180
    angle_edges = np.arange(-180, 180 + angle_bin_size, angle_bin_size)
    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    n_angle_bins = len(angle_centers)

    # Determine max slant range across all frames
    max_range_m = 0.0
    for frame in frames:
        slant_ranges = frame.slant_range()
        frame_max_slant = float(np.max(slant_ranges))
        max_range_m = max(max_range_m, frame_max_slant)

    # Create range bins for the range plot
    range_edges = np.arange(0, max_range_m + range_bin_size, range_bin_size)
    range_centers = (range_edges[:-1] + range_edges[1:]) / 2
    n_range_bins = len(range_centers)

    # Collect wind and ship statistics
    all_wind_speeds: list[float] = []
    all_wind_dirs: list[float] = []
    all_ship_speeds: list[float] = []
    all_ship_headings: list[float] = []

    # =========================================================================
    # FIRST PASS: Collect intensity vs range to compute median per range bin
    # =========================================================================
    range_intensities: list[list[float]] = [[] for _ in range(n_range_bins)]

    n_frames = len(frames)
    for frame_idx in range(n_frames):
        frame = frames[frame_idx]
        intensity = destreaked_intensities[frame_idx]
        slant_ranges = frame.slant_range()

        # Get shadow mask for this frame
        shadow_mask = theta.in_shadow(frame_idx)

        # Get wind and ship from metadata
        meta = frame.metadata
        wind_speed = meta.wind_speed or 0.0
        wind_dir = meta.wind_direction or 0.0
        ship_speed = meta.ship_speed or 0.0
        ship_heading = meta.heading or 0.0

        all_wind_speeds.append(wind_speed)
        all_wind_dirs.append(wind_dir)
        all_ship_speeds.append(ship_speed)
        all_ship_headings.append(ship_heading)

        # Get bearing values for this frame
        bearing = theta.bearing_for_frame(frame_idx)

        # Process each radial (excluding shadow)
        for radial_idx in range(len(bearing)):
            if shadow_mask[radial_idx]:
                continue

            # Bin intensity by slant range
            for range_idx in range(intensity.shape[1]):
                slant_range = slant_ranges[range_idx]
                range_bin_idx = int(slant_range / range_bin_size)
                range_bin_idx = min(max(0, range_bin_idx), n_range_bins - 1)

                val = intensity[radial_idx, range_idx]
                if np.isfinite(val):
                    range_intensities[range_bin_idx].append(float(val))

    # Compute mean and median per range bin
    range_means = np.zeros(n_range_bins)
    range_medians = np.zeros(n_range_bins)
    range_stds = np.zeros(n_range_bins)
    range_counts = np.zeros(n_range_bins, dtype=int)

    for i, values in enumerate(range_intensities):
        if len(values) > 0:
            range_means[i] = np.mean(values)
            range_medians[i] = np.median(values)
            range_stds[i] = np.std(values)
            range_counts[i] = len(values)

    # Apply smoothing to median
    range_medians_smoothed = smooth_array(range_medians, smooth_window)

    # =========================================================================
    # SECOND PASS: Compute range-corrected intensity vs wind-relative angle
    # =========================================================================
    combined_angle_intensities: list[list[float]] = [[] for _ in range(n_angle_bins)]

    # Per-range-selection for display
    first_slant_ranges = frames[0].slant_range()
    angle_intensities: dict[str, list[list[float]]] = {}
    for start_bin, end_bin in range_bins:
        start_m = float(first_slant_ranges[min(start_bin, len(first_slant_ranges) - 1)])
        end_m = float(first_slant_ranges[min(end_bin - 1, len(first_slant_ranges) - 1)])
        label = f"{start_m:.0f}-{end_m:.0f}m"
        angle_intensities[label] = [[] for _ in range(n_angle_bins)]

    for frame_idx in range(n_frames):
        frame = frames[frame_idx]
        intensity = destreaked_intensities[frame_idx]
        slant_ranges = frame.slant_range()

        # Get shadow mask and bearing
        shadow_mask = theta.in_shadow(frame_idx)
        bearing = theta.bearing_for_frame(frame_idx)

        # Get wind direction from metadata (per-frame)
        wind_dir = frame.metadata.wind_direction or 0.0

        # Calculate wind-relative angle
        wind_relative = wrap_angle_180(bearing - wind_dir)

        # Process each radial (excluding shadow)
        for radial_idx in range(len(bearing)):
            if shadow_mask[radial_idx]:
                continue

            angle = wind_relative[radial_idx]
            angle_bin_idx = int((angle + 180) / angle_bin_size)
            angle_bin_idx = min(max(0, angle_bin_idx), n_angle_bins - 1)

            # Collect ALL range bins for combined cosine fit
            for range_idx in range(intensity.shape[1]):
                val = intensity[radial_idx, range_idx]
                if np.isfinite(val):
                    slant_range = slant_ranges[range_idx]
                    range_bin_idx_for_correction = int(slant_range / range_bin_size)
                    range_bin_idx_for_correction = min(
                        max(0, range_bin_idx_for_correction), n_range_bins - 1
                    )
                    median_val = range_medians_smoothed[range_bin_idx_for_correction]
                    corrected = float(val) - median_val
                    combined_angle_intensities[angle_bin_idx].append(corrected)

            # For display: also collect per range selection
            for start_bin, end_bin in range_bins:
                start_m = float(slant_ranges[min(start_bin, len(slant_ranges) - 1)])
                end_m = float(slant_ranges[min(end_bin - 1, len(slant_ranges) - 1)])
                label = f"{start_m:.0f}-{end_m:.0f}m"

                actual_start = max(0, start_bin)
                actual_end = min(intensity.shape[1], end_bin)

                if actual_start >= actual_end:
                    continue

                corrected_values: list[float] = []
                for range_idx in range(actual_start, actual_end):
                    val = intensity[radial_idx, range_idx]
                    if np.isfinite(val):
                        slant_range = slant_ranges[range_idx]
                        range_bin_idx_for_correction = int(slant_range / range_bin_size)
                        range_bin_idx_for_correction = min(
                            max(0, range_bin_idx_for_correction), n_range_bins - 1
                        )
                        median_val = range_medians_smoothed[range_bin_idx_for_correction]
                        corrected_values.append(float(val) - median_val)

                if corrected_values:
                    mean_corrected = float(np.mean(corrected_values))
                    angle_intensities[label][angle_bin_idx].append(mean_corrected)

    # Compute combined mean for cosine fit across all ranges
    combined_means = np.zeros(n_angle_bins)
    combined_counts = np.zeros(n_angle_bins, dtype=int)
    for i, values in enumerate(combined_angle_intensities):
        if len(values) > 0:
            combined_means[i] = np.mean(values)
            combined_counts[i] = len(values)

    # Compute statistics for angle plot (per range selection for display)
    angle_means: dict[str, np.ndarray] = {}
    angle_stds: dict[str, np.ndarray] = {}
    angle_counts: dict[str, np.ndarray] = {}

    for label, bin_values in angle_intensities.items():
        means = np.zeros(n_angle_bins)
        stds = np.zeros(n_angle_bins)
        counts = np.zeros(n_angle_bins, dtype=int)

        for i, values in enumerate(bin_values):
            if len(values) > 0:
                means[i] = np.mean(values)
                stds[i] = np.std(values)
                counts[i] = len(values)

        angle_means[label] = means
        angle_stds[label] = stds
        angle_counts[label] = counts

    # Compute wind and ship statistics
    wind_speed_mean = float(np.mean(all_wind_speeds)) if all_wind_speeds else 0.0
    wind_speed_std = float(np.std(all_wind_speeds)) if all_wind_speeds else 0.0
    wind_dir_mean = float(np.mean(all_wind_dirs)) if all_wind_dirs else 0.0
    wind_dir_std = float(np.std(all_wind_dirs)) if all_wind_dirs else 0.0
    ship_speed_mean = float(np.mean(all_ship_speeds)) if all_ship_speeds else 0.0
    ship_speed_std = float(np.std(all_ship_speeds)) if all_ship_speeds else 0.0
    ship_heading_mean = float(np.mean(all_ship_headings)) if all_ship_headings else 0.0
    ship_heading_std = float(np.std(all_ship_headings)) if all_ship_headings else 0.0

    # Fit cosine to combined data (all range bins)
    from scipy.optimize import curve_fit

    def cosine_func(theta_rad: np.ndarray, a: float, phi: float, b: float) -> np.ndarray:
        return a * np.cos(theta_rad - phi) + b

    valid = np.isfinite(combined_means) & (combined_counts > 0)
    cosine_params: dict = {"a": 0.0, "b": 0.0, "phi": 0.0, "r_squared": 0.0}

    if np.sum(valid) >= 4:
        theta_valid = np.deg2rad(angle_centers[valid])
        y_valid = combined_means[valid]

        try:
            a0 = (np.max(y_valid) - np.min(y_valid)) / 2
            b0 = np.mean(y_valid)
            phi0 = 0.0

            popt, _ = curve_fit(
                cosine_func,
                theta_valid,
                y_valid,
                p0=[a0, phi0, b0],
                bounds=([-np.inf, -np.pi, -np.inf], [np.inf, np.pi, np.inf]),
                maxfev=5000,
            )
            a, phi, b = popt

            y_fit = cosine_func(theta_valid, a, phi, b)
            ss_res = np.sum((y_valid - y_fit) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            cosine_params = {
                "a": float(a),
                "b": float(b),
                "phi": float(np.rad2deg(phi)),
                "r_squared": float(r_squared),
            }
        except Exception:
            pass

    # =========================================================================
    # THIRD PASS: Compute per-range std after median and cosine subtraction
    # =========================================================================
    range_adjusted_intensities: list[list[float]] = [[] for _ in range(n_range_bins)]

    for frame_idx in range(n_frames):
        frame = frames[frame_idx]
        intensity = destreaked_intensities[frame_idx]
        slant_ranges = frame.slant_range()

        shadow_mask = theta.in_shadow(frame_idx)
        bearing = theta.bearing_for_frame(frame_idx)
        wind_dir = frame.metadata.wind_direction or 0.0
        wind_relative = wrap_angle_180(bearing - wind_dir)

        for radial_idx in range(len(bearing)):
            if shadow_mask[radial_idx]:
                continue

            # Compute cosine correction for this radial
            angle_rad = np.deg2rad(wind_relative[radial_idx])
            phi_rad = np.deg2rad(cosine_params["phi"])
            cosine_correction = (
                cosine_params["a"] * np.cos(angle_rad - phi_rad) + cosine_params["b"]
            )

            for range_idx in range(intensity.shape[1]):
                val = intensity[radial_idx, range_idx]
                if np.isfinite(val):
                    slant_range = slant_ranges[range_idx]
                    range_bin_idx_adj = int(slant_range / range_bin_size)
                    range_bin_idx_adj = min(max(0, range_bin_idx_adj), n_range_bins - 1)

                    median_val = range_medians_smoothed[range_bin_idx_adj]
                    adjusted = float(val) - median_val - cosine_correction
                    range_adjusted_intensities[range_bin_idx_adj].append(adjusted)

    # Compute per-range std after adjustment
    range_adjusted_stds = np.zeros(n_range_bins)
    for i, values in enumerate(range_adjusted_intensities):
        if len(values) > 1:
            range_adjusted_stds[i] = np.std(values)

    # Smooth the adjusted stds
    range_adjusted_stds_smoothed = smooth_array(range_adjusted_stds, smooth_window)

    return {
        "angle_centers": angle_centers,
        "angle_means": angle_means,
        "angle_stds": angle_stds,
        "angle_counts": angle_counts,
        "combined_angle_means": combined_means,
        "combined_angle_counts": combined_counts,
        "cosine_params": cosine_params,
        "range_centers": range_centers,
        "range_means": range_means,
        "range_medians": range_medians,
        "range_medians_smoothed": range_medians_smoothed,
        "range_stds": range_stds,
        "range_adjusted_stds": range_adjusted_stds,
        "range_adjusted_stds_smoothed": range_adjusted_stds_smoothed,
        "range_counts": range_counts,
        "smooth_window": smooth_window,
        # Wind and ship statistics
        "wind_speed_mean": wind_speed_mean,
        "wind_speed_std": wind_speed_std,
        "wind_dir_mean": wind_dir_mean,
        "wind_dir_std": wind_dir_std,
        "ship_speed_mean": ship_speed_mean,
        "ship_speed_std": ship_speed_std,
        "ship_heading_mean": ship_heading_mean,
        "ship_heading_std": ship_heading_std,
        "n_frames": n_frames,
    }


def compute_averaged_earth_intensity_new(
    frames: list,
    theta: Theta,
    combine: Combine,
    destreaked_intensities: list[np.ndarray],
    intensity_data: dict,
    cosine_params: dict,
    grid_size: int = 200,
) -> dict:
    """
    Compute averaged intensity maps in earth coordinates across all frames.

    Uses the new API with Frame objects, Theta, and Combine.

    Args:
        frames: List of Frame objects
        theta: Multi-frame Theta object
        combine: Combine object for coordinate transformation
        destreaked_intensities: List of destreaked intensity arrays
        intensity_data: Output from compute_wind_relative_intensity_new()
        cosine_params: Cosine fit parameters for wind correction
        grid_size: Grid resolution for earth coordinate averaging

    Returns:
        Dictionary with averaged intensity grids
    """
    # Get lat/lon bounds from Combine
    all_lats: list[float] = []
    all_lons: list[float] = []

    for frame_idx in range(len(frames)):
        lat, lon = combine.latlon(frame_idx)
        all_lats.extend(lat.flatten().tolist())
        all_lons.extend(lon.flatten().tolist())

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    # Add small margin
    lon_margin = (lon_max - lon_min) * 0.02
    lat_margin = (lat_max - lat_min) * 0.02
    lon_min -= lon_margin
    lon_max += lon_margin
    lat_min -= lat_margin
    lat_max += lat_margin

    # Create grid
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    # Accumulators for averaging
    destreak_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    adjusted_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size), dtype=np.int64)

    range_bin_size = intensity_data["range_centers"][1] - intensity_data["range_centers"][0]
    range_medians_smoothed = intensity_data["range_medians_smoothed"]
    range_adjusted_stds_smoothed = intensity_data["range_adjusted_stds_smoothed"]
    range_centers = intensity_data["range_centers"]
    n_range_bins = len(range_centers)

    # Get cosine parameters
    a = cosine_params.get("a", 0.0)
    phi_deg = cosine_params.get("phi", 0.0)
    b = cosine_params.get("b", 0.0)

    n_frames = len(frames)
    for frame_idx in range(n_frames):
        frame = frames[frame_idx]
        intensity = destreaked_intensities[frame_idx]
        slant_ranges = frame.slant_range()

        # Get lat/lon for this frame
        earth_lat, earth_lon = combine.latlon(frame_idx)

        # Get shadow mask and bearing
        shadow_mask = theta.in_shadow(frame_idx)
        bearing = theta.bearing_for_frame(frame_idx)

        # Get wind direction
        wind_dir = frame.metadata.wind_direction or 0.0
        wind_relative = wrap_angle_180(bearing - wind_dir)

        # Accumulate onto grid (excluding shadow)
        for bearing_idx in range(len(bearing)):
            if shadow_mask[bearing_idx]:
                continue

            # Compute cosine correction for this radial
            wind_rel_rad = np.deg2rad(wind_relative[bearing_idx])
            phi_rad = np.deg2rad(phi_deg)
            cosine_correction = a * np.cos(wind_rel_rad - phi_rad) + b

            for range_idx in range(intensity.shape[1]):
                lon_val = earth_lon[bearing_idx, range_idx]
                lat_val = earth_lat[bearing_idx, range_idx]

                # Find grid cell
                lon_idx = int((lon_val - lon_min) / (lon_max - lon_min) * grid_size)
                lat_idx = int((lat_val - lat_min) / (lat_max - lat_min) * grid_size)

                # Clamp to grid bounds
                lon_idx = max(0, min(grid_size - 1, lon_idx))
                lat_idx = max(0, min(grid_size - 1, lat_idx))

                destreak_val = intensity[bearing_idx, range_idx]

                # Compute adjusted intensity
                slant_range = slant_ranges[range_idx]
                range_bin_idx = int(slant_range / range_bin_size)
                range_bin_idx = min(max(0, range_bin_idx), n_range_bins - 1)

                median_correction = range_medians_smoothed[range_bin_idx]
                std_val = range_adjusted_stds_smoothed[range_bin_idx]

                corrected = destreak_val - median_correction - cosine_correction
                adjusted_val = corrected / std_val if std_val > 0 else corrected

                if np.isfinite(destreak_val) and np.isfinite(adjusted_val):
                    destreak_sum[lat_idx, lon_idx] += destreak_val
                    adjusted_sum[lat_idx, lon_idx] += adjusted_val
                    count_grid[lat_idx, lon_idx] += 1

        # Progress indicator
        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
            print_progress(frame_idx + 1, n_frames, prefix="Averaging")

    # Compute averages (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_destreaked = np.where(count_grid > 0, destreak_sum / count_grid, np.nan)
        avg_adjusted = np.where(count_grid > 0, adjusted_sum / count_grid, np.nan)

    return {
        "avg_destreaked": avg_destreaked,
        "avg_adjusted": avg_adjusted,
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
        "lon_centers": lon_centers,
        "lat_centers": lat_centers,
        "count_grid": count_grid,
        "n_frames": n_frames,
    }


# Legacy function - kept but not used by main()
def compute_wind_relative_intensity(
    results: list[dict],
    transformer: "CoordinateTransformer",  # type: ignore[name-defined]  # noqa: F821
    range_bins: list[tuple[int, int]],
    angle_bin_size: float = 2.0,
    range_bin_size: float = 50.0,
    smooth_window: int = 10,
) -> dict:
    """
    Compute average intensity vs wind-relative angle and vs range.

    Two-pass algorithm:
    1. First pass: Collect intensity vs slant range, compute median per range bin
    2. Apply smoothing to median
    3. Second pass: Subtract smoothed median from intensity, then bin by wind-relative angle

    Excludes shadow regions from each frame.

    Args:
        results: List of frame data dictionaries
        transformer: CoordinateTransformer for navigation interpolation
        range_bins: List of (start_bin, end_bin) tuples for angle plot
        angle_bin_size: Size of angle bins in degrees
        range_bin_size: Size of range bins in meters for range plot
        smooth_window: Window size for smoothing median (in range bins)

    Returns:
        Dictionary with angle and range data.
    """
    # Create angle bins from -180 to 180
    angle_edges = np.arange(-180, 180 + angle_bin_size, angle_bin_size)
    angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
    n_angle_bins = len(angle_centers)

    # Determine max slant range across all frames (use frame's slant_ranges)
    max_range_m = 0.0
    for frame in results:
        frame_max_slant = float(np.max(frame["slant_ranges"]))
        max_range_m = max(max_range_m, frame_max_slant)

    # Create range bins for the range plot
    range_edges = np.arange(0, max_range_m + range_bin_size, range_bin_size)
    range_centers = (range_edges[:-1] + range_edges[1:]) / 2
    n_range_bins = len(range_centers)

    # Collect wind and ship statistics
    all_wind_speeds: list[float] = []
    all_wind_dirs: list[float] = []
    all_ship_speeds: list[float] = []
    all_ship_headings: list[float] = []

    # =========================================================================
    # FIRST PASS: Collect intensity vs range to compute median per range bin
    # =========================================================================
    range_intensities: list[list[float]] = [[] for _ in range(n_range_bins)]

    n_frames = len(results)
    for frame_idx in range(n_frames):
        frame = results[frame_idx]
        intensity = frame["intensity"]
        wtheta = frame["wtheta"]
        slant_ranges = frame["slant_ranges"]

        # Get shadow region for this frame
        shadow = wtheta.shadow_region
        shadow_leading = shadow.leading if shadow.is_valid else None
        shadow_trailing = shadow.trailing if shadow.is_valid else None

        # Get per-radial navigation (includes interpolated wind direction)
        nav = transformer.get_frame_navigation(frame_idx)

        # Collect wind and ship statistics (use mean across radials for this frame)
        all_wind_speeds.append(float(np.mean(nav.wind_speed)))
        all_wind_dirs.append(float(np.mean(nav.wind_dir)))
        all_ship_speeds.append(float(np.mean(nav.ship_speed)))
        all_ship_headings.append(float(np.mean(nav.heading)))

        # Get adjusted theta values
        theta = wtheta.theta

        # Process each radial (excluding shadow)
        for radial_idx in range(len(theta)):
            if is_in_shadow(theta[radial_idx], shadow_leading, shadow_trailing):
                continue

            # Bin intensity by slant range
            for range_idx in range(intensity.shape[1]):
                slant_range = slant_ranges[range_idx]
                range_bin_idx = int(slant_range / range_bin_size)
                range_bin_idx = min(max(0, range_bin_idx), n_range_bins - 1)

                val = intensity[radial_idx, range_idx]
                if np.isfinite(val):
                    range_intensities[range_bin_idx].append(float(val))

    # Compute mean and median per range bin
    range_means = np.zeros(n_range_bins)
    range_medians = np.zeros(n_range_bins)
    range_stds = np.zeros(n_range_bins)
    range_counts = np.zeros(n_range_bins, dtype=int)

    for i, values in enumerate(range_intensities):
        if len(values) > 0:
            range_means[i] = np.mean(values)
            range_medians[i] = np.median(values)
            range_stds[i] = np.std(values)
            range_counts[i] = len(values)

    # Apply smoothing to median
    range_medians_smoothed = smooth_array(range_medians, smooth_window)

    # =========================================================================
    # SECOND PASS: Compute range-corrected intensity vs wind-relative angle
    # Collect ALL range bins together for a single cosine fit
    # =========================================================================
    # Combined angle intensities across all range bins
    combined_angle_intensities: list[list[float]] = [[] for _ in range(n_angle_bins)]

    # Also keep per-range-selection for display
    first_slant_ranges = results[0]["slant_ranges"]
    angle_intensities: dict[str, list[list[float]]] = {}
    for start_bin, end_bin in range_bins:
        start_m = float(first_slant_ranges[min(start_bin, len(first_slant_ranges) - 1)])
        end_m = float(first_slant_ranges[min(end_bin - 1, len(first_slant_ranges) - 1)])
        label = f"{start_m:.0f}-{end_m:.0f}m"
        angle_intensities[label] = [[] for _ in range(n_angle_bins)]

    for frame_idx in range(n_frames):
        frame = results[frame_idx]
        intensity = frame["intensity"]
        wtheta = frame["wtheta"]
        slant_ranges = frame["slant_ranges"]

        # Get shadow region for this frame
        shadow = wtheta.shadow_region
        shadow_leading = shadow.leading if shadow.is_valid else None
        shadow_trailing = shadow.trailing if shadow.is_valid else None

        # Get per-radial navigation
        nav = transformer.get_frame_navigation(frame_idx)
        theta = wtheta.theta
        wind_relative = wrap_angle_180(theta - nav.wind_dir)

        # Process each radial (excluding shadow)
        for radial_idx in range(len(theta)):
            if is_in_shadow(theta[radial_idx], shadow_leading, shadow_trailing):
                continue

            angle = wind_relative[radial_idx]
            angle_bin_idx = int((angle + 180) / angle_bin_size)
            angle_bin_idx = min(max(0, angle_bin_idx), n_angle_bins - 1)

            # Collect ALL range bins for combined cosine fit
            for range_idx in range(intensity.shape[1]):
                val = intensity[radial_idx, range_idx]
                if np.isfinite(val):
                    slant_range = slant_ranges[range_idx]
                    range_bin_idx_for_correction = int(slant_range / range_bin_size)
                    range_bin_idx_for_correction = min(
                        max(0, range_bin_idx_for_correction), n_range_bins - 1
                    )
                    median_val = range_medians_smoothed[range_bin_idx_for_correction]
                    corrected = float(val) - median_val
                    combined_angle_intensities[angle_bin_idx].append(corrected)

            # For display: also collect per range selection
            for start_bin, end_bin in range_bins:
                start_m = float(slant_ranges[min(start_bin, len(slant_ranges) - 1)])
                end_m = float(slant_ranges[min(end_bin - 1, len(slant_ranges) - 1)])
                label = f"{start_m:.0f}-{end_m:.0f}m"

                actual_start = max(0, start_bin)
                actual_end = min(intensity.shape[1], end_bin)

                if actual_start >= actual_end:
                    continue

                corrected_values: list[float] = []
                for range_idx in range(actual_start, actual_end):
                    val = intensity[radial_idx, range_idx]
                    if np.isfinite(val):
                        slant_range = slant_ranges[range_idx]
                        range_bin_idx_for_correction = int(slant_range / range_bin_size)
                        range_bin_idx_for_correction = min(
                            max(0, range_bin_idx_for_correction), n_range_bins - 1
                        )
                        median_val = range_medians_smoothed[range_bin_idx_for_correction]
                        corrected_values.append(float(val) - median_val)

                if corrected_values:
                    mean_corrected = float(np.mean(corrected_values))
                    angle_intensities[label][angle_bin_idx].append(mean_corrected)

    # Compute combined mean for cosine fit across all ranges
    combined_means = np.zeros(n_angle_bins)
    combined_counts = np.zeros(n_angle_bins, dtype=int)
    for i, values in enumerate(combined_angle_intensities):
        if len(values) > 0:
            combined_means[i] = np.mean(values)
            combined_counts[i] = len(values)

    # Compute statistics for angle plot (per range selection for display)
    angle_means: dict[str, np.ndarray] = {}
    angle_stds: dict[str, np.ndarray] = {}
    angle_counts: dict[str, np.ndarray] = {}

    for label, bin_values in angle_intensities.items():
        means = np.zeros(n_angle_bins)
        stds = np.zeros(n_angle_bins)
        counts = np.zeros(n_angle_bins, dtype=int)

        for i, values in enumerate(bin_values):
            if len(values) > 0:
                means[i] = np.mean(values)
                stds[i] = np.std(values)
                counts[i] = len(values)

        angle_means[label] = means
        angle_stds[label] = stds
        angle_counts[label] = counts

    # Compute wind and ship statistics
    wind_speed_mean = float(np.mean(all_wind_speeds))
    wind_speed_std = float(np.std(all_wind_speeds))
    wind_dir_mean = float(np.mean(all_wind_dirs))
    wind_dir_std = float(np.std(all_wind_dirs))
    ship_speed_mean = float(np.mean(all_ship_speeds))
    ship_speed_std = float(np.std(all_ship_speeds))
    ship_heading_mean = float(np.mean(all_ship_headings))
    ship_heading_std = float(np.std(all_ship_headings))

    # Fit cosine to combined data (all range bins)
    from scipy.optimize import curve_fit

    def cosine_func(theta_rad: np.ndarray, a: float, phi: float, b: float) -> np.ndarray:
        return a * np.cos(theta_rad - phi) + b

    # Fit combined data
    valid = np.isfinite(combined_means) & (combined_counts > 0)
    cosine_params: dict = {"a": 0.0, "b": 0.0, "phi": 0.0, "r_squared": 0.0}

    if np.sum(valid) >= 4:
        theta_valid = np.deg2rad(angle_centers[valid])
        y_valid = combined_means[valid]

        try:
            a0 = (np.max(y_valid) - np.min(y_valid)) / 2
            b0 = np.mean(y_valid)
            phi0 = 0.0

            popt, _ = curve_fit(
                cosine_func,
                theta_valid,
                y_valid,
                p0=[a0, phi0, b0],
                bounds=([-np.inf, -np.pi, -np.inf], [np.inf, np.pi, np.inf]),
                maxfev=5000,
            )
            a, phi, b = popt

            y_fit = cosine_func(theta_valid, a, phi, b)
            ss_res = np.sum((y_valid - y_fit) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            cosine_params = {
                "a": float(a),
                "b": float(b),
                "phi": float(np.rad2deg(phi)),
                "r_squared": float(r_squared),
            }
        except Exception:
            pass

    # =========================================================================
    # THIRD PASS: Compute per-range std after median and cosine subtraction
    # =========================================================================
    range_adjusted_intensities: list[list[float]] = [[] for _ in range(n_range_bins)]

    for frame_idx in range(n_frames):
        frame = results[frame_idx]
        intensity = frame["intensity"]
        wtheta = frame["wtheta"]
        slant_ranges = frame["slant_ranges"]

        shadow = wtheta.shadow_region
        shadow_leading = shadow.leading if shadow.is_valid else None
        shadow_trailing = shadow.trailing if shadow.is_valid else None

        nav = transformer.get_frame_navigation(frame_idx)
        theta = wtheta.theta
        wind_relative = wrap_angle_180(theta - nav.wind_dir)

        for radial_idx in range(len(theta)):
            if is_in_shadow(theta[radial_idx], shadow_leading, shadow_trailing):
                continue

            # Compute cosine correction for this radial
            angle_rad = np.deg2rad(wind_relative[radial_idx])
            phi_rad = np.deg2rad(cosine_params["phi"])
            cosine_correction = (
                cosine_params["a"] * np.cos(angle_rad - phi_rad) + cosine_params["b"]
            )

            for range_idx in range(intensity.shape[1]):
                val = intensity[radial_idx, range_idx]
                if np.isfinite(val):
                    slant_range = slant_ranges[range_idx]
                    range_bin_idx_adj = int(slant_range / range_bin_size)
                    range_bin_idx_adj = min(max(0, range_bin_idx_adj), n_range_bins - 1)

                    median_val = range_medians_smoothed[range_bin_idx_adj]
                    adjusted = float(val) - median_val - cosine_correction
                    range_adjusted_intensities[range_bin_idx_adj].append(adjusted)

    # Compute per-range std after adjustment
    range_adjusted_stds = np.zeros(n_range_bins)
    for i, values in enumerate(range_adjusted_intensities):
        if len(values) > 1:
            range_adjusted_stds[i] = np.std(values)

    # Smooth the adjusted stds
    range_adjusted_stds_smoothed = smooth_array(range_adjusted_stds, smooth_window)

    return {
        "angle_centers": angle_centers,
        "angle_means": angle_means,
        "angle_stds": angle_stds,
        "angle_counts": angle_counts,
        "combined_angle_means": combined_means,
        "combined_angle_counts": combined_counts,
        "cosine_params": cosine_params,
        "range_centers": range_centers,
        "range_means": range_means,
        "range_medians": range_medians,
        "range_medians_smoothed": range_medians_smoothed,
        "range_stds": range_stds,
        "range_adjusted_stds": range_adjusted_stds,
        "range_adjusted_stds_smoothed": range_adjusted_stds_smoothed,
        "range_counts": range_counts,
        "smooth_window": smooth_window,
        # Wind and ship statistics
        "wind_speed_mean": wind_speed_mean,
        "wind_speed_std": wind_speed_std,
        "wind_dir_mean": wind_dir_mean,
        "wind_dir_std": wind_dir_std,
        "ship_speed_mean": ship_speed_mean,
        "ship_speed_std": ship_speed_std,
        "ship_heading_mean": ship_heading_mean,
        "ship_heading_std": ship_heading_std,
        "n_frames": n_frames,
    }


def fit_hyperbolic(r: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Fit data to hyperbolic function: y = a/r + b.

    Args:
        r: Range values (x-axis)
        y: Intensity values (y-axis)

    Returns:
        Tuple of (fitted y values, parameters dict with 'a', 'b', 'r_squared')
    """
    from scipy.optimize import curve_fit

    def hyperbolic(r: np.ndarray, a: float, b: float) -> np.ndarray:
        return a / r + b

    # Filter out zeros and invalid values
    valid = (r > 0) & np.isfinite(y) & (y > 0)
    if np.sum(valid) < 3:
        return np.full_like(y, np.nan), {"a": np.nan, "b": np.nan, "r_squared": np.nan}

    r_valid = r[valid]
    y_valid = y[valid]

    try:
        # Initial guess: a = mean(y * r), b = min(y)
        a0 = float(np.mean(y_valid * r_valid))
        b0 = float(np.min(y_valid))
        popt, _ = curve_fit(hyperbolic, r_valid, y_valid, p0=[a0, b0], maxfev=5000)
        a, b = popt

        # Compute R-squared
        y_fit = hyperbolic(r_valid, a, b)
        ss_res = np.sum((y_valid - y_fit) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Generate fitted curve for all r values
        y_fitted = np.where(r > 0, hyperbolic(r, a, b), np.nan)

        return y_fitted, {"a": a, "b": b, "r_squared": r_squared}
    except Exception:
        return np.full_like(y, np.nan), {"a": np.nan, "b": np.nan, "r_squared": np.nan}


def fit_cosine(theta: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Fit data to cosine function: y = a * cos(theta - phi) + b.

    Args:
        theta: Angle values in degrees (x-axis)
        y: Intensity values (y-axis)

    Returns:
        Tuple of (fitted y values, parameters dict with 'a', 'b', 'phi', 'r_squared')
    """
    from scipy.optimize import curve_fit

    def cosine_func(theta_rad: np.ndarray, a: float, phi: float, b: float) -> np.ndarray:
        return a * np.cos(theta_rad - phi) + b

    # Filter invalid values
    valid = np.isfinite(y) & (y > 0)
    if np.sum(valid) < 4:
        return np.full_like(y, np.nan), {
            "a": np.nan,
            "b": np.nan,
            "phi": np.nan,
            "r_squared": np.nan,
        }

    theta_valid = np.deg2rad(theta[valid])
    y_valid = y[valid]

    try:
        # Initial guess
        a0 = (np.max(y_valid) - np.min(y_valid)) / 2
        b0 = np.mean(y_valid)
        phi0 = 0.0  # Assume max at upwind (0°)

        popt, _ = curve_fit(
            cosine_func,
            theta_valid,
            y_valid,
            p0=[a0, phi0, b0],
            bounds=([-np.inf, -np.pi, -np.inf], [np.inf, np.pi, np.inf]),
            maxfev=5000,
        )
        a, phi, b = popt

        # Compute R-squared
        y_fit = cosine_func(theta_valid, a, phi, b)
        ss_res = np.sum((y_valid - y_fit) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Generate fitted curve for all theta values
        theta_rad = np.deg2rad(theta)
        y_fitted = cosine_func(theta_rad, a, phi, b)

        return y_fitted, {"a": a, "b": b, "phi": np.rad2deg(phi), "r_squared": r_squared}
    except Exception:
        return np.full_like(y, np.nan), {
            "a": np.nan,
            "b": np.nan,
            "phi": np.nan,
            "r_squared": np.nan,
        }


def plot_wind_relative_intensity(
    data: dict,
    earth_data: dict | None = None,
    title: str = "Average Intensity",
    show_std: bool = True,
) -> None:
    """
    Plot average intensity vs wind-relative angle, vs range, and earth-averaged maps.

    Creates a 2x2 figure:
        Row 1: Intensity vs slant range, Range-corrected intensity vs wind-relative angle
        Row 2: Averaged destreaked intensity (earth), Averaged adjusted intensity (earth)

    Args:
        data: Output from compute_wind_relative_intensity()
        earth_data: Output from compute_averaged_earth_intensity() (optional)
        title: Overall plot title
        show_std: Whether to show standard deviation as shaded region
    """
    import matplotlib.pyplot as plt

    # Create 2x2 figure if we have earth data, otherwise 1x2
    if earth_data is not None:
        fig = plt.figure(figsize=(14, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax3, ax4 = None, None

    # --- Top-left: Intensity vs Range showing mean, median, and std ---
    range_centers = data["range_centers"]
    range_means = data["range_means"]
    range_medians = data["range_medians"]
    range_medians_smoothed = data["range_medians_smoothed"]
    range_stds = data["range_stds"]
    range_adjusted_stds = data["range_adjusted_stds"]
    range_adjusted_stds_smoothed = data["range_adjusted_stds_smoothed"]
    smooth_window = data.get("smooth_window", 10)

    ax1.plot(range_centers, range_means, "b-", linewidth=1.5, label="Mean")
    ax1.plot(range_centers, range_medians, "g-", linewidth=1.5, label="Median")
    ax1.plot(
        range_centers,
        range_medians_smoothed,
        "r-",
        linewidth=2.5,
        label=f"Smoothed Median (window={smooth_window})",
    )

    if show_std:
        ax1.fill_between(
            range_centers,
            range_means - range_stds,
            range_means + range_stds,
            alpha=0.15,
            color="blue",
            label="±1 std (mean)",
        )

    ax1.set_xlabel("Slant Range (m)", fontsize=11)
    ax1.set_ylabel("Intensity", fontsize=11, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("Intensity & Std vs Slant Range (shadow excluded)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis for std
    ax1_std = ax1.twinx()
    ax1_std.plot(range_centers, range_adjusted_stds, "m-", linewidth=1, alpha=0.6, label="Std")
    ax1_std.plot(
        range_centers, range_adjusted_stds_smoothed, "m-", linewidth=2.5, label="Smoothed Std"
    )
    ax1_std.set_ylabel("Std (after median+cosine)", fontsize=11, color="purple")
    ax1_std.tick_params(axis="y", labelcolor="purple")
    ax1_std.legend(loc="upper right", fontsize=8)

    # --- Top-right: Range-corrected Intensity vs Wind-Relative Angle ---
    angle_centers = data["angle_centers"]
    angle_means = data["angle_means"]
    angle_stds = data["angle_stds"]
    combined_angle_means = data["combined_angle_means"]
    cosine_params = data["cosine_params"]

    n_lines = len(angle_means)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_lines))  # type: ignore[attr-defined]

    # Plot per-range-selection data
    for i, (label, mean_values) in enumerate(angle_means.items()):
        std_values = angle_stds[label]
        color = colors[i]

        ax2.plot(angle_centers, mean_values, color=color, linewidth=1.5, label=label)

        if show_std:
            ax2.fill_between(
                angle_centers,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2,
                color=color,
            )

    # Plot combined data and its fit (the one actually used for correction)
    ax2.plot(angle_centers, combined_angle_means, "k-", linewidth=2, label="All ranges")

    # Plot combined cosine fit
    theta_rad = np.deg2rad(angle_centers)
    phi_rad = np.deg2rad(cosine_params["phi"])
    combined_fit = cosine_params["a"] * np.cos(theta_rad - phi_rad) + cosine_params["b"]
    ax2.plot(angle_centers, combined_fit, "k--", linewidth=2.5, label="Combined fit")

    # Format fit info
    fit_info = (
        f"Combined: {cosine_params['a']:.1f}·cos(θ-{cosine_params['phi']:.0f}°)"
        f"+{cosine_params['b']:.1f} (R²={cosine_params['r_squared']:.3f})"
    )

    # Add horizontal line at y=0 (since we subtracted median)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.5, linewidth=1)

    # Add vertical lines for key directions
    ax2.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax2.axvline(180, color="blue", linestyle="--", alpha=0.5)
    ax2.axvline(-180, color="blue", linestyle="--", alpha=0.5)
    ax2.axvline(90, color="green", linestyle=":", alpha=0.5)
    ax2.axvline(-90, color="green", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Wind-Relative Angle (°)", fontsize=11)
    ax2.set_ylabel("Range-Corrected Intensity (I - median)", fontsize=11)
    ax2.set_title("Range-Corrected Intensity vs Wind-Relative Angle", fontsize=12)
    ax2.set_xlim(-180, 180)
    ax2.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add text annotations for wind directions
    ylim = ax2.get_ylim()
    y_text = ylim[1] + (ylim[1] - ylim[0]) * 0.02
    ax2.text(0, y_text, "Upwind", ha="center", va="bottom", fontsize=9, color="red")
    ax2.text(180, y_text, "Down", ha="center", va="bottom", fontsize=9, color="blue")
    ax2.text(-180, y_text, "Down", ha="center", va="bottom", fontsize=9, color="blue")
    ax2.text(90, y_text, "Cross", ha="center", va="bottom", fontsize=9, color="green")
    ax2.text(-90, y_text, "Cross", ha="center", va="bottom", fontsize=9, color="green")

    # --- Bottom row: Earth-averaged intensity maps ---
    if earth_data is not None and ax3 is not None and ax4 is not None:
        lon_edges = earth_data["lon_edges"]
        lat_edges = earth_data["lat_edges"]
        avg_destreaked = earth_data["avg_destreaked"]
        avg_adjusted = earth_data["avg_adjusted"]

        # Normalize to [0, 1]
        valid_destreak = avg_destreaked[np.isfinite(avg_destreaked)]
        valid_adjusted = avg_adjusted[np.isfinite(avg_adjusted)]

        if len(valid_destreak) > 0:
            d_min = float(np.percentile(valid_destreak, 2))
            d_max = float(np.percentile(valid_destreak, 98))
            d_range = d_max - d_min
            if d_range > 0:
                destreak_norm = (avg_destreaked - d_min) / d_range
                destreak_norm = np.clip(destreak_norm, 0, 1)
            else:
                destreak_norm = np.zeros_like(avg_destreaked)
        else:
            destreak_norm = avg_destreaked

        if len(valid_adjusted) > 0:
            a_min = float(np.percentile(valid_adjusted, 2))
            a_max = float(np.percentile(valid_adjusted, 98))
            a_range = a_max - a_min
            if a_range > 0:
                adjusted_norm = (avg_adjusted - a_min) / a_range
                adjusted_norm = np.clip(adjusted_norm, 0, 1)
            else:
                adjusted_norm = np.zeros_like(avg_adjusted)
        else:
            adjusted_norm = avg_adjusted

        # Calculate center for range rings
        center_lon = (lon_edges[0] + lon_edges[-1]) / 2
        center_lat = (lat_edges[0] + lat_edges[-1]) / 2

        # Bottom-left: Averaged destreaked intensity
        mesh1 = ax3.pcolormesh(
            lon_edges,
            lat_edges,
            destreak_norm,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax3.set_xlabel("Longitude (°E)", fontsize=11)
        ax3.set_ylabel("Latitude (°N)", fontsize=11)
        ax3.set_title(
            f"Avg Destreaked Intensity [0,1] ({earth_data['n_frames']} frames)", fontsize=12
        )
        ax3.set_aspect("equal", adjustable="box")
        _add_range_rings_earth(ax3, center_lon, center_lat, lon_edges, lat_edges)
        plt.colorbar(mesh1, ax=ax3, shrink=0.8, label="Normalized Intensity")

        # Bottom-right: Averaged adjusted intensity
        mesh2 = ax4.pcolormesh(
            lon_edges,
            lat_edges,
            adjusted_norm,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax4.set_xlabel("Longitude (°E)", fontsize=11)
        ax4.set_ylabel("Latitude (°N)", fontsize=11)
        ax4.set_title(
            f"Avg Adjusted Intensity [0,1] ({earth_data['n_frames']} frames)", fontsize=12
        )
        ax4.set_aspect("equal", adjustable="box")
        _add_range_rings_earth(ax4, center_lon, center_lat, lon_edges, lat_edges)
        plt.colorbar(mesh2, ax=ax4, shrink=0.8, label="Normalized Intensity")

    # Add statistics text box
    stats_lines = [
        f"Wind: {data['wind_speed_mean']:.1f} ± {data['wind_speed_std']:.1f} m/s "
        f"from {data['wind_dir_mean']:.1f} ± {data['wind_dir_std']:.1f}°   |   "
        f"Ship: {data['ship_speed_mean']:.1f} ± {data['ship_speed_std']:.1f} m/s, "
        f"heading {data['ship_heading_mean']:.1f} ± {data['ship_heading_std']:.1f}°   |   "
        f"Frames: {data['n_frames']}",
    ]
    # Add cosine fit info
    if fit_info:
        stats_lines.append(fit_info)
    stats_text = "\n".join(stats_lines)
    fig.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=9,
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.show()


def _add_range_rings_earth(
    ax: object,
    center_lon: float,
    center_lat: float,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
) -> None:
    """Add 1km range rings to an earth coordinate plot centered on the grid center."""
    # Determine max range from grid extent
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(center_lat))

    max_lon_dist = max(abs(lon_edges[-1] - center_lon), abs(lon_edges[0] - center_lon))
    max_lat_dist = max(abs(lat_edges[-1] - center_lat), abs(lat_edges[0] - center_lat))
    max_range_m = max(max_lon_dist * meters_per_deg_lon, max_lat_dist * meters_per_deg_lat)

    ring_interval = 1000.0  # 1km rings
    ring_radii = np.arange(ring_interval, max_range_m + ring_interval, ring_interval)

    for r in ring_radii:
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        circle_lon = center_lon + (r * np.cos(circle_theta)) / meters_per_deg_lon
        circle_lat = center_lat + (r * np.sin(circle_theta)) / meters_per_deg_lat
        ax.plot(circle_lon, circle_lat, "w--", linewidth=0.5, alpha=0.5)  # type: ignore[attr-defined]


def compute_adjusted_intensity(
    intensity: np.ndarray,
    slant_ranges: np.ndarray,
    horizontal_ranges: np.ndarray,
    theta: np.ndarray,
    wind_dir: np.ndarray,
    range_medians_smoothed: np.ndarray,
    range_adjusted_stds_smoothed: np.ndarray,
    range_centers: np.ndarray,
    range_bin_size: float,
    cosine_params: dict,
) -> np.ndarray:
    """
    Compute adjusted intensity by removing range and wind-relative effects.

    adjusted = (intensity - smoothed_median(range) - cosine_fit(wind_relative_angle))
               / smoothed_std(range)

    Args:
        intensity: Raw intensity array (n_bearings, n_ranges)
        slant_ranges: Slant range for each range bin
        horizontal_ranges: Horizontal range for each range bin
        theta: Adjusted theta for each bearing
        wind_dir: Wind direction for each bearing (interpolated)
        range_medians_smoothed: Smoothed median intensity per range bin
        range_adjusted_stds_smoothed: Smoothed std per range bin (after median/cosine)
        range_centers: Centers of range bins used for median calculation
        range_bin_size: Size of range bins in meters
        cosine_params: Cosine fit parameters {'a': amplitude, 'phi': phase_deg, 'b': offset}

    Returns:
        Adjusted intensity array (n_bearings, n_ranges), normalized by std
    """
    n_bearings, n_ranges_data = intensity.shape
    adjusted = np.zeros_like(intensity, dtype=np.float64)

    # Get cosine parameters
    a = cosine_params.get("a", 0.0)
    phi_deg = cosine_params.get("phi", 0.0)
    b = cosine_params.get("b", 0.0)

    n_range_bins = len(range_centers)

    for bearing_idx in range(n_bearings):
        # Calculate wind-relative angle for this bearing
        wind_relative = wrap_angle_180(theta[bearing_idx] - wind_dir[bearing_idx])
        wind_relative_rad = np.deg2rad(wind_relative)
        phi_rad = np.deg2rad(phi_deg)

        # Cosine correction for this bearing
        cosine_correction = a * np.cos(wind_relative_rad - phi_rad) + b

        for range_idx in range(n_ranges_data):
            val = intensity[bearing_idx, range_idx]

            # Find range bin for corrections (use slant range for binning)
            slant_range = slant_ranges[range_idx]
            range_bin_idx = int(slant_range / range_bin_size)
            range_bin_idx = min(max(0, range_bin_idx), n_range_bins - 1)

            median_correction = range_medians_smoothed[range_bin_idx]
            std_val = range_adjusted_stds_smoothed[range_bin_idx]

            # Apply corrections and normalize by std
            corrected = val - median_correction - cosine_correction
            if std_val > 0:
                adjusted[bearing_idx, range_idx] = corrected / std_val
            else:
                adjusted[bearing_idx, range_idx] = corrected

    return adjusted


# Legacy function - kept but not used by main()
def compute_averaged_earth_intensity(
    results: list[dict],
    transformer: "CoordinateTransformer",  # type: ignore[name-defined]  # noqa: F821
    intensity_data: dict,
    cosine_params: dict,
    grid_size: int = 200,
) -> dict:
    """
    Compute averaged intensity maps in earth coordinates across all frames.

    Uses all range bins (intensity in bottom 12 bits is valid for all bins).

    Args:
        results: List of frame data dictionaries
        transformer: CoordinateTransformer for navigation interpolation
        intensity_data: Output from compute_wind_relative_intensity()
        cosine_params: Cosine fit parameters for wind correction
        grid_size: Grid resolution for earth coordinate averaging

    Returns:
        Dictionary with:
            - avg_destreaked: Averaged destreaked intensity on earth grid
            - avg_adjusted: Averaged adjusted intensity on earth grid
            - lon_grid: Longitude grid edges
            - lat_grid: Latitude grid edges
            - lon_centers: Longitude grid centers
            - lat_centers: Latitude grid centers
            - count_grid: Number of samples at each grid point
    """
    # Use all range bins (intensity in bottom 12 bits is valid for all bins)
    range_start = 0

    # First pass: determine bounding box across all frames
    all_lons: list[float] = []
    all_lats: list[float] = []

    for frame_idx, frame in enumerate(results):
        horizontal_ranges = frame["horizontal_ranges"]
        horiz_ranges_slice = horizontal_ranges[range_start:]

        earth_lon, earth_lat = transformer.get_earth_coords(frame_idx, horiz_ranges_slice)
        all_lons.extend(earth_lon.flatten().tolist())
        all_lats.extend(earth_lat.flatten().tolist())

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)

    # Add small margin
    lon_margin = (lon_max - lon_min) * 0.02
    lat_margin = (lat_max - lat_min) * 0.02
    lon_min -= lon_margin
    lon_max += lon_margin
    lat_min -= lat_margin
    lat_max += lat_margin

    # Create grid
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    # Accumulators for averaging
    destreak_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    adjusted_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    count_grid = np.zeros((grid_size, grid_size), dtype=np.int64)

    range_bin_size = intensity_data["range_centers"][1] - intensity_data["range_centers"][0]

    # Second pass: accumulate intensity onto grid
    n_frames = len(results)
    for frame_idx in range(n_frames):
        frame = results[frame_idx]
        intensity = frame["intensity"]
        wtheta = frame["wtheta"]
        horizontal_ranges = frame["horizontal_ranges"]
        slant_ranges = frame["slant_ranges"]

        # Use all range bins from range_start to end
        intensity_slice = intensity[:, range_start:]
        horiz_ranges_slice = horizontal_ranges[range_start:]
        slant_ranges_slice = slant_ranges[range_start:]

        # Get navigation data
        nav = transformer.get_frame_navigation(frame_idx)

        # Get adjusted theta values
        theta = wtheta.theta

        # Skip shadow region
        shadow = wtheta.shadow_region
        shadow_leading = shadow.leading if shadow.is_valid else None
        shadow_trailing = shadow.trailing if shadow.is_valid else None

        # Compute adjusted intensity
        adjusted_slice = compute_adjusted_intensity(
            intensity=intensity_slice,
            slant_ranges=slant_ranges_slice,
            horizontal_ranges=horiz_ranges_slice,
            theta=theta,
            wind_dir=nav.wind_dir,
            range_medians_smoothed=intensity_data["range_medians_smoothed"],
            range_adjusted_stds_smoothed=intensity_data["range_adjusted_stds_smoothed"],
            range_centers=intensity_data["range_centers"],
            range_bin_size=range_bin_size,
            cosine_params=cosine_params,
        )

        # Get earth coordinates
        earth_lon, earth_lat = transformer.get_earth_coords(frame_idx, horiz_ranges_slice)

        # Accumulate onto grid (excluding shadow)
        for bearing_idx in range(len(theta)):
            if is_in_shadow(theta[bearing_idx], shadow_leading, shadow_trailing):
                continue

            for range_idx in range(intensity_slice.shape[1]):
                lon_val = earth_lon[bearing_idx, range_idx]
                lat_val = earth_lat[bearing_idx, range_idx]

                # Find grid cell
                lon_idx = int((lon_val - lon_min) / (lon_max - lon_min) * grid_size)
                lat_idx = int((lat_val - lat_min) / (lat_max - lat_min) * grid_size)

                # Clamp to grid bounds
                lon_idx = max(0, min(grid_size - 1, lon_idx))
                lat_idx = max(0, min(grid_size - 1, lat_idx))

                destreak_val = intensity_slice[bearing_idx, range_idx]
                adjusted_val = adjusted_slice[bearing_idx, range_idx]

                if np.isfinite(destreak_val) and np.isfinite(adjusted_val):
                    destreak_sum[lat_idx, lon_idx] += destreak_val
                    adjusted_sum[lat_idx, lon_idx] += adjusted_val
                    count_grid[lat_idx, lon_idx] += 1

        # Progress indicator
        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
            print_progress(frame_idx + 1, n_frames, prefix="Averaging")

    # Compute averages (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_destreaked = np.where(count_grid > 0, destreak_sum / count_grid, np.nan)
        avg_adjusted = np.where(count_grid > 0, adjusted_sum / count_grid, np.nan)

    return {
        "avg_destreaked": avg_destreaked,
        "avg_adjusted": avg_adjusted,
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
        "lon_centers": lon_centers,
        "lat_centers": lat_centers,
        "count_grid": count_grid,
        "n_frames": n_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot average intensity vs wind-relative angle")

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--range-bins",
        type=str,
        nargs="+",
        default=["50-100", "100-200", "200-400"],
        help="Range bin selections as start-end pairs (default: 50-100 100-200 200-400)",
    )
    parser.add_argument(
        "--angle-bin-size",
        type=float,
        default=2.0,
        help="Angle bin size in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--range-bin-size",
        type=float,
        default=50.0,
        help="Range bin size in meters for range plot (default: 50.0)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Window size for smoothing median intensity (in range bins, default: 10)",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Don't show standard deviation shading",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Grid resolution for earth coordinate averaging (default: 200)",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable theta refinement using shadow detection",
    )

    args = parser.parse_args()
    setup_logging(args)

    try:
        # Load config
        config = Config(args.config) if args.config else Config()

        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        logger.info("Loading %d files with %d workers", n_files, args.workers or os.cpu_count())

        # Load frames in parallel
        frames: list = []
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame, fn): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    frame, _ = result
                    frames.append(frame)

        elapsed = time.perf_counter() - t0
        n_loaded = len(frames)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(
            f"Successfully loaded {n_loaded} of {n_files} frames in {elapsed:.2f}s ({fps:.1f} frames/sec)"
        )

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort by timestamp
        frames.sort(
            key=lambda f: f.timestamp if f.timestamp is not None else np.datetime64(0, "ns")
        )

        # Create multi-frame Theta for bearing calculation with shadow refinement
        print("Calculating bearings with shadow refinement...")
        refine = not args.no_refine
        theta = Theta(frames, config, refine=refine)

        if theta.shadow_stats:
            print(f"  Shadow: {theta.shadow_stats}")
            print(f"  Shadow offset correction: {theta.shadow_offset:.2f}°")

        # Apply destreaking to each frame
        print("Applying destreaking...")
        destreaked_intensities = []
        for frame in frames:
            destreaked = Destreak(frame)
            destreaked_intensities.append(destreaked.intensity)

        # Create Combine for coordinate transformation
        print("Creating coordinate transformer...")
        combine = Combine(frames, config, theta=theta)

        # Parse range bin selections
        range_bin_selections: list[tuple[int, int]] = []
        for rb in args.range_bins:
            try:
                start, end = parse_range_bins(rb)
                range_bin_selections.append((start, end))
            except ValueError:
                logger.error("Invalid range-bins format: %s", rb)
                return 1

        # Compute wind-relative intensity (includes cosine fit across all ranges)
        print("Computing wind-relative intensity (excluding shadow regions)...")
        data = compute_wind_relative_intensity_new(
            frames=frames,
            theta=theta,
            destreaked_intensities=destreaked_intensities,
            range_bins=range_bin_selections,
            angle_bin_size=args.angle_bin_size,
            range_bin_size=args.range_bin_size,
            smooth_window=args.smooth_window,
        )

        # Use the combined cosine fit from data (fitted across all range bins)
        cosine_params = data["cosine_params"]
        print(
            f"Cosine fit: {cosine_params['a']:.2f}·cos(θ-{cosine_params['phi']:.1f}°)"
            f"+{cosine_params['b']:.2f} (R²={cosine_params['r_squared']:.4f})"
        )

        # Compute averaged intensity in earth coordinates
        print("Computing averaged intensity in earth coordinates...")
        earth_data = compute_averaged_earth_intensity_new(
            frames=frames,
            theta=theta,
            combine=combine,
            destreaked_intensities=destreaked_intensities,
            intensity_data=data,
            cosine_params=cosine_params,
            grid_size=args.grid_size,
        )

        # Plot histograms and averaged intensity maps
        print("Plotting results...")
        title = f"Average Intensity ({len(frames)} frames, shadow excluded)"
        plot_wind_relative_intensity(
            data=data,
            earth_data=earth_data,
            title=title,
            show_std=not args.no_std,
        )

        # Report peak memory usage
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak_mem_mb = peak_mem / (1024 * 1024)
        else:
            peak_mem_mb = peak_mem / 1024
        print(f"Peak memory: {peak_mem_mb:.1f} MB")

        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
