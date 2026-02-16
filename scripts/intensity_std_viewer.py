#!/usr/bin/env python3
"""
Intensity and standard deviation viewer.

Reads frames, despikes intensity, finds shadow region, NANs out the shadow,
calculates standard deviation by distance bin (excluding shadow), and displays:
    1. Scatter plot of intensity (shadow region NANed out)
    2. Standard deviation vs distance bin
    3. Intensities divided by their respective standard deviation

Usage:
    python intensity_std_viewer.py 20220405 20220406 /path/to/POLAR --shadow

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw.args import add_time_range_arguments  # noqa: E402

from wamos_tpw import Filenames, PolarFrame  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.shadow import Shadow  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame(
    fn: str,
    config: Config | None = None,
) -> dict | None:
    """
    Load a single frame with all required data.

    Args:
        fn: Filename to load
        destreak_config: Optional destreak configuration
        shadow_config: Optional shadow detection configuration

    Returns:
        Dictionary with frame data, or None on error
    """
    try:
        t0 = time.perf_counter()
        polar_frame = PolarFrame(fn)
        t1 = time.perf_counter()

        my_config = config.refine(polar_frame.metadata)  # This tower's config

        theta = Theta(polar_frame, my_config.bias.theta)
        t2 = time.perf_counter()

        destreaked = Destreak(polar_frame)
        t3 = time.perf_counter()

        shadow = Shadow(destreaked, theta)
        t4 = time.perf_counter()

        return {
            "filename": fn,
            "polar_frame": polar_frame,
            "theta": theta,
            "destreaked": destreaked,
            "shadow": shadow,
            # Timing info
            "time_polar_load": t1 - t0,
            "time_theta": t2 - t1,
            "time_destreak": t3 - t2,
            "time_shadow": t4 - t3,
        }
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


def get_shadow_mask(
    theta: np.ndarray,
    shadow_leading: float | None,
    shadow_trailing: float | None,
) -> np.ndarray:
    """
    Get boolean mask for shadow region (vectorized).

    Args:
        theta: Array of theta values
        shadow_leading: Leading edge of shadow (or None)
        shadow_trailing: Trailing edge of shadow (or None)

    Returns:
        Boolean array where True = in shadow
    """
    if shadow_leading is None or shadow_trailing is None:
        return np.zeros(len(theta), dtype=bool)

    if shadow_leading <= shadow_trailing:
        return (theta >= shadow_leading) & (theta <= shadow_trailing)
    else:
        # Wraparound case
        return (theta >= shadow_leading) | (theta <= shadow_trailing)


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
        # Only average over valid (non-zero) values
        valid_mask = np.isfinite(arr[lo:hi]) & (arr[lo:hi] > 0)
        if np.any(valid_mask):
            result[i] = np.mean(arr[lo:hi][valid_mask])
        else:
            result[i] = arr[i]

    return result


def compute_std_by_range(
    intensity: np.ndarray,
    theta: np.ndarray,
    shadow_leading: float | None,
    shadow_trailing: float | None,
    n_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute standard deviation for each range bin, excluding shadow region.

    Performs two-pass calculation:
    1. Calculate initial std
    2. Exclude values exceeding n_sigma from mean, recalculate std

    Args:
        intensity: Intensity array (n_bearings, n_ranges)
        theta: Theta values for each bearing
        shadow_leading: Leading edge of shadow (or None)
        shadow_trailing: Trailing edge of shadow (or None)
        n_sigma: Number of sigma for outlier exclusion (default: 4.0)

    Returns:
        Tuple of (original_std, outlier_removed_std) arrays for each range bin
    """
    # Create mask for shadow bearings (vectorized)
    shadow_mask = get_shadow_mask(theta, shadow_leading, shadow_trailing)

    # Copy intensity and mask out shadow + invalid values
    data = intensity.copy()
    data[shadow_mask, :] = np.nan

    # First pass: compute mean and std along bearing axis (axis=0)
    with np.errstate(all="ignore"):
        mean1 = np.nanmean(data, axis=0)
        std_original = np.nanstd(data, axis=0)

    # Second pass: mask outliers and recompute std
    # Compute deviation from mean for all values
    deviation = np.abs(data - mean1[np.newaxis, :])
    threshold = n_sigma * std_original[np.newaxis, :]

    # Mask outliers (where deviation > threshold)
    data_cleaned = data.copy()
    with np.errstate(invalid="ignore"):
        outlier_mask = deviation > threshold
    data_cleaned[outlier_mask] = np.nan

    # Compute cleaned std
    with np.errstate(all="ignore"):
        std_cleaned = np.nanstd(data_cleaned, axis=0)

    # Where cleaned std is nan or 0, fall back to original
    fallback_mask = ~np.isfinite(std_cleaned) | (std_cleaned == 0)
    std_cleaned[fallback_mask] = std_original[fallback_mask]

    # Replace any remaining nans with 0
    std_original = np.nan_to_num(std_original, nan=0.0)
    std_cleaned = np.nan_to_num(std_cleaned, nan=0.0)

    return std_original, std_cleaned


def compute_std_by_angle(
    intensity: np.ndarray,
    theta: np.ndarray,
    shadow_leading: float | None,
    shadow_trailing: float | None,
    n_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute standard deviation for each bearing (angle), excluding shadow region.

    Performs two-pass calculation:
    1. Calculate initial std
    2. Exclude values exceeding n_sigma from mean, recalculate std

    Args:
        intensity: Intensity array (n_bearings, n_ranges)
        theta: Theta values for each bearing
        shadow_leading: Leading edge of shadow (or None)
        shadow_trailing: Trailing edge of shadow (or None)
        n_sigma: Number of sigma for outlier exclusion (default: 4.0)

    Returns:
        Tuple of (original_std, outlier_removed_std) arrays for each bearing
        (NaN for shadow region)
    """
    n_bearings = intensity.shape[0]

    # Get shadow mask (vectorized)
    shadow_mask = get_shadow_mask(theta, shadow_leading, shadow_trailing)
    non_shadow_indices = np.where(~shadow_mask)[0]

    # Initialize output arrays with NaN
    std_original = np.full(n_bearings, np.nan)
    std_cleaned = np.full(n_bearings, np.nan)

    if len(non_shadow_indices) == 0:
        return std_original, std_cleaned

    # Extract only non-shadow rows for computation
    data = intensity[non_shadow_indices, :]

    # First pass: compute mean and std along range axis (axis=1)
    with np.errstate(all="ignore"):
        mean1 = np.nanmean(data, axis=1)
        std1 = np.nanstd(data, axis=1)

    # Store original std
    std_original[non_shadow_indices] = std1

    # Second pass: mask outliers row by row (more memory efficient)
    std2 = np.zeros(len(non_shadow_indices))
    for i in range(len(non_shadow_indices)):
        if std1[i] > 0:
            row = data[i, :]
            valid_mask = np.isfinite(row) & (np.abs(row - mean1[i]) <= n_sigma * std1[i])
            valid = row[valid_mask]
            if len(valid) > 1:
                std2[i] = np.std(valid)
            else:
                std2[i] = std1[i]
        else:
            std2[i] = std1[i]

    std_cleaned[non_shadow_indices] = std2

    return std_original, std_cleaned


class IntensityStdViewer:
    """
    Interactive viewer showing intensity, std vs range/angle, and normalized intensity.

    Displays five plots:
        1. Polar scatter of intensity (shadow NANed out)
        2. Standard deviation vs distance bin (raw and smoothed)
        3. Polar scatter of intensity / smoothed_range_std
        4. Standard deviation vs angle
        5. Polar scatter of (intensity / range_std) / angular_std
    """

    def __init__(
        self,
        results: list[dict],
        smooth_range: int = 10,
        smooth_angle: int = 10,
        n_sigma: float = 4.0,
    ) -> None:
        """
        Initialize the viewer.

        Args:
            results: List of frame data dictionaries
            smooth_range: Window size for smoothing range std (default: 10)
            smooth_angle: Window size for smoothing angle std (default: 10)
            n_sigma: Number of sigma for outlier exclusion (default: 4.0)
        """
        self.results = results
        self.n_frames = len(results)
        self.smooth_range = smooth_range
        self.smooth_angle = smooth_angle
        self.n_sigma = n_sigma
        self.current_idx = 0
        self.playing = False
        self.timer: object | None = None

        # Set up the figure
        self._setup_figure()
        self._update_display()

    def _setup_figure(self) -> None:
        """Set up the matplotlib figure with five subplots and navigation."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.fig = plt.figure(figsize=(20, 8))

        # Create 2x3 grid (but only use 5 plots)
        # Row 1: Intensity, Std vs Range, Intensity/Range Std
        self.ax_intensity = self.fig.add_subplot(2, 3, 1, projection="polar")
        self.ax_std_range = self.fig.add_subplot(2, 3, 2)
        self.ax_normalized = self.fig.add_subplot(2, 3, 3, projection="polar")
        # Row 2: (empty), Std vs Angle, Fully Normalized
        self.ax_std_angle = self.fig.add_subplot(2, 3, 5)
        self.ax_fully_normalized = self.fig.add_subplot(2, 3, 6, projection="polar")

        # Colorbar references (created on first update)
        self.cbar_intensity: object | None = None
        self.cbar_normalized: object | None = None
        self.cbar_fully_normalized: object | None = None

        # Navigation buttons
        ax_prev = self.fig.add_axes((0.3, 0.02, 0.1, 0.03))
        ax_play = self.fig.add_axes((0.45, 0.02, 0.1, 0.03))
        ax_next = self.fig.add_axes((0.6, 0.02, 0.1, 0.03))

        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, "Next >")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play)
        self.btn_next.on_clicked(self._on_next)

        # Info text
        self.info_text = self.fig.text(
            0.5, 0.97, "", ha="center", va="top", fontsize=10, fontfamily="monospace"
        )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_prev(self, _event: object) -> None:
        """Handle previous button click."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_display()

    def _on_next(self, _event: object) -> None:
        """Handle next button click."""
        if self.current_idx < self.n_frames - 1:
            self.current_idx += 1
            self._update_display()

    def _on_play(self, _event: object) -> None:
        """Handle play/pause button click."""
        import matplotlib.pyplot as plt

        if self.playing:
            self.playing = False
            self.btn_play.label.set_text("Play")
            if self.timer is not None:
                self.timer.stop()  # type: ignore[attr-defined]
                self.timer = None
        else:
            self.playing = True
            self.btn_play.label.set_text("Pause")
            self.timer = self.fig.canvas.new_timer(interval=500)  # type: ignore[union-attr]
            self.timer.add_callback(self._advance_frame)  # type: ignore[union-attr]
            self.timer.start()  # type: ignore[union-attr]
        plt.draw()

    def _on_key(self, event: object) -> None:
        """Handle keyboard events."""
        key = getattr(event, "key", None)
        if key == "left":
            self._on_prev(None)
        elif key == "right":
            self._on_next(None)
        elif key == " ":
            self._on_play(None)

    def _advance_frame(self) -> None:
        """Advance to next frame (for animation)."""
        if self.current_idx < self.n_frames - 1:
            self.current_idx += 1
            self._update_display()
        else:
            self._on_play(None)

    def _update_display(self) -> None:
        """Update all five plots for the current frame."""
        import matplotlib.pyplot as plt

        t_start = time.perf_counter()

        frame = self.results[self.current_idx]
        intensity = frame["intensity"].copy()
        wtheta = frame["wtheta"]
        slant_ranges = frame["slant_ranges"]

        # Get theta values
        theta = wtheta.theta

        # Get shadow region
        shadow = wtheta.shadow_region
        shadow_leading = shadow.leading if shadow.is_valid else None
        shadow_trailing = shadow.trailing if shadow.is_valid else None

        # NAN out shadow region in intensity (vectorized)
        shadow_mask = get_shadow_mask(theta, shadow_leading, shadow_trailing)
        intensity[shadow_mask, :] = np.nan

        t_after_shadow_mask = time.perf_counter()

        # Compute std by range (excluding shadow) with outlier removal
        std_range_orig, std_range_cleaned = compute_std_by_range(
            frame["intensity"], theta, shadow_leading, shadow_trailing, n_sigma=self.n_sigma
        )

        t_after_std_range = time.perf_counter()

        # Smooth the cleaned range std (use cleaned for normalization)
        std_range_smoothed = smooth_array(std_range_cleaned, self.smooth_range)

        # Compute normalized intensity (intensity / smoothed_range_std)
        std_range_broadcast = std_range_smoothed[np.newaxis, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(std_range_broadcast > 0, intensity / std_range_broadcast, np.nan)

        t_after_range_norm = time.perf_counter()

        # Compute std by angle from the normalized data with outlier removal
        std_angle_orig, std_angle_cleaned = compute_std_by_angle(
            normalized, theta, shadow_leading, shadow_trailing, n_sigma=self.n_sigma
        )

        t_after_std_angle = time.perf_counter()

        # Smooth the cleaned angular std (use cleaned for normalization)
        std_angle_smoothed = smooth_array(std_angle_cleaned, self.smooth_angle)

        # Compute fully normalized intensity (normalized / smoothed_angular_std)
        std_angle_broadcast = std_angle_smoothed[:, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            fully_normalized = np.where(
                std_angle_broadcast > 0, normalized / std_angle_broadcast, np.nan
            )

        t_after_angle_norm = time.perf_counter()

        # Print timing on first frame only
        if self.current_idx == 0 and not hasattr(self, "_timing_printed"):
            self._timing_printed = True
            t_shadow = t_after_shadow_mask - t_start
            t_std_range = t_after_std_range - t_after_shadow_mask
            t_range_norm = t_after_range_norm - t_after_std_range
            t_std_angle = t_after_std_angle - t_after_range_norm
            t_angle_norm = t_after_angle_norm - t_after_std_angle
            t_total_std = t_after_angle_norm - t_start

            print("\nStd normalization timing (per frame):")
            pct_shadow = 100 * t_shadow / t_total_std
            pct_std_range = 100 * t_std_range / t_total_std
            pct_range_norm = 100 * t_range_norm / t_total_std
            pct_std_angle = 100 * t_std_angle / t_total_std
            pct_angle_norm = 100 * t_angle_norm / t_total_std
            print(f"  Shadow masking:     {t_shadow * 1000:7.2f}ms ({pct_shadow:5.1f}%)")
            print(f"  Std by range:       {t_std_range * 1000:7.2f}ms ({pct_std_range:5.1f}%)")
            print(f"  Range normalize:    {t_range_norm * 1000:7.2f}ms ({pct_range_norm:5.1f}%)")
            print(f"  Std by angle:       {t_std_angle * 1000:7.2f}ms ({pct_std_angle:5.1f}%)")
            print(f"  Angle normalize:    {t_angle_norm * 1000:7.2f}ms ({pct_angle_norm:5.1f}%)")
            print(f"  Total:              {t_total_std * 1000:7.2f}ms")

        # Remove old colorbars if they exist
        if self.cbar_intensity is not None:
            self.cbar_intensity.remove()  # type: ignore[attr-defined]
            self.cbar_intensity = None
        if self.cbar_normalized is not None:
            self.cbar_normalized.remove()  # type: ignore[attr-defined]
            self.cbar_normalized = None
        if self.cbar_fully_normalized is not None:
            self.cbar_fully_normalized.remove()  # type: ignore[attr-defined]
            self.cbar_fully_normalized = None

        # Clear all axes
        self.ax_intensity.clear()
        self.ax_std_range.clear()
        self.ax_normalized.clear()
        self.ax_std_angle.clear()
        self.ax_fully_normalized.clear()

        # Create meshgrid for polar plots
        theta_rad = np.deg2rad(theta)
        r_mesh, theta_mesh = np.meshgrid(slant_ranges, theta_rad)

        # Get ship and wind info
        ship_speed = frame.get("ship_speed", 0.0)
        ship_heading = frame.get("ship_heading", 0.0)
        wind_speed = frame.get("wind_speed", 0.0)
        wind_dir = frame.get("wind_dir", 0.0)

        # Compute wind direction relative to ship heading
        wind_relative = wind_dir - ship_heading
        wind_relative_rad = np.deg2rad(wind_relative)

        max_range = float(np.max(slant_ranges))
        arrow_len = max_range * 0.85

        # =====================================================================
        # Plot 1: Polar intensity (shadow NANed)
        # =====================================================================
        valid_intensity = intensity[np.isfinite(intensity)]
        if len(valid_intensity) > 0:
            vmin_int = float(np.percentile(valid_intensity, 2))
            vmax_int = float(np.percentile(valid_intensity, 98))
        else:
            vmin_int, vmax_int = 0, 1

        mesh1 = self.ax_intensity.pcolormesh(
            theta_mesh,
            r_mesh,
            intensity,
            shading="auto",
            cmap="viridis",
            vmin=vmin_int,
            vmax=vmax_int,
        )
        self.ax_intensity.set_theta_zero_location("N")  # type: ignore[attr-defined]
        self.ax_intensity.set_theta_direction(-1)  # type: ignore[attr-defined]
        self.ax_intensity.set_title("Intensity (shadow=NaN)", fontsize=10)
        self._add_range_rings(self.ax_intensity, max_range)
        self.cbar_intensity = plt.colorbar(mesh1, ax=self.ax_intensity, shrink=0.7, pad=0.1)

        # Add wind arrow relative to ship (ship heading is "up"/0 in ship frame)
        self.ax_intensity.annotate(
            "",
            xy=(wind_relative_rad, arrow_len),
            xytext=(wind_relative_rad, max_range * 0.15),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )
        self.ax_intensity.text(
            wind_relative_rad,
            max_range * 0.5,
            f"Wind\n{wind_speed:.1f} m/s\n({wind_relative:.0f}° rel)",
            ha="center",
            va="center",
            fontsize=7,
            color="red",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        # =====================================================================
        # Plot 2: Std vs range (original, outlier-removed, smoothed)
        # =====================================================================
        self.ax_std_range.plot(
            slant_ranges, std_range_orig, "b-", linewidth=1, alpha=0.4, label="Original"
        )
        self.ax_std_range.plot(
            slant_ranges,
            std_range_cleaned,
            "g-",
            linewidth=1,
            alpha=0.6,
            label=f"Outliers removed ({self.n_sigma}σ)",
        )
        self.ax_std_range.plot(
            slant_ranges,
            std_range_smoothed,
            "r-",
            linewidth=2,
            label=f"Smoothed (w={self.smooth_range})",
        )
        self.ax_std_range.set_xlabel("Slant Range (m)", fontsize=10)
        self.ax_std_range.set_ylabel("Std", fontsize=10)
        self.ax_std_range.set_title("Std vs Range", fontsize=10)
        self.ax_std_range.legend(loc="best", fontsize=8)
        self.ax_std_range.grid(True, alpha=0.3)

        # Add ship/wind info
        self.ax_std_range.text(
            0.98,
            0.98,
            f"Ship: {ship_speed:.1f} m/s @ {ship_heading:.0f}°\n"
            f"Wind: {wind_speed:.1f} m/s from {wind_dir:.0f}°",
            transform=self.ax_std_range.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        # =====================================================================
        # Plot 3: Polar normalized intensity (intensity / range_std)
        # =====================================================================
        valid_normalized = normalized[np.isfinite(normalized)]
        if len(valid_normalized) > 0:
            vmin_norm = float(np.percentile(valid_normalized, 2))
            vmax_norm = float(np.percentile(valid_normalized, 98))
        else:
            vmin_norm, vmax_norm = 0, 1

        mesh2 = self.ax_normalized.pcolormesh(
            theta_mesh,
            r_mesh,
            normalized,
            shading="auto",
            cmap="viridis",
            vmin=vmin_norm,
            vmax=vmax_norm,
        )
        self.ax_normalized.set_theta_zero_location("N")  # type: ignore[attr-defined]
        self.ax_normalized.set_theta_direction(-1)  # type: ignore[attr-defined]
        self.ax_normalized.set_title("I / Range_Std", fontsize=10)
        self._add_range_rings(self.ax_normalized, max_range)
        self.cbar_normalized = plt.colorbar(mesh2, ax=self.ax_normalized, shrink=0.7, pad=0.1)

        # Add wind arrow relative to ship
        self.ax_normalized.annotate(
            "",
            xy=(wind_relative_rad, arrow_len),
            xytext=(wind_relative_rad, max_range * 0.15),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )

        # =====================================================================
        # Plot 4: Std vs angle (original, outlier-removed, smoothed)
        # =====================================================================
        self.ax_std_angle.plot(
            theta, std_angle_orig, "b-", linewidth=1, alpha=0.4, label="Original"
        )
        self.ax_std_angle.plot(
            theta,
            std_angle_cleaned,
            "g-",
            linewidth=1,
            alpha=0.6,
            label=f"Outliers removed ({self.n_sigma}σ)",
        )
        self.ax_std_angle.plot(
            theta, std_angle_smoothed, "r-", linewidth=2, label=f"Smoothed (w={self.smooth_angle})"
        )
        self.ax_std_angle.set_xlabel("Theta (deg)", fontsize=10)
        self.ax_std_angle.set_ylabel("Std", fontsize=10)
        self.ax_std_angle.set_title("Std vs Angle (of I/Range_Std)", fontsize=10)
        self.ax_std_angle.grid(True, alpha=0.3)

        # Mark wind direction (relative to ship) and reciprocal on angle plot
        wind_rel_wrapped = wind_relative % 360
        wind_reciprocal = (wind_relative + 180) % 360
        self.ax_std_angle.axvline(
            wind_rel_wrapped,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Downwind ({wind_rel_wrapped:.0f}°)",
        )
        self.ax_std_angle.axvline(
            wind_reciprocal,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Upwind ({wind_reciprocal:.0f}°)",
        )
        # Update legend to include wind lines
        self.ax_std_angle.legend(loc="best", fontsize=8)

        # =====================================================================
        # Plot 5: Fully normalized (normalized / angular_std)
        # =====================================================================
        valid_fully = fully_normalized[np.isfinite(fully_normalized)]
        if len(valid_fully) > 0:
            vmin_full = float(np.percentile(valid_fully, 2))
            vmax_full = float(np.percentile(valid_fully, 98))
        else:
            vmin_full, vmax_full = 0, 1

        mesh3 = self.ax_fully_normalized.pcolormesh(
            theta_mesh,
            r_mesh,
            fully_normalized,
            shading="auto",
            cmap="viridis",
            vmin=vmin_full,
            vmax=vmax_full,
        )
        self.ax_fully_normalized.set_theta_zero_location("N")  # type: ignore[attr-defined]
        self.ax_fully_normalized.set_theta_direction(-1)  # type: ignore[attr-defined]
        self.ax_fully_normalized.set_title("(I/Range_Std) / Angle_Std", fontsize=10)
        self._add_range_rings(self.ax_fully_normalized, max_range)
        self.cbar_fully_normalized = plt.colorbar(
            mesh3, ax=self.ax_fully_normalized, shrink=0.7, pad=0.1
        )

        # Add wind arrow relative to ship
        self.ax_fully_normalized.annotate(
            "",
            xy=(wind_relative_rad, arrow_len),
            xytext=(wind_relative_rad, max_range * 0.15),
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
        )

        # =====================================================================
        # Update info text
        # =====================================================================
        frame_dt = frame.get("frame_datetime")
        dt_str = str(frame_dt)[:19] if frame_dt else "N/A"
        if shadow.is_valid:
            shadow_str = f"Shadow: {shadow.leading:.1f}-{shadow.trailing:.1f}°"
        else:
            shadow_str = "No shadow"

        info = (
            f"Frame {self.current_idx + 1}/{self.n_frames}  |  {dt_str}  |  "
            f"{shadow_str}  |  "
            f"Ship: {ship_speed:.1f} m/s @ {ship_heading:.0f}°  |  "
            f"Wind: {wind_speed:.1f} m/s from {wind_dir:.0f}° "
            f"({wind_relative:.0f}° relative)"
        )
        self.info_text.set_text(info)

        self.fig.suptitle(
            f"Intensity & Std Analysis - Frame {self.current_idx + 1}/{self.n_frames}",
            fontsize=12,
        )

        plt.tight_layout(rect=(0, 0.05, 1, 0.94))
        plt.draw()

    def _add_range_rings(self, ax: object, max_range: float) -> None:
        """Add 1km range rings to a polar plot."""
        ring_interval = 1000.0
        ring_radii = np.arange(ring_interval, max_range + ring_interval, ring_interval)
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        for r in ring_radii:
            ax.plot(  # type: ignore[attr-defined]
                theta_circle, np.full_like(theta_circle, r), "w--", linewidth=0.5, alpha=0.5
            )
        ax.set_rticks(ring_radii)  # type: ignore[attr-defined]

    def show(self) -> None:
        """Display the viewer."""
        import matplotlib.pyplot as plt

        plt.show()


class Tower_config:
    """Tower-specific configuration wrapper."""

    def __init__(self, config: dict) -> None:
        self._config = config or {}

    def __repr__(self) -> str:
        return f"Tower_config({self._config})"

    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


class Config:
    """Load YAML configuration file, if specified."""

    def __init__(self, filename: str) -> None:
        try:
            with open(filename) as f:
                self._config = yaml.safe_load(f)
        except Exception:
            self._config = {}
            logger.exception("Failed to load config file %s", filename)

    def __repr__(self) -> str:
        return f"Config({self._config})"

    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def refine(self, metadata: dict) -> Tower_config:
        """
        Refine configuration based on frame metadata's tower name.

        For now, just returns self. In practice, could adjust config based on metadata.
        """
        if "TOWER" not in metadata:
            return Tower_config({})
        tower = metadata["TOWER"].lower()
        if tower not in self._config:
            return Tower_config({})
        return Tower_config(self._config[tower])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="View intensity, std by range, and normalized intensity"
    )

    add_time_range_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )

    args = parser.parse_args()

    if not args.config:
        args.config = Path("default_wamos.yaml")

    setup_logging(args)

    try:
        config = None if args.config is None else Config(args.config)

        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        logger.info("Loading %d files with %d workers", n_files, args.workers or os.cpu_count())

        # Load frames in parallel
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_frame, fn, config): fn for fn in files}

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)

        n_loaded = len(results)
        print(f"Successfully loaded {n_loaded} of {n_files} frames")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Print timing summary
        total_polar = sum(r.get("time_polar_load", 0) for r in results)
        total_destreak = sum(r.get("time_destreak", 0) for r in results)
        total_theta = sum(r.get("time_theta_shadow", 0) for r in results)
        total_load = total_polar + total_destreak + total_theta

        print(f"\nLoading timing breakdown (total across {n_loaded} frames):")
        pct_polar = 100 * total_polar / total_load
        pct_destreak = 100 * total_destreak / total_load
        pct_theta = 100 * total_theta / total_load
        print(f"  PolarFrame load:    {total_polar:7.3f}s ({pct_polar:5.1f}%)")
        print(f"  Destreak:           {total_destreak:7.3f}s ({pct_destreak:5.1f}%)")
        print(f"  Theta+Shadow:       {total_theta:7.3f}s ({pct_theta:5.1f}%)")
        print(f"  Total:              {total_load:7.3f}s")

        # Sort by timestamp
        def sort_key(x: dict) -> str:
            dt = x.get("frame_datetime")
            return str(dt) if dt else ""

        results.sort(key=sort_key)

        # Launch viewer
        print("Launching viewer...")
        viewer = IntensityStdViewer(
            results,
            smooth_range=args.smooth_range,
            smooth_angle=args.smooth_angle,
            n_sigma=args.n_sigma,
        )
        viewer.show()

        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
