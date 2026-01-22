#! /usr/bin/env python3
#
# Shadow detection for WAMOS combined data
# Detects shadow region edges and computes offsets for bearing correction
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import Config
    from wamos_tpw.frame import Frame


def detect_shadow_edges(
    frame: "Frame", bearing: np.ndarray, config: "Config"
) -> tuple[float | None, float | None]:
    """
    Detect left and right shadow edges for a single frame.

    Uses intensity gradient analysis to find where the shadow region
    begins (left edge) and ends (right edge).

    Args:
        frame: Frame object with intensity data
        bearing: Bearing angles for this frame (degrees)
        config: Config with shadow settings

    Returns:
        Tuple of (left_edge, right_edge) in degrees, or None if not detected
    """
    from scipy.ndimage import gaussian_filter1d

    SHADOW_INTENSITY_BINS = 25
    GAUSSIAN_SIGMA = 3
    GRADIENT_THRESHOLD = 0.01

    shadow_cfg = config.get("shadow", {})
    expected_center = (
        shadow_cfg.get("center", 180.0)
        if isinstance(shadow_cfg, dict)
        else getattr(shadow_cfg, "center", 180.0)
    )
    search_range = (
        shadow_cfg.get("width", 90.0)
        if isinstance(shadow_cfg, dict)
        else getattr(shadow_cfg, "width", 90.0)
    )

    # Get intensity at first few distance bins (shadow is most visible there)
    intensity = frame.intensity[:, :SHADOW_INTENSITY_BINS].mean(axis=1).astype(float)

    # Sort by bearing for gradient calculation
    sort_idx = np.argsort(bearing)
    bearing_sorted = bearing[sort_idx]
    intensity_sorted = intensity[sort_idx]

    # Normalize intensity
    if intensity_sorted.max() > 0:
        intensity_norm = intensity_sorted / intensity_sorted.max()
    else:
        return None, None

    # Smooth intensity for edge detection
    intensity_smooth = gaussian_filter1d(intensity_norm, sigma=GAUSSIAN_SIGMA)

    # Calculate gradient
    gradient = np.gradient(intensity_smooth)

    # Search region around expected center
    search_start = (expected_center - search_range) % 360
    search_end = (expected_center + search_range) % 360

    if search_start < search_end:
        in_search = (bearing_sorted >= search_start) & (bearing_sorted <= search_end)
    else:
        in_search = (bearing_sorted >= search_start) | (bearing_sorted <= search_end)

    search_indices = np.where(in_search)[0]

    left_edge = None
    right_edge = None

    if len(search_indices) > 10:
        search_gradient = gradient[search_indices]

        # Left edge: most negative gradient (entering shadow)
        neg_grad_idx = np.argmin(search_gradient)
        if search_gradient[neg_grad_idx] < -GRADIENT_THRESHOLD:
            left_edge = bearing_sorted[search_indices[neg_grad_idx]]

        # Right edge: most positive gradient (exiting shadow)
        pos_grad_idx = np.argmax(search_gradient)
        if search_gradient[pos_grad_idx] > GRADIENT_THRESHOLD:
            right_edge = bearing_sorted[search_indices[pos_grad_idx]]

    return left_edge, right_edge


def compute_chunk_shadow_offset(
    chunk_frames: list, theta, config: "Config"
) -> tuple[float, float | None, float | None]:
    """
    Compute shadow offset for a chunk of frames.

    Analyzes multiple frames to get a robust estimate of the shadow
    position offset from the expected configuration values.

    Args:
        chunk_frames: List of frames in the chunk
        theta: Theta object with bearing calculations
        config: Config with shadow settings

    Returns:
        Tuple of (offset, shadow_left_mean, shadow_right_mean)
        - offset: bearing offset to apply (degrees)
        - shadow_left_mean: mean left edge angle (degrees)
        - shadow_right_mean: mean right edge angle (degrees)
    """
    shadow_cfg = config.get("shadow", {})
    expected_center = (
        shadow_cfg.get("center", 180.0)
        if isinstance(shadow_cfg, dict)
        else getattr(shadow_cfg, "center", 180.0)
    )

    left_edges = []
    right_edges = []

    for j, frame in enumerate(chunk_frames):
        bearing = theta.bearing_for_frame(j)
        left_edge, right_edge = detect_shadow_edges(frame, bearing, config)
        if left_edge is not None:
            left_edges.append(left_edge)
        if right_edge is not None:
            right_edges.append(right_edge)

    if not left_edges or not right_edges:
        # Fall back to config values
        return 0.0, config.shadow.start, config.shadow.end

    # Circular mean for left and right edges
    def circular_mean(angles):
        angles_rad = np.deg2rad(angles)
        mean_x = np.mean(np.cos(angles_rad))
        mean_y = np.mean(np.sin(angles_rad))
        return np.rad2deg(np.arctan2(mean_y, mean_x)) % 360

    mean_left = circular_mean(left_edges)
    mean_right = circular_mean(right_edges)

    # Calculate center from mean edges
    if mean_right < mean_left:
        width = mean_right + 360 - mean_left
    else:
        width = mean_right - mean_left
    detected_center = (mean_left + width / 2) % 360

    # Calculate offset from expected
    offset = detected_center - expected_center
    if offset > 180:
        offset -= 360
    elif offset < -180:
        offset += 360

    return offset, mean_left, mean_right
