#! /usr/bin/env python3
#
# Theta and Bearing classes for calculating radar beam angles
# Theta: beam angle relative to radar/ship
# Bearing: beam angle in earth coordinates with cartesian projections
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from wamos_tpw.config import Config
from wamos_tpw.frame import Frame
from wamos_tpw.plotting import (
    calc_bin_edges,
    format_nav_title,
    add_crosshairs,
    add_range_rings,
    sort_polar_data,
    get_radar_height,
)


class Theta:
    """
    Calculate radar beam angle (theta) relative to ship for contiguous frames.

    Uses bit 13 encoding in the first distance bin to determine degree transitions,
    then optionally refines the estimate using the shadow region alignment.

    Algorithm Overview
    ------------------

    **Step 1: Bearing Extraction from Bit 13**

    The WAMOS radar encodes bearing information in bit 13 of the first distance bin:
    - Bit 13 = 0: radial is within an even degree (e.g., 44.0° to 45.0°)
    - Bit 13 = 1: radial is within an odd degree (e.g., 45.0° to 46.0°)

    The algorithm detects transitions in bit 13 to identify degree boundaries, then
    interpolates bearing values within each segment. Missing transitions (due to
    noise or dropouts) are detected by comparing segment sizes to the median and
    synthetically inserted.

    **Step 2: Shadow Refinement (Optional)**

    The ship's superstructure creates a radar shadow typically centered at 180°
    (stern). The algorithm refines the bearing estimate by:

    1. For each frame, compute mean intensity in the first 25 range bins vs bearing
    2. Apply Gaussian smoothing (σ=3) to the intensity profile
    3. Calculate the intensity gradient (derivative)
    4. Detect the left edge (most negative gradient = entering shadow)
    5. Detect the right edge (most positive gradient = exiting shadow)
    6. Compute the shadow center from the mean of detected edges
    7. Calculate offset between detected and expected shadow center
    8. Apply the offset correction to all bearings

    The refinement requires a minimum number of frames (default: 3) with successfully
    detected edges. Edge detection uses a gradient threshold of 0.01 to reject
    noise. Results are reported using circular statistics (mean and standard
    deviation) to properly handle the 360°/0° wraparound.

    All bearing values are wrapped to [0, 360).

    Example:
        >>> config = Config('radar_config.yaml')
        >>> theta = Theta(frames, config)
        >>> bearings = theta.bearing  # Array of bearing angles per radial [0, 360)
        >>> print(f"Shadow offset correction: {theta.shadow_offset:.2f}°")
    """

    # Bit mask for bearing pulse (bit 13)
    _MASK_BIT13 = np.uint16(0x2000)

    # Algorithm constants
    _SEGMENT_SUSPICIOUS_MULTIPLIER = 1.8  # Flag segments > 1.8x median as suspicious
    _SMOOTHING_WEIGHTS = (0.6, 0.2, 0.2)  # Center, prev, next for adaptive smoothing
    _GRADIENT_THRESHOLD = 0.01  # Minimum gradient magnitude for edge detection
    _SHADOW_INTENSITY_BINS = 25  # Number of distance bins for shadow analysis
    _GAUSSIAN_SIGMA = 3  # Sigma for intensity smoothing in edge detection

    def __init__(self, frames: list[Frame], config: Config | None = None, refine: bool = True):
        """
        Initialize Theta calculator for a set of contiguous frames.

        Args:
            frames: List of contiguous Frame objects
            config: Config object (uses defaults if None)
            refine: Whether to refine bearing using shadow alignment
        """
        if not frames:
            raise ValueError("At least one frame is required")

        self._frames = frames
        self._config = config or Config()
        theta_refine_cfg = self._config.get("theta_refinement", {})
        refine_enabled = theta_refine_cfg.get("enabled", False) if isinstance(theta_refine_cfg, dict) else getattr(theta_refine_cfg, "enabled", False)
        self._refine = refine and refine_enabled

        # Calculated values (computed lazily)
        self._bearing: np.ndarray | None = None
        self._bearing_per_frame: list[np.ndarray] | None = None
        self._shadow_offset: float = 0.0
        self._shadow_quality: float = 0.0

        # Shadow edge detection results
        self._shadow_data: list[dict] = []
        self._shadow_left_edges: list[float] = []
        self._shadow_right_edges: list[float] = []
        self._shadow_left_mean: float | None = None
        self._shadow_left_std: float | None = None
        self._shadow_right_mean: float | None = None
        self._shadow_right_std: float | None = None

        # Calculate bearing
        self._calculate()

    def _calculate(self) -> None:
        """Calculate bearing angles for all frames."""
        self._bearing_per_frame = []

        for i, frame in enumerate(self._frames):
            # Calculate bearing for this frame
            bearing = self._calculate_frame_bearing(frame)
            # Wrap to [0, 360)
            bearing = bearing % 360
            self._bearing_per_frame.append(bearing)

        # Concatenate all bearings (each frame is 0-360)
        self._bearing = np.concatenate(self._bearing_per_frame)

        # Refine using shadow region if enabled
        if self._refine:
            theta_refine_cfg = self._config.get("theta_refinement", {})
            min_frames = theta_refine_cfg.get("min_frames", 3) if isinstance(theta_refine_cfg, dict) else getattr(theta_refine_cfg, "min_frames", 3)
            if len(self._frames) >= min_frames:
                self._refine_with_shadow()

    def _calculate_frame_bearing(self, frame: Frame) -> np.ndarray:
        """
        Calculate bearing angles for a single frame using bit 13 transitions.

        Args:
            frame: Frame object

        Returns:
            Array of bearing angles for each radial (may exceed 360)
        """
        data = frame.raw
        n_radials = frame.n_bearings

        # Step 1: Extract bit 13 transitions
        bit_13 = (data[:, 0] & self._MASK_BIT13) != 0
        transitions = self._extract_transitions(bit_13, n_radials)

        # Step 2: Fix missing transitions
        transitions = self._fix_missing_transitions(transitions)

        # Step 3: Interpolate bearing values
        return self._interpolate_bearings(transitions, bit_13[0], n_radials)

    def _extract_transitions(self, bit_13: np.ndarray, n_radials: int) -> np.ndarray:
        """
        Extract degree transition points from bit 13 pattern.

        Args:
            bit_13: Boolean array of bit 13 values for each radial
            n_radials: Total number of radials

        Returns:
            Array of transition indices including boundaries
        """
        # Find all degree transitions (bit changes)
        transitions = np.where(np.diff(bit_13.astype(int)) != 0)[0] + 1

        # Add boundaries (start and end)
        return np.concatenate([[0], transitions, [n_radials]])

    def _fix_missing_transitions(self, transitions: np.ndarray) -> np.ndarray:
        """
        Detect and insert missing transitions based on segment size analysis.

        Args:
            transitions: Array of transition indices

        Returns:
            Updated transitions array with missing transitions inserted
        """
        if len(transitions) <= 2:
            return transitions

        segment_sizes = np.diff(transitions)
        median_size = np.median(segment_sizes)

        # Find segments that are likely missing a transition
        suspicious = np.where(segment_sizes > median_size * self._SEGMENT_SUSPICIOUS_MULTIPLIER)[0]

        new_transitions = []
        for seg_idx in suspicious:
            start_idx = transitions[seg_idx]
            end_idx = transitions[seg_idx + 1]
            segment_size = end_idx - start_idx

            # Estimate number of missing transitions
            expected_segments = int(np.round(segment_size / median_size))
            if expected_segments > 1:
                for j in range(1, expected_segments):
                    insert_idx = start_idx + int(j * segment_size / expected_segments)
                    new_transitions.append(insert_idx)

        if new_transitions:
            return np.sort(np.concatenate([transitions, new_transitions]))

        return transitions

    def _interpolate_bearings(
        self, transitions: np.ndarray, first_bit_odd: bool, n_radials: int
    ) -> np.ndarray:
        """
        Interpolate bearing values within each transition segment.

        Args:
            transitions: Array of transition indices
            first_bit_odd: Whether first radial has odd degree (bit 13 = 1)
            n_radials: Total number of radials

        Returns:
            Array of bearing angles for each radial
        """
        bearing = np.zeros(n_radials)

        # Determine starting degree based on bit 13 parity
        current_degree = 1.0 if first_bit_odd else 0.0

        # Calculate nominal radials per degree
        n_segments = len(transitions) - 1
        avg_radials_per_degree = n_radials / n_segments if n_segments > 0 else n_radials / 360

        # Process each segment with adaptive width
        for i in range(len(transitions) - 1):
            start_idx = transitions[i]
            end_idx = transitions[i + 1]
            n_radials_in_segment = end_idx - start_idx

            # Apply smoothing for interior segments
            if i > 0 and i < len(transitions) - 2:
                prev_size = transitions[i] - transitions[i - 1]
                next_size = transitions[i + 2] - transitions[i + 1]
                w = self._SMOOTHING_WEIGHTS
                smoothed_size = w[0] * n_radials_in_segment + w[1] * prev_size + w[2] * next_size
                degree_width = smoothed_size / avg_radials_per_degree
            else:
                degree_width = n_radials_in_segment / avg_radials_per_degree

            # Distribute bearing within segment
            sub_bearings = (
                current_degree
                + (np.arange(n_radials_in_segment) + 0.5) / n_radials_in_segment * degree_width
            )
            bearing[start_idx:end_idx] = sub_bearings

            current_degree += degree_width

        return bearing

    def _refine_with_shadow(self) -> None:
        """
        Refine bearing estimate using shadow edge detection across frames.

        Detects left and right edges of the shadow region using intensity
        gradients, then calculates the center and applies offset correction.
        """
        # Step 1: Detect shadow edges for all frames
        left_edges, right_edges = self._detect_all_shadow_edges()

        # Step 2: Compute statistics from detected edges
        self._compute_shadow_statistics(left_edges, right_edges)

        # Step 3: Apply correction if sufficient detections
        self._apply_shadow_correction(left_edges, right_edges)

    def _detect_all_shadow_edges(self) -> Tuple[list[float], list[float]]:
        """
        Detect shadow edges for all frames.

        Returns:
            Tuple of (left_edges, right_edges) lists
        """
        self._shadow_data = []
        left_edges = []
        right_edges = []

        shadow_cfg = self._config.get("shadow", {})
        expected_center = shadow_cfg.get("center", 180.0) if isinstance(shadow_cfg, dict) else getattr(shadow_cfg, "center", 180.0)
        search_range = shadow_cfg.get("width", 90.0) if isinstance(shadow_cfg, dict) else getattr(shadow_cfg, "width", 90.0)

        for frame_idx, frame in enumerate(self._frames):
            bearing = self._bearing_per_frame[frame_idx]
            edge_data = self._detect_frame_shadow_edges(
                frame, bearing, frame_idx, expected_center, search_range
            )

            if edge_data["left_edge"] is not None:
                left_edges.append(edge_data["left_edge"])
            if edge_data["right_edge"] is not None:
                right_edges.append(edge_data["right_edge"])

            self._shadow_data.append(edge_data)

        self._shadow_left_edges = left_edges
        self._shadow_right_edges = right_edges

        return left_edges, right_edges

    def _detect_frame_shadow_edges(
        self,
        frame: Frame,
        bearing: np.ndarray,
        frame_idx: int,
        expected_center: float,
        search_range: float,
    ) -> dict:
        """
        Detect shadow edges for a single frame using intensity gradients.

        Args:
            frame: Frame object
            bearing: Bearing array for this frame
            frame_idx: Index of the frame
            expected_center: Expected shadow center in degrees
            search_range: Angular range to search around expected center

        Returns:
            Dictionary with edge detection results and diagnostic data
        """
        from scipy.ndimage import gaussian_filter1d

        # Get intensity at first few distance bins (shadow is most visible there)
        intensity = frame.intensity[:, : self._SHADOW_INTENSITY_BINS].mean(axis=1).astype(float)

        # Sort by bearing for gradient calculation
        sort_idx = np.argsort(bearing)
        bearing_sorted = bearing[sort_idx]
        intensity_sorted = intensity[sort_idx]

        # Normalize intensity
        intensity_norm = (
            intensity_sorted / intensity_sorted.max()
            if intensity_sorted.max() > 0
            else intensity_sorted
        )

        # Smooth intensity for edge detection
        intensity_smooth = gaussian_filter1d(intensity_norm, sigma=self._GAUSSIAN_SIGMA)

        # Calculate gradient (derivative)
        gradient = np.gradient(intensity_smooth)

        # Find edges in search region
        left_edge, right_edge, left_edge_idx, right_edge_idx = self._find_edges_in_region(
            bearing_sorted, gradient, expected_center, search_range
        )

        # Calculate center from edges if both found
        detected_center = None
        detected_width = None
        if left_edge is not None and right_edge is not None:
            right_edge_unwrap = right_edge + 360 if right_edge < left_edge else right_edge
            detected_center = ((left_edge + right_edge_unwrap) / 2) % 360
            detected_width = (right_edge_unwrap - left_edge) % 360

        return {
            "frame_idx": frame_idx,
            "bearing_sorted": bearing_sorted,
            "intensity_norm": intensity_norm,
            "intensity_smooth": intensity_smooth,
            "gradient": gradient,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "left_edge_idx": left_edge_idx,
            "right_edge_idx": right_edge_idx,
            "detected_center": detected_center,
            "detected_width": detected_width,
            "timestamp": frame.timestamp,
        }

    def _find_edges_in_region(
        self,
        bearing_sorted: np.ndarray,
        gradient: np.ndarray,
        expected_center: float,
        search_range: float,
    ) -> Tuple[float | None, float | None, int | None, int | None]:
        """
        Find left and right shadow edges within search region.

        Args:
            bearing_sorted: Sorted bearing array
            gradient: Intensity gradient array
            expected_center: Expected shadow center
            search_range: Angular range to search

        Returns:
            Tuple of (left_edge, right_edge, left_edge_idx, right_edge_idx)
        """
        # Define search region
        search_start = (expected_center - search_range) % 360
        search_end = (expected_center + search_range) % 360

        if search_start < search_end:
            in_search = (bearing_sorted >= search_start) & (bearing_sorted <= search_end)
        else:
            in_search = (bearing_sorted >= search_start) | (bearing_sorted <= search_end)

        search_indices = np.where(in_search)[0]

        left_edge = None
        right_edge = None
        left_edge_idx = None
        right_edge_idx = None

        if len(search_indices) > 10:
            search_gradient = gradient[search_indices]

            # Left edge: most negative gradient (entering shadow)
            neg_grad_idx = np.argmin(search_gradient)
            if search_gradient[neg_grad_idx] < -self._GRADIENT_THRESHOLD:
                left_edge_idx = search_indices[neg_grad_idx]
                left_edge = bearing_sorted[left_edge_idx]

            # Right edge: most positive gradient (exiting shadow)
            pos_grad_idx = np.argmax(search_gradient)
            if search_gradient[pos_grad_idx] > self._GRADIENT_THRESHOLD:
                right_edge_idx = search_indices[pos_grad_idx]
                right_edge = bearing_sorted[right_edge_idx]

        return left_edge, right_edge, left_edge_idx, right_edge_idx

    def _compute_shadow_statistics(self, left_edges: list[float], right_edges: list[float]) -> None:
        """
        Compute circular statistics for detected shadow edges.

        Args:
            left_edges: List of detected left edge angles
            right_edges: List of detected right edge angles
        """
        if left_edges:
            self._shadow_left_mean, self._shadow_left_std = self._circular_stats(left_edges)
        else:
            self._shadow_left_mean = None
            self._shadow_left_std = None

        if right_edges:
            self._shadow_right_mean, self._shadow_right_std = self._circular_stats(right_edges)
        else:
            self._shadow_right_mean = None
            self._shadow_right_std = None

    def _apply_shadow_correction(self, left_edges: list[float], right_edges: list[float]) -> None:
        """
        Apply shadow-based bearing correction if sufficient detections available.

        Args:
            left_edges: List of detected left edge angles
            right_edges: List of detected right edge angles
        """
        theta_refine_cfg = self._config.get("theta_refinement", {})
        min_frames = theta_refine_cfg.get("min_frames", 3) if isinstance(theta_refine_cfg, dict) else getattr(theta_refine_cfg, "min_frames", 3)
        shadow_cfg = self._config.get("shadow", {})
        expected_center = shadow_cfg.get("center", 180.0) if isinstance(shadow_cfg, dict) else getattr(shadow_cfg, "center", 180.0)

        if len(left_edges) < min_frames or len(right_edges) < min_frames:
            logging.warning(
                f"Insufficient shadow detections (LHS={len(left_edges)}, RHS={len(right_edges)}) "
                f"for refinement (need {min_frames})"
            )
            return

        mean_left = self._shadow_left_mean
        mean_right = self._shadow_right_mean

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

        self._shadow_offset = offset
        self._shadow_quality = (self._shadow_left_std + self._shadow_right_std) / 2

        # Apply correction to all bearings and rewrap to [0, 360)
        for i in range(len(self._bearing_per_frame)):
            self._bearing_per_frame[i] = (self._bearing_per_frame[i] - offset) % 360

        self._bearing = np.concatenate(self._bearing_per_frame)

        logging.info(
            f"Shadow refinement: LHS={mean_left:.1f}°±{self._shadow_left_std:.1f}°, "
            f"RHS={mean_right:.1f}°±{self._shadow_right_std:.1f}°, "
            f"offset={offset:.2f}°"
        )

    @staticmethod
    def _circular_stats(angles: list[float]) -> Tuple[float, float]:
        """
        Calculate circular mean and standard deviation of angles in degrees.

        Returns both values efficiently by computing shared intermediate values once.

        Args:
            angles: List of angles in degrees

        Returns:
            Tuple of (mean, std) in degrees
        """
        angles_rad = np.deg2rad(angles)
        mean_x = np.mean(np.cos(angles_rad))
        mean_y = np.mean(np.sin(angles_rad))

        # Circular mean
        mean_angle = np.rad2deg(np.arctan2(mean_y, mean_x)) % 360

        # Circular std (from mean resultant length)
        R = np.sqrt(mean_x**2 + mean_y**2)
        if R >= 1:
            std_angle = 0.0
        else:
            std_angle = np.rad2deg(np.sqrt(-2 * np.log(R)))

        return mean_angle, std_angle

    @staticmethod
    def _circular_mean(angles: list[float]) -> float:
        """Calculate circular mean of angles in degrees."""
        mean_angle, _ = Theta._circular_stats(angles)
        return mean_angle

    @staticmethod
    def _circular_std(angles: list[float]) -> float:
        """Calculate circular standard deviation of angles in degrees."""
        _, std_angle = Theta._circular_stats(angles)
        return std_angle

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def frames(self) -> list[Frame]:
        """Return the frames."""
        return self._frames

    @property
    def bearing(self) -> np.ndarray:
        """
        Get bearing angles for all radials across all frames.

        Returns:
            Array of bearing angles in degrees [0, 360)
        """
        return self._bearing

    @property
    def bearing_per_frame(self) -> list[np.ndarray]:
        """
        Get bearing angles separated by frame.

        Returns:
            List of arrays, one per frame, each in [0, 360)
        """
        return self._bearing_per_frame

    def bearing_for_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get bearing angles for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Array of bearing angles in degrees [0, 360)
        """
        return self._bearing_per_frame[frame_idx]

    @property
    def shadow_offset(self) -> float:
        """Get the shadow-based bearing offset correction applied."""
        return self._shadow_offset

    @property
    def shadow_quality(self) -> float:
        """Get the shadow alignment quality metric (std dev, lower is better)."""
        return self._shadow_quality

    @property
    def shadow_left_mean(self) -> float | None:
        """Get the mean left edge of shadow region."""
        return self._shadow_left_mean

    @property
    def shadow_left_std(self) -> float | None:
        """Get the std dev of left edge of shadow region."""
        return self._shadow_left_std

    @property
    def shadow_right_mean(self) -> float | None:
        """Get the mean right edge of shadow region."""
        return self._shadow_right_mean

    @property
    def shadow_right_std(self) -> float | None:
        """Get the std dev of right edge of shadow region."""
        return self._shadow_right_std

    @property
    def shadow_stats(self) -> str | None:
        """Get formatted shadow statistics string, or None if not available."""
        if self._shadow_left_mean is not None and self._shadow_right_mean is not None:
            return (
                f"LHS: {self._shadow_left_mean:.2f}°±{self._shadow_left_std:.2f}°  "
                f"RHS: {self._shadow_right_mean:.2f}°±{self._shadow_right_std:.2f}°  "
                f"(n={len(self._shadow_left_edges)})"
            )
        return None

    def plot_shadow_diagnostics(self) -> None:
        """
        Display diagnostic plots for shadow edge detection.

        Shows per-frame intensity profiles with detected edges and
        summary statistics across all frames.
        """
        import matplotlib.pyplot as plt

        if not self._shadow_data:
            logging.warning("No shadow data available (refinement may be disabled)")
            return

        shadow_cfg = self._config.get("shadow", {})
        expected_center = shadow_cfg.get("center", 180.0) if isinstance(shadow_cfg, dict) else getattr(shadow_cfg, "center", 180.0)
        expected_width = shadow_cfg.get("width", 90.0) if isinstance(shadow_cfg, dict) else getattr(shadow_cfg, "width", 90.0)
        search_range = expected_width

        # Create figure with subplots
        n_frames = len(self._frames)
        n_cols = min(3, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols + 2  # Extra rows for summary

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle("Shadow Edge Detection Diagnostics", fontsize=14, fontweight="bold")

        # Plot each frame's intensity vs bearing with edge detection
        for i, data in enumerate(self._shadow_data):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            bearing = data["bearing_sorted"]
            intensity_norm = data["intensity_norm"]
            intensity_smooth = data["intensity_smooth"]

            # Plot raw and smoothed intensity
            ax.plot(bearing, intensity_norm, "b-", linewidth=0.5, alpha=0.5, label="Raw")
            ax.plot(bearing, intensity_smooth, "b-", linewidth=1.5, label="Smoothed")

            # Mark expected shadow region (lighter)
            ax.axvspan(
                expected_center - expected_width / 2,
                expected_center + expected_width / 2,
                alpha=0.1,
                color="gray",
                label="Expected region",
            )
            ax.axvline(expected_center, color="gray", linestyle=":", linewidth=1)

            # Mark detected edges
            if data["left_edge"] is not None:
                ax.axvline(
                    data["left_edge"],
                    color="red",
                    linestyle="-",
                    linewidth=2,
                    label=f"Left edge ({data['left_edge']:.1f}°)",
                )
                ax.scatter(
                    [data["left_edge"]],
                    [intensity_smooth[data["left_edge_idx"]]],
                    color="red",
                    s=80,
                    zorder=5,
                    marker="<",
                )

            if data["right_edge"] is not None:
                ax.axvline(
                    data["right_edge"],
                    color="blue",
                    linestyle="-",
                    linewidth=2,
                    label=f"Right edge ({data['right_edge']:.1f}°)",
                )
                ax.scatter(
                    [data["right_edge"]],
                    [intensity_smooth[data["right_edge_idx"]]],
                    color="blue",
                    s=80,
                    zorder=5,
                    marker=">",
                )

            # Mark detected center
            if data["detected_center"] is not None:
                ax.axvline(
                    data["detected_center"],
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Center ({data['detected_center']:.1f}°)",
                )

            ax.set_xlabel("Bearing (°)")
            ax.set_ylabel("Normalized Intensity")
            ax.set_xlim(expected_center - search_range - 10, expected_center + search_range + 10)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, alpha=0.3)

        # Summary plot 1: Edge positions across frames
        ax_edges = fig.add_subplot(n_rows, 1, n_rows - 1)

        left_edges = [
            (d["frame_idx"], d["left_edge"])
            for d in self._shadow_data
            if d["left_edge"] is not None
        ]
        right_edges = [
            (d["frame_idx"], d["right_edge"])
            for d in self._shadow_data
            if d["right_edge"] is not None
        ]
        centers = [
            (d["frame_idx"], d["detected_center"])
            for d in self._shadow_data
            if d["detected_center"] is not None
        ]

        if left_edges:
            ax_edges.scatter(*zip(*left_edges), s=60, c="red", marker="<", label="Left edges")
            if self._shadow_left_mean is not None:
                ax_edges.axhline(
                    self._shadow_left_mean, color="red", linestyle="-", linewidth=1, alpha=0.7
                )

        if right_edges:
            ax_edges.scatter(*zip(*right_edges), s=60, c="blue", marker=">", label="Right edges")
            if self._shadow_right_mean is not None:
                ax_edges.axhline(
                    self._shadow_right_mean, color="blue", linestyle="-", linewidth=1, alpha=0.7
                )

        if centers:
            ax_edges.scatter(
                *zip(*centers), s=60, c="green", marker="o", label="Centers (from edges)"
            )
            mean_center = self._circular_mean([c[1] for c in centers])
            ax_edges.axhline(mean_center, color="green", linestyle="--", linewidth=2)

        ax_edges.axhline(
            expected_center,
            color="gray",
            linestyle=":",
            linewidth=2,
            label=f"Expected ({expected_center}°)",
        )

        ax_edges.set_xlabel("Frame Index")
        ax_edges.set_ylabel("Bearing (°)")
        ax_edges.legend(loc="upper right")
        ax_edges.grid(True, alpha=0.3)

        # Summary plot 2: Statistics
        ax_stats = fig.add_subplot(n_rows, 1, n_rows)

        stats_text = []
        if self._shadow_left_mean is not None:
            stats_text.append(
                f"Left Edge:  mean={self._shadow_left_mean:.2f}°, std={self._shadow_left_std:.2f}°, n={len(self._shadow_left_edges)}"
            )

        if self._shadow_right_mean is not None:
            stats_text.append(
                f"Right Edge: mean={self._shadow_right_mean:.2f}°, std={self._shadow_right_std:.2f}°, n={len(self._shadow_right_edges)}"
            )

        if centers:
            center_vals = [c[1] for c in centers]
            mean_center, std_center = self._circular_stats(center_vals)
            stats_text.append(
                f"Center:     mean={mean_center:.2f}°, std={std_center:.2f}°, n={len(centers)}"
            )

        if self._shadow_left_mean is not None and self._shadow_right_mean is not None:
            mean_left = self._shadow_left_mean
            mean_right = self._shadow_right_mean
            if mean_right < mean_left:
                width = mean_right + 360 - mean_left
            else:
                width = mean_right - mean_left
            computed_center = (mean_left + width / 2) % 360
            offset = computed_center - expected_center
            if offset > 180:
                offset -= 360
            elif offset < -180:
                offset += 360
            stats_text.append("\nComputed from edges:")
            stats_text.append(f"  Shadow width: {width:.2f}°")
            stats_text.append(f"  Shadow center: {computed_center:.2f}°")
            stats_text.append(f"  Offset from expected ({expected_center}°): {offset:.2f}°")

        ax_stats.text(
            0.05,
            0.95,
            "\n".join(stats_text),
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis("off")

        plt.tight_layout()
        plt.show()

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    def in_shadow(self, frame_idx: int | None = None) -> np.ndarray:
        """
        Get boolean mask indicating radials in the shadow region.

        Args:
            frame_idx: If provided, return mask for specific frame only

        Returns:
            Boolean array (True = in shadow region)
        """
        if frame_idx is not None:
            bearing = self._bearing_per_frame[frame_idx]
        else:
            bearing = self._bearing

        shadow_start = self._config.shadow.start
        shadow_end = self._config.shadow.end

        if shadow_start < shadow_end:
            return (bearing >= shadow_start) & (bearing <= shadow_end)
        else:
            # Wrap around 360
            return (bearing >= shadow_start) | (bearing <= shadow_end)

    def __len__(self) -> int:
        """Return total number of radials across all frames."""
        return len(self._bearing)

    def __repr__(self) -> str:
        return (
            f"Theta(frames={len(self._frames)}, "
            f"radials={len(self)}, "
            f"shadow_offset={self._shadow_offset:.2f}°)"
        )

    def clear_shadow_data(self) -> None:
        """
        Clear shadow diagnostic data to free memory.

        Call this after theta refinement is complete if you don't need
        to call plot_shadow_diagnostics(). Frees ~600KB per 1000 frames.
        """
        self._shadow_data = []
        logging.debug("Shadow diagnostic data cleared")

    def __enter__(self) -> "Theta":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - clears shadow data to free memory."""
        self.clear_shadow_data()


class Bearing:
    """
    Calculate radar beam angles in ship and earth reference frames with cartesian coordinates.

    Converts theta (radar-relative angle) to:
    - Ship-relative heading: theta + BO2RA (bow-to-radar angle)
    - Image-adjusted heading: + HDGDL (heading delay for start of image)
    - Earth heading: + GYROC (gyro compass heading)

    Provides x/y cartesian coordinates for each radar pixel in both ship and earth frames.

    Coordinate calculations are cached/memoized for performance. Use clear_cache()
    to free memory if needed.

    Example:
        >>> config = Config('radar_config.yaml')
        >>> theta = Theta(frames, config)
        >>> bearing = Bearing(theta, radar_height=25.0)
        >>> x_ship, y_ship = bearing.xy_ship(frame_idx=0)
        >>> x_earth, y_earth = bearing.xy_earth(frame_idx=0)
    """

    def __init__(
        self, theta: Theta, radar_height: float | None = None, cache_coordinates: bool = True
    ):
        """
        Initialize Bearing calculator.

        Args:
            theta: Theta object with calculated beam angles
            radar_height: Radar height above water (m). If None, uses config or metadata.
            cache_coordinates: If True, cache xy_ship/xy_earth results (default True)
        """
        self._theta = theta
        self._config = theta.config
        self._frames = theta.frames
        self._cache_enabled = cache_coordinates

        # Coordinate caches
        self._xy_ship_cache: dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._xy_earth_cache: dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._heading_ship_cache: dict[int, np.ndarray] = {}
        self._heading_earth_cache: dict[int, np.ndarray] = {}

        # Determine radar height: parameter > config > metadata > wind sensor height
        if radar_height is not None:
            self._radar_height = radar_height
        elif self._config.get("radar.height") is not None:
            self._radar_height = self._config.get("radar.height")
        elif self._config.get("tower.height") is not None:
            self._radar_height = self._config.get("tower.height")
        elif self._frames[0].metadata.radar_height is not None:
            self._radar_height = self._frames[0].metadata.radar_height
        else:
            # Fall back to wind sensor height
            self._radar_height = self._frames[0].metadata.wind_sensor_height

        logging.debug(
            f"Bearing initialized: {len(self._frames)} frames, "
            f"radar_height={self._radar_height}, cache={cache_coordinates}"
        )

    @property
    def theta(self) -> Theta:
        """Return the underlying Theta object."""
        return self._theta

    @property
    def config(self) -> Config:
        """Return the configuration."""
        return self._config

    @property
    def radar_height(self) -> float | None:
        """Return the radar height."""
        return self._radar_height

    def _get_radar_height(self, frame_idx: int) -> float | None:
        """Get radar height with fallback to frame metadata."""
        return get_radar_height(self._radar_height, self._frames[frame_idx])

    def clear_cache(self) -> None:
        """Clear all cached coordinate calculations to free memory."""
        self._xy_ship_cache.clear()
        self._xy_earth_cache.clear()
        self._heading_ship_cache.clear()
        self._heading_earth_cache.clear()
        logging.debug("Bearing cache cleared")

    def heading_ship(self, frame_idx: int) -> np.ndarray:
        """
        Get beam heading relative to ship bow.

        heading_ship = theta + BO2RA

        Args:
            frame_idx: Frame index

        Returns:
            Array of ship-relative headings in degrees [0, 360)
        """
        if self._cache_enabled and frame_idx in self._heading_ship_cache:
            return self._heading_ship_cache[frame_idx]

        theta = self._theta.bearing_for_frame(frame_idx)
        bo2ra = self._config.offsets.bow_to_radar
        result = (theta + bo2ra) % 360

        if self._cache_enabled:
            self._heading_ship_cache[frame_idx] = result
        return result

    def heading_image(self, frame_idx: int) -> np.ndarray:
        """
        Get beam heading adjusted for image start.

        heading_image = theta + BO2RA + HDGDL

        Args:
            frame_idx: Frame index

        Returns:
            Array of image-adjusted headings in degrees [0, 360)
        """
        theta = self._theta.bearing_for_frame(frame_idx)
        bo2ra = self._config.offsets.bow_to_radar
        hdgdl = self._config.offsets.heading_delay
        return (theta + bo2ra + hdgdl) % 360

    def heading_earth(self, frame_idx: int) -> np.ndarray:
        """
        Get beam heading in earth coordinates (true heading).

        heading_earth = theta + BO2RA + HDGDL + GYROC

        Args:
            frame_idx: Frame index

        Returns:
            Array of earth headings in degrees [0, 360), where 0=North, 90=East
        """
        if self._cache_enabled and frame_idx in self._heading_earth_cache:
            return self._heading_earth_cache[frame_idx]

        frame = self._frames[frame_idx]
        theta = self._theta.bearing_for_frame(frame_idx)
        bo2ra = self._config.offsets.bow_to_radar
        hdgdl = self._config.offsets.heading_delay
        gyroc = frame.metadata.heading or 0.0
        compass_offset = self._config.offsets.compass

        result = (theta + bo2ra + hdgdl + gyroc + compass_offset) % 360

        if self._cache_enabled:
            self._heading_earth_cache[frame_idx] = result
        return result

    def _heading_to_xy(self, frame_idx: int, heading: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert heading angles to x/y cartesian coordinates.

        Common implementation for xy_ship and xy_earth.
        Convention: +Y = heading 0°, +X = heading 90°

        Args:
            frame_idx: Frame index (for range calculation)
            heading: Array of heading angles in degrees

        Returns:
            Tuple of (x, y) arrays, each shape (n_bearings, n_distances)
        """
        frame = self._frames[frame_idx]
        height = self._get_radar_height(frame_idx)
        range_vals = frame.ground_range(height) if height else frame.slant_range()

        # Convert to radians
        heading_rad = np.deg2rad(heading)

        # Create 2D grids: heading is (n_bearings,), range is (n_distances,)
        heading_2d = heading_rad[:, np.newaxis]
        range_2d = range_vals[np.newaxis, :]

        # x = range * sin(heading), y = range * cos(heading)
        x = range_2d * np.sin(heading_2d)
        y = range_2d * np.cos(heading_2d)

        return x, y

    def xy_ship(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get x/y cartesian coordinates in ship reference frame.

        Ship frame: +X = starboard, +Y = bow (forward)
        Origin at ship center.

        Results are cached for performance.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (x, y) arrays, each shape (n_bearings, n_distances)
            in meters relative to ship center
        """
        if self._cache_enabled and frame_idx in self._xy_ship_cache:
            return self._xy_ship_cache[frame_idx]

        result = self._heading_to_xy(frame_idx, self.heading_ship(frame_idx))

        if self._cache_enabled:
            self._xy_ship_cache[frame_idx] = result
        return result

    def xy_earth(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get x/y cartesian coordinates in earth reference frame.

        Earth frame: +X = East, +Y = North
        Origin at ship position.

        Results are cached for performance.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (x, y) arrays, each shape (n_bearings, n_distances)
            in meters relative to ship position
        """
        if self._cache_enabled and frame_idx in self._xy_earth_cache:
            return self._xy_earth_cache[frame_idx]

        result = self._heading_to_xy(frame_idx, self.heading_earth(frame_idx))

        if self._cache_enabled:
            self._xy_earth_cache[frame_idx] = result
        return result

    def plot_polar(
        self,
        frame_idx: int,
        ax=None,
        vmin: float = 0,
        vmax: float = 4095,
        cmap: str = "viridis",
        colorbar: bool = True,
        title: bool = True,
    ):
        """
        Plot frame in polar coordinates (bearing vs distance).

        Args:
            frame_idx: Frame index
            ax: Matplotlib axes (creates new if None)
            vmin, vmax: Colorbar limits
            cmap: Colormap name
            colorbar: Whether to add colorbar
            title: Whether to add subplot title

        Returns:
            Tuple of (figure, axes, image)
        """
        import matplotlib.pyplot as plt

        frame = self._frames[frame_idx]
        bearing = self._theta.bearing_for_frame(frame_idx)
        height = self._get_radar_height(frame_idx)
        if height is not None:
            range_vals = frame.ground_range(height)
            x_label = "Ground Distance (m)"
        else:
            range_vals = frame.slant_range()
            x_label = "Slant Range (m)"
        intensity = frame.intensity.astype(np.float64)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Sort bearing and data to ensure monotonic coordinates for pcolormesh
        sorted_bearing, sorted_intensity = sort_polar_data(bearing, intensity)

        # Create bin edges for pcolormesh
        bearing_edges = calc_bin_edges(sorted_bearing)
        range_edges = calc_bin_edges(range_vals)

        im = ax.pcolormesh(
            range_edges,
            bearing_edges,
            sorted_intensity,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading="flat",
        )

        if colorbar:
            fig.colorbar(im, ax=ax, label="Intensity")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Bearing (°)")
        if title:
            ax.set_title(self._format_title(frame_idx, "Polar"))

        return fig, ax, im

    def plot_ship(
        self,
        frame_idx: int,
        ax=None,
        vmin: float = 0,
        vmax: float = 4095,
        cmap: str = "viridis",
        colorbar: bool = True,
        title: bool = True,
    ):
        """
        Plot frame in ship-relative x/y coordinates.

        Ship frame: +X = starboard, +Y = bow (forward)

        Args:
            frame_idx: Frame index
            ax: Matplotlib axes (creates new if None)
            vmin, vmax: Colorbar limits
            cmap: Colormap name
            colorbar: Whether to add colorbar
            title: Whether to add subplot title

        Returns:
            Tuple of (figure, axes, image)
        """
        import matplotlib.pyplot as plt

        frame = self._frames[frame_idx]
        intensity = frame.intensity
        x, y = self.xy_ship(frame_idx)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        # Use shading='nearest' for radial coordinates - they're not monotonic
        # Suppress the matplotlib warning about non-monotonic coordinates
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*input coordinates.*pcolormesh.*")
            im = ax.pcolormesh(x, y, intensity, vmin=vmin, vmax=vmax, cmap=cmap, shading="nearest")

        if colorbar:
            fig.colorbar(im, ax=ax, label="Intensity")

        ax.set_xlabel("X - Starboard (m)")
        ax.set_ylabel("Y - Bow (m)")
        ax.set_aspect("equal")
        add_crosshairs(ax)

        # Add range rings every 1km
        max_range = np.sqrt(x**2 + y**2).max()
        add_range_rings(ax, max_range, interval=1000.0)

        if title:
            ax.set_title(self._format_title(frame_idx, "Ship Coordinates"))

        return fig, ax, im

    def plot_earth(
        self,
        frame_idx: int,
        ax=None,
        vmin: float = 0,
        vmax: float = 4095,
        cmap: str = "viridis",
        colorbar: bool = True,
        title: bool = True,
    ):
        """
        Plot frame in earth x/y coordinates.

        Earth frame: +X = East, +Y = North

        Args:
            frame_idx: Frame index
            ax: Matplotlib axes (creates new if None)
            vmin, vmax: Colorbar limits
            cmap: Colormap name
            colorbar: Whether to add colorbar
            title: Whether to add subplot title

        Returns:
            Tuple of (figure, axes, image)
        """
        import matplotlib.pyplot as plt

        frame = self._frames[frame_idx]
        intensity = frame.intensity
        x, y = self.xy_earth(frame_idx)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        # Use shading='nearest' for radial coordinates - they're not monotonic
        # Suppress the matplotlib warning about non-monotonic coordinates
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*input coordinates.*pcolormesh.*")
            im = ax.pcolormesh(x, y, intensity, vmin=vmin, vmax=vmax, cmap=cmap, shading="nearest")

        if colorbar:
            fig.colorbar(im, ax=ax, label="Intensity")

        ax.set_xlabel("X - East (m)")
        ax.set_ylabel("Y - North (m)")
        ax.set_aspect("equal")
        add_crosshairs(ax)

        # Add range rings every 1km
        max_range = np.sqrt(x**2 + y**2).max()
        add_range_rings(ax, max_range, interval=1000.0)

        if title:
            ax.set_title(self._format_title(frame_idx, "Earth Coordinates"))

        return fig, ax, im

    def _format_title(self, frame_idx: int, coord_type: str) -> str:
        """Format plot title with frame info."""
        frame = self._frames[frame_idx]

        parts = [f"Frame {frame_idx + 1}/{len(self._frames)}: {frame.timestamp}"]
        parts.append(f"[{coord_type}]")

        nav_info = format_nav_title(frame)
        if nav_info:
            parts.append(nav_info)

        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"Bearing(frames={len(self._frames)}, radar_height={self._radar_height})"

    def __enter__(self) -> "Bearing":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        pass


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument(
        "--radar-height", type=float, default=None, help="Radar height above water (m)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Maximum frames to process (default: 10)"
    )
    parser.add_argument("--no-refine", action="store_true", help="Disable shadow refinement")
    parser.add_argument(
        "--plot", choices=["polar", "ship", "earth", "all"], default=None, help="Plot type"
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to plot (default: 0)")


def add_subparser(subparsers) -> None:
    """Register the 'bearing' subcommand."""
    p = subparsers.add_parser(
        "bearing",
        help="Bearing analysis and plotting",
        description="Calculate radar bearing from polar files",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'bearing' command."""
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.polarfile import load_polar_file

    # Find files (args.stime/etime already parsed by argparse)
    filenames = Filenames(args.stime, args.etime, args.polar_path)
    logging.info(f"Found {len(filenames)} files")

    if not filenames:
        logging.warning("No files found")
        return

    # Load frames
    logging.info(f"Loading up to {args.max_frames} frames...")
    frames = []
    for filepath in filenames.files[: args.max_frames]:
        frame = load_polar_file(filepath)
        if frame is not None:
            frames.append(frame)

    logging.info(f"Loaded {len(frames)} frames")

    if not frames:
        logging.warning("No valid frames")
        return

    # Load config
    config = Config(args.config) if args.config else Config()
    logging.debug(f"Config: {config}")

    # Calculate theta
    theta = Theta(frames, config, refine=not args.no_refine)
    logging.info(f"{theta}")

    # Show statistics
    logging.info("Theta Statistics:")
    logging.info(f"  Total radials: {len(theta)}")
    for i in range(min(3, len(frames))):
        bearing = theta.bearing_for_frame(i)
        logging.info(f"  Frame {i}: bearing range [{bearing.min():.1f}deg, {bearing.max():.1f}deg]")

    if theta.shadow_offset != 0:
        logging.info("Shadow Refinement:")
        logging.info(f"  Offset applied: {theta.shadow_offset:.2f}deg")
        logging.info(f"  Quality (std): {theta.shadow_quality:.2f}deg")

    # Create Bearing object
    bearing_obj = Bearing(theta, radar_height=args.radar_height)
    logging.info(f"{bearing_obj}")

    # Show heading conversions for first frame
    logging.info("Heading conversions (frame 0):")
    heading_ship = bearing_obj.heading_ship(0)
    heading_image = bearing_obj.heading_image(0)
    heading_earth = bearing_obj.heading_earth(0)
    logging.info(f"  Ship heading:  [{heading_ship.min():.1f}deg, {heading_ship.max():.1f}deg]")
    logging.info(f"  Image heading: [{heading_image.min():.1f}deg, {heading_image.max():.1f}deg]")
    logging.info(f"  Earth heading: [{heading_earth.min():.1f}deg, {heading_earth.max():.1f}deg]")

    # Plot if requested
    if args.plot:
        import matplotlib.pyplot as plt

        frame_idx = min(args.frame, len(frames) - 1)

        if args.plot == "all":
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            _, _, im1 = bearing_obj.plot_polar(frame_idx, ax=axes[0], title=False, colorbar=False)
            bearing_obj.plot_ship(frame_idx, ax=axes[1], title=False, colorbar=False)
            bearing_obj.plot_earth(frame_idx, ax=axes[2], title=False, colorbar=False)
            # Add single figure title and shared colorbar
            frame = frames[frame_idx]
            fig.suptitle(f"Frame {frame_idx + 1}/{len(frames)}: {frame.timestamp}")
            fig.subplots_adjust(right=0.92)
            fig.colorbar(im1, ax=axes, label="Intensity", shrink=0.8)
        elif args.plot == "polar":
            bearing_obj.plot_polar(frame_idx)
        elif args.plot == "ship":
            bearing_obj.plot_ship(frame_idx)
        elif args.plot == "earth":
            bearing_obj.plot_earth(frame_idx)

        plt.show()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Calculate radar bearing from polar files")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
