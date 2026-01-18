#! /usr/bin/env python3
#
# Theta - Calculate radar beam angle (theta) for a single frame
# Uses bit 13 encoding to determine degree transitions and interpolates bearings
#
# Jan-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time as _time

import numpy as np

from wamos_tpw.config import Config
from wamos_tpw.frame import Frame


class Theta:
    """
    Calculate radar beam angle (theta) relative to radar for a single frame.

    Uses bit 13 encoding in the first distance bin to determine degree transitions,
    then interpolates bearing values within each segment.

    Algorithm Overview
    ------------------

    The WAMOS radar encodes bearing information in bit 13 of the first distance bin:
    - Bit 13 = 0: radial is within an even degree (e.g., 44.0° to 45.0°)
    - Bit 13 = 1: radial is within an odd degree (e.g., 45.0° to 46.0°)

    The algorithm detects transitions in bit 13 to identify degree boundaries, then
    interpolates bearing values within each segment. Missing transitions (due to
    noise or dropouts) are detected by comparing segment sizes to the median and
    synthetically inserted.

    All bearing values are wrapped to [0, 360).

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> theta = Theta(frame)
        >>> print(f"Theta range: [{theta.theta.min():.1f}, {theta.theta.max():.1f}]")
    """

    # Bit mask for bearing pulse (bit 13)
    _MASK_BIT13 = np.uint16(0x2000)

    # Algorithm constants
    _SEGMENT_SUSPICIOUS_MULTIPLIER = 1.8  # Flag segments > 1.8x median as suspicious
    _SMOOTHING_WEIGHTS = (0.6, 0.2, 0.2)  # Center, prev, next for adaptive smoothing

    def __init__(self, frame: Frame) -> None:
        """
        Initialize Theta calculator for a single frame.

        Args:
            frame: Frame object containing radar data
                   (config is obtained from frame.config)
        """
        self._config = frame.config
        self._timing: dict[str, float] = {}

        # Calculate theta
        self._theta = self._calculate(frame)

    def _calculate(self, frame: Frame) -> np.ndarray:
        """
        Calculate theta angles for all radials in the frame.

        Returns:
            Array of theta angles in degrees [0, 360)
        """
        data = frame.raw
        n_radials = frame.n_bearings

        # Step 1: Extract bit 13 transitions
        t0 = _time.perf_counter()
        bit_13 = (data[:, 0] & self._MASK_BIT13) != 0
        transitions = self._extract_transitions(bit_13, n_radials)
        self._timing["extract_transitions"] = _time.perf_counter() - t0

        # Step 2: Fix missing transitions
        t0 = _time.perf_counter()
        transitions = self._fix_missing_transitions(transitions)
        self._timing["fix_transitions"] = _time.perf_counter() - t0

        # Step 3: Interpolate theta values and wrap to [0, 360)
        t0 = _time.perf_counter()
        theta = self._interpolate_theta(transitions, bit_13[0], n_radials) % 360
        self._timing["interpolate"] = _time.perf_counter() - t0

        return theta

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
        transitions = np.where(np.diff(bit_13))[0] + 1

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

    def _interpolate_theta(
        self, transitions: np.ndarray, first_bit_odd: bool, n_radials: int
    ) -> np.ndarray:
        """
        Interpolate theta values within each transition segment.

        Args:
            transitions: Array of transition indices
            first_bit_odd: Whether first radial has odd degree (bit 13 = 1)
            n_radials: Total number of radials

        Returns:
            Array of theta angles for each radial
        """
        theta = np.zeros(n_radials, dtype=np.float32)

        # Determine starting degree based on bit 13 parity
        current_degree = 1.0 if first_bit_odd else 0.0

        # Pre-compute segment sizes
        segment_sizes = np.diff(transitions)
        n_segments = len(segment_sizes)
        avg_radials_per_degree = n_radials / n_segments if n_segments > 0 else n_radials / 360

        # Process each segment with adaptive width
        for i in range(n_segments):
            start_idx = transitions[i]
            end_idx = transitions[i + 1]
            n_radials_in_segment = segment_sizes[i]

            # Apply smoothing for interior segments
            if i > 0 and i < n_segments - 1:
                prev_size = segment_sizes[i - 1]
                next_size = segment_sizes[i + 1]
                w = self._SMOOTHING_WEIGHTS
                smoothed_size = w[0] * n_radials_in_segment + w[1] * prev_size + w[2] * next_size
                degree_width = smoothed_size / avg_radials_per_degree
            else:
                degree_width = n_radials_in_segment / avg_radials_per_degree

            # Distribute theta within segment
            sub_theta = (
                current_degree
                + (np.arange(n_radials_in_segment) + 0.5) / n_radials_in_segment * degree_width
            )
            theta[start_idx:end_idx] = sub_theta

            current_degree += degree_width

        return theta

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def index(self, thetas: np.ndarray) -> np.ndarray:
        """
        Get radial indices for given theta angles.

        Args:
            thetas: Array of theta angles in degrees

        Returns:
            Array of radial indices corresponding to input theta angles
        """
        thetas = np.asarray(thetas) % 360  # Map onto [0, 360)
        # Claude I really do want axis=1 here
        indices = np.argmin(np.abs(self._theta[:, np.newaxis] - thetas[np.newaxis, :]), axis=1)
        indices = np.clip(indices, 0, self._theta.size - 1)
        return indices

    @property
    def theta(self) -> np.ndarray:
        """
        Get theta angles for all radials.

        Returns:
            Array of theta angles in degrees [0, 360)
        """
        return self._theta

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    @property
    def timing(self) -> dict[str, float]:
        """Return timing information for sub-steps."""
        return self._timing

    def __len__(self) -> int:
        """Return number of radials."""
        return len(self._theta)

    def __repr__(self) -> str:
        return f"radials={len(self)}, range=[{self._theta.min():.1f}, {self._theta.max():.1f}])"


class ThetaDiag:
    """
    Diagnostic visualization for theta calculation results.

    Plots theta angle vs bin number to visualize the bearing distribution.

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> theta = Theta(frame)
        >>> diag = ThetaDiag(frame, theta)
        >>> diag.plot()
    """

    def __init__(self, frame: Frame, theta: Theta) -> None:
        """
        Initialize diagnostic viewer.

        Args:
            frame: Original frame
            theta: Theta object with calculated angles
        """
        self._frame = frame
        self._theta = theta

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @property
    def theta(self) -> Theta:
        """Return the Theta object."""
        return self._theta

    def plot(self, figsize: tuple[float, float] = (12, 6)) -> None:
        """
        Plot theta angle vs bin number.

        Args:
            figsize: Figure size as (width, height) in inches
        """
        import matplotlib.pyplot as plt

        theta_values = self._theta.theta
        bin_numbers = np.arange(len(theta_values))

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(bin_numbers, theta_values, linewidth=0.5)
        ax.set_xlabel("Bin number")
        ax.set_ylabel("Theta (degrees)")
        ax.set_title(f"Theta vs Bin: {self._frame.timestamp}")
        ax.set_xlim(0, len(theta_values) - 1)
        ax.set_ylim(0, 360)
        ax.grid(True, alpha=0.3)

        # Add stats annotation
        stats_text = (
            f"Radials: {len(theta_values)}\n"
            f"Range: [{theta_values.min():.1f}, {theta_values.max():.1f}]°"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"ThetaDiag(frame={self._frame.timestamp}, "
            f"radials={len(self._theta)}, "
            f"range=[{self._theta.theta.min():.1f}, {self._theta.theta.max():.1f}])"
        )


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", type=str, help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument("--plot", action="store_true", help="Plot theta vs bin number")


def add_subparser(subparsers) -> None:
    """Register the 'theta' subcommand."""
    p = subparsers.add_parser(
        "theta",
        help="Calculate theta angles for a frame",
        description="Calculate radar beam angle (theta) from polar file",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'theta' command."""
    from wamos_tpw.polarfile import PolarFile

    # Load polar file
    config = Config(args.config) if args.config else None
    pf = PolarFile(args.filename, config=config)

    if not pf:
        logging.error("No frames found in %s", args.filename)
        return

    frame_idx = min(args.frame, len(pf) - 1)
    frame = pf[frame_idx]

    # Calculate theta
    theta = Theta(frame)

    # Display results
    logging.info("File: %s", args.filename)
    logging.info("Frame: %s (index %s)", frame.timestamp, frame_idx)
    logging.info("Shape: %s", frame.shape)
    logging.info("Theta range: [%.2f, %.2f] degrees", theta.theta.min(), theta.theta.max())
    logging.info("Number of radials: %s", len(theta))

    # Show some sample values
    n_samples = min(5, len(theta))
    logging.info("Sample theta values (first %s):", n_samples)
    for i in range(n_samples):
        logging.info("  Radial %s: %.2f degrees", i, theta.theta[i])

    if args.plot:
        diag = ThetaDiag(frame, theta)
        diag.plot()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Calculate radar beam angle (theta) from polar file")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
