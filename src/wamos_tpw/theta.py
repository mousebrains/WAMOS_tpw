#! /usr/bin/env python3
#
# Theta - Calculate radar beam angle (theta) for a single frame
# Uses 12-bit counter from top nibbles of bins 18, 19, 20 to get whole degrees,
# then distributes fractional degrees using run lengths.
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

    Uses the 12-bit counter encoded in the top nibbles of distance bins 18, 19, 20
    to extract whole degree values, then distributes fractional degrees within each
    run using run-length interpolation.

    Algorithm Overview
    ------------------

    The WAMOS radar encodes a 12-bit bearing counter across 3 distance bins:
    - Bin 18, bits 12-15 → bits 8-11 of the counter
    - Bin 19, bits 12-15 → bits 4-7 of the counter
    - Bin 20, bits 12-15 → bits 0-3 of the counter

    The 12-bit counter starts at an initial value and increments by 1-2 degrees
    per step as the radar rotates. The counter value can exceed 359. Within each
    run of consecutive radials with the same counter value, fractional degrees
    are distributed proportionally based on position within the run.

    For a run of N radials at degree D, radial i gets theta = D + (i + 0.5) / N

    Final bearing values are wrapped to [0, 360).

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> theta = Theta(frame)
        >>> print(f"Theta range: [{theta.theta.min():.1f}, {theta.theta.max():.1f}]")
    """

    # Distance bins containing the 12-bit counter (top nibble of each)
    _COUNTER_BINS = (18, 19, 20)

    # Mask for top nibble (bits 12-15)
    _NIBBLE_MASK = np.uint16(0xF000)

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

        # Step 1: Extract 12-bit counter from bins 18, 19, 20
        t0 = _time.perf_counter()
        degrees = self._extract_degrees(data)
        self._timing["extract_degrees"] = _time.perf_counter() - t0

        # Step 2: Interpolate fractional degrees within each run
        t0 = _time.perf_counter()
        theta = self._interpolate_theta(degrees)
        self._timing["interpolate"] = _time.perf_counter() - t0

        # Step 3: Build sorted index for fast lookups
        t0 = _time.perf_counter()
        self._sorted_indices = np.argsort(theta)
        self._sorted_theta = theta[self._sorted_indices]
        self._timing["sort"] = _time.perf_counter() - t0

        return theta

    def _extract_degrees(self, data: np.ndarray) -> np.ndarray:
        """
        Extract whole degree values from the 12-bit counter in bins 18, 19, 20.

        The 12-bit counter is constructed from the top nibble (bits 12-15) of each bin:
        - Bin 18: bits 8-11 of counter
        - Bin 19: bits 4-7 of counter
        - Bin 20: bits 0-3 of counter

        The counter starts at some initial value and increments by 1-2 degrees per step.
        Values can exceed 359 - wrapping to [0, 360) happens during interpolation.

        Args:
            data: Raw frame data array (n_bearings, n_distances)

        Returns:
            Array of whole degree counter values for each radial
        """
        nibbles = data[:, self._COUNTER_BINS] & self._NIBBLE_MASK
        nibbles = np.right_shift(nibbles, [4, 8, 12]).astype(np.uint16)
        return nibbles[:, 0] | nibbles[:, 1] | nibbles[:, 2]

    def _interpolate_theta(self, degrees: np.ndarray) -> np.ndarray:
        """
        Disperse degrees within each run of degrees.

        For a run of N radials at degree D, radial i gets theta = D + (i + 0.5) / N
        This places radial values at the center of their fractional range.

        Args:
            degrees: Array of whole degree values for each radial (monotonic)

        Returns:
            Array of theta angles for each radial, wrapped to [0, 360)
        """
        unique_degrees, degree_indices, run_counts = np.unique(
            degrees, return_inverse=True, return_counts=True
        )

        mean_run_length = np.mean(run_counts)

        # Convert to float for fractional calculations
        run_counts = run_counts.astype(np.float32)
        first_run_count = run_counts[0]

        # Extend first/last runs if too short (partial rotations at edges)
        run_counts[0] = max(run_counts[0], mean_run_length)
        run_counts[-1] = max(run_counts[-1], mean_run_length)

        # Calculate step size per radial in each run
        degree_diffs = np.diff(unique_degrees, append=unique_degrees[-1] + 1)
        step_sizes = degree_diffs / run_counts
        delta = step_sizes[degree_indices]

        # Adjust first radial's delta to account for extended first run.
        # When first run is extended, cumsum starts too early, so we add
        # the extra steps that would have occurred before the actual start.
        delta[0] += (run_counts[0] - first_run_count) * delta[0]

        return ((unique_degrees[0] + np.cumsum(delta) - (delta / 2)) % 360).astype(np.float32)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def index(self, thetas: np.ndarray) -> np.ndarray:
        """
        Get radial indices for given theta angles.

        Uses binary search on sorted theta for O(m log n) performance.

        Args:
            thetas: Array of theta angles in degrees

        Returns:
            Array of radial indices corresponding to input theta angles
        """
        input_shape = thetas.shape
        thetas = np.asarray(thetas).ravel() % 360  # Map onto [0, 360)

        # Binary search in sorted theta array
        n = len(self._sorted_theta)
        insert_pos = np.searchsorted(self._sorted_theta, thetas)

        # Clamp to valid range and get candidates (position and position-1)
        insert_pos = np.clip(insert_pos, 0, n - 1)
        prev_pos = np.clip(insert_pos - 1, 0, n - 1)

        # Compare distances to find closest
        dist_curr = np.abs(self._sorted_theta[insert_pos] - thetas)
        dist_prev = np.abs(self._sorted_theta[prev_pos] - thetas)

        # Use previous position where it's closer
        best_pos = np.where(dist_prev < dist_curr, prev_pos, insert_pos)

        # Map back to original (unsorted) indices
        indices = self._sorted_indices[best_pos]

        return indices.reshape(input_shape)

    def set_bias(self, bias: float) -> None:
        """
        Apply a bias offset to all theta angles.

        Also updates the sorted index arrays used by index() for lookups.

        Args:
            bias: Bias angle in degrees to add to all theta values
        """
        self._theta = (self._theta + bias) % 360
        # Re-sort after bias adjustment to maintain correct index() behavior
        self._sorted_indices = np.argsort(self._theta)
        self._sorted_theta = self._theta[self._sorted_indices]

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
        return f"Theta(radials={len(self)}, range=[{self._theta.min():.1f}, {self._theta.max():.1f}])"


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
