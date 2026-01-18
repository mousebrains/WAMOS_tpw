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

    The 12-bit value directly represents the whole degree (0-359). Within each
    run of consecutive radials with the same degree value, fractional degrees
    are distributed proportionally based on position within the run.

    For a run of N radials at degree D, radial i gets theta = D + (i + 0.5) / N

    All bearing values are wrapped to [0, 360).

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
        n_radials = frame.n_bearings

        # Step 1: Extract 12-bit counter from bins 18, 19, 20
        t0 = _time.perf_counter()
        degrees = self._extract_degrees(data)
        self._timing["extract_degrees"] = _time.perf_counter() - t0

        # Step 2: Find run boundaries (where degree changes)
        t0 = _time.perf_counter()
        transitions = self._find_transitions(degrees, n_radials)
        self._timing["find_transitions"] = _time.perf_counter() - t0

        # Step 3: Interpolate fractional degrees within each run
        t0 = _time.perf_counter()
        theta = self._interpolate_theta(degrees, transitions, n_radials)
        self._timing["interpolate"] = _time.perf_counter() - t0

        return theta

    def _extract_degrees(self, data: np.ndarray) -> np.ndarray:
        """
        Extract whole degree values from the 12-bit counter in bins 18, 19, 20.

        The 12-bit counter is constructed from the top nibble (bits 12-15) of each bin:
        - Bin 18: bits 8-11 of counter
        - Bin 19: bits 4-7 of counter
        - Bin 20: bits 0-3 of counter

        Args:
            data: Raw frame data array (n_bearings, n_distances)

        Returns:
            Array of whole degree values (0-359) for each radial
        """
        b18, b19, b20 = self._COUNTER_BINS

        # Extract top nibble from each bin and shift to correct position
        # Bin 18: shift right 4 to get bits 8-11 of result
        counter = np.right_shift(data[:, b18] & self._NIBBLE_MASK, 4).astype(np.uint16)
        # Bin 19: shift right 8 to get bits 4-7 of result
        counter += np.right_shift(data[:, b19] & self._NIBBLE_MASK, 8).astype(np.uint16)
        # Bin 20: shift right 12 to get bits 0-3 of result
        counter += np.right_shift(data[:, b20] & self._NIBBLE_MASK, 12).astype(np.uint16)

        # Counter is whole degrees (0-359), wrap to handle any overflow
        return counter % 360

    def _find_transitions(self, degrees: np.ndarray, n_radials: int) -> np.ndarray:
        """
        Find indices where the degree value changes.

        Args:
            degrees: Array of whole degree values for each radial
            n_radials: Total number of radials

        Returns:
            Array of transition indices including boundaries [0, ..., n_radials]
        """
        # Find where degree changes
        transitions = np.where(degrees[:-1] != degrees[1:])[0] + 1

        # Add boundaries (start and end)
        return np.concatenate([[0], transitions, [n_radials]])

    def _interpolate_theta(
        self, degrees: np.ndarray, transitions: np.ndarray, n_radials: int
    ) -> np.ndarray:
        """
        Interpolate fractional degrees within each run.

        For a run of N radials at degree D, radial i gets theta = D + (i + 0.5) / N
        This places radial values at the center of their fractional range.

        Args:
            degrees: Array of whole degree values for each radial
            transitions: Array of transition indices
            n_radials: Total number of radials

        Returns:
            Array of theta angles for each radial, wrapped to [0, 360)
        """
        segment_sizes = np.diff(transitions)
        n_segments = len(segment_sizes)

        if n_segments == 0:
            return np.zeros(n_radials, dtype=np.float32)

        # Get the degree value for each segment (from first radial in segment)
        segment_degrees = degrees[transitions[:-1]].astype(np.float32)

        # Compute local position within each segment
        segment_starts_idx = transitions[:-1]
        local_pos = np.arange(n_radials) - np.repeat(segment_starts_idx, segment_sizes)

        # Compute theta: degree + (local_pos + 0.5) / segment_size
        theta = np.repeat(segment_degrees, segment_sizes) + (local_pos + 0.5) / np.repeat(
            segment_sizes.astype(np.float32), segment_sizes
        )

        return (theta % 360).astype(np.float32)

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
