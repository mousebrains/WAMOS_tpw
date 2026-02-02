#! /usr/bin/env python3
#
# Range - Calculate radar range values for a single frame
# Computes slant range and ground (horizontal) range with optional bias correction
#
# Jan-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging

import numpy as np

from wamos_tpw.config import Config
from wamos_tpw.constants import C_AIR
from wamos_tpw.frame import Frame


class Range:
    """
    Calculate radar range values for a single frame.

    Provides slant range (line-of-sight distance) and ground range (horizontal
    distance) calculations. When a config is provided with bias.range, the
    bias correction is applied to the ground range.

    Range Calculations
    ------------------

    **Slant Range**: The direct line-of-sight distance from the radar to the target.
    Calculated using the speed of light in air at standard conditions.

        slant_range = sample_delay_range + bin_index * range_resolution

    **Ground Range**: The horizontal distance from the radar to the target,
    accounting for radar height above the water surface.

        ground_range = sqrt(slant_range² - radar_height²) + bias

    The bias.range from config is applied to the ground range if present.

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> rng = Range(frame)
        >>> print(f"Ground range: [{rng.ground_range.min():.1f}, {rng.ground_range.max():.1f}] m")
    """

    def __init__(self, frame: Frame) -> None:
        """
        Initialize Range calculator for a single frame.

        Args:
            frame: Frame object containing radar data and metadata
                   (config is obtained from frame.config, including tower.height)
        """
        self._frame = frame
        self._config = frame.config or Config()

        # Get radar/tower height from config (set from WINDH header by PolarFile)
        self._radar_height = self._config.get("tower.height")

        # Get range bias from config (applied to ground range)
        self._range_bias = self._config.get("bias.range", 0.0)

        # Calculate ranges
        self._slant_range = self._calculate_slant_range()
        self._ground_range = self._calculate_ground_range()

    def _calculate_range_resolution(self) -> float:
        """
        Calculate the range resolution (meters per distance bin).

        Uses the speed of light in air at standard conditions and the
        sampling frequency from frame metadata.

        Returns:
            Range resolution in meters per bin, or 0.0 if sampling_frequency not set.
        """
        sfreq_mhz = self._frame.metadata.sampling_frequency
        if sfreq_mhz <= 0:
            return 0.0
        # Convert MHz to Hz
        sfreq_hz = sfreq_mhz * 1e6
        # Round-trip time per sample = 1/sfreq
        # One-way distance = c_air * t / 2 = c_air / (2 * sfreq)
        return C_AIR / (2.0 * sfreq_hz)

    def _calculate_slant_range(self) -> np.ndarray:
        """
        Calculate slant range (straight-line distance) for all bins.

        Returns:
            Array of slant ranges in meters, shape (n_distances,)
        """
        n_distances = self._frame.n_distances
        bin_indices = np.arange(n_distances)
        delta_r = self._calculate_range_resolution()
        sdrng = self._frame.metadata.sample_delay_range

        # slant_range = sample_delay_range + bin_index * range_resolution
        return sdrng + bin_indices * delta_r

    def _calculate_ground_range(self) -> np.ndarray:
        """
        Calculate ground range (horizontal distance) for all bins.

        Applies bias.range correction from config if present.

        Returns:
            Array of ground ranges in meters, shape (n_distances,), dtype float32
        """
        if self._radar_height is None:
            # No radar height - return slant range + bias
            return (self._slant_range + self._range_bias).astype(np.float32)

        # Calculate ground range: sqrt(slant² - height²)
        height_sq = self._radar_height**2
        slant_sq = self._slant_range**2

        # Where slant > height, compute ground range; otherwise 0
        ground = np.where(slant_sq > height_sq, np.sqrt(slant_sq - height_sq), 0.0)

        # Apply bias correction and convert to float32 (sufficient precision for ~0-3000m)
        return (ground + self._range_bias).astype(np.float32)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    @property
    def radar_height(self) -> float | None:
        """Return the radar height used for ground range calculation."""
        return self._radar_height

    @property
    def range_bias(self) -> float:
        """Return the range bias applied to ground range (from config)."""
        return self._range_bias

    @property
    def range_resolution(self) -> float:
        """Return the range resolution in meters per bin."""
        return self._calculate_range_resolution()

    @property
    def slant_range(self) -> np.ndarray:
        """
        Get slant range (line-of-sight distance) for all bins.

        Returns:
            Array of slant ranges in meters, shape (n_distances,)
        """
        return self._slant_range

    @property
    def ground_range(self) -> np.ndarray:
        """
        Get ground range (horizontal distance) for all bins.

        Includes bias.range correction from config if present.

        Returns:
            Array of ground ranges in meters, shape (n_distances,)
        """
        return self._ground_range

    def slant_range_at_bin(self, bin_index: int) -> float:
        """Get slant range for a specific bin index."""
        return self._slant_range[bin_index]

    def ground_range_at_bin(self, bin_index: int) -> float:
        """Get ground range for a specific bin index."""
        return self._ground_range[bin_index]

    def __len__(self) -> int:
        """Return number of distance bins."""
        return len(self._slant_range)

    def __repr__(self) -> str:
        return (
            f"Range(frame={self._frame.metadata.filename}, "
            f"bins={len(self)}, "
            f"ground=[{self._ground_range.min():.1f}, {self._ground_range.max():.1f}]m, "
            f"bias={self._range_bias}m)"
        )


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", type=str, help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")


def add_subparser(subparsers) -> None:
    """Register the 'range' subcommand."""
    p = subparsers.add_parser(
        "range",
        help="Calculate range values for a frame",
        description="Calculate radar range (slant and ground) from polar file",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'range' command."""
    from wamos_tpw.polarfile import PolarFile

    # Load polar file
    config = Config(args.config) if args.config else None
    pf = PolarFile(args.filename, config=config)

    if not pf:
        logging.error("No frames found in %s", args.filename)
        return

    frame_idx = min(args.frame, len(pf) - 1)
    frame = pf[frame_idx]

    # Calculate range (config is obtained from frame.config)
    rng = Range(frame)

    # Display results
    logging.info("File: %s", args.filename)
    logging.info("Frame: %s (index %s)", frame.timestamp, frame_idx)
    logging.info("Shape: %s", frame.shape)
    logging.info("Radar height: %s m", rng.radar_height)
    logging.info("Range bias (from config): %s m", rng.range_bias)
    logging.info("Range resolution: %.4f m/bin", rng.range_resolution)
    logging.info("Number of bins: %s", len(rng))
    logging.info("Slant range: [%.2f, %.2f] m", rng.slant_range.min(), rng.slant_range.max())
    logging.info("Ground range: [%.2f, %.2f] m", rng.ground_range.min(), rng.ground_range.max())

    # Show some sample values
    n_samples = min(5, len(rng))
    logging.info("Sample range values (first %s bins):", n_samples)
    for i in range(n_samples):
        logging.info(
            "  Bin %s: slant=%.2f m, ground=%.2f m",
            i,
            rng.slant_range_at_bin(i),
            rng.ground_range_at_bin(i),
        )


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Calculate radar range from polar file")

if __name__ == "__main__":
    main()
