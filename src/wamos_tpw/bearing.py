#! /usr/bin/env python3
#
# Bearing - Convert radar theta to ship/earth headings and x/y coordinates
#
# Provides functions and classes to transform radar-relative beam angles (theta)
# to ship-relative and earth-referenced headings, with optional cartesian
# coordinate projection.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import Config

logger = logging.getLogger(__name__)


def theta_to_heading_ship(
    theta: np.ndarray,
    bow_to_radar: float = 0.0,
) -> np.ndarray:
    """
    Convert radar theta to ship-relative heading.

    heading_ship = theta + BO2RA

    Args:
        theta: Radar beam angles in degrees
        bow_to_radar: BO2RA offset from bow to radar beam (degrees)

    Returns:
        Ship-relative headings in degrees [0, 360)
    """
    return (theta + bow_to_radar) % 360


def theta_to_heading_earth(
    theta: np.ndarray,
    ship_heading: float,
    bow_to_radar: float = 0.0,
    heading_delay: float = 0.0,
    compass_offset: float = 0.0,
) -> np.ndarray:
    """
    Convert radar theta to earth-referenced heading.

    heading_earth = theta + BO2RA + HDGDL + GYROC + compass

    Args:
        theta: Radar beam angles in degrees
        ship_heading: Ship's gyro compass heading (GYROC) in degrees
        bow_to_radar: BO2RA offset from bow to radar beam (degrees)
        heading_delay: HDGDL heading delay correction (degrees)
        compass_offset: Compass offset correction (degrees)

    Returns:
        Earth headings in degrees [0, 360), where 0=North, 90=East
    """
    return (theta + bow_to_radar + heading_delay + ship_heading + compass_offset) % 360


def heading_to_xy(
    heading: np.ndarray,
    ground_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert heading angles and range to x/y cartesian coordinates.

    Convention: +X = East (heading 90°), +Y = North (heading 0°)

    Args:
        heading: Heading angles in degrees, shape (n_radials,)
        ground_range: Ground range values in meters, shape (n_distances,)

    Returns:
        Tuple of (x, y) arrays, each shape (n_radials, n_distances)
        x: East component in meters
        y: North component in meters
    """
    heading_rad = np.deg2rad(heading)

    # Broadcast: heading (n_radials, 1) * range (1, n_distances)
    heading_2d = heading_rad[:, np.newaxis]
    range_2d = ground_range[np.newaxis, :]

    x = range_2d * np.sin(heading_2d)  # East
    y = range_2d * np.cos(heading_2d)  # North

    return x, y


class Bearing:
    """
    Convert radar theta to ship/earth headings with coordinate projections.

    This is a convenience class that wraps the heading conversion functions
    and caches results for repeated access.

    Example:
        >>> from wamos_tpw.theta import Theta
        >>> theta = Theta(frame)
        >>> bearing = Bearing(theta.theta, frame.metadata, ground_range)
        >>> x, y = bearing.xy_earth()
    """

    def __init__(
        self,
        theta: np.ndarray,
        ship_heading: float,
        ground_range: np.ndarray,
        config: "Config | None" = None,
    ) -> None:
        """
        Initialize Bearing converter.

        Args:
            theta: Radar beam angles in degrees, shape (n_radials,)
            ship_heading: Ship's gyro compass heading in degrees
            ground_range: Ground range values in meters, shape (n_distances,)
            config: Optional config for offset values (defaults to 0)
        """
        self._theta = theta
        self._ship_heading = ship_heading
        self._ground_range = ground_range

        # Get offsets from config or use defaults
        if config is not None:
            self._bow_to_radar = config.get("offsets.bow_to_radar", 0.0)
            self._heading_delay = config.get("offsets.heading_delay", 0.0)
            self._compass_offset = config.get("offsets.compass", 0.0)
        else:
            self._bow_to_radar = 0.0
            self._heading_delay = 0.0
            self._compass_offset = 0.0

        # Cached results
        self._heading_ship: np.ndarray | None = None
        self._heading_earth: np.ndarray | None = None
        self._xy_ship: tuple[np.ndarray, np.ndarray] | None = None
        self._xy_earth: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def theta(self) -> np.ndarray:
        """Return the radar theta angles."""
        return self._theta

    @property
    def ship_heading(self) -> float:
        """Return the ship's heading."""
        return self._ship_heading

    @property
    def ground_range(self) -> np.ndarray:
        """Return the ground range array."""
        return self._ground_range

    def heading_ship(self) -> np.ndarray:
        """
        Get beam heading relative to ship bow.

        Returns:
            Ship-relative headings in degrees [0, 360)
        """
        if self._heading_ship is None:
            self._heading_ship = theta_to_heading_ship(
                self._theta,
                self._bow_to_radar,
            )
        return self._heading_ship

    def heading_earth(self) -> np.ndarray:
        """
        Get beam heading in earth coordinates.

        Returns:
            Earth headings in degrees [0, 360), where 0=North, 90=East
        """
        if self._heading_earth is None:
            self._heading_earth = theta_to_heading_earth(
                self._theta,
                self._ship_heading,
                self._bow_to_radar,
                self._heading_delay,
                self._compass_offset,
            )
        return self._heading_earth

    def xy_ship(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x/y coordinates in ship reference frame.

        Ship frame: +X = starboard, +Y = bow

        Returns:
            Tuple of (x, y) arrays in meters
        """
        if self._xy_ship is None:
            self._xy_ship = heading_to_xy(self.heading_ship(), self._ground_range)
        return self._xy_ship

    def xy_earth(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x/y coordinates in earth reference frame.

        Earth frame: +X = East, +Y = North

        Returns:
            Tuple of (x, y) arrays in meters
        """
        if self._xy_earth is None:
            self._xy_earth = heading_to_xy(self.heading_earth(), self._ground_range)
        return self._xy_earth

    def __repr__(self) -> str:
        return (
            f"Bearing(n_radials={len(self._theta)}, "
            f"n_distances={len(self._ground_range)}, "
            f"ship_heading={self._ship_heading:.1f})"
        )


# =============================================================================
# Backward compatibility: Legacy Theta class wrapper
# =============================================================================
# The old Theta class has been replaced by wamos_tpw.theta.Theta
# This import maintains backward compatibility for code that does:
#   from wamos_tpw.bearing import Theta

try:
    from wamos_tpw.theta import Theta
except ImportError:
    # Theta not available - define a stub that raises helpful error
    class Theta:  # type: ignore[no-redef]
        """Theta has moved to wamos_tpw.theta."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Theta has moved from wamos_tpw.bearing to wamos_tpw.theta. "
                "Use: from wamos_tpw.theta import Theta"
            )


# =============================================================================
# CLI
# =============================================================================


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", type=str, help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")


def add_subparser(subparsers) -> None:
    """Register the 'bearing' subcommand."""
    p = subparsers.add_parser(
        "bearing",
        help="Calculate bearing from theta",
        description="Convert radar theta to earth heading",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'bearing' command."""
    from wamos_tpw.config import Config
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.theta import Theta
    from wamos_tpw.range import Range

    # Load polar file
    config = Config(args.config) if args.config else Config()
    pf = PolarFile(args.filename, config=config)

    if not pf:
        logging.error("No frames found in %s", args.filename)
        return

    frame_idx = min(args.frame, len(pf) - 1)
    frame = pf[frame_idx]

    # Calculate theta and range
    theta_obj = Theta(frame)
    range_obj = Range(frame)

    # Get ship heading from metadata
    ship_heading = frame.metadata.heading or 0.0

    # Create bearing converter
    bearing = Bearing(
        theta=theta_obj.theta,
        ship_heading=ship_heading,
        ground_range=range_obj.ground_range,
        config=config,
    )

    # Display results
    logging.info("File: %s", args.filename)
    logging.info("Frame: %s (index %d)", frame.timestamp, frame_idx)
    logging.info("Shape: %s", frame.shape)
    logging.info("")
    logging.info("Theta range: [%.2f, %.2f] degrees", theta_obj.theta.min(), theta_obj.theta.max())
    logging.info("Ship heading: %.1f degrees", ship_heading)
    logging.info("")

    heading_ship = bearing.heading_ship()
    heading_earth = bearing.heading_earth()
    logging.info("Heading ship:  [%.1f, %.1f] degrees", heading_ship.min(), heading_ship.max())
    logging.info("Heading earth: [%.1f, %.1f] degrees", heading_earth.min(), heading_earth.max())

    x_earth, y_earth = bearing.xy_earth()
    logging.info("")
    logging.info("Earth coordinates:")
    logging.info("  X (East):  [%.1f, %.1f] m", x_earth.min(), x_earth.max())
    logging.info("  Y (North): [%.1f, %.1f] m", y_earth.min(), y_earth.max())


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Convert radar theta to earth heading")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
