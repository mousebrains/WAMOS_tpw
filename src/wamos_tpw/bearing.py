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

from wamos_tpw.config import Config
from wamos_tpw.frame import Frame
from wamos_tpw.theta import Theta as SingleTheta
from wamos_tpw.shadow import Shadow
from wamos_tpw.destreak import Destreak
from wamos_tpw.range import Range

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "Bearing",
    "MultiTheta",
    "MultiBearing",
    "theta_to_heading_ship",
    "theta_to_heading_earth",
    "heading_to_xy",
]


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


# ============================================================================
# Multi-frame classes (originally from multi_theta.py)
# ============================================================================


class MultiTheta:
    """
    Multi-frame theta (bearing angle) calculation with shadow refinement.

    Provides the legacy multi-frame API that combine.py uses, internally
    using the single-frame Theta and Shadow classes for each frame.

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frames = list(pf)
        >>> theta = MultiTheta(frames, config, refine=True)
        >>> bearing_frame0 = theta.bearing_for_frame(0)
        >>> shadow_mask = theta.in_shadow(0)
    """

    def __init__(
        self,
        frames: list[Frame],
        config: Config | None = None,
        refine: bool = True,
    ) -> None:
        """
        Initialize MultiTheta for multiple frames.

        Args:
            frames: List of Frame objects
            config: Configuration object
            refine: Whether to refine theta using shadow detection
        """
        if not frames:
            raise ValueError("At least one frame is required")

        self._frames = frames
        self._config = config or Config()
        self._refine = refine

        # Process each frame
        self._theta_per_frame: list[np.ndarray] = []
        self._shadow_per_frame: list[Shadow | None] = []
        self._shadow_indices_per_frame: list[np.ndarray] = []
        self._shadow_offset = 0.0

        biases = []
        for frame in frames:
            # Set config on frame if not set
            if frame.config is None:
                frame._config = self._config

            # Calculate theta for this frame
            theta_obj = SingleTheta(frame)

            if refine:
                # Get destreaked intensity for shadow detection
                destreak = Destreak(frame)
                shadow = Shadow(destreak.intensity, theta_obj)

                # Apply shadow bias to theta
                if shadow.theta_bias is not None and shadow.theta_bias != 0:
                    theta_obj.set_bias(shadow.theta_bias)
                    biases.append(shadow.theta_bias)

                self._shadow_per_frame.append(shadow)
                self._shadow_indices_per_frame.append(shadow.indices.copy())
            else:
                self._shadow_per_frame.append(None)
                self._shadow_indices_per_frame.append(np.empty((0, 2), dtype=int))

            self._theta_per_frame.append(theta_obj.theta.copy())

        # Store mean shadow offset
        if biases:
            self._shadow_offset = np.mean(biases)

        # Build concatenated bearing array
        self._bearing = np.concatenate(self._theta_per_frame)

        # Build frame boundaries for indexing
        self._frame_boundaries = np.cumsum([0] + [len(t) for t in self._theta_per_frame])

    @property
    def frames(self) -> list[Frame]:
        """Return the frames."""
        return self._frames

    @property
    def config(self) -> Config:
        """Return the configuration."""
        return self._config

    @property
    def bearing(self) -> np.ndarray:
        """Return all bearing angles concatenated across frames."""
        return self._bearing

    @property
    def bearing_per_frame(self) -> list[np.ndarray]:
        """Return list of bearing arrays, one per frame."""
        return self._theta_per_frame

    @property
    def shadow_offset(self) -> float:
        """Return the mean shadow-based theta offset applied during refinement."""
        return self._shadow_offset

    def bearing_for_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get bearing angles for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Array of bearing angles in degrees [0, 360)
        """
        return self._theta_per_frame[frame_idx]

    def in_shadow(self, frame_idx: int) -> np.ndarray:
        """
        Get shadow mask for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Boolean array where True indicates shadow region
        """
        n_bearings = len(self._theta_per_frame[frame_idx])
        mask = np.zeros(n_bearings, dtype=bool)

        shadow_indices = self._shadow_indices_per_frame[frame_idx]
        for start, end in shadow_indices:
            mask[start : end + 1] = True

        return mask

    def clear_shadow_data(self) -> None:
        """Clear shadow data to free memory."""
        self._shadow_per_frame = [None] * len(self._frames)

    def __len__(self) -> int:
        """Return total number of radials across all frames."""
        return len(self._bearing)

    def __repr__(self) -> str:
        return (
            f"MultiTheta(frames={len(self._frames)}, "
            f"radials={len(self._bearing)}, "
            f"shadow_offset={self._shadow_offset:.2f})"
        )


class MultiBearing:
    """
    Multi-frame bearing coordinate calculator.

    Provides the legacy multi-frame API for coordinate transformation,
    calculating earth coordinates for each frame with caching.

    Example:
        >>> theta = MultiTheta(frames, config)
        >>> bearing = MultiBearing(theta, radar_height=25.0)
        >>> x, y = bearing.xy_earth(0)  # Earth coordinates for frame 0
    """

    def __init__(
        self,
        theta: MultiTheta,
        radar_height: float | None = None,
        cache_coordinates: bool = True,
    ) -> None:
        """
        Initialize MultiBearing for coordinate transformation.

        Args:
            theta: MultiTheta object with bearing angles
            radar_height: Radar height above water in meters (optional)
            cache_coordinates: Whether to cache computed coordinates
        """
        self._theta = theta
        self._radar_height = radar_height
        self._cache_coordinates = cache_coordinates
        self._config = theta.config

        # Get offset values from config
        self._bow_to_radar = self._config.get("offsets.bow_to_radar", 0.0)
        self._heading_delay = self._config.get("offsets.heading_delay", 0.0)
        self._compass_offset = self._config.get("offsets.compass", 0.0)

        # Caches
        self._xy_ship_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._xy_earth_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._ground_range_cache: dict[int, np.ndarray] = {}

    @property
    def theta(self) -> MultiTheta:
        """Return the MultiTheta object."""
        return self._theta

    @property
    def config(self) -> Config:
        """Return the configuration."""
        return self._config

    def _get_radar_height(self, frame_idx: int) -> float | None:
        """
        Get radar height for a frame.

        Priority: init param > config > frame metadata.

        Args:
            frame_idx: Frame index

        Returns:
            Radar height in meters, or None if not available
        """
        if self._radar_height is not None:
            return self._radar_height

        # Try config
        height = self._config.get("radar_height", None)
        if height is not None:
            return float(height)

        # Try frame metadata
        frame = self._theta.frames[frame_idx]
        if hasattr(frame.metadata, "radar_height") and frame.metadata.radar_height is not None:
            return frame.metadata.radar_height

        # Fall back to WINDH if available
        if hasattr(frame.metadata, "windh") and frame.metadata.windh is not None:
            return frame.metadata.windh

        return None

    def _get_ground_range(self, frame_idx: int) -> np.ndarray:
        """
        Get ground range array for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Ground range values in meters
        """
        if frame_idx in self._ground_range_cache:
            return self._ground_range_cache[frame_idx]

        frame = self._theta.frames[frame_idx]
        range_obj = Range(frame)
        ground_range = range_obj.ground_range

        if self._cache_coordinates:
            self._ground_range_cache[frame_idx] = ground_range

        return ground_range

    def heading_ship(self, frame_idx: int) -> np.ndarray:
        """
        Get ship-relative heading for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Ship-relative headings in degrees [0, 360)
        """
        theta = self._theta.bearing_for_frame(frame_idx)
        return (theta + self._bow_to_radar) % 360

    def heading_earth(self, frame_idx: int) -> np.ndarray:
        """
        Get earth-relative heading for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Earth headings in degrees [0, 360), where 0=North, 90=East
        """
        frame = self._theta.frames[frame_idx]
        ship_heading = frame.metadata.heading or 0.0

        theta = self._theta.bearing_for_frame(frame_idx)
        return (
            theta + self._bow_to_radar + self._heading_delay + ship_heading + self._compass_offset
        ) % 360

    def heading_image(self, frame_idx: int) -> np.ndarray:
        """
        Get image heading for a frame (ship heading + heading delay).

        This is the heading used for image rotation/display.

        Args:
            frame_idx: Frame index

        Returns:
            Image headings in degrees [0, 360)
        """
        return (self.heading_ship(frame_idx) + self._heading_delay) % 360

    def xy_ship(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x/y coordinates in ship reference frame.

        Ship frame: +X = starboard, +Y = bow

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (x, y) arrays, each shape (n_radials, n_distances)
        """
        if frame_idx in self._xy_ship_cache:
            return self._xy_ship_cache[frame_idx]

        heading = self.heading_ship(frame_idx)
        ground_range = self._get_ground_range(frame_idx)

        heading_rad = np.deg2rad(heading)
        heading_2d = heading_rad[:, np.newaxis]
        range_2d = ground_range[np.newaxis, :]

        x = range_2d * np.sin(heading_2d)
        y = range_2d * np.cos(heading_2d)

        if self._cache_coordinates:
            self._xy_ship_cache[frame_idx] = (x, y)

        return x, y

    def xy_earth(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x/y coordinates in earth reference frame.

        Earth frame: +X = East, +Y = North

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (x, y) arrays, each shape (n_radials, n_distances)
        """
        if frame_idx in self._xy_earth_cache:
            return self._xy_earth_cache[frame_idx]

        heading = self.heading_earth(frame_idx)
        ground_range = self._get_ground_range(frame_idx)

        heading_rad = np.deg2rad(heading)
        heading_2d = heading_rad[:, np.newaxis]
        range_2d = ground_range[np.newaxis, :]

        x = range_2d * np.sin(heading_2d)  # East
        y = range_2d * np.cos(heading_2d)  # North

        if self._cache_coordinates:
            self._xy_earth_cache[frame_idx] = (x, y)

        return x, y

    def clear_cache(self) -> None:
        """Clear all coordinate caches."""
        self._xy_ship_cache.clear()
        self._xy_earth_cache.clear()
        self._ground_range_cache.clear()

    def __repr__(self) -> str:
        return f"MultiBearing(frames={len(self._theta.frames)}, radar_height={self._radar_height})"


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Convert radar theta to earth heading")

if __name__ == "__main__":
    main()
