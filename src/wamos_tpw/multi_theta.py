#! /usr/bin/env python3
#
# MultiTheta and MultiBearing - Multi-frame wrappers for Theta and Bearing
#
# Provides backward-compatible multi-frame API for combine.py and related modules.
# Internally uses single-frame Theta from theta.py and Shadow from shadow.py.
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
    def _theta(self) -> MultiTheta:
        """Return the MultiTheta object (internal access)."""
        return self.__theta

    @_theta.setter
    def _theta(self, value: MultiTheta) -> None:
        """Set the MultiTheta object."""
        self.__theta = value

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


# Backward compatibility aliases
Theta = MultiTheta
Bearing = MultiBearing
