#! /usr/bin/env python3
#
# Combine class for combining frames into earth coordinate images
# Accounts for ship motion during radar scan
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from wamos_tpw.multi_theta import MultiTheta as Theta, MultiBearing as Bearing
from wamos_tpw.config import Config
from wamos_tpw.frame import Frame
from wamos_tpw.timestamp import Timestamp


class Combine:
    """
    Combine radar frames into earth coordinate images with ship motion compensation.

    Calculates x/y coordinates in earth reference frame for each radar pixel,
    accounting for ship motion during the scan. The ship moves continuously
    while the radar sweeps, so each radial beam originates from a slightly
    different position.

    Earth frame convention:
    - +X = East
    - +Y = North
    - Origin = ship position at start of first frame

    Example:
        >>> config = Config('radar_config.yaml')
        >>> combine = Combine(frames, config)
        >>> x_earth, y_earth = combine.xy_earth_all()  # (n_total_pixels,), (n_total_pixels,)
        >>> lat, lon = combine.latlon_all()  # (n_total_pixels,), (n_total_pixels,)
        >>> combine.plot_diagnostics()  # Show earth coordinate plot
    """

    # Earth radius in meters (WGS84 mean radius)
    _EARTH_RADIUS = 6371000.0

    def __init__(
        self,
        frames: list[Frame],
        config: Config | None = None,
        radar_height: float | None = None,
        theta: Theta | None = None,
        cache_coordinates: bool = True,
    ):
        """
        Initialize Combine for a set of contiguous frames.

        Args:
            frames: List of contiguous Frame objects (sorted by time)
            config: Config object (uses defaults if None)
            radar_height: Radar height above water (m). If None, uses config or metadata.
            theta: Existing Theta object to reuse (avoids duplicate computation).
                   If None, creates a new one.
            cache_coordinates: If True, cache xy coordinates in Bearing (uses more memory).
                              Set to False for memory-constrained scenarios.
        """
        if not frames:
            raise ValueError("At least one frame is required")

        self._frames = frames
        self._config = config or Config()

        # Create or reuse Theta/Bearing for coordinate calculation
        if theta is not None:
            self._theta = theta
        else:
            self._theta = Theta(frames, config, refine=True)
        self._bearing = Bearing(
            self._theta, radar_height=radar_height, cache_coordinates=cache_coordinates
        )

        # Create Timestamp for ship position calculation
        self._timestamp = Timestamp(frames, config)

        # Get reference position (first frame's recorded position)
        self._ref_lat = frames[0].metadata.latitude or 0.0
        self._ref_lon = frames[0].metadata.longitude or 0.0

        # Cached results
        self._xy_earth_cache: list[Tuple[np.ndarray, np.ndarray]] | None = None
        self._latlon_cache: list[Tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def frames(self) -> list[Frame]:
        """Return the frames."""
        return self._frames

    @property
    def config(self) -> Config:
        """Return the configuration."""
        return self._config

    @property
    def timestamp(self) -> Timestamp:
        """Return the Timestamp object."""
        return self._timestamp

    @property
    def bearing(self) -> Bearing:
        """Return the Bearing object."""
        return self._bearing

    @property
    def reference_position(self) -> Tuple[float, float]:
        """Return the reference (lat, lon) position."""
        return (self._ref_lat, self._ref_lon)

    def _compute_all(self) -> None:
        """Compute earth coordinates for all frames with ship motion compensation."""
        if self._xy_earth_cache is not None:
            return

        self._xy_earth_cache = []
        self._latlon_cache = []

        for frame_idx in range(len(self._frames)):
            x_earth, y_earth = self._compute_frame_xy_earth(frame_idx)
            lat, lon = self._compute_frame_latlon(frame_idx, x_earth, y_earth)

            self._xy_earth_cache.append((x_earth, y_earth))
            self._latlon_cache.append((lat, lon))

    def _compute_frame_xy_earth(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute x/y earth coordinates for a frame with ship motion compensation.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (x_earth, y_earth) arrays, each shape (n_bearings, n_distances)
            in meters relative to reference position
        """
        # Get radar pixel positions relative to ship
        x_rel, y_rel = self._bearing.xy_earth(frame_idx)

        # Get ship position offset for each radial
        ship_lat, ship_lon = self._timestamp.position_for_frame(frame_idx)

        # Convert ship position to meters relative to reference
        # delta_lat in degrees -> meters (North)
        # delta_lon in degrees -> meters (East)
        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(self._ref_lat))

        ship_y = (ship_lat - self._ref_lat) * meters_per_deg_lat  # North offset
        ship_x = (ship_lon - self._ref_lon) * meters_per_deg_lon  # East offset

        # Ship offsets are 1D (per radial), expand to 2D
        # x_rel, y_rel are shape (n_bearings, n_distances)
        ship_x_2d = ship_x[:, np.newaxis]
        ship_y_2d = ship_y[:, np.newaxis]

        # Add ship position to relative coordinates
        x_earth = x_rel + ship_x_2d
        y_earth = y_rel + ship_y_2d

        return x_earth, y_earth

    def _compute_frame_latlon(
        self, frame_idx: int, x_earth: np.ndarray, y_earth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert x/y earth coordinates to lat/lon.

        Args:
            frame_idx: Frame index
            x_earth: East offset in meters from reference
            y_earth: North offset in meters from reference

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(self._ref_lat))

        delta_lat = y_earth / meters_per_deg_lat
        delta_lon = x_earth / meters_per_deg_lon if meters_per_deg_lon > 0 else 0.0

        lat = self._ref_lat + delta_lat
        lon = self._ref_lon + delta_lon

        return lat, lon

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def xy_earth(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get x/y earth coordinates for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (x_earth, y_earth) arrays in meters relative to reference
        """
        self._compute_all()
        return self._xy_earth_cache[frame_idx]

    def xy_earth_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get x/y earth coordinates for all frames concatenated.

        Returns:
            Tuple of (x_earth, y_earth) arrays, each shape (total_pixels,)
        """
        self._compute_all()
        x_all = np.concatenate([x.ravel() for x, y in self._xy_earth_cache])
        y_all = np.concatenate([y.ravel() for x, y in self._xy_earth_cache])
        return x_all, y_all

    def latlon(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lat/lon coordinates for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        self._compute_all()
        return self._latlon_cache[frame_idx]

    def latlon_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lat/lon coordinates for all frames concatenated.

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        self._compute_all()
        lat_all = np.concatenate([lat.ravel() for lat, lon in self._latlon_cache])
        lon_all = np.concatenate([lon.ravel() for lat, lon in self._latlon_cache])
        return lat_all, lon_all

    def intensity_all(self) -> np.ndarray:
        """
        Get intensity values for all frames concatenated.

        Uses corrected_intensity if available on frames.

        Returns:
            Array of intensity values, shape (total_pixels,)
        """
        # Pre-allocate to avoid multiple copies from list append + concatenate
        total = sum(f.n_bearings * f.n_distances for f in self._frames)
        result = np.empty(total, dtype=np.float32)
        offset = 0
        for frame in self._frames:
            data = (
                frame.corrected_intensity
                if frame.corrected_intensity is not None
                else frame.deramped_intensity
                if frame.deramped_intensity is not None
                else frame.intensity
            )
            size = data.size
            result[offset : offset + size] = data.ravel()
            offset += size
        return result

    def ship_track_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get continuous ship track (x, y) in meters for each radial across all frames.

        Integrates ship motion from the first radial of the first frame,
        ensuring continuity between frames.

        Returns:
            Tuple of (x, y) arrays in meters, where x=East, y=North
            Origin is at ship position at start of first frame
        """
        # Get all radial times
        all_times = self._timestamp.times  # seconds from start of first frame

        # Build arrays of ship speed and heading for each radial
        n_total = len(all_times)
        ship_speeds = np.zeros(n_total)
        ship_headings = np.zeros(n_total)

        idx = 0
        for frame in self._frames:
            n_radials = frame.n_bearings
            # Use ship course if available, otherwise heading
            heading = frame.metadata.ship_course
            if heading is None:
                heading = frame.metadata.heading or 0.0
            speed = frame.metadata.ship_speed or 0.0

            ship_speeds[idx : idx + n_radials] = speed
            ship_headings[idx : idx + n_radials] = heading
            idx += n_radials

        # Calculate time deltas between consecutive radials
        dt = np.diff(all_times, prepend=0.0)
        dt[0] = 0.0  # First radial starts at origin

        # Calculate displacement for each time step
        heading_rad = np.deg2rad(ship_headings)
        dx = ship_speeds * dt * np.sin(heading_rad)  # East
        dy = ship_speeds * dt * np.cos(heading_rad)  # North

        # Integrate to get position
        x = np.cumsum(dx)
        y = np.cumsum(dy)

        return x, y

    def ship_track(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get continuous ship track (lat, lon) for each radial across all frames.

        Returns:
            Tuple of (latitude, longitude) arrays for ship position
        """
        x, y = self.ship_track_xy()

        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(self._ref_lat))

        lat = self._ref_lat + y / meters_per_deg_lat
        lon = self._ref_lon + x / meters_per_deg_lon

        return lat, lon

    def travel_distance(self) -> dict:
        """
        Calculate total distance traveled during the scan.

        Returns:
            Dictionary with travel statistics:
            - 'total_m': Total distance in meters
            - 'x_m': East-West distance in meters
            - 'y_m': North-South distance in meters
            - 'duration_s': Total duration in seconds
            - 'speed_m_s': Average speed in m/s
        """
        ship_x, ship_y = self.ship_track_xy()

        # Calculate distances
        dx = ship_x[-1] - ship_x[0]
        dy = ship_y[-1] - ship_y[0]
        total = np.sqrt(dx**2 + dy**2)

        # Duration
        times = self._timestamp.times
        duration = times[-1] - times[0]
        speed = total / duration if duration > 0 else 0.0

        return {
            "total_m": total,
            "x_m": dx,
            "y_m": dy,
            "duration_s": duration,
            "speed_m_s": speed,
        }

    def frame_velocities(self) -> dict:
        """
        Calculate per-frame speed and heading statistics.

        Returns:
            Dictionary with:
            - 'speeds': Array of speeds (m/s) for each frame interval
            - 'headings': Array of headings (degrees from north) for each frame interval
            - 'speed_mean', 'speed_std': Speed statistics
            - 'heading_mean', 'heading_std': Heading statistics (circular mean/std)
        """
        ship_x, ship_y = self.ship_track_xy()
        times = self._timestamp.times

        # Per-frame deltas
        dx = np.diff(ship_x)
        dy = np.diff(ship_y)
        dt = np.diff(times)

        # Speeds
        distances = np.sqrt(dx**2 + dy**2)
        speeds = distances / dt
        speeds = speeds[dt > 0]  # Filter zero-duration intervals

        # Headings (from north, clockwise)
        headings = np.degrees(np.arctan2(dx, dy)) % 360

        # Circular mean for heading
        heading_rad = np.radians(headings)
        mean_sin = np.mean(np.sin(heading_rad))
        mean_cos = np.mean(np.cos(heading_rad))
        heading_mean = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360

        # Circular standard deviation
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        heading_std = np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else 0.0

        return {
            "speeds": speeds,
            "headings": headings,
            "speed_mean": np.mean(speeds) if len(speeds) > 0 else 0.0,
            "speed_std": np.std(speeds) if len(speeds) > 0 else 0.0,
            "heading_mean": heading_mean,
            "heading_std": heading_std,
            "duration_s": times[-1] - times[0] if len(times) > 1 else 0.0,
        }

    def compute_grid_bounds(self, padding: float = 1.1) -> Tuple[float, float, float, float]:
        """
        Compute grid bounds in meters from ship track and radar range.

        Much faster than computing all pixel coordinates - only needs
        ship track and max radar range.

        Args:
            padding: Multiplier for radar range to add margin (default 1.1 = 10% margin)

        Returns:
            Tuple of (x_min, x_max, y_min, y_max) in meters from reference
        """
        # Get ship track in meters
        ship_x, ship_y = self.ship_track_xy()

        # Get max radar range from first frame
        frame = self._frames[0]
        height = self._bearing._get_radar_height(0)
        if height is not None:
            max_range = frame.ground_range(height)[-1]
        else:
            max_range = frame.slant_range()[-1]

        # Add padding
        max_range *= padding

        # Compute bounds
        x_min = ship_x.min() - max_range
        x_max = ship_x.max() + max_range
        y_min = ship_y.min() - max_range
        y_max = ship_y.max() + max_range

        return x_min, x_max, y_min, y_max

    def _grid_frame(
        self, frame_idx: int, x_edges: np.ndarray, y_edges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grid a single frame onto the predefined grid.

        Args:
            frame_idx: Frame index
            x_edges: X bin edges in meters
            y_edges: Y bin edges in meters

        Returns:
            Tuple of (sum_grid, count_grid) for averaging later
        """
        # Compute frame coordinates
        x_earth, y_earth = self._compute_frame_xy_earth(frame_idx)

        # Get intensity (use corrected if available)
        frame = self._frames[frame_idx]
        intensity = (
            frame.corrected_intensity
            if frame.corrected_intensity is not None
            else frame.deramped_intensity
            if frame.deramped_intensity is not None
            else frame.intensity
        )

        # Get shadow mask (1D per bearing) and expand to 2D
        shadow_mask_1d = self._bearing._theta.in_shadow(frame_idx)
        n_distances = intensity.shape[1]
        shadow_mask_2d = np.repeat(shadow_mask_1d[:, np.newaxis], n_distances, axis=1)

        # Flatten for binning
        x_flat = x_earth.ravel()
        y_flat = y_earth.ravel()
        values_flat = intensity.ravel().astype(np.float64)
        shadow_flat = shadow_mask_2d.ravel()

        # Compute bin indices
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        x_idx = np.searchsorted(x_edges, x_flat) - 1
        y_idx = np.searchsorted(y_edges, y_flat) - 1

        # Clip to valid range and exclude shadow region
        valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y) & ~shadow_flat
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        values_valid = values_flat[valid]

        # Accumulate into grids
        sum_grid = np.zeros((n_y, n_x), dtype=np.float64)
        count_grid = np.zeros((n_y, n_x), dtype=np.int32)

        # Use np.add.at for accumulation (faster than loops)
        np.add.at(sum_grid, (y_idx, x_idx), values_valid)
        np.add.at(count_grid, (y_idx, x_idx), 1)

        return sum_grid, count_grid

    def grid_parallel(
        self, grid_size: int = 800, workers: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Grid all frames in parallel onto a common grid.

        Args:
            grid_size: Number of bins in each dimension
            workers: Number of parallel workers (None = auto)

        Returns:
            Tuple of (x_edges, y_edges, gridded_values) in meters
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Compute grid bounds
        x_min, x_max, y_min, y_max = self.compute_grid_bounds()

        # Create grid edges in meters
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)

        n_frames = len(self._frames)
        if workers is None:
            workers = min(n_frames, 8)

        # Grid frames in parallel
        sum_total = np.zeros((grid_size, grid_size), dtype=np.float64)
        count_total = np.zeros((grid_size, grid_size), dtype=np.int32)

        if workers > 1 and n_frames > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._grid_frame, i, x_edges, y_edges): i
                    for i in range(n_frames)
                }
                for future in as_completed(futures):
                    sum_grid, count_grid = future.result()
                    sum_total += sum_grid
                    count_total += count_grid
        else:
            # Sequential fallback
            for i in range(n_frames):
                sum_grid, count_grid = self._grid_frame(i, x_edges, y_edges)
                sum_total += sum_grid
                count_total += count_grid

        # Compute mean (avoid division by zero)
        with np.errstate(invalid="ignore"):
            gridded = sum_total / count_total
        gridded[count_total == 0] = np.nan

        return x_edges, y_edges, gridded

    def compute_track_angle(self) -> float:
        """
        Compute the angle to rotate ship track to align with Y-axis.

        Returns:
            Rotation angle in radians (counter-clockwise positive)
        """
        ship_x, ship_y = self.ship_track_xy()

        # Use linear regression to find track direction
        # Or simpler: use start-to-end vector
        dx = ship_x[-1] - ship_x[0]
        dy = ship_y[-1] - ship_y[0]

        # Angle of track from Y-axis (North)
        # We want to rotate so track aligns with Y-axis
        track_angle = np.arctan2(dx, dy)  # angle from North

        return -track_angle  # negate to rotate track onto Y-axis

    def _rotate_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        angle: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate coordinates by angle around center point.

        Args:
            x, y: Coordinates to rotate
            angle: Rotation angle in radians (counter-clockwise positive)
            center_x, center_y: Center of rotation

        Returns:
            Tuple of (x_rotated, y_rotated)
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        x_centered = x - center_x
        y_centered = y - center_y

        x_rot = x_centered * cos_a - y_centered * sin_a + center_x
        y_rot = x_centered * sin_a + y_centered * cos_a + center_y

        return x_rot, y_rot

    def _create_nonuniform_edges(
        self,
        min_val: float,
        max_val: float,
        n_bins: int,
        center: float = 0.0,
        fine_fraction: float = 0.3,
        coarse_ratio: float = 4.0,
    ) -> np.ndarray:
        """
        Create non-uniform grid edges - finer near center, coarser at edges.

        Uses a piecewise linear scheme:
        - Inner region (fine_fraction of range): fine spacing
        - Outer regions: coarser spacing (coarse_ratio times coarser)

        Args:
            min_val, max_val: Range of values
            n_bins: Total number of bins
            center: Center point for fine resolution
            fine_fraction: Fraction of total range for fine resolution
            coarse_ratio: Ratio of coarse to fine bin size

        Returns:
            Array of bin edges
        """
        total_range = max_val - min_val
        fine_range = total_range * fine_fraction

        # Determine fine region bounds
        fine_min = max(min_val, center - fine_range / 2)
        fine_max = min(max_val, center + fine_range / 2)
        actual_fine_range = fine_max - fine_min

        # Calculate bin allocation
        # Let f = fine bin size, c = coarse bin size = coarse_ratio * f
        # n_fine * f + n_coarse * c = total_range
        # n_fine + n_coarse = n_bins
        # Solve for distribution

        left_range = fine_min - min_val
        right_range = max_val - fine_max

        # Allocate bins proportionally but account for coarse ratio
        # Effective "work" for each region
        fine_work = actual_fine_range
        coarse_work = (left_range + right_range) / coarse_ratio

        total_work = fine_work + coarse_work
        if total_work <= 0:
            return np.linspace(min_val, max_val, n_bins + 1)

        n_fine = int(n_bins * fine_work / total_work)
        n_coarse_left = int(n_bins * (left_range / coarse_ratio) / total_work)
        n_coarse_right = n_bins - n_fine - n_coarse_left

        # Ensure minimums
        n_fine = max(n_fine, 10)
        n_coarse_left = max(n_coarse_left, 1) if left_range > 0 else 0
        n_coarse_right = max(n_coarse_right, 1) if right_range > 0 else 0

        # Build edges
        edges = []

        if n_coarse_left > 0:
            left_edges = np.linspace(min_val, fine_min, n_coarse_left + 1)
            edges.extend(left_edges[:-1])

        fine_edges = np.linspace(fine_min, fine_max, n_fine + 1)
        edges.extend(fine_edges[:-1])

        if n_coarse_right > 0:
            right_edges = np.linspace(fine_max, max_val, n_coarse_right + 1)
            edges.extend(right_edges)
        else:
            edges.append(fine_max)

        return np.array(edges)

    def _grid_frame_rotated(
        self, frame_idx: int, x_edges: np.ndarray, y_edges: np.ndarray, angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grid a single frame onto the rotated non-uniform grid.

        Memory-optimized: deletes intermediate arrays as soon as possible.

        Args:
            frame_idx: Frame index
            x_edges: X bin edges (in rotated coordinates)
            y_edges: Y bin edges (in rotated coordinates)
            angle: Rotation angle applied to coordinates

        Returns:
            Tuple of (sum_grid, count_grid) for averaging later
        """
        # Compute frame coordinates
        x_earth, y_earth = self._compute_frame_xy_earth(frame_idx)

        # Rotate to grid coordinates (in-place to save memory)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        x_rot = x_earth * cos_a - y_earth * sin_a
        y_rot = x_earth * sin_a + y_earth * cos_a
        del x_earth, y_earth  # Free memory immediately

        # Get intensity (use processed if available)
        frame = self._frames[frame_idx]
        intensity = (
            frame.corrected_intensity
            if frame.corrected_intensity is not None
            else frame.deramped_intensity
            if frame.deramped_intensity is not None
            else frame.intensity
        )

        # Get shadow mask (1D per bearing)
        shadow_mask_1d = self._bearing._theta.in_shadow(frame_idx)

        # Bin indices using searchsorted (works with non-uniform edges)
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        # Process in flattened form to reduce memory copies
        x_flat = x_rot.ravel()
        del x_rot
        y_flat = y_rot.ravel()
        del y_rot

        x_idx = np.searchsorted(x_edges, x_flat) - 1
        del x_flat
        y_idx = np.searchsorted(y_edges, y_flat) - 1
        del y_flat

        # Expand shadow mask efficiently (broadcast instead of repeat)
        n_bearings, n_distances = intensity.shape
        shadow_flat = np.broadcast_to(
            shadow_mask_1d[:, np.newaxis], (n_bearings, n_distances)
        ).ravel()

        # Clip to valid range and exclude shadow region
        valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y) & ~shadow_flat
        del shadow_flat

        x_idx_valid = x_idx[valid]
        del x_idx
        y_idx_valid = y_idx[valid]
        del y_idx
        values_valid = intensity.ravel()[valid].astype(np.float64)
        del valid

        # Accumulate sum and count
        sum_grid = np.zeros((n_y, n_x), dtype=np.float64)
        count_grid = np.zeros((n_y, n_x), dtype=np.int32)

        np.add.at(sum_grid, (y_idx_valid, x_idx_valid), values_valid)
        np.add.at(count_grid, (y_idx_valid, x_idx_valid), 1)

        return sum_grid, count_grid

    def grid_parallel_rotated(
        self, n_along: int = 600, n_cross: int = 800, workers: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Grid all frames in parallel onto a rotated non-uniform grid.

        Rotates coordinates to align ship track with Y-axis, then uses:
        - Uniform spacing along track (Y)
        - Non-uniform spacing cross track (X) - finer near ship track

        Args:
            n_along: Number of bins along track direction
            n_cross: Number of bins cross track direction
            workers: Number of parallel workers (None = auto)

        Returns:
            Tuple of (x_edges, y_edges, gridded_values, rotation_angle)
            Edges are in rotated coordinates
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Compute rotation angle to align track with Y-axis
        angle = self.compute_track_angle()

        # Get ship track and rotate
        ship_x, ship_y = self.ship_track_xy()
        ship_x_rot, ship_y_rot = self._rotate_coords(ship_x, ship_y, angle)

        # Get max radar range
        frame = self._frames[0]
        height = self._bearing._get_radar_height(0)
        max_range = frame.ground_range(height)[-1] if height else frame.slant_range()[-1]
        max_range *= 1.1  # padding

        # Compute bounds in rotated coordinates
        # Along track (Y): based on ship track extent + radar range
        y_min = ship_y_rot.min() - max_range
        y_max = ship_y_rot.max() + max_range

        # Cross track (X): based on ship track extent + radar range
        # (accounts for lateral ship movement, not just track center)
        x_min = ship_x_rot.min() - max_range
        x_max = ship_x_rot.max() + max_range

        # Keep track center for non-uniform grid
        track_x_center = (ship_x_rot.min() + ship_x_rot.max()) / 2

        # Create grid edges
        # Along track: uniform (ship moves at constant speed)
        y_edges = np.linspace(y_min, y_max, n_along + 1)

        # Cross track: non-uniform (finer near ship track)
        x_edges = self._create_nonuniform_edges(
            x_min,
            x_max,
            n_cross,
            center=track_x_center,
            fine_fraction=0.2,  # 20% of width at fine resolution
            coarse_ratio=3.0,  # edges 3x coarser than center
        )

        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        n_frames = len(self._frames)
        if workers is None:
            workers = min(n_frames, 8)

        # Grid frames in parallel
        sum_total = np.zeros((n_y, n_x), dtype=np.float64)
        count_total = np.zeros((n_y, n_x), dtype=np.int32)

        if workers > 1 and n_frames > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._grid_frame_rotated, i, x_edges, y_edges, angle): i
                    for i in range(n_frames)
                }
                for future in as_completed(futures):
                    sum_grid, count_grid = future.result()
                    sum_total += sum_grid
                    count_total += count_grid
        else:
            import gc

            for i in range(n_frames):
                sum_grid, count_grid = self._grid_frame_rotated(i, x_edges, y_edges, angle)
                sum_total += sum_grid
                count_total += count_grid
                del sum_grid, count_grid
                # Periodically collect garbage to keep memory usage down
                if i % 50 == 0:
                    gc.collect()

        # Compute mean
        with np.errstate(invalid="ignore"):
            gridded = sum_total / count_total
        gridded[count_total == 0] = np.nan

        return x_edges, y_edges, gridded, angle

    def _grid_data(
        self, lon: np.ndarray, lat: np.ndarray, values: np.ndarray, grid_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Grid irregular data onto a regular lat/lon grid using binned statistics.

        Much faster than scatter plots for large datasets.

        Args:
            lon: Longitude values (1D)
            lat: Latitude values (1D)
            values: Data values to grid (1D)
            grid_size: Number of bins in each dimension

        Returns:
            Tuple of (lon_edges, lat_edges, gridded_values)
        """
        from scipy.stats import binned_statistic_2d

        # Create grid edges
        lon_edges = np.linspace(lon.min(), lon.max(), grid_size + 1)
        lat_edges = np.linspace(lat.min(), lat.max(), grid_size + 1)

        # Bin the data using mean
        result, _, _, _ = binned_statistic_2d(
            lon, lat, values, statistic="mean", bins=[lon_edges, lat_edges]
        )

        return lon_edges, lat_edges, result.T  # Transpose to match pcolormesh expectation

    def _xy_to_latlon_edges(
        self, x_edges: np.ndarray, y_edges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert x/y edges in meters to lat/lon edges.

        Args:
            x_edges: X bin edges in meters (East)
            y_edges: Y bin edges in meters (North)

        Returns:
            Tuple of (lon_edges, lat_edges) in degrees
        """
        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(self._ref_lat))

        lon_edges = self._ref_lon + x_edges / meters_per_deg_lon
        lat_edges = self._ref_lat + y_edges / meters_per_deg_lat

        return lon_edges, lat_edges

    # -------------------------------------------------------------------------
    # Plotting methods (delegate to combine_plot module)
    # -------------------------------------------------------------------------

    def plot_diagnostics(
        self,
        figsize: tuple[float, float] = (10, 10),
        n_along: int = 1200,
        n_cross: int = 1600,
        workers: int | None = None,
        show_track: bool = False,
    ) -> None:
        """
        Show diagnostic plots for the combined frames in earth coordinates.

        Uses parallel gridding for fast rendering of large datasets.

        Args:
            figsize: Figure size
            n_along: Grid bins along ship track (default 1200)
            n_cross: Grid bins cross track (default 1600)
            workers: Number of parallel workers for gridding (None = auto)
            show_track: Show separate ship track plot (default False)
        """
        from wamos_tpw.combine_plot import plot_diagnostics as _plot_diagnostics

        _plot_diagnostics(
            self,
            figsize=figsize,
            n_along=n_along,
            n_cross=n_cross,
            workers=workers,
            show_track=show_track,
        )

    def save_frame(
        self,
        output_path: str,
        figsize: tuple[float, float] = (10, 10),
        n_along: int = 1200,
        n_cross: int = 1600,
        workers: int | None = None,
        dpi: int = 100,
    ) -> None:
        """
        Save a single frame image non-interactively.

        Args:
            output_path: Path to save the image (e.g., frame_001.png)
            figsize: Figure size in inches
            n_along: Grid bins along ship track
            n_cross: Grid bins cross track
            workers: Number of parallel workers for gridding
            dpi: Image resolution
        """
        from wamos_tpw.combine_plot import save_frame as _save_frame

        _save_frame(
            self,
            output_path,
            figsize=figsize,
            n_along=n_along,
            n_cross=n_cross,
            workers=workers,
            dpi=dpi,
        )

    def clear_cache(self) -> None:
        """
        Clear all cached data to free memory.

        Call this after you're done with the Combine object to release memory
        used by coordinate caches and nested objects.
        """
        self._xy_earth_cache = None
        self._latlon_cache = None
        self._bearing.clear_cache()
        self._bearing._theta.clear_shadow_data()
        for frame in self._frames:
            frame.clear_cache()
            frame.deramped_intensity = None
            frame.corrected_intensity = None

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self._frames)

    def __repr__(self) -> str:
        vel = self.frame_velocities()
        return (
            f"Combine(frames={len(self._frames)}, "
            f"duration={vel['duration_s']:.1f}s, "
            f"speed={vel['speed_mean']:.2f}±{vel['speed_std']:.2f}m/s, "
            f"heading={vel['heading_mean']:.0f}±{vel['heading_std']:.0f}°)"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument(
        "--groupby", "-g", type=str, default="h", help="Groupby frequency (default: h)"
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument(
        "--radar-height", type=float, default=None, help="Radar height above water (m)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process per group"
    )
    parser.add_argument(
        "--no-process",
        dest="process",
        action="store_false",
        help="Skip processing (deramp + destreak)",
    )
    parser.set_defaults(process=True)
    parser.add_argument(
        "--plot", action="store_true", help="Show interactive viewer with prev/next/play buttons"
    )
    parser.add_argument(
        "--movie", type=str, default=None, help="Output movie file (e.g., output.mp4)"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory to save frame images (persistent, not deleted)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip existing frames in --frames-dir)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for movie (default: 10)"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers for movie generation (default: auto)",
    )
    parser.add_argument(
        "--netcdf", type=str, default=None, help="Output NetCDF file (e.g., output.nc)"
    )
    parser.add_argument("--show-track", action="store_true", help="Show separate ship track plot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling (shows peak memory usage)",
    )


def add_subparser(subparsers) -> None:
    """Register the 'combine' subcommand."""
    p = subparsers.add_parser(
        "combine",
        help="Combine frames into earth coordinates",
        description="Combine multiple frames with ship motion compensation",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'combine' command."""
    import gc

    from wamos_tpw.combine_netcdf import NetCDFWriter
    from wamos_tpw.combine_plot import CombineViewer, grid_group
    from wamos_tpw.processed import ProcessedFrames

    # Load config
    config = Config(args.config) if args.config else Config()

    # Handle --dry-run: show what would be processed without doing it
    if hasattr(args, "dry_run") and args.dry_run:
        with ProcessedFrames(
            stime=args.stime,
            etime=args.etime,
            polar_path=args.polar_path,
            groupby=args.groupby,
            config=config,
            radar_height=args.radar_height,
        ) as pframes:
            logging.info("=== DRY RUN MODE ===")
            logging.info(f"Discovered {len(pframes)} files")

            groups = pframes.groups()
            logging.info(f"Would process {len(groups)} groups")

            total_files = 0
            for period, file_list in groups.items():
                file_count = len(file_list)
                total_files += file_count
                logging.info(f"  {period}: {file_count} files")

            logging.info(f"Total: {total_files} files in {len(groups)} groups")

            if args.movie:
                logging.info(f"Would generate movie: {args.movie}")
            if args.frames_dir:
                logging.info(f"Would save frames to: {args.frames_dir}")
            if args.netcdf:
                logging.info(f"Would save NetCDF to: {args.netcdf}")
            if args.plot:
                logging.info("Would show interactive viewer")

            logging.info("=== END DRY RUN ===")
        return

    # For --plot: grid each group immediately to minimize memory usage
    if args.plot:
        # Create viewer (we'll count groups as we go)
        viewer = CombineViewer(total_groups=0)
        n_groups = 0

        with ProcessedFrames(
            stime=args.stime,
            etime=args.etime,
            polar_path=args.polar_path,
            groupby=args.groupby,
            config=config,
            radar_height=args.radar_height,
        ) as pframes:
            logging.info(f"Discovered {len(pframes)} files")

            for period, frames in pframes.itergroups():
                frames = list(frames)
                if args.max_frames:
                    frames = frames[: args.max_frames]
                logging.info(f"{period}: {len(frames)} frames")

                if not frames:
                    continue

                # Optionally process frames
                if args.process:
                    logging.debug("Processing: refine_theta, deramp, destreak...")
                    corrected = pframes.process_group(frames)
                    for frame, corr in zip(frames, corrected):
                        frame.corrected_intensity = corr

                # Create Combine, grid immediately, then discard frames
                combine = Combine(frames, config, radar_height=args.radar_height)
                logging.debug(f"{combine}")

                # Grid this group immediately
                group_data = grid_group(str(period), combine, 1200, 1600)
                n_groups += 1
                logging.info(f"Gridded group {n_groups}: {group_data['period']}")

                # Add to viewer (only gridded data retained, frames discarded)
                viewer.add_group_data(group_data)

                # Clear caches and free memory
                combine.clear_cache()
                del combine, frames, group_data
                gc.collect()

        if n_groups == 0:
            logging.warning("No groups to display")
            return

        # Mark loading complete
        viewer.set_loading_complete()
        logging.info(f"All {n_groups} groups loaded.")
        logging.info("Navigation: <- -> keys, Prev/Next buttons, Space=Play/Stop")

        # Show viewer
        viewer.show()
        return

    # For --netcdf: grid each group and append immediately to minimize memory
    if args.netcdf:
        n_groups = 0

        with NetCDFWriter(args.netcdf) as nc_writer:
            with ProcessedFrames(
                stime=args.stime,
                etime=args.etime,
                polar_path=args.polar_path,
                groupby=args.groupby,
                config=config,
                radar_height=args.radar_height,
            ) as pframes:
                logging.info(f"Discovered {len(pframes)} files")
                logging.info(f"Writing to: {args.netcdf}")

                for period, frames in pframes.itergroups():
                    frames = list(frames)
                    if args.max_frames:
                        frames = frames[: args.max_frames]
                    logging.info(f"{period}: {len(frames)} frames")

                    if not frames:
                        continue

                    # Optionally process frames
                    if args.process:
                        logging.debug("Processing: refine_theta, deramp, destreak...")
                        corrected = pframes.process_group(frames)
                        for frame, corr in zip(frames, corrected):
                            frame.corrected_intensity = corr

                    # Create Combine, grid immediately, then discard frames
                    combine = Combine(frames, config, radar_height=args.radar_height)
                    logging.debug(f"{combine}")

                    # Grid this group immediately
                    group_data = grid_group(str(period), combine, 1200, 1600)
                    n_groups += 1
                    logging.info(f"Gridded and saved group {n_groups}: {group_data['period']}")

                    # Append to NetCDF (writes to disk immediately)
                    nc_writer.append_group(group_data)

                    # Clear caches and free memory
                    combine.clear_cache()
                    del combine, frames, group_data
                    gc.collect()

        if n_groups == 0:
            logging.warning("No groups to save")
        else:
            logging.info(f"Saved {n_groups} groups to: {args.netcdf}")
        return

    # Movie/frames generation mode
    if args.movie or args.frames_dir:
        from wamos_tpw.combine_movie import generate_movie

        generate_movie(args, config)
        return


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Combine frames into earth coordinate image")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
