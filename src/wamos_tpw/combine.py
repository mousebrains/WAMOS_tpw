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

from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.config import WamosConfig
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
        >>> config = WamosConfig('radar_config.yaml')
        >>> combine = Combine(frames, config)
        >>> x_earth, y_earth = combine.xy_earth_all()  # (n_total_pixels,), (n_total_pixels,)
        >>> lat, lon = combine.latlon_all()  # (n_total_pixels,), (n_total_pixels,)
        >>> combine.plot_diagnostics()  # Show earth coordinate plot
    """

    # Earth radius in meters (WGS84 mean radius)
    _EARTH_RADIUS = 6371000.0

    def __init__(self,
                 frames: list[Frame],
                 config: WamosConfig | None = None,
                 radar_height: float | None = None):
        """
        Initialize Combine for a set of contiguous frames.

        Args:
            frames: List of contiguous Frame objects (sorted by time)
            config: WamosConfig object (uses defaults if None)
            radar_height: Radar height above water (m). If None, uses config or metadata.
        """
        if not frames:
            raise ValueError("At least one frame is required")

        self._frames = frames
        self._config = config or WamosConfig()

        # Create Theta/Bearing for coordinate calculation
        self._theta = Theta(frames, config, refine=True)
        self._bearing = Bearing(self._theta, radar_height=radar_height)

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
    def config(self) -> WamosConfig:
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

    def _compute_frame_latlon(self,
                               frame_idx: int,
                               x_earth: np.ndarray,
                               y_earth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        intensities = []
        for frame in self._frames:
            data = getattr(frame, 'corrected_intensity',
                          getattr(frame, 'deramped_intensity', frame.intensity))
            intensities.append(data.ravel())
        return np.concatenate(intensities)

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

            ship_speeds[idx:idx + n_radials] = speed
            ship_headings[idx:idx + n_radials] = heading
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
            'total_m': total,
            'x_m': dx,
            'y_m': dy,
            'duration_s': duration,
            'speed_m_s': speed,
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
            'speeds': speeds,
            'headings': headings,
            'speed_mean': np.mean(speeds) if len(speeds) > 0 else 0.0,
            'speed_std': np.std(speeds) if len(speeds) > 0 else 0.0,
            'heading_mean': heading_mean,
            'heading_std': heading_std,
            'duration_s': times[-1] - times[0] if len(times) > 1 else 0.0,
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

    def _grid_frame(self,
                    frame_idx: int,
                    x_edges: np.ndarray,
                    y_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        intensity = getattr(frame, 'corrected_intensity',
                           getattr(frame, 'deramped_intensity', frame.intensity))

        # Flatten for binning
        x_flat = x_earth.ravel()
        y_flat = y_earth.ravel()
        values_flat = intensity.ravel().astype(np.float64)

        # Compute bin indices
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        x_idx = np.searchsorted(x_edges, x_flat) - 1
        y_idx = np.searchsorted(y_edges, y_flat) - 1

        # Clip to valid range
        valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y)
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

    def grid_parallel(self,
                      grid_size: int = 800,
                      workers: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        with np.errstate(invalid='ignore'):
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

    def _rotate_coords(self,
                       x: np.ndarray,
                       y: np.ndarray,
                       angle: float,
                       center_x: float = 0.0,
                       center_y: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
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

    def _create_nonuniform_edges(self,
                                  min_val: float,
                                  max_val: float,
                                  n_bins: int,
                                  center: float = 0.0,
                                  fine_fraction: float = 0.3,
                                  coarse_ratio: float = 4.0) -> np.ndarray:
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

    def _grid_frame_rotated(self,
                            frame_idx: int,
                            x_edges: np.ndarray,
                            y_edges: np.ndarray,
                            angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grid a single frame onto the rotated non-uniform grid.

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

        # Rotate to grid coordinates
        x_rot, y_rot = self._rotate_coords(x_earth, y_earth, angle)

        # Get intensity (use processed if available)
        frame = self._frames[frame_idx]
        intensity = getattr(frame, 'corrected_intensity',
                           getattr(frame, 'deramped_intensity', frame.intensity))

        # Flatten
        x_flat = x_rot.ravel()
        y_flat = y_rot.ravel()
        values_flat = intensity.ravel().astype(np.float64)

        # Bin indices using searchsorted (works with non-uniform edges)
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        x_idx = np.searchsorted(x_edges, x_flat) - 1
        y_idx = np.searchsorted(y_edges, y_flat) - 1

        # Clip to valid range
        valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y)
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        values_valid = values_flat[valid]

        # Accumulate sum and count
        sum_grid = np.zeros((n_y, n_x), dtype=np.float64)
        count_grid = np.zeros((n_y, n_x), dtype=np.int32)

        np.add.at(sum_grid, (y_idx, x_idx), values_valid)
        np.add.at(count_grid, (y_idx, x_idx), 1)

        return sum_grid, count_grid

    def grid_parallel_rotated(self,
                               n_along: int = 600,
                               n_cross: int = 800,
                               workers: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
        # Along track (Y): based on ship track extent
        y_min = ship_y_rot.min() - max_range
        y_max = ship_y_rot.max() + max_range

        # Cross track (X): based on radar range from track center
        track_x_center = (ship_x_rot.min() + ship_x_rot.max()) / 2
        x_min = track_x_center - max_range
        x_max = track_x_center + max_range

        # Create grid edges
        # Along track: uniform (ship moves at constant speed)
        y_edges = np.linspace(y_min, y_max, n_along + 1)

        # Cross track: non-uniform (finer near ship track)
        x_edges = self._create_nonuniform_edges(
            x_min, x_max, n_cross,
            center=track_x_center,
            fine_fraction=0.2,  # 20% of width at fine resolution
            coarse_ratio=3.0    # edges 3x coarser than center
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
            for i in range(n_frames):
                sum_grid, count_grid = self._grid_frame_rotated(i, x_edges, y_edges, angle)
                sum_total += sum_grid
                count_total += count_grid

        # Compute mean
        with np.errstate(invalid='ignore'):
            gridded = sum_total / count_total
        gridded[count_total == 0] = np.nan

        return x_edges, y_edges, gridded, angle

    def _grid_data(self,
                   lon: np.ndarray,
                   lat: np.ndarray,
                   values: np.ndarray,
                   grid_size: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            lon, lat, values,
            statistic='mean',
            bins=[lon_edges, lat_edges]
        )

        return lon_edges, lat_edges, result.T  # Transpose to match pcolormesh expectation

    def _xy_to_latlon_edges(self,
                             x_edges: np.ndarray,
                             y_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def plot_diagnostics(self,
                         figsize: tuple[float, float] = (10, 10),
                         n_along: int = 1200,
                         n_cross: int = 1600,
                         workers: int | None = None,
                         show_track: bool = False) -> None:
        """
        Show diagnostic plots for the combined frames in earth coordinates.

        Uses parallel gridding for fast rendering of large datasets.

        Args:
            figsize: Figure size
            n_along: Grid bins along ship track (default 600)
            n_cross: Grid bins cross track (default 800)
            workers: Number of parallel workers for gridding (None = auto)
            show_track: Show separate ship track plot (default False)

        Displays:
        - Combined intensity in lat/lon coordinates (gridded)
        - Polar scatter plots for ship speed/heading and wind
        - Optional ship track detail plot
        - Statistics panel
        """
        import matplotlib.pyplot as plt

        # Get ship track (fast - doesn't require full coordinate computation)
        print("Computing ship track...", flush=True)
        ship_x, ship_y = self.ship_track_xy()
        ship_lat, ship_lon = self.ship_track()
        travel = self.travel_distance()
        n_radials = len(ship_x)
        n_frames = len(self._frames)

        # Estimate pixel count
        n_pixels = sum(f.n_bearings * f.n_distances for f in self._frames)

        # Use rotated grid for speed, then display in lat/lon
        print(f"Gridding {n_pixels:,} pixels from {n_frames} frames "
              f"to {n_cross}x{n_along} rotated grid (parallel)...", flush=True)

        # Grid all frames in parallel with rotation
        x_edges, y_edges, gridded, angle = self.grid_parallel_rotated(
            n_along=n_along, n_cross=n_cross, workers=workers
        )

        # Convert edges to lat/lon for display
        # For rotated grid, we need to rotate back first
        # Create corner points of the grid and rotate back
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        # Create meshgrid of rotated coordinates
        xx_rot, yy_rot = np.meshgrid(x_edges, y_edges)

        # Rotate back to earth coordinates
        xx_earth = xx_rot * cos_a - yy_rot * sin_a
        yy_earth = xx_rot * sin_a + yy_rot * cos_a

        # Convert to lat/lon
        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(self._ref_lat))

        lon_grid = self._ref_lon + xx_earth / meters_per_deg_lon
        lat_grid = self._ref_lat + yy_earth / meters_per_deg_lat

        # Calculate intensity limits from gridded data (ignoring NaN)
        valid_data = gridded[~np.isnan(gridded)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [1, 99])
        else:
            vmin, vmax = 0, 1

        print("Creating plot...", flush=True)

        # Collect ship and wind data from all frames for polar plots
        ship_speeds = []
        ship_headings = []
        wind_speeds = []
        wind_dirs = []
        for frame in self._frames:
            meta = frame.metadata
            # Ship data
            speed = meta.ship_speed
            heading = meta.ship_course if meta.ship_course is not None else meta.heading
            if speed is not None and heading is not None:
                ship_speeds.append(speed)
                ship_headings.append(heading)
            # Wind data
            wind_spd = meta.wind_speed
            wind_dir = meta.wind_direction
            if wind_spd is not None and wind_dir is not None:
                wind_speeds.append(wind_spd)
                wind_dirs.append(wind_dir)

        # Create figure layout
        if show_track:
            # With track: 2 columns top, stats bottom
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.3)
            ax_main = fig.add_subplot(gs[0, 0])
            ax_track = fig.add_subplot(gs[0, 1])
            ax_info = fig.add_subplot(gs[1, :])
        else:
            # No track: main plot with polar insets, stats bottom
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 1, height_ratios=[12, 1], hspace=0.02)
            ax_main = fig.add_subplot(gs[0, 0])
            ax_info = fig.add_subplot(gs[1, 0])
            fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)

        # Round start time down, end time up to nearest second
        import pandas as pd
        start_ts = pd.Timestamp(self._frames[0].timestamp).floor('s')
        end_ts = pd.Timestamp(self._frames[-1].timestamp).ceil('s')
        title_str = f'{start_ts} to {end_ts}'

        # Main plot: Combined intensity in lat/lon
        im = ax_main.pcolormesh(lon_grid, lat_grid, gridded,
                                cmap='viridis', vmin=vmin, vmax=vmax,
                                shading='flat')
        plt.colorbar(im, ax=ax_main, label='Intensity')

        # Overlay ship track (subsample if too many points)
        if n_radials > 5000:
            step = n_radials // 5000
            track_lon = ship_lon[::step]
            track_lat = ship_lat[::step]
        else:
            track_lon = ship_lon
            track_lat = ship_lat

        ax_main.plot(track_lon, track_lat, 'r-', linewidth=1.5)

        # Find extent of valid (non-NaN, non-zero) intensity data
        valid_mask = np.logical_and(
                np.logical_not(np.isnan(gridded)),
                (gridded != 0))
        valid_rows, valid_cols = np.where(valid_mask)

        if len(valid_rows) > 0:
            # Get lon/lat bounds from all valid data pixels
            lon_vals = lon_grid[valid_rows, valid_cols]
            lon_min, lon_max = lon_vals.min(), lon_vals.max()
            lat_vals = lat_grid[valid_rows, valid_cols]
            lat_min, lat_max = lat_vals.min(), lat_vals.max()
        else:
            lon_min, lon_max = lon_grid.min(), lon_grid.max()
            lat_min, lat_max = lat_grid.min(), lat_grid.max()

        # Aspect ratio for lat/lon: 1/cos(lat) because longitude degrees shrink with latitude
        mean_lat = (lat_min + lat_max) / 2
        aspect_ratio = 1.0 / np.cos(np.deg2rad(mean_lat))

        data_lon_range = lon_max - lon_min
        data_lat_range = lat_max - lat_min

        # Get axes dimensions in figure coordinates to determine available space
        fig_width, fig_height = fig.get_size_inches()
        plot_width = fig_width * (0.88 - 0.12)
        plot_height = fig_height * (0.88 - 0.12) * (12/13)

        # Current data dimensions in "distance units" (lat as reference)
        data_width_dist = data_lon_range / aspect_ratio
        data_height_dist = data_lat_range

        # Determine which dimension to expand to fill plot
        plot_aspect = plot_width / plot_height
        data_aspect = data_width_dist / data_height_dist if data_height_dist > 0 else 1.0

        if plot_aspect > data_aspect:
            # Plot is wider than data - expand longitude
            new_lon_range = data_lat_range * plot_aspect * aspect_ratio
            lon_center = (lon_min + lon_max) / 2
            lon_min = lon_center - new_lon_range / 2
            lon_max = lon_center + new_lon_range / 2
        else:
            # Plot is taller than data - expand latitude
            new_lat_range = data_width_dist / plot_aspect
            lat_center = (lat_min + lat_max) / 2
            lat_min = lat_center - new_lat_range / 2
            lat_max = lat_center + new_lat_range / 2

        ax_main.set_xlim(lon_min, lon_max)
        ax_main.set_ylim(lat_min, lat_max)

        ax_main.set_xlabel('Longitude (°)')
        ax_main.set_ylabel('Latitude (°)')
        ax_main.set_title(title_str, fontsize=11)
        ax_main.set_aspect(aspect_ratio)
        ax_main.margins(0)
        ax_main.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Add polar scatter plots as insets (if not showing track)
        if not show_track:
            # Ship polar in upper right corner of main axes
            if ship_speeds and ship_headings:
                ax_ship = ax_main.inset_axes([0.88, 0.85, 0.11, 0.14], projection='polar')
                ax_ship.set_theta_zero_location('N')
                ax_ship.set_theta_direction(-1)
                headings_rad = np.deg2rad(ship_headings)
                # Color by age - viridis is colorblind-friendly (dark=oldest, bright=newest)
                ages = np.linspace(0, 1, len(ship_speeds))
                ax_ship.scatter(headings_rad, ship_speeds, c=ages, cmap='viridis',
                               s=6, alpha=0.8)
                ax_ship.set_xticklabels([])  # Remove angular labels
                ax_ship.tick_params(labelsize=4)
                ax_ship.set_facecolor('white')
                ax_ship.patch.set_alpha(0.8)
                ax_ship.grid(True, linewidth=0.3, alpha=0.5)
                # Label northwest of polar plot (left of and near top)
                ax_main.text(0.87, 0.99, 'Ship', transform=ax_main.transAxes,
                            fontsize=7, ha='right', va='top')

            # Wind polar in lower right corner of main axes
            if wind_speeds and wind_dirs:
                ax_wind = ax_main.inset_axes([0.88, 0.01, 0.11, 0.14], projection='polar')
                ax_wind.set_theta_zero_location('N')
                ax_wind.set_theta_direction(-1)
                wind_rad = np.deg2rad(wind_dirs)
                # Color by age - viridis is colorblind-friendly (dark=oldest, bright=newest)
                ages = np.linspace(0, 1, len(wind_speeds))
                ax_wind.scatter(wind_rad, wind_speeds, c=ages, cmap='viridis',
                               s=6, alpha=0.8)
                ax_wind.set_xticklabels([])  # Remove angular labels
                ax_wind.tick_params(labelsize=4)
                ax_wind.set_facecolor('white')
                ax_wind.patch.set_alpha(0.8)
                ax_wind.grid(True, linewidth=0.3, alpha=0.5)
                # Label southwest of polar plot (left of and near bottom)
                ax_main.text(0.87, 0.01, 'Wind', transform=ax_main.transAxes,
                            fontsize=7, ha='right', va='bottom')

        # Optional track detail plot
        if show_track:
            if n_radials > 10000:
                step = n_radials // 10000
                track_x = ship_x[::step]
                track_y = ship_y[::step]
                colors = np.linspace(0, 1, len(track_x))
            else:
                track_x = ship_x
                track_y = ship_y
                colors = np.linspace(0, 1, n_radials)

            ax_track.scatter(track_x, track_y, c=colors, cmap='coolwarm', s=2)
            ax_track.plot(ship_x[0], ship_y[0], 'go', markersize=10, label='Start', zorder=5)
            ax_track.plot(ship_x[-1], ship_y[-1], 'r^', markersize=10, label='End', zorder=5)

            ax_track.set_xlabel('X - East (m)')
            ax_track.set_ylabel('Y - North (m)')
            ax_track.set_title('Ship Track During Scan')
            ax_track.legend(fontsize=8, loc='best')
            ax_track.set_aspect('equal')
            ax_track.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax_track.axvline(0, color='gray', linestyle=':', alpha=0.5)

        # Statistics text
        ax_info.axis('off')

        # Get frame metadata
        meta = self._frames[0].metadata
        ship_heading = meta.heading or 0.0
        ship_speed_val = meta.ship_speed or 0.0
        ship_course = meta.ship_course
        wind_speed_val = meta.wind_speed
        wind_dir_val = meta.wind_direction

        # Calculate coverage
        x_range = x_edges[-1] - x_edges[0]
        y_range = y_edges[-1] - y_edges[0]
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1

        info_text = (
            f"Reference: ({self._ref_lat:.6f}°, {self._ref_lon:.6f}°)    "
            f"Coverage: {x_range:.0f}m × {y_range:.0f}m    "
            f"Grid: {n_x}×{n_y}\n"
            f"Ship Motion: {travel['duration_s']:.1f}s, {travel['total_m']:.1f}m total "
            f"(E: {travel['x_m']:.1f}m, N: {travel['y_m']:.1f}m), "
            f"avg {travel['speed_m_s']:.2f} m/s\n"
            f"Navigation: heading {ship_heading:.1f}°, speed {ship_speed_val:.2f} m/s"
        )
        if ship_course is not None:
            info_text += f", course {ship_course:.1f}°"
        if wind_speed_val is not None and wind_dir_val is not None:
            info_text += f"    Wind: {wind_speed_val:.1f} m/s from {wind_dir_val:.0f}°"
        info_text += f"\nFrames: {n_frames}, Radials: {n_radials:,}, Pixels: {n_pixels:,}"

        ax_info.text(0.5, 0.5, info_text,
                     transform=ax_info.transAxes,
                     ha='center', va='center',
                     fontfamily='monospace',
                     fontsize=9)

        print("Displaying...", flush=True)
        plt.show()

    def save_frame(self,
                   output_path: str,
                   figsize: tuple[float, float] = (10, 10),
                   n_along: int = 1200,
                   n_cross: int = 1600,
                   workers: int | None = None,
                   dpi: int = 100) -> None:
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
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Get ship track
        ship_x, ship_y = self.ship_track_xy()
        ship_lat, ship_lon = self.ship_track()
        n_radials = len(ship_x)
        n_frames = len(self._frames)

        # Grid all frames in parallel with rotation
        x_edges, y_edges, gridded, angle = self.grid_parallel_rotated(
            n_along=n_along, n_cross=n_cross, workers=workers
        )

        # Rotate grid back to earth coordinates and convert to lat/lon
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        xx_rot, yy_rot = np.meshgrid(x_edges, y_edges)
        xx_earth = xx_rot * cos_a - yy_rot * sin_a
        yy_earth = xx_rot * sin_a + yy_rot * cos_a

        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(self._ref_lat))
        lon_grid = self._ref_lon + xx_earth / meters_per_deg_lon
        lat_grid = self._ref_lat + yy_earth / meters_per_deg_lat

        # Calculate intensity limits
        valid_data = gridded[~np.isnan(gridded)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [1, 99])
        else:
            vmin, vmax = 0, 1

        # Collect ship and wind data
        ship_speeds, ship_headings, wind_speeds, wind_dirs = [], [], [], []
        for frame in self._frames:
            meta = frame.metadata
            speed = meta.ship_speed
            heading = meta.ship_course if meta.ship_course is not None else meta.heading
            if speed is not None and heading is not None:
                ship_speeds.append(speed)
                ship_headings.append(heading)
            if meta.wind_speed is not None and meta.wind_direction is not None:
                wind_speeds.append(meta.wind_speed)
                wind_dirs.append(meta.wind_direction)

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[12, 1], hspace=0.02)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[1, 0])
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)

        # Title with rounded timestamps
        import pandas as pd
        start_ts = pd.Timestamp(self._frames[0].timestamp).floor('s')
        end_ts = pd.Timestamp(self._frames[-1].timestamp).ceil('s')
        title_str = f'{start_ts} to {end_ts}'

        # Main plot
        im = ax_main.pcolormesh(lon_grid, lat_grid, gridded,
                                cmap='viridis', vmin=vmin, vmax=vmax,
                                shading='flat')
        plt.colorbar(im, ax=ax_main, label='Intensity')

        # Ship track overlay
        if n_radials > 5000:
            step = n_radials // 5000
            track_lon, track_lat = ship_lon[::step], ship_lat[::step]
        else:
            track_lon, track_lat = ship_lon, ship_lat
        ax_main.plot(track_lon, track_lat, 'r-', linewidth=1.5)

        # Find valid data extent
        valid_mask = np.logical_and(np.logical_not(np.isnan(gridded)), gridded != 0)
        valid_rows, valid_cols = np.where(valid_mask)
        if len(valid_rows) > 0:
            lon_vals = lon_grid[valid_rows, valid_cols]
            lon_min, lon_max = lon_vals.min(), lon_vals.max()
            lat_vals = lat_grid[valid_rows, valid_cols]
            lat_min, lat_max = lat_vals.min(), lat_vals.max()
        else:
            lon_min, lon_max = lon_grid.min(), lon_grid.max()
            lat_min, lat_max = lat_grid.min(), lat_grid.max()

        # Aspect ratio and limits
        mean_lat = (lat_min + lat_max) / 2
        aspect_ratio = 1.0 / np.cos(np.deg2rad(mean_lat))
        data_lon_range = lon_max - lon_min
        data_lat_range = lat_max - lat_min

        fig_width, fig_height = fig.get_size_inches()
        plot_width = fig_width * (0.88 - 0.12)
        plot_height = fig_height * (0.88 - 0.12) * (12/13)
        data_width_dist = data_lon_range / aspect_ratio
        data_height_dist = data_lat_range
        plot_aspect = plot_width / plot_height
        data_aspect = data_width_dist / data_height_dist if data_height_dist > 0 else 1.0

        if plot_aspect > data_aspect:
            new_lon_range = data_lat_range * plot_aspect * aspect_ratio
            lon_center = (lon_min + lon_max) / 2
            lon_min, lon_max = lon_center - new_lon_range/2, lon_center + new_lon_range/2
        else:
            new_lat_range = data_width_dist / plot_aspect
            lat_center = (lat_min + lat_max) / 2
            lat_min, lat_max = lat_center - new_lat_range/2, lat_center + new_lat_range/2

        ax_main.set_xlim(lon_min, lon_max)
        ax_main.set_ylim(lat_min, lat_max)
        ax_main.set_xlabel('Longitude (°)')
        ax_main.set_ylabel('Latitude (°)')
        ax_main.set_title(title_str, fontsize=11)
        ax_main.set_aspect(aspect_ratio)
        ax_main.margins(0)
        ax_main.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Polar insets
        if ship_speeds and ship_headings:
            ax_ship = ax_main.inset_axes([0.88, 0.85, 0.11, 0.14], projection='polar')
            ax_ship.set_theta_zero_location('N')
            ax_ship.set_theta_direction(-1)
            ages = np.linspace(0, 1, len(ship_speeds))
            ax_ship.scatter(np.deg2rad(ship_headings), ship_speeds, c=ages,
                           cmap='viridis', s=6, alpha=0.8)
            ax_ship.set_xticklabels([])
            ax_ship.tick_params(labelsize=4)
            ax_ship.set_facecolor('white')
            ax_ship.patch.set_alpha(0.8)
            ax_ship.grid(True, linewidth=0.3, alpha=0.5)
            ax_main.text(0.87, 0.99, 'Ship', transform=ax_main.transAxes,
                        fontsize=7, ha='right', va='top')

        if wind_speeds and wind_dirs:
            ax_wind = ax_main.inset_axes([0.88, 0.01, 0.11, 0.14], projection='polar')
            ax_wind.set_theta_zero_location('N')
            ax_wind.set_theta_direction(-1)
            ages = np.linspace(0, 1, len(wind_speeds))
            ax_wind.scatter(np.deg2rad(wind_dirs), wind_speeds, c=ages,
                           cmap='viridis', s=6, alpha=0.8)
            ax_wind.set_xticklabels([])
            ax_wind.tick_params(labelsize=4)
            ax_wind.set_facecolor('white')
            ax_wind.patch.set_alpha(0.8)
            ax_wind.grid(True, linewidth=0.3, alpha=0.5)
            ax_main.text(0.87, 0.01, 'Wind', transform=ax_main.transAxes,
                        fontsize=7, ha='right', va='bottom')

        # Info panel
        ax_info.axis('off')
        travel = self.travel_distance()
        meta = self._frames[0].metadata
        info_text = (
            f"Ship: {travel['total_m']:.0f}m in {travel['duration_s']:.0f}s "
            f"({travel['speed_m_s']:.2f} m/s)    "
            f"Frames: {n_frames}"
        )
        ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                     ha='center', va='center', fontfamily='monospace', fontsize=9)

        # Save
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

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


def _process_and_save_group(args_tuple) -> str:
    """
    Process a single group and save its frame image.
    Used for parallel movie generation.

    Args:
        args_tuple: (group_idx, period, frames, config, radar_height, output_path, process_frames)

    Returns:
        Path to saved frame image
    """
    import matplotlib
    matplotlib.use('Agg')

    from wamos_tpw.processed import ProcessedFrames

    group_idx, period, frames, config, radar_height, output_path, process_frames = args_tuple

    # Process frames if requested
    if process_frames:
        pframes = ProcessedFrames.__new__(ProcessedFrames)
        pframes._config = config
        pframes._radar_height = radar_height
        pframes.refine_theta(frames)
        pframes.deramp_frames(frames)
        corrected = pframes.destreak_frames(frames)
        normalized = pframes.normalize_frames(corrected)
        for frame, corr in zip(frames, normalized):
            frame.corrected_intensity = corr

    # Create Combine and save frame
    combine = Combine(frames, config, radar_height=radar_height)
    combine.save_frame(output_path)

    return output_path


def add_subparser(subparsers) -> None:
    """Register the 'combine' subcommand."""
    p = subparsers.add_parser(
        'combine',
        help='Combine frames into earth coordinate image',
        description="Combine multiple radar frames into earth-referenced images with ship motion compensation"
    )
    p.add_argument("stime", type=str, help="Start time (YYYYMMDDHHMM or ISO format)")
    p.add_argument("etime", type=str, help="End time (YYYYMMDDHHMM or ISO format)")
    p.add_argument("polar_path", type=str, help="Path to POLAR files directory")
    p.add_argument("--groupby", "-g", type=str, default='h',
                   help="Groupby frequency (default: h)")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="YAML configuration file")
    p.add_argument("--radar-height", type=float, default=None,
                   help="Radar height above water (m)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximum frames to process per group")
    p.add_argument("--process", action="store_true",
                   help="Apply processing (deramp + destreak)")
    p.add_argument("--plot", action="store_true",
                   help="Show diagnostic plots")
    p.add_argument("--movie", type=str, default=None,
                   help="Output movie file (e.g., output.mp4)")
    p.add_argument("--fps", type=int, default=10,
                   help="Frames per second for movie (default: 10)")
    p.add_argument("--show-track", action="store_true",
                   help="Show separate ship track plot")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose output")
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'combine' command."""
    from wamos_tpw.processed import ProcessedFrames

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # Load config
    config = WamosConfig(args.config) if args.config else WamosConfig()

    with ProcessedFrames(
        stime=args.stime,
        etime=args.etime,
        polar_path=args.polar_path,
        groupby=args.groupby,
        config=config,
        radar_height=args.radar_height,
    ) as pframes:
        print(f"Discovered {len(pframes)} files")

        for period, frames in pframes.itergroups():
            frames = list(frames)
            if args.max_frames:
                frames = frames[:args.max_frames]
            print(f"\n{period}: {len(frames)} frames")

            if not frames:
                continue

            # Optionally process frames
            if args.process:
                print("  Processing: refine_theta, deramp, destreak...")
                corrected = pframes.process_group(frames)
                for frame, corr in zip(frames, corrected):
                    frame.corrected_intensity = corr

            # Create Combine object
            combine = Combine(frames, config, radar_height=args.radar_height)
            print(f"  {combine}")

            # Plot if requested
            if args.plot:
                combine.plot_diagnostics(show_track=args.show_track)

    # Movie generation mode
    if args.movie:
        import tempfile
        import subprocess
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"\nGenerating movie: {args.movie}")

        with ProcessedFrames(
            stime=args.stime,
            etime=args.etime,
            polar_path=args.polar_path,
            groupby=args.groupby,
            config=config,
            radar_height=args.radar_height,
        ) as pframes:
            # Collect all groups
            groups = []
            for period, frames in pframes.itergroups():
                frames = list(frames)
                if args.max_frames:
                    frames = frames[:args.max_frames]
                if frames:
                    groups.append((period, frames))

            if not groups:
                print("No frames to process")
                return

            print(f"Found {len(groups)} groups to render")

            # Create temp directory for frames
            with tempfile.TemporaryDirectory() as tmpdir:
                # Prepare arguments for parallel processing
                tasks = []
                for idx, (period, frames) in enumerate(groups):
                    output_path = f"{tmpdir}/frame_{idx:06d}.png"
                    tasks.append((
                        idx, period, frames, config, args.radar_height,
                        output_path, args.process
                    ))

                # Process groups in parallel
                n_workers = min(len(tasks), 8)
                print(f"Rendering {len(tasks)} frames with {n_workers} workers...")

                completed = 0
                frame_paths = [None] * len(tasks)
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(_process_and_save_group, task): task[0]
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            path = future.result()
                            frame_paths[idx] = path
                            completed += 1
                            print(f"  Rendered frame {completed}/{len(tasks)}", flush=True)
                        except Exception as e:
                            print(f"  Error rendering frame {idx}: {e}")

                # Filter out failed frames
                frame_paths = [p for p in frame_paths if p is not None]

                if not frame_paths:
                    print("No frames were rendered successfully")
                    return

                # Create movie with ffmpeg
                print(f"Creating MP4 with ffmpeg ({args.fps} fps)...")

                # Check if ffmpeg is available
                import shutil
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path is None:
                    print("Error: ffmpeg not found. Install with: brew install ffmpeg")
                    print("Frame images saved in temp directory (will be deleted)")
                    # Copy frames to output directory as fallback
                    import os
                    frames_dir = args.movie.replace('.mp4', '_frames')
                    os.makedirs(frames_dir, exist_ok=True)
                    for i, path in enumerate(frame_paths):
                        if path:
                            import shutil as sh
                            sh.copy(path, f"{frames_dir}/frame_{i:06d}.png")
                    print(f"Frames saved to: {frames_dir}/")
                    print("To create movie manually:")
                    print(f"  ffmpeg -framerate {args.fps} -i {frames_dir}/frame_%06d.png "
                          f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
                          f"-c:v libx264 -pix_fmt yuv420p {args.movie}")
                else:
                    # Use ffmpeg with H.264 codec for web compatibility
                    ffmpeg_cmd = [
                        ffmpeg_path, '-y',  # Overwrite output
                        '-framerate', str(args.fps),
                        '-i', f'{tmpdir}/frame_%06d.png',
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # H.264 needs even dimensions
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',  # Quality (lower = better, 18-28 typical)
                        '-pix_fmt', 'yuv420p',  # Web compatibility
                        '-movflags', '+faststart',  # Web streaming
                        args.movie
                    ]

                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"ffmpeg error: {result.stderr}")
                    else:
                        print(f"Movie saved to: {args.movie}")


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Combine frames into earth coordinate image")
    parser.add_argument("stime", type=str, help="Start time (YYYYMMDDHHMM or ISO format)")
    parser.add_argument("etime", type=str, help="End time (YYYYMMDDHHMM or ISO format)")
    parser.add_argument("polar_path", type=str, help="Path to POLAR files directory")
    parser.add_argument("--groupby", "-g", type=str, default='h',
                        help="Groupby frequency (default: h)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML configuration file")
    parser.add_argument("--radar-height", type=float, default=None,
                        help="Radar height above water (m)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process per group")
    parser.add_argument("--process", action="store_true",
                        help="Apply processing (deramp + destreak)")
    parser.add_argument("--plot", action="store_true",
                        help="Show diagnostic plots")
    parser.add_argument("--movie", type=str, default=None,
                        help="Output movie file (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for movie (default: 10)")
    parser.add_argument("--show-track", action="store_true",
                        help="Show separate ship track plot")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
