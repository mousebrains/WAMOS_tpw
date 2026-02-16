#! /usr/bin/env python3
#
# Earth projection for WAMOS radar data
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pyproj import CRS, Transformer

if TYPE_CHECKING:
    from wamos_tpw.frame_pipeline import FramePipeline
    from wamos_tpw.interpolator import FrameInterpolator

logger = logging.getLogger(__name__)

# Earth radius in meters (WGS84 mean radius)
EARTH_RADIUS = 6371000.0


def _circular_mean(angles: np.ndarray) -> float:
    """Calculate circular mean of angles in degrees."""
    angles_rad = np.deg2rad(angles)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360)


def get_utm_zone(longitude: float) -> int:
    """
    Calculate UTM zone number from longitude.

    Args:
        longitude: Longitude in degrees (-180 to 180)

    Returns:
        UTM zone number (1-60)
    """
    return int((longitude + 180) / 6) % 60 + 1


def get_utm_crs(latitude: float, longitude: float) -> CRS:
    """
    Get the appropriate UTM CRS for a given location.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees

    Returns:
        pyproj CRS object for the UTM zone
    """
    zone = get_utm_zone(longitude)
    hemisphere = "north" if latitude >= 0 else "south"
    return CRS.from_proj4(f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84")


@dataclass
class UTMGrid:
    """UTM-referenced Cartesian coordinate system for projected radar data."""

    # Grid definition
    x_edges: np.ndarray  # Easting bin edges in meters
    y_edges: np.ndarray  # Northing bin edges in meters
    grid_spacing: float  # Grid cell size in meters

    # Coordinate system
    utm_zone: int
    hemisphere: str  # 'north' or 'south'
    crs: CRS

    # Reference position (for display/metadata)
    ref_latitude: float = 0.0
    ref_longitude: float = 0.0

    # Projected data (sum and count for averaging)
    intensity_sum: np.ndarray = field(default=None)
    intensity_count: np.ndarray = field(default=None)

    @property
    def x_centers(self) -> np.ndarray:
        """Return x (easting) bin centers."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        """Return y (northing) bin centers."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def intensity(self) -> np.ndarray:
        """Return averaged projected intensity (NaN where no data)."""
        with np.errstate(invalid="ignore"):
            result = self.intensity_sum / self.intensity_count
        result[self.intensity_count == 0] = np.nan
        return result

    @property
    def n_x(self) -> int:
        """Return number of x bins."""
        return len(self.x_edges) - 1

    @property
    def n_y(self) -> int:
        """Return number of y bins."""
        return len(self.y_edges) - 1

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in meters."""
        return (self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1])

    @property
    def crs_wkt(self) -> str:
        """Return CRS as WKT string for embedding in files."""
        return self.crs.to_wkt()

    @property
    def crs_proj4(self) -> str:
        """Return CRS as PROJ4 string."""
        return self.crs.to_proj4()

    @property
    def epsg_code(self) -> int | None:
        """Return EPSG code if available."""
        try:
            return self.crs.to_epsg()
        except Exception:
            return None

    def to_latlon(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert UTM coordinates to lat/lon.

        Args:
            x: Easting values in meters
            y: Northing values in meters

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        crs_wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(self.crs, crs_wgs84, always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon

    def plot(
        self,
        title: str | None = None,
        ship_speed: float | None = None,
        ship_heading: float | None = None,
        wind_speed: float | None = None,
        wind_direction: float | None = None,
        timestamp: str | None = None,
        figsize: tuple[float, float] = (12, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot the projected intensity with coordinates as distance from center.

        Args:
            title: Optional custom title (replaces default)
            ship_speed: Ship speed in m/s (for title)
            ship_heading: Ship heading in degrees (for title)
            wind_speed: Wind speed in m/s (for title)
            wind_direction: Wind direction in degrees (for title)
            timestamp: Timestamp string (for title)
            figsize: Figure size as (width, height) in inches
            cmap: Colormap name
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt

        intensity = self.intensity

        # Auto-scale if not specified
        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        # Compute grid center in UTM coordinates
        x_center = (self.x_edges[0] + self.x_edges[-1]) / 2
        y_center = (self.y_edges[0] + self.y_edges[-1]) / 2

        # Translate edges to distance from center
        x_edges_centered = self.x_edges - x_center
        y_edges_centered = self.y_edges - y_center

        # Get center lat/lon
        center_lat, center_lon = self.to_latlon(np.array([x_center]), np.array([y_center]))
        center_lat = float(center_lat[0])
        center_lon = float(center_lon[0])

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.pcolormesh(
            x_edges_centered,
            y_edges_centered,
            intensity,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        ax.set_xlabel("Distance East (m)")
        ax.set_ylabel("Distance North (m)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        # Build title
        if title is None:
            title_lines = []

            # Location line
            title_lines.append(
                f"Center: {abs(center_lat):.4f}°{'N' if center_lat >= 0 else 'S'}, "
                f"{abs(center_lon):.4f}°{'E' if center_lon >= 0 else 'W'} "
                f"(UTM {self.utm_zone}{self.hemisphere[0].upper()})"
            )

            # Timestamp line
            if timestamp:
                title_lines.append(timestamp)

            # Ship and wind line
            nav_parts = []
            if ship_speed is not None and ship_heading is not None:
                nav_parts.append(f"Ship: {ship_speed:.1f} m/s @ {ship_heading:.0f}°")
            if wind_speed is not None and wind_direction is not None:
                nav_parts.append(f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}°")
            if nav_parts:
                title_lines.append(" | ".join(nav_parts))

            title = "\n".join(title_lines)

        ax.set_title(title)

        # Grid info in corner
        ax.text(
            0.02,
            0.98,
            f"Grid: {self.n_x}x{self.n_y}\n"
            f"Spacing: {self.grid_spacing:.1f}m\n"
            f"EPSG: {self.epsg_code}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"UTMGrid(zone={self.utm_zone}{self.hemisphere[0].upper()}, "
            f"grid={self.n_x}x{self.n_y}, spacing={self.grid_spacing:.1f}m)"
        )


def create_utm_grid(
    latitudes: list[np.ndarray],
    longitudes: list[np.ndarray],
    range_resolutions: list[float],
    max_ground_ranges: list[float],
    padding: float = 1.1,
) -> UTMGrid:
    """
    Create a UTM-referenced coordinate system for radar projection.

    The UTM zone is determined from the center longitude of the data.
    Grid spacing is set to the average range resolution.

    Args:
        latitudes: List of per-radial latitude arrays (one per frame)
        longitudes: List of per-radial longitude arrays (one per frame)
        range_resolutions: Range resolution per frame in meters
        max_ground_ranges: Maximum ground range per frame in meters
        padding: Multiplier for radar range to add margin (default 1.1 = 10% margin)

    Returns:
        UTMGrid object with coordinate system and grid defined
    """
    if not latitudes:
        raise ValueError("No spatial data available")

    # Calculate grid spacing from average range resolution
    grid_spacing = float(np.mean(range_resolutions))
    if grid_spacing <= 0:
        raise ValueError("Invalid range resolution")

    # Get reference position (center of all data)
    all_lats = np.concatenate(latitudes)
    all_lons = np.concatenate(longitudes)
    ref_lat = float(np.mean(all_lats))
    ref_lon = float(np.mean(all_lons))

    # Determine UTM zone and create CRS
    utm_zone = get_utm_zone(ref_lon)
    hemisphere = "north" if ref_lat >= 0 else "south"
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84")

    # Create transformer from WGS84 to UTM
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, utm_crs, always_xy=True)

    # Transform all positions to UTM
    all_x, all_y = transformer.transform(all_lons, all_lats)

    # Grid extent: data extent + max radar range + padding
    max_range = float(np.max(max_ground_ranges)) * padding

    x_min = all_x.min() - max_range
    x_max = all_x.max() + max_range
    y_min = all_y.min() - max_range
    y_max = all_y.max() + max_range

    # Create bin edges aligned to grid spacing
    n_x = int(np.ceil((x_max - x_min) / grid_spacing))
    n_y = int(np.ceil((y_max - y_min) / grid_spacing))

    x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
    y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)

    # Initialize accumulation arrays
    intensity_sum = np.zeros((n_y, n_x), dtype=np.float64)
    intensity_count = np.zeros((n_y, n_x), dtype=np.int32)

    grid = UTMGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        grid_spacing=grid_spacing,
        utm_zone=utm_zone,
        hemisphere=hemisphere,
        crs=utm_crs,
        ref_latitude=ref_lat,
        ref_longitude=ref_lon,
        intensity_sum=intensity_sum,
        intensity_count=intensity_count,
    )

    logger.debug(
        "Created UTM grid: zone=%d%s, %dx%d cells, %.2fm spacing",
        utm_zone,
        hemisphere[0].upper(),
        n_x,
        n_y,
        grid_spacing,
    )

    return grid


def project_to_utm(
    grid: UTMGrid,
    intensity: np.ndarray,
    theta: np.ndarray,
    ground_range: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings: np.ndarray,
) -> int:
    """
    Project a single frame onto the UTM grid.

    Args:
        grid: UTMGrid to project onto
        intensity: 2D intensity array (n_bearings, n_distances)
        theta: Beam angles relative to ship (degrees) for each bearing
        ground_range: Ground range (meters) for each distance bin
        latitudes: Per-radial latitudes
        longitudes: Per-radial longitudes
        headings: Per-radial ship headings (degrees)

    Returns:
        Number of points projected
    """
    # Create transformer from WGS84 to UTM
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_wgs84, grid.crs, always_xy=True)

    # Convert ship positions to UTM
    ship_x, ship_y = transformer.transform(longitudes, latitudes)

    # Pre-compute grid origin and spacing for direct index calculation
    x_origin = grid.x_edges[0]
    y_origin = grid.y_edges[0]
    inv_spacing = 1.0 / grid.grid_spacing

    # Grid dimensions
    n_x = grid.n_x
    n_y = grid.n_y

    # Compute earth bearing for each radial (theta is relative to ship, add heading)
    earth_bearing_rad = np.deg2rad((theta + headings) % 360)

    # Vectorized projection: compute x, y for all points
    sin_bearing = np.sin(earth_bearing_rad)
    cos_bearing = np.cos(earth_bearing_rad)

    # x, y offsets from ship position for each (bearing, range) point
    # Shape: (n_bearings, n_distances)
    x_coords = np.outer(sin_bearing, ground_range) + ship_x[:, np.newaxis]
    y_coords = np.outer(cos_bearing, ground_range) + ship_y[:, np.newaxis]

    # Convert to grid indices
    x_idx = ((x_coords - x_origin) * inv_spacing).astype(np.int64)
    y_idx = ((y_coords - y_origin) * inv_spacing).astype(np.int64)

    # Flatten arrays
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    values_flat = intensity.ravel()

    # Filter valid indices
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~np.isnan(values_flat)

    n_valid = np.sum(valid)
    if n_valid > 0:
        linear_idx = y_flat[valid] * n_x + x_flat[valid]
        grid_size = n_x * n_y

        # Accumulate using bincount
        batch_sum = np.bincount(linear_idx, weights=values_flat[valid], minlength=grid_size)
        batch_count = np.bincount(linear_idx, minlength=grid_size)

        grid.intensity_sum.ravel()[:] += batch_sum
        grid.intensity_count.ravel()[:] += batch_count

    return int(n_valid)


@dataclass
class EarthGrid:
    """Earth-referenced Cartesian coordinate system for projected radar data."""

    # Grid definition
    x_edges: np.ndarray  # East-west bin edges in meters
    y_edges: np.ndarray  # North-south bin edges in meters
    grid_spacing: float  # Grid cell size in meters

    # Projected data (sum and count for averaging)
    intensity_sum: np.ndarray = field(default=None)
    intensity_count: np.ndarray = field(default=None)

    # Reference position (origin)
    ref_latitude: float = 0.0
    ref_longitude: float = 0.0

    @property
    def x_centers(self) -> np.ndarray:
        """Return x (east) bin centers."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        """Return y (north) bin centers."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def intensity(self) -> np.ndarray:
        """Return averaged projected intensity (NaN where no data)."""
        with np.errstate(invalid="ignore"):
            result = self.intensity_sum / self.intensity_count
        result[self.intensity_count == 0] = np.nan
        return result

    @property
    def n_x(self) -> int:
        """Return number of x bins."""
        return len(self.x_edges) - 1

    @property
    def n_y(self) -> int:
        """Return number of y bins."""
        return len(self.y_edges) - 1

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in meters."""
        return (self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1])


@dataclass
class ProjectionResult:
    """
    Retained data after earth projection and memory cleanup.

    Contains only the essential data for analysis:
    - Earth coordinate system and projected intensities
    - Ship navigation data (speeds, headings)
    - Wind data (speeds, directions)
    - Frame timing information
    """

    # Earth grid with projected data
    earth_grid: EarthGrid

    # Ship navigation per frame
    ship_speeds: np.ndarray  # m/s per frame
    ship_headings: np.ndarray  # degrees per frame (mean heading during frame)

    # Wind data per frame
    wind_speeds: np.ndarray  # m/s per frame
    wind_directions: np.ndarray  # degrees per frame

    # Timing
    frame_start_times: np.ndarray  # datetime64 per frame
    frame_end_times: np.ndarray  # datetime64 per frame

    # Statistics
    n_frames: int = 0
    n_radials_total: int = 0


def create_earth_grid(
    latitudes: list[np.ndarray],
    longitudes: list[np.ndarray],
    range_resolutions: list[float],
    max_ground_ranges: list[float],
    padding: float = 1.1,
) -> EarthGrid:
    """
    Create an earth-referenced Cartesian coordinate system.

    The grid spacing is set to the average range resolution across all frames.
    The extent is determined by the ship track plus the maximum radar range.

    Args:
        latitudes: List of per-radial latitude arrays (one per frame)
        longitudes: List of per-radial longitude arrays (one per frame)
        range_resolutions: Range resolution per frame in meters
        max_ground_ranges: Maximum ground range per frame in meters
        padding: Multiplier for radar range to add margin (default 1.1 = 10% margin)

    Returns:
        EarthGrid object with x/y edges defined
    """
    if not latitudes:
        raise ValueError("No spatial data available")

    # Calculate grid spacing from average range resolution
    grid_spacing = float(np.mean(range_resolutions))
    if grid_spacing <= 0:
        raise ValueError("Invalid range resolution")

    # Get reference position (first radial of first frame)
    ref_lat = latitudes[0][0]
    ref_lon = longitudes[0][0]

    # Convert lat/lon to meters (flat earth approximation for local area)
    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    # Calculate ship track extent in meters
    all_lats = np.concatenate(latitudes)
    all_lons = np.concatenate(longitudes)

    ship_x = (all_lons - ref_lon) * meters_per_deg_lon
    ship_y = (all_lats - ref_lat) * meters_per_deg_lat

    # Grid extent: ship track + max radar range + padding
    max_range = float(np.max(max_ground_ranges)) * padding

    x_min = ship_x.min() - max_range
    x_max = ship_x.max() + max_range
    y_min = ship_y.min() - max_range
    y_max = ship_y.max() + max_range

    # Create bin edges aligned to grid spacing
    n_x = int(np.ceil((x_max - x_min) / grid_spacing))
    n_y = int(np.ceil((y_max - y_min) / grid_spacing))

    x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
    y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)

    # Initialize accumulation arrays
    intensity_sum = np.zeros((n_y, n_x), dtype=np.float64)
    intensity_count = np.zeros((n_y, n_x), dtype=np.int32)

    grid = EarthGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        grid_spacing=grid_spacing,
        intensity_sum=intensity_sum,
        intensity_count=intensity_count,
        ref_latitude=ref_lat,
        ref_longitude=ref_lon,
    )

    logger.debug(
        "Created earth grid: %dx%d cells, %.2fm spacing, extent: [%.1f, %.1f] x [%.1f, %.1f] m",
        n_x,
        n_y,
        grid_spacing,
        x_edges[0],
        x_edges[-1],
        y_edges[0],
        y_edges[-1],
    )

    return grid


def project_frame_to_grid(
    grid: EarthGrid,
    frame: FramePipeline,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    headings: np.ndarray,
) -> int:
    """
    Project a single frame onto the earth grid.

    Args:
        grid: EarthGrid to project onto
        frame: FramePipeline with intensity data
        latitudes: Per-radial latitudes for this frame
        longitudes: Per-radial longitudes for this frame
        headings: Per-radial headings for this frame

    Returns:
        Number of points projected
    """
    # Meters per degree for coordinate conversion
    meters_per_deg_lat = np.pi * EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(grid.ref_latitude))

    # Pre-compute grid origin and spacing for direct index calculation
    x_origin = grid.x_edges[0]
    y_origin = grid.y_edges[0]
    inv_spacing = 1.0 / grid.grid_spacing

    # Grid dimensions
    n_x = grid.n_x
    n_y = grid.n_y

    # Get frame data
    theta_ship = frame.theta_array  # Beam angles relative to ship (degrees)
    ground_range = frame.ground_range  # Distance per range bin (meters)
    intensity = frame.final_intensity  # Dewinded intensity

    # Convert ship positions to meters from reference
    ship_x = (longitudes - grid.ref_longitude) * meters_per_deg_lon
    ship_y = (latitudes - grid.ref_latitude) * meters_per_deg_lat

    # Compute earth bearing for each radial
    earth_bearing_rad = np.deg2rad((theta_ship + headings) % 360)

    # Vectorized projection
    sin_bearing = np.sin(earth_bearing_rad)
    cos_bearing = np.cos(earth_bearing_rad)

    # Compute x, y for all points
    x_coords = np.outer(sin_bearing, ground_range) + ship_x[:, np.newaxis]
    y_coords = np.outer(cos_bearing, ground_range) + ship_y[:, np.newaxis]

    # Convert to grid indices
    x_idx = ((x_coords - x_origin) * inv_spacing).astype(np.int64)
    y_idx = ((y_coords - y_origin) * inv_spacing).astype(np.int64)

    # Flatten arrays
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    values_flat = intensity.ravel()

    # Filter valid indices
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~np.isnan(values_flat)

    n_valid = np.sum(valid)
    if n_valid > 0:
        linear_idx = y_flat[valid] * n_x + x_flat[valid]
        grid_size = n_x * n_y

        # Accumulate using bincount
        batch_sum = np.bincount(linear_idx, weights=values_flat[valid], minlength=grid_size)
        batch_count = np.bincount(linear_idx, minlength=grid_size)

        grid.intensity_sum.ravel()[:] += batch_sum
        grid.intensity_count.ravel()[:] += batch_count

    return int(n_valid)


def project_frames_to_grid(
    grid: EarthGrid,
    frames: list[FramePipeline],
    latitudes: list[np.ndarray],
    longitudes: list[np.ndarray],
    headings: list[np.ndarray],
) -> int:
    """
    Project multiple frames onto the earth grid.

    Args:
        grid: EarthGrid to project onto
        frames: List of FramePipeline objects with intensity data
        latitudes: List of per-radial latitude arrays (one per frame)
        longitudes: List of per-radial longitude arrays (one per frame)
        headings: List of per-radial heading arrays (one per frame)

    Returns:
        Total number of points projected
    """
    n_projected = 0
    for i, frame in enumerate(frames):
        n_projected += project_frame_to_grid(grid, frame, latitudes[i], longitudes[i], headings[i])
    return n_projected


def create_projection_result(
    grid: EarthGrid,
    frames: list[FramePipeline],
    interpolators: list[FrameInterpolator],
    headings: list[np.ndarray],
) -> ProjectionResult:
    """
    Create a ProjectionResult from processed frames.

    Args:
        grid: EarthGrid with projected data
        frames: List of FramePipeline objects
        interpolators: List of FrameInterpolator objects
        headings: List of per-radial heading arrays

    Returns:
        ProjectionResult with metadata
    """
    n_frames = len(frames)

    # Collect ship navigation data per frame
    ship_speeds = np.array([f.metadata.ship_speed or 0.0 for f in frames])
    ship_headings = np.array([np.mean(headings[i]) for i in range(n_frames)])

    # Collect wind data per frame
    wind_speeds = np.array([f.metadata.wind_speed or 0.0 for f in frames])
    wind_directions = np.array([f.metadata.wind_direction or 0.0 for f in frames])

    # Collect timing data
    frame_start_times = np.array(
        [interp.times[0] for interp in interpolators], dtype="datetime64[ns]"
    )
    frame_end_times = np.array(
        [interp.times[-1] for interp in interpolators], dtype="datetime64[ns]"
    )

    # Count total radials
    n_radials_total = sum(len(headings[i]) for i in range(n_frames))

    return ProjectionResult(
        earth_grid=grid,
        ship_speeds=ship_speeds,
        ship_headings=ship_headings,
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        frame_start_times=frame_start_times,
        frame_end_times=frame_end_times,
        n_frames=n_frames,
        n_radials_total=n_radials_total,
    )


class ProjectionDiagnostics:
    """
    Diagnostic visualization for earth projection results.

    Provides plotting for:
    - Projected intensity on earth grid
    - Ship speed and heading polar plots
    - Wind speed and direction polar plots
    - Time series of navigation and environmental data
    """

    def __init__(self, result: ProjectionResult) -> None:
        """
        Initialize diagnostics viewer.

        Args:
            result: ProjectionResult from projection functions
        """
        self._result = result
        self._grid = result.earth_grid

    @property
    def result(self) -> ProjectionResult:
        """Return the projection result."""
        return self._result

    def plot(
        self,
        figsize: tuple[float, float] = (16, 12),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Create comprehensive diagnostic plot.

        Shows:
        - Main: Projected intensity on earth grid
        - Top-right: Ship speed/heading polar plot
        - Bottom-right: Wind speed/direction polar plot

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plot
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        result = self._result
        grid = self._grid

        # Get intensity data
        intensity = grid.intensity

        # Auto-scale if not specified
        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

        # Main intensity plot (spans left 2 columns)
        ax_main = fig.add_subplot(gs[:, 0])
        self._plot_intensity(ax_main, intensity, cmap, vmin, vmax)

        # Ship speed/heading polar plot (top-right)
        ax_ship = fig.add_subplot(gs[0, 1], projection="polar")
        self._plot_ship_polar(ax_ship)

        # Wind speed/direction polar plot (bottom-right)
        ax_wind = fig.add_subplot(gs[1, 1], projection="polar")
        self._plot_wind_polar(ax_wind)

        # Time series plot (right column, spans both rows)
        ax_time = fig.add_subplot(gs[:, 2])
        self._plot_time_series(ax_time)

        # Title
        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        fig.suptitle(
            f"Earth Projection: {result.n_frames} frames, {result.n_radials_total} radials\n"
            f"{start_time} to {end_time}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()

    def _plot_intensity(
        self,
        ax,
        intensity: np.ndarray,
        cmap: str,
        vmin: float,
        vmax: float,
    ) -> None:
        """Plot projected intensity on earth grid."""
        grid = self._grid

        im = ax.pcolormesh(
            grid.x_edges, grid.y_edges, intensity, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Projected Intensity")
        ax.set_aspect("equal")

        # Add colorbar
        ax.figure.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        # Add grid info
        ax.text(
            0.02,
            0.98,
            f"Grid: {grid.n_x}x{grid.n_y}\nSpacing: {grid.grid_spacing:.1f}m",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def _plot_ship_polar(self, ax) -> None:
        """
        Plot ship speed and heading as a polar scatter plot.

        Heading is the angle (0=North, clockwise), speed is the radius.
        """
        result = self._result

        # Convert headings to radians (matplotlib polar: 0=East, counter-clockwise)
        # We want 0=North, clockwise, so: theta = 90 - heading (in degrees) then to radians
        headings_rad = np.deg2rad(90 - result.ship_headings)

        # Speed is the radius
        speeds = result.ship_speeds

        # Color by frame index for temporal information
        colors = np.arange(len(speeds))

        ax.scatter(headings_rad, speeds, c=colors, cmap="viridis", s=20, alpha=0.7)

        # Configure polar plot
        ax.set_theta_zero_location("N")  # 0 degrees at top
        ax.set_theta_direction(-1)  # Clockwise

        ax.set_title("Ship Speed/Heading", pad=10)
        ax.set_xlabel("")

        # Add statistics
        mean_speed = np.mean(speeds)
        mean_heading = _circular_mean(result.ship_headings)
        ax.text(
            0.5,
            -0.15,
            f"Mean: {mean_speed:.1f} m/s @ {mean_heading:.0f}\u00b0",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    def _plot_wind_polar(self, ax) -> None:
        """
        Plot wind speed and direction as a polar scatter plot.

        Direction is the angle (0=North, clockwise - direction wind comes FROM),
        speed is the radius.
        """
        result = self._result

        # Convert directions to radians
        directions_rad = np.deg2rad(90 - result.wind_directions)

        # Speed is the radius
        speeds = result.wind_speeds

        # Color by frame index
        colors = np.arange(len(speeds))

        ax.scatter(directions_rad, speeds, c=colors, cmap="coolwarm", s=20, alpha=0.7)

        # Configure polar plot
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.set_title("Wind Speed/Direction", pad=10)

        # Add statistics
        mean_speed = np.mean(speeds)
        mean_dir = _circular_mean(result.wind_directions)
        ax.text(
            0.5,
            -0.15,
            f"Mean: {mean_speed:.1f} m/s from {mean_dir:.0f}\u00b0",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    def _plot_time_series(self, ax) -> None:
        """Plot time series of ship and wind data."""
        result = self._result

        # Convert times to relative seconds from start
        t0 = result.frame_start_times[0]
        times = (result.frame_start_times - t0) / np.timedelta64(1, "s")

        # Create twin axis for speeds
        ax2 = ax.twinx()

        # Plot headings on primary axis
        ax.plot(times, result.ship_headings, "b-", label="Ship heading", alpha=0.7)
        ax.plot(times, result.wind_directions, "r-", label="Wind direction", alpha=0.7)
        ax.set_ylabel("Direction (degrees)")
        ax.set_ylim(0, 360)
        ax.legend(loc="upper left", fontsize=8)

        # Plot speeds on secondary axis
        ax2.plot(times, result.ship_speeds, "b--", label="Ship speed", alpha=0.7)
        ax2.plot(times, result.wind_speeds, "r--", label="Wind speed", alpha=0.7)
        ax2.set_ylabel("Speed (m/s)")
        ax2.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("Time (s)")
        ax.set_title("Time Series")
        ax.grid(True, alpha=0.3)

    def plot_intensity_only(
        self,
        figsize: tuple[float, float] = (12, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot only the projected intensity.

        Args:
            figsize: Figure size
            cmap: Colormap name
            vmin: Minimum intensity value
            vmax: Maximum intensity value
        """
        import matplotlib.pyplot as plt

        grid = self._grid
        result = self._result
        intensity = grid.intensity

        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.pcolormesh(
            grid.x_edges, grid.y_edges, intensity, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")

        fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        ax.set_title(f"Projected Intensity\n{result.n_frames} frames, {start_time} to {end_time}")

        # Add statistics
        stats_text = (
            f"Grid: {grid.n_x}x{grid.n_y} ({grid.grid_spacing:.1f}m)\n"
            f"Extent: [{grid.x_edges[0]:.0f}, {grid.x_edges[-1]:.0f}] x "
            f"[{grid.y_edges[0]:.0f}, {grid.y_edges[-1]:.0f}] m"
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

    def save_intensity(
        self,
        output_path: str,
        figsize: tuple[float, float] = (12, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        dpi: int = 150,
    ) -> None:
        """
        Save projected intensity plot to file.

        Args:
            output_path: Path to save image (e.g., 'projection.png')
            figsize: Figure size
            cmap: Colormap name
            vmin: Minimum intensity value
            vmax: Maximum intensity value
            dpi: Image resolution
        """
        import matplotlib.pyplot as plt

        grid = self._grid
        result = self._result
        intensity = grid.intensity

        if vmin is None:
            vmin = np.nanpercentile(intensity, 2)
        if vmax is None:
            vmax = np.nanpercentile(intensity, 98)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.pcolormesh(
            grid.x_edges, grid.y_edges, intensity, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
        )

        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")

        fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        ax.set_title(f"Projected Intensity: {result.n_frames} frames\n{start_time} to {end_time}")

        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved intensity plot to %s", output_path)

    def __repr__(self) -> str:
        return (
            f"ProjectionDiagnostics(frames={self._result.n_frames}, "
            f"grid={self._grid.n_x}x{self._grid.n_y}, "
            f"spacing={self._grid.grid_spacing:.1f}m)"
        )


class ProjectionViewer:
    """
    Interactive viewer for multiple earth projection results.

    Provides navigation buttons (Back, Play/Pause, Next) and keyboard shortcuts
    to walk through a sequence of ProjectionResult objects.

    Keyboard shortcuts:
    - Left arrow: Previous result
    - Right arrow: Next result
    - Space: Play/Pause
    - q/Escape: Close viewer
    """

    def __init__(
        self,
        results: list[ProjectionResult],
        labels: list[str] | None = None,
    ) -> None:
        """
        Initialize the projection viewer.

        Args:
            results: List of ProjectionResult objects to display
            labels: Optional labels for each result (e.g., group periods)
        """
        if not results:
            raise ValueError("No results to display")

        self._results = results
        self._labels = labels or [f"Result {i + 1}" for i in range(len(results))]
        self._current_idx = 0
        self._playing = False
        self._play_interval = 1000  # milliseconds
        self._timer = None

        # Plot settings (set during show())
        self._cmap = "viridis"
        self._vmin: float | None = None
        self._vmax: float | None = None

        # Figure and axes (created during show())
        self._fig = None
        self._ax_main = None
        self._ax_ship = None
        self._ax_wind = None
        self._ax_time = None
        self._ax_time_twin = None  # Track twin axis for proper clearing
        self._colorbar = None
        self._im = None

    @property
    def n_results(self) -> int:
        """Return the number of results."""
        return len(self._results)

    @property
    def current_index(self) -> int:
        """Return the current result index."""
        return self._current_idx

    @property
    def current_result(self) -> ProjectionResult:
        """Return the current result."""
        return self._results[self._current_idx]

    @property
    def current_label(self) -> str:
        """Return the current label."""
        return self._labels[self._current_idx]

    def show(
        self,
        figsize: tuple[float, float] = (16, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        play_interval: int = 1000,
    ) -> None:
        """
        Display the interactive viewer.

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plot
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
            play_interval: Interval between frames in play mode (milliseconds)
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import Button

        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax
        self._play_interval = play_interval

        # Compute global vmin/vmax across all results if not specified
        if self._vmin is None or self._vmax is None:
            all_intensities = []
            for r in self._results:
                intensity = r.earth_grid.intensity
                all_intensities.append(intensity[~np.isnan(intensity)])
            combined = np.concatenate(all_intensities)
            if self._vmin is None:
                self._vmin = np.percentile(combined, 2)
            if self._vmax is None:
                self._vmax = np.percentile(combined, 98)

        # Create figure with custom layout: main plot (left), time series (right)
        self._fig = plt.figure(figsize=figsize)

        gs_plots = GridSpec(
            1,
            2,
            width_ratios=[2, 1],
            wspace=0.15,
            top=0.92,
            bottom=0.12,
            left=0.06,
            right=0.98,
        )

        # Main intensity plot
        self._ax_main = self._fig.add_subplot(gs_plots[0, 0])

        # Time series plot (right side)
        self._ax_time = self._fig.add_subplot(gs_plots[0, 1])

        # Polar insets will be created in _update_plot as they need proper positioning

        # Create navigation buttons
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        center_x = 0.5

        ax_back = self._fig.add_axes([center_x - btn_width * 1.6, btn_y, btn_width, btn_height])
        ax_play = self._fig.add_axes([center_x - btn_width * 0.5, btn_y, btn_width, btn_height])
        ax_next = self._fig.add_axes([center_x + btn_width * 0.6, btn_y, btn_width, btn_height])

        self._btn_back = Button(ax_back, "< Back")
        self._btn_play = Button(ax_play, "Play")
        self._btn_next = Button(ax_next, "Next >")

        self._btn_back.on_clicked(self._on_back)
        self._btn_play.on_clicked(self._on_play)
        self._btn_next.on_clicked(self._on_next)

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("close_event", self._on_close)

        # Initial plot
        self._update_plot()

        plt.show()

    def _update_plot(self) -> None:
        """Update all plots for the current result."""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        result = self.current_result
        grid = result.earth_grid
        intensity = grid.intensity

        # Clear main axis
        self._ax_main.clear()

        # Clear and remove old polar inset axes if they exist
        if self._ax_ship is not None:
            self._ax_ship.remove()
            self._ax_ship = None
        if self._ax_wind is not None:
            self._ax_wind.remove()
            self._ax_wind = None

        # Clear time series and twin axis
        self._ax_time.clear()
        if self._ax_time_twin is not None:
            self._ax_time_twin.remove()
            self._ax_time_twin = None

        # Main intensity plot
        self._im = self._ax_main.pcolormesh(
            grid.x_edges,
            grid.y_edges,
            intensity,
            cmap=self._cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            shading="flat",
        )

        self._ax_main.set_xlabel("East (m)")
        self._ax_main.set_ylabel("North (m)")
        self._ax_main.set_title("Dewinded Intensity")
        self._ax_main.set_aspect("equal")

        # Add colorbar (only on first update, then reuse)
        if self._colorbar is None:
            self._colorbar = self._fig.colorbar(
                self._im, ax=self._ax_main, label="Intensity", shrink=0.8
            )
        else:
            self._colorbar.update_normal(self._im)

        # Grid info in SW corner
        self._ax_main.text(
            0.02,
            0.02,
            f"Grid: {grid.n_x}x{grid.n_y}\nSpacing: {grid.grid_spacing:.1f}m",
            transform=self._ax_main.transAxes,
            verticalalignment="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Create small polar inset for ship in NE corner (top-right of main plot)
        self._ax_ship = inset_axes(
            self._ax_main,
            width="20%",
            height="20%",
            loc="upper right",
            borderpad=1.0,
            axes_class=None,
        )
        # Convert to polar projection by replacing the inset
        pos = self._ax_ship.get_position()
        self._ax_ship.remove()
        self._ax_ship = self._fig.add_axes(pos, projection="polar")
        self._plot_ship_polar(self._ax_ship, result)

        # Create small polar inset for wind in SE corner (bottom-right of main plot)
        self._ax_wind = inset_axes(
            self._ax_main,
            width="20%",
            height="20%",
            loc="lower right",
            borderpad=1.0,
            axes_class=None,
        )
        # Convert to polar projection
        pos = self._ax_wind.get_position()
        self._ax_wind.remove()
        self._ax_wind = self._fig.add_axes(pos, projection="polar")
        self._plot_wind_polar(self._ax_wind, result)

        # Time series with twin axis
        self._plot_time_series(self._ax_time, result)

        # Title with navigation info
        start_time = np.datetime_as_string(result.frame_start_times[0], unit="s")
        end_time = np.datetime_as_string(result.frame_end_times[-1], unit="s")
        play_status = " [Playing]" if self._playing else ""
        self._fig.suptitle(
            f"{self.current_label} ({self._current_idx + 1}/{self.n_results}){play_status}\n"
            f"{result.n_frames} frames, {result.n_radials_total} radials | "
            f"{start_time} to {end_time}",
            fontsize=12,
        )

        self._fig.canvas.draw_idle()

    def _plot_ship_polar(self, ax, result: ProjectionResult) -> None:
        """Plot ship speed and heading as a polar scatter plot (compact for inset)."""
        headings_rad = np.deg2rad(90 - result.ship_headings)
        speeds = result.ship_speeds
        colors = np.arange(len(speeds))

        ax.scatter(headings_rad, speeds, c=colors, cmap="viridis", s=10, alpha=0.7)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Compact title inside the plot
        mean_speed = np.mean(speeds)
        mean_heading = _circular_mean(result.ship_headings)
        ax.set_title(f"Ship\n{mean_speed:.1f}m/s @ {mean_heading:.0f}\u00b0", fontsize=7, pad=2)

        # Make tick labels smaller
        ax.tick_params(axis="both", labelsize=6)
        # Reduce radial tick labels
        ax.set_yticklabels([])

    def _plot_wind_polar(self, ax, result: ProjectionResult) -> None:
        """Plot wind speed and direction as a polar scatter plot (compact for inset)."""
        directions_rad = np.deg2rad(90 - result.wind_directions)
        speeds = result.wind_speeds
        colors = np.arange(len(speeds))

        ax.scatter(directions_rad, speeds, c=colors, cmap="coolwarm", s=10, alpha=0.7)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Compact title inside the plot
        mean_speed = np.mean(speeds)
        mean_dir = _circular_mean(result.wind_directions)
        ax.set_title(f"Wind\n{mean_speed:.1f}m/s from {mean_dir:.0f}\u00b0", fontsize=7, pad=2)

        # Make tick labels smaller
        ax.tick_params(axis="both", labelsize=6)
        # Reduce radial tick labels
        ax.set_yticklabels([])

    def _plot_time_series(self, ax, result: ProjectionResult) -> None:
        """Plot time series of ship and wind data."""
        t0 = result.frame_start_times[0]
        times = (result.frame_start_times - t0) / np.timedelta64(1, "s")

        # Create twin axis and store reference for proper clearing
        self._ax_time_twin = ax.twinx()

        ax.plot(times, result.ship_headings, "b-", label="Ship heading", alpha=0.7)
        ax.plot(times, result.wind_directions, "r-", label="Wind direction", alpha=0.7)
        ax.set_ylabel("Direction (degrees)")
        ax.set_ylim(0, 360)
        ax.legend(loc="upper left", fontsize=8)

        self._ax_time_twin.plot(times, result.ship_speeds, "b--", label="Ship speed", alpha=0.7)
        self._ax_time_twin.plot(times, result.wind_speeds, "r--", label="Wind speed", alpha=0.7)
        self._ax_time_twin.set_ylabel("Speed (m/s)")
        self._ax_time_twin.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("Time (s)")
        ax.set_title("Time Series")
        ax.grid(True, alpha=0.3)

    def _on_back(self, event) -> None:
        """Handle back button click."""
        self._go_back()

    def _on_next(self, event) -> None:
        """Handle next button click."""
        self._go_next()

    def _on_play(self, event) -> None:
        """Handle play/pause button click."""
        self._toggle_play()

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == "left":
            self._go_back()
        elif event.key == "right":
            self._go_next()
        elif event.key == " ":
            self._toggle_play()
        elif event.key in ("q", "escape"):
            self._stop_play()
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _on_close(self, event) -> None:
        """Handle window close event."""
        self._stop_play()

    def _go_back(self) -> None:
        """Go to the previous result."""
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_plot()

    def _go_next(self) -> None:
        """Go to the next result."""
        if self._current_idx < self.n_results - 1:
            self._current_idx += 1
            self._update_plot()

    def _toggle_play(self) -> None:
        """Toggle play/pause mode."""
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        """Start auto-play mode."""
        if self._playing:
            return

        self._playing = True
        self._btn_play.label.set_text("Pause")

        # Use matplotlib timer for animation
        self._timer = self._fig.canvas.new_timer(interval=self._play_interval)
        self._timer.add_callback(self._play_step)
        self._timer.start()

        self._update_plot()

    def _stop_play(self) -> None:
        """Stop auto-play mode."""
        if not self._playing:
            return

        self._playing = False
        self._btn_play.label.set_text("Play")

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        self._update_plot()

    def _play_step(self) -> None:
        """Advance to next frame during play mode."""
        if not self._playing:
            return

        if self._current_idx < self.n_results - 1:
            self._current_idx += 1
            self._update_plot()
        else:
            # Loop back to start
            self._current_idx = 0
            self._update_plot()

    def __repr__(self) -> str:
        return (
            f"ProjectionViewer(n_results={self.n_results}, "
            f"current={self._current_idx + 1}/{self.n_results})"
        )


class InterpolatorViewer:
    """
    Interactive streaming viewer for interpolated frame projections.

    Displays results as they arrive, showing the first frame immediately.
    Provides navigation buttons (Back, Play/Pause, Next) and keyboard shortcuts.

    Results must include pre-projected data:
    - projected_intensity: 2D numpy array
    - grid_params: dict with x_edges, y_edges, center_lat, center_lon, etc.

    Keyboard shortcuts:
    - Left arrow: Previous frame
    - Right arrow: Next frame
    - Space: Play/Pause
    - q/Escape: Close viewer
    """

    def __init__(self) -> None:
        """Initialize an empty streaming viewer."""
        self._results: list[dict] = []
        self._current_idx = 0
        self._playing = False
        self._play_interval = 500  # milliseconds
        self._timer = None
        self._loading_complete = False
        self._expected_total: int | None = None

        # Plot settings
        self._cmap = "viridis"
        self._vmin: float | None = None
        self._vmax: float | None = None
        self._auto_scale = True

        # Figure and axes (created during show())
        self._fig = None
        self._ax_main = None
        self._colorbar = None
        self._im = None
        self._shown = False

    def add_result(self, result: dict) -> None:
        """
        Add an interpolation result with pre-projected data.

        If the viewer is shown, updates the display. Shows first frame immediately.

        Args:
            result: Dict with projected_intensity, grid_params, timestamp, etc.
        """
        if result.get("projected_intensity") is None:
            return  # Skip results without projection

        self._results.append(result)

        # Update auto-scaling with new data
        if self._auto_scale and self._vmin is None:
            intensity = result["projected_intensity"]
            valid = intensity[~np.isnan(intensity)]
            if len(valid) > 0:
                self._vmin = np.percentile(valid, 2)
                self._vmax = np.percentile(valid, 98)

        # If shown and this is first result, display it
        if self._shown and len(self._results) == 1:
            self._update_plot()

    def set_expected_total(self, total: int) -> None:
        """Set the expected total number of frames (for progress display)."""
        self._expected_total = total

    def mark_loading_complete(self) -> None:
        """Mark that all results have been added."""
        self._loading_complete = True
        if self._shown:
            self._update_plot()

    @property
    def n_frames(self) -> int:
        """Return the number of frames loaded so far."""
        return len(self._results)

    @property
    def current_index(self) -> int:
        """Return the current frame index."""
        return self._current_idx

    @property
    def current_result(self) -> dict | None:
        """Return the current interpolation result."""
        if not self._results or self._current_idx >= len(self._results):
            return None
        return self._results[self._current_idx]

    def show(
        self,
        figsize: tuple[float, float] = (10, 10),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        play_interval: int = 500,
    ) -> None:
        """
        Display the interactive viewer (non-blocking).

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plot
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
            play_interval: Interval between frames in play mode (milliseconds)
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._cmap = cmap
        self._play_interval = play_interval

        if vmin is not None:
            self._vmin = vmin
            self._auto_scale = False
        if vmax is not None:
            self._vmax = vmax
            self._auto_scale = False

        # Enable interactive mode for non-blocking display
        plt.ion()

        # Create figure with manually-positioned square axes so the colorbar
        # cannot steal space from the main plot.
        self._fig = plt.figure(figsize=figsize)
        #                       left  bot   w     h
        self._ax_main = self._fig.add_axes([0.07, 0.12, 0.78, 0.78])
        self._cax = self._fig.add_axes([0.87, 0.12, 0.025, 0.78])

        # Create navigation buttons
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        center_x = 0.5

        ax_back = self._fig.add_axes([center_x - btn_width * 1.6, btn_y, btn_width, btn_height])
        ax_play = self._fig.add_axes([center_x - btn_width * 0.5, btn_y, btn_width, btn_height])
        ax_next = self._fig.add_axes([center_x + btn_width * 0.6, btn_y, btn_width, btn_height])

        self._btn_back = Button(ax_back, "< Back")
        self._btn_play = Button(ax_play, "Play")
        self._btn_next = Button(ax_next, "Next >")

        self._btn_back.on_clicked(self._on_back)
        self._btn_play.on_clicked(self._on_play)
        self._btn_next.on_clicked(self._on_next)

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._fig.canvas.mpl_connect("close_event", self._on_close)

        self._shown = True

        # Initial plot (might show "loading" if no results yet)
        self._update_plot()

        # Show figure but don't block
        self._fig.show()
        self._fig.canvas.flush_events()

    def wait(self) -> None:
        """Block until the viewer is closed (call after loading is complete)."""
        import matplotlib.pyplot as plt

        plt.ioff()  # Disable interactive mode
        plt.show()  # This blocks

    def _update_plot(self) -> None:
        """Update the plot for the current frame."""
        import matplotlib.pyplot as plt

        self._ax_main.clear()

        result = self.current_result

        if result is None:
            # No data yet - show loading message
            loading_msg = "Loading first frame..."
            if self._expected_total:
                loading_msg = f"Loading... (0/{self._expected_total})"
            self._ax_main.text(
                0.5,
                0.5,
                loading_msg,
                transform=self._ax_main.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            return

        intensity = result["projected_intensity"]
        params = result["grid_params"]

        # Plot intensity with centered coordinates
        self._im = self._ax_main.pcolormesh(
            params["x_edges"],
            params["y_edges"],
            intensity,
            cmap=self._cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            shading="flat",
        )

        self._ax_main.set_xlabel("Distance East (m)")
        self._ax_main.set_ylabel("Distance North (m)")
        self._ax_main.set_aspect("equal", adjustable="datalim")

        # Add range rings every 1km
        max_extent = max(
            abs(params["x_edges"][0]),
            abs(params["x_edges"][-1]),
            abs(params["y_edges"][0]),
            abs(params["y_edges"][-1]),
        )
        for r in range(1000, int(max_extent) + 1000, 1000):
            circle = plt.Circle((0, 0), r, fill=False, color="white", linewidth=0.5, alpha=0.5)
            self._ax_main.add_patch(circle)
            # Label at 45 degrees
            label_x = r * 0.707
            label_y = r * 0.707
            if label_x < max_extent * 0.9 and label_y < max_extent * 0.9:
                self._ax_main.text(
                    label_x,
                    label_y,
                    f"{r // 1000}km",
                    color="white",
                    fontsize=8,
                    alpha=0.7,
                    ha="center",
                    va="center",
                )

        # Add colorbar in dedicated axes (doesn't steal from main plot)
        if self._colorbar is None:
            self._colorbar = self._fig.colorbar(self._im, cax=self._cax, label="Intensity")
        else:
            self._colorbar.update_normal(self._im)

        # Build title
        title_lines = []

        # Location line
        center_lat = params["center_lat"]
        center_lon = params["center_lon"]
        title_lines.append(
            f"Center: {abs(center_lat):.4f}\u00b0{'N' if center_lat >= 0 else 'S'}, "
            f"{abs(center_lon):.4f}\u00b0{'E' if center_lon >= 0 else 'W'} "
            f"(UTM {params['utm_zone']}{params['hemisphere'][0].upper()})"
        )

        # Timestamp
        timestamp = result.get("timestamp")
        if timestamp is not None:
            title_lines.append(np.datetime_as_string(timestamp, unit="ms"))

        # Ship and wind info
        nav_parts = []
        ship_speed = result.get("ship_speed")
        ship_heading = float(np.mean(result["headings"])) if "headings" in result else None
        if ship_speed is not None and ship_heading is not None:
            nav_parts.append(f"Ship: {ship_speed:.1f} m/s @ {ship_heading:.0f}\u00b0")
        elif ship_heading is not None:
            nav_parts.append(f"Ship heading: {ship_heading:.0f}\u00b0")

        wind_speed = result.get("wind_speed")
        wind_direction = result.get("wind_direction")
        if wind_speed is not None and wind_direction is not None:
            nav_parts.append(f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}\u00b0")

        if nav_parts:
            title_lines.append(" | ".join(nav_parts))

        self._ax_main.set_title("\n".join(title_lines))

        # Frame counter with loading status
        total_str = str(self._expected_total) if self._expected_total else "?"
        if self._loading_complete:
            total_str = str(self.n_frames)
        loading_indicator = "" if self._loading_complete else " [Loading...]"

        self._ax_main.text(
            0.02,
            0.98,
            f"Frame {self._current_idx + 1}/{total_str}{loading_indicator}\n"
            f"Grid: {params['n_x']}x{params['n_y']}\n"
            f"Spacing: {params['grid_spacing']:.1f}m",
            transform=self._ax_main.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Play status
        if self._playing:
            self._ax_main.text(
                0.98,
                0.98,
                "[Playing]",
                transform=self._ax_main.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color="green",
            )

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _on_back(self, event) -> None:
        """Handle back button click."""
        self._go_back()

    def _on_next(self, event) -> None:
        """Handle next button click."""
        self._go_next()

    def _on_play(self, event) -> None:
        """Handle play/pause button click."""
        self._toggle_play()

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == "left":
            self._go_back()
        elif event.key == "right":
            self._go_next()
        elif event.key == " ":
            self._toggle_play()
        elif event.key in ("q", "escape"):
            self._stop_play()
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _on_close(self, event) -> None:
        """Handle window close event."""
        self._stop_play()

    def _go_back(self) -> None:
        """Go to the previous frame."""
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_plot()

    def _go_next(self) -> None:
        """Go to the next frame."""
        if self._current_idx < self.n_frames - 1:
            self._current_idx += 1
            self._update_plot()

    def _toggle_play(self) -> None:
        """Toggle play/pause mode."""
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        """Start auto-play mode."""
        if self._playing:
            return

        self._playing = True
        self._btn_play.label.set_text("Pause")

        self._timer = self._fig.canvas.new_timer(interval=self._play_interval)
        self._timer.add_callback(self._play_step)
        self._timer.start()

        self._update_plot()

    def _stop_play(self) -> None:
        """Stop auto-play mode."""
        if not self._playing:
            return

        self._playing = False
        self._btn_play.label.set_text("Play")

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        self._update_plot()

    def _play_step(self) -> None:
        """Advance to next frame during play mode."""
        if not self._playing:
            return

        if self._current_idx < self.n_frames - 1:
            self._current_idx += 1
            self._update_plot()
        else:
            # Loop back to start
            self._current_idx = 0
            self._update_plot()

    def __repr__(self) -> str:
        status = "loading" if not self._loading_complete else "complete"
        return (
            f"InterpolatorViewer(n_frames={self.n_frames}, "
            f"current={self._current_idx + 1}, status={status})"
        )
