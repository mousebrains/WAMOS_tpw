#! /usr/bin/env python3
#
# Plotting functions for WAMOS combined data
# Includes interactive viewer and frame saving
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.combine import Combine


# Earth radius in meters (WGS84 mean radius)
_EARTH_RADIUS = 6371000.0


def grid_group(period: str, combine: 'Combine', n_along: int = 1200, n_cross: int = 1600) -> dict:
    """
    Grid a single group's frames.

    Args:
        period: Time period identifier string
        combine: Combine object with frames
        n_along: Grid bins along ship track
        n_cross: Grid bins cross track

    Returns:
        Dictionary with gridded data and metadata
    """
    # Grid the frames
    x_edges, y_edges, gridded, angle = combine.grid_parallel_rotated(
        n_along=n_along, n_cross=n_cross
    )

    # Rotate grid back to earth coordinates
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    xx_rot, yy_rot = np.meshgrid(x_edges, y_edges)
    xx_earth = xx_rot * cos_a - yy_rot * sin_a
    yy_earth = xx_rot * sin_a + yy_rot * cos_a

    # Convert to lat/lon
    ref_lat, ref_lon = combine.reference_position
    meters_per_deg_lat = np.pi * _EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))
    lon_grid = ref_lon + xx_earth / meters_per_deg_lon
    lat_grid = ref_lat + yy_earth / meters_per_deg_lat

    # Get ship track
    ship_lat, ship_lon = combine.ship_track()
    travel = combine.travel_distance()

    # Count pixels
    n_pixels = sum(f.n_bearings * f.n_distances for f in combine.frames)

    return {
        'period': str(period),
        'gridded': gridded,
        'lat_grid': lat_grid,
        'lon_grid': lon_grid,
        'ship_lat': ship_lat,
        'ship_lon': ship_lon,
        'ref_lat': ref_lat,
        'ref_lon': ref_lon,
        'travel': travel,
        'n_frames': len(combine.frames),
        'n_pixels': n_pixels,
        'x_range': x_edges[-1] - x_edges[0],
        'y_range': y_edges[-1] - y_edges[0],
        'grid_shape': (len(x_edges) - 1, len(y_edges) - 1),
    }


def plot_diagnostics(combine: 'Combine',
                     figsize: tuple[float, float] = (10, 10),
                     n_along: int = 1200,
                     n_cross: int = 1600,
                     workers: int | None = None,
                     show_track: bool = False) -> None:
    """
    Show diagnostic plots for combined frames in earth coordinates.

    Uses parallel gridding for fast rendering of large datasets.

    Args:
        combine: Combine object with frames
        figsize: Figure size
        n_along: Grid bins along ship track (default 1200)
        n_cross: Grid bins cross track (default 1600)
        workers: Number of parallel workers for gridding (None = auto)
        show_track: Show separate ship track plot (default False)

    Displays:
    - Combined intensity in lat/lon coordinates (gridded)
    - Polar scatter plots for ship speed/heading and wind
    - Optional ship track detail plot
    - Statistics panel
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Get ship track (fast - doesn't require full coordinate computation)
    logging.debug("Computing ship track...")
    ship_x, ship_y = combine.ship_track_xy()
    ship_lat, ship_lon = combine.ship_track()
    travel = combine.travel_distance()
    n_radials = len(ship_x)
    n_frames = len(combine.frames)

    # Estimate pixel count
    n_pixels = sum(f.n_bearings * f.n_distances for f in combine.frames)

    # Use rotated grid for speed, then display in lat/lon
    logging.debug(f"Gridding {n_pixels:,} pixels from {n_frames} frames "
                  f"to {n_cross}x{n_along} rotated grid (parallel)...")

    # Grid all frames in parallel with rotation
    x_edges, y_edges, gridded, angle = combine.grid_parallel_rotated(
        n_along=n_along, n_cross=n_cross, workers=workers
    )

    # Convert edges to lat/lon for display
    # For rotated grid, we need to rotate back first
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # Create meshgrid of rotated coordinates
    xx_rot, yy_rot = np.meshgrid(x_edges, y_edges)

    # Rotate back to earth coordinates
    xx_earth = xx_rot * cos_a - yy_rot * sin_a
    yy_earth = xx_rot * sin_a + yy_rot * cos_a

    # Convert to lat/lon
    ref_lat, ref_lon = combine.reference_position
    meters_per_deg_lat = np.pi * _EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))

    lon_grid = ref_lon + xx_earth / meters_per_deg_lon
    lat_grid = ref_lat + yy_earth / meters_per_deg_lat

    # Calculate intensity limits from gridded data (ignoring NaN)
    valid_data = gridded[~np.isnan(gridded)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [1, 99])
    else:
        vmin, vmax = 0, 1

    logging.debug("Creating plot...")

    # Collect ship and wind data from all frames for polar plots
    ship_speeds = []
    ship_headings = []
    wind_speeds = []
    wind_dirs = []
    for frame in combine.frames:
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
    start_ts = pd.Timestamp(combine.frames[0].timestamp).floor('s')
    end_ts = pd.Timestamp(combine.frames[-1].timestamp).ceil('s')
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
    meta = combine.frames[0].metadata
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
        f"Reference: ({ref_lat:.6f}°, {ref_lon:.6f}°)    "
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

    logging.debug("Displaying...")
    plt.show()


def save_frame(combine: 'Combine',
               output_path: str,
               figsize: tuple[float, float] = (10, 10),
               n_along: int = 1200,
               n_cross: int = 1600,
               workers: int | None = None,
               dpi: int = 100) -> None:
    """
    Save a single frame image non-interactively.

    Args:
        combine: Combine object with frames
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
    import pandas as pd

    # Get ship track
    ship_x, ship_y = combine.ship_track_xy()
    ship_lat, ship_lon = combine.ship_track()
    n_radials = len(ship_x)
    n_frames = len(combine.frames)

    # Grid all frames in parallel with rotation
    x_edges, y_edges, gridded, angle = combine.grid_parallel_rotated(
        n_along=n_along, n_cross=n_cross, workers=workers
    )

    # Rotate grid back to earth coordinates and convert to lat/lon
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    xx_rot, yy_rot = np.meshgrid(x_edges, y_edges)
    xx_earth = xx_rot * cos_a - yy_rot * sin_a
    yy_earth = xx_rot * sin_a + yy_rot * cos_a

    ref_lat, ref_lon = combine.reference_position
    meters_per_deg_lat = np.pi * _EARTH_RADIUS / 180.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(ref_lat))
    lon_grid = ref_lon + xx_earth / meters_per_deg_lon
    lat_grid = ref_lat + yy_earth / meters_per_deg_lat

    # Calculate intensity limits
    valid_data = gridded[~np.isnan(gridded)]
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [1, 99])
    else:
        vmin, vmax = 0, 1

    # Collect ship and wind data
    ship_speeds, ship_headings, wind_speeds, wind_dirs = [], [], [], []
    for frame in combine.frames:
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
    start_ts = pd.Timestamp(combine.frames[0].timestamp).floor('s')
    end_ts = pd.Timestamp(combine.frames[-1].timestamp).ceil('s')
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
    travel = combine.travel_distance()
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


class CombineViewer:
    """
    Interactive viewer for combined/gridded group images with navigation buttons.

    Shows the gridded combined image for each time group (e.g., 10-minute periods).
    Provides prev/next/play buttons to navigate between groups.
    Supports dynamic group addition while the viewer is displayed.

    Example:
        >>> viewer = CombineViewer(total_groups=3)
        >>> viewer.add_group_data(group_data)  # Add pre-gridded data
        >>> viewer.show()  # Can also add more groups while showing

    Keyboard controls:
        - Left, p, b: Previous group
        - Right, n, f: Next group
        - Space: Play/Stop animation
    """

    def __init__(self,
                 total_groups: int = 0,
                 cmap: str = 'viridis',
                 figsize: tuple[float, float] = (10, 10)):
        """
        Initialize the viewer.

        Args:
            total_groups: Expected total number of groups (for progress display)
            cmap: Colormap name
            figsize: Figure size in inches
        """
        self._cmap = cmap
        self._figsize = figsize
        self._groups: list[dict] = []
        self._current_idx = 0
        self._total_groups = total_groups

        # Will be initialized when show() is called
        self._fig = None
        self._ax = None
        self._ax_info = None
        self._im = None
        self._cbar = None
        self._playing = False
        self._timer = None
        self._play_interval = 1000  # milliseconds between groups

        # For dynamic group addition
        self._check_timer = None
        self._pending_groups: list[dict] = []
        self._loading_complete = False

    def add_group_data(self, group_data: dict) -> None:
        """
        Add pre-gridded group data to the viewer.

        Args:
            group_data: Dictionary with gridded data (from grid_group)
        """
        self._groups.append(group_data)

        # If viewer is already showing, redraw to update group count
        if self._fig is not None:
            self._draw_plot()

    def set_loading_complete(self) -> None:
        """Mark that all groups have been loaded."""
        self._loading_complete = True
        if self._fig is not None:
            self._draw_plot()  # Redraw to update title

    def _add_nav_buttons(self) -> None:
        """Add prev/next/play navigation buttons."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        ax_prev = plt.axes([0.12, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.23, 0.02, 0.1, 0.04])
        ax_play = plt.axes([0.34, 0.02, 0.1, 0.04])

        self._btn_prev = Button(ax_prev, '← Prev')
        self._btn_next = Button(ax_next, 'Next →')
        self._btn_play = Button(ax_play, '▶ Play')

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_next.on_clicked(self._on_next)
        self._btn_play.on_clicked(self._on_play)

    def _on_prev(self, event) -> None:
        """Handle previous button click."""
        if self._current_idx > 0:
            self._current_idx -= 1
        else:
            self._current_idx = len(self._groups) - 1
        self._draw_plot()

    def _on_next(self, event) -> None:
        """Handle next button click."""
        if self._current_idx < len(self._groups) - 1:
            self._current_idx += 1
        else:
            self._current_idx = 0
        self._draw_plot()

    def _on_play(self, event) -> None:
        """Handle play/stop button click."""
        if self._playing:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self) -> None:
        """Start auto-advancing through groups."""
        self._playing = True
        self._btn_play.label.set_text('■ Stop')
        self._fig.canvas.draw_idle()

        self._timer = self._fig.canvas.new_timer(interval=self._play_interval)
        self._timer.add_callback(self._animation_step)
        self._timer.start()

    def _stop_animation(self) -> None:
        """Stop auto-advancing through groups."""
        self._playing = False
        self._btn_play.label.set_text('▶ Play')
        self._fig.canvas.draw_idle()

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _animation_step(self) -> None:
        """Advance to next group."""
        if not self._playing:
            return
        self._on_next(None)

    def _on_key(self, event) -> None:
        """Handle keyboard navigation."""
        if event.key in ('left', 'p', 'b'):
            self._on_prev(event)
        elif event.key in ('right', 'n', 'f'):
            self._on_next(event)
        elif event.key == ' ':
            self._on_play(event)

    def _draw_plot(self) -> None:
        """Draw the current group's combined image."""
        if not self._groups:
            return

        group = self._groups[self._current_idx]

        # Clear previous plot
        self._ax.clear()
        self._ax_info.clear()

        # Calculate intensity limits
        valid_data = group['gridded'][~np.isnan(group['gridded'])]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [1, 99])
        else:
            vmin, vmax = 0, 1

        # Plot gridded intensity
        self._im = self._ax.pcolormesh(
            group['lon_grid'], group['lat_grid'], group['gridded'],
            cmap=self._cmap, vmin=vmin, vmax=vmax, shading='flat'
        )

        # Overlay ship track (subsample if too many points)
        ship_lon = group['ship_lon']
        ship_lat = group['ship_lat']
        n_points = len(ship_lon)
        if n_points > 5000:
            step = n_points // 5000
            ship_lon = ship_lon[::step]
            ship_lat = ship_lat[::step]

        self._ax.plot(ship_lon, ship_lat, 'r-', linewidth=1.5)

        # Set axis limits based on valid data
        gridded = group['gridded']
        valid_mask = np.logical_and(~np.isnan(gridded), gridded != 0)
        valid_rows, valid_cols = np.where(valid_mask)

        if len(valid_rows) > 0:
            lon_vals = group['lon_grid'][valid_rows, valid_cols]
            lat_vals = group['lat_grid'][valid_rows, valid_cols]
            lon_min, lon_max = lon_vals.min(), lon_vals.max()
            lat_min, lat_max = lat_vals.min(), lat_vals.max()

            # Compute aspect ratio
            mean_lat = (lat_min + lat_max) / 2
            aspect_ratio = 1.0 / np.cos(np.deg2rad(mean_lat))

            self._ax.set_xlim(lon_min, lon_max)
            self._ax.set_ylim(lat_min, lat_max)
            self._ax.set_aspect(aspect_ratio)

        self._ax.set_xlabel('Longitude (°)')
        self._ax.set_ylabel('Latitude (°)')
        self._ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Title with loading progress
        n_loaded = len(self._groups)
        if self._loading_complete or self._total_groups == 0:
            title = f"Group {self._current_idx + 1}/{n_loaded}: {group['period']}"
        else:
            title = f"Group {self._current_idx + 1}/{n_loaded} (loading {n_loaded}/{self._total_groups}): {group['period']}"
        self._ax.set_title(title, fontsize=11)

        # Add colorbar only once
        if self._cbar is None:
            self._cbar = self._fig.colorbar(self._im, ax=self._ax, label='Intensity')
        else:
            self._cbar.mappable = self._im
            self._im.set_clim(vmin, vmax)

        # Info panel
        self._ax_info.axis('off')
        travel = group['travel']
        info_text = (
            f"Reference: ({group['ref_lat']:.6f}°, {group['ref_lon']:.6f}°)    "
            f"Coverage: {group['x_range']:.0f}m × {group['y_range']:.0f}m    "
            f"Grid: {group['grid_shape'][0]}×{group['grid_shape'][1]}\n"
            f"Ship Motion: {travel['duration_s']:.1f}s, {travel['total_m']:.1f}m total, "
            f"avg {travel['speed_m_s']:.2f} m/s    "
            f"Frames: {group['n_frames']}, Pixels: {group['n_pixels']:,}"
        )
        self._ax_info.text(0.5, 0.5, info_text,
                           transform=self._ax_info.transAxes,
                           ha='center', va='center',
                           fontfamily='monospace', fontsize=9)

        self._fig.canvas.draw_idle()

    def show(self) -> None:
        """Display the interactive viewer."""
        import matplotlib.pyplot as plt

        if not self._groups:
            logging.warning("No groups to display")
            return

        # Create figure
        self._fig = plt.figure(figsize=self._figsize)
        gs = self._fig.add_gridspec(2, 1, height_ratios=[12, 1], hspace=0.02)
        self._ax = self._fig.add_subplot(gs[0, 0])
        self._ax_info = self._fig.add_subplot(gs[1, 0])
        self._fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.12)

        # Draw initial plot
        self._draw_plot()

        # Add navigation buttons
        self._add_nav_buttons()

        # Connect keyboard navigation
        self._fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()
