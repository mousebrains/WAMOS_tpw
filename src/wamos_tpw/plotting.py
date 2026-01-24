#! /usr/bin/env python3
#
# Common plotting utilities and base viewer class for WAMOS visualization
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.colorbar import Colorbar
    from wamos_tpw.frame import Frame
    from wamos_tpw.bearing import MultiTheta as Theta, MultiBearing as Bearing


# -----------------------------------------------------------------------------
# Radar Height Utility
# -----------------------------------------------------------------------------


def get_radar_height(radar_height: float | None, frame: "Frame") -> float | None:
    """
    Get radar height with fallback priority.

    Priority order:
    1. Explicitly provided radar_height parameter
    2. Frame metadata radar_height
    3. Frame metadata wind_sensor_height (as fallback)

    Args:
        radar_height: Explicitly provided radar height (takes priority if not None)
        frame: Frame object to get fallback heights from

    Returns:
        Radar height in meters, or None if not available from any source
    """
    if radar_height is not None:
        return radar_height
    if frame.metadata.radar_height is not None:
        return frame.metadata.radar_height
    if frame.metadata.wind_sensor_height is not None:
        return frame.metadata.wind_sensor_height
    return None


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def quantile_limits(
    data: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0
) -> Tuple[float, float]:
    """
    Calculate quantile-based colorbar limits.

    Args:
        data: Intensity data array
        low_pct: Lower percentile (default: 1%)
        high_pct: Upper percentile (default: 99%)

    Returns:
        Tuple of (vmin, vmax) based on quantiles
    """
    vmin = float(np.nanpercentile(data, low_pct))
    vmax = float(np.nanpercentile(data, high_pct))
    return vmin, vmax


def calc_bin_edges(centers: np.ndarray) -> np.ndarray:
    """
    Calculate bin edges from bin centers for pcolormesh.

    For N centers, returns N+1 edges. Edges are placed at midpoints
    between centers, with extrapolation at boundaries.

    Args:
        centers: Array of bin center values

    Returns:
        Array of bin edges (length = len(centers) + 1)
    """
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges


def sort_polar_data(bearing: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort bearing values and reorder data to match.

    Radar sweep data may have bearings that wrap around 360° -> 0°,
    making them non-monotonic. pcolormesh requires monotonic coordinates.
    This function sorts bearings to [0, 360) and reorders data rows accordingly.

    Args:
        bearing: 1D array of bearing values in degrees (length N)
        data: 2D array of shape (N, M) where N is number of bearings

    Returns:
        Tuple of (sorted_bearing, reordered_data)
    """
    # Get sort indices for bearing
    sort_idx = np.argsort(bearing)

    # Sort bearing and reorder data rows
    sorted_bearing = bearing[sort_idx]
    sorted_data = data[sort_idx, :]

    return sorted_bearing, sorted_data


def format_nav_title(frame: Frame) -> str:
    """
    Format ship and wind navigation info for plot titles.

    Args:
        frame: Frame object with metadata

    Returns:
        Formatted string with heading, speed, wind info
    """
    meta = frame.metadata
    parts = []

    # Ship info
    ship_parts = []
    if meta.heading is not None:
        ship_parts.append(f"Hdg: {meta.heading:.1f}°")
    if meta.ship_speed is not None:
        ship_parts.append(f"Spd: {meta.ship_speed:.1f} m/s")  # Already in m/s
    if ship_parts:
        parts.append(f"Ship: {', '.join(ship_parts)}")

    # Wind info
    wind_parts = []
    if meta.wind_direction is not None:
        wind_parts.append(f"Dir: {meta.wind_direction:.1f}°")
    if meta.wind_speed is not None:
        wind_parts.append(f"Spd: {meta.wind_speed:.1f} m/s")
    if wind_parts:
        parts.append(f"Wind: {', '.join(wind_parts)}")

    return " | ".join(parts) if parts else ""


def add_crosshairs(ax, color: str = "gray", linewidth: float = 0.5, linestyle: str = "--") -> None:
    """
    Add crosshairs at origin for ship/earth coordinate plots.

    Args:
        ax: Matplotlib axes
        color: Line color
        linewidth: Line width
        linestyle: Line style
    """
    ax.axhline(0, color=color, linewidth=linewidth, linestyle=linestyle)
    ax.axvline(0, color=color, linewidth=linewidth, linestyle=linestyle)


def add_range_rings(
    ax,
    max_range: float,
    interval: float = 1000.0,
    color: str = "gray",
    linewidth: float = 0.5,
    linestyle: str = ":",
    label: bool = True,
) -> None:
    """
    Add concentric range rings at regular intervals for ship/earth plots.

    Args:
        ax: Matplotlib axes
        max_range: Maximum range to show rings for (meters)
        interval: Distance between rings (default: 1000m = 1km)
        color: Ring color
        linewidth: Ring line width
        linestyle: Ring line style
        label: If True, add distance labels to rings
    """
    from matplotlib.patches import Circle

    # Calculate ring distances
    n_rings = int(max_range / interval)

    for i in range(1, n_rings + 1):
        radius = i * interval
        circle = Circle(
            (0, 0), radius, fill=False, color=color, linewidth=linewidth, linestyle=linestyle
        )
        ax.add_patch(circle)

        if label:
            # Add label at top of ring
            label_text = f"{radius / 1000:.0f}km" if radius >= 1000 else f"{radius:.0f}m"
            ax.text(0, radius, label_text, ha="center", va="bottom", fontsize=8, color=color)


# -----------------------------------------------------------------------------
# Plotting Helpers
# -----------------------------------------------------------------------------


def plot_polar(
    ax,
    data: np.ndarray,
    bearing: np.ndarray,
    range_vals: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    x_label: str = "Distance (m)",
    y_label: str = "Bearing (°)",
    shading: str = "auto",
):
    """
    Plot data in polar coordinates (bearing vs range).

    Args:
        ax: Matplotlib axes
        data: 2D data array (n_bearings, n_distances)
        bearing: Array of bearing values (degrees)
        range_vals: Array of range/distance values
        vmin: Minimum colorbar value
        vmax: Maximum colorbar value
        cmap: Colormap name
        x_label: X-axis label
        y_label: Y-axis label
        shading: pcolormesh shading mode

    Returns:
        QuadMesh object from pcolormesh
    """
    im = ax.pcolormesh(range_vals, bearing, data, vmin=vmin, vmax=vmax, cmap=cmap, shading=shading)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return im


def plot_cartesian(
    ax,
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    x_label: str = "X (m)",
    y_label: str = "Y (m)",
    equal_aspect: bool = True,
    crosshairs: bool = True,
    shading: str = "auto",
):
    """
    Plot data in cartesian coordinates.

    Args:
        ax: Matplotlib axes
        data: 2D data array
        x: X coordinate array (2D, same shape as data)
        y: Y coordinate array (2D, same shape as data)
        vmin: Minimum colorbar value
        vmax: Maximum colorbar value
        cmap: Colormap name
        x_label: X-axis label
        y_label: Y-axis label
        equal_aspect: If True, set equal aspect ratio
        crosshairs: If True, add crosshairs at origin
        shading: pcolormesh shading mode

    Returns:
        QuadMesh object from pcolormesh
    """
    im = ax.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap, shading=shading)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if equal_aspect:
        ax.set_aspect("equal")

    if crosshairs:
        add_crosshairs(ax)

    return im


# -----------------------------------------------------------------------------
# Base Viewer Class
# -----------------------------------------------------------------------------


class BaseViewer(ABC):
    """
    Abstract base class for interactive frame viewers with navigation.

    Provides common functionality for frame navigation, keyboard handling,
    button creation, and standard polar/ship/earth view rendering.

    Subclasses should:
    1. Call super().__init__() first
    2. Set up frames, theta, bearing, vmin/vmax
    3. Create figure and axes
    4. Call _draw_plot(), _add_nav_buttons(), _connect_keyboard()

    Attributes that subclasses should set for standard views:
        _frames: list[Frame] - Frame objects to display
        _theta: Theta - For bearing calculations
        _bearing: Bearing - For coordinate transformations
        _vmin, _vmax: float - Colorbar limits
        _radar_height: float | None - Radar height for ground range
        _ax: Axes - Main plot axes
        _im: QuadMesh - Current image (for colorbar)
        _cbar: Colorbar - Colorbar instance

    Example:
        class MyViewer(BaseViewer):
            def __init__(self, frames, ...):
                super().__init__(n_frames=len(frames), cmap='viridis')
                self._frames = frames
                self._theta = Theta(frames, config)
                self._bearing = Bearing(self._theta)
                # ... create figure, call _draw_plot(), etc.
    """

    def __init__(
        self, n_frames: int, cmap: str = "viridis", figsize: Tuple[float, float] = (10, 8)
    ):
        """
        Initialize base viewer.

        Args:
            n_frames: Total number of frames
            cmap: Colormap name
            figsize: Default figure size
        """
        self._n_frames = n_frames
        self._current_idx = 0
        self._cmap = cmap
        self._figsize = figsize

        # Matplotlib objects (set by subclasses after figure creation)
        self._fig: Figure | None = None
        self._axes: list[Axes] | None = None
        self._ax: Axes | None = None
        self._im: AxesImage | None = None
        self._cbar: Colorbar | None = None

        # View mode support
        self._view: str | None = None
        self._view_keys: dict[str, str] = {}  # Maps keyboard keys to view names

        # Subclasses should set these for standard draw methods
        self._frames: list[Frame] | None = None
        self._theta: Theta | None = None
        self._bearing: Bearing | None = None
        self._vmin: float = 0
        self._vmax: float = 4095
        self._radar_height: float | None = None

        # Animation support
        self._play_interval: int = 500  # milliseconds between frames

    @property
    def current_index(self) -> int:
        """Return current frame index."""
        return self._current_idx

    @property
    def n_frames(self) -> int:
        """Return total number of frames."""
        return self._n_frames

    # -------------------------------------------------------------------------
    # Abstract methods - subclasses must implement
    # -------------------------------------------------------------------------

    @abstractmethod
    def _draw_plot(self) -> None:
        """Draw or redraw the current plot. Subclasses must implement."""
        pass

    @abstractmethod
    def _update_title(self) -> None:
        """Update the figure title. Subclasses must implement."""
        pass

    @abstractmethod
    def _get_frame(self, idx: int):
        """
        Get the Frame object at given index.

        Args:
            idx: Frame index

        Returns:
            Frame object (used for title formatting)
        """
        pass

    # -------------------------------------------------------------------------
    # Common draw methods for polar/ship/earth views
    # -------------------------------------------------------------------------

    def _get_radar_height(self, frame_idx: int) -> float | None:
        """Get radar height with fallback to frame metadata."""
        if self._frames is None:
            return self._radar_height
        return get_radar_height(self._radar_height, self._frames[frame_idx])

    def _draw_polar(self, frame: "Frame", data: np.ndarray) -> None:
        """
        Draw polar view (bearing vs ground distance).

        Uses self._theta for bearing and self._ax for plotting.
        Requires _theta and _ax to be set by subclass.
        """
        assert self._theta is not None, "_theta must be set before calling _draw_polar"
        assert self._ax is not None, "_ax must be set before calling _draw_polar"

        height = self._get_radar_height(self._current_idx)

        if height is not None:
            distances = frame.ground_range(height)
            x_label = "Ground Distance (m)"
        else:
            distances = frame.slant_range()
            x_label = "Slant Range (m)"

        bearing_centers = self._theta.bearing_for_frame(self._current_idx)

        # Sort bearing and data to ensure monotonic coordinates for pcolormesh
        sorted_bearing, sorted_data = sort_polar_data(bearing_centers, data)

        dist_edges = calc_bin_edges(distances)
        bearing_edges = calc_bin_edges(sorted_bearing)

        self._im = self._ax.pcolormesh(
            dist_edges,
            bearing_edges,
            sorted_data,
            vmin=self._vmin,
            vmax=self._vmax,
            cmap=self._cmap,
            shading="flat",
        )
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel("Bearing (°)")

    def _draw_ship(self, frame: "Frame", data: np.ndarray) -> None:
        """
        Draw ship-relative x/y view (+X=starboard, +Y=bow).

        Uses self._bearing for coordinates and self._ax for plotting.
        Requires _bearing and _ax to be set by subclass.
        """
        assert self._bearing is not None, "_bearing must be set before calling _draw_ship"
        assert self._ax is not None, "_ax must be set before calling _draw_ship"

        import warnings

        x, y = self._bearing.xy_ship(self._current_idx)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*input coordinates.*pcolormesh.*")
            self._im = self._ax.pcolormesh(
                x, y, data, vmin=self._vmin, vmax=self._vmax, cmap=self._cmap, shading="nearest"
            )
        self._ax.set_xlabel("X - Starboard (m)")
        self._ax.set_ylabel("Y - Bow (m)")
        self._ax.set_aspect("equal")
        add_crosshairs(self._ax)

        max_range = np.sqrt(x**2 + y**2).max()
        add_range_rings(self._ax, max_range, interval=1000.0)

    def _draw_earth(self, frame: "Frame", data: np.ndarray) -> None:
        """
        Draw earth-relative x/y view (+X=East, +Y=North).

        Uses self._bearing for coordinates and self._ax for plotting.
        Requires _bearing and _ax to be set by subclass.
        """
        assert self._bearing is not None, "_bearing must be set before calling _draw_earth"
        assert self._ax is not None, "_ax must be set before calling _draw_earth"

        import warnings

        x, y = self._bearing.xy_earth(self._current_idx)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*input coordinates.*pcolormesh.*")
            self._im = self._ax.pcolormesh(
                x, y, data, vmin=self._vmin, vmax=self._vmax, cmap=self._cmap, shading="nearest"
            )
        self._ax.set_xlabel("X - East (m)")
        self._ax.set_ylabel("Y - North (m)")
        self._ax.set_aspect("equal")
        add_crosshairs(self._ax)

        max_range = np.sqrt(x**2 + y**2).max()
        add_range_rings(self._ax, max_range, interval=1000.0)

    def _draw_view(self) -> None:
        """
        Draw the current view based on self._view.

        Dispatches to _draw_polar, _draw_ship, or _draw_earth.
        Clears axes, draws plot, updates title, and handles colorbar.
        Requires _frames, _ax, and _fig to be set by subclass.
        """
        assert self._frames is not None, "_frames must be set before calling _draw_view"
        assert self._ax is not None, "_ax must be set before calling _draw_view"
        assert self._fig is not None, "_fig must be set before calling _draw_view"

        frame = self._frames[self._current_idx]
        data = frame.intensity

        # Clear previous plot
        self._ax.clear()

        if self._view == "polar":
            self._draw_polar(frame, data)
        elif self._view == "ship":
            self._draw_ship(frame, data)
        elif self._view == "earth":
            self._draw_earth(frame, data)

        # Update title
        self._update_title()

        # Add colorbar only once, but update mappable each time
        if self._cbar is None:
            self._cbar = self._fig.colorbar(self._im, ax=self._ax, label="Intensity (0-4095)")
        else:
            # Update colorbar to reference new image while keeping same limits
            self._cbar.mappable = self._im
            self._im.set_clim(self._vmin, self._vmax)

        self._fig.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Navigation - shared implementation
    # -------------------------------------------------------------------------

    def _on_prev(self, event) -> None:
        """Handle previous button click. Wraps to last frame."""
        if self._current_idx > 0:
            self._current_idx -= 1
        else:
            self._current_idx = self._n_frames - 1
        self._draw_plot()

    def _on_next(self, event) -> None:
        """Handle next button click. Wraps to first frame."""
        if self._current_idx < self._n_frames - 1:
            self._current_idx += 1
        else:
            self._current_idx = 0
        self._draw_plot()

    def _on_play(self, event) -> None:
        """Handle play/stop button click. Toggle auto-advance."""
        if self._playing:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self) -> None:
        """Start auto-advancing through frames."""
        self._playing = True
        self._btn_play.label.set_text("■ Stop")
        self._fig.canvas.draw_idle()

        # Create timer for animation
        self._timer = self._fig.canvas.new_timer(interval=self._play_interval)
        self._timer.add_callback(self._animation_step)
        self._timer.start()

    def _stop_animation(self) -> None:
        """Stop auto-advancing through frames."""
        self._playing = False
        self._btn_play.label.set_text("▶ Play")
        self._fig.canvas.draw_idle()

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _animation_step(self) -> None:
        """Advance to next frame (called by timer)."""
        if not self._playing:
            return
        self._on_next(None)

    def _on_key(self, event) -> None:
        """
        Handle keyboard navigation.

        Default bindings:
        - left, p, b: Previous frame
        - right, n, f: Next frame
        - space: Toggle play/stop
        - 1, 2, 3, ...: View modes (if _view_keys is set)
        """
        if event.key in ("left", "p", "b"):
            self._on_prev(event)
        elif event.key in ("right", "n", "f"):
            self._on_next(event)
        elif event.key == " ":
            self._on_play(event)
        elif event.key in self._view_keys:
            self._set_view(self._view_keys[event.key])

    def _set_view(self, view: str) -> None:
        """
        Change the view mode. Override in subclass if needed.

        Args:
            view: View mode name
        """
        if view != self._view:
            self._view = view
            self._draw_plot()

    # -------------------------------------------------------------------------
    # Button and keyboard setup
    # -------------------------------------------------------------------------

    def _add_nav_buttons(
        self, prev_pos: list = None, next_pos: list = None, play_pos: list = None
    ) -> None:
        """
        Add previous/next/play navigation buttons.

        Args:
            prev_pos: Position [left, bottom, width, height] for prev button
            next_pos: Position [left, bottom, width, height] for next button
            play_pos: Position [left, bottom, width, height] for play button
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        if prev_pos is None:
            prev_pos = [0.1, 0.02, 0.1, 0.04]
        if next_pos is None:
            next_pos = [0.21, 0.02, 0.1, 0.04]
        if play_pos is None:
            play_pos = [0.32, 0.02, 0.1, 0.04]

        ax_prev = plt.axes(prev_pos)
        ax_next = plt.axes(next_pos)
        ax_play = plt.axes(play_pos)

        self._btn_prev = Button(ax_prev, "← Prev")
        self._btn_next = Button(ax_next, "Next →")
        self._btn_play = Button(ax_play, "▶ Play")

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_next.on_clicked(self._on_next)
        self._btn_play.on_clicked(self._on_play)

        # Animation state
        self._playing = False
        self._timer = None
        self._play_interval = 500  # milliseconds between frames

    def _add_view_buttons(self, views: list[Tuple[str, str, list]]) -> None:
        """
        Add view mode buttons.

        Args:
            views: List of (view_name, label, position) tuples
                   position is [left, bottom, width, height]
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._view_buttons = {}
        for view_name, label, pos in views:
            ax = plt.axes(pos)
            btn = Button(ax, label)
            btn.on_clicked(lambda e, v=view_name: self._set_view(v))
            self._view_buttons[view_name] = btn

    def _connect_keyboard(self) -> None:
        """Connect keyboard event handler to figure."""
        if self._fig is not None:
            self._fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def show(self) -> None:
        """Display the interactive viewer."""
        import matplotlib.pyplot as plt

        plt.show()


# -----------------------------------------------------------------------------
# Main (testing)
# -----------------------------------------------------------------------------


def main() -> None:
    """Test plotting utilities."""
    import matplotlib.pyplot as plt

    # Test quantile_limits
    data = np.random.randint(0, 4096, (100, 200))
    vmin, vmax = quantile_limits(data)
    print(f"Quantile limits: {vmin:.1f} - {vmax:.1f}")

    # Test calc_bin_edges
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    edges = calc_bin_edges(centers)
    print(f"Centers: {centers}")
    print(f"Edges: {edges}")

    # Test plot_polar and plot_cartesian
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bearing = np.linspace(0, 360, 100)
    range_vals = np.linspace(100, 2000, 200)
    data = np.random.rand(100, 200) * 4095

    vmin, vmax = quantile_limits(data)

    # Polar plot
    plot_polar(ax1, data, bearing, range_vals, vmin, vmax, x_label="Slant Range (m)")
    ax1.set_title("Polar Plot Test")

    # Cartesian plot
    theta = np.deg2rad(bearing)
    r = range_vals
    theta_2d = theta[:, np.newaxis]
    r_2d = r[np.newaxis, :]
    x = r_2d * np.sin(theta_2d)
    y = r_2d * np.cos(theta_2d)

    plot_cartesian(ax2, data, x, y, vmin, vmax, x_label="X - East (m)", y_label="Y - North (m)")
    ax2.set_title("Cartesian Plot Test")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
