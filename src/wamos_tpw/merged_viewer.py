#! /usr/bin/env python3
#
# Interactive viewer for merged WAMOS radar images
#
# Jan-2026, Pat Welch, pat@mousebrains.com

"""Interactive viewer for merged WAMOS radar images."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.merged_image import MergedImage

logger = logging.getLogger(__name__)


def _draw_range_rings(ax, extent: list[float]) -> None:
    """
    Draw range rings on the plot.

    Args:
        ax: Matplotlib axes
        extent: [x_min, x_max, y_min, y_max] in meters
    """
    # Determine appropriate ring spacing based on extent
    max_extent = max(abs(extent[0]), abs(extent[1]), abs(extent[2]), abs(extent[3]))

    if max_extent > 5000:
        ring_spacing = 1000  # 1 km rings
    elif max_extent > 2000:
        ring_spacing = 500  # 500 m rings
    elif max_extent > 1000:
        ring_spacing = 250  # 250 m rings
    else:
        ring_spacing = 100  # 100 m rings

    # Draw rings
    theta = np.linspace(0, 2 * np.pi, 100)
    max_ring_radius = int(max_extent / ring_spacing) * ring_spacing + ring_spacing

    for radius in range(ring_spacing, int(max_ring_radius) + 1, ring_spacing):
        x_ring = radius * np.cos(theta)
        y_ring = radius * np.sin(theta)
        ax.plot(x_ring, y_ring, "w-", alpha=0.3, linewidth=0.5)

        # Add label at top of ring
        if radius <= max_extent * 0.9:
            if radius >= 1000:
                label = f"{radius / 1000:.0f} km"
            else:
                label = f"{radius} m"
            ax.text(
                0,
                radius,
                label,
                ha="center",
                va="bottom",
                color="white",
                alpha=0.5,
                fontsize=7,
            )


def show_single_merged_image(merged: "MergedImage") -> None:
    """
    Display a single merged image in a non-blocking window.

    Used to show the first image while processing continues.

    Args:
        merged: MergedImage to display
    """
    import matplotlib.pyplot as plt

    # Find bounds of actual data (non-NaN values)
    valid_mask = ~np.isnan(merged.intensity)
    valid_rows = np.any(valid_mask, axis=1)
    valid_cols = np.any(valid_mask, axis=0)

    if np.any(valid_rows) and np.any(valid_cols):
        row_min, row_max = np.where(valid_rows)[0][[0, -1]]
        col_min, col_max = np.where(valid_cols)[0][[0, -1]]

        cropped = merged.intensity[row_min : row_max + 1, col_min : col_max + 1]
        extent = [
            merged.x_edges[col_min],
            merged.x_edges[col_max + 1],
            merged.y_edges[row_min],
            merged.y_edges[row_max + 1],
        ]
    else:
        cropped = merged.intensity
        extent = [merged.x_edges[0], merged.x_edges[-1], merged.y_edges[0], merged.y_edges[-1]]

    # Compute intensity range
    valid_data = cropped[~np.isnan(cropped)]
    if len(valid_data) > 0:
        vmin = float(np.percentile(valid_data, 2))
        vmax = float(np.percentile(valid_data, 98))
    else:
        vmin, vmax = 0, 1

    # Enable interactive mode
    plt.ion()

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        cropped,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="lower",
        aspect="equal",
    )

    # Add range rings
    _draw_range_rings(ax, extent)

    ax.set_xlabel("Distance East (m)")
    ax.set_ylabel("Distance North (m)")

    start_time = np.datetime_as_string(merged.start_time, unit="s")
    end_time = np.datetime_as_string(merged.end_time, unit="s")

    ax.set_title(f"First Merged Image: {merged.n_frames} frames\n{start_time} to {end_time}")

    fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)

    # Force draw and show non-blocking
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)
    plt.pause(0.5)  # Give time to render


def show_merged_viewer(merged_images: list["MergedImage"], interval_ms: int = 500) -> None:
    """
    Show an interactive viewer for merged images.

    Simple matplotlib-based viewer with navigation between windows.
    Includes play/stop button for automatic playback.

    Args:
        merged_images: List of merged images to display
        interval_ms: Playback interval in milliseconds (default 500ms = 2 fps)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button

    # Turn off interactive mode for blocking display
    plt.ioff()
    # Close any existing figures from single image preview
    plt.close("all")

    if not merged_images:
        logger.warning("No merged images to display")
        return

    # Compute global intensity range across all images
    all_valid = []
    for merged in merged_images:
        valid_data = merged.intensity[~np.isnan(merged.intensity)]
        if len(valid_data) > 0:
            all_valid.extend(valid_data.ravel())

    if not all_valid:
        logger.warning("All merged images have no valid data")
        # Still show the empty images for debugging
        vmin, vmax = 0, 1
    else:
        vmin = float(np.percentile(all_valid, 2))
        vmax = float(np.percentile(all_valid, 98))

    # Compute max speeds for consistent polar plot scaling
    max_ship_speed = 0.0
    max_wind_speed = 0.0
    for merged in merged_images:
        if merged.mean_ship_speed is not None:
            max_ship_speed = max(max_ship_speed, merged.mean_ship_speed)
        if merged.mean_wind_speed is not None:
            max_wind_speed = max(max_wind_speed, merged.mean_wind_speed)

    # Use reasonable defaults if no data
    if max_ship_speed == 0:
        max_ship_speed = 10.0  # m/s
    if max_wind_speed == 0:
        max_wind_speed = 20.0  # m/s

    # Create figure with explicit axes positioning
    # Use square figure for equal aspect ratio
    fig = plt.figure(figsize=(12, 11))

    # Main axes: make it square in figure coordinates
    # Height: 0.78 of figure (11 inches) = 8.58 inches
    # Width: 8.58/12 = 0.715 of figure width
    ax = fig.add_axes((0.08, 0.12, 0.715, 0.78))

    # Colorbar axes (fixed position, won't affect main axes)
    cax = fig.add_axes((0.82, 0.12, 0.03, 0.78))

    # Mutable containers for inset axes (recreated on each update since ax.clear() removes them)
    inset_axes = {"ship": None, "wind": None}

    current_idx = [0]  # Mutable container for callback
    is_playing = [False]  # Mutable container for play state
    animation = [None]  # Mutable container for animation object

    def create_polar_insets():
        """Create polar inset axes inside the main plot."""
        # Position: [left, bottom, width, height] relative to parent axes (0-1)
        # 70% of original size (0.14 vs 0.20), positioned closer to corners
        ax_ship = ax.inset_axes((0.85, 0.85, 0.14, 0.14), projection="polar")
        ax_wind = ax.inset_axes((0.85, 0.01, 0.14, 0.14), projection="polar")
        inset_axes["ship"] = ax_ship
        inset_axes["wind"] = ax_wind
        return ax_ship, ax_wind

    def update_polar_plots(merged):
        """Update the ship and wind polar inset plots."""
        ax_ship = inset_axes["ship"]
        ax_wind = inset_axes["wind"]

        # Ship plot (NE corner of main plot, label in NW corner of polar plot)
        ax_ship.clear()
        ax_ship.set_facecolor((1, 1, 1, 0.85))  # White with transparency
        ax_ship.set_theta_zero_location("N")
        ax_ship.set_theta_direction(-1)
        ax_ship.set_yticklabels([])
        ax_ship.set_xticklabels([])  # No N/E/S/W labels
        ax_ship.set_ylim(0, max_ship_speed * 1.1)

        if merged.mean_ship_speed is not None and merged.mean_ship_speed > 0:
            heading_rad = np.radians(merged.mean_heading)
            ax_ship.annotate(
                "",
                xy=(heading_rad, merged.mean_ship_speed),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": "blue", "lw": 2},
            )
            label_text = f"Ship\n{merged.mean_ship_speed:.1f} m/s"
        else:
            label_text = "Ship\nN/A"
        # Label in NW corner of polar plot
        ax_ship.text(
            0.02,
            0.98,
            label_text,
            transform=ax_ship.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            color="blue",
        )

        # Wind plot (SE corner of main plot, label in SW corner of polar plot)
        ax_wind.clear()
        ax_wind.set_facecolor((1, 1, 1, 0.85))  # White with transparency
        ax_wind.set_theta_zero_location("N")
        ax_wind.set_theta_direction(-1)
        ax_wind.set_yticklabels([])
        ax_wind.set_xticklabels([])  # No N/E/S/W labels
        ax_wind.set_ylim(0, max_wind_speed * 1.1)

        if merged.mean_wind_speed is not None and merged.mean_wind_speed > 0:
            # Wind direction is where wind comes FROM, arrow points in that direction
            wind_dir_rad = np.radians(merged.mean_wind_direction or 0)
            ax_wind.annotate(
                "",
                xy=(wind_dir_rad, merged.mean_wind_speed),
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": "green", "lw": 2},
            )
            label_text = f"Wind\n{merged.mean_wind_speed:.1f} m/s"
        else:
            label_text = "Wind\nN/A"
        # Label in SW corner of polar plot
        ax_wind.text(
            0.02,
            0.02,
            label_text,
            transform=ax_wind.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            color="green",
        )

    def update_plot():
        merged = merged_images[current_idx[0]]
        ax.clear()

        # Recreate inset axes (cleared by ax.clear())
        create_polar_insets()

        # Find bounds of actual data (non-NaN values)
        valid_mask = ~np.isnan(merged.intensity)
        valid_rows = np.any(valid_mask, axis=1)
        valid_cols = np.any(valid_mask, axis=0)

        if np.any(valid_rows) and np.any(valid_cols):
            row_min, row_max = np.where(valid_rows)[0][[0, -1]]
            col_min, col_max = np.where(valid_cols)[0][[0, -1]]

            # Crop to valid data region
            cropped = merged.intensity[row_min : row_max + 1, col_min : col_max + 1]
            extent = [
                merged.x_edges[col_min],
                merged.x_edges[col_max + 1],
                merged.y_edges[row_min],
                merged.y_edges[row_max + 1],
            ]
        else:
            # No valid data, show full extent
            cropped = merged.intensity
            extent = [merged.x_edges[0], merged.x_edges[-1], merged.y_edges[0], merged.y_edges[-1]]

        im = ax.imshow(
            cropped,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="lower",
            aspect="equal",  # Fill the axes
        )

        # Add range rings
        _draw_range_rings(ax, extent)

        ax.set_xlabel("Distance East (m)")
        ax.set_ylabel("Distance North (m)")

        start_time = np.datetime_as_string(merged.start_time, unit="s")
        end_time = np.datetime_as_string(merged.end_time, unit="s")

        ax.set_title(
            f"Window {current_idx[0] + 1}/{len(merged_images)}: {merged.n_frames} frames\n"
            f"{start_time} to {end_time}"
        )

        # Update colorbar (using dedicated axes so it doesn't resize main plot)
        if not hasattr(fig, "_colorbar"):
            fig._colorbar = fig.colorbar(im, cax=cax, label="Intensity")
        else:
            fig._colorbar.update_normal(im)

        # Update polar inset plots
        update_polar_plots(merged)

        fig.canvas.draw_idle()

    def on_prev(event):
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_plot()

    def on_next(event):
        if current_idx[0] < len(merged_images) - 1:
            current_idx[0] += 1
            update_plot()

    def animate_frame(frame_num):
        """Animation callback - advance to next frame."""
        if is_playing[0]:
            if current_idx[0] < len(merged_images) - 1:
                current_idx[0] += 1
            else:
                # Loop back to start
                current_idx[0] = 0
            update_plot()

    def on_play(event):
        """Toggle play/stop state."""
        is_playing[0] = not is_playing[0]

        if is_playing[0]:
            btn_play.label.set_text("Stop")
            # Start animation
            animation[0] = FuncAnimation(
                fig,
                animate_frame,
                interval=interval_ms,
                cache_frame_data=False,
            )
        else:
            btn_play.label.set_text("Play")
            # Stop animation
            if animation[0] is not None:
                animation[0].event_source.stop()
                animation[0] = None

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "left":
            on_prev(event)
        elif event.key == "right":
            on_next(event)
        elif event.key == " ":  # Spacebar toggles play
            on_play(event)

    # Add navigation buttons using explicit figure reference
    ax_prev = fig.add_axes((0.2, 0.02, 0.12, 0.05))
    ax_play = fig.add_axes((0.37, 0.02, 0.12, 0.05))
    ax_next = fig.add_axes((0.54, 0.02, 0.12, 0.05))

    btn_prev = Button(ax_prev, "Previous")
    btn_play = Button(ax_play, "Play")
    btn_next = Button(ax_next, "Next")

    btn_prev.on_clicked(on_prev)
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(on_next)

    # Key bindings
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial plot
    update_plot()

    # Show with blocking
    plt.show(block=True)
