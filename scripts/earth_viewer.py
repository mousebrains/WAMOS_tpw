#!/usr/bin/env python3
"""
Interactive frame viewer in earth coordinates for WAMOS radar data.

Displays radar frames with earth-referenced bearings by combining:
- Radar-relative theta (from Theta class)
- Ship heading interpolated between frames

The earth bearing for each radial is: earth_bearing = (theta + heading) % 360

Features:
  - Side-by-side views: ship-relative (original) vs earth-referenced
  - Previous/Next navigation buttons
  - Play/Pause animation
  - Keyboard shortcuts (Left/Right arrows, Space for play/pause, Q to quit)
  - Heading interpolation between consecutive frames

Usage:
    python earth_viewer.py 20220405 20220406 /path/to/POLAR
    python earth_viewer.py 20220405 20220406 /path/to/POLAR --interval 500

Jan-2026, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

# Suppress warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.deramp import Deramp  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.dewind import Dewind  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.range import Range  # noqa: E402
from wamos_tpw.shadow import Shadow  # noqa: E402
from wamos_tpw.theta import Theta  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Processed frame data for display."""

    filename: str
    timestamp: np.datetime64
    theta: np.ndarray  # Radar-relative angles (degrees)
    earth_bearing: np.ndarray  # Earth-referenced angles (degrees)
    range_vals: np.ndarray  # Slant range (meters)
    intensity: np.ndarray  # Processed intensity
    heading: float  # Ship heading at frame time
    heading_interpolated: np.ndarray  # Interpolated heading per radial
    latitude: float | None
    longitude: float | None


class EarthViewer:
    """Interactive viewer for radar frames in earth coordinates."""

    def __init__(
        self,
        files: list[str],
        config: Config,
        interval: int = 200,
        figsize: tuple[float, float] = (12, 6),
    ) -> None:
        """
        Initialize the earth-referenced frame viewer.

        Args:
            files: List of polar file paths (should be sorted by time)
            config: Configuration object
            interval: Animation interval in milliseconds
            figsize: Figure size as (width, height)
        """
        self._files = files
        self._config = config
        self._interval = interval
        self._current_idx = 0
        self._playing = False
        self._timer = None

        # Pre-load frame metadata for heading interpolation
        self._frame_metadata = self._load_metadata()

        # Create figure with 1x2 polar plots + button row
        self._fig = plt.figure(figsize=figsize)
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 0.06], hspace=0.15, wspace=0.15)

        # Polar views
        self._ax_ship = self._fig.add_subplot(gs[0, 0], projection="polar")
        self._ax_earth = self._fig.add_subplot(gs[0, 1], projection="polar")

        # Button axes (bottom row)
        btn_width = 0.1
        btn_height = 0.04
        btn_y = 0.02
        btn_spacing = 0.02

        # Center the buttons
        total_width = 3 * btn_width + 2 * btn_spacing
        start_x = 0.5 - total_width / 2

        ax_prev = self._fig.add_axes([start_x, btn_y, btn_width, btn_height])
        ax_play = self._fig.add_axes(
            [start_x + btn_width + btn_spacing, btn_y, btn_width, btn_height]
        )
        ax_next = self._fig.add_axes(
            [start_x + 2 * (btn_width + btn_spacing), btn_y, btn_width, btn_height]
        )

        self._btn_prev = Button(ax_prev, "< Previous")
        self._btn_play = Button(ax_play, "> Play")
        self._btn_next = Button(ax_next, "Next >")

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_play.on_clicked(self._on_play)
        self._btn_next.on_clicked(self._on_next)

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Cache for processed frames
        self._cache: dict[int, FrameData] = {}
        self._cache_size = 10

        # Load and display first frame
        self._update_display()

    def _load_metadata(self) -> list[dict]:
        """Load basic metadata from all frames for heading interpolation."""
        metadata = []
        for fn in self._files:
            try:
                pf = PolarFile(fn, config=self._config)
                frame = pf.frame()
                meta = frame.metadata
                metadata.append(
                    {
                        "filename": fn,
                        "timestamp": frame.timestamp,
                        "heading": meta.heading,
                        "repeat_time": meta.repeat_time if meta.repeat_time > 0 else 1.5,
                        "n_radials": frame.n_bearings,
                        "latitude": meta.latitude,
                        "longitude": meta.longitude,
                    }
                )
            except Exception:
                logger.exception("Failed to load metadata from %s", fn)
                metadata.append(None)
        return metadata

    def _interpolate_heading(
        self,
        n_radials: int,
        repeat_time: float,
        curr_heading: float,
        next_heading: float | None,
    ) -> np.ndarray:
        """
        Interpolate heading across radials within a frame.

        Args:
            n_radials: Number of radials in frame
            repeat_time: Frame duration in seconds
            curr_heading: Heading at start of frame
            next_heading: Heading at start of next frame (None if last frame)

        Returns:
            Array of headings for each radial
        """
        if next_heading is None:
            return np.full(n_radials, curr_heading, dtype=np.float32)

        # Handle wraparound (e.g., 359 -> 1 degrees)
        delta = next_heading - curr_heading
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360

        # Linear interpolation across frame
        t_frac = np.linspace(0, 1, n_radials, endpoint=False)
        headings = (curr_heading + delta * t_frac) % 360

        return headings.astype(np.float32)

    def _load_frame(self, idx: int) -> FrameData | None:
        """Load and process a single frame with earth-referenced bearings."""
        if idx in self._cache:
            return self._cache[idx]

        fn = self._files[idx]
        try:
            pf = PolarFile(fn, config=self._config)
            frame = pf.frame()
            meta = frame.metadata

            # Calculate theta
            theta_obj = Theta(frame)

            # Apply shadow bias if available
            destreaked = Destreak(frame)
            shadow = Shadow(destreaked.intensity, theta_obj)
            if shadow.theta_bias:
                theta_obj.set_bias(shadow.theta_bias)

            # Process intensity
            rng = Range(frame)
            masked_intensity = shadow.mask(destreaked.intensity)
            deramp = Deramp(masked_intensity, rng)
            dewind = Dewind(deramp.intensity, theta_obj)

            theta = theta_obj.theta

            # Get heading for this frame and next frame
            curr_meta = self._frame_metadata[idx]
            next_meta = self._frame_metadata[idx + 1] if idx < len(self._files) - 1 else None

            curr_heading = (
                curr_meta["heading"] if curr_meta and curr_meta["heading"] is not None else 0.0
            )
            next_heading = (
                next_meta["heading"] if next_meta and next_meta["heading"] is not None else None
            )

            # Interpolate heading across radials
            n_radials = len(theta)
            repeat_time = curr_meta["repeat_time"] if curr_meta else 1.5
            heading_interp = self._interpolate_heading(
                n_radials, repeat_time, curr_heading, next_heading
            )

            # Calculate earth-referenced bearing
            # Earth bearing = theta (radar-relative) + heading (ship orientation)
            earth_bearing = (theta + heading_interp) % 360

            result = FrameData(
                filename=fn,
                timestamp=frame.timestamp,
                theta=theta,
                earth_bearing=earth_bearing,
                range_vals=rng.slant_range,
                intensity=dewind.intensity.astype(np.float32),
                heading=curr_heading,
                heading_interpolated=heading_interp,
                latitude=meta.latitude,
                longitude=meta.longitude,
            )

            # Manage cache size
            if len(self._cache) >= self._cache_size:
                oldest = min(self._cache.keys())
                del self._cache[oldest]

            self._cache[idx] = result
            return result

        except Exception:
            logger.exception("Failed to load %s", fn)
            return None

    def _intensity_to_polar_mesh(
        self,
        intensity: np.ndarray,
        angles: np.ndarray,
        range_vals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert intensity array to polar mesh coordinates."""
        # Sort by angle for proper polar display
        sort_idx = np.argsort(angles)
        angles_sorted = np.deg2rad(angles[sort_idx])
        intensity_sorted = intensity[sort_idx, :]

        # Create mesh grids
        angles_mesh, range_mesh = np.meshgrid(
            np.append(angles_sorted, angles_sorted[0] + 2 * np.pi),  # Close the circle
            range_vals,
            indexing="ij",
        )

        # Append first row to close the circle
        intensity_closed = np.vstack([intensity_sorted, intensity_sorted[0:1, :]])

        return angles_mesh, range_mesh, intensity_closed

    def _update_display(self) -> None:
        """Update the display with the current frame."""
        data = self._load_frame(self._current_idx)
        if data is None:
            return

        intensity = data.intensity
        range_vals = data.range_vals

        # Compute common color limits (ignoring NaN)
        vmin = np.nanpercentile(intensity, 1)
        vmax = np.nanpercentile(intensity, 99)

        # Clear axes
        for ax in [self._ax_ship, self._ax_earth]:
            ax.clear()

        # Ship-relative view (using theta)
        theta_mesh, range_mesh, intensity_closed = self._intensity_to_polar_mesh(
            intensity, data.theta, range_vals
        )
        self._ax_ship.pcolormesh(
            theta_mesh,
            range_mesh,
            intensity_closed,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self._ax_ship.set_title(f"Ship-Relative (heading={data.heading:.1f}°)")
        self._ax_ship.set_theta_zero_location("N")
        self._ax_ship.set_theta_direction(-1)

        # Earth-referenced view (using earth_bearing)
        earth_mesh, range_mesh, intensity_closed = self._intensity_to_polar_mesh(
            intensity, data.earth_bearing, range_vals
        )
        self._ax_earth.pcolormesh(
            earth_mesh,
            range_mesh,
            intensity_closed,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self._ax_earth.set_title("Earth-Referenced (North up)")
        self._ax_earth.set_theta_zero_location("N")
        self._ax_earth.set_theta_direction(-1)

        # Update title with position info
        pos_str = ""
        if data.latitude is not None and data.longitude is not None:
            pos_str = f" | Pos: ({data.latitude:.4f}°, {data.longitude:.4f}°)"

        self._fig.suptitle(
            f"Frame {self._current_idx + 1}/{len(self._files)}: {data.timestamp}{pos_str}",
            fontsize=11,
            fontweight="bold",
        )

        self._fig.canvas.draw_idle()

    def _on_prev(self, event=None) -> None:
        """Handle Previous button click."""
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_display()

    def _on_next(self, event=None) -> None:
        """Handle Next button click."""
        if self._current_idx < len(self._files) - 1:
            self._current_idx += 1
            self._update_display()

    def _on_play(self, event=None) -> None:
        """Handle Play/Pause button click."""
        self._playing = not self._playing
        if self._playing:
            self._btn_play.label.set_text("|| Pause")
            self._timer = self._fig.canvas.new_timer(interval=self._interval)
            self._timer.add_callback(self._animate_step)
            self._timer.start()
        else:
            self._btn_play.label.set_text("> Play")
            if self._timer:
                self._timer.stop()
                self._timer = None
        self._fig.canvas.draw_idle()

    def _animate_step(self) -> None:
        """Advance one frame during animation."""
        if self._current_idx < len(self._files) - 1:
            self._current_idx += 1
            self._update_display()
        else:
            # Stop at end
            self._on_play()  # Toggle back to paused

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == "left":
            self._on_prev()
        elif event.key == "right":
            self._on_next()
        elif event.key == " ":
            self._on_play()
        elif event.key == "q":
            plt.close(self._fig)

    def show(self) -> None:
        """Display the viewer."""
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive frame viewer in earth coordinates")

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--interval",
        type=int,
        default=200,
        help="Animation interval in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,6",
        help="Figure size as 'width,height' in inches (default: 12,6)",
    )

    args = parser.parse_args()
    setup_logging(args)

    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        logger.info("Found %d files", n_files)

        config = Config(args.config)

        try:
            figsize = tuple(float(x) for x in args.figsize.split(","))
        except ValueError:
            figsize = (12, 6)

        viewer = EarthViewer(
            files=files,
            config=config,
            interval=args.interval,
            figsize=figsize,
        )
        viewer.show()

        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
