#!/usr/bin/env python3
"""
Interactive frame viewer for WAMOS radar data.

Displays radar frames in polar coordinates, showing the original (destreaked)
image alongside the processed (deramped + dewinded) image.

Features:
  - Side-by-side polar views: original vs processed
  - Previous/Next navigation buttons
  - Play/Pause animation
  - Keyboard shortcuts (Left/Right arrows, Space for play/pause, Q to quit)

Usage:
    python frame_viewer.py 20220405 20220406 /path/to/POLAR
    python frame_viewer.py 20220405 20220406 /path/to/POLAR --interval 500

Jan-2026, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

# Suppress "Mean of empty slice" warnings from shadow-masked regions
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


class FrameViewer:
    """Interactive viewer for radar frames with polar and Cartesian views."""

    def __init__(
        self,
        files: list[str],
        config: Config,
        interval: int = 200,
        figsize: tuple[float, float] = (12, 6),
    ) -> None:
        """
        Initialize the frame viewer.

        Args:
            files: List of polar file paths
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

        # Create figure with 1x2 polar plots + button row
        self._fig = plt.figure(figsize=figsize)
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 0.06], hspace=0.15, wspace=0.15)

        # Polar views
        self._ax_polar_orig = self._fig.add_subplot(gs[0, 0], projection="polar")
        self._ax_polar_proc = self._fig.add_subplot(gs[0, 1], projection="polar")

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

        # Store image references for updating
        self._img_polar_orig = None
        self._img_polar_proc = None
        self._colorbar = None

        # Cache for processed frames
        self._cache: dict[int, dict] = {}
        self._cache_size = 10

        # Load and display first frame
        self._update_display()

    def _load_frame(self, idx: int) -> dict | None:
        """Load and process a single frame."""
        if idx in self._cache:
            return self._cache[idx]

        fn = self._files[idx]
        try:
            pf = PolarFile(fn, config=self._config)
            frame = pf.frame()

            theta = Theta(frame)
            destreaked = Destreak(frame)

            shadow = Shadow(destreaked.intensity, theta)
            if shadow.theta_bias:
                theta.set_bias(shadow.theta_bias)

            rng = Range(frame)
            masked_intensity = shadow.mask(destreaked.intensity)
            deramp = Deramp(masked_intensity, rng)
            dewind = Dewind(deramp.intensity, theta)

            result = {
                "filename": fn,
                "timestamp": frame.timestamp,
                "theta": theta.theta,
                "range": rng.slant_range,
                "original": destreaked.intensity.astype(np.float32),
                "processed": dewind.intensity.astype(np.float32),
                "shadow_indices": shadow.indices,
            }

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
        self, intensity: np.ndarray, theta: np.ndarray, range_vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert intensity array to polar mesh coordinates."""
        # Sort by theta for proper polar display
        sort_idx = np.argsort(theta)
        theta_sorted = np.deg2rad(theta[sort_idx])
        intensity_sorted = intensity[sort_idx, :]

        # Create mesh grids
        theta_mesh, range_mesh = np.meshgrid(
            np.append(theta_sorted, theta_sorted[0] + 2 * np.pi),  # Close the circle
            range_vals,
            indexing="ij",
        )

        # Append first row to close the circle
        intensity_closed = np.vstack([intensity_sorted, intensity_sorted[0:1, :]])

        return theta_mesh, range_mesh, intensity_closed

    def _update_display(self) -> None:
        """Update the display with the current frame."""
        data = self._load_frame(self._current_idx)
        if data is None:
            return

        theta = data["theta"]
        range_vals = data["range"]
        original = data["original"]
        processed = data["processed"]

        # Compute common color limits (ignoring NaN)
        vmin = min(np.nanpercentile(original, 1), np.nanpercentile(processed, 1))
        vmax = max(np.nanpercentile(original, 99), np.nanpercentile(processed, 99))

        # Clear axes
        for ax in [self._ax_polar_orig, self._ax_polar_proc]:
            ax.clear()

        # Polar plots
        theta_mesh, range_mesh, orig_closed = self._intensity_to_polar_mesh(
            original, theta, range_vals
        )
        _, _, proc_closed = self._intensity_to_polar_mesh(processed, theta, range_vals)

        self._ax_polar_orig.pcolormesh(
            theta_mesh,
            range_mesh,
            orig_closed,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self._ax_polar_orig.set_title("Original (Destreaked)")
        self._ax_polar_orig.set_theta_zero_location("N")
        self._ax_polar_orig.set_theta_direction(-1)

        self._ax_polar_proc.pcolormesh(
            theta_mesh,
            range_mesh,
            proc_closed,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self._ax_polar_proc.set_title("Processed (Dewinded)")
        self._ax_polar_proc.set_theta_zero_location("N")
        self._ax_polar_proc.set_theta_direction(-1)

        # Update title
        timestamp = data["timestamp"]
        self._fig.suptitle(
            f"Frame {self._current_idx + 1}/{len(self._files)}: {timestamp}",
            fontsize=12,
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
    parser = argparse.ArgumentParser(description="Interactive frame viewer for WAMOS radar data")

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

        viewer = FrameViewer(
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
