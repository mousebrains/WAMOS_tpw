#! /usr/bin/env python3
#
# Diagnostic viewers for frame pipeline visualization
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import Config
    from wamos_tpw.frame_pipeline import FramePipeline

logger = logging.getLogger(__name__)


class PipelineDiagnosticViewer:
    """
    Interactive viewer for pipeline diagnostic stages.

    Displays a 3x4 grid showing:
    - Row 0: Raw Intensity, Destreaked, Shadow Mask, Theta
    - Row 1: Shadow Applied, Deramped, Dewinded, Final (Normalized)
    - Row 2: Difference plots between stages

    Navigation: Arrow keys, Space for play/pause, or Prev/Play/Next buttons.
    """

    def __init__(
        self,
        files: list[str],
        config: Config,
        frame_index: int = 0,
        interval: int = 500,
    ) -> None:
        """
        Initialize the diagnostic viewer.

        Args:
            files: List of polar file paths
            config: Configuration object
            frame_index: Which frame to extract from each file
            interval: Animation interval in milliseconds
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        from wamos_tpw.frame_pipeline import FramePipeline
        from wamos_tpw.polarfile import PolarFile

        # Suppress "Mean of empty slice" warnings from shadow-masked regions
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

        self._files = files
        self._config = config
        self._frame_index = frame_index
        self._current_idx = 0
        self._playing = False
        self._timer = None
        self._interval = interval
        self._cache: dict[int, tuple[FramePipeline, np.ndarray]] = {}
        self._cache_size = 5
        self._PolarFile = PolarFile
        self._FramePipeline = FramePipeline

        # Create figure with 3x4 grid of plots + button row
        self._fig = plt.figure(figsize=(16, 12))
        gs = self._fig.add_gridspec(
            4, 4,
            height_ratios=[1, 1, 1, 0.06],
            hspace=0.3,
            wspace=0.2,
        )

        # Stage plots (3 rows x 4 columns)
        # Row 0: Raw, Destreaked, Shadow Mask, Theta
        # Row 1: Shadow Applied, Deramped, Dewinded, Final
        # Row 2: Differences
        self._axes = {}
        stage_names = [
            ("Raw Intensity", 0, 0),
            ("Destreaked", 0, 1),
            ("Shadow Mask", 0, 2),
            ("Theta (beam angle)", 0, 3),
            ("Shadow Applied", 1, 0),
            ("Deramped", 1, 1),
            ("Dewinded", 1, 2),
            ("Final (Processed)", 1, 3),
            ("Destreaked - Raw", 2, 0),
            ("Shadow - Destreaked", 2, 1),
            ("Deramped - Shadow", 2, 2),
            ("Dewinded - Deramped", 2, 3),
        ]

        # Create first intensity axis as reference for sharing
        ref_ax = self._fig.add_subplot(gs[0, 0])
        ref_ax.set_title("Raw Intensity", fontsize=10)
        self._axes["Raw Intensity"] = ref_ax

        for name, row, col in stage_names[1:]:  # Skip first (already created)
            if name == "Theta (beam angle)":
                # Theta plot doesn't share axes
                ax = self._fig.add_subplot(gs[row, col])
            else:
                # All other intensity plots share axes with reference
                ax = self._fig.add_subplot(gs[row, col], sharex=ref_ax, sharey=ref_ax)
            ax.set_title(name, fontsize=10)
            self._axes[name] = ax

        # Button axes (bottom row)
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        btn_spacing = 0.02

        total_width = 3 * btn_width + 2 * btn_spacing
        start_x = 0.5 - total_width / 2

        ax_prev = self._fig.add_axes([start_x, btn_y, btn_width, btn_height])
        ax_play = self._fig.add_axes(
            [start_x + btn_width + btn_spacing, btn_y, btn_width, btn_height]
        )
        ax_next = self._fig.add_axes(
            [start_x + 2 * (btn_width + btn_spacing), btn_y, btn_width, btn_height]
        )

        self._btn_prev = Button(ax_prev, "< Prev")
        self._btn_play = Button(ax_play, "> Play")
        self._btn_next = Button(ax_next, "Next >")

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_play.on_clicked(self._on_play)
        self._btn_next.on_clicked(self._on_next)

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Load and display first frame
        self._update_display()

    def _load_pipeline(self, idx: int) -> tuple[FramePipeline, np.ndarray] | None:
        """Load and process a single frame through the pipeline."""
        if idx in self._cache:
            return self._cache[idx]

        fn = self._files[idx]
        try:
            pf = self._PolarFile(fn, config=self._config)
            frames = list(pf)
            if self._frame_index >= len(frames):
                logger.warning("Frame %d not in %s (has %d frames)", self._frame_index, fn, len(frames))
                return None

            frame = frames[self._frame_index]
            # Save raw intensity before pipeline processing
            raw_intensity = frame.intensity.astype(np.float32)
            # Use qSave=True to keep intermediate results for visualization
            fp = self._FramePipeline(frame, config=self._config, qSave=True, qTiming=True)

            # Manage cache size
            if len(self._cache) >= self._cache_size:
                oldest = min(self._cache.keys())
                del self._cache[oldest]

            self._cache[idx] = (fp, raw_intensity)
            return (fp, raw_intensity)

        except Exception:
            logger.exception("Failed to load %s", fn)
            return None

    def _update_display(self) -> None:
        """Update the display with the current frame's pipeline stages."""
        result = self._load_pipeline(self._current_idx)
        if result is None:
            return

        fp, raw_intensity = result

        # Clear all axes
        for ax in self._axes.values():
            ax.clear()

        # Get data for each stage
        metadata = fp.metadata
        n_bearings, n_distances = fp.shape

        # Common colormap settings
        cmap = "viridis"

        # 1. Raw Intensity (before any processing) - top-left: ylabel only
        ax = self._axes["Raw Intensity"]
        ax.imshow(raw_intensity, aspect="auto", cmap=cmap, origin="lower")
        ax.set_ylabel("Bearing")
        ax.set_title("Raw Intensity", fontsize=10)

        # 2. Destreaked - top row, no labels
        ax = self._axes["Destreaked"]
        if fp.destreak is not None:
            ax.imshow(fp.destreak.intensity, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title("Destreaked", fontsize=10)

        # 3. Shadow Mask - top row, no labels
        ax = self._axes["Shadow Mask"]
        if fp.shadow is not None:
            mask = np.zeros((n_bearings, n_distances), dtype=np.float32)
            for start, end in fp.shadow.indices:
                mask[start:end, :] = 1.0
            ax.imshow(mask, aspect="auto", cmap="Reds", origin="lower", vmin=0, vmax=1)
            if fp.shadow.theta_bias is not None:
                ax.text(
                    0.02, 0.98, f"bias: {fp.shadow.theta_bias:.1f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
        ax.set_title("Shadow Mask", fontsize=10)

        # 4. Theta (beam angles) - top-right, separate axes
        ax = self._axes["Theta (beam angle)"]
        if fp.theta is not None:
            theta = fp.theta.theta
            ax.plot(theta, "b-", linewidth=0.5)
            ax.set_ylabel("Angle", labelpad=-1)
            ax.yaxis.set_label_coords(-0.08, 0.5)
            ax.set_ylim(0, 360)
            ax.grid(True, alpha=0.3)
            # Compute step sizes (handle wrap-around at 360)
            dtheta = np.diff(theta)
            dtheta = np.where(dtheta < -180, dtheta + 360, dtheta)
            dtheta = np.where(dtheta > 180, dtheta - 360, dtheta)
            mean_step = np.mean(dtheta)
            median_step = np.median(dtheta)
            ax.text(
                0.02, 0.98,
                f"$\\delta\\theta$\nmean: {mean_step:.3f}\u00b0\nmedian: {median_step:.3f}\u00b0",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
        ax.set_title("Theta", fontsize=10)

        # 5. Shadow Applied - middle row left: ylabel only
        ax = self._axes["Shadow Applied"]
        if fp.intensity_shadowed is not None:
            ax.imshow(fp.intensity_shadowed, aspect="auto", cmap=cmap, origin="lower")
        ax.set_ylabel("Bearing")
        ax.set_title("Shadow Applied", fontsize=10)

        # 6. Deramped - middle row
        ax = self._axes["Deramped"]
        if fp.deramp is not None:
            ax.imshow(fp.deramp.intensity, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title("Deramped", fontsize=10)

        # 7. Dewinded - middle row
        ax = self._axes["Dewinded"]
        if fp.dewind is not None:
            ax.imshow(fp.dewind.intensity, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title("Dewinded", fontsize=10)

        # 8. Final Normalized - middle row
        ax = self._axes["Final (Normalized)"]
        if fp.final_intensity is not None:
            ax.imshow(fp.final_intensity, aspect="auto", cmap=cmap, origin="lower")
        ax.set_title("Final [0,1]", fontsize=10)

        # Difference colormap (diverging, centered on zero)
        diff_cmap = "RdBu_r"

        # 9. Destreaked - Raw - bottom left: both labels
        ax = self._axes["Destreaked - Raw"]
        if fp.destreak is not None:
            diff = fp.destreak.intensity - raw_intensity
            vlim = np.nanmax(np.abs(diff))
            ax.imshow(diff, aspect="auto", cmap=diff_cmap, origin="lower", vmin=-vlim, vmax=vlim)
        ax.set_xlabel("Range bin")
        ax.set_ylabel("Bearing")
        ax.set_title("Destreaked - Raw", fontsize=10)

        # 10. Shadow Applied - Destreaked - bottom row: xlabel only
        ax = self._axes["Shadow - Destreaked"]
        if fp.intensity_shadowed is not None and fp.destreak is not None:
            diff = fp.intensity_shadowed - fp.destreak.intensity
            vlim = np.nanmax(np.abs(diff[~np.isnan(diff)])) if np.any(~np.isnan(diff)) else 1
            ax.imshow(diff, aspect="auto", cmap=diff_cmap, origin="lower", vmin=-vlim, vmax=vlim)
        ax.set_xlabel("Range bin")
        ax.set_title("Shadow - Destreaked", fontsize=10)

        # 11. Deramped - Shadow Applied - bottom row: xlabel only
        ax = self._axes["Deramped - Shadow"]
        if fp.deramp is not None and fp.intensity_shadowed is not None:
            diff = fp.deramp.intensity - fp.intensity_shadowed
            vlim = np.nanmax(np.abs(diff[~np.isnan(diff)])) if np.any(~np.isnan(diff)) else 1
            ax.imshow(diff, aspect="auto", cmap=diff_cmap, origin="lower", vmin=-vlim, vmax=vlim)
        ax.set_xlabel("Range bin")
        ax.set_title("Deramped - Shadow", fontsize=10)

        # 12. Dewinded - Deramped - bottom row: xlabel only
        ax = self._axes["Dewinded - Deramped"]
        if fp.dewind is not None and fp.deramp is not None:
            diff = fp.dewind.intensity - fp.deramp.intensity
            vlim = np.nanmax(np.abs(diff[~np.isnan(diff)])) if np.any(~np.isnan(diff)) else 1
            ax.imshow(diff, aspect="auto", cmap=diff_cmap, origin="lower", vmin=-vlim, vmax=vlim)
        ax.set_xlabel("Range bin")
        ax.set_title("Dewinded - Deramped", fontsize=10)

        # Update main title with wind and ship info
        wind_parts = []
        if metadata.wind_speed is not None:
            wind_parts.append(f"Wind: {metadata.wind_speed:.1f} m/s")
        if metadata.wind_direction is not None:
            wind_parts.append(f"{metadata.wind_direction:.0f}\u00b0")
        wind_str = " @ ".join(wind_parts) if wind_parts else "Wind: N/A"

        ship_parts = []
        if metadata.ship_speed is not None:
            ship_parts.append(f"Ship: {metadata.ship_speed:.1f} m/s")
        if metadata.ship_course is not None:
            ship_parts.append(f"{metadata.ship_course:.0f}\u00b0")
        ship_str = " @ ".join(ship_parts) if ship_parts else "Ship: N/A"

        self._fig.suptitle(
            f"Frame {self._current_idx + 1}/{len(self._files)}: {metadata.timestamp}\n"
            f"{wind_str} | {ship_str}",
            fontsize=10,
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
            self._on_play()  # Toggle back to paused

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

        plt.show()


class PolarDiagnosticViewer:
    """
    Interactive viewer for polar plots of pipeline stages.

    Displays a 2x2 grid of polar plots showing:
    - Raw Intensity, Destreaked, Deramped, Dewinded

    Navigation: Arrow keys, Space for play/pause, or Prev/Play/Next buttons.
    """

    def __init__(
        self,
        files: list[str],
        config: Config,
        frame_index: int = 0,
        interval: int = 500,
    ) -> None:
        """
        Initialize the polar diagnostic viewer.

        Args:
            files: List of polar file paths
            config: Configuration object
            frame_index: Which frame to extract from each file
            interval: Animation interval in milliseconds
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        from wamos_tpw.frame_pipeline import FramePipeline
        from wamos_tpw.polarfile import PolarFile

        # Suppress "Mean of empty slice" warnings from shadow-masked regions
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

        self._files = files
        self._config = config
        self._frame_index = frame_index
        self._current_idx = 0
        self._playing = False
        self._timer = None
        self._interval = interval
        self._cache: dict[int, tuple[FramePipeline, np.ndarray]] = {}
        self._cache_size = 5
        self._PolarFile = PolarFile
        self._FramePipeline = FramePipeline

        # Create figure with 2x2 grid of polar plots + buttons
        self._fig = plt.figure(figsize=(12, 12))
        gs = self._fig.add_gridspec(
            3, 2,
            height_ratios=[1, 1, 0.04],
            hspace=0.12,
            wspace=0.15,
            top=0.92,
            bottom=0.06,
            left=0.04,
            right=0.96,
        )

        # Polar plots (2 rows x 2 columns)
        self._axes = {}
        stage_names = [
            ("Raw Intensity", 0, 0),
            ("Destreaked", 0, 1),
            ("Deramped", 1, 0),
            ("Dewinded", 1, 1),
        ]
        for name, row, col in stage_names:
            ax = self._fig.add_subplot(gs[row, col], projection="polar")
            if row == 0:
                ax.set_title(name, fontsize=10)
            else:
                ax.set_xlabel(name, fontsize=10, labelpad=10)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            self._axes[name] = ax

        # Navigation button axes (bottom row)
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        btn_spacing = 0.02

        total_width = 3 * btn_width + 2 * btn_spacing
        start_x = 0.5 - total_width / 2

        ax_prev = self._fig.add_axes([start_x, btn_y, btn_width, btn_height])
        ax_play = self._fig.add_axes(
            [start_x + btn_width + btn_spacing, btn_y, btn_width, btn_height]
        )
        ax_next = self._fig.add_axes(
            [start_x + 2 * (btn_width + btn_spacing), btn_y, btn_width, btn_height]
        )

        self._btn_prev = Button(ax_prev, "< Prev")
        self._btn_play = Button(ax_play, "> Play")
        self._btn_next = Button(ax_next, "Next >")

        self._btn_prev.on_clicked(self._on_prev)
        self._btn_play.on_clicked(self._on_play)
        self._btn_next.on_clicked(self._on_next)

        # Connect keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Load and display first frame
        self._update_display()

    def _load_pipeline(self, idx: int) -> tuple[FramePipeline, np.ndarray] | None:
        """Load and process a single frame through the pipeline."""
        if idx in self._cache:
            return self._cache[idx]

        fn = self._files[idx]
        try:
            pf = self._PolarFile(fn, config=self._config)
            frames = list(pf)
            if self._frame_index >= len(frames):
                logger.warning("Frame %d not in %s (has %d frames)", self._frame_index, fn, len(frames))
                return None

            frame = frames[self._frame_index]
            # Save raw intensity before pipeline processing
            raw_intensity = frame.intensity.astype(np.float32)
            # Use qSave=True to keep intermediate results for visualization
            fp = self._FramePipeline(frame, config=self._config, qSave=True, qTiming=False)

            # Manage cache size
            if len(self._cache) >= self._cache_size:
                oldest = min(self._cache.keys())
                del self._cache[oldest]

            self._cache[idx] = (fp, raw_intensity)
            return (fp, raw_intensity)

        except Exception:
            logger.exception("Failed to load %s", fn)
            return None

    def _plot_polar(self, ax, intensity: np.ndarray, theta: np.ndarray, ground_range: np.ndarray) -> None:
        """Plot intensity data in polar coordinates."""
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Sort by theta for proper polar display
        sort_idx = np.argsort(theta)
        theta_sorted = np.deg2rad(theta[sort_idx])
        intensity_sorted = intensity[sort_idx, :]

        # Create mesh grids - close the circle by appending first row
        theta_mesh, range_mesh = np.meshgrid(
            np.append(theta_sorted, theta_sorted[0] + 2 * np.pi),
            ground_range,
            indexing="ij",
        )
        intensity_closed = np.vstack([intensity_sorted, intensity_sorted[0:1, :]])

        ax.pcolormesh(
            theta_mesh,
            range_mesh,
            intensity_closed,
            shading="auto",
            cmap="viridis",
        )

    def _update_display(self) -> None:
        """Update the display with the current frame's polar plots."""
        result = self._load_pipeline(self._current_idx)
        if result is None:
            return

        fp, raw_intensity = result
        metadata = fp.metadata
        theta = fp.theta_array
        ground_range = fp.ground_range

        # All 8 standard tick positions, with selective label blanking
        all_ticks = [0, 45, 90, 135, 180, 225, 270, 315]
        all_labels = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
        top_labels = [("" if a == 180 else lbl) for a, lbl in zip(all_ticks, all_labels)]
        bottom_labels = [("" if a == 0 else lbl) for a, lbl in zip(all_ticks, all_labels)]

        # 1. Raw Intensity (top row - title on top)
        ax = self._axes["Raw Intensity"]
        self._plot_polar(ax, raw_intensity, theta, ground_range)
        ax.set_thetagrids(all_ticks, top_labels)
        ax.set_title("Raw Intensity", fontsize=10)

        # 2. Destreaked (top row - title on top)
        ax = self._axes["Destreaked"]
        if fp.destreak is not None:
            self._plot_polar(ax, fp.destreak.intensity, theta, ground_range)
        ax.set_thetagrids(all_ticks, top_labels)
        ax.set_title("Destreaked", fontsize=10)

        # 3. Deramped (bottom row - title on bottom to avoid overlap)
        ax = self._axes["Deramped"]
        if fp.deramp is not None:
            self._plot_polar(ax, fp.deramp.intensity, theta, ground_range)
        ax.set_thetagrids(all_ticks, bottom_labels)
        ax.set_xlabel("Deramped", fontsize=10, labelpad=10)

        # 4. Dewinded (bottom row - title on bottom to avoid overlap)
        ax = self._axes["Dewinded"]
        if fp.dewind is not None:
            self._plot_polar(ax, fp.dewind.intensity, theta, ground_range)
        ax.set_thetagrids(all_ticks, bottom_labels)
        ax.set_xlabel("Dewinded", fontsize=10, labelpad=10)

        # Update main title with wind and ship info
        wind_parts = []
        if metadata.wind_speed is not None:
            wind_parts.append(f"Wind: {metadata.wind_speed:.1f} m/s")
        if metadata.wind_direction is not None:
            wind_parts.append(f"{metadata.wind_direction:.0f}\u00b0")
        wind_str = " @ ".join(wind_parts) if wind_parts else "Wind: N/A"

        ship_parts = []
        if metadata.ship_speed is not None:
            ship_parts.append(f"Ship: {metadata.ship_speed:.1f} m/s")
        if metadata.ship_course is not None:
            ship_parts.append(f"{metadata.ship_course:.0f}\u00b0")
        ship_str = " @ ".join(ship_parts) if ship_parts else "Ship: N/A"

        self._fig.suptitle(
            f"Frame {self._current_idx + 1}/{len(self._files)}: {metadata.timestamp}\n"
            f"{wind_str} | {ship_str}",
            fontsize=10,
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
            self._on_play()  # Toggle back to paused

    def _on_key(self, event) -> None:
        """Handle keyboard events."""
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

        plt.show()


def run_plot_mode(files: list[str], config: Config, frame_index: int) -> None:
    """Run interactive plot mode showing pipeline stages for each frame."""
    viewer = PipelineDiagnosticViewer(files, config, frame_index)
    viewer.show()


def run_polar_mode(files: list[str], config: Config, frame_index: int) -> None:
    """Run interactive polar plot mode showing Raw, Destreaked, Deramped, Dewinded."""
    viewer = PolarDiagnosticViewer(files, config, frame_index)
    viewer.show()
