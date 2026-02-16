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
            4,
            4,
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
                logger.warning(
                    "Frame %d not in %s (has %d frames)", self._frame_index, fn, len(frames)
                )
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
                    0.02,
                    0.98,
                    f"bias: {fp.shadow.theta_bias:.1f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
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
                0.02,
                0.98,
                f"$\\delta\\theta$\nmean: {mean_step:.3f}\u00b0\nmedian: {median_step:.3f}\u00b0",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
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
        ax = self._axes["Final (Processed)"]
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
            3,
            2,
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
                logger.warning(
                    "Frame %d not in %s (has %d frames)", self._frame_index, fn, len(frames)
                )
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

    def _plot_polar(
        self, ax, intensity: np.ndarray, theta: np.ndarray, ground_range: np.ndarray
    ) -> None:
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
        top_labels = [
            ("" if a == 180 else lbl) for a, lbl in zip(all_ticks, all_labels, strict=False)
        ]
        bottom_labels = [
            ("" if a == 0 else lbl) for a, lbl in zip(all_ticks, all_labels, strict=False)
        ]

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


class FinalIntensityViewer:
    """
    Interactive viewer for pipeline-processed intensity with polar/ship/earth views.

    Displays the final processed intensity (destreaked, shadow-masked, deramped,
    dewinded) in a single plot with switchable coordinate views:
    - Polar: bearing vs ground distance
    - Ship: ship-relative x/y (+X=starboard, +Y=bow)
    - Earth: earth-referenced x/y (+X=East, +Y=North)

    Frames are loaded lazily on demand and cached (LRU, default 5 frames).

    Navigation: Arrow keys or Prev/Play/Next buttons.
    Views: 1=Polar, 2=Ship, 3=Earth keys.
    """

    def __init__(
        self,
        files: list[str],
        config: Config,
        frame_index: int = 0,
        view: str = "polar",
        ship_data_dir: str | None = None,
        radar_height: float | None = None,
    ) -> None:
        from pathlib import Path

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        from wamos_tpw.bearing import (
            heading_to_xy,
            theta_to_heading_earth,
            theta_to_heading_ship,
        )
        from wamos_tpw.frame_pipeline import FramePipeline
        from wamos_tpw.plotting import (
            add_crosshairs,
            add_range_rings,
            calc_bin_edges,
            quantile_limits,
            sort_polar_data,
        )
        from wamos_tpw.polarfile import PolarFile

        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

        self._files = files
        self._config = config
        self._frame_index = frame_index
        self._view = view
        self._radar_height = radar_height
        self._current_idx = 0
        self._playing = False
        self._timer = None

        # Store class/function references for lazy loading
        self._PolarFile = PolarFile
        self._FramePipeline = FramePipeline
        self._theta_to_heading_ship = theta_to_heading_ship
        self._theta_to_heading_earth = theta_to_heading_earth
        self._heading_to_xy = heading_to_xy
        self._sort_polar_data = sort_polar_data
        self._calc_bin_edges = calc_bin_edges
        self._quantile_limits = quantile_limits
        self._add_crosshairs = add_crosshairs
        self._add_range_rings = add_range_rings

        # Get config offsets
        self._bow_to_radar = config.get("offsets.bow_to_radar", 0.0)
        self._heading_delay = config.get("offsets.heading_delay", 0.0)
        self._compass_offset = config.get("offsets.compass", 0.0)

        # Load ship data if requested
        self._ship_data = None
        if ship_data_dir is not None:
            from wamos_tpw.instruments.ship_data import ShipData

            self._ship_data = ShipData(Path(ship_data_dir))
            logger.info("Loaded ship data: %s", self._ship_data)

        # Lazy-loading cache: idx -> (FramePipeline, per_radial_heading | None)
        self._cache: dict[int, tuple[FramePipeline, np.ndarray | None]] = {}
        self._cache_size = 5

        # Create figure
        self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 8))
        self._fig.subplots_adjust(bottom=0.12)
        self._cbar = None
        self._im = None

        # Navigation buttons
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.02
        btn_spacing = 0.02
        total_width = 3 * btn_width + 2 * btn_spacing
        start_x = 0.15

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

        # View buttons
        view_start_x = start_x + total_width + 0.06
        view_width = 0.07
        view_spacing = 0.015

        ax_polar = self._fig.add_axes([view_start_x, btn_y, view_width, btn_height])
        ax_ship = self._fig.add_axes(
            [view_start_x + view_width + view_spacing, btn_y, view_width, btn_height]
        )
        ax_earth = self._fig.add_axes(
            [view_start_x + 2 * (view_width + view_spacing), btn_y, view_width, btn_height]
        )

        self._btn_polar = Button(ax_polar, "1 Polar")
        self._btn_ship = Button(ax_ship, "2 Ship")
        self._btn_earth = Button(ax_earth, "3 Earth")

        self._btn_polar.on_clicked(lambda e: self._set_view("polar"))
        self._btn_ship.on_clicked(lambda e: self._set_view("ship"))
        self._btn_earth.on_clicked(lambda e: self._set_view("earth"))

        # Keyboard
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Draw initial frame
        self._update_display()

    def _load_pipeline(self, idx: int):
        """Load and process a single frame through the pipeline (lazy, cached)."""
        if idx in self._cache:
            return self._cache[idx]

        fn = self._files[idx]
        try:
            pf = self._PolarFile(fn, config=self._config)
            frames = list(pf)
            if self._frame_index >= len(frames):
                logger.warning(
                    "Frame %d not in %s (has %d frames)", self._frame_index, fn, len(frames)
                )
                return None

            frame = frames[self._frame_index]
            fp = self._FramePipeline(frame, config=self._config, qSave=False, qTiming=False)

            # Compute per-radial heading from ship data if available.
            # The metadata timestamp is the END of the frame, so radial
            # times run from (timestamp - repeat_time) to timestamp.
            per_radial = None
            if self._ship_data is not None:
                repeat_time = fp.metadata.repeat_time or 1.43
                n_bearings = fp.n_bearings
                frame_duration = np.timedelta64(int(repeat_time * 1e9), "ns")
                offsets_ns = (np.arange(n_bearings) * repeat_time / n_bearings * 1e9).astype(
                    "timedelta64[ns]"
                )
                times = fp.metadata.timestamp - frame_duration + offsets_ns
                per_radial = self._ship_data.interpolate(times, "heading")

            # Manage cache size — evict oldest entry
            if len(self._cache) >= self._cache_size:
                oldest = min(self._cache.keys())
                del self._cache[oldest]

            self._cache[idx] = (fp, per_radial)
            return (fp, per_radial)

        except Exception:
            logger.exception("Failed to load %s", fn)
            return None

    def _set_view(self, view: str) -> None:
        if view != self._view:
            self._view = view
            self._update_display()

    def _update_display(self) -> None:
        """Update the display with the current frame."""
        result = self._load_pipeline(self._current_idx)
        if result is None:
            return

        fp, per_radial = result

        self._ax.clear()

        intensity = fp.final_intensity
        theta = fp.theta_array
        ground_range = fp.ground_range
        metadata = fp.metadata

        vmin, vmax = self._quantile_limits(intensity)

        if self._view == "polar":
            sorted_bearing, sorted_data = self._sort_polar_data(theta, intensity)
            dist_edges = self._calc_bin_edges(ground_range)
            bearing_edges = self._calc_bin_edges(sorted_bearing)
            self._im = self._ax.pcolormesh(
                dist_edges,
                bearing_edges,
                sorted_data,
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
                shading="flat",
            )
            self._ax.set_xlabel("Ground Distance (m)")
            self._ax.set_ylabel("Bearing (\u00b0)")
            self._ax.set_aspect("auto")

        elif self._view == "ship":
            heading = self._theta_to_heading_ship(theta, self._bow_to_radar)
            x, y = self._heading_to_xy(heading, ground_range)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input coordinates.*pcolormesh.*")
                self._im = self._ax.pcolormesh(
                    x,
                    y,
                    intensity,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    shading="nearest",
                )
            self._ax.set_xlabel("X - Starboard (m)")
            self._ax.set_ylabel("Y - Bow (m)")
            self._ax.set_aspect("equal")
            self._add_crosshairs(self._ax)
            max_range = np.sqrt(x**2 + y**2).max()
            self._add_range_rings(self._ax, max_range, interval=1000.0)

        elif self._view == "earth":
            if per_radial is not None:
                # Per-radial heading from ship data
                heading = self._theta_to_heading_earth(
                    theta,
                    per_radial,
                    bow_to_radar=self._bow_to_radar,
                    heading_delay=self._heading_delay,
                    compass_offset=self._compass_offset,
                )
            else:
                # Scalar heading from metadata
                ship_heading = metadata.heading or 0.0
                heading = self._theta_to_heading_earth(
                    theta,
                    ship_heading,
                    bow_to_radar=self._bow_to_radar,
                    heading_delay=self._heading_delay,
                    compass_offset=self._compass_offset,
                )
            x, y = self._heading_to_xy(heading, ground_range)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*input coordinates.*pcolormesh.*")
                self._im = self._ax.pcolormesh(
                    x,
                    y,
                    intensity,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                    shading="nearest",
                )
            self._ax.set_xlabel("X - East (m)")
            self._ax.set_ylabel("Y - North (m)")
            self._ax.set_aspect("equal")
            self._add_crosshairs(self._ax)
            max_range = np.sqrt(x**2 + y**2).max()
            self._add_range_rings(self._ax, max_range, interval=1000.0)

        # Title
        heading_str = f"Hdg: {metadata.heading:.1f}\u00b0" if metadata.heading is not None else ""
        speed_str = f"Spd: {metadata.ship_speed:.1f} m/s" if metadata.ship_speed is not None else ""
        ship_info = ", ".join(s for s in [heading_str, speed_str] if s)
        ship_data_note = (
            " [per-radial]" if (self._view == "earth" and per_radial is not None) else ""
        )

        self._fig.suptitle(
            f"Frame {self._current_idx + 1}/{len(self._files)}: "
            f"{metadata.timestamp}  [{self._view}]{ship_data_note}\n"
            f"{ship_info}",
            fontsize=10,
            fontweight="bold",
        )

        # Colorbar
        if self._cbar is None and self._im is not None:
            self._cbar = self._fig.colorbar(self._im, ax=self._ax, label="Intensity")
        elif self._cbar is not None and self._im is not None:
            self._cbar.mappable = self._im
            self._im.set_clim(vmin, vmax)

        self._fig.canvas.draw_idle()

    def _on_prev(self, event=None) -> None:
        if self._current_idx > 0:
            self._current_idx -= 1
        else:
            self._current_idx = len(self._files) - 1
        self._update_display()

    def _on_next(self, event=None) -> None:
        if self._current_idx < len(self._files) - 1:
            self._current_idx += 1
        else:
            self._current_idx = 0
        self._update_display()

    def _on_play(self, event=None) -> None:
        self._playing = not self._playing
        if self._playing:
            self._btn_play.label.set_text("|| Pause")
            self._timer = self._fig.canvas.new_timer(interval=500)
            self._timer.add_callback(self._animate_step)
            self._timer.start()
        else:
            self._btn_play.label.set_text("> Play")
            if self._timer:
                self._timer.stop()
                self._timer = None
        self._fig.canvas.draw_idle()

    def _animate_step(self) -> None:
        if self._current_idx < len(self._files) - 1:
            self._current_idx += 1
            self._update_display()
        else:
            self._on_play()

    def _on_key(self, event) -> None:
        import matplotlib.pyplot as plt

        if event.key in ("left", "p", "b"):
            self._on_prev()
        elif event.key in ("right", "n", "f"):
            self._on_next()
        elif event.key == " ":
            self._on_play()
        elif event.key == "1":
            self._set_view("polar")
        elif event.key == "2":
            self._set_view("ship")
        elif event.key == "3":
            self._set_view("earth")
        elif event.key == "q":
            plt.close(self._fig)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def run_view_mode(
    files: list[str],
    config: Config,
    frame_index: int,
    view: str,
    ship_data_dir: str | None = None,
    radar_height: float | None = None,
) -> None:
    """Run interactive view mode showing final processed intensity."""
    viewer = FinalIntensityViewer(
        files,
        config,
        frame_index,
        view,
        ship_data_dir,
        radar_height,
    )
    viewer.show()
