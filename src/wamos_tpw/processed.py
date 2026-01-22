#! /usr/bin/env python3
#
# ProcessedFrames class for loading and processing WAMOS polar frames
# Extends Files with processing capabilities and plotting.
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from tqdm import tqdm

from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.config import Config
from wamos_tpw.deramp import Deramp
from wamos_tpw.destreak import Destreak
from wamos_tpw.files import Files
from wamos_tpw.frame import Frame
from wamos_tpw.plotting import BaseViewer, quantile_limits, format_nav_title


class ProcessedFrames(Files):
    """
    Load and process WAMOS polar files in time-based groups.

    Extends Files with processing capabilities for radar intensity data.

    Example:
        >>> with ProcessedFrames(
        ...     stime='20241215100000',
        ...     etime='20241215120000',
        ...     polar_path='/data/wamos',
        ...     groupby='30m',
        ... ) as pframes:
        ...     for period, frames in pframes.itergroups():
        ...         process(frames)
    """

    def __init__(
        self,
        stime: str | np.datetime64,
        etime: str | np.datetime64,
        polar_path: str,
        groupby: str = "h",
        workers: int | None = None,
        loader: Callable[[str], Frame | None] | None = None,
        config: Config | None = None,
        radar_height: float | None = None,
    ) -> None:
        """
        Initialize ProcessedFrames.

        Args:
            stime: Start time (string or np.datetime64)
            etime: End time (string or np.datetime64)
            polar_path: Base path to polar file directory
            groupby: Time grouping frequency (e.g., 'h', '30m', 'D')
            workers: Number of parallel workers (None = auto, 1 = sequential)
            loader: Optional custom file loader function
            config: Config for processing parameters
            radar_height: Height of radar above water in meters
        """
        super().__init__(stime, etime, polar_path, groupby, workers, loader)
        self._config = config or Config()
        self._radar_height = radar_height

    @property
    def config(self) -> Config:
        """Return the configuration."""
        return self._config

    @property
    def radar_height(self) -> float | None:
        """Return the radar height."""
        return self._radar_height

    def refine_theta(self, frames: list[Frame], shadow_diagnostics: bool = False) -> Theta:
        """
        Refine the angle with respect to the vessel by using the shadowed regions.

        The detected shadow edges are stored and can be used for deramping.

        Args:
            frames: List of Frame objects to optimize the theta for
            shadow_diagnostics: If True, show shadow detection diagnostic plots and stop

        Returns:
            Theta object with detected shadow edges (access via shadow_left_mean, shadow_right_mean)
        """
        # Create Theta to get shadow analysis (refinement does the edge detection)
        theta = Theta(frames, self._config, refine=True)

        # Log shadow statistics
        if theta.shadow_stats:
            logging.debug(f"Shadow {theta.shadow_stats}")

        if shadow_diagnostics:
            theta.plot_shadow_diagnostics()
            raise RuntimeError("Shadow diagnostic plots complete - stopping execution")

        return theta

    def deramp_frames(
        self,
        frames: list[Frame],
        diagnostics: bool = False,
        show_progress: bool = True,
        parallel: bool = True,
    ) -> None:
        """
        Remove range-dependent intensity fall-off from frames.

        Applies empirical range profile normalization to correct for
        radar signal attenuation with distance. Stores the deramped
        intensity as frame.deramped_intensity for use by destreak.

        Args:
            frames: List of Frame objects to deramp
            diagnostics: If True, show diagnostic plots for each frame
            show_progress: If True, show progress bar
            parallel: If True, process frames in parallel using threads
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        from wamos_tpw.range import Range

        n_frames = len(frames)

        # Diagnostics require sequential processing
        if diagnostics or not parallel or n_frames < 4:
            frame_iter = tqdm(frames, desc="Deramping", disable=not show_progress)
            for frame in frame_iter:
                intensity = frame.intensity.astype(np.float32)
                rng = Range(frame)
                deramp = Deramp(intensity, rng)
                frame.deramped_intensity = deramp.intensity
            return

        # Parallel processing using threads (numpy/scipy release GIL)
        def process_frame(args):
            i, frame = args
            intensity = frame.intensity.astype(np.float32)
            rng = Range(frame)
            deramp = Deramp(intensity, rng)
            return i, deramp.intensity

        n_workers = min(os.cpu_count() or 4, n_frames)
        results = [None] * n_frames

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_frame, (i, f)): i for i, f in enumerate(frames)}

            with tqdm(total=n_frames, desc="Deramping", disable=not show_progress) as pbar:
                for future in as_completed(futures):
                    i, corrected = future.result()
                    results[i] = corrected
                    pbar.update(1)

        # Assign results to frames
        for i, frame in enumerate(frames):
            frame.deramped_intensity = results[i]

    def destreak_frames(
        self,
        frames: list[Frame],
        diagnostics: bool = False,
        show_progress: bool = True,
        parallel: bool = True,
    ) -> list[np.ndarray]:
        """
        Remove radial streak artifacts from a list of frames.

        Args:
            frames: List of Frame objects to destreak
            diagnostics: If True, show diagnostic plots for each frame
            show_progress: If True, show progress bar
            parallel: If True, process frames in parallel using threads

        Returns:
            List of corrected intensity arrays (same order as input frames)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        n_frames = len(frames)

        # Diagnostics require sequential processing
        if diagnostics or not parallel or n_frames < 4:
            corrected = []
            frame_iter = tqdm(
                enumerate(frames), total=n_frames, desc="Destreaking", disable=not show_progress
            )
            for i, frame in frame_iter:
                ds = Destreak(frame)
                corrected.append(ds.intensity)
            return corrected

        # Parallel processing using threads (numpy/scipy release GIL)
        def process_frame(args):
            i, frame = args
            ds = Destreak(frame)
            return i, ds.intensity

        n_workers = min(os.cpu_count() or 4, n_frames)
        results = [None] * n_frames

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_frame, (i, f)): i for i, f in enumerate(frames)}

            with tqdm(total=n_frames, desc="Destreaking", disable=not show_progress) as pbar:
                for future in as_completed(futures):
                    i, corrected = future.result()
                    results[i] = corrected
                    pbar.update(1)

        return results

    def normalize_frames(
        self,
        corrected: list[np.ndarray],
        low_percentile: float = 2.0,
        high_percentile: float = 98.0,
        parallel: bool = True,
        show_progress: bool = False,
    ) -> list[np.ndarray]:
        """
        Normalize frames using global statistics for consistent distribution.

        Uses percentile-based normalization across all frames to ensure
        the intensity distribution is approximately the same across frames.
        Values below low_percentile map to 0, above high_percentile map to 1.

        Args:
            corrected: List of intensity arrays to normalize
            low_percentile: Lower percentile for normalization (default: 2.0)
            high_percentile: Upper percentile for normalization (default: 98.0)
            parallel: If True, process frames in parallel using threads
            show_progress: If True, show progress bar

        Returns:
            List of normalized intensity arrays in [0, 1]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        if not corrected:
            return []

        n_frames = len(corrected)

        # Compute global percentiles using sampling for efficiency
        # Sample up to 1M values total across all frames
        max_samples = 1_000_000
        samples_per_frame = max(1000, max_samples // n_frames)

        sampled_values = []
        for frame in corrected:
            flat = frame.ravel()
            valid = flat[~np.isnan(flat)]
            if len(valid) > samples_per_frame:
                # Random sample without replacement
                idx = np.random.choice(len(valid), samples_per_frame, replace=False)
                sampled_values.append(valid[idx])
            elif len(valid) > 0:
                sampled_values.append(valid)

        if not sampled_values:
            return [np.full_like(frame, 0.5) for frame in corrected]

        all_samples = np.concatenate(sampled_values)

        # Use partition-based percentile for O(n) performance
        low_val = self._fast_percentile(all_samples, low_percentile)
        high_val = self._fast_percentile(all_samples, high_percentile)

        logging.debug(
            f"Global normalization: p{low_percentile}={low_val:.2f}, "
            f"p{high_percentile}={high_val:.2f}"
        )

        if high_val <= low_val:
            return [np.full_like(frame, 0.5) for frame in corrected]

        # Normalize frames
        scale = 1.0 / (high_val - low_val)

        if not parallel or n_frames < 4:
            # Sequential processing
            normalized = []
            frame_iter = tqdm(corrected, desc="Normalizing", disable=not show_progress)
            for intensity in frame_iter:
                norm = np.clip((intensity - low_val) * scale, 0.0, 1.0)
                normalized.append(norm)
            return normalized

        # Parallel processing
        def normalize_frame(args):
            i, intensity = args
            norm = np.clip((intensity - low_val) * scale, 0.0, 1.0)
            return i, norm

        n_workers = min(os.cpu_count() or 4, n_frames)
        results = [None] * n_frames

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(normalize_frame, (i, f)): i for i, f in enumerate(corrected)}

            with tqdm(total=n_frames, desc="Normalizing", disable=not show_progress) as pbar:
                for future in as_completed(futures):
                    i, norm = future.result()
                    results[i] = norm
                    pbar.update(1)

        return results

    @staticmethod
    def _fast_percentile(data: np.ndarray, percentile: float) -> float:
        """
        Compute percentile using partition for O(n) performance.

        Args:
            data: 1D array of values
            percentile: Percentile value (0-100)

        Returns:
            Percentile value
        """
        n = len(data)
        if n == 0:
            return 0.0

        k = int((percentile / 100.0) * (n - 1))
        k = max(0, min(k, n - 1))

        # Partition to get k-th smallest element
        partitioned = np.partition(data, k)
        return partitioned[k]

    def process_group(
        self,
        frames: list[Frame],
        shadow_diagnostics: bool = False,
        deramp_diagnostics: bool = False,
        destreak_diagnostics: bool = False,
    ) -> list[np.ndarray]:
        """
        Process a single group of frames: refine theta, deramp, destreak, normalize.

        The shadow edges detected during theta refinement are propagated to the
        deramp phase for accurate shadow region masking.

        Args:
            frames: List of Frame objects to process
            shadow_diagnostics: If True, show shadow detection plots
            deramp_diagnostics: If True, show deramp plots
            destreak_diagnostics: If True, show destreak plots

        Returns:
            List of normalized intensity arrays in [0, 1]
        """
        frames = list(frames)  # Materialize generator if needed

        # Refine theta for this group - returns Theta with detected shadow edges
        theta = self.refine_theta(frames, shadow_diagnostics=shadow_diagnostics)

        # Extract detected shadow edges from theta (if available)
        shadow_start = theta.shadow_left_mean
        shadow_end = theta.shadow_right_mean

        if shadow_start is not None and shadow_end is not None:
            logging.info(f"Using detected shadow region: {shadow_start:.1f}° - {shadow_end:.1f}°")
        else:
            logging.info("Using config-based shadow region (detection failed)")

        # Deramp frames
        self.deramp_frames(
            frames,
            diagnostics=deramp_diagnostics,
        )

        # Destreak frames (uses deramped_intensity if available)
        corrected = self.destreak_frames(frames, diagnostics=destreak_diagnostics)

        # Normalize each frame to [0, 1]
        normalized = self.normalize_frames(corrected)

        return normalized

    def process(
        self,
        shadow_diagnostics: bool = False,
        deramp_diagnostics: bool = False,
        destreak_diagnostics: bool = False,
        parallel: bool = True,
    ) -> dict:
        """
        Process all frame groups: refine theta, deramp, destreak, and normalize.

        For each itergroup, this method:
        1. Calls refine_theta() to align bearing using shadow regions
        2. Calls deramp_frames() to remove range-dependent intensity fall-off
        3. Calls destreak_frames() to remove radial streak artifacts
        4. Calls normalize_frames() to scale each frame to [0, 1]

        When diagnostics are disabled and parallel=True, groups are processed
        concurrently using multiple workers.

        Args:
            shadow_diagnostics: If True, show shadow detection plots (disables parallel)
            deramp_diagnostics: If True, show deramp plots (disables parallel)
            destreak_diagnostics: If True, show destreak plots (disables parallel)
            parallel: If True and no diagnostics, process groups in parallel

        Returns:
            Dictionary mapping period to list of normalized intensity arrays in [0, 1]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        # Diagnostics require sequential processing for interactive plots
        if shadow_diagnostics or deramp_diagnostics or destreak_diagnostics:
            for period, frames in self.itergroups():
                frames = list(frames)
                logging.info(f"Processing {period}: {len(frames)} frames")
                corrected = self.process_group(
                    frames,
                    shadow_diagnostics=shadow_diagnostics,
                    deramp_diagnostics=deramp_diagnostics,
                    destreak_diagnostics=destreak_diagnostics,
                )
                results[period] = corrected
            return results

        # Collect all groups first (need to materialize for parallel processing)
        groups = [(period, list(frames)) for period, frames in self.itergroups()]

        if len(groups) == 0:
            return results

        # Determine worker count
        n_workers = self._workers or min(len(groups), 4)
        use_parallel = parallel and n_workers > 1 and len(groups) > 1

        if use_parallel:
            # Parallel processing using threads
            logging.info(f"Processing {len(groups)} groups with {n_workers} workers...")

            def process_one(period_frames):
                period, frames = period_frames
                corrected = self.process_group(frames)
                return (period, corrected)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_one, g): g[0] for g in groups}

                for future in as_completed(futures):
                    period = futures[future]
                    try:
                        _, corrected = future.result()
                        results[period] = corrected
                        logging.debug(f"  Completed {period}")
                    except Exception as e:
                        logging.error(f"Error processing {period}: {e}")
        else:
            # Sequential processing
            for period, frames in groups:
                logging.info(f"Processing {period}: {len(frames)} frames")
                corrected = self.process_group(frames)
                results[period] = corrected

        return results


# -----------------------------------------------------------------------------
# Interactive Viewer
# -----------------------------------------------------------------------------


class ProcessedViewer(BaseViewer):
    """
    Interactive viewer for processed radar frame data with navigation.

    Displays radar intensity data with polar, ship, and earth coordinate views.

    Example:
        >>> frames = [frame1, frame2, frame3]
        >>> config = Config()
        >>> viewer = ProcessedViewer(frames, config=config, view='polar')
        >>> viewer.show()
    """

    def __init__(
        self,
        frames: list[Frame],
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (10, 9),
        radar_height: float | None = None,
        config: Config | None = None,
        view: str = "polar",
    ):
        """
        Initialize the processed viewer.

        Args:
            frames: List of Frame objects to navigate through
            vmin: Minimum value for colorbar (default: 1% quantile of all frames)
            vmax: Maximum value for colorbar (default: 99% quantile of all frames)
            cmap: Colormap name
            figsize: Figure size
            radar_height: Height of radar above water in meters
            config: Config for bearing calculation
            view: Initial view type ('polar', 'ship', or 'earth')
        """
        import matplotlib.pyplot as plt

        # Initialize base viewer
        super().__init__(n_frames=len(frames), cmap=cmap, figsize=figsize)

        self._frames = frames
        self._view = view
        self._view_keys = {"1": "polar", "2": "ship", "3": "earth"}

        # Calculate quantile limits from all frames if not specified
        # Use corrected_intensity if available, otherwise use original intensity
        if vmin is None or vmax is None:
            all_intensities = np.concatenate(
                [
                    (
                        f.corrected_intensity if f.corrected_intensity is not None else f.intensity
                    ).ravel()
                    for f in frames
                ]
            )
            q_vmin, q_vmax = quantile_limits(all_intensities)
            self._vmin = vmin if vmin is not None else q_vmin
            self._vmax = vmax if vmax is not None else q_vmax
        else:
            self._vmin = vmin
            self._vmax = vmax

        self._config = config or Config()

        # Determine radar height (CLI arg > config > metadata)
        if radar_height is not None:
            self._radar_height = radar_height
        elif self._config.get("radar.height") is not None:
            self._radar_height = self._config.get("radar.height")
        elif self._config.get("tower.height") is not None:
            self._radar_height = self._config.get("tower.height")
        else:
            self._radar_height = None

        # Calculate bearing using Theta and Bearing classes
        self._theta = Theta(frames, self._config, refine=True)
        self._bearing = Bearing(self._theta, radar_height=self._radar_height)

        # Create figure with space for buttons
        self._fig, self._ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.15)

        # Initial plot
        self._im = None
        self._cbar = None
        self._draw_plot()

        # Add navigation buttons
        self._add_nav_buttons(
            prev_pos=[0.12, 0.02, 0.1, 0.05],
            next_pos=[0.23, 0.02, 0.1, 0.05],
            play_pos=[0.34, 0.02, 0.1, 0.05],
        )

        # Add view buttons
        self._add_view_buttons(
            [
                ("polar", "Polar", [0.45, 0.02, 0.1, 0.05]),
                ("ship", "Ship", [0.56, 0.02, 0.1, 0.05]),
                ("earth", "Earth", [0.67, 0.02, 0.1, 0.05]),
            ]
        )

        # Connect keyboard navigation
        self._connect_keyboard()

    def _get_frame(self, idx: int) -> Frame:
        """Get the Frame object at given index."""
        return self._frames[idx]

    def _draw_plot(self) -> None:
        """Draw or redraw the plot based on current view and frame."""
        self._draw_view()

    def _draw_view(self) -> None:
        """
        Draw the current view using corrected intensity if available.

        Overrides BaseViewer._draw_view to use corrected_intensity
        attribute on frames if it exists (set by process_group).
        """
        frame = self._frames[self._current_idx]

        # Use corrected_intensity if available, otherwise original intensity
        data = (
            frame.corrected_intensity if frame.corrected_intensity is not None else frame.intensity
        )

        # Clear previous plot and remove old insets
        self._ax.clear()
        self._clear_insets()

        if self._view == "polar":
            self._draw_polar(frame, data)
        elif self._view == "ship":
            self._draw_ship(frame, data)
            self._draw_nav_insets(frame)
        elif self._view == "earth":
            self._draw_earth(frame, data)
            self._draw_nav_insets(frame)

        # Update title
        self._update_title()

        # Add colorbar only once, but update mappable each time
        if self._cbar is None:
            self._cbar = self._fig.colorbar(self._im, ax=self._ax, label="Intensity")
        else:
            # Update colorbar to reference new image while keeping same limits
            self._cbar.mappable = self._im
            self._im.set_clim(self._vmin, self._vmax)

        self._fig.canvas.draw_idle()

    def _clear_insets(self) -> None:
        """Remove any existing inset axes."""
        if hasattr(self, "_ship_inset") and self._ship_inset is not None:
            self._ship_inset.remove()
            self._ship_inset = None
        if hasattr(self, "_wind_inset") and self._wind_inset is not None:
            self._wind_inset.remove()
            self._wind_inset = None

    def _draw_nav_insets(self, frame: Frame) -> None:
        """
        Draw small polar inset plots for ship and wind vectors.

        Ship inset in upper right, wind inset in lower right.
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        meta = frame.metadata
        inset_size = "12%"  # Size relative to parent axes

        # Determine max speed for scaling (use reasonable defaults, all in m/s)
        ship_max = 15.0  # m/s max ship speed
        wind_max = 25.0  # m/s max wind speed

        # Ship heading/speed inset (upper right)
        # Note: ship_speed is already in m/s (converted at parse time)
        if meta.heading is not None:
            self._ship_inset = inset_axes(
                self._ax,
                width=inset_size,
                height=inset_size,
                loc="upper right",
                borderpad=0.5,
                axes_class=None,
            )
            self._draw_vector_inset(
                self._ship_inset,
                heading=meta.heading,
                speed=meta.ship_speed,
                max_speed=ship_max,
                color="blue",
                label="Ship",
                label_pos="nw",
            )
        else:
            self._ship_inset = None

        # Wind direction/speed inset (lower right)
        if meta.wind_direction is not None:
            self._wind_inset = inset_axes(
                self._ax,
                width=inset_size,
                height=inset_size,
                loc="lower right",
                borderpad=0.5,
                axes_class=None,
            )
            self._draw_vector_inset(
                self._wind_inset,
                heading=meta.wind_direction,
                speed=meta.wind_speed,
                max_speed=wind_max,
                color="green",
                label="Wind",
                label_pos="sw",
            )
        else:
            self._wind_inset = None

    def _draw_vector_inset(
        self,
        ax,
        heading: float,
        speed: float | None,
        max_speed: float,
        color: str,
        label: str,
        label_pos: str = "ne",
    ) -> None:
        """
        Draw a polar-style vector inset showing direction and magnitude.

        Args:
            ax: Axes to draw on
            heading: Direction in degrees (0=North, clockwise)
            speed: Magnitude (optional, for arrow length)
            max_speed: Maximum speed for scaling rings and arrow
            color: Arrow color
            label: Label for the inset
            label_pos: Label position ('ne' for northeast, 'se' for southeast)
        """
        # Convert heading to radians (0=North, clockwise -> math convention)
        # Math: 0=East, counter-clockwise. So: theta = 90 - heading
        theta_rad = np.radians(90 - heading)

        # Draw speed rings
        ring_speeds = [max_speed * 0.33, max_speed * 0.67, max_speed]
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        for i, ring_speed in enumerate(ring_speeds):
            radius = ring_speed / max_speed
            ax.plot(
                radius * np.cos(circle_theta),
                radius * np.sin(circle_theta),
                "k-",
                linewidth=0.4,
                alpha=0.3,
            )
            # Label outer ring only
            if i == len(ring_speeds) - 1:
                ax.text(
                    0,
                    -radius - 0.08,
                    f"{ring_speed:.0f}",
                    ha="center",
                    va="top",
                    fontsize=5,
                    alpha=0.6,
                )

        # Draw crosshairs
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.3, alpha=0.4)

        # Draw arrow with length proportional to speed
        if speed is not None and speed > 0:
            arrow_len = min(speed / max_speed, 1.0)  # Clamp to max
        else:
            arrow_len = 0.1  # Minimum visible length

        dx = arrow_len * np.cos(theta_rad)
        dy = arrow_len * np.sin(theta_rad)
        ax.arrow(0, 0, dx, dy, head_width=0.08, head_length=0.06, fc=color, ec=color, linewidth=1.2)

        # Add cardinal directions (smaller)
        ax.text(0, 1.08, "N", ha="center", va="bottom", fontsize=5, fontweight="bold")
        ax.text(1.08, 0, "E", ha="left", va="center", fontsize=5)
        ax.text(0, -1.08, "S", ha="center", va="top", fontsize=5)
        ax.text(-1.08, 0, "W", ha="right", va="center", fontsize=5)

        # Add label with speed info positioned outside the plot
        if speed is not None:
            label_text = f"{label}\n{heading:.0f}°\n{speed:.1f} m/s"
        else:
            label_text = f"{label}\n{heading:.0f}°"

        if label_pos == "nw":
            ax.text(
                -1.3,
                1.3,
                label_text,
                ha="right",
                va="top",
                fontsize=6,
                fontweight="bold",
                color=color,
            )
        elif label_pos == "sw":
            ax.text(
                -1.3,
                -1.3,
                label_text,
                ha="right",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                color=color,
            )
        elif label_pos == "ne":
            ax.text(
                1.3,
                1.3,
                label_text,
                ha="left",
                va="top",
                fontsize=6,
                fontweight="bold",
                color=color,
            )
        elif label_pos == "se":
            ax.text(
                1.3,
                -1.3,
                label_text,
                ha="left",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                color=color,
            )

        # Set equal aspect and limits
        ax.set_xlim(-1.8, 1.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    def _update_title(self) -> None:
        """Update the plot title with current frame info."""
        frame = self._frames[self._current_idx]
        nav_info = format_nav_title(frame)

        view_labels = {"polar": "Polar", "ship": "Ship", "earth": "Earth"}
        view_label = view_labels.get(self._view, self._view)

        title = (
            f"Frame {self._current_idx + 1}/{len(self._frames)}: {frame.timestamp} [{view_label}]"
        )
        if nav_info:
            title += f"\n{nav_info}"

        self._ax.set_title(title)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument(
        "--groupby", "-g", type=str, default="h", help="Groupby frequency (default: h)"
    )
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of workers")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument(
        "--radar-height", type=float, default=None, help="Radar height above water (m)"
    )
    parser.add_argument("--plot", action="store_true", help="Launch interactive viewer")
    parser.add_argument(
        "--view",
        type=str,
        default="polar",
        choices=["polar", "ship", "earth"],
        help="Initial view type (default: polar)",
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap (default: viridis)")
    parser.add_argument(
        "--shadow-diagnostics", action="store_true", help="Show shadow detection diagnostic plots"
    )
    parser.add_argument(
        "--destreak-diagnostics", action="store_true", help="Show destreak diagnostic plots"
    )
    parser.add_argument(
        "--deramp-diagnostics",
        action="store_true",
        help="Show deramp (range correction) diagnostic plots",
    )


def add_subparser(subparsers) -> None:
    """Register the 'process' subcommand."""
    p = subparsers.add_parser(
        "process",
        help="Process frames (destreak, deramp) with viewer",
        description="Load and process WAMOS polar frames",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'process' command."""
    import time

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load config
    config = Config(args.config) if args.config else Config()

    t0 = time.time()

    with ProcessedFrames(
        stime=args.stime,
        etime=args.etime,
        polar_path=args.polar_path,
        groupby=args.groupby,
        workers=args.workers,
        config=config,
        radar_height=args.radar_height,
    ) as pframes:
        t1 = time.time()
        logging.info(f"Discovered {len(pframes)} files in {t1 - t0:.3f}s")
        logging.debug(f"{pframes}")

        if args.plot:
            # Process and plot group by group
            for period, frames in pframes.itergroups():
                frames = list(frames)
                logging.info(f"{period}: {len(frames)} frames")

                # Process the group and store corrected intensities on frames
                corrected = pframes.process_group(
                    frames,
                    shadow_diagnostics=args.shadow_diagnostics,
                    deramp_diagnostics=args.deramp_diagnostics,
                    destreak_diagnostics=args.destreak_diagnostics,
                )
                for frame, corr_intensity in zip(frames, corrected):
                    frame.corrected_intensity = corr_intensity

                # Show viewer for this group
                logging.info(f"Plotting {len(frames)} frames...")
                viewer = ProcessedViewer(
                    frames,
                    cmap=args.cmap,
                    radar_height=args.radar_height,
                    config=config,
                    view=args.view,
                )
                logging.info("Navigation: <- -> keys, Prev/Next buttons, Space=Play/Stop")
                logging.info("Views: 1=Polar, 2=Ship, 3=Earth (or click buttons)")
                logging.info("Close window to continue to next group...")
                viewer.show()
        else:
            # Process all groups
            pframes.process(
                shadow_diagnostics=args.shadow_diagnostics,
                deramp_diagnostics=args.deramp_diagnostics,
                destreak_diagnostics=args.destreak_diagnostics,
            )


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Load and process WAMOS polar frames")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
