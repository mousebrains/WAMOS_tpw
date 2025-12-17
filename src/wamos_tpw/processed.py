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

from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.config import WamosConfig
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

    def __init__(self,
                 stime: str | np.datetime64,
                 etime: str | np.datetime64,
                 polar_path: str,
                 groupby: str = 'h',
                 workers: int | None = None,
                 loader: Callable[[str], Frame | None] | None = None,
                 config: WamosConfig | None = None,
                 radar_height: float | None = None) -> None:
        """
        Initialize ProcessedFrames.

        Args:
            stime: Start time (string or np.datetime64)
            etime: End time (string or np.datetime64)
            polar_path: Base path to polar file directory
            groupby: Time grouping frequency (e.g., 'h', '30m', 'D')
            workers: Number of parallel workers (None = auto, 1 = sequential)
            loader: Optional custom file loader function
            config: WamosConfig for processing parameters
            radar_height: Height of radar above water in meters
        """
        super().__init__(stime, etime, polar_path, groupby, workers, loader)
        self._config = config or WamosConfig()
        self._radar_height = radar_height

    @property
    def config(self) -> WamosConfig:
        """Return the configuration."""
        return self._config

    @property
    def radar_height(self) -> float | None:
        """Return the radar height."""
        return self._radar_height

    def refine_theta(self, frames: list[Frame], shadow_diagnostics: bool = False) -> None:
        """
        Refine the angle with respect to the vessel by using the shadowed regions

        Args:
            frames: List of Frame objects to optimize the theta for
            shadow_diagnostics: If True, show shadow detection diagnostic plots and stop
        """
        # Create Theta to get shadow analysis (refinement does the edge detection)
        theta = Theta(frames, self._config, refine=True)

        # Print shadow statistics
        if theta.shadow_stats:
            print(f"Shadow {theta.shadow_stats}")

        if shadow_diagnostics:
            theta.plot_shadow_diagnostics()
            raise RuntimeError("Shadow diagnostic plots complete - stopping execution")

    def deramp_frames(self, frames: list[Frame],
                      diagnostics: bool = False) -> None:
        """
        Remove range-dependent intensity fall-off from frames.

        Applies empirical range profile normalization to correct for
        radar signal attenuation with distance. Stores the deramped
        intensity as frame.deramped_intensity for use by destreak.

        Args:
            frames: List of Frame objects to deramp
            diagnostics: If True, show diagnostic plots for each frame
        """
        for i, frame in enumerate(frames):
            deramp = Deramp(frame, self._config)
            # Store deramped intensity on frame for destreak to use
            frame.deramped_intensity = deramp.corrected_intensity

            logging.debug(f"Frame {i}: smooth profile range [{deramp.smooth_profile.min():.1f}, "
                         f"{deramp.smooth_profile.max():.1f}]")

            if diagnostics:
                deramp.plot_diagnostics()

    def destreak_frames(self, frames: list[Frame],
                        diagnostics: bool = False) -> list[np.ndarray]:
        """
        Remove radial streak artifacts from a list of frames.

        Each frame is destreaked using its temporal neighbors (previous and next
        frames in the list) to enable edge detection at the first and last
        bearing bins.

        If frames have deramped_intensity attribute (from deramp_frames),
        that will be used instead of the original intensity.

        Args:
            frames: List of Frame objects to destreak
            diagnostics: If True, show diagnostic plots for each frame

        Returns:
            List of corrected intensity arrays (same order as input frames)
        """
        corrected = []

        for i, center in enumerate(frames):
            prev_frame = frames[i - 1] if i > 0 else None
            next_frame = frames[i + 1] if i < len(frames) - 1 else None

            ds = Destreak(prev_frame, center, next_frame, self._config)
            corrected.append(ds.corrected_intensity)

            n_streaks = ds.streak_mask.sum()
            total_pixels = ds.streak_mask.size
            logging.debug(f"Frame {i}: {n_streaks}/{total_pixels} streak pixels "
                         f"({100*n_streaks/total_pixels:.2f}%)")

            if diagnostics:
                ds.plot_diagnostics()

        return corrected

    def normalize_frames(self, corrected: list[np.ndarray]) -> list[np.ndarray]:
        """
        Normalize each frame's intensity to the [0, 1] interval.

        Uses min-max normalization: (x - min) / (max - min)

        Args:
            corrected: List of intensity arrays to normalize

        Returns:
            List of normalized intensity arrays in [0, 1]
        """
        normalized = []
        for intensity in corrected:
            min_val = intensity.min()
            max_val = intensity.max()
            if max_val > min_val:
                norm = (intensity - min_val) / (max_val - min_val)
            else:
                # Constant array - set to 0.5
                norm = np.full_like(intensity, 0.5)
            normalized.append(norm)
            logging.debug(f"Normalized: [{min_val:.2f}, {max_val:.2f}] -> [0, 1]")
        return normalized

    def process_group(self,
                      frames: list[Frame],
                      shadow_diagnostics: bool = False,
                      deramp_diagnostics: bool = False,
                      destreak_diagnostics: bool = False) -> list[np.ndarray]:
        """
        Process a single group of frames: refine theta, deramp, destreak, normalize.

        Args:
            frames: List of Frame objects to process
            shadow_diagnostics: If True, show shadow detection plots
            deramp_diagnostics: If True, show deramp plots
            destreak_diagnostics: If True, show destreak plots

        Returns:
            List of normalized intensity arrays in [0, 1]
        """
        frames = list(frames)  # Materialize generator if needed

        # Refine theta for this group
        self.refine_theta(frames, shadow_diagnostics=shadow_diagnostics)

        # Deramp frames first (remove range-dependent intensity fall-off)
        self.deramp_frames(frames, diagnostics=deramp_diagnostics)

        # Destreak frames (uses deramped_intensity if available)
        corrected = self.destreak_frames(frames, diagnostics=destreak_diagnostics)

        # Normalize each frame to [0, 1]
        normalized = self.normalize_frames(corrected)

        return normalized

    def process(self,
                shadow_diagnostics: bool = False,
                deramp_diagnostics: bool = False,
                destreak_diagnostics: bool = False,
                parallel: bool = True) -> dict:
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
                print(f"Processing {period}: {len(frames)} frames")
                corrected = self.process_group(frames,
                                               shadow_diagnostics=shadow_diagnostics,
                                               deramp_diagnostics=deramp_diagnostics,
                                               destreak_diagnostics=destreak_diagnostics)
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
            print(f"Processing {len(groups)} groups with {n_workers} workers...")

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
                        print(f"  Completed {period}")
                    except Exception as e:
                        logging.error(f"Error processing {period}: {e}")
        else:
            # Sequential processing
            for period, frames in groups:
                print(f"Processing {period}: {len(frames)} frames")
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
        >>> config = WamosConfig()
        >>> viewer = ProcessedViewer(frames, config=config, view='polar')
        >>> viewer.show()
    """

    def __init__(self,
                 frames: list[Frame],
                 vmin: float | None = None,
                 vmax: float | None = None,
                 cmap: str = 'viridis',
                 figsize: tuple[float, float] = (10, 9),
                 radar_height: float | None = None,
                 config: WamosConfig | None = None,
                 view: str = 'polar'):
        """
        Initialize the processed viewer.

        Args:
            frames: List of Frame objects to navigate through
            vmin: Minimum value for colorbar (default: 1% quantile of all frames)
            vmax: Maximum value for colorbar (default: 99% quantile of all frames)
            cmap: Colormap name
            figsize: Figure size
            radar_height: Height of radar above water in meters
            config: WamosConfig for bearing calculation
            view: Initial view type ('polar', 'ship', or 'earth')
        """
        import matplotlib.pyplot as plt

        # Initialize base viewer
        super().__init__(n_frames=len(frames), cmap=cmap, figsize=figsize)

        self._frames = frames
        self._view = view
        self._view_keys = {'1': 'polar', '2': 'ship', '3': 'earth'}

        # Calculate quantile limits from all frames if not specified
        # Use corrected_intensity if available, otherwise use original intensity
        if vmin is None or vmax is None:
            all_intensities = np.concatenate([
                getattr(f, 'corrected_intensity', f.intensity).ravel()
                for f in frames
            ])
            q_vmin, q_vmax = quantile_limits(all_intensities)
            self._vmin = vmin if vmin is not None else q_vmin
            self._vmax = vmax if vmax is not None else q_vmax
        else:
            self._vmin = vmin
            self._vmax = vmax

        self._config = config or WamosConfig()

        # Determine radar height (CLI arg > config > metadata)
        if radar_height is not None:
            self._radar_height = radar_height
        elif self._config.radar.height is not None:
            self._radar_height = self._config.radar.height
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
            play_pos=[0.34, 0.02, 0.1, 0.05]
        )

        # Add view buttons
        self._add_view_buttons([
            ('polar', 'Polar', [0.45, 0.02, 0.1, 0.05]),
            ('ship', 'Ship', [0.56, 0.02, 0.1, 0.05]),
            ('earth', 'Earth', [0.67, 0.02, 0.1, 0.05]),
        ])

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
        data = getattr(frame, 'corrected_intensity', frame.intensity)

        # Clear previous plot and remove old insets
        self._ax.clear()
        self._clear_insets()

        if self._view == 'polar':
            self._draw_polar(frame, data)
        elif self._view == 'ship':
            self._draw_ship(frame, data)
            self._draw_nav_insets(frame)
        elif self._view == 'earth':
            self._draw_earth(frame, data)
            self._draw_nav_insets(frame)

        # Update title
        self._update_title()

        # Add colorbar only once, but update mappable each time
        if self._cbar is None:
            self._cbar = self._fig.colorbar(self._im, ax=self._ax,
                                             label='Intensity')
        else:
            # Update colorbar to reference new image while keeping same limits
            self._cbar.mappable = self._im
            self._im.set_clim(self._vmin, self._vmax)

        self._fig.canvas.draw_idle()

    def _clear_insets(self) -> None:
        """Remove any existing inset axes."""
        if hasattr(self, '_ship_inset') and self._ship_inset is not None:
            self._ship_inset.remove()
            self._ship_inset = None
        if hasattr(self, '_wind_inset') and self._wind_inset is not None:
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
                self._ax, width=inset_size, height=inset_size,
                loc='upper right', borderpad=0.5,
                axes_class=None
            )
            self._draw_vector_inset(
                self._ship_inset,
                heading=meta.heading,
                speed=meta.ship_speed,
                max_speed=ship_max,
                color='blue',
                label='Ship',
                label_pos='nw'
            )
        else:
            self._ship_inset = None

        # Wind direction/speed inset (lower right)
        if meta.wind_direction is not None:
            self._wind_inset = inset_axes(
                self._ax, width=inset_size, height=inset_size,
                loc='lower right', borderpad=0.5,
                axes_class=None
            )
            self._draw_vector_inset(
                self._wind_inset,
                heading=meta.wind_direction,
                speed=meta.wind_speed,
                max_speed=wind_max,
                color='green',
                label='Wind',
                label_pos='sw'
            )
        else:
            self._wind_inset = None

    def _draw_vector_inset(self, ax, heading: float, speed: float | None,
                           max_speed: float, color: str, label: str,
                           label_pos: str = 'ne') -> None:
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
            ax.plot(radius * np.cos(circle_theta), radius * np.sin(circle_theta),
                   'k-', linewidth=0.4, alpha=0.3)
            # Label outer ring only
            if i == len(ring_speeds) - 1:
                ax.text(0, -radius - 0.08, f'{ring_speed:.0f}', ha='center', va='top',
                       fontsize=5, alpha=0.6)

        # Draw crosshairs
        ax.axhline(0, color='gray', linewidth=0.3, alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.3, alpha=0.4)

        # Draw arrow with length proportional to speed
        if speed is not None and speed > 0:
            arrow_len = min(speed / max_speed, 1.0)  # Clamp to max
        else:
            arrow_len = 0.1  # Minimum visible length

        dx = arrow_len * np.cos(theta_rad)
        dy = arrow_len * np.sin(theta_rad)
        ax.arrow(0, 0, dx, dy, head_width=0.08, head_length=0.06,
                fc=color, ec=color, linewidth=1.2)

        # Add cardinal directions (smaller)
        ax.text(0, 1.08, 'N', ha='center', va='bottom', fontsize=5, fontweight='bold')
        ax.text(1.08, 0, 'E', ha='left', va='center', fontsize=5)
        ax.text(0, -1.08, 'S', ha='center', va='top', fontsize=5)
        ax.text(-1.08, 0, 'W', ha='right', va='center', fontsize=5)

        # Add label with speed info positioned outside the plot
        if speed is not None:
            label_text = f'{label}\n{heading:.0f}°\n{speed:.1f} m/s'
        else:
            label_text = f'{label}\n{heading:.0f}°'

        if label_pos == 'nw':
            ax.text(-1.3, 1.3, label_text, ha='right', va='top', fontsize=6,
                   fontweight='bold', color=color)
        elif label_pos == 'sw':
            ax.text(-1.3, -1.3, label_text, ha='right', va='bottom', fontsize=6,
                   fontweight='bold', color=color)
        elif label_pos == 'ne':
            ax.text(1.3, 1.3, label_text, ha='left', va='top', fontsize=6,
                   fontweight='bold', color=color)
        elif label_pos == 'se':
            ax.text(1.3, -1.3, label_text, ha='left', va='bottom', fontsize=6,
                   fontweight='bold', color=color)

        # Set equal aspect and limits
        ax.set_xlim(-1.8, 1.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _update_title(self) -> None:
        """Update the plot title with current frame info."""
        frame = self._frames[self._current_idx]
        nav_info = format_nav_title(frame)

        view_labels = {'polar': 'Polar', 'ship': 'Ship', 'earth': 'Earth'}
        view_label = view_labels.get(self._view, self._view)

        title = f'Frame {self._current_idx + 1}/{len(self._frames)}: {frame.timestamp} [{view_label}]'
        if nav_info:
            title += f'\n{nav_info}'

        self._ax.set_title(title)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def add_subparser(subparsers) -> None:
    """Register the 'process' subcommand."""
    p = subparsers.add_parser(
        'process',
        help='Process frames (destreak, deramp) with viewer',
        description="Load and process WAMOS polar frames"
    )
    p.add_argument("stime", type=str, help="Start time")
    p.add_argument("etime", type=str, help="End time")
    p.add_argument("polar_path", type=str, help="Path to polar files")
    p.add_argument("--groupby", "-g", type=str, default='h',
                   help="Groupby frequency (default: h)")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="Number of workers")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="YAML configuration file")
    p.add_argument("--radar-height", type=float, default=None,
                   help="Radar height above water (m)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose output")
    p.add_argument("--plot", action="store_true",
                   help="Launch interactive viewer")
    p.add_argument("--view", type=str, default='polar',
                   choices=['polar', 'ship', 'earth'],
                   help="Initial view type (default: polar)")
    p.add_argument("--cmap", type=str, default='viridis',
                   help="Colormap (default: viridis)")
    p.add_argument("--shadow-diagnostics", action="store_true",
                   help="Show shadow detection diagnostic plots")
    p.add_argument("--destreak-diagnostics", action="store_true",
                   help="Show destreak diagnostic plots")
    p.add_argument("--deramp-diagnostics", action="store_true",
                   help="Show deramp (range correction) diagnostic plots")
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'process' command."""
    import time

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # Load config
    config = WamosConfig(args.config) if args.config else WamosConfig()

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
        print(f"Discovered {len(pframes)} files in {t1-t0:.3f}s")
        print(pframes)

        if args.plot:
            # Process and plot group by group
            for period, frames in pframes.itergroups():
                frames = list(frames)
                print(f"\n{period}: {len(frames)} frames")

                # Process the group and store corrected intensities on frames
                corrected = pframes.process_group(frames,
                                                  shadow_diagnostics=args.shadow_diagnostics,
                                                  deramp_diagnostics=args.deramp_diagnostics,
                                                  destreak_diagnostics=args.destreak_diagnostics)
                for frame, corr_intensity in zip(frames, corrected):
                    frame.corrected_intensity = corr_intensity

                # Show viewer for this group
                print(f"Plotting {len(frames)} frames...")
                viewer = ProcessedViewer(
                    frames,
                    cmap=args.cmap,
                    radar_height=args.radar_height,
                    config=config,
                    view=args.view
                )
                print("Navigation: ← → keys, Prev/Next buttons, Space=Play/Stop")
                print("Views: 1=Polar, 2=Ship, 3=Earth (or click buttons)")
                print("Close window to continue to next group...")
                viewer.show()
        else:
            # Process all groups
            pframes.process(shadow_diagnostics=args.shadow_diagnostics,
                           deramp_diagnostics=args.deramp_diagnostics,
                           destreak_diagnostics=args.destreak_diagnostics)


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Load and process WAMOS polar frames")
    parser.add_argument("stime", type=str, help="Start time")
    parser.add_argument("etime", type=str, help="End time")
    parser.add_argument("polar_path", type=str, help="Path to polar files")
    parser.add_argument("--groupby", "-g", type=str, default='h',
                        help="Groupby frequency (default: h)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of workers")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML configuration file")
    parser.add_argument("--radar-height", type=float, default=None,
                        help="Radar height above water (m)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--plot", action="store_true",
                        help="Launch interactive viewer")
    parser.add_argument("--view", type=str, default='polar',
                        choices=['polar', 'ship', 'earth'],
                        help="Initial view type (default: polar)")
    parser.add_argument("--cmap", type=str, default='viridis',
                        help="Colormap (default: viridis)")
    parser.add_argument("--shadow-diagnostics", action="store_true",
                        help="Show shadow detection diagnostic plots")
    parser.add_argument("--destreak-diagnostics", action="store_true",
                        help="Show destreak diagnostic plots")
    parser.add_argument("--deramp-diagnostics", action="store_true",
                        help="Show deramp (range correction) diagnostic plots")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
