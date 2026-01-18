#! /usr/bin/env python3
#
# Files class for loading WAMOS polar files in time-based groups
# Uses Filenames class for efficient file discovery and groupby intervals
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterator, Callable, Any

import numpy as np

from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.config import Config
from wamos_tpw.filenames import Filenames, _parse_timestamp
from wamos_tpw.frame import Frame
from wamos_tpw.plotting import BaseViewer, quantile_limits, calc_bin_edges, format_nav_title
from wamos_tpw.polarfile import load_polar_file


class Files:
    """
    Load WAMOS polar files in time-based groups using the Filenames class.

    Supports efficient parallel loading and time-based grouping for
    processing large datasets in manageable chunks.

    Example:
        >>> files = Files(
        ...     stime='20241215100000',
        ...     etime='20241215120000',
        ...     polar_path='/data/wamos',
        ...     groupby='30m'
        ... )
        >>> for period, frames in files.itergroups():
        ...     process_frames(frames)
    """

    def __init__(
        self,
        stime: str | np.datetime64,
        etime: str | np.datetime64,
        polar_path: str,
        groupby: str = "h",
        workers: int | None = None,
        loader: Callable[[str], Frame | None] | None = None,
    ) -> None:
        """
        Initialize Files with a time range and grouping interval.

        Args:
            stime: Start time (string or np.datetime64)
            etime: End time (string or np.datetime64)
            polar_path: Base path to polar file directory
            groupby: Time grouping frequency (e.g., 'h', '30m', 'D')
            workers: Number of parallel workers (None = auto, 1 = sequential)
            loader: Optional custom file loader function
        """
        # Parse timestamps if strings
        if isinstance(stime, str):
            stime = _parse_timestamp(stime)
        if isinstance(etime, str):
            etime = _parse_timestamp(etime)

        self._stime = stime
        self._etime = etime
        self._polar_path = polar_path
        self._groupby = groupby
        self._workers = workers
        self._loader = loader or load_polar_file

        # Initialize Filenames for file discovery
        self._filenames = Filenames(
            stime=stime, etime=etime, polar_path=polar_path, workers=workers
        )

        # Check GIL status for optimal parallelization
        self._gil_disabled = self._check_gil_status()
        if self._gil_disabled:
            logging.info("GIL-free mode detected - using optimized parallelization")

    @staticmethod
    def _check_gil_status() -> bool:
        """Check if Python is running in GIL-free mode (Python 3.13+)."""
        try:
            return sys._is_gil_enabled() is False  # type: ignore[attr-defined]
        except AttributeError:
            return False

    @property
    def filenames(self) -> Filenames:
        """Return the underlying Filenames instance."""
        return self._filenames

    @property
    def files(self) -> list[str]:
        """Return all matching file paths."""
        return self._filenames.files

    def __len__(self) -> int:
        """Return total number of matching files."""
        return len(self._filenames)

    def __bool__(self) -> bool:
        """Return True if any files were found."""
        return bool(self._filenames)

    # -------------------------------------------------------------------------
    # Group-based loading
    # -------------------------------------------------------------------------

    def groups(self) -> dict[np.datetime64, list[str]]:
        """
        Get all file groups based on the groupby frequency.

        Returns:
            Dict mapping period start times to lists of file paths
        """
        return self._filenames.groupby(self._groupby)

    def itergroups(self) -> Iterator[tuple[np.datetime64, list[Frame]]]:
        """
        Iterate over time groups, loading and yielding frames for each group.

        Yields:
            Tuples of (period_start, list of Frame objects)

        Example:
            >>> for period, frames in files.itergroups():
            ...     print(f"{period}: {len(frames)} frames")
            ...     for frame in frames:
            ...         process(frame.intensity)
        """
        for period, file_list in self._filenames.groupby(self._groupby).items():
            frames = self._load_files(file_list)
            yield period, frames

    def itergroups_parallel(
        self, process_fn: Callable[[np.datetime64, list[Frame]], Any], max_groups: int | None = None
    ) -> Iterator[tuple[np.datetime64, Any]]:
        """
        Process groups in parallel, yielding results as they complete.

        Args:
            process_fn: Function to process each group (period, frames) -> result
            max_groups: Maximum number of groups to process concurrently

        Yields:
            Tuples of (period, result from process_fn)
        """
        groups = self._filenames.groupby(self._groupby)

        if max_groups is None:
            max_groups = self._workers or 4

        with ProcessPoolExecutor(max_workers=max_groups) as executor:
            # Submit group processing tasks
            future_to_period = {}

            for period, file_list in groups.items():
                future = executor.submit(self._process_group, period, file_list, process_fn)
                future_to_period[future] = period

            # Yield results as they complete
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    result = future.result()
                    yield period, result
                except Exception as e:
                    logging.error(f"Error processing group {period}: {e}")

    def _process_group(
        self, period: np.datetime64, file_list: list[str], process_fn: Callable
    ) -> Any:
        """Process a single group (for parallel execution)."""
        frames = self._load_files(file_list)
        return process_fn(period, frames)

    # -------------------------------------------------------------------------
    # File loading
    # -------------------------------------------------------------------------

    def _load_files(self, file_list: list[str]) -> list[Frame]:
        """Load a list of files, potentially in parallel."""
        if not file_list:
            return []

        # Use parallel loading if we have multiple files and workers != 1
        if len(file_list) > 1 and self._workers != 1:
            return self._load_files_parallel(file_list)
        else:
            return self._load_files_sequential(file_list)

    def _load_files_sequential(self, file_list: list[str]) -> list[Frame]:
        """Load files sequentially."""
        frames = []
        for filepath in file_list:
            frame = self._loader(filepath)
            if frame is not None:
                frames.append(frame)
        return frames

    def _load_files_parallel(self, file_list: list[str]) -> list[Frame]:
        """Load files in parallel using ThreadPoolExecutor (I/O-bound)."""
        frames = []
        workers = self._workers or 4

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(self._loader, fp): fp for fp in file_list}

            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    frame = future.result()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")

        # Sort by timestamp
        frames.sort(key=lambda f: f.timestamp)
        return frames

    def load_group(self, period: np.datetime64, max_frames: int | None = None) -> list[Frame]:
        """
        Load frames for a specific time period.

        Args:
            period: The period start time (must be a key from groups())
            max_frames: Maximum number of frames to load (None = all)

        Returns:
            List of Frame objects for that period
        """
        groups = self.groups()
        if period not in groups:
            raise KeyError(f"Period {period} not found in groups")
        file_list = groups[period]
        if max_frames is not None:
            file_list = file_list[:max_frames]
        return self._load_files(file_list)

    def load_files(self, file_list: list[str]) -> list[Frame]:
        """
        Load frames from a list of file paths.

        Args:
            file_list: List of file paths to load

        Returns:
            List of Frame objects
        """
        return self._load_files(file_list)

    def load_all(self) -> list[Frame]:
        """
        Load all frames from all files.

        Warning: This may use significant memory for large time ranges.

        Returns:
            List of all Frame objects, sorted by timestamp
        """
        return self._load_files(self.files)

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary of the file collection."""
        groups = self.groups()
        return {
            "stime": self._stime,
            "etime": self._etime,
            "polar_path": self._polar_path,
            "groupby": self._groupby,
            "total_files": len(self),
            "n_groups": len(groups),
            "groups": {str(k): len(v) for k, v in groups.items()},
        }

    def __repr__(self) -> str:
        return (
            f"Files(stime={self._stime}, etime={self._etime}, "
            f"n_files={len(self)}, groupby='{self._groupby}')"
        )

    def __str__(self) -> str:
        groups = self.groups()
        lines = [
            f"Files: {len(self)} files from {self._stime} to {self._etime}",
            f"  Path: {self._polar_path}",
            f"  Groupby: {self._groupby} ({len(groups)} groups)",
        ]
        for period, files in list(groups.items())[:5]:
            lines.append(f"    {period}: {len(files)} files")
        if len(groups) > 5:
            lines.append(f"    ... and {len(groups) - 5} more groups")
        return "\n".join(lines)

    def __enter__(self) -> "Files":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        pass


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------


def plot_frame_intensity(
    frame: Frame,
    ax=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    colorbar: bool = True,
    radar_height: float | None = None,
    config: Config | None = None,
):
    """
    Plot frame intensity data using pcolormesh with ground distance axis.

    Args:
        frame: Frame object to plot
        ax: Matplotlib axes (creates new figure if None)
        vmin: Minimum value for colorbar (default: 1% quantile)
        vmax: Maximum value for colorbar (default: 99% quantile)
        cmap: Colormap name (default: 'viridis')
        title: Plot title (default: frame timestamp)
        colorbar: Whether to add colorbar (default: True)
        radar_height: Height of radar above water in meters.
                     Priority: this arg > config.radar.height > frame.metadata.radar_height
                     Falls back to slant range if not available.
        config: Config for radar height fallback.

    Returns:
        Tuple of (figure, axes, pcolormesh object)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Get intensity data (bottom 12 bits)
    data = frame.intensity

    # Use quantile limits if not specified
    if vmin is None or vmax is None:
        q_vmin, q_vmax = quantile_limits(data)
        if vmin is None:
            vmin = q_vmin
        if vmax is None:
            vmax = q_vmax

    # Calculate ground distances for x-axis (priority: arg > config > metadata)
    if radar_height is None and config is not None:
        radar_height = config.radar.height
    if radar_height is None:
        radar_height = frame.metadata.radar_height

    if radar_height is not None:
        distances = frame.ground_range(radar_height)
        x_label = "Ground Distance (m)"
    else:
        # Fall back to slant range if no radar height
        distances = frame.slant_range()
        x_label = "Slant Range (m)"

    # Create coordinate arrays for pcolormesh
    # bearings on y-axis (0 to n_bearings)
    # distances on x-axis
    bearings = np.arange(frame.n_bearings + 1)

    # Need n_distances + 1 edges for pcolormesh
    dist_edges = calc_bin_edges(distances)

    # Create pcolormesh plot with distance coordinates
    im = ax.pcolormesh(dist_edges, bearings, data, vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")

    # Add colorbar
    if colorbar:
        fig.colorbar(im, ax=ax, label="Intensity (0-4095)")

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("Bearing bin")
    if title is None:
        nav_info = format_nav_title(frame)
        title = f"Frame: {frame.timestamp}"
        if nav_info:
            title += f"\n{nav_info}"
    ax.set_title(title)

    return fig, ax, im


class IntensityViewer(BaseViewer):
    """
    Interactive viewer for frame intensity plots with forward/backward navigation.

    Uses Bearing class for coordinate transformations and supports multiple views:
    - Polar: bearing (degrees) vs ground distance
    - Ship: x/y coordinates relative to ship (+X=starboard, +Y=bow)
    - Earth: x/y coordinates in earth frame (+X=East, +Y=North)

    Navigation wraps around (first <-> last frame).

    Example:
        >>> config = Config('radar_config.yaml')
        >>> frames = [frame1, frame2, frame3]
        >>> viewer = IntensityViewer(frames, config=config, view='polar')
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
        Initialize the intensity viewer.

        Args:
            frames: List of Frame objects to navigate through
            vmin: Minimum value for colorbar (default: 1% quantile of all frames)
            vmax: Maximum value for colorbar (default: 99% quantile of all frames)
            cmap: Colormap name
            figsize: Figure size
            radar_height: Height of radar above water in meters.
                         Priority: this arg > config.radar.height > frame.metadata.radar_height
            config: Config for bearing calculation and other settings.
            view: Initial view type ('polar', 'ship', or 'earth')
        """
        import matplotlib.pyplot as plt

        # Initialize base viewer
        super().__init__(n_frames=len(frames), cmap=cmap, figsize=figsize)

        self._frames = frames
        self._view = view
        self._view_keys = {"1": "polar", "2": "ship", "3": "earth"}

        # Calculate quantile limits from all frames if not specified
        if vmin is None or vmax is None:
            all_intensities = np.concatenate([f.intensity.ravel() for f in frames])
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
            prev_pos=[0.10, 0.02, 0.10, 0.05],
            next_pos=[0.21, 0.02, 0.10, 0.05],
            play_pos=[0.32, 0.02, 0.10, 0.05],
        )

        # Add view buttons
        self._add_view_buttons(
            [
                ("polar", "Polar", [0.50, 0.02, 0.10, 0.05]),
                ("ship", "Ship", [0.61, 0.02, 0.10, 0.05]),
                ("earth", "Earth", [0.72, 0.02, 0.10, 0.05]),
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

    def _update_title(self) -> None:
        """Update the plot title with current frame info."""
        frame = self._frames[self._current_idx]
        nav_info = format_nav_title(frame)

        view_labels = {"polar": "Polar", "ship": "Ship Coords", "earth": "Earth Coords"}
        view_label = view_labels.get(self._view, self._view)

        # Check if using ground or slant range for cartesian views
        if self._view in ("ship", "earth"):
            height = self._get_radar_height(self._current_idx)
            range_type = "ground" if height is not None else "slant"
            view_label += f" ({range_type})"

        title = (
            f"Frame {self._current_idx + 1}/{len(self._frames)}: {frame.timestamp} [{view_label}]"
        )
        if nav_info:
            title += f"\n{nav_info}"
        self._ax.set_title(title)


def plot_frame_bits(
    frame: Frame,
    distance_bins: list[int] | None = None,
    figsize: tuple[float, float] = (14, 10),
    title_prefix: str = "",
):
    """
    Plot the top 4 bits for specified distance bins.

    Args:
        frame: Frame object to plot
        distance_bins: List of distance bin indices to plot (default: 0-20)
        figsize: Figure size
        title_prefix: Prefix for figure title

    Returns:
        Tuple of (figure, axes array)
    """
    import matplotlib.pyplot as plt

    if distance_bins is None:
        distance_bins = list(range(min(21, frame.n_distances)))

    n_bins = len(distance_bins)

    # Create figure with 4 subplots (one for each bit)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    bit_data = [
        ("Bit 12 (PPS)", frame.bit12),
        ("Bit 13 (Bearing)", frame.bit13),
        ("Bit 14", frame.bit14),
        ("Bit 15", frame.bit15),
    ]

    for ax, (bit_name, data) in zip(axes, bit_data):
        # Extract the specified distance bins across all bearings
        # Shape: (n_bearings, n_selected_bins)
        selected = data[:, distance_bins]

        # Plot as image (no colorbar needed - values are only 0 or 1)
        ax.imshow(
            selected.T,
            aspect="auto",
            cmap="binary",
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Bearing bin")
        ax.set_ylabel("Distance bin index")
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels([str(b) for b in distance_bins])
        ax.set_title(f"{bit_name}")

    nav_info = format_nav_title(frame)
    suptitle = f"{title_prefix}Top 4 bits - Frame: {frame.timestamp}"
    if nav_info:
        suptitle += f"\n{nav_info}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes


def plot_distance_bins_detail(
    frame: Frame, distance_bins: list[int] | None = None, figsize: tuple[float, float] = (16, 12)
):
    """
    Plot detailed view of top 4 bits for each distance bin.

    Creates a grid showing each distance bin as a row for each bit type.

    Args:
        frame: Frame object to plot
        distance_bins: Distance bins to show (default: 0-20)
        figsize: Figure size

    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt

    if distance_bins is None:
        distance_bins = list(range(min(21, frame.n_distances)))

    n_bins = len(distance_bins)

    # Create figure with subplots: rows=distance bins, cols=4 bits
    fig, axes = plt.subplots(n_bins, 4, figsize=figsize, sharex=True)

    bit_extractors = [
        ("Bit12 (PPS)", frame.bit12),
        ("Bit13 (Bearing)", frame.bit13),
        ("Bit14", frame.bit14),
        ("Bit15", frame.bit15),
    ]

    for row_idx, dist_bin in enumerate(distance_bins):
        for col_idx, (bit_name, bit_data) in enumerate(bit_extractors):
            ax = axes[row_idx, col_idx] if n_bins > 1 else axes[col_idx]

            # Get data for this distance bin across all bearings
            row_data = bit_data[:, dist_bin].astype(int)

            # Plot as line
            ax.fill_between(range(len(row_data)), row_data, alpha=0.7)
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1])

            # Labels
            if row_idx == 0:
                ax.set_title(bit_name)
            if col_idx == 0:
                ax.set_ylabel(f"D={dist_bin}", fontsize=8)
            if row_idx == n_bins - 1:
                ax.set_xlabel("Bearing")

    nav_info = format_nav_title(frame)
    suptitle = f"Distance bins {distance_bins[0]}-{distance_bins[-1]} - Frame: {frame.timestamp}"
    if nav_info:
        suptitle += f"\n{nav_info}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes


def parse_distance_bins(spec: str, max_bins: int) -> list[int]:
    """
    Parse a distance bin specification into a list of bin indices.

    Supports Python-like slice notation:
        "5"     -> [5]
        ":21"   -> [0, 1, ..., 20]
        "2:5"   -> [2, 3, 4]
        "10:"   -> [10, 11, ..., max_bins-1]
        ":"     -> [0, 1, ..., max_bins-1]

    Args:
        spec: Distance bin specification string
        max_bins: Maximum number of bins available

    Returns:
        List of bin indices

    Raises:
        ValueError: If specification is invalid
    """
    spec = spec.strip()

    if ":" not in spec:
        # Single bin
        try:
            idx = int(spec)
            if idx < 0 or idx >= max_bins:
                raise ValueError(f"Bin index {idx} out of range [0, {max_bins - 1}]")
            return [idx]
        except ValueError:
            raise ValueError(f"Invalid bin specification: {spec}")

    # Slice notation
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice specification: {spec}")

    start_str, end_str = parts
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else max_bins

    if start < 0:
        start = 0
    if end > max_bins:
        end = max_bins
    if start >= end:
        raise ValueError(f"Invalid range: start ({start}) >= end ({end})")

    return list(range(start, end))


def plot_bits_across_frames(
    frames: list[Frame],
    distance_bins: list[int] | None = None,
    figsize: tuple[float, float] = (16, 10),
    show_frame_boundaries: bool = True,
):
    """
    Plot all 4 top bits across multiple frames with frame transitions marked.

    Similar to plot_frame_bits but concatenates data from multiple frames,
    with alternating background colors and vertical lines at frame boundaries.

    Args:
        frames: List of Frame objects to plot
        distance_bins: List of distance bin indices to plot (default: 0-20)
        figsize: Figure size
        show_frame_boundaries: Whether to show frame boundaries

    Returns:
        Tuple of (figure, axes array)
    """
    import matplotlib.pyplot as plt

    if not frames:
        raise ValueError("No frames provided")

    if distance_bins is None:
        distance_bins = list(range(min(21, frames[0].n_distances)))

    n_bins = len(distance_bins)

    # Collect bit data across all frames
    bit_data_combined = {12: [], 13: [], 14: [], 15: []}
    frame_boundaries = [0]

    for frame in frames:
        for bit in (12, 13, 14, 15):
            data = getattr(frame, f"bit{bit}")
            # Extract specified distance bins: shape (n_bearings, n_selected_bins)
            selected = data[:, distance_bins]
            bit_data_combined[bit].append(selected)
        frame_boundaries.append(frame_boundaries[-1] + frame.n_bearings)

    # Concatenate along bearing axis
    for bit in (12, 13, 14, 15):
        bit_data_combined[bit] = np.vstack(bit_data_combined[bit])

    # Create figure with 4 subplots (one for each bit), linked x-axes
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    bit_info = [
        (12, "Bit 12 (PPS)"),
        (13, "Bit 13 (Bearing)"),
        (14, "Bit 14"),
        (15, "Bit 15"),
    ]

    for ax, (bit, bit_name) in zip(axes, bit_info):
        data = bit_data_combined[bit]

        # Add alternating background colors for frame regions
        if show_frame_boundaries:
            colors = ["#e8e8e8", "#ffffff"]
            for i in range(len(frame_boundaries) - 1):
                start = frame_boundaries[i]
                end = frame_boundaries[i + 1]
                ax.axvspan(start, end, color=colors[i % 2], zorder=0)

        # Plot as image (no colorbar - values are only 0 or 1)
        ax.imshow(
            data.T,
            aspect="auto",
            cmap="binary",
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=1,
        )

        # Add frame boundary lines
        if show_frame_boundaries:
            for boundary in frame_boundaries[1:-1]:
                ax.axvline(boundary, color="red", linestyle="-", linewidth=0.5, alpha=0.7)

        ax.set_xlabel("Radial index (continuous)")
        ax.set_ylabel("Distance bin")
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels([str(b) for b in distance_bins])
        ax.set_title(bit_name)

    # Create figure title with time range
    start_time = frames[0].timestamp
    end_time = frames[-1].timestamp
    suptitle = f"Top 4 bits across {len(frames)} frames (D={distance_bins[0]}-{distance_bins[-1]})"
    suptitle += f"\n{start_time} to {end_time}"
    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument(
        "--groupby", "-g", type=str, default="h", help="Groupby frequency (default: h)"
    )
    parser.add_argument("--workers", "-w", type=int, default=None, help="Number of workers")
    parser.add_argument(
        "--load", action="store_true", help="Actually load the files (not just discover)"
    )
    parser.add_argument(
        "--plot-intensity",
        action="store_true",
        help="Plot intensity (bottom 12 bits) for each frame",
    )
    parser.add_argument(
        "--plot-bits", action="store_true", help="Plot top 4 bits per frame (uses --distance-bins)"
    )
    parser.add_argument(
        "--plot-bits-detail",
        action="store_true",
        help="Plot detailed view of top 4 bits per distance bin per frame",
    )
    parser.add_argument(
        "--plot-bits-across",
        action="store_true",
        help="Plot top 4 bits across all frames with frame boundaries",
    )
    parser.add_argument(
        "--distance-bins",
        type=str,
        default=":21",
        help="Distance bins: N, :N, M:N (default: :21 for D=0-20)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for plots (shows interactively if not set)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=10, help="Maximum frames to plot (default: 10)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for intensity plots (default: viridis)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved plots (default: 150)")
    parser.add_argument(
        "--radar-height",
        type=float,
        default=None,
        help="Radar height above water (m) for ground distance calculation",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML configuration file for bearing calculation",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="polar",
        choices=["polar", "ship", "earth"],
        help="Initial view type for intensity viewer (default: polar)",
    )


def add_subparser(subparsers) -> None:
    """Register the 'view' subcommand."""
    p = subparsers.add_parser(
        "view",
        help="View raw intensity data",
        description="Load WAMOS polar files in time-based groups",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'view' command."""
    from pathlib import Path
    import time

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Check if any plotting requested
    do_plot = (
        args.plot_intensity or args.plot_bits or args.plot_bits_detail or args.plot_bits_across
    )

    if do_plot:
        import matplotlib

        if args.output_dir:
            matplotlib.use("Agg")  # Non-interactive backend for saving
        import matplotlib.pyplot as plt

        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving plots to: {output_path}")

    t0 = time.time()

    files = Files(
        stime=args.stime,
        etime=args.etime,
        polar_path=args.polar_path,
        groupby=args.groupby,
        workers=args.workers,
    )

    t1 = time.time()
    logging.info(f"Discovered {len(files)} files in {t1 - t0:.3f}s")
    logging.debug(f"{files}")

    if args.load or do_plot:
        logging.info("Loading files by group...")
        t2 = time.time()
        total_frames = 0
        frames_collected = []  # Collect frames for interactive viewer

        for period, frames in files.itergroups():
            logging.debug(f"  {period}: loaded {len(frames)} frames")
            total_frames += len(frames)

            if do_plot:
                for frame in frames:
                    if len(frames_collected) >= args.max_frames:
                        break
                    frames_collected.append(frame)

                if len(frames_collected) >= args.max_frames:
                    logging.info(f"Reached max frames ({args.max_frames}), stopping load.")
                    break

        t3 = time.time()
        logging.info(f"Loaded {total_frames} frames in {t3 - t2:.3f}s")

        if do_plot and frames_collected:
            logging.info(f"Plotting {len(frames_collected)} frames...")

            # Load configuration if provided
            config = Config(args.config) if args.config else Config()

            # Interactive intensity viewer with navigation
            if args.plot_intensity and not args.output_dir:
                viewer = IntensityViewer(
                    frames_collected,
                    cmap=args.cmap,
                    radar_height=args.radar_height,
                    config=config,
                    view=args.view,
                )
                logging.info("Navigation: <- -> keys or Prev/Next buttons (wraps around)")
                logging.info("Views: 1=Polar, 2=Ship, 3=Earth (or click buttons)")
                viewer.show()

            # File output mode or other plot types
            else:
                # Parse distance bins specification
                max_bins = frames_collected[0].n_distances
                distance_bins = parse_distance_bins(args.distance_bins, max_bins)

                frames_plotted = 0
                for frame in frames_collected:
                    ts_str = str(frame.timestamp).replace(":", "-").replace(" ", "_")

                    # Plot intensity (file output only - interactive handled above)
                    if args.plot_intensity and args.output_dir:
                        fig, ax, im = plot_frame_intensity(
                            frame, cmap=args.cmap, radar_height=args.radar_height, config=config
                        )
                        fname = output_path / f"intensity_{ts_str}.png"
                        fig.savefig(fname, dpi=args.dpi, bbox_inches="tight")
                        logging.debug(f"    Saved: {fname.name}")
                        plt.close(fig)

                    # Plot top 4 bits
                    if args.plot_bits:
                        fig, axes = plot_frame_bits(frame, distance_bins=distance_bins)
                        if args.output_dir:
                            fname = output_path / f"bits_{ts_str}.png"
                            fig.savefig(fname, dpi=args.dpi, bbox_inches="tight")
                            logging.debug(f"    Saved: {fname.name}")
                            plt.close(fig)
                        else:
                            plt.show()

                    # Plot detailed bits view
                    if args.plot_bits_detail:
                        fig, axes = plot_distance_bins_detail(frame, distance_bins=distance_bins)
                        if args.output_dir:
                            fname = output_path / f"bits_detail_{ts_str}.png"
                            fig.savefig(fname, dpi=args.dpi, bbox_inches="tight")
                            logging.debug(f"    Saved: {fname.name}")
                            plt.close(fig)
                        else:
                            plt.show()

                    frames_plotted += 1

                logging.info(f"Plotted {frames_plotted} frames")

                # Plot bits across all frames
                if args.plot_bits_across and frames_collected:
                    fig, axes = plot_bits_across_frames(
                        frames_collected, distance_bins=distance_bins
                    )
                    if args.output_dir:
                        bin_spec = args.distance_bins.replace(":", "-")
                        fname = output_path / f"bits_across_D{bin_spec}.png"
                        fig.savefig(fname, dpi=args.dpi, bbox_inches="tight")
                        logging.debug(f"    Saved: {fname.name}")
                        plt.close(fig)
                    else:
                        plt.show()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Load WAMOS polar files in time-based groups")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
