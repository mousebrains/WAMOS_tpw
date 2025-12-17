#! /usr/bin/env python3
#
# Destreak class for removing radial streak artifacts from WAMOS polar frames
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging

import numpy as np

from wamos_tpw.config import WamosConfig
from wamos_tpw.frame import Frame


class Destreak:
    """
    Remove radial streak artifacts from radar frames using angular gradient analysis.

    Streaks appear as radial lines of anomalously high or low intensity,
    typically caused by interference or hardware artifacts. The algorithm
    detects streaks by looking for sharp intensity transitions along the
    bearing direction (spikes that go up then down quickly).

    Algorithm (from Matlab tpw01.m):
    1. Compute angular derivative (intensity change between adjacent bearings)
    2. Compute second difference to find spike patterns (large positive followed by negative)
    3. Find threshold dynamically from histogram minimum (valley between normal and streak values)
    4. Require at least 10 contiguous flagged bins per radial to confirm as streak
    5. Replace streak pixels with moving average of neighbors

    The prev_frame and next_frame parameters are reserved for future
    temporal-based destreaking extensions.

    Example:
        >>> config = WamosConfig()
        >>> destreak = Destreak(
        ...     prev_frame=frame_t0,
        ...     center_frame=frame_t1,
        ...     next_frame=frame_t2,
        ...     config=config
        ... )
        >>> corrected = destreak.corrected_intensity
    """

    # Algorithm constants
    _FILL_WINDOW = 3  # Window size for moving average fill
    _MIN_STREAK_LENGTH = 10  # Minimum number of contiguous flagged bins required
    _HISTOGRAM_BINS = 100  # Number of bins for threshold histogram
    _THRESHOLD_SIGMA = 7.5  # Number of one-sided standard deviations for threshold

    def __init__(self,
                 prev_frame: Frame | None,
                 center_frame: Frame,
                 next_frame: Frame | None,
                 config: WamosConfig | None = None):
        """
        Initialize destreaking with temporal frame triplet.

        Args:
            prev_frame: Previous frame in time (None if not available, reserved for future use)
            center_frame: Frame to be destreaked
            next_frame: Next frame in time (None if not available, reserved for future use)
            config: WamosConfig for algorithm parameters
        """
        if center_frame is None:
            raise ValueError("center_frame is required")

        self._prev = prev_frame
        self._center = center_frame
        self._next = next_frame
        self._config = config or WamosConfig()

        # Get parameters from config (with class defaults as fallback)
        self._min_streak_length = self._config.destreak.min_streak_length
        self._threshold_sigma = self._config.destreak.threshold_sigma

        # Results (computed lazily)
        self._corrected: np.ndarray | None = None
        self._streak_mask: np.ndarray | None = None
        self._derivative: np.ndarray | None = None  # For diagnostics
        self._threshold: float | None = None  # Dynamic threshold value
        self._one_sided_std: float | None = None  # One-sided standard deviation

    @property
    def center_frame(self) -> Frame:
        """Return the center frame being destreaked."""
        return self._center

    @property
    def corrected_intensity(self) -> np.ndarray:
        """
        Return the destreaked intensity data.

        Returns:
            2D array of corrected intensity values
        """
        if self._corrected is None:
            self._compute_destreak()
        return self._corrected

    @property
    def streak_mask(self) -> np.ndarray:
        """
        Return boolean mask indicating detected streaks.

        Returns:
            2D boolean array (True = streak detected)
        """
        if self._streak_mask is None:
            self._compute_destreak()
        return self._streak_mask

    def _compute_destreak(self) -> None:
        """
        Compute destreaking using angular gradient analysis.

        Streaks show up as bright returns with a constant angle (radial lines).
        The algorithm detects them by looking for a derivative signature where
        there's a large positive value followed by a large negative value
        (i.e., intensity spikes along the bearing direction).

        Edge handling:
        - If prev_frame exists: prepend its last theta row to enable streak
          detection in center_frame's first theta bin
        - If next_frame exists: append its first theta row to enable streak
          detection in center_frame's last theta bin
        - If prev_frame is None: assume no streaks in first theta bin
        - If next_frame is None: assume no streaks in last theta bin

        The second difference operation (detecting spikes) at index i is centered
        on the original data at index i+1. So:
        - With prev_frame: we can detect streaks starting at center[0]
        - Without prev_frame: detection starts at center[1], pad False at start
        - With next_frame: we can detect streaks up to center[n-1]
        - Without next_frame: detection ends at center[n-2], pad False at end

        Based on destreak_frame() from tpw01.m:
        - Matlab frame shape: (range, theta) - operations along theta (dim 2)
        - Python frame shape: (bearings, distances) - operations along bearings (axis 0)
        """
        # Get intensity as float for calculations
        # Use deramped_intensity if available, otherwise use intensity
        # Frame shape: (n_bearings, n_distances)
        deramped = getattr(self._center, 'deramped_intensity', None)
        if deramped is not None:
            center_data = deramped.astype(np.float64)
        else:
            center_data = self._center.intensity.astype(np.float64)
        n_bearings, n_distances = center_data.shape

        # Build extended frame with neighbor data for edge detection
        has_prev = self._prev is not None
        has_next = self._next is not None

        frame = center_data

        if has_prev:
            # Prepend last theta row from previous frame
            prev_intensity = getattr(self._prev, 'deramped_intensity',
                                     self._prev.intensity)
            prev_last = prev_intensity[-1:, :].astype(np.float64)
            frame = np.vstack([prev_last, frame])

        if has_next:
            # Append first theta row from next frame
            next_intensity = getattr(self._next, 'deramped_intensity',
                                     self._next.intensity)
            next_first = next_intensity[:1, :].astype(np.float64)
            frame = np.vstack([frame, next_first])

        # Calculate derivative along bearing direction
        # Matlab: a = diff(frame, 1, 2)  -> diff along theta (dim 2)
        # Python: axis 0 is bearings (theta equivalent)
        a = np.diff(frame, axis=0)

        # Enhance adjacent bins across streak: find spike patterns
        # Where there's a large positive derivative followed by a large negative one
        # Matlab: a = a(:,1:end-1) - a(:,2:end)
        # second_diff[i] is centered on frame[i+1]
        a = a[:-1, :] - a[1:, :]

        # Keep only positive values (streak signature has this pattern)
        a = np.maximum(a, 0)

        # Dynamically determine threshold by finding histogram minimum
        # The histogram typically has a peak of normal values and a tail of streak values
        # The minimum between them is a natural threshold point
        threshold = self._find_histogram_threshold(a)

        # Create mask for streak locations
        q = a > threshold

        # Filter out radials that don't have at least min_streak_length contiguous flagged bins
        # A real streak should have a continuous run of detections, not scattered pixels
        for i in range(q.shape[0]):
            if not self._has_contiguous_run(q[i, :], self._min_streak_length):
                q[i, :] = False  # Unflag radials without sufficient contiguous detections

        # Store derivative for diagnostics
        self._derivative = a

        # The mask q now covers:
        # - With both neighbors: center indices 0 to n-1 (full coverage)
        # - With prev only: center indices 0 to n-2
        # - With next only: center indices 1 to n-1
        # - With neither: center indices 1 to n-2

        # Pad to match center_frame dimensions
        if not has_prev:
            # No prev frame - assume no streak in first theta bin
            q = np.vstack([np.zeros((1, n_distances), dtype=bool), q])

        if not has_next:
            # No next frame - assume no streak in last theta bin
            q = np.vstack([q, np.zeros((1, n_distances), dtype=bool)])

        # Ensure q matches center_frame dimensions
        assert q.shape == (n_bearings, n_distances), \
            f"Mask shape {q.shape} doesn't match frame shape {(n_bearings, n_distances)}"

        # Copy center frame and mark streak pixels as NaN
        b = center_data.copy()
        b[q] = np.nan

        # Fill missing values with moving mean along bearing direction
        # Matlab: c = fillmissing(b, "movmean", 3, 2) - along theta (dim 2)
        # Python: along axis 0 (bearings)
        c = self._fill_missing_movmean(b, window=self._FILL_WINDOW, axis=0)

        self._corrected = c
        self._streak_mask = q

    def _find_histogram_threshold(self, derivative: np.ndarray) -> float:
        """
        Find threshold using one-sided standard deviation.

        The one-sided standard deviation is calculated from values below the median,
        which represents the "normal" variation without being skewed by streak outliers
        in the upper tail. The threshold is set to N sigma above zero.

        Args:
            derivative: 2D array of derivative values (after max(a, 0))

        Returns:
            Threshold value for streak detection
        """
        # Flatten and get positive values only
        deriv_flat = derivative.ravel()
        deriv_pos = deriv_flat[deriv_flat > 0]

        if len(deriv_pos) == 0:
            self._threshold = 0.0
            self._one_sided_std = 0.0
            return 0.0

        # Calculate one-sided standard deviation using values below the median
        # This gives the "normal" spread without streak outlier contamination
        median = np.median(deriv_pos)
        lower_half = deriv_pos[deriv_pos <= median]

        if len(lower_half) > 1:
            # One-sided std: sqrt(mean of squared deviations from median, for lower half)
            one_sided_std = np.sqrt(np.mean((lower_half - median) ** 2))
        else:
            one_sided_std = np.std(deriv_pos)

        # Threshold is N sigma above zero (median + N * one_sided_std as reference)
        threshold = self._threshold_sigma * one_sided_std

        # Store for diagnostics
        self._threshold = threshold
        self._one_sided_std = one_sided_std

        return threshold

    @staticmethod
    def _has_contiguous_run(arr: np.ndarray, min_length: int) -> bool:
        """
        Check if a 1D boolean array has a contiguous run of True values.

        Args:
            arr: 1D boolean array
            min_length: Minimum length of contiguous True values required

        Returns:
            True if there's at least one run of min_length consecutive True values
        """
        if not arr.any():
            return False

        # Find runs of consecutive True values
        # Pad with False at ends to detect runs at boundaries
        padded = np.concatenate([[False], arr, [False]])
        # Find where values change
        changes = np.diff(padded.astype(int))
        # Rising edges (0->1) mark start of runs, falling edges (1->0) mark ends
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        # Calculate run lengths
        run_lengths = ends - starts
        return np.any(run_lengths >= min_length)

    @staticmethod
    def _fill_missing_movmean(data: np.ndarray,
                               window: int = 3,
                               axis: int = 0) -> np.ndarray:
        """
        Fill NaN values using moving mean along specified axis.

        For each NaN value, computes the mean of non-NaN values within
        the window centered on that position.

        Args:
            data: Input array with NaN values to fill
            window: Size of the moving average window (should be odd)
            axis: Axis along which to compute moving mean

        Returns:
            Array with NaN values filled
        """
        result = data.copy()
        half_window = window // 2

        # Get indices of NaN values
        nan_mask = np.isnan(data)

        if not nan_mask.any():
            return result

        # Get coordinates of NaN values
        nan_coords = np.argwhere(nan_mask)

        for coord in nan_coords:
            # Build slice for the window along the specified axis
            slices = [slice(None)] * data.ndim

            # Window bounds along the fill axis
            idx = coord[axis]
            start = max(0, idx - half_window)
            end = min(data.shape[axis], idx + half_window + 1)
            slices[axis] = slice(start, end)

            # For other axes, use the specific coordinate
            for i, c in enumerate(coord):
                if i != axis:
                    slices[i] = c

            # Get window values and compute mean of non-NaN values
            window_vals = data[tuple(slices)]
            valid_vals = window_vals[~np.isnan(window_vals)]

            if len(valid_vals) > 0:
                result[tuple(coord)] = np.mean(valid_vals)
            # If all values in window are NaN, leave as NaN (will be handled by edge cases)

        # Handle any remaining NaN values (edge cases) with nearest valid value
        remaining_nans = np.isnan(result)
        if remaining_nans.any():
            # Simple fill with axis-wise interpolation
            from scipy.ndimage import generic_filter

            def nanmean_filter(x):
                valid = x[~np.isnan(x)]
                return np.mean(valid) if len(valid) > 0 else np.nan

            # This is slower but handles edge cases
            filled = generic_filter(result, nanmean_filter, size=window,
                                    mode='nearest')
            result[remaining_nans] = filled[remaining_nans]

        return result

    def plot_diagnostics(self,
                         figsize: tuple[float, float] = (16, 10),
                         cmap: str = 'viridis') -> None:
        """
        Plot diagnostic comparison of before and after destreaking.

        Creates a 2x3 figure showing:
        - Top left: Original intensity
        - Top center: Destreaked intensity
        - Top right: Histogram of derivative values
        - Bottom left: Streak mask
        - Bottom center: Difference (original - destreaked)
        - Bottom right: Info text

        Args:
            figsize: Figure size (width, height)
            cmap: Colormap for intensity plots
        """
        import matplotlib.pyplot as plt
        from wamos_tpw.plotting import quantile_limits

        # Ensure computation is done
        # Use deramped_intensity if available, otherwise use intensity
        original = getattr(self._center, 'deramped_intensity',
                           self._center.intensity).astype(np.float64)
        corrected = self.corrected_intensity
        mask = self.streak_mask
        derivative = self._derivative

        # Calculate common color limits
        vmin, vmax = quantile_limits(original)

        # Create figure with linked axes for image panels only
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Destreak Diagnostics: {self._center.timestamp}', fontsize=14)

        # Create axes - images share x/y, histogram is separate
        ax_orig = fig.add_subplot(2, 3, 1)
        ax_destreaked = fig.add_subplot(2, 3, 2, sharex=ax_orig, sharey=ax_orig)
        ax_hist = fig.add_subplot(2, 3, 3)
        ax_mask = fig.add_subplot(2, 3, 4, sharex=ax_orig, sharey=ax_orig)
        ax_diff = fig.add_subplot(2, 3, 5, sharex=ax_orig, sharey=ax_orig)
        ax_info = fig.add_subplot(2, 3, 6)

        # Top left: Original
        im = ax_orig.imshow(original, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax_orig.set_title('Original Intensity')
        ax_orig.set_xlabel('Distance bin')
        ax_orig.set_ylabel('Bearing bin')
        plt.colorbar(im, ax=ax_orig, label='Intensity')

        # Top center: Destreaked
        im = ax_destreaked.imshow(corrected, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax_destreaked.set_title('Destreaked Intensity')
        ax_destreaked.set_xlabel('Distance bin')
        ax_destreaked.set_ylabel('Bearing bin')
        plt.colorbar(im, ax=ax_destreaked, label='Intensity')

        # Top right: Histogram of derivative
        if derivative is not None:
            deriv_flat = derivative.ravel()
            deriv_flat = deriv_flat[deriv_flat > 0]  # Only positive values (after max(a, 0))
            if len(deriv_flat) > 0:
                ax_hist.hist(deriv_flat, bins=self._HISTOGRAM_BINS, color='steelblue',
                            alpha=0.7, edgecolor='none')
                # Show threshold (N × one-sided std)
                if self._threshold is not None:
                    ax_hist.axvline(self._threshold, color='red', linestyle='--', linewidth=2,
                                   label=f'Threshold ({self._threshold_sigma:.1f}σ): {self._threshold:.1f}')
                ax_hist.set_xlabel('Derivative value')
                ax_hist.set_ylabel('Count')
                ax_hist.set_title('Derivative Histogram')
                ax_hist.legend(loc='upper right', fontsize=8)
                ax_hist.set_yscale('log')
            else:
                ax_hist.text(0.5, 0.5, 'No positive\nderivative values',
                            ha='center', va='center', transform=ax_hist.transAxes)
                ax_hist.set_title('Derivative Histogram')
        else:
            ax_hist.text(0.5, 0.5, 'No derivative data',
                        ha='center', va='center', transform=ax_hist.transAxes)
            ax_hist.set_title('Derivative Histogram')

        # Bottom left: Streak mask
        im = ax_mask.imshow(mask, aspect='auto', cmap='Reds')
        ax_mask.set_title(f'Streak Mask ({mask.sum()} pixels, {100*mask.sum()/mask.size:.2f}%)')
        ax_mask.set_xlabel('Distance bin')
        ax_mask.set_ylabel('Bearing bin')
        plt.colorbar(im, ax=ax_mask, label='Streak detected')

        # Bottom center: Difference
        diff = original - corrected
        diff_max = max(abs(diff.min()), abs(diff.max()))
        if diff_max > 0:
            im = ax_diff.imshow(diff, aspect='auto', cmap='RdBu_r',
                               vmin=-diff_max, vmax=diff_max)
        else:
            im = ax_diff.imshow(diff, aspect='auto', cmap='RdBu_r')
        ax_diff.set_title('Difference (Original - Destreaked)')
        ax_diff.set_xlabel('Distance bin')
        ax_diff.set_ylabel('Bearing bin')
        plt.colorbar(im, ax=ax_diff, label='Difference')

        # Bottom right: Info text
        ax_info.axis('off')
        info_lines = [
            f'Frame: {self._center.metadata.filename}',
            f'Shape: {original.shape}',
            f'Prev frame: {"Yes" if self._prev else "No"}',
            f'Next frame: {"Yes" if self._next else "No"}',
            f'Min contiguous streak: {self._min_streak_length} bins',
            '',
            f'Threshold ({self._threshold_sigma:.1f}σ):',
            f'  One-sided std: {self._one_sided_std:.1f}' if self._one_sided_std else '  One-sided std: N/A',
            f'  Threshold: {self._threshold:.1f}' if self._threshold else '  Threshold: N/A',
            '',
            f'Streaks detected: {mask.sum()} pixels',
            f'Percentage: {100*mask.sum()/mask.size:.3f}%',
        ]
        if derivative is not None:
            deriv_flat = derivative.ravel()
            deriv_pos = deriv_flat[deriv_flat > 0]
            if len(deriv_pos) > 0:
                info_lines.extend([
                    '',
                    'Derivative stats (positive):',
                    f'  Min: {deriv_pos.min():.1f}',
                    f'  Max: {deriv_pos.max():.1f}',
                    f'  Mean: {deriv_pos.mean():.1f}',
                ])
        ax_info.text(0.1, 0.95, '\n'.join(info_lines), fontsize=10,
                    verticalalignment='top', family='monospace',
                    transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        has_prev = self._prev is not None
        has_next = self._next is not None
        return f"Destreak(prev={has_prev}, center={self._center.timestamp}, next={has_next})"


def destreak_frame(prev_frame: Frame | None,
                   center_frame: Frame,
                   next_frame: Frame | None,
                   config: WamosConfig | None = None) -> np.ndarray:
    """
    Convenience function to destreak a single frame.

    Args:
        prev_frame: Previous frame in time (None if not available)
        center_frame: Frame to be destreaked
        next_frame: Next frame in time (None if not available)
        config: WamosConfig for algorithm parameters

    Returns:
        2D array of destreaked intensity values
    """
    ds = Destreak(prev_frame, center_frame, next_frame, config)
    return ds.corrected_intensity


def add_subparser(subparsers) -> None:
    """Register the 'destreak' subcommand."""
    p = subparsers.add_parser(
        'destreak',
        help='Standalone destreak tool',
        description="Test destreak algorithm"
    )
    p.add_argument("polar_files", nargs="+", help="Polar files to process")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="YAML configuration file")
    p.add_argument("--plot", "-p", action="store_true",
                   help="Show diagnostic plots for each frame")
    p.add_argument("--cmap", type=str, default="viridis",
                   help="Colormap for plots (default: viridis)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose output")
    p.add_argument("--deramp", "-d", action="store_true",
                   help="Apply deramping before destreaking")
    p.add_argument("--quantile", "-q", type=float, default=0.10,
                   help="Deramp quantile (0.0-1.0, default: 0.10)")
    p.add_argument("--smooth-window", "-s", type=int, default=None,
                   help="Deramp smoothing window in bins (default: 2%% of range bins)")
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'destreak' command."""
    from wamos_tpw.polarfile import load_polar_file

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # Load config
    config = WamosConfig(args.config) if args.config else WamosConfig()

    # Load frames
    frames = []
    for filepath in args.polar_files:
        frame = load_polar_file(filepath)
        if frame is not None:
            frames.append(frame)
            print(f"Loaded: {filepath} -> {frame.timestamp}")

    if len(frames) < 1:
        print("No frames loaded")
        return

    print(f"\nLoaded {len(frames)} frames")

    # Apply deramping if requested
    if args.deramp:
        from wamos_tpw.deramp import Deramp
        print(f"\nApplying deramp (quantile={args.quantile*100:.0f}%, smooth_window={args.smooth_window or 'auto'})")
        for frame in frames:
            deramp = Deramp(frame, config, quantile=args.quantile, smooth_window=args.smooth_window)
            frame.deramped_intensity = deramp.corrected_intensity

    # Process each frame with its neighbors
    for i, center in enumerate(frames):
        prev_frame = frames[i - 1] if i > 0 else None
        next_frame = frames[i + 1] if i < len(frames) - 1 else None

        ds = Destreak(prev_frame, center, next_frame, config)
        print(f"\nFrame {i}: {ds}")
        print(f"  Original intensity range: [{center.intensity.min():.1f}, {center.intensity.max():.1f}]")

        corrected = ds.corrected_intensity
        print(f"  Corrected intensity range: [{corrected.min():.1f}, {corrected.max():.1f}]")

        n_streaks = ds.streak_mask.sum()
        total_pixels = ds.streak_mask.size
        print(f"  Streaks detected: {n_streaks} / {total_pixels} ({100*n_streaks/total_pixels:.2f}%)")

        if args.plot:
            ds.plot_diagnostics(cmap=args.cmap)


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Test destreak algorithm")
    parser.add_argument("polar_files", nargs="+", help="Polar files to process")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML configuration file")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Show diagnostic plots for each frame")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Colormap for plots (default: viridis)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--deramp", "-d", action="store_true",
                        help="Apply deramping before destreaking")
    parser.add_argument("--quantile", "-q", type=float, default=0.10,
                        help="Deramp quantile (0.0-1.0, default: 0.10)")
    parser.add_argument("--smooth-window", "-s", type=int, default=None,
                        help="Deramp smoothing window in bins (default: 2%% of range bins)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
