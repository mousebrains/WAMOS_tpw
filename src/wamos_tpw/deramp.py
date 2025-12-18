#! /usr/bin/env python3
#
# Deramp class for removing range-dependent intensity fall-off
# Subtracts smoothed quantile profile from intensity data
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging

import numpy as np

from wamos_tpw.config import WamosConfig
from wamos_tpw.frame import Frame


class Deramp:
    """
    Remove range-dependent intensity fall-off from radar data.

    Calculates a quantile intensity profile as a function of range
    (excluding shadowed regions), smooths it, and subtracts from the data.

    Algorithm Overview
    ------------------

    Marine radar intensity decreases with range due to the radar equation:
    received power falls off as 1/r⁴ (for surface targets). This creates
    a radial intensity gradient that obscures sea surface features.

    **Step 1: Shadow Mask Creation**

    The ship's superstructure blocks radar returns in a sector (typically
    around 180° from bow). These bearings are excluded from profile
    calculation since they don't represent true sea surface returns.

    The shadow region is defined by:
    - center: Direction of shadow center from bow (default: 180° = stern)
    - width: Total angular width of shadow (default: 90°)

    The mask handles 360°/0° wraparound correctly.

    **Step 2: Quantile Profile Calculation**

    For each range bin (column), compute the specified quantile (default: 10th
    percentile) of intensity values across all non-shadowed bearings. The
    quantile is used instead of the mean because:

    - Ocean waves create high-intensity returns that skew the mean
    - The lower quantile better represents the "background" intensity level
    - It's more robust to outliers from ships, rain, or artifacts

    The result is a 1D profile: intensity vs. range.

    **Step 3: Profile Smoothing**

    Apply a moving average filter to the profile to reduce noise while
    preserving the overall trend. The window size is 2% of the range bins
    (minimum 3 bins). Edge handling uses 'edge' padding mode to avoid
    boundary artifacts.

    **Step 4: Correction**

    Subtract the smoothed profile from each bearing's intensity:
        corrected[bearing, range] = intensity[bearing, range] - profile[range]

    This flattens the range-dependent falloff while preserving relative
    intensity variations (waves, ships, etc.).

    Example:
        >>> deramp = Deramp(frame, config, quantile=0.25)
        >>> corrected = deramp.corrected_intensity
        >>> deramp.plot_diagnostics()  # Optional visualization
    """

    # Default smoothing window as fraction of range bins
    _SMOOTH_WINDOW_FRACTION = 0.02  # 2% of range bins

    def __init__(
        self,
        frame: Frame,
        config: WamosConfig | None = None,
        bearing: np.ndarray | None = None,
        quantile: float = 0.10,
        smooth_window: int | None = None,
        shadow_start: float | None = None,
        shadow_end: float | None = None,
    ) -> None:
        """
        Initialize Deramp for a single frame.

        Args:
            frame: Frame object with intensity data to correct
            config: WamosConfig for shadow region parameters
            bearing: Optional bearing array for each row (degrees).
                     If None, assumes uniform 0-360 distribution.
            quantile: Quantile to use for profile (0.0-1.0, default 0.10)
            smooth_window: Smoothing window size in bins (default: 2% of range bins)
            shadow_start: Detected shadow start angle (degrees). Overrides config if provided.
            shadow_end: Detected shadow end angle (degrees). Overrides config if provided.
        """
        self._frame = frame
        self._config = config or WamosConfig()
        self._bearing = bearing
        self._quantile = quantile
        # Store detected shadow bounds (override config if provided)
        self._shadow_start = shadow_start
        self._shadow_end = shadow_end
        # Calculate default window as 2% of range bins, minimum 3
        if smooth_window is None:
            smooth_window = max(3, int(frame.n_distances * self._SMOOTH_WINDOW_FRACTION))
        self._smooth_window = smooth_window

        # Results (computed lazily)
        self._corrected: np.ndarray | None = None
        self._raw_profile: np.ndarray | None = None
        self._smooth_profile: np.ndarray | None = None
        self._shadow_mask: np.ndarray | None = None
        self._slant_range: np.ndarray | None = None

    @property
    def quantile(self) -> float:
        """Return the quantile used for profile calculation."""
        return self._quantile

    @property
    def corrected_intensity(self) -> np.ndarray:
        """Return the range-corrected intensity array."""
        if self._corrected is None:
            self._compute_deramp()
        return self._corrected

    @property
    def raw_profile(self) -> np.ndarray:
        """Return the raw quantile range profile (before smoothing)."""
        if self._raw_profile is None:
            self._compute_deramp()
        return self._raw_profile

    @property
    def smooth_profile(self) -> np.ndarray:
        """Return the smoothed range profile used for correction."""
        if self._smooth_profile is None:
            self._compute_deramp()
        return self._smooth_profile

    @property
    def slant_range(self) -> np.ndarray:
        """Return the slant range values in meters."""
        if self._slant_range is None:
            self._slant_range = self._frame.slant_range()
        return self._slant_range

    @property
    def shadow_mask(self) -> np.ndarray:
        """Return boolean mask of shadow region (True = shadowed)."""
        if self._shadow_mask is None:
            self._compute_shadow_mask()
        return self._shadow_mask

    def _compute_shadow_mask(self) -> None:
        """Compute mask for shadow region based on detected edges or config."""
        n_bearings = self._frame.n_bearings

        # Use detected shadow bounds if provided, otherwise fall back to config
        if self._shadow_start is not None and self._shadow_end is not None:
            # Use detected shadow edges directly
            shadow_start = self._shadow_start
            shadow_end = self._shadow_end
        else:
            # Fall back to config-based shadow region
            shadow_center = self._config.shadow.center  # degrees from bow
            shadow_width = self._config.shadow.width  # total width in degrees

            if shadow_center is None or shadow_width is None or shadow_width <= 0:
                # No shadow defined - no masking
                self._shadow_mask = np.zeros(n_bearings, dtype=bool)
                return

            # Calculate shadow bounds from center and width
            half_width = shadow_width / 2.0
            shadow_start = (shadow_center - half_width) % 360
            shadow_end = (shadow_center + half_width) % 360

        # Determine bearing for each row
        if self._bearing is not None:
            bearings = self._bearing
        else:
            # Assume uniform distribution 0-360
            bearings = np.linspace(0, 360, n_bearings, endpoint=False)

        # Create mask (handle wrap-around)
        if shadow_start < shadow_end:
            # Normal case: shadow doesn't wrap around 360
            self._shadow_mask = (bearings >= shadow_start) & (bearings <= shadow_end)
        else:
            # Shadow wraps around 360
            self._shadow_mask = (bearings >= shadow_start) | (bearings <= shadow_end)

    def _compute_deramp(self) -> None:
        """Compute the range-corrected intensity using smoothed quantile profile."""
        # Use float32 - sufficient for 12-bit intensity data (0-4095), saves 50% memory
        intensity = self._frame.intensity.astype(np.float32)
        n_bearings, n_distances = intensity.shape

        # Get shadow mask
        shadow_mask = self.shadow_mask

        # Select non-shadow bearings for profile calculation
        non_shadow_idx = ~shadow_mask
        if non_shadow_idx.sum() == 0:
            # All bearings are shadowed - use all data
            non_shadow_idx = np.ones(n_bearings, dtype=bool)

        # Calculate quantile profile at each range bin (excluding shadow)
        non_shadow_data = intensity[non_shadow_idx, :]
        self._raw_profile = self._fast_quantile(non_shadow_data, self._quantile)

        # Smooth the profile
        self._smooth_profile = self._smooth_moving_average(self._raw_profile, self._smooth_window)

        # Subtract the smoothed profile from intensity data
        self._corrected = intensity - self._smooth_profile[np.newaxis, :]

    @staticmethod
    def _fast_quantile(data: np.ndarray, q: float, max_samples: int = 500) -> np.ndarray:
        """
        Compute quantile along axis 0 using np.partition for O(n) performance.

        Much faster than np.quantile for large arrays since partition is O(n)
        vs O(n log n) for full sorting. Uses vectorized operations across all
        columns simultaneously.

        For arrays with many rows (bearings), uses strided sampling to reduce
        computation while maintaining statistical accuracy.

        Args:
            data: 2D array of shape (n_samples, n_features)
            q: Quantile value between 0 and 1
            max_samples: Maximum samples to use along axis 0 (default 500)

        Returns:
            1D array of quantile values for each column
        """
        n_samples, n_features = data.shape

        if n_samples == 0:
            return np.zeros(n_features)

        if n_samples == 1:
            return data[0, :].copy()

        # Use strided sampling for large arrays (3x speedup for typical radar data)
        if n_samples > max_samples * 2:
            stride = n_samples // max_samples
            data = data[::stride, :]
            n_samples = data.shape[0]

        # Calculate indices for linear interpolation
        idx = q * (n_samples - 1)
        k_low = int(np.floor(idx))
        k_high = min(k_low + 1, n_samples - 1)
        frac = idx - k_low

        # Transpose to (n_features, n_samples) for row-wise partition
        data_t = data.T

        if k_high > k_low and frac > 0:
            # Need both k_low and k_high for interpolation
            # Partition both indices simultaneously in O(n) - more efficient than
            # partitioning once then scanning for min
            partitioned = np.partition(data_t, [k_low, k_high], axis=1)
            low_vals = partitioned[:, k_low]
            high_vals = partitioned[:, k_high]
            return low_vals + frac * (high_vals - low_vals)
        else:
            # Only need k_low
            partitioned = np.partition(data_t, k_low, axis=1)
            return partitioned[:, k_low]

    @staticmethod
    def _smooth_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing with edge handling."""
        if window <= 1:
            return data.copy()

        # Use convolution for moving average
        kernel = np.ones(window) / window
        # Pad edges to avoid boundary effects
        pad_width = window // 2
        padded = np.pad(data, pad_width, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

        # Ensure output length matches input
        if len(smoothed) > len(data):
            smoothed = smoothed[: len(data)]
        elif len(smoothed) < len(data):
            smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode="edge")

        return smoothed

    def plot_diagnostics(self, figsize: tuple[float, float] = (14, 10)) -> None:
        """
        Show diagnostic plots for the deramping process.

        Displays:
        - Original intensity image
        - Corrected intensity image
        - Range profiles (raw and smoothed)
        - Smoothing residuals
        - Histogram comparison
        """
        import matplotlib.pyplot as plt

        # Ensure computation is done
        _ = self.corrected_intensity

        original = self._frame.intensity.astype(np.float32)
        corrected = self._corrected
        distances = self.slant_range

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f"Deramp Diagnostics: {self._frame.metadata.filename}", fontsize=12)

        # Top left: Original intensity
        ax_orig = axes[0, 0]
        vmin, vmax = np.percentile(original, [1, 99])
        im1 = ax_orig.imshow(
            original,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=[distances[0], distances[-1], original.shape[0], 0],
        )
        ax_orig.set_title("Original Intensity")
        ax_orig.set_xlabel("Range (m)")
        ax_orig.set_ylabel("Bearing bin")
        plt.colorbar(im1, ax=ax_orig)

        # Top middle: Corrected intensity
        ax_corr = axes[0, 1]
        vmin_c, vmax_c = np.percentile(corrected, [1, 99])
        im2 = ax_corr.imshow(
            corrected,
            aspect="auto",
            cmap="viridis",
            vmin=vmin_c,
            vmax=vmax_c,
            extent=[distances[0], distances[-1], corrected.shape[0], 0],
        )
        ax_corr.set_title("Corrected Intensity")
        ax_corr.set_xlabel("Range (m)")
        ax_corr.set_ylabel("Bearing bin")
        plt.colorbar(im2, ax=ax_corr)

        # Top right: Range profiles
        ax_prof = axes[0, 2]
        ax_prof.plot(
            distances, self._raw_profile, "b-", alpha=0.5, label=f"Raw {self._quantile * 100:.0f}%"
        )
        ax_prof.plot(
            distances,
            self._smooth_profile,
            "r-",
            linewidth=2,
            label=f"Smoothed ({self._smooth_window} bins)",
        )
        ax_prof.set_xlabel("Range (m)")
        ax_prof.set_ylabel("Intensity")
        ax_prof.set_title("Range Profile (excl. shadow)")
        ax_prof.legend()
        ax_prof.grid(True, alpha=0.3)

        # Bottom left: Smoothing residuals
        ax_resid = axes[1, 0]
        residuals = self._raw_profile - self._smooth_profile
        ax_resid.plot(distances, residuals, "g-", linewidth=1)
        ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_resid.set_xlabel("Range (m)")
        ax_resid.set_ylabel("Residual (raw - smooth)")
        ax_resid.set_title("Smoothing Residuals")
        ax_resid.grid(True, alpha=0.3)

        # Bottom middle: Histogram comparison
        ax_hist = axes[1, 1]
        ax_hist.hist(original.ravel(), bins=100, alpha=0.5, label="Original", density=True)
        ax_hist.hist(corrected.ravel(), bins=100, alpha=0.5, label="Corrected", density=True)
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Intensity Distribution")
        ax_hist.legend()
        ax_hist.set_yscale("log")

        # Bottom right: Info text
        ax_info = axes[1, 2]
        ax_info.axis("off")

        # Calculate statistics
        n_shadow = self.shadow_mask.sum()
        n_total = len(self.shadow_mask)
        rms_residual = np.sqrt(np.mean(residuals**2))

        info_lines = [
            f"Frame: {self._frame.metadata.filename}",
            f"Shape: {original.shape}",
            f"Range: {distances[0]:.0f} - {distances[-1]:.0f} m",
            "",
            "Shadow region:",
            f"  Center: {self._config.shadow.center}°",
            f"  Width: {self._config.shadow.width}°",
            f"  Masked bearings: {n_shadow}/{n_total}",
            "",
            "Profile parameters:",
            f"  Quantile: {self._quantile * 100:.0f}%",
            f"  Smooth window: {self._smooth_window} bins",
            f"  RMS residual: {rms_residual:.2f}",
            "",
            "Profile range:",
            f"  Raw: [{self._raw_profile.min():.1f}, {self._raw_profile.max():.1f}]",
            f"  Smooth: [{self._smooth_profile.min():.1f}, {self._smooth_profile.max():.1f}]",
        ]
        ax_info.text(
            0.05,
            0.95,
            "\n".join(info_lines),
            transform=ax_info.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
        )

        plt.tight_layout()
        plt.show()


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument(
        "--quantile",
        "-q",
        type=float,
        default=0.10,
        help="Quantile for profile (0.0-1.0, default: 0.10)",
    )
    parser.add_argument(
        "--smooth-window",
        "-s",
        type=int,
        default=None,
        help="Smoothing window size in bins (default: 2%% of range bins)",
    )
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots")


def add_subparser(subparsers) -> None:
    """Register the 'deramp' subcommand."""
    p = subparsers.add_parser(
        "deramp", help="Standalone deramp tool", description="Test deramp on a polar file"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'deramp' command."""
    from wamos_tpw.polarfile import PolarFile

    # Load config
    config = WamosConfig(args.config) if args.config else WamosConfig()

    # Load polar file
    pf = PolarFile(args.filename)
    if not pf:
        logging.warning(f"No frames in {args.filename}")
        return

    frame = pf.frame()
    logging.info(f"Loaded: {frame}")

    # Apply deramp
    deramp = Deramp(frame, config, quantile=args.quantile, smooth_window=args.smooth_window)
    corrected = deramp.corrected_intensity

    logging.info(f"Quantile: {args.quantile * 100:.0f}%")
    logging.info(f"Smooth window: {args.smooth_window} bins")
    logging.info(f"Original intensity range: [{frame.intensity.min()}, {frame.intensity.max()}]")
    logging.info(f"Corrected intensity range: [{corrected.min():.1f}, {corrected.max():.1f}]")
    logging.info(
        f"Smooth profile range: [{deramp.smooth_profile.min():.1f}, {deramp.smooth_profile.max():.1f}]"
    )

    if args.plot:
        deramp.plot_diagnostics()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test deramp on a polar file")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
