#! /usr/bin/env python3
#
# Deramp class for removing range-dependent intensity fall-off
# Subtracts smoothed quantile profile from intensity data
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import numpy as np

from wamos_tpw.config import WamosConfig
from wamos_tpw.frame import Frame


class Deramp:
    """
    Remove range-dependent intensity fall-off from radar data.

    Calculates a quantile intensity profile as a function of range
    (excluding shadowed regions), smooths it, and subtracts from the data.

    Algorithm:
    1. Identify shadow region from config (bearings to exclude)
    2. Calculate quantile intensity at each range bin (excluding shadow)
    3. Smooth the profile with a moving average
    4. Subtract smoothed profile from intensity data

    Example:
        >>> deramp = Deramp(frame, config, quantile=0.25)
        >>> corrected = deramp.corrected_intensity
        >>> deramp.plot_diagnostics()  # Optional visualization
    """

    # Default smoothing window as fraction of range bins
    _SMOOTH_WINDOW_FRACTION = 0.02  # 2% of range bins

    def __init__(self,
                 frame: Frame,
                 config: WamosConfig | None = None,
                 bearing: np.ndarray | None = None,
                 quantile: float = 0.10,
                 smooth_window: int | None = None) -> None:
        """
        Initialize Deramp for a single frame.

        Args:
            frame: Frame object with intensity data to correct
            config: WamosConfig for shadow region parameters
            bearing: Optional bearing array for each row (degrees).
                     If None, assumes uniform 0-360 distribution.
            quantile: Quantile to use for profile (0.0-1.0, default 0.10)
            smooth_window: Smoothing window size in bins (default: 2% of range bins)
        """
        self._frame = frame
        self._config = config or WamosConfig()
        self._bearing = bearing
        self._quantile = quantile
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
        """Compute mask for shadow region based on config."""
        n_bearings = self._frame.n_bearings

        # Get shadow parameters from config
        shadow_center = self._config.shadow.center  # degrees from bow
        shadow_width = self._config.shadow.width    # total width in degrees

        if shadow_center is None or shadow_width is None or shadow_width <= 0:
            # No shadow defined - no masking
            self._shadow_mask = np.zeros(n_bearings, dtype=bool)
            return

        # Calculate shadow bounds
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
        intensity = self._frame.intensity.astype(np.float64)
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
        self._raw_profile = np.quantile(non_shadow_data, self._quantile, axis=0)

        # Smooth the profile
        self._smooth_profile = self._smooth_moving_average(
            self._raw_profile, self._smooth_window
        )

        # Subtract the smoothed profile from intensity data
        self._corrected = intensity - self._smooth_profile[np.newaxis, :]

    @staticmethod
    def _smooth_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing with edge handling."""
        if window <= 1:
            return data.copy()

        # Use convolution for moving average
        kernel = np.ones(window) / window
        # Pad edges to avoid boundary effects
        pad_width = window // 2
        padded = np.pad(data, pad_width, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')

        # Ensure output length matches input
        if len(smoothed) > len(data):
            smoothed = smoothed[:len(data)]
        elif len(smoothed) < len(data):
            smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')

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

        original = self._frame.intensity.astype(np.float64)
        corrected = self._corrected
        distances = self.slant_range

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Deramp Diagnostics: {self._frame.metadata.filename}', fontsize=12)

        # Top left: Original intensity
        ax_orig = axes[0, 0]
        vmin, vmax = np.percentile(original, [1, 99])
        im1 = ax_orig.imshow(original, aspect='auto', cmap='viridis',
                             vmin=vmin, vmax=vmax,
                             extent=[distances[0], distances[-1], original.shape[0], 0])
        ax_orig.set_title('Original Intensity')
        ax_orig.set_xlabel('Range (m)')
        ax_orig.set_ylabel('Bearing bin')
        plt.colorbar(im1, ax=ax_orig)

        # Top middle: Corrected intensity
        ax_corr = axes[0, 1]
        vmin_c, vmax_c = np.percentile(corrected, [1, 99])
        im2 = ax_corr.imshow(corrected, aspect='auto', cmap='viridis',
                             vmin=vmin_c, vmax=vmax_c,
                             extent=[distances[0], distances[-1], corrected.shape[0], 0])
        ax_corr.set_title('Corrected Intensity')
        ax_corr.set_xlabel('Range (m)')
        ax_corr.set_ylabel('Bearing bin')
        plt.colorbar(im2, ax=ax_corr)

        # Top right: Range profiles
        ax_prof = axes[0, 2]
        ax_prof.plot(distances, self._raw_profile, 'b-', alpha=0.5,
                     label=f'Raw {self._quantile*100:.0f}%')
        ax_prof.plot(distances, self._smooth_profile, 'r-', linewidth=2,
                     label=f'Smoothed ({self._smooth_window} bins)')
        ax_prof.set_xlabel('Range (m)')
        ax_prof.set_ylabel('Intensity')
        ax_prof.set_title('Range Profile (excl. shadow)')
        ax_prof.legend()
        ax_prof.grid(True, alpha=0.3)

        # Bottom left: Smoothing residuals
        ax_resid = axes[1, 0]
        residuals = self._raw_profile - self._smooth_profile
        ax_resid.plot(distances, residuals, 'g-', linewidth=1)
        ax_resid.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_resid.set_xlabel('Range (m)')
        ax_resid.set_ylabel('Residual (raw - smooth)')
        ax_resid.set_title('Smoothing Residuals')
        ax_resid.grid(True, alpha=0.3)

        # Bottom middle: Histogram comparison
        ax_hist = axes[1, 1]
        ax_hist.hist(original.ravel(), bins=100, alpha=0.5, label='Original', density=True)
        ax_hist.hist(corrected.ravel(), bins=100, alpha=0.5, label='Corrected', density=True)
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Intensity Distribution')
        ax_hist.legend()
        ax_hist.set_yscale('log')

        # Bottom right: Info text
        ax_info = axes[1, 2]
        ax_info.axis('off')

        # Calculate statistics
        n_shadow = self.shadow_mask.sum()
        n_total = len(self.shadow_mask)
        rms_residual = np.sqrt(np.mean(residuals**2))

        info_lines = [
            f'Frame: {self._frame.metadata.filename}',
            f'Shape: {original.shape}',
            f'Range: {distances[0]:.0f} - {distances[-1]:.0f} m',
            '',
            'Shadow region:',
            f'  Center: {self._config.shadow.center}°',
            f'  Width: {self._config.shadow.width}°',
            f'  Masked bearings: {n_shadow}/{n_total}',
            '',
            'Profile parameters:',
            f'  Quantile: {self._quantile*100:.0f}%',
            f'  Smooth window: {self._smooth_window} bins',
            f'  RMS residual: {rms_residual:.2f}',
            '',
            'Profile range:',
            f'  Raw: [{self._raw_profile.min():.1f}, {self._raw_profile.max():.1f}]',
            f'  Smooth: [{self._smooth_profile.min():.1f}, {self._smooth_profile.max():.1f}]',
        ]
        ax_info.text(0.05, 0.95, '\n'.join(info_lines),
                    transform=ax_info.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=9)

        plt.tight_layout()
        plt.show()


def add_subparser(subparsers) -> None:
    """Register the 'deramp' subcommand."""
    p = subparsers.add_parser(
        'deramp',
        help='Standalone deramp tool',
        description="Test deramp on a polar file"
    )
    p.add_argument("filename", help="Polar file to process")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="YAML configuration file")
    p.add_argument("--quantile", "-q", type=float, default=0.10,
                   help="Quantile for profile (0.0-1.0, default: 0.10)")
    p.add_argument("--smooth-window", "-s", type=int, default=None,
                   help="Smoothing window size in bins (default: 2%% of range bins)")
    p.add_argument("--plot", action="store_true",
                   help="Show diagnostic plots")
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'deramp' command."""
    from wamos_tpw.polarfile import PolarFile

    # Load config
    config = WamosConfig(args.config) if args.config else WamosConfig()

    # Load polar file
    pf = PolarFile(args.filename)
    if not pf:
        print(f"No frames in {args.filename}")
        return

    frame = pf.frame()
    print(f"Loaded: {frame}")

    # Apply deramp
    deramp = Deramp(frame, config, quantile=args.quantile, smooth_window=args.smooth_window)
    corrected = deramp.corrected_intensity

    print(f"\nQuantile: {args.quantile*100:.0f}%")
    print(f"Smooth window: {args.smooth_window} bins")
    print(f"\nOriginal intensity range: [{frame.intensity.min()}, {frame.intensity.max()}]")
    print(f"Corrected intensity range: [{corrected.min():.1f}, {corrected.max():.1f}]")
    print(f"Smooth profile range: [{deramp.smooth_profile.min():.1f}, {deramp.smooth_profile.max():.1f}]")

    if args.plot:
        deramp.plot_diagnostics()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Test deramp on a polar file")
    parser.add_argument("filename", help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML configuration file")
    parser.add_argument("--quantile", "-q", type=float, default=0.10,
                        help="Quantile for profile (0.0-1.0, default: 0.10)")
    parser.add_argument("--smooth-window", "-s", type=int, default=None,
                        help="Smoothing window size in bins (default: 2%% of range bins)")
    parser.add_argument("--plot", action="store_true",
                        help="Show diagnostic plots")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
