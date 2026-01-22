#! /usr/bin/env python3
#
# Dewind class for removing look-angle-dependent intensity variation
#
# Jan-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.optimize import curve_fit

from wamos_tpw.config import Config
from wamos_tpw.theta import Theta


def _sin_model(theta_deg: np.ndarray, amplitude: float, phi: float) -> np.ndarray:
    """Sinusoidal model: amplitude * sin(theta + phi)."""
    theta_rad = np.deg2rad(theta_deg)
    return amplitude * np.sin(theta_rad + phi)


class Dewind:
    """
    Remove look-angle-dependent intensity variation from radar data.

    Algorithm Overview
    ------------------

    Marine radar intensity varies with look angle (bearing) due to wind-wave
    interaction. Waves aligned with the wind direction have different radar
    backscatter characteristics than waves perpendicular to the wind. This
    creates an azimuthal intensity modulation that can obscure sea surface
    features.

    The algorithm:
        1. Compute mean intensity at each bearing (ignoring NaN from shadow mask)
        2. Fit a sinusoidal model: amplitude * sin(theta + phi)
        3. Subtract the fitted profile from each bearing's intensity

    The fitted parameters are:
        - amplitude: strength of the look-angle modulation
        - phi: phase offset (related to wind direction)

    This flattens the look-angle-dependent variation while preserving relative
    intensity variations (waves, ships, etc.).

    Example:
        >>> from wamos_tpw.theta import Theta
        >>> theta = Theta(frame)
        >>> dewind = Dewind(deramp.intensity, theta)
        >>> corrected = dewind.intensity
    """

    def __init__(
        self,
        intensity: np.ndarray,
        theta: Theta,
        copy: bool = False,
    ) -> None:
        """
        Dewind a single frame.

        Args:
            intensity: Deramped intensity array (n_bearings, n_distances)
            theta: Theta object for bearing angles
                   (config is obtained from theta.config)
            copy: If True, copy the input array before modifying.
                  If False (default), modify in-place for memory efficiency.
        """
        self._config = theta.config

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice')
            mu = np.nanmean(intensity, axis=1)  # Mean over distances
        q = ~np.isnan(mu)

        theta_values = theta.theta

        # Initial guess: amplitude from data range, phi = 0
        amplitude_guess = (np.nanmax(mu) - np.nanmin(mu)) / 2
        p0 = [amplitude_guess, 0.0]

        try:
            popt, _ = curve_fit(
                _sin_model,
                theta_values[q],
                mu[q],
                p0=p0,
            )
            self._amplitude = popt[0]
            self._phi = popt[1]
        except RuntimeError:
            # Fit failed - use zero correction
            logging.warning("Dewind sinusoidal fit failed, using zero correction")
            self._amplitude = 0.0
            self._phi = 0.0

        # Compute fitted values and subtract
        fit_values = _sin_model(theta_values, self._amplitude, self._phi)
        if copy:
            intensity = intensity.copy()
        intensity -= fit_values[:, np.newaxis]
        self._intensity = intensity

    @property
    def intensity(self) -> np.ndarray:
        """Return the look-angle-corrected intensity array."""
        return self._intensity

    @property
    def amplitude(self) -> float:
        """Return the fitted amplitude."""
        return self._amplitude

    @property
    def phi(self) -> float:
        """Return the fitted phase offset (radians)."""
        return self._phi

    @property
    def phi_degrees(self) -> float:
        """Return the fitted phase offset (degrees)."""
        return np.rad2deg(self._phi)

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    def fit(self, theta_values: np.ndarray) -> np.ndarray:
        """Evaluate the fitted sinusoidal model at given theta values."""
        return _sin_model(theta_values, self._amplitude, self._phi)

    def __repr__(self) -> str:
        return (
            f"Dewind(amplitude={self._amplitude:.2f}, "
            f"phi={self.phi_degrees:.1f}°, shape={self._intensity.shape})"
        )


class DewindDiag:
    """
    Diagnostic visualization for dewind results.

    Provides plotting for comparing original and dewind-corrected intensity.

    Example:
        >>> from wamos_tpw.theta import Theta
        >>> theta = Theta(frame)
        >>> dewind = Dewind(deramp.intensity, theta)
        >>> diag = DewindDiag(deramp.intensity, theta, dewind)
        >>> diag.plot()
    """

    def __init__(
        self,
        intensity: np.ndarray,
        theta: Theta,
        dewind: Dewind,
    ) -> None:
        """
        Initialize diagnostic viewer.

        Args:
            intensity: Original intensity array (before dewind)
            theta: Theta object for bearing angles
            dewind: Dewind object with corrected intensity
        """
        self._intensity = intensity
        self._theta = theta
        self._dewind = dewind

    @property
    def intensity(self) -> np.ndarray:
        """Return the original intensity array."""
        return self._intensity

    @property
    def dewind(self) -> Dewind:
        """Return the Dewind object."""
        return self._dewind

    def plot(
        self,
        figsize: tuple[float, float] = (14, 8),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot diagnostic comparison of original and dewind-corrected intensity.

        Creates a 2-row figure showing:
        - Top row: Original and dewind-corrected intensity images
        - Bottom row: Theta vs mean intensity with sinusoidal fit

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plots
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt

        original = self._intensity.astype(np.float32, copy=False)
        corrected = self._dewind.intensity

        # Sort by theta for display
        theta_values = self._theta.theta
        sort_idx = np.argsort(theta_values)
        theta_sorted = theta_values[sort_idx]
        original_sorted = original[sort_idx, :]
        corrected_sorted = corrected[sort_idx, :]

        # Auto-scale if not specified
        if vmin is None:
            vmin = min(np.nanmin(original), np.nanmin(corrected))
        if vmax is None:
            vmax = max(np.nanmax(original), np.nanmax(corrected))

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 0.6])

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
        cax = fig.add_subplot(gs[0, 2])
        ax2 = fig.add_subplot(gs[1, :2])

        n_distances = original.shape[1]
        extent = [0, n_distances, theta_sorted[-1], theta_sorted[0]]

        # Original intensity (sorted by theta)
        ax0.imshow(original_sorted, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax0.set_title("Original")
        ax0.set_xlabel("Distance bin")
        ax0.set_ylabel("Theta (degrees)")

        # Dewind-corrected intensity (sorted by theta)
        im1 = ax1.imshow(
            corrected_sorted, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent
        )
        ax1.set_title(f"Dewind (A={self._dewind.amplitude:.1f}, φ={self._dewind.phi_degrees:.1f}°)")
        ax1.set_xlabel("Distance bin")

        # Single colorbar for images
        fig.colorbar(im1, cax=cax, label="Intensity")

        # Theta profile plot (already sorted above)
        pre_mean = np.nanmean(original_sorted, axis=1)
        post_mean = np.nanmean(corrected_sorted, axis=1)

        # Sinusoidal fit values
        fit_values = self._dewind.fit(theta_sorted)

        ax2.plot(theta_sorted, pre_mean, label="Pre-dewind mean", alpha=0.7)
        ax2.plot(
            theta_sorted,
            fit_values,
            label=f"Sinusoidal fit (A={self._dewind.amplitude:.1f}, φ={self._dewind.phi_degrees:.1f}°)",
            linestyle="--",
            linewidth=2,
        )
        ax2.plot(theta_sorted, post_mean, label="Post-dewind mean", alpha=0.7)
        ax2.set_xlabel("Theta (degrees)")
        ax2.set_ylabel("Mean intensity")
        ax2.set_title("Look Angle Profile")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Dewind Correction")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"DewindDiag(amplitude={self._dewind.amplitude:.2f}, "
            f"phi={self._dewind.phi_degrees:.1f}°, shape={self._intensity.shape})"
        )


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--plot", action="store_true", help="Plot dewind results")


def add_subparser(subparsers) -> None:
    """Register the 'dewind' subcommand."""
    p = subparsers.add_parser(
        "dewind", help="Standalone dewind tool", description="Test dewind on a polar file"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'dewind' command."""
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.destreak import Destreak
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.range import Range
    from wamos_tpw.deramp import Deramp

    # Load config
    config = Config(args.config) if args.config else Config()

    # Load polar file
    pf = PolarFile(args.filename, config=config)
    if not pf:
        logging.warning("No frames in %s", args.filename)
        return

    frame = pf.frame()
    theta = Theta(frame)
    destreak = Destreak(frame)

    shadow = Shadow(destreak.intensity, theta)

    masked_intensity = shadow.mask(destreak.intensity)
    rng = Range(frame)
    deramp = Deramp(masked_intensity, rng)

    # Save original for diagnostics (Dewind modifies intensity in-place)
    deramped_intensity = deramp.intensity.copy() if args.plot else deramp.intensity

    dewind = Dewind(deramp.intensity, theta)

    # Display results
    logging.info("File: %s", args.filename)
    logging.info("Frame: %s", frame.timestamp)
    logging.info("Shape: %s", frame.shape)
    logging.info("Fit: amplitude=%.2f, phi=%.1f°", dewind.amplitude, dewind.phi_degrees)
    logging.info(
        "Pre-dewind intensity: [%.1f, %.1f]",
        np.nanmin(deramped_intensity),
        np.nanmax(deramped_intensity),
    )
    logging.info(
        "Post-dewind intensity: [%.1f, %.1f]",
        np.nanmin(dewind.intensity),
        np.nanmax(dewind.intensity),
    )

    if args.plot:
        diag = DewindDiag(deramped_intensity, theta, dewind)
        diag.plot()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test dewind on a polar file")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
