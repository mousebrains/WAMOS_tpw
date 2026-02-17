#! /usr/bin/env python3
#
# Deramp class for removing range-dependent intensity fall-off
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging

import numpy as np
from numpy.polynomial import Polynomial

from wamos_tpw.backend import HAS_TORCH_GPU
from wamos_tpw.config import Config
from wamos_tpw.range import Range

logger = logging.getLogger(__name__)

__all__ = ["Deramp"]


def _deramp_cpu(
    intensity: np.ndarray,
    x: np.ndarray,
    order: int,
    copy: bool,
) -> tuple[np.ndarray, Polynomial]:
    """CPU deramp path."""
    mu = np.nanmean(intensity, axis=0)
    q = np.isnan(mu)
    p = Polynomial.fit(x[~q], mu[~q], deg=order)
    py = p(x)
    if copy:
        intensity = intensity.copy()
    intensity -= py[np.newaxis, :]
    return intensity, p


def _deramp_gpu(
    intensity: np.ndarray,
    x: np.ndarray,
    order: int,
    copy: bool,
) -> tuple[np.ndarray, Polynomial]:
    """GPU deramp path: nanmean on GPU, polyfit on CPU, subtract on GPU."""
    import torch

    from wamos_tpw.backend import get_device, to_numpy

    dev = get_device()

    # nanmean on GPU
    t_intensity = torch.from_numpy(np.ascontiguousarray(intensity, dtype=np.float32)).to(dev)
    mu = to_numpy(torch.nanmean(t_intensity, dim=0))

    # Polynomial fit on CPU (small 1D array)
    q = np.isnan(mu)
    p = Polynomial.fit(x[~q], mu[~q], deg=order)
    py = p(x).astype(np.float32)

    # Broadcast subtract on GPU
    t_py = torch.from_numpy(py).to(dev)
    if copy:
        t_result = t_intensity - t_py.unsqueeze(0)
    else:
        t_intensity -= t_py.unsqueeze(0)
        t_result = t_intensity
    result = to_numpy(t_result)
    return result, p


class Deramp:
    """
    Remove range-dependent intensity fall-off from radar data.

    Algorithm Overview
    ------------------

    Marine radar intensity decreases with range due to the radar equation:
    received power falls off as 1/r (for surface waves). This creates
    a radial intensity gradient that obscures sea surface features.

    The algorithm:
        1. Compute mean intensity at each range bin (ignoring NaN from shadow mask)
        2. Fit a polynomial to mean intensity vs 1/range
        3. Subtract the fitted profile from each bearing's intensity

    This flattens the range-dependent falloff while preserving relative
    intensity variations (waves, ships, etc.).

    Example:
        >>> from wamos_tpw.range import Range
        >>> rng = Range(frame)
        >>> deramp = Deramp(shadow.mask(intensity), rng)
        >>> corrected = deramp.intensity
    """

    # Constants
    _DERAMP_ORDER_DEFAULT = 4  # Fit to 4th order polynomial

    def __init__(
        self,
        intensity: np.ndarray,
        rng: Range,
        copy: bool = False,
    ) -> None:
        """
        Deramp a single frame.

        Args:
            intensity: Shadow-masked intensity array (n_bearings, n_distances)
            rng: Range object for slant range values
                 (config is obtained from rng.config)
            copy: If True, copy the input array before modifying.
                  If False (default), modify in-place for memory efficiency.
        """
        self._config = rng.config
        order = int(self._config.get("deramp.order", self._DERAMP_ORDER_DEFAULT))

        slant = rng.slant_range
        x = 1 / slant  # 1/range fall-off

        if HAS_TORCH_GPU:
            self._intensity, self._polynomial = _deramp_gpu(intensity, x, order, copy)
        else:
            self._intensity, self._polynomial = _deramp_cpu(intensity, x, order, copy)
        self._order = order

    @property
    def intensity(self) -> np.ndarray:
        """Return the range-corrected intensity array."""
        return self._intensity

    @property
    def polynomial(self) -> Polynomial:
        """Return the fitted polynomial."""
        return self._polynomial

    @property
    def order(self) -> int:
        """Return the polynomial order used."""
        return self._order

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    def __repr__(self) -> str:
        return f"Deramp(order={self._order}, shape={self._intensity.shape})"


class DerampDiag:
    """
    Diagnostic visualization for deramp results.

    Provides plotting for comparing original and deramped intensity.

    Example:
        >>> from wamos_tpw.range import Range
        >>> rng = Range(frame)
        >>> deramp = Deramp(shadow.mask(intensity), rng)
        >>> diag = DerampDiag(shadow.mask(intensity), rng, deramp)
        >>> diag.plot()
    """

    def __init__(
        self,
        intensity: np.ndarray,
        rng: Range,
        deramp: Deramp,
    ) -> None:
        """
        Initialize diagnostic viewer.

        Args:
            intensity: Original intensity array (before deramping)
            rng: Range object for slant range values
            deramp: Deramp object with corrected intensity
        """
        self._intensity = intensity
        self._rng = rng
        self._deramp = deramp

    @property
    def intensity(self) -> np.ndarray:
        """Return the original intensity array."""
        return self._intensity

    @property
    def deramp(self) -> Deramp:
        """Return the Deramp object."""
        return self._deramp

    def plot(
        self,
        figsize: tuple[float, float] = (14, 8),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot diagnostic comparison of original and deramped intensity.

        Creates a 2-row figure showing:
        - Top row: Original and deramped intensity images
        - Bottom row: Slant range vs mean intensity with polynomial fit

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plots
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt

        original = self._intensity.astype(np.float32)
        corrected = self._deramp.intensity

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

        # Original intensity
        ax0.imshow(original, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax0.set_title("Original")
        ax0.set_xlabel("Distance bin")
        ax0.set_ylabel("Bearing bin")

        # Deramped intensity
        im1 = ax1.imshow(corrected, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title(f"Deramped (order={self._deramp.order})")
        ax1.set_xlabel("Distance bin")

        # Single colorbar for images
        fig.colorbar(im1, cax=cax, label="Intensity")

        # Range profile plot
        slant = self._rng.slant_range
        pre_mean = np.nanmean(original, axis=0)
        post_mean = np.nanmean(corrected, axis=0)

        # Polynomial fit values
        x = 1 / slant
        fit_values = self._deramp.polynomial(x)

        ax2.plot(slant, pre_mean, label="Pre-deramp mean", alpha=0.7)
        ax2.plot(
            slant,
            fit_values,
            label=f"Polynomial fit (order={self._deramp.order})",
            linestyle="--",
            linewidth=2,
        )
        ax2.plot(slant, post_mean, label="Post-deramp mean", alpha=0.7)
        ax2.set_xlabel("Slant range (m)")
        ax2.set_ylabel("Mean intensity")
        ax2.set_title("Range Profile")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Deramp Correction")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return f"DerampDiag(order={self._deramp.order}, shape={self._intensity.shape})"


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--plot", action="store_true", help="Plot deramp results")


def add_subparser(subparsers) -> None:
    """Register the 'deramp' subcommand."""
    p = subparsers.add_parser(
        "deramp", help="Standalone deramp tool", description="Test deramp on a polar file"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'deramp' command."""
    from wamos_tpw.destreak import Destreak
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.theta import Theta

    # Load config
    config = Config(args.config) if args.config else Config()

    # Load polar file
    pf = PolarFile(args.filename, config=config)
    if not pf:
        logger.warning("No frames in %s", args.filename)
        return

    frame = pf.frame()
    theta = Theta(frame)
    destreak = Destreak(frame)

    shadow = Shadow(destreak.intensity, theta)

    masked_intensity = shadow.mask(destreak.intensity)
    rng = Range(frame)

    # Save original for diagnostics (Deramp modifies intensity in-place)
    original_intensity = masked_intensity.copy() if args.plot else None

    deramp = Deramp(masked_intensity, rng)

    # Display results
    logger.info("File: %s", args.filename)
    logger.info("Frame: %s", frame.timestamp)
    logger.info("Shape: %s", frame.shape)
    logger.info("Polynomial order: %d", deramp.order)
    logger.info(
        "Original intensity: [%.1f, %.1f]",
        np.nanmin(original_intensity if original_intensity is not None else masked_intensity),
        np.nanmax(original_intensity if original_intensity is not None else masked_intensity),
    )
    logger.info(
        "Deramped intensity: [%.1f, %.1f]", np.nanmin(deramp.intensity), np.nanmax(deramp.intensity)
    )

    if args.plot:
        diag = DerampDiag(original_intensity, rng, deramp)
        diag.plot()


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Test deramp on a polar file")

if __name__ == "__main__":
    main()
