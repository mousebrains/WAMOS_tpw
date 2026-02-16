#! /usr/bin/env python3
#
# Shadow - Detect and mask shadow regions in radar data
# Shadow regions are areas blocked by ship structure (mast, antenna, etc.)
#
# Jan-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING

import numpy as np

from wamos_tpw.config import Config

if TYPE_CHECKING:
    from wamos_tpw.theta import Theta

__all__ = ["Shadow"]


class Shadow:
    """
    Detect and mask shadow regions in radar data.

    Shadow regions are angular sectors where the radar beam is blocked by
    ship structure (mast, antenna, exhaust stack, etc.). These regions
    contain invalid or attenuated returns that should be excluded from
    analysis.

    Shadow regions are defined in the config as angular ranges in degrees
    relative to the radar's reference direction (typically ship's bow).

    Config structure:
        shadow:
          range_fraction: 0.1  # Fraction of range bins for detection
          aft:                 # Name of shadow region
            LHS: 130           # Left hand side angle (degrees)
            RHS: 230           # Right hand side angle (degrees)
            expected_LHS: 145  # Expected LHS for theta adjustment (optional)
            expected_RHS: 215  # Expected RHS for theta adjustment (optional)

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> from wamos_tpw.theta import Theta
        >>> from wamos_tpw.destreak import Destreak
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> theta = Theta(frame)
        >>> destreak = Destreak(frame)
        >>> shadow = Shadow(destreak.intensity, theta)
        >>> masked_intensity = shadow.mask(destreak.intensity)
    """

    # Constants
    _RANGE_FRACTION_DEFAULT = 0.05  # Default fraction of range for shadow detection
    _ANGLE_RANGE_DEFAULT = 10.0  # Default angle range around shadow region
    _THETA_REFINE_DEFAULT = True  # Default to refine theta based on expected edges
    _KNOWN_IGNORES = ["range_fraction", "angle_range", "theta_refinement"]  # Skip non-region keys

    def __init__(
        self,
        intensity: np.ndarray,
        theta: Theta,
    ) -> None:
        """
        Initialize Shadow detector.

        Args:
            intensity: Destreaked intensity
            theta: Theta object containing radar beam angles
                   (config is obtained from theta.config)
        """
        self._config = theta.config
        self._timing: dict[str, float] = {}

        # Get shadow config - try direct access first (tower-specific config)
        config = self._config.get("shadow", Config())

        self._thetas = np.empty([0, 2], dtype=float)
        self._indices = np.empty_like(self._thetas, dtype=int)
        self._theta_bias = 0.0

        # Parse shadow regions from config (new format with LHS/RHS keys)

        t0 = _time.perf_counter()

        signs = []
        centroids = []

        for key in config:
            if key in self._KNOWN_IGNORES:  # Skip non-region keys
                continue
            region = config[key]
            if "LHS" in region:
                centroids.append(region["LHS"])
                signs.append(+1)
            if "RHS" in region:
                centroids.append(region["RHS"])
                signs.append(-1)

        if not centroids:
            return

        centroids = np.array(centroids)
        angle_range = config.get("angle_range", self._ANGLE_RANGE_DEFAULT)
        angle_delta = np.array([-1, 1]) * angle_range
        edges = centroids[:, np.newaxis] + angle_delta[np.newaxis, :]

        signs = np.array(signs, dtype=int)
        indices = theta.index(edges)  # Indices for these thetas

        self._timing["index_lookup"] = _time.perf_counter() - t0

        range_fraction = config.get("range_fraction", self._RANGE_FRACTION_DEFAULT)
        range_slice = int(
            np.clip(np.ceil(range_fraction * intensity.shape[1]), 1, intensity.shape[1])
        )

        t0 = _time.perf_counter()
        kernel_1d = np.array([1, 1, 0, -1, -1], dtype=np.float32)

        index = []

        for row in range(indices.shape[0]):
            ii = indices[row, :]
            sgn = signs[row]
            aLocal = intensity[ii[0] : (ii[1] + 1), :range_slice]
            row_sums = aLocal.sum(axis=1)
            row_padded = np.pad(row_sums, 2, mode="reflect")
            b = np.convolve(row_padded, kernel_1d, mode="same")[2:-2]
            if sgn > 0:
                iEdge = np.argmin(b)
            else:
                iEdge = np.argmax(b)
            index.append(iEdge + ii[0])

        self._timing["convolve"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        # Reshape to pair LHS and RHS edges into regions: [[lhs0, rhs0], [lhs1, rhs1], ...]
        self._indices = np.array(index, dtype=int).reshape(-1, 2)
        self._thetas = theta.theta[self._indices]
        self._timing["extract"] = _time.perf_counter() - t0

        if config.get("theta_refinement", self._THETA_REFINE_DEFAULT):
            # Refine theta bias based on expected edges
            self._theta_bias = np.mean(centroids - self._thetas.ravel())

    def mask(self, intensity: np.ndarray) -> np.ndarray:
        """Return intensity with shadow regions masked as NaN."""
        masked_intensity = intensity.astype(np.float32, copy=True)
        mask = np.zeros(masked_intensity.shape[0], dtype=bool)
        for start, end in self._indices:
            mask[start : end + 1] = True
        masked_intensity[mask, :] = np.nan
        return masked_intensity

    @property
    def theta_bias(self) -> float:
        """Return the theta bias applied during shadow detection (0.0 if none)."""
        return self._theta_bias

    @property
    def indices(self) -> np.ndarray:
        """Return the shadow regions as Nx2 array of [start, end] index pairs."""
        return self._indices

    @property
    def thetas(self) -> np.ndarray:
        """Return the shadow regions as Nx2 array of [start, end] angle pairs in degrees."""
        return self._thetas

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    @property
    def timing(self) -> dict[str, float]:
        """Return timing information for sub-steps (index_lookup, convolve, extract)."""
        return self._timing

    def __repr__(self) -> str:
        regions_str = ", ".join(f"[{s:.2f}-{e:.2f}]" for s, e in self._thetas)
        indices_str = ", ".join(f"[{s:.0f}-{e:.0f}]" for s, e in self._indices)
        return f"Shadow(thetas=[{regions_str}], indices=[{indices_str}])"


class ShadowDiag:
    """
    Diagnostic visualization and statistics for shadow detection results.

    Provides plotting and statistics for analyzing shadow regions.

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> from wamos_tpw.theta import Theta
        >>> from wamos_tpw.destreak import Destreak
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> theta = Theta(frame)
        >>> destreak = Destreak(frame)
        >>> shadow = Shadow(destreak.intensity, theta)
        >>> diag = ShadowDiag(destreak.intensity, shadow)
        >>> diag.plot()
    """

    def __init__(
        self,
        intensity: np.ndarray,
        shadow: Shadow,
    ) -> None:
        """
        Initialize diagnostic viewer.

        Args:
            intensity: Original intensity array (before masking)
            shadow: Shadow object with detected regions
        """
        self._intensity = intensity
        self._shadow = shadow

        # Cache statistics
        n_distances = intensity.shape[1]
        self._n_shadow_pixels = sum(
            (region[1] - region[0] + 1) * n_distances for region in shadow.indices
        )
        total_pixels = intensity.size
        self._shadow_fraction = self._n_shadow_pixels / total_pixels if total_pixels > 0 else 0.0

    @property
    def intensity(self) -> np.ndarray:
        """Return the original intensity array."""
        return self._intensity

    @property
    def shadow(self) -> Shadow:
        """Return the Shadow object."""
        return self._shadow

    @property
    def n_shadow_pixels(self) -> int:
        """Return the number of pixels in shadow regions."""
        return self._n_shadow_pixels

    @property
    def shadow_fraction(self) -> float:
        """Return the fraction of pixels in shadow regions."""
        return self._shadow_fraction

    def plot(
        self,
        figsize: tuple[float, float] = (14, 5),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot diagnostic comparison of original and shadow-masked intensity.

        Creates a 1x2 figure showing:
        - Left: Original intensity
        - Right: Shadow-masked intensity

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plots
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt

        original = self._intensity.astype(np.float32)
        masked = self._shadow.mask(self._intensity)

        # Auto-scale if not specified
        if vmin is None:
            vmin = np.nanmin(original)
        if vmax is None:
            vmax = np.nanmax(original)

        fig, (ax0, ax1, cax) = plt.subplots(
            1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1, 1, 0.05]}
        )

        # Original intensity
        ax0.imshow(original, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax0.set_title("Original")
        ax0.set_xlabel("Distance bin")
        ax0.set_ylabel("Bearing bin")

        # Shadow-masked intensity
        im1 = ax1.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        theta_str = ", ".join(f"θ=[{t[0]:.1f}°, {t[1]:.1f}°]" for t in self._shadow.thetas)
        ax1.set_title(
            f"Shadow Masked ({self.n_shadow_pixels} pixels, {self.shadow_fraction:.2%}) {theta_str}"
        )
        ax1.set_xlabel("Distance bin")
        ax1.sharex(ax0)
        ax1.sharey(ax0)

        # Single colorbar in dedicated axes
        fig.colorbar(im1, cax=cax, label="Intensity")

        fig.suptitle(
            f"Shadow Detection | Regions: {len(self._shadow.indices)} | "
            f"Masked: {self.shadow_fraction:.2%}"
        )
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"ShadowDiag(regions={len(self._shadow.indices)}, "
            f"shadow_pixels={self.n_shadow_pixels} ({self.shadow_fraction:.2%}))"
        )


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", type=str, help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument("--plot", action="store_true", help="Plot shadow detection results")


def add_subparser(subparsers) -> None:
    """Register the 'shadow' subcommand."""
    p = subparsers.add_parser(
        "shadow",
        help="Detect shadow regions in radar data",
        description="Detect and display shadow regions from radar data",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'shadow' command."""
    from wamos_tpw.destreak import Destreak
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.theta import Theta

    # Load polar file
    pf = PolarFile(args.filename, config=Config(args.config) if args.config else Config())

    if not pf:
        logging.error("No frames found in %s", args.filename)
        return

    frame_idx = min(args.frame, len(pf) - 1)
    frame = pf[frame_idx]

    # Calculate theta and destreak
    theta = Theta(frame)
    destreak = Destreak(frame)

    shadow = Shadow(destreak.intensity, theta)
    diag = ShadowDiag(destreak.intensity, shadow)

    # Display results
    logging.info("File: %s", args.filename)
    logging.info("Frame: %s (index %s)", frame.timestamp, frame_idx)
    logging.info("Shape: %s", frame.shape)
    logging.info("Shadow regions: %d", len(shadow.indices))
    logging.info("Shadow pixels: %d (%.2f%%)", diag.n_shadow_pixels, diag.shadow_fraction * 100)
    for i, (t, idx) in enumerate(zip(shadow.thetas, shadow.indices, strict=False)):
        logging.info(
            "  Region %d: theta=[%.2f, %.2f], indices=[%d, %d]", i, t[0], t[1], idx[0], idx[1]
        )

    if args.plot:
        diag.plot()


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Detect shadow regions in radar data")

if __name__ == "__main__":
    main()
