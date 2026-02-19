#! /usr/bin/env python3
#
# Diagnostic visualizations for surface current extraction
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Diagnostic visualizations for surface current extraction.

Provides:

- **CurrentDiag**: Spectral diagnostics for a single sub-region extraction
  (kx-omega and ky-omega slices, dispersion curve overlay, SNR search surface).
- **CurrentMapDiag**: Spatial diagnostics for a tiled current map
  (quiver plot, speed colormap, SNR quality map).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.current import CurrentExtractor, CurrentMap

logger = logging.getLogger(__name__)

__all__ = [
    "CurrentDiag",
    "CurrentMapDiag",
]


class CurrentDiag:
    """Spectral diagnostics for a single CurrentExtractor result.

    Produces:
    1. kx-omega slice of the 3D power spectrum (at ky=0) with dispersion
       curve overlay for the estimated current.
    2. ky-omega slice (at kx=0) with dispersion curve overlay.
    3. (Ux, Uy) search surface showing which current maximizes shell energy.

    Args:
        extractor: A :class:`CurrentExtractor` that has already run.
    """

    def __init__(self, extractor: CurrentExtractor) -> None:
        self._ext = extractor

    def plot(self, fig=None) -> None:
        """Draw all three diagnostic panels.

        Args:
            fig: Optional matplotlib Figure. Creates one if not provided.
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            axes = fig.subplots(1, 3)

        self._plot_kx_omega(axes[0])
        self._plot_ky_omega(axes[1])
        self._plot_search_surface(axes[2])

        fig.tight_layout()

    def _plot_kx_omega(self, ax) -> None:
        """Plot kx-omega slice at ky=0 with dispersion curve overlay."""
        from scipy.fft import fftshift

        from wamos_tpw.current import dispersion_relation

        ext = self._ext
        P = ext.power_spectrum  # (n_t, n_ky, n_kx)
        kx = ext.kx
        omega = ext.omega

        # Find ky=0 index (should be index 0 for fftfreq)
        ky_zero_idx = np.argmin(np.abs(ext.ky))

        # Extract slice P(omega, kx) at ky=0
        slice_2d = P[:, ky_zero_idx, :]  # (n_t, n_kx)

        # fftshift for display
        kx_shifted = fftshift(kx)
        omega_shifted = fftshift(omega)
        slice_shifted = fftshift(slice_2d)

        log_power = np.log10(np.maximum(slice_shifted, 1e-10))

        ax.pcolormesh(
            kx_shifted,
            omega_shifted,
            log_power,
            shading="auto",
            cmap="viridis",
        )

        # Overlay dispersion curve for the estimated current
        est = ext.estimate
        k_plot = np.linspace(kx_shifted.min(), kx_shifted.max(), 200)
        omega_0 = dispersion_relation(np.abs(k_plot), est.depth)
        # Positive branch: omega = -(omega_0 + kx*Ux) at ky=0
        omega_disp_pos = -(omega_0 + k_plot * est.ux)
        omega_disp_neg = omega_0 - k_plot * est.ux

        ax.plot(k_plot, omega_disp_pos, "r-", linewidth=1.5, label="Dispersion (+)")
        ax.plot(k_plot, omega_disp_neg, "r--", linewidth=1.0, label="Dispersion (-)")

        ax.set_xlabel("kx (rad/m)")
        ax.set_ylabel("omega (rad/s)")
        ax.set_title("P(kx, omega) at ky=0")
        ax.legend(fontsize=7)

    def _plot_ky_omega(self, ax) -> None:
        """Plot ky-omega slice at kx=0 with dispersion curve overlay."""
        from scipy.fft import fftshift

        from wamos_tpw.current import dispersion_relation

        ext = self._ext
        P = ext.power_spectrum  # (n_t, n_ky, n_kx)
        ky = ext.ky
        omega = ext.omega

        kx_zero_idx = np.argmin(np.abs(ext.kx))

        # Extract slice P(omega, ky) at kx=0
        slice_2d = P[:, :, kx_zero_idx]  # (n_t, n_ky)

        ky_shifted = fftshift(ky)
        omega_shifted = fftshift(omega)
        slice_shifted = fftshift(slice_2d)

        log_power = np.log10(np.maximum(slice_shifted, 1e-10))

        ax.pcolormesh(
            ky_shifted,
            omega_shifted,
            log_power,
            shading="auto",
            cmap="viridis",
        )

        est = ext.estimate
        k_plot = np.linspace(ky_shifted.min(), ky_shifted.max(), 200)
        omega_0 = dispersion_relation(np.abs(k_plot), est.depth)
        omega_disp_pos = -(omega_0 + k_plot * est.uy)
        omega_disp_neg = omega_0 - k_plot * est.uy

        ax.plot(k_plot, omega_disp_pos, "r-", linewidth=1.5, label="Dispersion (+)")
        ax.plot(k_plot, omega_disp_neg, "r--", linewidth=1.0, label="Dispersion (-)")

        ax.set_xlabel("ky (rad/m)")
        ax.set_ylabel("omega (rad/s)")
        ax.set_title("P(ky, omega) at kx=0")
        ax.legend(fontsize=7)

    def _plot_search_surface(self, ax) -> None:
        """Plot the (Ux, Uy) search surface with the maximum marked."""
        ext = self._ext
        surface = ext._search_surface
        est = ext.estimate

        radius = ext._search_radius
        step = ext._search_step
        candidates = np.arange(-radius, radius + step / 2, step)

        ax.pcolormesh(
            candidates,
            candidates,
            surface,
            shading="auto",
            cmap="hot",
        )

        # Mark the estimated current
        ax.plot(est.ux, est.uy, "cx", markersize=12, markeredgewidth=2, label="Estimate")
        ax.set_xlabel("Ux (m/s)")
        ax.set_ylabel("Uy (m/s)")
        ax.set_title(
            f"Shell energy surface\n"
            f"Ux={est.ux:.2f}, Uy={est.uy:.2f}, "
            f"speed={est.speed:.2f} m/s, SNR={est.snr:.2f}"
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

        # Draw search radius circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "w--", linewidth=0.5, alpha=0.5)


class CurrentMapDiag:
    """Spatial diagnostics for a CurrentMap result.

    Produces:
    1. Quiver plot of current vectors with speed color-coding.
    2. Speed colormap.
    3. SNR quality map.

    Args:
        current_map: A :class:`CurrentMap` with extraction results.
    """

    def __init__(self, current_map: CurrentMap) -> None:
        self._map = current_map

    def plot(self, fig=None) -> None:
        """Draw all three diagnostic panels.

        Args:
            fig: Optional matplotlib Figure. Creates one if not provided.
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            axes = fig.subplots(1, 3)

        self._plot_quiver(axes[0])
        self._plot_speed(axes[1])
        self._plot_snr(axes[2])

        fig.tight_layout()

    def _plot_quiver(self, ax) -> None:
        """Quiver plot of current vectors colored by speed."""
        cmap = self._map
        X, Y = np.meshgrid(cmap.tile_x_centers, cmap.tile_y_centers)

        valid = ~np.isnan(cmap.speed)
        if not np.any(valid):
            ax.set_title("No valid current estimates")
            return

        q = ax.quiver(
            X[valid],
            Y[valid],
            cmap.ux[valid],
            cmap.uy[valid],
            cmap.speed[valid],
            cmap="coolwarm",
            scale_units="xy",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Current vectors")
        ax.set_aspect("equal")

        import matplotlib.pyplot as plt

        plt.colorbar(q, ax=ax, label="Speed (m/s)")

    def _plot_speed(self, ax) -> None:
        """Speed colormap over tile grid."""
        cmap = self._map
        ax.pcolormesh(
            cmap.tile_x_centers,
            cmap.tile_y_centers,
            cmap.speed,
            shading="auto",
            cmap="viridis",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Current speed (m/s)")
        ax.set_aspect("equal")

    def _plot_snr(self, ax) -> None:
        """SNR quality map over tile grid."""
        cmap = self._map
        im = ax.pcolormesh(
            cmap.tile_x_centers,
            cmap.tile_y_centers,
            cmap.snr,
            shading="auto",
            cmap="RdYlGn",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("SNR quality")
        ax.set_aspect("equal")

        import matplotlib.pyplot as plt

        plt.colorbar(im, ax=ax, label="SNR")
