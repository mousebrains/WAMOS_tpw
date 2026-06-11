#! /usr/bin/env python3
#
# Diagnostic visualizations for surface current extraction
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Diagnostic visualizations for surface current extraction.

Provides:

- **CurrentDiag**: Spectral diagnostics for a single sub-region extraction
  (kx-omega and ky-omega slices, dispersion curve overlay, SNR search surface).
- **CubeCurrentDiag**: Combined intensity + spectral diagnostics for a whole
  FrameCube (no tiling): time-avg intensity with current vector, kx-omega,
  ky-omega, and (Ux, Uy) search surface.
- **CurrentMapDiag**: Spatial diagnostics for a tiled current map
  (quiver plot, speed colormap, SNR quality map).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.current import CurrentExtractor, CurrentMap, FrameCube

logger = logging.getLogger(__name__)

__all__ = [
    "CubeCurrentDiag",
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
            f"speed={est.speed:.2f} m/s, SNR={est.snr:.2f}, FOM={est.fom:.1f}"
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


class CubeCurrentDiag:
    """Diagnostic 2x2 figure for a single FrameCube: intensity + current vectors + spectra.

    Produces:
    1. Time-averaged intensity I(x, y) with quiver arrows for sub-cube
       current estimates (color-coded by speed).
    2. kx-omega spectrum slice with dispersion curve overlay (whole cube).
    3. ky-omega spectrum slice with dispersion curve overlay (whole cube).
    4. (Ux, Uy) search surface with the estimated current marked (whole cube).

    The spectral panels use the whole cube.  The intensity panel overlays
    current vectors extracted from spatial sub-cubes via
    :func:`compute_tile_specs`.

    After construction the heavy arrays (power spectrum, cube intensity) are
    freed.  Only lightweight 2-D slices (~1.5 MB each), the search surface
    (~30 KB), and the scalar estimate are retained for plotting.

    Args:
        cube: A :class:`FrameCube` with radar intensity data.
        config: Optional configuration object for :class:`CurrentExtractor`.
        n_workers: Number of threads for parallel sub-cube extraction.
    """

    def __init__(
        self,
        cube: FrameCube,
        config: Any = None,
        n_workers: int | None = None,
    ) -> None:
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from scipy.fft import fftshift

        from wamos_tpw.current import CurrentExtractor, compute_tile_specs

        # Tile geometry
        specs = compute_tile_specs(cube, config)
        self._min_snr = specs["min_snr"]
        self._min_fom = specs.get("min_fom", 3.0)

        # Run whole-cube + all sub-cube extractions in parallel
        n_workers = n_workers or os.cpu_count() or 1
        n_tasks = 1 + len(specs["tiles"])  # whole-cube + tiles

        with ThreadPoolExecutor(max_workers=min(n_tasks, n_workers)) as pool:
            # Submit whole-cube extraction
            whole_future = pool.submit(CurrentExtractor, cube, config=config)

            # Submit sub-cube extractions (skip seam/radar-masked tiles)
            tile_futures: dict[Any, dict] = {}
            for tile in specs["tiles"]:
                if tile.get("masked"):
                    continue
                sub = cube.sub_cube(
                    tile["x_start"],
                    tile["x_end"],
                    tile["y_start"],
                    tile["y_end"],
                )
                future = pool.submit(CurrentExtractor, sub, config=config)
                tile_futures[future] = tile

            # Collect whole-cube result
            extractor = whole_future.result()

            # Collect sub-cube results
            self._tile_estimates: list[dict] = []
            for future in as_completed(tile_futures):
                tile = tile_futures[future]
                try:
                    ext = future.result()
                    est = ext.estimate
                    self._tile_estimates.append(
                        {
                            "center_x": tile["center_x"],
                            "center_y": tile["center_y"],
                            "ux": est.ux,
                            "uy": est.uy,
                            "speed": est.speed,
                            "direction": est.direction,
                            "snr": est.snr,
                            "peak_ratio": est.peak_ratio,
                            "fom": est.fom,
                        }
                    )
                except Exception:
                    logger.debug(
                        "Tile (%d, %d) extraction failed",
                        tile["ix"],
                        tile["iy"],
                        exc_info=True,
                    )

        # ---- Cache lightweight plotting data, then free heavy arrays ----

        # Pre-compute mean intensity (n_y, n_x float64) before dropping cube
        self._mean_intensity = np.nanmean(cube.intensity, axis=0)
        self._x_centers = cube.x_centers
        self._y_centers = cube.y_centers

        # Cache the estimate (a tiny dataclass)
        self._estimate = extractor.estimate

        # Cache fftshift-ed 2-D spectral slices (~1.5 MB total)
        P = extractor.power_spectrum  # (n_t, n_ky, n_kx)
        ky_zero_idx = int(np.argmin(np.abs(extractor.ky)))
        kx_zero_idx = int(np.argmin(np.abs(extractor.kx)))
        self._kx_omega_slice = fftshift(P[:, ky_zero_idx, :])  # (n_t, n_kx)
        self._ky_omega_slice = fftshift(P[:, :, kx_zero_idx])  # (n_t, n_ky)
        self._kx_shifted = fftshift(extractor.kx)
        self._ky_shifted = fftshift(extractor.ky)
        self._omega_shifted = fftshift(extractor.omega)

        # Cache search surface (~30 KB) and its parameters
        self._search_surface = extractor._search_surface
        self._search_radius = extractor._search_radius
        self._search_step = extractor._search_step

        # Heavy objects are now unreferenced and eligible for GC
        del extractor

    @property
    def estimate(self) -> Any:
        """The whole-cube :class:`CurrentEstimate`."""
        return self._estimate

    @property
    def tile_estimates(self) -> list[dict]:
        """Per-tile extraction results (center_x/y, ux, uy, speed, snr)."""
        return self._tile_estimates

    def plot(self, fig: Any = None) -> None:
        """Draw the 2x2 diagnostic figure.

        Args:
            fig: Optional matplotlib Figure. Creates one if not provided.
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        else:
            axes = fig.subplots(2, 2)

        self._plot_intensity_with_current(axes[0, 0])
        self._plot_kx_omega(axes[0, 1])
        self._plot_ky_omega(axes[1, 0])
        self._plot_search_surface(axes[1, 1])

        fig.tight_layout()

    def _plot_intensity_with_current(self, ax: Any) -> None:
        """Plot time-averaged intensity with sub-cube current vectors."""
        import matplotlib.pyplot as plt

        est = self._estimate

        im = ax.pcolormesh(
            self._x_centers,
            self._y_centers,
            self._mean_intensity,
            shading="auto",
            cmap="gray",
        )
        plt.colorbar(im, ax=ax, label="Intensity")

        # Sub-cube current vectors (filtered by min_fom)
        valid = [t for t in self._tile_estimates if t["fom"] >= self._min_fom]
        if valid:
            cx = np.array([t["center_x"] for t in valid])
            cy = np.array([t["center_y"] for t in valid])
            ux = np.array([t["ux"] for t in valid])
            uy = np.array([t["uy"] for t in valid])
            speed = np.array([t["speed"] for t in valid])

            q = ax.quiver(
                cx,
                cy,
                ux,
                uy,
                speed,
                cmap="coolwarm",
                scale_units="xy",
                angles="xy",
                width=0.004,
                zorder=5,
            )
            plt.colorbar(q, ax=ax, label="Speed (m/s)")

        n_valid = len(valid)
        n_total = len(self._tile_estimates)

        # Annotate with whole-cube summary
        ax.annotate(
            f"whole: {est.speed:.2f} m/s, {est.direction:.1f}\u00b0, "
            f"SNR={est.snr:.2f}, FOM={est.fom:.1f}\n"
            f"tiles: {n_valid}/{n_total} (FOM\u2265{self._min_fom:.1f})",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            verticalalignment="top",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.6},
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Time-avg intensity + current")
        ax.set_aspect("equal")

    def _plot_kx_omega(self, ax: Any) -> None:
        """Plot kx-omega slice at ky=0 with dispersion curve overlay."""
        from wamos_tpw.current import dispersion_relation

        est = self._estimate
        log_power = np.log10(np.maximum(self._kx_omega_slice, 1e-10))

        ax.pcolormesh(
            self._kx_shifted,
            self._omega_shifted,
            log_power,
            shading="auto",
            cmap="viridis",
        )

        k_plot = np.linspace(self._kx_shifted.min(), self._kx_shifted.max(), 200)
        omega_0 = dispersion_relation(np.abs(k_plot), est.depth)
        omega_disp_pos = -(omega_0 + k_plot * est.ux)
        omega_disp_neg = omega_0 - k_plot * est.ux

        ax.plot(k_plot, omega_disp_pos, "r-", linewidth=1.5, label="Dispersion (+)")
        ax.plot(k_plot, omega_disp_neg, "r--", linewidth=1.0, label="Dispersion (-)")

        ax.set_xlabel("kx (rad/m)")
        ax.set_ylabel("omega (rad/s)")
        ax.set_title("P(kx, omega) at ky=0")
        ax.legend(fontsize=7)

    def _plot_ky_omega(self, ax: Any) -> None:
        """Plot ky-omega slice at kx=0 with dispersion curve overlay."""
        from wamos_tpw.current import dispersion_relation

        est = self._estimate
        log_power = np.log10(np.maximum(self._ky_omega_slice, 1e-10))

        ax.pcolormesh(
            self._ky_shifted,
            self._omega_shifted,
            log_power,
            shading="auto",
            cmap="viridis",
        )

        k_plot = np.linspace(self._ky_shifted.min(), self._ky_shifted.max(), 200)
        omega_0 = dispersion_relation(np.abs(k_plot), est.depth)
        omega_disp_pos = -(omega_0 + k_plot * est.uy)
        omega_disp_neg = omega_0 - k_plot * est.uy

        ax.plot(k_plot, omega_disp_pos, "r-", linewidth=1.5, label="Dispersion (+)")
        ax.plot(k_plot, omega_disp_neg, "r--", linewidth=1.0, label="Dispersion (-)")

        ax.set_xlabel("ky (rad/m)")
        ax.set_ylabel("omega (rad/s)")
        ax.set_title("P(ky, omega) at kx=0")
        ax.legend(fontsize=7)

    def _plot_search_surface(self, ax: Any) -> None:
        """Plot the (Ux, Uy) search surface with the maximum marked."""
        est = self._estimate
        radius = self._search_radius
        step = self._search_step
        candidates = np.arange(-radius, radius + step / 2, step)

        ax.pcolormesh(
            candidates,
            candidates,
            self._search_surface,
            shading="auto",
            cmap="hot",
        )

        ax.plot(est.ux, est.uy, "cx", markersize=12, markeredgewidth=2, label="Estimate")
        ax.set_xlabel("Ux (m/s)")
        ax.set_ylabel("Uy (m/s)")
        ax.set_title(
            f"Shell energy surface\n"
            f"Ux={est.ux:.2f}, Uy={est.uy:.2f}, "
            f"speed={est.speed:.2f} m/s, SNR={est.snr:.2f}, FOM={est.fom:.1f}"
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "w--", linewidth=0.5, alpha=0.5)
