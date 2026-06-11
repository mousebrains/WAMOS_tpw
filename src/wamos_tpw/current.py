#! /usr/bin/env python3
#
# Surface current extraction via 3D FFT dispersion fitting
#
# Extracts surface current vectors from sequential radar images by fitting
# the gravity wave dispersion relation to the 3D (kx, ky, omega) power spectrum.
# Based on Young (1985), Senet (2001), Nieto-Borge (2004).
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""
Surface current extraction from sequential radar images via 3D FFT dispersion fitting.

This module provides:

1. **FrameCube** - 3D space-time cube I(x, y, t) from sequential projected frames
2. **CurrentEstimate** - Single current vector estimate with quality metrics
3. **CurrentExtractor** - Core algorithm: 3D FFT, dispersion fitting, grid search + refinement
4. **CurrentMap** - Spatial field of current vectors from tiled extraction

Algorithm Summary::

    Individual projected frames -> FrameCube I(x, y, t)
      -> tile into sub-regions (e.g. 2km x 2km)
      -> per sub-region:
          remove temporal mean, apply 3D Hann window, handle NaN
          -> 3D FFT -> power spectrum P(kx, ky, omega)
          -> grid search over (Ux, Uy) candidates:
              for each candidate, mean P per bin on the Doppler-shifted
              dispersion shell (k_min < k <= k_max, omega_min <= |omega| < Nyquist)
          -> (Ux, Uy) that maximizes mean shell power -> refine with scipy.optimize
          -> CurrentEstimate(ux, uy, speed, direction, snr)
      -> assemble into CurrentMap (spatial vector field)

Example::

    from wamos_tpw.current import FrameCube, CurrentMap

    cube = FrameCube.from_frame_dicts(frames, grid_params)
    current_map = CurrentMap.from_cube(cube, config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal

if TYPE_CHECKING:
    from wamos_tpw.config import Config
    from wamos_tpw.grid import GridParams

logger = logging.getLogger(__name__)

__all__ = [
    "CurrentEstimate",
    "CurrentExtractor",
    "CurrentMap",
    "FrameCube",
    "compute_multiscale_tile_specs",
    "compute_tile_specs",
    "dispersion_relation",
]

# Physical constants
_G = 9.81  # gravitational acceleration (m/s^2)

# Default configuration values
_BLOCK_FRAMES_DEFAULT = 32
_BLOCK_OVERLAP_DEFAULT = 0.5
_SUB_REGION_SIZE_DEFAULT = 2000.0  # meters
_SUB_REGION_OVERLAP_DEFAULT = 0.5
_DEPTH_DEFAULT = np.inf
_SEARCH_RADIUS_DEFAULT = 3.0  # m/s
_SEARCH_STEP_DEFAULT = 0.1  # m/s
_REFINE_DEFAULT = True
_MIN_SNR_DEFAULT = 1.5
_MIN_FOM_DEFAULT = 3.0
_FFT_WINDOW_DEFAULT = "hann"
_K_MIN_FACTOR_DEFAULT = 2.0
# Exclude shell samples with |omega| < omega_min_factor * d_omega: the
# residual static pattern and its window leakage concentrate enormous
# power near omega = 0, which would otherwise attract the shell fit.
_OMEGA_MIN_FACTOR_DEFAULT = 2.0
# Reject (Ux, Uy) candidates whose dispersion shell retains fewer than
# this fraction of the maximum possible shell bins (2 branches x n_valid):
# means over very few bins are noise-dominated.
_MIN_SHELL_FRACTION_DEFAULT = 0.25
# Mask tiles containing the radar or crossed by the antenna rotation seam
# (pixels on either side of the seam were observed a full rotation apart)
_MASK_SEAM_DEFAULT = True
# Mask tiles whose center is farther than this from the radar (meters);
# None disables. Real-data validation (Revelle 2022, Thompson 2025) shows
# the wave signal degrades steadily with range — tile scatter grows from
# ~0.5 m/s inside 2 km to >2 m/s beyond 4 km — so ~3000 is a good value
# for these installations.
_MAX_TILE_RANGE_DEFAULT = None


# ============================================================
# Data Structures
# ============================================================


@dataclass
class FrameCube:
    """3D space-time cube of radar intensity I(x, y, t).

    Attributes:
        intensity: 3D array (n_t, n_y, n_x) of radar intensity values.
        timestamps: 1D array of timestamps for each time step.
        dt: Mean time step in seconds.
        x_centers: 1D array of x grid centers in meters.
        y_centers: 1D array of y grid centers in meters.
        grid_spacing: Grid cell size in meters.
        center_lat: Grid center latitude in degrees.
        center_lon: Grid center longitude in degrees.
        ship_x: Per-frame radar x position in cube coordinates (meters),
                or None when unknown (e.g. synthetic cubes).
        ship_y: Per-frame radar y position in cube coordinates (meters).
        seam_bearing: Per-frame earth bearing (deg) of the antenna rotation
                seam (radial 0); NaN when unknown. Pixels on either side of
                the seam radial were observed a full rotation apart.
    """

    intensity: np.ndarray  # (n_t, n_y, n_x)
    timestamps: np.ndarray
    dt: float
    x_centers: np.ndarray
    y_centers: np.ndarray
    grid_spacing: float
    center_lat: float
    center_lon: float
    ship_x: np.ndarray | None = None
    ship_y: np.ndarray | None = None
    seam_bearing: np.ndarray | None = None

    @property
    def n_t(self) -> int:
        """Number of time steps."""
        return self.intensity.shape[0]

    @property
    def n_y(self) -> int:
        """Number of y grid cells."""
        return self.intensity.shape[1]

    @property
    def n_x(self) -> int:
        """Number of x grid cells."""
        return self.intensity.shape[2]

    @classmethod
    def from_frame_dicts(
        cls,
        frames: list[dict],
        grid_params: GridParams,
    ) -> FrameCube:
        """Build a FrameCube by remapping individual frame dicts onto a common grid.

        Each frame dict should contain the keys produced by the interpolation
        pipeline: ``projected_intensity``, ``projected_count``, ``grid_params``,
        ``timestamp``.

        Args:
            frames: List of interpolated frame data dicts.
            grid_params: Common grid parameters (from ``compute_common_grid*``).

        Returns:
            FrameCube with individual (not averaged) frames stacked along time.
        """
        from wamos_tpw.grid import remap_to_common_grid

        _DEG2M = 111_319.5

        # Sort frames by timestamp
        frames = sorted(frames, key=lambda f: f["timestamp"])

        n_y = grid_params["n_y"]
        n_x = grid_params["n_x"]
        n_t = len(frames)

        intensity_cube = np.full((n_t, n_y, n_x), np.nan, dtype=np.float64)
        timestamps = np.empty(n_t, dtype="datetime64[ns]")
        ship_x = np.full(n_t, np.nan)
        ship_y = np.full(n_t, np.nan)
        seam_bearing = np.full(n_t, np.nan)

        # Grid center in absolute equirectangular meters, for converting
        # ship positions to cube-centered coordinates
        x_edges_abs = grid_params.get("x_edges_abs")
        y_edges_abs = grid_params.get("y_edges_abs")
        ref_lat = grid_params.get("ref_lat")
        ref_lon = grid_params.get("ref_lon")
        m_per_deg = grid_params.get("m_per_deg_lon")
        have_abs = (
            x_edges_abs is not None
            and y_edges_abs is not None
            and ref_lat is not None
            and ref_lon is not None
            and m_per_deg is not None
        )
        if have_abs:
            x_center_abs = (x_edges_abs[0] + x_edges_abs[-1]) / 2
            y_center_abs = (y_edges_abs[0] + y_edges_abs[-1]) / 2

        for i, frame_data in enumerate(frames):
            timestamps[i] = frame_data["timestamp"]

            stats = frame_data.get("position_stats")
            if have_abs and stats is not None:
                ship_x[i] = (stats["lon_mean"] - ref_lon) * m_per_deg - x_center_abs
                ship_y[i] = (stats["lat_mean"] - ref_lat) * _DEG2M - y_center_abs
            seam = frame_data.get("seam_bearing")
            if seam is not None:
                seam_bearing[i] = seam

            proj_intensity = frame_data.get("projected_intensity")
            if proj_intensity is None:
                continue

            proj_count = frame_data.get("projected_count")
            frame_gp = frame_data.get("grid_params") or {}

            frame_x_edges = frame_gp.get("x_edges")
            frame_y_edges = frame_gp.get("y_edges")
            frame_center_lat = frame_gp.get("center_lat")
            frame_center_lon = frame_gp.get("center_lon")

            if (
                frame_x_edges is not None
                and frame_y_edges is not None
                and frame_center_lat is not None
                and frame_center_lon is not None
            ):
                ref_lat = grid_params["ref_lat"]
                ref_lon = grid_params["ref_lon"]
                m_per_deg_lon = grid_params["m_per_deg_lon"]

                frame_center_x = (frame_center_lon - ref_lon) * m_per_deg_lon
                frame_center_y = (frame_center_lat - ref_lat) * _DEG2M
                frame_x_abs = frame_x_edges + frame_center_x
                frame_y_abs = frame_y_edges + frame_center_y

                frame_sum, frame_count = remap_to_common_grid(
                    proj_intensity,
                    proj_count,
                    frame_x_abs,
                    frame_y_abs,
                    grid_params["x_edges_abs"],
                    grid_params["y_edges_abs"],
                    n_x,
                    n_y,
                )
            else:
                frame_sum = np.zeros((n_y, n_x), dtype=np.float64)
                frame_count = np.zeros((n_y, n_x), dtype=np.int32)
                if proj_intensity.shape == frame_sum.shape:
                    valid = ~np.isnan(proj_intensity)
                    frame_sum[valid] = proj_intensity[valid]
                    if proj_count is not None:
                        frame_count[valid] = proj_count[valid]
                    else:
                        frame_count[valid] = 1

            # Convert sum/count to averaged intensity for this single frame
            with np.errstate(invalid="ignore"):
                avg = frame_sum / frame_count
            avg[frame_count == 0] = np.nan
            intensity_cube[i] = avg

        # Compute dt from timestamps
        if n_t >= 2:
            diffs = np.diff(timestamps.astype("datetime64[ms]").astype(np.float64)) / 1000.0
            dt = float(np.median(diffs))
            # The 3D FFT assumes uniform temporal sampling; warn when the
            # antenna rotation period jitters or frames are missing, since
            # that smears energy along the omega axis and biases the fit.
            spread = float(np.max(np.abs(diffs - dt))) if len(diffs) else 0.0
            if dt > 0 and spread > 0.1 * dt:
                logger.warning(
                    "Non-uniform frame intervals: median dt=%.2fs, max deviation=%.2fs "
                    "(%.0f%%); FFT assumes uniform sampling",
                    dt,
                    spread,
                    100.0 * spread / dt,
                )
        else:
            dt = 1.0

        x_centers = (grid_params["x_edges"][:-1] + grid_params["x_edges"][1:]) / 2
        y_centers = (grid_params["y_edges"][:-1] + grid_params["y_edges"][1:]) / 2

        return cls(
            intensity=intensity_cube,
            timestamps=timestamps,
            dt=dt,
            x_centers=x_centers,
            y_centers=y_centers,
            grid_spacing=grid_params["grid_spacing"],
            center_lat=grid_params["center_lat"],
            center_lon=grid_params["center_lon"],
            ship_x=ship_x if np.any(np.isfinite(ship_x)) else None,
            ship_y=ship_y if np.any(np.isfinite(ship_y)) else None,
            seam_bearing=seam_bearing if np.any(np.isfinite(seam_bearing)) else None,
        )

    @classmethod
    def from_arrays(
        cls,
        intensity: np.ndarray,
        dt: float,
        dx: float,
        dy: float | None = None,
        center_lat: float = 0.0,
        center_lon: float = 0.0,
    ) -> FrameCube:
        """Build a FrameCube from raw arrays (useful for testing).

        Args:
            intensity: 3D array (n_t, n_y, n_x).
            dt: Time step in seconds.
            dx: Grid spacing in x (meters).
            dy: Grid spacing in y (meters). Defaults to dx.
            center_lat: Grid center latitude.
            center_lon: Grid center longitude.

        Returns:
            FrameCube.
        """
        if dy is None:
            dy = dx
        n_t, n_y, n_x = intensity.shape
        x_centers = np.arange(n_x) * dx - (n_x - 1) * dx / 2
        y_centers = np.arange(n_y) * dy - (n_y - 1) * dy / 2
        timestamps = np.arange(n_t).astype("timedelta64[ms]") * int(dt * 1000) + np.datetime64(
            "2022-01-01"
        )

        return cls(
            intensity=intensity.astype(np.float64),
            timestamps=timestamps,
            dt=dt,
            x_centers=x_centers,
            y_centers=y_centers,
            grid_spacing=dx,
            center_lat=center_lat,
            center_lon=center_lon,
        )

    def sub_cube(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ) -> FrameCube:
        """Extract a spatial sub-region of the cube.

        Args:
            x_start: Start x index (inclusive).
            x_end: End x index (exclusive).
            y_start: Start y index (inclusive).
            y_end: End y index (exclusive).

        Returns:
            New FrameCube for the sub-region.
        """
        return FrameCube(
            intensity=self.intensity[:, y_start:y_end, x_start:x_end].copy(),
            timestamps=self.timestamps,
            dt=self.dt,
            x_centers=self.x_centers[x_start:x_end],
            y_centers=self.y_centers[y_start:y_end],
            grid_spacing=self.grid_spacing,
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            # Ship track and seam are in whole-cube coordinates, which the
            # sliced x/y centers preserve (sub_cube does not re-center)
            ship_x=self.ship_x,
            ship_y=self.ship_y,
            seam_bearing=self.seam_bearing,
        )


@dataclass
class CurrentEstimate:
    """A single surface current vector estimate.

    Attributes:
        ux: Eastward current component (m/s).
        uy: Northward current component (m/s).
        speed: Current speed (m/s).
        direction: Current direction in degrees (direction current flows TO,
                   measured clockwise from north).
        snr: Signal-to-noise ratio (energy on dispersion shell / off-shell energy).
        depth: Water depth used for the estimate (meters, inf for deep water).
        center_x: X position of the sub-region center (meters).
        center_y: Y position of the sub-region center (meters).
        peak_ratio: Max/mean of the (Ux, Uy) search surface.
        fom: Figure of merit (snr * peak_ratio).
        ux_err: 1-sigma uncertainty of ux from the least-squares dispersion
                fit (m/s, NaN when refinement did not run).
        uy_err: 1-sigma uncertainty of uy (m/s, NaN when unavailable).
        cov_uxuy: Covariance between ux and uy from the least-squares fit
                (m^2/s^2, NaN when unavailable).
        n_ls_points: Number of spectral peaks used by the least-squares fit.
        ls_rms: RMS frequency residual of the least-squares fit (rad/s).
        window_size: Side length (m) of the analysis window that produced
                this estimate (0 when unknown); the estimate is an average
                of the current over that footprint.
    """

    ux: float
    uy: float
    speed: float
    direction: float
    snr: float
    depth: float
    center_x: float = 0.0
    center_y: float = 0.0
    peak_ratio: float = 1.0
    fom: float = 0.0
    ux_err: float = float("nan")
    uy_err: float = float("nan")
    cov_uxuy: float = float("nan")
    n_ls_points: int = 0
    ls_rms: float = float("nan")
    window_size: float = 0.0


@dataclass
class CurrentMap:
    """Spatial field of current vectors from tiled extraction.

    Attributes:
        ux: 2D array of eastward current (m/s), shape (n_tiles_y, n_tiles_x).
        uy: 2D array of northward current (m/s).
        speed: 2D array of current speed (m/s).
        direction: 2D array of current direction (degrees, TO convention).
        snr: 2D array of signal-to-noise ratio.
        tile_x_centers: 1D array of tile center x positions (meters).
        tile_y_centers: 1D array of tile center y positions (meters).
        depth: Water depth used (meters).
        center_lat: Grid center latitude.
        center_lon: Grid center longitude.
        start_time: Start time of the analysis block.
        end_time: End time of the analysis block.
        estimates: List of all CurrentEstimate objects (including rejected).
        ux_err: 2D array of 1-sigma ux uncertainties (m/s) from the
                least-squares dispersion fit (None for legacy maps).
        uy_err: 2D array of 1-sigma uy uncertainties (m/s).
    """

    ux: np.ndarray
    uy: np.ndarray
    speed: np.ndarray
    direction: np.ndarray
    snr: np.ndarray
    tile_x_centers: np.ndarray
    tile_y_centers: np.ndarray
    depth: float
    center_lat: float
    center_lon: float
    start_time: np.datetime64 = field(default_factory=lambda: np.datetime64("NaT"))
    end_time: np.datetime64 = field(default_factory=lambda: np.datetime64("NaT"))
    estimates: list[CurrentEstimate] = field(default_factory=list)
    ux_err: np.ndarray | None = None
    uy_err: np.ndarray | None = None

    @property
    def n_tiles_x(self) -> int:
        """Number of tiles in x direction."""
        return len(self.tile_x_centers)

    @property
    def n_tiles_y(self) -> int:
        """Number of tiles in y direction."""
        return len(self.tile_y_centers)

    @classmethod
    def from_cube(
        cls,
        cube: FrameCube,
        config: Config | None = None,
    ) -> CurrentMap:
        """Extract surface currents from a FrameCube using tiled analysis.

        Divides the spatial domain into overlapping sub-regions, runs
        :class:`CurrentExtractor` on each, and assembles the results
        into a spatial current map.

        Args:
            cube: 3D space-time cube of radar intensity.
            config: Configuration object for extraction parameters.

        Returns:
            CurrentMap with current vectors at each tile center.
        """
        from wamos_tpw.config import NullConfig

        cfg = config or NullConfig()
        specs = compute_tile_specs(cube, cfg)
        min_snr = specs["min_snr"]

        # Allocate output arrays
        n_tiles_y = specs["n_tiles_y"]
        n_tiles_x = specs["n_tiles_x"]
        ux_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        uy_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        speed_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        dir_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        snr_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        ux_err_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        uy_err_map = np.full((n_tiles_y, n_tiles_x), np.nan)

        estimates: list[CurrentEstimate] = []

        for tile in specs["tiles"]:
            if tile.get("masked"):
                continue
            iy = tile["iy"]
            ix = tile["ix"]
            sub = cube.sub_cube(tile["x_start"], tile["x_end"], tile["y_start"], tile["y_end"])

            try:
                extractor = CurrentExtractor(sub, config=cfg)
                est = extractor.estimate
            except Exception:
                logger.debug(
                    "Extraction failed for tile (%d, %d)",
                    ix,
                    iy,
                    exc_info=True,
                )
                continue

            est = CurrentEstimate(
                ux=est.ux,
                uy=est.uy,
                speed=est.speed,
                direction=est.direction,
                snr=est.snr,
                depth=est.depth,
                center_x=tile["center_x"],
                center_y=tile["center_y"],
                peak_ratio=est.peak_ratio,
                fom=est.fom,
                ux_err=est.ux_err,
                uy_err=est.uy_err,
                n_ls_points=est.n_ls_points,
                ls_rms=est.ls_rms,
            )
            estimates.append(est)

            snr_map[iy, ix] = est.snr
            if est.snr >= min_snr:
                ux_map[iy, ix] = est.ux
                uy_map[iy, ix] = est.uy
                speed_map[iy, ix] = est.speed
                dir_map[iy, ix] = est.direction
                ux_err_map[iy, ix] = est.ux_err
                uy_err_map[iy, ix] = est.uy_err

        return cls(
            ux=ux_map,
            uy=uy_map,
            speed=speed_map,
            direction=dir_map,
            snr=snr_map,
            tile_x_centers=specs["tile_x_centers"],
            tile_y_centers=specs["tile_y_centers"],
            depth=specs["depth"],
            center_lat=cube.center_lat,
            center_lon=cube.center_lon,
            start_time=cube.timestamps[0],
            end_time=cube.timestamps[-1],
            estimates=estimates,
            ux_err=ux_err_map,
            uy_err=uy_err_map,
        )

    @classmethod
    def from_tile_results(
        cls,
        tile_specs: dict,
        tile_results: list[dict],
        cube_metadata: dict,
    ) -> CurrentMap:
        """Assemble a CurrentMap from parallel tile extraction results.

        Args:
            tile_specs: Tile geometry from :func:`compute_tile_specs`.
            tile_results: List of result dicts from parallel tile workers.
                Each dict has keys: ix, iy, ux, uy, speed, direction, snr,
                depth, center_x, center_y. May have ``error`` key for failures.
            cube_metadata: Dict with ``center_lat``, ``center_lon``,
                ``start_time``, ``end_time``.

        Returns:
            CurrentMap with current vectors at each tile center.
        """
        n_tiles_y = tile_specs["n_tiles_y"]
        n_tiles_x = tile_specs["n_tiles_x"]
        min_snr = tile_specs["min_snr"]

        ux_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        uy_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        speed_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        dir_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        snr_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        ux_err_map = np.full((n_tiles_y, n_tiles_x), np.nan)
        uy_err_map = np.full((n_tiles_y, n_tiles_x), np.nan)

        estimates: list[CurrentEstimate] = []

        for result in tile_results:
            if result.get("error"):
                continue

            iy = result["iy"]
            ix = result["ix"]
            est = CurrentEstimate(
                ux=result["ux"],
                uy=result["uy"],
                speed=result["speed"],
                direction=result["direction"],
                snr=result["snr"],
                depth=result["depth"],
                center_x=result["center_x"],
                center_y=result["center_y"],
                peak_ratio=result.get("peak_ratio", 1.0),
                fom=result.get("fom", 0.0),
                ux_err=result.get("ux_err", float("nan")),
                uy_err=result.get("uy_err", float("nan")),
                cov_uxuy=result.get("cov_uxuy", float("nan")),
                n_ls_points=result.get("n_ls_points", 0),
                ls_rms=result.get("ls_rms", float("nan")),
                window_size=result.get("window_size", 0.0),
            )
            estimates.append(est)

            # Only primary-scale tiles populate the regular map arrays;
            # extra scales contribute estimates for the field inversion
            if result.get("scale", 0) != 0:
                continue

            snr_map[iy, ix] = est.snr
            if est.snr >= min_snr:
                ux_map[iy, ix] = est.ux
                uy_map[iy, ix] = est.uy
                speed_map[iy, ix] = est.speed
                dir_map[iy, ix] = est.direction
                ux_err_map[iy, ix] = est.ux_err
                uy_err_map[iy, ix] = est.uy_err

        return cls(
            ux=ux_map,
            uy=uy_map,
            speed=speed_map,
            direction=dir_map,
            snr=snr_map,
            tile_x_centers=tile_specs["tile_x_centers"],
            tile_y_centers=tile_specs["tile_y_centers"],
            depth=tile_specs["depth"],
            center_lat=cube_metadata["center_lat"],
            center_lon=cube_metadata["center_lon"],
            start_time=cube_metadata["start_time"],
            end_time=cube_metadata["end_time"],
            estimates=estimates,
            ux_err=ux_err_map,
            uy_err=uy_err_map,
        )


# ============================================================
# Tile Geometry
# ============================================================


def compute_tile_specs(
    cube: FrameCube,
    config: Config | None = None,
    sub_region_size: float | None = None,
) -> dict:
    """Compute tile geometry for spatial current extraction.

    Divides the cube's spatial domain into overlapping sub-regions
    suitable for independent current extraction.

    Args:
        cube: 3D space-time cube of radar intensity.
        config: Configuration object for extraction parameters.
        sub_region_size: Override the configured window size (meters);
            used by :func:`compute_multiscale_tile_specs`.

    Returns:
        Dict with keys:
        - ``n_tiles_x``, ``n_tiles_y``: Number of tiles in each direction.
        - ``tiles``: List of tile dicts with ``ix``, ``iy``, ``x_start``,
          ``x_end``, ``y_start``, ``y_end``, ``center_x``, ``center_y``,
          ``masked``, ``scale``.
        - ``tile_x_centers``, ``tile_y_centers``: 1D arrays of tile center
          positions (meters).
        - ``min_snr``: Minimum SNR threshold.
        - ``depth``: Water depth (meters).
    """
    from wamos_tpw.config import NullConfig

    cfg = config or NullConfig()

    if sub_region_size is None:
        sub_region_size = cfg.get("current.sub_region_size", _SUB_REGION_SIZE_DEFAULT)
    sub_region_overlap = cfg.get("current.sub_region_overlap", _SUB_REGION_OVERLAP_DEFAULT)
    min_snr = cfg.get("current.min_snr", _MIN_SNR_DEFAULT)
    min_fom = cfg.get("current.min_fom", _MIN_FOM_DEFAULT)
    depth = cfg.get("current.depth", _DEPTH_DEFAULT)
    mask_seam = cfg.get("current.mask_seam", _MASK_SEAM_DEFAULT)
    max_tile_range = cfg.get("current.max_tile_range", _MAX_TILE_RANGE_DEFAULT)

    # Median radar position in cube coordinates, for range gating
    radar_x = radar_y = None
    if (
        max_tile_range is not None
        and cube.ship_x is not None
        and cube.ship_y is not None
        and np.any(np.isfinite(cube.ship_x))
    ):
        radar_x = float(np.nanmedian(cube.ship_x))
        radar_y = float(np.nanmedian(cube.ship_y))

    stride = sub_region_size * (1.0 - sub_region_overlap)
    tile_size_pixels = int(round(sub_region_size / cube.grid_spacing))

    # X tiles
    x_range = cube.x_centers[-1] - cube.x_centers[0]
    n_tiles_x = max(1, int((x_range - sub_region_size) / stride) + 1)
    if x_range < sub_region_size:
        n_tiles_x = 1

    # Y tiles
    y_range = cube.y_centers[-1] - cube.y_centers[0]
    n_tiles_y = max(1, int((y_range - sub_region_size) / stride) + 1)
    if y_range < sub_region_size:
        n_tiles_y = 1

    # Compute tile start positions
    if n_tiles_x == 1:
        tile_x_starts = [0]
    else:
        tile_x_starts = [int(round(i * stride / cube.grid_spacing)) for i in range(n_tiles_x)]
    if n_tiles_y == 1:
        tile_y_starts = [0]
    else:
        tile_y_starts = [int(round(i * stride / cube.grid_spacing)) for i in range(n_tiles_y)]

    tile_x_centers = np.empty(n_tiles_x)
    tile_y_centers = np.empty(n_tiles_y)
    tiles: list[dict] = []

    for iy, y_start in enumerate(tile_y_starts):
        y_end = min(y_start + tile_size_pixels, cube.n_y)
        if y_end - y_start < tile_size_pixels // 2:
            y_start = max(0, cube.n_y - tile_size_pixels)
            y_end = cube.n_y

        cy = float(np.mean(cube.y_centers[y_start:y_end]))
        tile_y_centers[iy] = cy

        for ix, x_start in enumerate(tile_x_starts):
            x_end = min(x_start + tile_size_pixels, cube.n_x)
            if x_end - x_start < tile_size_pixels // 2:
                x_start = max(0, cube.n_x - tile_size_pixels)
                x_end = cube.n_x

            cx = float(np.mean(cube.x_centers[x_start:x_end]))
            tile_x_centers[ix] = cx

            half = cube.grid_spacing / 2
            tile_bounds = (
                float(cube.x_centers[x_start] - half),
                float(cube.x_centers[x_end - 1] + half),
                float(cube.y_centers[y_start] - half),
                float(cube.y_centers[y_end - 1] + half),
            )

            masked = mask_seam and _tile_contains_seam(cube, *tile_bounds)
            if not masked and radar_x is not None:
                masked = np.hypot(cx - radar_x, cy - radar_y) > max_tile_range

            tiles.append(
                {
                    "ix": ix,
                    "iy": iy,
                    "x_start": x_start,
                    "x_end": x_end,
                    "y_start": y_start,
                    "y_end": y_end,
                    "center_x": cx,
                    "center_y": cy,
                    "masked": masked,
                    "scale": 0,
                }
            )

    n_masked = sum(1 for t in tiles if t["masked"])
    if n_masked:
        logger.debug(
            "Masked %d/%d tiles (seam, radar inside, or beyond max range)",
            n_masked,
            len(tiles),
        )

    return {
        "n_tiles_x": n_tiles_x,
        "n_tiles_y": n_tiles_y,
        "tiles": tiles,
        "tile_x_centers": tile_x_centers,
        "tile_y_centers": tile_y_centers,
        "min_snr": min_snr,
        "min_fom": min_fom,
        "depth": depth,
    }


def compute_multiscale_tile_specs(cube: FrameCube, config: Config | None = None) -> dict:
    """Compute tile geometry over one or more window sizes.

    The first entry of ``current.window_sizes`` is the primary scale and
    defines the regular tile grid of the resulting :class:`CurrentMap`;
    tiles from additional sizes are appended with ``scale >= 1`` and only
    contribute estimates (for the joint field inversion in
    :mod:`wamos_tpw.current_field`), never the map arrays.

    Falls back to plain :func:`compute_tile_specs` when
    ``current.window_sizes`` is not configured.

    Args:
        cube: 3D space-time cube of radar intensity.
        config: Configuration object.

    Returns:
        Spec dict as :func:`compute_tile_specs`, with extra-scale tiles
        appended to ``tiles`` and a ``window_sizes`` key.
    """
    from wamos_tpw.config import NullConfig

    cfg = config or NullConfig()
    sizes = cfg.get("current.window_sizes", None)
    if not sizes:
        return compute_tile_specs(cube, config)

    sizes = [float(s) for s in sizes]
    specs = compute_tile_specs(cube, config, sub_region_size=sizes[0])

    for scale, size in enumerate(sizes[1:], start=1):
        extra = compute_tile_specs(cube, config, sub_region_size=size)
        for tile in extra["tiles"]:
            tile["scale"] = scale
            specs["tiles"].append(tile)

    specs["window_sizes"] = sizes
    return specs


def _ray_intersects_rect(
    px: float,
    py: float,
    bearing_deg: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Test whether a ray from (px, py) at a compass bearing crosses a rectangle.

    Bearing is degrees clockwise from north, so the direction vector is
    (sin(bearing), cos(bearing)). Uses the slab method; a ray starting
    inside the rectangle counts as intersecting.
    """
    b = np.deg2rad(bearing_deg)
    dx = float(np.sin(b))
    dy = float(np.cos(b))

    t_min = 0.0
    t_max = np.inf

    for p, d, lo, hi in ((px, dx, x_min, x_max), (py, dy, y_min, y_max)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return False
        else:
            t1 = (lo - p) / d
            t2 = (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False

    return True


def _tile_contains_seam(
    cube: FrameCube,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Check whether any frame's rotation seam (or the radar itself) is in the tile.

    Pixels on either side of the seam radial were observed a full antenna
    rotation apart, so a tile the seam crosses contains a time discontinuity
    of ~dt within single "snapshots". A tile containing the radar position
    spans all azimuths and always contains the seam.

    Returns False when the cube has no ship-track information (synthetic
    cubes), leaving such cubes unmasked.
    """
    if cube.ship_x is None or cube.ship_y is None:
        return False

    seam = cube.seam_bearing if cube.seam_bearing is not None else np.full(len(cube.ship_x), np.nan)

    for px, py, bearing in zip(cube.ship_x, cube.ship_y, seam, strict=True):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if x_min <= px <= x_max and y_min <= py <= y_max:
            return True
        if np.isfinite(bearing) and _ray_intersects_rect(
            float(px), float(py), float(bearing), x_min, x_max, y_min, y_max
        ):
            return True

    return False


# ============================================================
# Dispersion Relation
# ============================================================


def dispersion_relation(k: np.ndarray, depth: float = np.inf) -> np.ndarray:
    """Compute intrinsic angular frequency from the gravity wave dispersion relation.

    .. math::

        \\omega_0 = \\sqrt{g |k| \\tanh(|k| d)}

    For deep water (depth -> inf), simplifies to:

    .. math::

        \\omega_0 = \\sqrt{g |k|}

    Args:
        k: Wavenumber magnitude (rad/m). Can be scalar or array.
        depth: Water depth in meters. Use ``np.inf`` for deep water.

    Returns:
        Intrinsic angular frequency (rad/s), same shape as *k*.
    """
    k = np.asarray(k, dtype=np.float64)
    k_abs = np.abs(k)

    if not np.isfinite(depth):
        return np.sqrt(_G * k_abs)

    # Finite depth: clamp argument to avoid overflow in tanh
    kd = np.minimum(k_abs * depth, 700.0)
    return np.sqrt(_G * k_abs * np.tanh(kd))


# ============================================================
# Current Extractor
# ============================================================


class CurrentExtractor:
    """Extract surface current from a single sub-region 3D cube.

    Runs the full pipeline: NaN fill, detrend, window, 3D FFT,
    grid search over (Ux, Uy), optional refinement, SNR computation.

    Args:
        cube: FrameCube for a single sub-region.
        config: Configuration object.

    Attributes:
        estimate: The best-fit CurrentEstimate.
        power_spectrum: 3D power spectrum P(omega, ky, kx) — full 3D FFT.
        kx: 1D wavenumber array (rad/m) along x axis (full FFT frequencies).
        ky: 1D wavenumber array (rad/m) along y axis (full FFT frequencies).
        omega: 1D frequency array (rad/s) along time axis (full FFT frequencies).
    """

    def __init__(self, cube: FrameCube, config: Any = None) -> None:
        from wamos_tpw.config import NullConfig

        self._cube = cube
        self._cfg = config or NullConfig()

        self._depth = self._cfg.get("current.depth", _DEPTH_DEFAULT)
        self._search_radius = self._cfg.get("current.search_radius", _SEARCH_RADIUS_DEFAULT)
        self._search_step = self._cfg.get("current.search_step", _SEARCH_STEP_DEFAULT)
        self._do_refine = self._cfg.get("current.refine", _REFINE_DEFAULT)
        self._k_min_factor = self._cfg.get("current.k_min_factor", _K_MIN_FACTOR_DEFAULT)
        self._k_max = self._cfg.get("current.k_max", None)  # rad/m; None -> pi/(2*dx)
        self._omega_min_factor = self._cfg.get(
            "current.omega_min_factor", _OMEGA_MIN_FACTOR_DEFAULT
        )
        self._min_shell_fraction = self._cfg.get(
            "current.min_shell_fraction", _MIN_SHELL_FRACTION_DEFAULT
        )
        self._window_name = self._cfg.get("current.fft_window", _FFT_WINDOW_DEFAULT)

        # Run the extraction pipeline
        prepared = self._prepare_cube()
        self.power_spectrum, self.kx, self.ky, self.omega = self._compute_3d_fft(prepared)
        self._cache_spectral_geometry()
        coarse_ux, coarse_uy, self._search_surface = self._grid_search()

        self._ls_stats: dict | None = None
        if self._do_refine:
            ux, uy = self._refine(coarse_ux, coarse_uy)
        else:
            ux, uy = coarse_ux, coarse_uy

        speed = float(np.sqrt(ux**2 + uy**2))
        direction = float(np.rad2deg(np.arctan2(ux, uy)) % 360)  # TO convention, CW from N
        snr = self._compute_snr(ux, uy)

        nonzero = self._search_surface[self._search_surface > 0]
        if len(nonzero) > 0:
            peak_ratio = float(np.max(nonzero) / np.mean(nonzero))
        else:
            peak_ratio = 1.0
        fom = snr * peak_ratio

        ls = self._ls_stats or {}
        self.estimate = CurrentEstimate(
            ux=ux,
            uy=uy,
            speed=speed,
            direction=direction,
            snr=snr,
            depth=self._depth,
            center_x=float(np.mean(cube.x_centers)),
            center_y=float(np.mean(cube.y_centers)),
            peak_ratio=peak_ratio,
            fom=fom,
            ux_err=ls.get("ux_err", float("nan")),
            uy_err=ls.get("uy_err", float("nan")),
            cov_uxuy=ls.get("cov_uxuy", float("nan")),
            n_ls_points=ls.get("n_points", 0),
            ls_rms=ls.get("rms", float("nan")),
            window_size=float(
                max(
                    cube.x_centers[-1] - cube.x_centers[0],
                    cube.y_centers[-1] - cube.y_centers[0],
                )
                + cube.grid_spacing
            ),
        )

    def _prepare_cube(self) -> np.ndarray:
        """Prepare the intensity cube for FFT analysis.

        Steps:
        1. Remove the per-pixel temporal mean (computed over valid samples
           only), which removes the static backscatter pattern.
        2. Set missing samples (NaN) to zero anomaly so data gaps — shadow
           sectors, frames with no coverage, pixels outside the radar disc —
           contribute no spurious spectral energy.
        3. Apply 3D window function (e.g. Hann).

        Returns:
            Prepared 3D array ready for FFT.
        """
        data = self._cube.intensity.copy()
        n_t, n_y, n_x = data.shape

        # Steps 1+2: per-pixel temporal mean over valid samples; gaps -> 0
        valid = np.isfinite(data)
        counts = valid.sum(axis=0)
        sums = np.where(valid, data, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            temporal_mean = sums / counts
        data -= temporal_mean
        data[~np.isfinite(data)] = 0.0

        # Step 3: Apply 3D window
        win_t = signal.windows.get_window(self._window_name, n_t, fftbins=False)
        win_y = signal.windows.get_window(self._window_name, n_y, fftbins=False)
        win_x = signal.windows.get_window(self._window_name, n_x, fftbins=False)
        # Apply separable window in-place (avoids full 3D temporary)
        data *= win_t[:, np.newaxis, np.newaxis]
        data *= win_y[np.newaxis, :, np.newaxis]
        data *= win_x[np.newaxis, np.newaxis, :]

        return data

    def _compute_3d_fft(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute full 3D FFT and return power spectrum with frequency axes.

        Uses ``scipy.fft.fftn`` to compute the full 3D power spectrum
        including both positive and negative temporal frequencies.
        The shell energy method evaluates both dispersion branches directly.

        Args:
            data: Prepared 3D array (n_t, n_y, n_x).

        Returns:
            Tuple of (power_spectrum, kx, ky, omega):
            - power_spectrum: 3D array P(n_t, n_y, n_x) — full spectrum.
            - kx: 1D wavenumber array (rad/m), full FFT frequencies.
            - ky: 1D wavenumber array (rad/m), full FFT frequencies.
            - omega: 1D angular frequency array (rad/s), full FFT frequencies.
        """
        n_t, n_y, n_x = data.shape
        dt = self._cube.dt
        dx = self._cube.grid_spacing

        # Full 3D FFT: output shape (n_t, n_y, n_x)
        spectrum = scipy_fft.fftn(data, workers=-1)
        power = np.abs(spectrum) ** 2

        # Frequency axes (full, including negative)
        omega = 2.0 * np.pi * scipy_fft.fftfreq(n_t, d=dt)
        ky = 2.0 * np.pi * scipy_fft.fftfreq(n_y, d=dx)
        kx = 2.0 * np.pi * scipy_fft.fftfreq(n_x, d=dx)

        return power, kx, ky, omega

    def _cache_spectral_geometry(self) -> None:
        """Pre-compute spectral geometry arrays reused across shell energy evaluations.

        Caches wavenumber grids, dispersion relation values, and valid-bin
        indices so that ``_shell_energy`` and ``_grid_search`` avoid
        recomputing meshgrid, sqrt, and dispersion_relation on every call.
        """
        kx_2d, ky_2d = np.meshgrid(self.kx, self.ky)  # (n_ky, n_kx)
        k_mag = np.sqrt(kx_2d**2 + ky_2d**2)

        sub_region_size = max(
            self._cube.x_centers[-1] - self._cube.x_centers[0],
            self._cube.y_centers[-1] - self._cube.y_centers[0],
        )
        k_min = self._k_min_factor * 2.0 * np.pi / sub_region_size if sub_region_size > 0 else 0.0
        # High-wavenumber cutoff: bins near the spatial Nyquist are dominated
        # by speckle, gridding artifacts, and temporally aliased short waves.
        # Default keeps wavelengths >= 4 grid cells.
        k_max = self._k_max
        if k_max is None:
            k_max = np.pi / (2.0 * self._cube.grid_spacing)
        valid = (k_mag > k_min) & (k_mag <= k_max)

        n_t = self.power_spectrum.shape[0]
        d_omega = 2.0 * np.pi / (n_t * self._cube.dt)
        omega_nyq = np.pi / self._cube.dt
        omega_min = self._omega_min_factor * d_omega

        # Extract valid-bin indices and pre-compute per-bin constants
        iy_v, ix_v = np.where(valid)
        self._se_kx: np.ndarray = kx_2d[iy_v, ix_v]  # (n_valid,)
        self._se_ky: np.ndarray = ky_2d[iy_v, ix_v]  # (n_valid,)
        self._se_om0: np.ndarray = dispersion_relation(k_mag[iy_v, ix_v], self._depth)  # (n_valid,)
        self._se_iy: np.ndarray = iy_v
        self._se_ix: np.ndarray = ix_v
        self._se_d_omega: float = d_omega
        self._se_omega_nyq: float = omega_nyq
        self._se_omega_min: float = omega_min
        self._se_n_t: int = n_t

        # Total energy and bin count over the analysis set — the valid k
        # annulus restricted to omega_min <= |omega| < omega_nyq — used by
        # _compute_snr so that noise is estimated from bins the shell could
        # actually occupy (not the static pattern at omega ~ 0 or k bins
        # outside [k_min, k_max]).
        omega_abs = np.abs(self.omega)
        rows_ok = (omega_abs >= omega_min) & (omega_abs < omega_nyq)
        if len(iy_v) > 0 and np.any(rows_ok):
            cols = self.power_spectrum[:, iy_v, ix_v]  # (n_t, n_valid)
            self._se_analysis_energy: float = float(cols[rows_ok].sum())
            self._se_n_analysis: int = int(rows_ok.sum()) * len(iy_v)
            # Noise-floor-subtracted gather table for the fit objective:
            # subtract each (kx, ky) column's median power over omega and
            # clip at zero. Wave signal occupies one or two omega bins per
            # column and stands far above the median, so it survives; the
            # broadband noise pedestal and most static-pattern leakage are
            # removed, so they no longer pull the shell fit toward hot
            # off-dispersion regions.
            self._se_cols_filtered: np.ndarray = np.maximum(
                cols - np.median(cols, axis=0, keepdims=True), 0.0
            )
        else:
            self._se_analysis_energy = 0.0
            self._se_n_analysis = 0
            self._se_cols_filtered = np.zeros((n_t, len(iy_v)))

    def _shell_energy(self, ux: float, uy: float, filtered: bool = False) -> tuple[float, int]:
        """Compute total energy on the Doppler-shifted dispersion shell.

        Uses cached spectral geometry from ``_cache_spectral_geometry``
        to avoid redundant meshgrid / dispersion_relation computation.

        Args:
            ux: Eastward current candidate (m/s).
            uy: Northward current candidate (m/s).
            filtered: When True, sample the noise-floor-subtracted gather
                table (the fit objective); when False, sum raw spectral
                power (for SNR).

        Returns:
            Tuple of (shell energy, number of shell bins sampled).
        """
        doppler = self._se_kx * ux + self._se_ky * uy  # (n_valid,)

        P = self.power_spectrum
        n_t = self._se_n_t
        d_omega = self._se_d_omega
        omega_nyq = self._se_omega_nyq
        omega_min = self._se_omega_min
        iy_v, ix_v = self._se_iy, self._se_ix

        energy = 0.0
        n_shell = 0

        for omega_pred in [-(self._se_om0 + doppler), self._se_om0 - doppler]:
            abs_pred = np.abs(omega_pred)
            ok = (abs_pred < omega_nyq) & (abs_pred >= omega_min)
            op = omega_pred[ok]
            frac = op / d_omega
            lo = np.floor(frac).astype(np.intp)
            w_hi = frac - lo
            lo_w = lo % n_t
            hi_w = (lo + 1) % n_t
            if filtered:
                cols_f = self._se_cols_filtered
                col_idx = np.where(ok)[0]
                contrib = (1.0 - w_hi) * cols_f[lo_w, col_idx] + w_hi * cols_f[hi_w, col_idx]
            else:
                iy_ok, ix_ok = iy_v[ok], ix_v[ok]
                contrib = (1.0 - w_hi) * P[lo_w, iy_ok, ix_ok] + w_hi * P[hi_w, iy_ok, ix_ok]
            energy += float(np.sum(contrib))
            n_shell += int(ok.sum())

        return energy, n_shell

    def _grid_search(self) -> tuple[float, float, np.ndarray]:
        """Vectorized grid search over (Ux, Uy) candidates.

        For each ``uy`` row, batches all valid ``ux`` candidates and
        evaluates both dispersion branches in a single vectorized gather
        from the power spectrum, eliminating the per-candidate Python
        loop and redundant array construction.

        The objective is the *mean* power per shell bin, not the raw sum:
        the number of shell bins inside the temporal Nyquist varies with
        the candidate current (large Doppler shifts move bins in or out of
        range), so a raw sum systematically favors candidates that merely
        sample more bins. Candidates retaining fewer than
        ``min_shell_fraction`` of the possible shell bins are rejected.

        Returns:
            Tuple of (best_ux, best_uy, search_surface).
            search_surface is a 2D array of mean shell power indexed by
            (uy, ux) grid.
        """
        radius = self._search_radius
        step = self._search_step

        candidates = np.arange(-radius, radius + step / 2, step)
        n = len(candidates)
        search_surface = np.zeros((n, n))

        kx_v = self._se_kx
        ky_v = self._se_ky
        om0_v = self._se_om0
        cols_f = self._se_cols_filtered

        n_t = self._se_n_t
        d_omega = self._se_d_omega
        omega_nyq = self._se_omega_nyq
        omega_min = self._se_omega_min

        n_valid = len(kx_v)

        best_energy = -1.0
        best_ux = 0.0
        best_uy = 0.0

        if n_valid == 0:
            logger.debug(
                "Grid search best: Ux=%.2f, Uy=%.2f, energy=%.2e",
                best_ux,
                best_uy,
                best_energy,
            )
            return best_ux, best_uy, search_surface

        min_count = max(1, int(self._min_shell_fraction * 2 * n_valid))
        col_idx = np.arange(n_valid)

        # Auto-chunk ux batch to cap memory (~6 working arrays per element)
        _MAX_BATCH_BYTES = 256 * 1024 * 1024
        chunk_size = max(1, _MAX_BATCH_BYTES // (n_valid * 8 * 6))

        for iy_idx, uy_cand in enumerate(candidates):
            max_ux = np.sqrt(max(0.0, radius**2 - uy_cand**2))
            inside = np.abs(candidates) <= max_ux + step * 0.01
            ix_indices = np.where(inside)[0]
            ux_batch = candidates[inside]
            if len(ux_batch) == 0:
                continue

            ky_contrib = ky_v * uy_cand  # (n_valid,)
            row_energy = np.zeros(len(ux_batch))

            for start in range(0, len(ux_batch), chunk_size):
                end = min(start + chunk_size, len(ux_batch))
                ux_chunk = ux_batch[start:end]  # (n_chunk,)

                # Doppler shift — shape (n_chunk, n_valid)
                doppler = kx_v[np.newaxis, :] * ux_chunk[:, np.newaxis] + ky_contrib[np.newaxis, :]

                chunk_energy = np.zeros(len(ux_chunk))
                chunk_count = np.zeros(len(ux_chunk), dtype=np.intp)
                for sign in (-1.0, 1.0):
                    omega_pred = sign * om0_v[np.newaxis, :] - doppler
                    abs_pred = np.abs(omega_pred)
                    ok = (abs_pred < omega_nyq) & (abs_pred >= omega_min)

                    frac = omega_pred / d_omega
                    lo = np.floor(frac).astype(np.intp)
                    w_hi = frac - lo
                    lo_w = lo % n_t
                    hi_w = (lo + 1) % n_t

                    P_lo = cols_f[lo_w, col_idx[np.newaxis, :]]
                    P_hi = cols_f[hi_w, col_idx[np.newaxis, :]]
                    contrib = (1.0 - w_hi) * P_lo + w_hi * P_hi
                    contrib[~ok] = 0.0
                    chunk_energy += contrib.sum(axis=1)
                    chunk_count += ok.sum(axis=1)

                # Mean power per shell bin; reject sparse shells
                enough = chunk_count >= min_count
                with np.errstate(invalid="ignore", divide="ignore"):
                    chunk_mean = np.where(enough, chunk_energy / chunk_count, 0.0)
                row_energy[start:end] = chunk_mean

            search_surface[iy_idx, ix_indices] = row_energy
            row_best_idx = int(np.argmax(row_energy))
            if row_energy[row_best_idx] > best_energy:
                best_energy = row_energy[row_best_idx]
                best_ux = float(ux_batch[row_best_idx])
                best_uy = float(uy_cand)

        logger.debug(
            "Grid search best: Ux=%.2f, Uy=%.2f, energy=%.2e",
            best_ux,
            best_uy,
            best_energy,
        )

        return best_ux, best_uy, search_surface

    def _refine(self, coarse_ux: float, coarse_uy: float) -> tuple[float, float]:
        """Refine the coarse estimate by weighted least squares on peak frequencies.

        The coarse energy-maximization is limited by the omega bin width:
        one bin corresponds to delta_U = d_omega / k, which for typical
        record lengths (~1 minute) and wave numbers is 0.5-1 m/s. Following
        Young et al. (1985) and Senet et al. (2001), this method instead

        1. predicts, for each valid (kx, ky) column, where the positive
           dispersion branch ``omega = omega_0 - k . U`` should lie,
        2. locates the actual spectral peak within a velocity-scaled window
           of the prediction and interpolates its coordinates to sub-bin
           accuracy in omega AND in (kx, ky) — without the latter, a wave
           whose true k falls between grid columns produces a frequency
           mismatch of up to c_g * d_k / 2,
        3. solves the weighted linear least-squares problem
           ``omega_0(k_obs) - omega_peak = kx*Ux + ky*Uy`` with
           inverse-variance weights and sigma-clipping in velocity units,

        iterating three times so the search windows re-center on the
        improved estimate. Falls back to the coarse estimate when too few
        usable peaks exist or the solution leaves the search disc.

        Sets ``self._ls_stats`` with keys ``ux_err``, ``uy_err``,
        ``n_points``, ``rms`` (residual rms, rad/s) when successful.

        Args:
            coarse_ux: Coarse eastward current estimate (m/s).
            coarse_uy: Coarse northward current estimate (m/s).

        Returns:
            Refined (ux, uy) tuple.
        """
        cols_f = self._se_cols_filtered  # (n_t, n_valid)
        kx_v = self._se_kx
        ky_v = self._se_ky
        om0_v = self._se_om0
        iy_v, ix_v = self._se_iy, self._se_ix
        n_t = self._se_n_t
        d_omega = self._se_d_omega
        omega_nyq = self._se_omega_nyq
        omega_min = self._se_omega_min

        P = self.power_spectrum
        n_ky = len(self.ky)
        n_kx = len(self.kx)
        d_k = float(self.kx[1] - self.kx[0]) if n_kx > 1 else 0.0
        # Per-column noise floor for sub-bin k interpolation across columns
        med2d = np.median(P, axis=0)  # (n_ky, n_kx)

        _MAX_SEARCH_BINS = 3
        _DU_WINDOW = 1.5  # m/s: peak search window in velocity units
        _MIN_POINTS = 6
        _CLIP_SIGMA = 2.5
        # Modeled per-point frequency standard deviation: sub-bin
        # interpolation accuracy in omega plus the k-quantization error
        # mapped through the group velocity (dominant at low k).
        _C_OMEGA = 0.3  # fraction of d_omega
        _C_K = 0.3  # fraction of d_k

        def _parabolic(qm: np.ndarray, q0: np.ndarray, qp: np.ndarray) -> np.ndarray:
            denom = qm - 2.0 * q0 + qp
            with np.errstate(invalid="ignore", divide="ignore"):
                delta = np.where(denom < 0, 0.5 * (qm - qp) / denom, 0.0)
            return np.clip(np.nan_to_num(delta), -0.5, 0.5)

        ux, uy = coarse_ux, coarse_uy
        stats: dict | None = None

        for _ in range(3):
            k_mag = np.sqrt(kx_v**2 + ky_v**2)
            omega_pred = om0_v - (kx_v * ux + ky_v * uy)

            # Peak search half-width in bins, scaled so the window spans
            # ~ +- _DU_WINDOW m/s of current at each wavenumber. This keeps
            # low-k columns from locking onto leakage peaks that would map
            # to huge velocity errors.
            half_w = np.clip(
                np.round(_DU_WINDOW * k_mag / d_omega).astype(np.intp),
                1,
                _MAX_SEARCH_BINS,
            )

            margin = (half_w + 1) * d_omega
            usable = (np.abs(omega_pred) < omega_nyq - margin) & (np.abs(omega_pred) >= omega_min)
            idx = np.where(usable)[0]
            if len(idx) < _MIN_POINTS:
                return coarse_ux, coarse_uy

            base = np.round(omega_pred[idx] / d_omega).astype(np.intp)
            offsets = np.arange(-_MAX_SEARCH_BINS, _MAX_SEARCH_BINS + 1)
            cand_rows = (base[:, np.newaxis] + offsets[np.newaxis, :]) % n_t
            powers = cols_f[cand_rows, idx[:, np.newaxis]].copy()  # (n_pts, n_off)
            powers[np.abs(offsets)[np.newaxis, :] > half_w[idx][:, np.newaxis]] = -1.0

            best = powers.argmax(axis=1)
            peak_row = base + offsets[best]
            p0 = powers[np.arange(len(idx)), best]

            pm = cols_f[(peak_row - 1) % n_t, idx]
            pp = cols_f[(peak_row + 1) % n_t, idx]

            # Keep genuine interior peaks with positive power
            good = (p0 > 0) & (p0 >= pm) & (p0 >= pp)
            if good.sum() < _MIN_POINTS:
                return coarse_ux, coarse_uy
            s = np.where(good)[0]

            # Sub-bin omega via parabolic interpolation
            delta_w = _parabolic(pm[s], p0[s], pp[s])
            omega_obs = (peak_row[s] + delta_w) * d_omega

            # Sub-bin (kx, ky) via parabolic interpolation across neighbor
            # columns at the peak's omega row. Without this, a wave whose
            # true k falls between grid columns is assigned the bin-center
            # k, and omega_0(k_bin) is wrong by up to c_g * d_k / 2.
            row0 = peak_row[s] % n_t
            iy_c = iy_v[idx][s]
            ix_c = ix_v[idx][s]

            def _fval(rows: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
                return np.maximum(P[rows, iy, ix] - med2d[iy, ix], 0.0)

            qxm = _fval(row0, iy_c, (ix_c - 1) % n_kx)
            qx0 = _fval(row0, iy_c, ix_c)
            qxp = _fval(row0, iy_c, (ix_c + 1) % n_kx)
            qym = _fval(row0, (iy_c - 1) % n_ky, ix_c)
            qyp = _fval(row0, (iy_c + 1) % n_ky, ix_c)

            kx_obs = kx_v[idx][s] + _parabolic(qxm, qx0, qxp) * d_k
            ky_obs = ky_v[idx][s] + _parabolic(qym, qx0, qyp) * d_k
            km_obs = np.sqrt(kx_obs**2 + ky_obs**2)
            om0_obs = dispersion_relation(km_obs, self._depth)

            y = om0_obs - omega_obs
            a = np.column_stack([kx_obs, ky_obs])

            # Inverse-variance weights: quality (peak power) over modeled
            # frequency variance. The c_g term downweights low-k columns,
            # whose k-quantization errors map to large velocity errors.
            c_g = 0.5 * np.sqrt(_G / np.maximum(km_obs, 1e-12))
            sigma2_omega = (_C_OMEGA * d_omega) ** 2 + (_C_K * c_g * d_k) ** 2
            w = p0[s] / sigma2_omega

            # Weighted LS with sigma-clipping in *velocity* units: a
            # frequency residual r maps to a velocity error r / |k|, so
            # clipping in velocity space removes low-k leakage points whose
            # frequency residuals look innocuous.
            keep = np.ones(len(y), dtype=bool)
            sol = np.array([ux, uy])
            for _clip in range(4):
                resid = y - a @ sol
                resid_vel = resid / np.maximum(km_obs, 1e-12)
                w_vel = w * km_obs**2
                sigma_u = np.sqrt(max(1e-12, np.average(resid_vel[keep] ** 2, weights=w_vel[keep])))
                keep = np.abs(resid_vel) <= _CLIP_SIGMA * sigma_u
                if keep.sum() < _MIN_POINTS:
                    return coarse_ux, coarse_uy
                aw = a[keep] * np.sqrt(w[keep])[:, np.newaxis]
                yw = y[keep] * np.sqrt(w[keep])
                sol, *_ = np.linalg.lstsq(aw, yw, rcond=None)

            ux, uy = float(sol[0]), float(sol[1])

            # Parameter uncertainty from the weighted normal equations:
            # Cov(beta) = sigma2 * (A^T W A)^-1 with sigma2 = sum(w r^2)/dof
            # (scale-invariant in the relative weights w).
            aw = a[keep] * np.sqrt(w[keep])[:, np.newaxis]
            resid = (y - a @ sol)[keep]
            n_pts = int(keep.sum())
            dof = max(1, n_pts - 2)
            sigma2 = float(np.sum(w[keep] * resid**2) / dof)
            try:
                cov = sigma2 * np.linalg.inv(aw.T @ aw)
                ux_err = float(np.sqrt(max(0.0, cov[0, 0])))
                uy_err = float(np.sqrt(max(0.0, cov[1, 1])))
                cov_uxuy = float(cov[0, 1])
            except np.linalg.LinAlgError:
                ux_err = uy_err = float("nan")
                cov_uxuy = float("nan")

            stats = {
                "ux_err": ux_err,
                "uy_err": uy_err,
                "cov_uxuy": cov_uxuy,
                "n_points": n_pts,
                "rms": float(np.sqrt(np.mean(resid**2))),
            }

        if not (np.isfinite(ux) and np.isfinite(uy)):
            return coarse_ux, coarse_uy
        if np.sqrt(ux**2 + uy**2) > self._search_radius:
            logger.debug(
                "LS refinement left the search disc (%.2f, %.2f); using coarse estimate",
                ux,
                uy,
            )
            return coarse_ux, coarse_uy

        self._ls_stats = stats
        logger.debug(
            "LS refinement: (%.3f, %.3f) -> (%.3f, %.3f), n=%s",
            coarse_ux,
            coarse_uy,
            ux,
            uy,
            stats["n_points"] if stats else "?",
        )
        return ux, uy

    def _compute_snr(self, ux: float, uy: float) -> float:
        """Compute signal-to-noise ratio for the estimated current.

        Uses a per-bin normalized ratio so the SNR is independent of
        the number of spectral bins::

            SNR = (shell_energy / N_shell) / (noise_energy / N_noise)

        where ``N_shell`` is the number of bins sampled on the
        dispersion shell and ``N_noise = N_analysis - N_shell``.

        Both shell and noise are restricted to the analysis set — the
        valid wavenumber annulus ``k_min < |k| <= k_max`` with
        ``omega_min <= |omega| < omega_nyq`` — so the SNR compares the
        shell against bins the shell could actually occupy. Including
        the (enormous) static-pattern energy near omega = 0 or k bins
        outside the annulus would deflate the SNR by an arbitrary,
        scene-dependent factor and make thresholds meaningless.

        This gives SNR ~ 1 for pure noise and SNR >> 1 when energy
        is concentrated on the dispersion shell.

        Args:
            ux: Eastward current (m/s).
            uy: Northward current (m/s).

        Returns:
            Signal-to-noise ratio (>= 0). Returns 0 if total energy is zero.
        """
        shell_energy, n_shell = self._shell_energy(ux, uy)
        total_energy = self._se_analysis_energy
        n_total = self._se_n_analysis

        if total_energy <= 0 or n_shell <= 0:
            return 0.0

        noise_energy = total_energy - shell_energy
        n_noise = n_total - n_shell

        if noise_energy <= 0 or n_noise <= 0:
            return float("inf")

        return (shell_energy / n_shell) / (noise_energy / n_noise)
