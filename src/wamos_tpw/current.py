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
              for each candidate, sum P on the Doppler-shifted dispersion shell
          -> (Ux, Uy) that maximizes shell energy -> refine with scipy.optimize
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
from scipy import optimize, signal

if TYPE_CHECKING:
    from wamos_tpw.config import Config
    from wamos_tpw.grid import GridParams

logger = logging.getLogger(__name__)

__all__ = [
    "CurrentEstimate",
    "CurrentExtractor",
    "CurrentMap",
    "FrameCube",
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
_FFT_WINDOW_DEFAULT = "hann"
_K_MIN_FACTOR_DEFAULT = 2.0


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
    """

    intensity: np.ndarray  # (n_t, n_y, n_x)
    timestamps: np.ndarray
    dt: float
    x_centers: np.ndarray
    y_centers: np.ndarray
    grid_spacing: float
    center_lat: float
    center_lon: float

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

        for i, frame_data in enumerate(frames):
            timestamps[i] = frame_data["timestamp"]

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
    """

    ux: float
    uy: float
    speed: float
    direction: float
    snr: float
    depth: float
    center_x: float = 0.0
    center_y: float = 0.0


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

        estimates: list[CurrentEstimate] = []

        for tile in specs["tiles"]:
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
            )
            estimates.append(est)

            snr_map[iy, ix] = est.snr
            if est.snr >= min_snr:
                ux_map[iy, ix] = est.ux
                uy_map[iy, ix] = est.uy
                speed_map[iy, ix] = est.speed
                dir_map[iy, ix] = est.direction

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
            )
            estimates.append(est)

            snr_map[iy, ix] = est.snr
            if est.snr >= min_snr:
                ux_map[iy, ix] = est.ux
                uy_map[iy, ix] = est.uy
                speed_map[iy, ix] = est.speed
                dir_map[iy, ix] = est.direction

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
        )


# ============================================================
# Tile Geometry
# ============================================================


def compute_tile_specs(cube: FrameCube, config: Config | None = None) -> dict:
    """Compute tile geometry for spatial current extraction.

    Divides the cube's spatial domain into overlapping sub-regions
    suitable for independent current extraction.

    Args:
        cube: 3D space-time cube of radar intensity.
        config: Configuration object for extraction parameters.

    Returns:
        Dict with keys:
        - ``n_tiles_x``, ``n_tiles_y``: Number of tiles in each direction.
        - ``tiles``: List of tile dicts with ``ix``, ``iy``, ``x_start``,
          ``x_end``, ``y_start``, ``y_end``, ``center_x``, ``center_y``.
        - ``tile_x_centers``, ``tile_y_centers``: 1D arrays of tile center
          positions (meters).
        - ``min_snr``: Minimum SNR threshold.
        - ``depth``: Water depth (meters).
    """
    from wamos_tpw.config import NullConfig

    cfg = config or NullConfig()

    sub_region_size = cfg.get("current.sub_region_size", _SUB_REGION_SIZE_DEFAULT)
    sub_region_overlap = cfg.get("current.sub_region_overlap", _SUB_REGION_OVERLAP_DEFAULT)
    min_snr = cfg.get("current.min_snr", _MIN_SNR_DEFAULT)
    depth = cfg.get("current.depth", _DEPTH_DEFAULT)

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
                }
            )

    return {
        "n_tiles_x": n_tiles_x,
        "n_tiles_y": n_tiles_y,
        "tiles": tiles,
        "tile_x_centers": tile_x_centers,
        "tile_y_centers": tile_y_centers,
        "min_snr": min_snr,
        "depth": depth,
    }


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
        self._window_name = self._cfg.get("current.fft_window", _FFT_WINDOW_DEFAULT)

        # Run the extraction pipeline
        prepared = self._prepare_cube()
        self.power_spectrum, self.kx, self.ky, self.omega = self._compute_3d_fft(prepared)
        coarse_ux, coarse_uy, self._search_surface = self._grid_search()

        if self._do_refine:
            ux, uy = self._refine(coarse_ux, coarse_uy)
        else:
            ux, uy = coarse_ux, coarse_uy

        speed = float(np.sqrt(ux**2 + uy**2))
        direction = float(np.rad2deg(np.arctan2(ux, uy)) % 360)  # TO convention, CW from N
        snr = self._compute_snr(ux, uy)

        self.estimate = CurrentEstimate(
            ux=ux,
            uy=uy,
            speed=speed,
            direction=direction,
            snr=snr,
            depth=self._depth,
            center_x=float(np.mean(cube.x_centers)),
            center_y=float(np.mean(cube.y_centers)),
        )

    def _prepare_cube(self) -> np.ndarray:
        """Prepare the intensity cube for FFT analysis.

        Steps:
        1. Replace NaN values with spatial mean per time step.
        2. Remove temporal mean (detrend).
        3. Apply 3D window function (e.g. Hann).

        Returns:
            Prepared 3D array ready for FFT.
        """
        data = self._cube.intensity.copy()
        n_t, n_y, n_x = data.shape

        # Step 1: Fill NaN with per-timestep spatial mean
        for t in range(n_t):
            frame = data[t]
            nan_mask = np.isnan(frame)
            if np.any(nan_mask):
                valid_mean = np.nanmean(frame)
                if np.isfinite(valid_mean):
                    frame[nan_mask] = valid_mean
                else:
                    frame[nan_mask] = 0.0

        # Step 2: Remove temporal mean
        temporal_mean = np.mean(data, axis=0)
        data -= temporal_mean

        # Step 3: Apply 3D window
        win_t = signal.windows.get_window(self._window_name, n_t, fftbins=False)
        win_y = signal.windows.get_window(self._window_name, n_y, fftbins=False)
        win_x = signal.windows.get_window(self._window_name, n_x, fftbins=False)
        # 3D separable window via outer products
        window_3d = (
            win_t[:, np.newaxis, np.newaxis]
            * win_y[np.newaxis, :, np.newaxis]
            * win_x[np.newaxis, np.newaxis, :]
        )
        data *= window_3d

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
        spectrum = scipy_fft.fftn(data)
        power = np.abs(spectrum) ** 2

        # Frequency axes (full, including negative)
        omega = 2.0 * np.pi * scipy_fft.fftfreq(n_t, d=dt)
        ky = 2.0 * np.pi * scipy_fft.fftfreq(n_y, d=dx)
        kx = 2.0 * np.pi * scipy_fft.fftfreq(n_x, d=dx)

        return power, kx, ky, omega

    def _shell_energy(self, ux: float, uy: float) -> tuple[float, int]:
        """Compute total energy on the Doppler-shifted dispersion shell.

        For each (kx, ky) bin, the expected observed frequency is:

        .. math::

            \\omega = \\pm \\omega_0(|k|, d) + k_x U_x + k_y U_y

        Energy is summed from the nearest omega bin for all valid
        wavenumber bins on both dispersion branches.

        Args:
            ux: Eastward current candidate (m/s).
            uy: Northward current candidate (m/s).

        Returns:
            Tuple of (total shell energy, number of shell bins sampled).
        """
        P = self.power_spectrum  # (n_t, n_y, n_x)
        kx = self.kx
        ky = self.ky
        depth = self._depth

        n_t, n_ky, n_kx = P.shape

        # Minimum wavenumber to avoid low-frequency contamination
        sub_region_size = max(
            self._cube.x_centers[-1] - self._cube.x_centers[0],
            self._cube.y_centers[-1] - self._cube.y_centers[0],
        )
        k_min = self._k_min_factor * 2.0 * np.pi / sub_region_size if sub_region_size > 0 else 0.0

        # Vectorized computation over all (ky, kx) pairs
        kx_2d, ky_2d = np.meshgrid(kx, ky)  # (n_ky, n_kx)
        k_mag = np.sqrt(kx_2d**2 + ky_2d**2)

        # Mask: skip DC and low wavenumbers
        valid = k_mag > k_min

        # Intrinsic frequency
        omega_0 = dispersion_relation(k_mag, depth)

        # Dispersion shell in the full 3D FFT.
        #
        # A physical wave cos(kx*x + ky*y - omega_obs*t) produces DFT peaks at:
        #   (omega_fft = -omega_obs, ky_fft = +ky, kx_fft = +kx)
        # and its conjugate. The Doppler-shifted observed frequency is:
        #   omega_obs = omega_0(|k|) + kx*Ux + ky*Uy
        #
        # So the DFT peak in the omega axis is at:
        #   omega_fft = -(omega_0(|k|) + kx_fft*Ux + ky_fft*Uy)
        #
        # For the negative dispersion branch (backward-propagating waves):
        #   omega_fft = -(-omega_0(|k|) + kx_fft*Ux + ky_fft*Uy)
        #             =   omega_0(|k|) - kx_fft*Ux - ky_fft*Uy
        doppler = kx_2d * ux + ky_2d * uy
        omega_pred_pos = -(omega_0 + doppler)  # positive dispersion branch
        omega_pred_neg = omega_0 - doppler  # negative dispersion branch

        d_omega = 2.0 * np.pi / (n_t * self._cube.dt)
        omega_nyquist = np.pi / self._cube.dt

        energy = 0.0
        n_shell_bins = 0

        for omega_pred in [omega_pred_pos, omega_pred_neg]:
            # Linearly interpolate between the two nearest FFT bins
            # for sub-bin precision.
            frac_idx = omega_pred / d_omega
            idx_lo = np.floor(frac_idx).astype(int)
            weight_hi = frac_idx - idx_lo  # fractional part
            weight_lo = 1.0 - weight_hi

            idx_lo_wrapped = idx_lo % n_t
            idx_hi_wrapped = (idx_lo + 1) % n_t

            ok = valid & (np.abs(omega_pred) < omega_nyquist)

            iy_arr, ix_arr = np.where(ok)
            lo_idx = idx_lo_wrapped[ok]
            hi_idx = idx_hi_wrapped[ok]
            w_lo = weight_lo[ok]
            w_hi = weight_hi[ok]

            energy += float(
                np.sum(w_lo * P[lo_idx, iy_arr, ix_arr] + w_hi * P[hi_idx, iy_arr, ix_arr])
            )
            n_shell_bins += int(ok.sum())

        return energy, n_shell_bins

    def _grid_search(self) -> tuple[float, float, np.ndarray]:
        """Brute-force grid search over (Ux, Uy) candidates.

        Tests all combinations on a regular grid within the search radius
        and returns the candidate with maximum shell energy.

        Returns:
            Tuple of (best_ux, best_uy, search_surface).
            search_surface is a 2D array of energies indexed by (uy, ux) grid.
        """
        radius = self._search_radius
        step = self._search_step

        # Create candidate grid
        candidates = np.arange(-radius, radius + step / 2, step)
        n = len(candidates)

        search_surface = np.zeros((n, n))
        best_energy = -1.0
        best_ux = 0.0
        best_uy = 0.0

        for iy, uy_cand in enumerate(candidates):
            for ix, ux_cand in enumerate(candidates):
                # Skip candidates outside the circular search region
                if ux_cand**2 + uy_cand**2 > radius**2:
                    continue
                energy, _ = self._shell_energy(ux_cand, uy_cand)
                search_surface[iy, ix] = energy
                if energy > best_energy:
                    best_energy = energy
                    best_ux = float(ux_cand)
                    best_uy = float(uy_cand)

        logger.debug(
            "Grid search best: Ux=%.2f, Uy=%.2f, energy=%.2e",
            best_ux,
            best_uy,
            best_energy,
        )

        return best_ux, best_uy, search_surface

    def _refine(self, coarse_ux: float, coarse_uy: float) -> tuple[float, float]:
        """Refine the coarse grid search result using Nelder-Mead optimization.

        Args:
            coarse_ux: Coarse eastward current estimate (m/s).
            coarse_uy: Coarse northward current estimate (m/s).

        Returns:
            Refined (ux, uy) tuple.
        """

        def neg_energy(uxy: np.ndarray) -> float:
            energy, _ = self._shell_energy(float(uxy[0]), float(uxy[1]))
            return -energy

        result = optimize.minimize(
            neg_energy,
            x0=np.array([coarse_ux, coarse_uy]),
            method="Nelder-Mead",
            options={
                "xatol": self._search_step / 10.0,
                "fatol": 1e-6,
                "maxiter": 200,
            },
        )

        refined_ux, refined_uy = float(result.x[0]), float(result.x[1])

        # Reject refinement if it moved more than one coarse step from the
        # grid search best — large jumps indicate chasing noise, not signal.
        max_shift = self._search_step * 1.5
        shift = np.sqrt((refined_ux - coarse_ux) ** 2 + (refined_uy - coarse_uy) ** 2)

        if not result.success or shift > max_shift:
            if not result.success:
                logger.debug("Refinement did not converge, using coarse estimate")
            else:
                logger.debug(
                    "Refinement jumped too far (%.2f > %.2f), using coarse estimate",
                    shift,
                    max_shift,
                )
            return coarse_ux, coarse_uy

        logger.debug(
            "Refinement: (%.3f, %.3f) -> (%.3f, %.3f)",
            coarse_ux,
            coarse_uy,
            refined_ux,
            refined_uy,
        )
        return refined_ux, refined_uy

    def _compute_snr(self, ux: float, uy: float) -> float:
        """Compute signal-to-noise ratio for the estimated current.

        Uses a per-bin normalized ratio so the SNR is independent of
        the number of spectral bins::

            SNR = (shell_energy / N_shell) / (noise_energy / N_noise)

        where ``N_shell`` is the number of bins sampled on the
        dispersion shell and ``N_noise = N_total - N_shell``.

        This gives SNR ~ 1 for pure noise and SNR >> 1 when energy
        is concentrated on the dispersion shell.

        Args:
            ux: Eastward current (m/s).
            uy: Northward current (m/s).

        Returns:
            Signal-to-noise ratio (>= 0). Returns 0 if total energy is zero.
        """
        shell_energy, n_shell = self._shell_energy(ux, uy)
        total_energy = float(np.sum(self.power_spectrum))
        n_total = self.power_spectrum.size

        if total_energy <= 0 or n_shell <= 0:
            return 0.0

        noise_energy = total_energy - shell_energy
        n_noise = n_total - n_shell

        if noise_energy <= 0 or n_noise <= 0:
            return float("inf")

        return (shell_energy / n_shell) / (noise_energy / n_noise)
