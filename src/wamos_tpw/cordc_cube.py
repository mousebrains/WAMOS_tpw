#! /usr/bin/env python3
#
# Build FrameCubes from CORDC fixed-station radar sweeps (CSS Angaur)
#
# Jul-2026, Pat Welch, pat@mousebrains.com

"""
Earth-gridded space-time cubes from CORDC fixed-station sweeps.

Resamples each :class:`~wamos_tpw.cordc.CordcFile` sweep from polar
(encoder bearing, range) onto a regular east/north grid centered on a
chosen point, producing the :class:`~wamos_tpw.current.FrameCube` the
surface-current machinery consumes. The tower is fixed, so there is no
ship-motion compensation — bearings come straight from the calibrated
encoder words and ranges from the calibrated cell size.

Sampling caveat: the recorder saved every OTHER antenna rotation
(~5.02 s cadence -> 10 s Nyquist). The wind sea is aliased; the swell
band (>= ~11 s) is usable, which is what the dispersion fit keys on.

Usage::

    from wamos_tpw.cordc import FixedStation
    from wamos_tpw.cordc_cube import build_cube

    station = FixedStation.from_yaml("css_angaur_2023")
    cube = build_cube(files, station, center_lat=6.93017,
                      center_lon=134.19942, half_size=1500.0)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wamos_tpw.cordc import ENCODER_COUNTS, CordcFile, FixedStation
from wamos_tpw.prefetch import read_ahead

logger = logging.getLogger(__name__)

__all__ = ["build_cube"]

_DEG2M = 111_319.5  # meters per degree latitude; matches current.py


def build_cube(
    files: list[str | Path],
    station: FixedStation,
    center_lat: float,
    center_lon: float,
    half_size: float = 1500.0,
    grid_spacing: float = 20.0,
):
    """Resample CORDC sweeps onto an earth grid as a FrameCube.

    Args:
        files: Sweep files, in time order.
        station: Calibrated fixed-station geometry.
        center_lat: Grid center latitude (degrees).
        center_lon: Grid center longitude (degrees).
        half_size: Half-width of the square grid (meters).
        grid_spacing: Grid cell size (meters).

    Returns:
        FrameCube with intensity ``(n_files, ny, nx)`` (float32, NaN
        where a cell falls outside the radar window).
    """
    from wamos_tpw.current import FrameCube

    n = int(round(2 * half_size / grid_spacing))
    x = (np.arange(n) + 0.5) * grid_spacing - half_size  # east of center
    y = (np.arange(n) + 0.5) * grid_spacing - half_size  # north of center
    gx, gy = np.meshgrid(x, y)

    # tower -> grid-cell geometry (equirectangular around the center)
    tow_x = (station.longitude - center_lon) * _DEG2M * np.cos(np.radians(center_lat))
    tow_y = (station.latitude - center_lat) * _DEG2M
    dx = gx - tow_x
    dy = gy - tow_y
    cell_rng = np.hypot(dx, dy)
    cell_brg = np.degrees(np.arctan2(dx, dy)) % 360.0
    # bearing -> encoder word (float) under the station's encoder zero
    cell_word = ((cell_brg - station.theta0_deg) % 360.0) / 360.0 * ENCODER_COUNTS

    intensity = np.full((len(files), n, n), np.nan, dtype=np.float32)
    times = np.empty(len(files), dtype="datetime64[ms]")
    flat_word = cell_word.ravel()
    # warm the page cache ahead of the parse loop; cold SMB reads
    # otherwise serialize with the per-sweep resampling below
    for k, fn in enumerate(read_ahead(files)):
        sweep = CordcFile(fn, station)
        times[k] = np.datetime64(sweep.file_time.replace(tzinfo=None), "ms")
        rng_idx = (cell_rng - sweep.ranges[0]) / sweep.cell_m
        ri = np.rint(rng_idx).astype(np.intp)
        n_cells = sweep.intensity.shape[1]
        in_rng = (ri >= 0) & (ri < n_cells)
        # nearest radial by encoder word (words are monotonic per sweep)
        wi = np.searchsorted(sweep.words, flat_word).reshape(cell_word.shape)
        wi = np.clip(wi, 0, len(sweep.words) - 1)
        # choose the closer of the two neighboring radials (mod 8192)
        lo = np.clip(wi - 1, 0, len(sweep.words) - 1)
        d_hi = np.abs(sweep.words[wi] - cell_word)
        d_lo = np.abs(sweep.words[lo] - cell_word)
        wi = np.where(d_lo < d_hi, lo, wi)
        frame = np.full((n, n), np.nan, dtype=np.float32)
        frame[in_rng] = sweep.intensity[wi[in_rng], ri[in_rng]].astype(np.float32)
        intensity[k] = frame

    dt = float(np.median(np.diff(times).astype("timedelta64[ms]").astype(float))) / 1000.0
    logger.info(
        "CORDC cube: %d sweeps, %dx%d cells @ %.0f m, dt=%.2f s (Nyquist %.1f s)",
        len(files),
        n,
        n,
        grid_spacing,
        dt,
        2 * dt,
    )
    return FrameCube(
        intensity=intensity,
        timestamps=times,
        dt=dt,
        x_centers=x,
        y_centers=y,
        grid_spacing=grid_spacing,
        center_lat=center_lat,
        center_lon=center_lon,
    )
