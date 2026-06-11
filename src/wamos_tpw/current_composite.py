#! /usr/bin/env python3
#
# Temporal compositing of surface current maps
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Temporal compositing of surface current maps.

Surface currents evolve over tens of minutes, while the wave field that
samples them decorrelates block to block. Averaging the current vectors
from successive analysis blocks therefore increases statistical power
without paying the spatial-smearing penalty of longer analysis records
(a wave group traverses ``c_g * T`` during a block, so longer blocks blur
the current field spatially; more blocks do not).

This module composites a sequence of :class:`~wamos_tpw.current.CurrentMap`
objects onto a fixed earth-referenced lattice using inverse-variance
weighting of the per-tile least-squares uncertainties:

    u_hat = sum(w_i u_i) / sum(w_i),   w_i = 1 / (err_i^2 + floor^2)
    err_hat = sqrt(1 / sum(w_i))
    chi2_red = sum(w_i (u_i - u_hat)^2) / (n - 1)

``chi2_red >> 1`` in a cell means the per-block vectors disagree beyond
their formal errors — either the current changed during the composite
window or the formal errors are optimistic there.

Estimates without finite uncertainties (least-squares refinement fell
back to the coarse search) carry no weight and are dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from wamos_tpw.current import CurrentMap
from wamos_tpw.grid import _DEG2M, quantize_anchor, snap_origin

logger = logging.getLogger(__name__)

__all__ = [
    "CompositeMap",
    "composite_current_maps",
    "write_composite_netcdf",
]

# Added in quadrature to the formal errors so a single optimistic tile
# cannot dominate a cell (m/s)
_ERR_FLOOR_DEFAULT = 0.02
_MIN_OBS_DEFAULT = 2


@dataclass
class CompositeMap:
    """Inverse-variance composite of several current maps.

    Attributes:
        ux: 2D array of composited eastward current (m/s).
        uy: 2D array of composited northward current (m/s).
        ux_err: 2D array of posterior 1-sigma errors of ux (m/s).
        uy_err: 2D array of posterior 1-sigma errors of uy (m/s).
        n_obs: 2D array of the number of estimates per cell.
        chi2: 2D array of reduced chi-square consistency per cell
              (NaN where n_obs < 2).
        x_centers: Cell center x in meters east of (ref_lat, ref_lon).
        y_centers: Cell center y in meters north of (ref_lat, ref_lon).
        grid_spacing: Composite cell size (meters).
        ref_lat: Latitude of the equirectangular reference.
        ref_lon: Longitude of the equirectangular reference.
        start_time: Earliest block start among the inputs.
        end_time: Latest block end among the inputs.
        n_maps: Number of input maps.
    """

    ux: np.ndarray
    uy: np.ndarray
    ux_err: np.ndarray
    uy_err: np.ndarray
    n_obs: np.ndarray
    chi2: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    grid_spacing: float
    ref_lat: float
    ref_lon: float
    start_time: np.datetime64 = field(default_factory=lambda: np.datetime64("NaT"))
    end_time: np.datetime64 = field(default_factory=lambda: np.datetime64("NaT"))
    n_maps: int = 0


def _collect_observations(
    maps: list[CurrentMap],
    ref_lat: float,
    ref_lon: float,
    min_snr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather (x, y, ux, uy, ux_err, uy_err) from all maps in absolute meters."""
    m_per_deg = _DEG2M * np.cos(np.deg2rad(ref_lat))

    xs, ys, uxs, uys, exs, eys = [], [], [], [], [], []
    for cm in maps:
        map_x = (cm.center_lon - ref_lon) * m_per_deg
        map_y = (cm.center_lat - ref_lat) * _DEG2M
        for est in cm.estimates:
            if est.snr < min_snr:
                continue
            if not (
                np.isfinite(est.ux)
                and np.isfinite(est.uy)
                and np.isfinite(est.ux_err)
                and np.isfinite(est.uy_err)
            ):
                continue
            xs.append(map_x + est.center_x)
            ys.append(map_y + est.center_y)
            uxs.append(est.ux)
            uys.append(est.uy)
            exs.append(est.ux_err)
            eys.append(est.uy_err)

    return (
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(uxs),
        np.asarray(uys),
        np.asarray(exs),
        np.asarray(eys),
    )


def composite_current_maps(
    maps: list[CurrentMap],
    grid_spacing: float | None = None,
    min_obs: int = _MIN_OBS_DEFAULT,
    min_snr: float = 0.0,
    err_floor: float = _ERR_FLOOR_DEFAULT,
) -> CompositeMap | None:
    """Composite successive current maps onto a fixed earth lattice.

    Args:
        maps: CurrentMaps from successive analysis blocks.
        grid_spacing: Composite cell size in meters. Defaults to the
            median tile stride of the inputs.
        min_obs: Minimum estimates per cell; cells below are NaN.
        min_snr: Minimum per-estimate SNR to include.
        err_floor: Added in quadrature to formal errors (m/s).

    Returns:
        CompositeMap, or None when no usable estimates exist.
    """
    if not maps:
        return None

    # Shared reference: quantized anchor of the mean map center
    ref_lat, ref_lon = quantize_anchor(
        float(np.mean([cm.center_lat for cm in maps])),
        float(np.mean([cm.center_lon for cm in maps])),
    )

    x, y, ux, uy, ux_err, uy_err = _collect_observations(maps, ref_lat, ref_lon, min_snr)
    if len(x) == 0:
        logger.warning("No usable estimates to composite (of %d maps)", len(maps))
        return None

    if grid_spacing is None:
        strides = []
        for cm in maps:
            if len(cm.tile_x_centers) > 1:
                strides.append(float(np.median(np.diff(np.sort(cm.tile_x_centers)))))
            if len(cm.tile_y_centers) > 1:
                strides.append(float(np.median(np.diff(np.sort(cm.tile_y_centers)))))
        grid_spacing = float(np.median(strides)) if strides else 1000.0

    x0 = snap_origin(float(x.min()), grid_spacing)
    y0 = snap_origin(float(y.min()), grid_spacing)
    n_x = int(np.floor((float(x.max()) - x0) / grid_spacing)) + 1
    n_y = int(np.floor((float(y.max()) - y0) / grid_spacing)) + 1

    ix = np.floor((x - x0) / grid_spacing).astype(np.intp)
    iy = np.floor((y - y0) / grid_spacing).astype(np.intp)
    cell = iy * n_x + ix
    n_cells = n_y * n_x

    wx = 1.0 / (ux_err**2 + err_floor**2)
    wy = 1.0 / (uy_err**2 + err_floor**2)

    sum_wx = np.bincount(cell, weights=wx, minlength=n_cells)
    sum_wy = np.bincount(cell, weights=wy, minlength=n_cells)
    sum_wxu = np.bincount(cell, weights=wx * ux, minlength=n_cells)
    sum_wyu = np.bincount(cell, weights=wy * uy, minlength=n_cells)
    counts = np.bincount(cell, minlength=n_cells)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ux = sum_wxu / sum_wx
        mean_uy = sum_wyu / sum_wy
        err_ux = np.sqrt(1.0 / sum_wx)
        err_uy = np.sqrt(1.0 / sum_wy)

    # Reduced chi-square consistency across blocks (both components pooled)
    resid2 = wx * (ux - mean_ux[cell]) ** 2 + wy * (uy - mean_uy[cell]) ** 2
    sum_resid2 = np.bincount(cell, weights=resid2, minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2 = sum_resid2 / (2.0 * (counts - 1))
    chi2[counts < 2] = np.nan

    bad = counts < min_obs
    for arr in (mean_ux, mean_uy, err_ux, err_uy):
        arr[bad] = np.nan

    start = min(cm.start_time for cm in maps)
    end = max(cm.end_time for cm in maps)

    logger.info(
        "Composited %d maps, %d estimates -> %dx%d cells (%d populated)",
        len(maps),
        len(x),
        n_y,
        n_x,
        int(np.sum(counts >= min_obs)),
    )

    return CompositeMap(
        ux=mean_ux.reshape(n_y, n_x),
        uy=mean_uy.reshape(n_y, n_x),
        ux_err=err_ux.reshape(n_y, n_x),
        uy_err=err_uy.reshape(n_y, n_x),
        n_obs=counts.reshape(n_y, n_x),
        chi2=chi2.reshape(n_y, n_x),
        x_centers=x0 + (np.arange(n_x) + 0.5) * grid_spacing,
        y_centers=y0 + (np.arange(n_y) + 0.5) * grid_spacing,
        grid_spacing=grid_spacing,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        start_time=start,
        end_time=end,
        n_maps=len(maps),
    )


def write_composite_netcdf(comp: CompositeMap, output_dir: str) -> str:
    """Write a CompositeMap to a CF-1.8 NetCDF file.

    Args:
        comp: CompositeMap to write.
        output_dir: Output directory.

    Returns:
        Path to created file, or "" when xarray is unavailable.
    """
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not installed, skipping composite NetCDF output")
        return ""

    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_str = np.datetime_as_string(comp.start_time, unit="s").replace(":", "-").replace("T", "_")
    end_str = np.datetime_as_string(comp.end_time, unit="s").replace(":", "-").replace("T", "_")
    filepath = out / f"current_composite_{start_str}_to_{end_str}.nc"

    m_per_deg = _DEG2M * np.cos(np.deg2rad(comp.ref_lat))
    lats = comp.ref_lat + comp.y_centers / _DEG2M
    lons = comp.ref_lon + comp.x_centers / m_per_deg

    err_comment = (
        "Posterior error of the inverse-variance mean; compare with chi2 "
        "to judge whether block-to-block scatter exceeds the formal errors"
    )

    ds = xr.Dataset(
        data_vars={
            "ux": (
                ["y", "x"],
                comp.ux.astype(np.float32),
                {
                    "long_name": "Eastward sea water velocity (composite)",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "uy": (
                ["y", "x"],
                comp.uy.astype(np.float32),
                {
                    "long_name": "Northward sea water velocity (composite)",
                    "standard_name": "northward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "ux_err": (
                ["y", "x"],
                comp.ux_err.astype(np.float32),
                {
                    "long_name": "1-sigma uncertainty of composite ux",
                    "units": "m s-1",
                    "comment": err_comment,
                },
            ),
            "uy_err": (
                ["y", "x"],
                comp.uy_err.astype(np.float32),
                {
                    "long_name": "1-sigma uncertainty of composite uy",
                    "units": "m s-1",
                    "comment": err_comment,
                },
            ),
            "n_obs": (
                ["y", "x"],
                comp.n_obs.astype(np.int32),
                {"long_name": "Number of block estimates per cell", "units": "1"},
            ),
            "chi2": (
                ["y", "x"],
                comp.chi2.astype(np.float32),
                {
                    "long_name": "Reduced chi-square of block estimates",
                    "units": "1",
                },
            ),
        },
        coords={
            "x": (
                ["x"],
                comp.x_centers.astype(np.float32),
                {"long_name": "Meters east of reference", "units": "m", "axis": "X"},
            ),
            "y": (
                ["y"],
                comp.y_centers.astype(np.float32),
                {"long_name": "Meters north of reference", "units": "m", "axis": "Y"},
            ),
            "latitude": (["y"], lats, {"units": "degrees_north"}),
            "longitude": (["x"], lons, {"units": "degrees_east"}),
            "time_start": comp.start_time,
            "time_end": comp.end_time,
        },
        attrs={
            "title": "WAMOS composite surface current estimates",
            "institution": "WAMOS TPW",
            "source": "wamos current (temporal composite)",
            "history": f"Created {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            "reference_latitude": comp.ref_lat,
            "reference_longitude": comp.ref_lon,
            "n_input_maps": comp.n_maps,
        },
    )

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(filepath, encoding=encoding)

    logger.debug("Wrote composite current estimates to %s", filepath)
    return str(filepath)
