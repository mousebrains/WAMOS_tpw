#! /usr/bin/env python3
#
# Joint regularized inversion of windowed current estimates onto a fine grid
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Joint regularized inversion of windowed current estimates.

Per-window dispersion fits assume the current is homogeneous over the
analysis window, and tiling the domain into independent windows ties the
output resolution to the window size. This module decouples the two:
every window estimate (possibly from several window sizes) is treated as
an observation of the *footprint average* of an unknown current field
U(x, y) on a fine grid, and the field is recovered by minimizing

    J(u) = sum_w (A_w u - d_w)^T C_w^-1 (A_w u - d_w)
         + sum_edges ((u_i - u_j) * L_c / (sigma_prior * h))^2

where ``A_w`` is a Hann^2-weighted average over window w's footprint —
the kernel the dispersion estimator actually applies: the extractor uses
a spatial Hann window and works with the power spectrum, so its
effective sensing footprint is concentrated toward the window center
(empirically a 2 km Hann window recovers a 2 km sinusoidal current at
~0.9 amplitude, where a boxcar kernel would predict zero). ``d_w`` is
the window's (ux, uy) with its 2x2 least-squares covariance ``C_w``, and
the second term is a first-difference smoothness prior: gradients of
order ``sigma_prior / L_c`` cost about one unit per cell edge,
comparable to a 1-sigma data misfit.

Because small windows resolve only short waves and large windows resolve
long waves, mixing window sizes lets short waves carry high-resolution
information where they have it while large windows stabilize the
large-scale field.

The normal equations are sparse and solved directly; posterior 1-sigma
errors come from the diagonal of the inverse Hessian (computed when the
problem is small enough, NaN otherwise).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from wamos_tpw.current import CurrentEstimate

logger = logging.getLogger(__name__)

__all__ = [
    "CurrentField",
    "solve_current_field",
    "write_field_netcdf",
]

_FIELD_SPACING_DEFAULT = 250.0  # meters
_CORRELATION_LENGTH_DEFAULT = 1000.0  # meters
_SIGMA_PRIOR_DEFAULT = 0.3  # m/s of expected variation over correlation length
_ERR_FLOOR_DEFAULT = 0.02  # m/s, added in quadrature to observation errors
# Above this many unknowns the dense posterior-variance computation is skipped
_MAX_POSTERIOR_UNKNOWNS = 20_000
# Robust reweighting: observations with squared Mahalanobis residual above
# this (chi-square 2-dof ~99th percentile) are scaled down proportionally
_IRLS_M2_CUTOFF = 9.0
_IRLS_ITERATIONS = 3


@dataclass
class CurrentField:
    """Current field from joint inversion of windowed estimates.

    Attributes:
        ux: 2D array (n_y, n_x) of eastward current (m/s).
        uy: 2D array of northward current (m/s).
        ux_err: 2D array of posterior 1-sigma errors (m/s); NaN when the
                problem was too large for the dense computation.
        uy_err: 2D array of posterior 1-sigma errors (m/s).
        coverage: 2D int array, number of observation footprints covering
                each cell (0 = pure prior extrapolation; treat with care).
        x_centers: Cell centers (m), same coordinate frame as the inputs.
        y_centers: Cell centers (m).
        grid_spacing: Cell size (m).
        center_lat: Latitude of the coordinate origin.
        center_lon: Longitude of the coordinate origin.
        n_obs: Number of window estimates used.
        chi2: Reduced data misfit sum_w r^T C^-1 r / (2 n_obs).
        correlation_length: Smoothness prior length scale used (m).
        sigma_prior: Smoothness prior amplitude used (m/s).
        start_time: Start of the analysis block.
        end_time: End of the analysis block.
    """

    ux: np.ndarray
    uy: np.ndarray
    ux_err: np.ndarray
    uy_err: np.ndarray
    coverage: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    grid_spacing: float
    center_lat: float
    center_lon: float
    n_obs: int
    chi2: float
    correlation_length: float
    sigma_prior: float
    start_time: np.datetime64 = field(default_factory=lambda: np.datetime64("NaT"))
    end_time: np.datetime64 = field(default_factory=lambda: np.datetime64("NaT"))


def _usable_estimates(
    estimates: list[CurrentEstimate],
    min_snr: float,
) -> list[CurrentEstimate]:
    """Filter to estimates with finite values, errors, and a footprint."""
    out = []
    for est in estimates:
        if est.snr < min_snr:
            continue
        if est.window_size <= 0:
            continue
        if not (
            np.isfinite(est.ux)
            and np.isfinite(est.uy)
            and np.isfinite(est.ux_err)
            and np.isfinite(est.uy_err)
        ):
            continue
        out.append(est)
    return out


def solve_current_field(
    estimates: list[CurrentEstimate],
    field_spacing: float = _FIELD_SPACING_DEFAULT,
    correlation_length: float = _CORRELATION_LENGTH_DEFAULT,
    sigma_prior: float = _SIGMA_PRIOR_DEFAULT,
    min_snr: float = 0.0,
    err_floor: float = _ERR_FLOOR_DEFAULT,
    center_lat: float = 0.0,
    center_lon: float = 0.0,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
) -> CurrentField | None:
    """Solve for U(x, y) on a fine grid from windowed current estimates.

    Args:
        estimates: Window estimates (e.g. ``CurrentMap.estimates``),
            possibly from several window sizes. Each must carry
            ``center_x/center_y`` (m), ``window_size`` (m), and finite
            least-squares errors; others are dropped.
        field_spacing: Output cell size in meters.
        correlation_length: Smoothness prior length scale L_c (m).
        sigma_prior: Expected current variation over L_c (m/s).
        min_snr: Minimum estimate SNR to include.
        err_floor: Added in quadrature to observation errors (m/s).
        center_lat: Latitude of the coordinate origin (metadata).
        center_lon: Longitude of the coordinate origin (metadata).
        start_time: Analysis block start (metadata).
        end_time: Analysis block end (metadata).

    Returns:
        CurrentField, or None when no usable estimates exist.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    obs = _usable_estimates(estimates, min_snr)
    if not obs:
        logger.warning("No usable estimates for field inversion")
        return None

    # ---- Field grid over the union of observation footprints ----
    x_lo = min(e.center_x - e.window_size / 2 for e in obs)
    x_hi = max(e.center_x + e.window_size / 2 for e in obs)
    y_lo = min(e.center_y - e.window_size / 2 for e in obs)
    y_hi = max(e.center_y + e.window_size / 2 for e in obs)

    h = field_spacing
    n_x = max(1, int(np.ceil((x_hi - x_lo) / h)))
    n_y = max(1, int(np.ceil((y_hi - y_lo) / h)))
    x_centers = x_lo + (np.arange(n_x) + 0.5) * h
    y_centers = y_lo + (np.arange(n_y) + 0.5) * h

    n_cells = n_y * n_x
    n_unknowns = 2 * n_cells  # [ux block; uy block]

    # ---- Per-observation footprints and 2x2 weights ----
    coverage = np.zeros(n_cells, dtype=np.int32)
    obs_cells: list[np.ndarray] = []
    obs_fpw: list[np.ndarray] = []  # footprint weights (sum to 1)
    obs_weights: list[tuple[float, float, float]] = []  # (w00, w01, w11)
    obs_data: list[tuple[float, float]] = []

    for est in obs:
        half = est.window_size / 2
        in_x = np.where(np.abs(x_centers - est.center_x) <= half)[0]
        in_y = np.where(np.abs(y_centers - est.center_y) <= half)[0]
        if len(in_x) == 0 or len(in_y) == 0:
            continue
        cells = (in_y[:, np.newaxis] * n_x + in_x[np.newaxis, :]).ravel()
        coverage[cells] += 1

        # Footprint weights: the window estimate is not a boxcar average of
        # the current. The extractor applies a spatial Hann window and works
        # with the POWER spectrum, so the local Doppler information is
        # weighted approximately by Hann^2 — heavily concentrated toward the
        # window center (equivalent width ~0.4 of the window). Validated
        # empirically: 2 km Hann windows recover a 2 km sinusoidal current
        # at ~0.9 amplitude where a boxcar model predicts ~0.
        wx_t = np.cos(np.pi * (x_centers[in_x] - est.center_x) / est.window_size) ** 4
        wy_t = np.cos(np.pi * (y_centers[in_y] - est.center_y) / est.window_size) ** 4
        w_fp = (wy_t[:, np.newaxis] * wx_t[np.newaxis, :]).ravel()
        w_sum = float(w_fp.sum())
        if w_sum <= 0:
            continue
        w_fp /= w_sum

        # 2x2 observation covariance -> weight matrix
        var_x = est.ux_err**2 + err_floor**2
        var_y = est.uy_err**2 + err_floor**2
        cov = est.cov_uxuy if np.isfinite(est.cov_uxuy) else 0.0
        det = var_x * var_y - cov**2
        if det <= 0:
            cov = 0.0
            det = var_x * var_y

        obs_cells.append(cells)
        obs_fpw.append(w_fp)
        obs_weights.append((var_y / det, -cov / det, var_x / det))
        obs_data.append((est.ux, est.uy))

    n_used = len(obs_cells)
    if n_used == 0:
        logger.warning("No observation footprint overlaps the field grid")
        return None

    # ---- Smoothness prior: first differences, both components ----
    w_s = (correlation_length / (sigma_prior * h)) ** 2

    d_rows: list[int] = []
    d_cols: list[int] = []
    d_vals: list[float] = []

    def _add_edge(i: int, j: int) -> None:
        d_rows.extend((i, j, i, j))
        d_cols.extend((i, j, j, i))
        d_vals.extend((w_s, w_s, -w_s, -w_s))

    for iy in range(n_y):
        for ix_ in range(n_x):
            cell = iy * n_x + ix_
            if ix_ + 1 < n_x:
                for off in (0, n_cells):
                    _add_edge(off + cell, off + cell + 1)
            if iy + 1 < n_y:
                for off in (0, n_cells):
                    _add_edge(off + cell, off + cell + n_x)

    h_smooth = sparse.coo_matrix((d_vals, (d_rows, d_cols)), shape=(n_unknowns, n_unknowns)).tocsr()

    def _assemble_and_solve(factors: np.ndarray):
        """Assemble the data term with per-obs robust factors and solve."""
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        rhs = np.zeros(n_unknowns)

        for cells, w_fp, (w00, w01, w11), (dx_, dy_), f in zip(
            obs_cells, obs_fpw, obs_weights, obs_data, factors, strict=True
        ):
            if f <= 0:
                continue

            # Normal-equation contribution: for the 2 x n row block R with
            # entries w_fp at [cells] (ux row) and [n_cells + cells]
            # (uy row), R^T W R adds (f * w_i * w_j * W_ab) over the outer
            # product of cells.
            ci, cj = np.meshgrid(cells, cells, indexing="ij")
            ci = ci.ravel()
            cj = cj.ravel()
            a2 = f * np.outer(w_fp, w_fp).ravel()

            rows.append(ci)
            cols.append(cj)
            vals.append(a2 * w00)
            rows.append(n_cells + ci)
            cols.append(n_cells + cj)
            vals.append(a2 * w11)
            rows.append(ci)
            cols.append(n_cells + cj)
            vals.append(a2 * w01)
            rows.append(n_cells + ci)
            cols.append(cj)
            vals.append(a2 * w01)

            rhs[cells] += f * w_fp * (w00 * dx_ + w01 * dy_)
            rhs[n_cells + cells] += f * w_fp * (w01 * dx_ + w11 * dy_)

        h_data = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_unknowns, n_unknowns),
        ).tocsr()

        # Tiny ridge for numerical safety (pins fully unconstrained problems)
        ridge = 1e-9 * (h_data.diagonal().max() + w_s)
        hessian = (h_data + h_smooth + ridge * sparse.identity(n_unknowns)).tocsc()
        lu = splu(hessian)
        return lu, lu.solve(rhs)

    def _mahalanobis2(solution: np.ndarray) -> np.ndarray:
        """Per-observation squared Mahalanobis residual (2 dof, expect ~2)."""
        m2 = np.empty(n_used)
        for i, (cells, w_fp, (w00, w01, w11), (dx_, dy_)) in enumerate(
            zip(obs_cells, obs_fpw, obs_weights, obs_data, strict=True)
        ):
            rx = dx_ - float(np.dot(w_fp, solution[cells]))
            ry = dy_ - float(np.dot(w_fp, solution[n_cells + cells]))
            m2[i] = w00 * rx * rx + 2 * w01 * rx * ry + w11 * ry * ry
        return m2

    # ---- Robust (IRLS) solve: down-weight gross outlier observations.
    #
    # Residuals are first normalized by a robust global scale (the bulk
    # median of m2 against the chi-square 2-dof median, 1.386) so that a
    # uniformly optimistic error calibration — common, since the window
    # least-squares errors ignore correlated spectral leakage — does not
    # mark the whole population as inconsistent. Only observations that
    # are outliers RELATIVE TO THE BULK (scaled m2 above the chi-square
    # 2-dof ~99th percentile) are scaled down: typically a window whose
    # fit locked onto the wrong spectral feature.
    #
    # Note: per-window-size variance-component recalibration was tried
    # and rejected — with few large windows and many small ones, the
    # consensus field is dominated by the small windows, and residuals
    # against a contaminated consensus punish the most accurate group.
    _CHI2_2DOF_MEDIAN = 1.386
    factors = np.ones(n_used)
    try:
        lu, solution = _assemble_and_solve(factors)
        for _ in range(_IRLS_ITERATIONS):
            m2 = _mahalanobis2(solution)
            bulk_scale = max(1.0, float(np.median(m2)) / _CHI2_2DOF_MEDIAN)
            m2_scaled = m2 / bulk_scale
            new_factors = np.minimum(1.0, _IRLS_M2_CUTOFF / np.maximum(m2_scaled, 1e-12))
            if np.allclose(new_factors, factors, rtol=0.05, atol=0.01):
                factors = new_factors
                break
            factors = new_factors
            lu, solution = _assemble_and_solve(factors)
    except RuntimeError:
        logger.exception("Field inversion factorization failed")
        return None

    ux_field = solution[:n_cells].reshape(n_y, n_x)
    uy_field = solution[n_cells:].reshape(n_y, n_x)

    # ---- Data misfit with the final robust weights ----
    m2 = _mahalanobis2(solution)
    sum_f = float(np.sum(factors))
    chi2 = float(np.sum(factors * m2) / (2 * sum_f)) if sum_f > 0 else float("nan")
    n_down = int(np.sum(factors < 0.999))
    if n_down:
        logger.info("Down-weighted %d/%d inconsistent observations", n_down, n_used)

    # ---- Posterior errors (dense path under a size cap) ----
    # Scaled by sqrt(max(1, chi2)): when observations scatter beyond their
    # formal errors even after robust weighting, the posterior inherits it.
    ux_err = np.full(n_cells, np.nan)
    uy_err = np.full(n_cells, np.nan)
    if n_unknowns <= _MAX_POSTERIOR_UNKNOWNS:
        err_scale = float(np.sqrt(max(1.0, chi2)))
        block = 512
        diag = np.empty(n_unknowns)
        eye = np.eye(n_unknowns, dtype=np.float64)
        for start in range(0, n_unknowns, block):
            end = min(start + block, n_unknowns)
            inv_block = lu.solve(eye[:, start:end])
            diag[start:end] = inv_block[np.arange(start, end), np.arange(end - start)]
        diag = np.maximum(diag, 0.0)
        ux_err = err_scale * np.sqrt(diag[:n_cells])
        uy_err = err_scale * np.sqrt(diag[n_cells:])
    else:
        logger.info(
            "Skipping posterior errors: %d unknowns > %d",
            n_unknowns,
            _MAX_POSTERIOR_UNKNOWNS,
        )

    logger.info(
        "Field inversion: %d obs -> %dx%d cells at %.0f m, chi2=%.2f",
        n_used,
        n_y,
        n_x,
        h,
        chi2,
    )

    return CurrentField(
        ux=ux_field,
        uy=uy_field,
        ux_err=ux_err.reshape(n_y, n_x),
        uy_err=uy_err.reshape(n_y, n_x),
        coverage=coverage.reshape(n_y, n_x),
        x_centers=x_centers,
        y_centers=y_centers,
        grid_spacing=h,
        center_lat=center_lat,
        center_lon=center_lon,
        n_obs=n_used,
        chi2=chi2,
        correlation_length=correlation_length,
        sigma_prior=sigma_prior,
        start_time=start_time if start_time is not None else np.datetime64("NaT"),
        end_time=end_time if end_time is not None else np.datetime64("NaT"),
    )


def write_field_netcdf(fld: CurrentField, output_dir: str) -> str:
    """Write a CurrentField to a CF-1.8 NetCDF file.

    Args:
        fld: CurrentField to write.
        output_dir: Output directory.

    Returns:
        Path to created file, or "" when xarray is unavailable.
    """
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not installed, skipping field NetCDF output")
        return ""

    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_str = np.datetime_as_string(fld.start_time, unit="s").replace(":", "-").replace("T", "_")
    end_str = np.datetime_as_string(fld.end_time, unit="s").replace(":", "-").replace("T", "_")
    filepath = out / f"current_field_{start_str}_to_{end_str}.nc"

    ds = xr.Dataset(
        data_vars={
            "ux": (
                ["y", "x"],
                fld.ux.astype(np.float32),
                {
                    "long_name": "Eastward sea water velocity (field inversion)",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "uy": (
                ["y", "x"],
                fld.uy.astype(np.float32),
                {
                    "long_name": "Northward sea water velocity (field inversion)",
                    "standard_name": "northward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "ux_err": (
                ["y", "x"],
                fld.ux_err.astype(np.float32),
                {
                    "long_name": "Posterior 1-sigma uncertainty of ux",
                    "units": "m s-1",
                },
            ),
            "uy_err": (
                ["y", "x"],
                fld.uy_err.astype(np.float32),
                {
                    "long_name": "Posterior 1-sigma uncertainty of uy",
                    "units": "m s-1",
                },
            ),
            "coverage": (
                ["y", "x"],
                fld.coverage.astype(np.int32),
                {
                    "long_name": "Number of observation footprints covering cell",
                    "units": "1",
                    "comment": "0 = pure prior extrapolation",
                },
            ),
        },
        coords={
            "x": (
                ["x"],
                fld.x_centers.astype(np.float32),
                {"long_name": "Distance east from center", "units": "m", "axis": "X"},
            ),
            "y": (
                ["y"],
                fld.y_centers.astype(np.float32),
                {"long_name": "Distance north from center", "units": "m", "axis": "Y"},
            ),
            "time_start": fld.start_time,
            "time_end": fld.end_time,
        },
        attrs={
            "title": "WAMOS surface current field (joint inversion)",
            "institution": "WAMOS TPW",
            "source": "wamos current --field",
            "history": f"Created {np.datetime64('now')}",
            "Conventions": "CF-1.8",
            "center_latitude": fld.center_lat,
            "center_longitude": fld.center_lon,
            "n_observations": fld.n_obs,
            "reduced_chi2": fld.chi2,
            "correlation_length_m": fld.correlation_length,
            "sigma_prior_ms": fld.sigma_prior,
        },
    )

    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(filepath, encoding=encoding)

    logger.debug("Wrote current field to %s", filepath)
    return str(filepath)
