#! /usr/bin/env python3
#
# Data-driven quality-threshold analysis for current block maps
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Sweep SNR thresholds and range rings against internal consistency.

Given a directory of per-block ``current_*.nc`` maps (run with
``--min-snr 0`` so the full distribution is captured), this measures,
for each candidate threshold and for each range ring:

- the fraction of tile vectors retained,
- the per-cell *temporal* scatter of retained vectors (the same earth
  cell observed across many blocks; the true current changes slowly, so
  scatter ~ error),
- the implied calibration of the formal least-squares errors.

This is how the recommended operating point in
``docs/realdata_findings.md`` (min_snr ~ 1.1, max_tile_range ~ 3000 m)
was derived; rerun it when conditions, ship, or radar settings change.

Usage::

    wamos current <stime> <etime> <polar> -o <dir> --min-snr 0 ...
    python3 tools/current_quality_sweep.py <dir>
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import xarray as xr

_DEG2M = 111_319.5


def load_observations(directory: str):
    """All tile vectors from block maps, with absolute coords and range."""
    xs, ys, rs, uxs, uys, snrs, errs = [], [], [], [], [], [], []
    ref = None
    files = sorted(glob.glob(f"{directory}/current_*.nc"))
    files = [f for f in files if "composite" not in f and "field" not in f]
    for f in files:
        ds = xr.open_dataset(f)
        lat0 = float(ds.attrs["center_latitude"])
        lon0 = float(ds.attrs["center_longitude"])
        if ref is None:
            ref = (lat0, lon0)
        map_x = (lon0 - ref[1]) * _DEG2M * np.cos(np.deg2rad(ref[0]))
        map_y = (lat0 - ref[0]) * _DEG2M
        xg, yg = np.meshgrid(ds["x"].values, ds["y"].values)
        ux = ds["ux"].values
        ok = np.isfinite(ux)
        xs.append(xg[ok] + map_x)
        ys.append(yg[ok] + map_y)
        # Tile coordinates are cube-relative; the ship is near the cube
        # center, so |(x, y)| approximates range from the radar
        rs.append(np.hypot(xg[ok], yg[ok]))
        uxs.append(ux[ok])
        uys.append(ds["uy"].values[ok])
        snrs.append(ds["snr"].values[ok])
        if "ux_err" in ds:
            errs.append(ds["ux_err"].values[ok])
        else:
            errs.append(np.full(int(ok.sum()), np.nan))
        ds.close()
    if not xs:
        raise SystemExit(f"No block maps found in {directory}")
    return tuple(np.concatenate(a) for a in (xs, ys, rs, uxs, uys, snrs, errs))


def cell_scatter(x, y, ux, uy, cell: float) -> float:
    """Median per-cell robust scatter of vectors binned on an earth lattice."""
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    key = (ix - ix.min()) * 1_000_000 + (iy - iy.min())
    scatters = []
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= 5:
            mu, mv = np.median(ux[sel]), np.median(uy[sel])
            scatters.append(float(np.median(np.hypot(ux[sel] - mu, uy[sel] - mv))))
    return float(np.median(scatters)) if scatters else float("nan")


def snr_sweep(obs, thresholds, cell: float) -> None:
    x, y, _r, ux, uy, snr, err = obs
    print(f"\n{'min_snr':>8} {'kept%':>6} {'scatter(m/s)':>13} {'formal(m/s)':>12} {'calib':>6}")
    for thr in thresholds:
        keep = snr >= thr
        if keep.sum() < 20:
            print(f"{thr:8.2f}  too few observations")
            continue
        scat = cell_scatter(x[keep], y[keep], ux[keep], uy[keep], cell)
        formal = float(np.nanmedian(err[keep]))
        print(
            f"{thr:8.2f} {100 * keep.mean():6.0f} {scat:13.3f} {formal:12.3f} "
            f"{scat / max(formal, 1e-6):6.1f}"
        )


def range_rings(obs, rings, min_snr: float) -> None:
    x, y, r, ux, uy, snr, _err = obs
    print(f"\nrange rings at min_snr >= {min_snr}:")
    print(f"{'ring (km)':>10} {'n':>7} {'scatter(m/s)':>13} {'med_snr':>8}")
    for lo, hi in rings:
        sel = (r >= lo * 1000) & (r < hi * 1000) & (snr >= min_snr)
        if sel.sum() < 20:
            print(f"{lo}-{hi:>7} {int(sel.sum()):7d}  too few")
            continue
        mu, mv = np.median(ux[sel]), np.median(uy[sel])
        scat = float(np.median(np.hypot(ux[sel] - mu, uy[sel] - mv)))
        print(f"{lo}-{hi:>7} {int(sel.sum()):7d} {scat:13.3f} {float(np.median(snr[sel])):8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory of per-block current_*.nc maps")
    parser.add_argument("--cell", type=float, default=1000.0, help="Earth cell size (m)")
    parser.add_argument(
        "--ring-snr", type=float, default=1.1, help="SNR gate for the range-ring table"
    )
    args = parser.parse_args()

    obs = load_observations(args.directory)
    print(f"{len(obs[0])} tile vectors from {args.directory}")
    snr_sweep(obs, [0.0, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0], args.cell)
    range_rings(obs, [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8)], args.ring_snr)


if __name__ == "__main__":
    main()
