#! /usr/bin/env python3
"""Convert WAMOS composite current maps to hourly trajectory NetCDF products.

Walks directories of ``current_composite_*.nc`` files (from ``wamos
current --composite-minutes``), flattens every populated composite cell
into one measurement, groups measurements by UTC hour, and writes one
CF-1.13 trajectory file per hour, patterned on the CSTARS/Hereon
near-surface current product layout (measurement dimension with
time/longitude/latitude/velocity/error/quality variables).

Formal composite errors are inflated by sqrt(max(chi2, 1)) per cell:
real-data validation showed block-to-block scatter exceeds the formal
least-squares errors by an order of magnitude, and the composite
reduced chi-square measures exactly that excess.

Example::

    python tools/composites_to_trajectory.py \
        '/path/products/currents_windows/202204*' \
        -o /path/products/currents --prefix rr
"""

from __future__ import annotations

import argparse
import glob
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ATTRS = {
    "instrument": "WaMoS II marine X-band radar",
    "platform": "R/V Roger Revelle",
    "source": "Shipboard marine X-band radar",
    "title": ("Marine X-band radar near-surface current measurements from R/V Roger Revelle"),
    "comment": (
        "Near-surface current vectors from least-squares fits of the "
        "wave signal in radar backscatter wavenumber-frequency spectra "
        "to the linear deep-water dispersion shell (wamos_tpw: 32-frame "
        "blocks, 2 km analysis tiles, sub-bin spectral peak refinement). "
        "Successive block estimates are combined into inverse-variance "
        "weighted composites; each populated composite cell is one "
        "measurement. Tiles beyond 3 km range, crossed by the antenna "
        "rotation seam, or overlapping land (static radar-derived land "
        "mask) are excluded. Velocity standard errors are the formal "
        "composite errors inflated by sqrt(reduced chi-square) of the "
        "contributing block estimates. The effective depth of the "
        "measurement is a weighted mean over roughly the upper 4-16 m "
        "of the water column, set by the wavenumbers of the fitted "
        "wave signal."
    ),
    "institution": "Oregon State University, CEOAS",
    "originator": "Pat Welch",
    "contact": "pat@mousebrains.com",
    "project": "ARCTERX",
    "Conventions": "CF-1.13",
    "featureType": "trajectory",
    "licence": ("Creative Commons Attribution 4.0 International Public License (CC BY 4.0)"),
}


def collect(window_dirs: list[Path]) -> dict[str, list[dict]]:
    """Read composites and bucket per-cell measurements by UTC hour."""
    import pandas as pd
    import xarray as xr

    hours: dict[str, list[dict]] = defaultdict(list)
    n_files = 0
    for wdir in window_dirs:
        for fn in sorted(wdir.glob("current_composite_*.nc")):
            with xr.open_dataset(fn) as ds:
                t0 = pd.to_datetime(ds.time_start.values)
                t1 = pd.to_datetime(ds.time_end.values)
                tmid = t0 + (t1 - t0) / 2
                lat2d, lon2d = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
                ok = np.isfinite(ds.ux.values)
                if not ok.any():
                    continue
                iy, ix = np.nonzero(ok)
                infl = np.sqrt(np.maximum(ds.chi2.values[ok], 1.0))
                hours[tmid.strftime("%Y-%m-%d-%H")].append(
                    dict(
                        time=np.full(ok.sum(), tmid.to_datetime64()),
                        t0=t0,
                        t1=t1,
                        lat=lat2d[ok],
                        lon=lon2d[ok],
                        u=ds.ux.values[ok],
                        v=ds.uy.values[ok],
                        u_err=ds.ux_err.values[ok] * infl,
                        v_err=ds.uy_err.values[ok] * infl,
                        n_obs=ds.n_obs.values[ok],
                        chi2=ds.chi2.values[ok],
                    )
                )
                n_files += 1
    logger.info("Collected %d composites -> %d hours", n_files, len(hours))
    return hours


def write_hour(key: str, chunks: list[dict], outdir: Path, prefix: str) -> Path:
    """Write one hourly trajectory NetCDF."""
    import xarray as xr

    cat = {
        k: np.concatenate([c[k] for c in chunks])
        for k in ("time", "lat", "lon", "u", "v", "u_err", "v_err", "n_obs", "chi2")
    }
    n = len(cat["u"])
    order = np.argsort(cat["time"])
    for k in cat:
        cat[k] = cat[k][order]

    enc_f: dict[str, object] = dict(zlib=True, complevel=4)
    ds = xr.Dataset(
        {
            "time": (
                "measurement",
                cat["time"],
                {
                    "long_name": "mid time of the contributing composite window",
                    "standard_name": "time",
                },
            ),
            "longitude": (
                "measurement",
                cat["lon"],
                {
                    "long_name": "center longitude of current measurement",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "latitude": (
                "measurement",
                cat["lat"],
                {
                    "long_name": "center latitude of current measurement",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "eastward_sea_water_velocity": (
                "measurement",
                cat["u"],
                {
                    "long_name": "eastward component of the near surface current velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "northward_sea_water_velocity": (
                "measurement",
                cat["v"],
                {
                    "long_name": "northward component of the near surface current velocity",
                    "standard_name": "northward_sea_water_velocity",
                    "units": "m s-1",
                },
            ),
            "eastward_sea_water_velocity_standard_error": (
                "measurement",
                cat["u_err"],
                {
                    "long_name": "chi-square inflated standard error of the eastward component",
                    "units": "m s-1",
                },
            ),
            "northward_sea_water_velocity_standard_error": (
                "measurement",
                cat["v_err"],
                {
                    "long_name": "chi-square inflated standard error of the northward component",
                    "units": "m s-1",
                },
            ),
            "number_of_block_estimates": (
                "measurement",
                cat["n_obs"].astype(np.int32),
                {
                    "long_name": "number of block estimates in the composite cell",
                    "units": "1",
                },
            ),
            "reduced_chi_square": (
                "measurement",
                cat["chi2"].astype(np.float32),
                {
                    "long_name": "reduced chi-square of block estimates in the composite cell",
                    "units": "1",
                },
            ),
        },
        attrs=dict(
            _ATTRS,
            time_coverage_start=str(min(c["t0"] for c in chunks)),
            time_coverage_end=str(max(c["t1"] for c in chunks)),
            geospatial_lat_min=float(cat["lat"].min()),
            geospatial_lat_max=float(cat["lat"].max()),
            geospatial_lon_min=float(cat["lon"].min()),
            geospatial_lon_max=float(cat["lon"].max()),
            history=f"Created {np.datetime64('now')} by tools/composites_to_trajectory.py",
        ),
    )
    out = outdir / f"{prefix}_{key}_currents.nc"
    encoding = {k: enc_f for k in ds.data_vars if ds[k].dtype.kind == "f"}
    encoding["time"] = dict(
        units="days since 2022-01-01T00:00:00Z", calendar="standard", dtype="float64", **enc_f
    )
    ds.to_netcdf(out, encoding=encoding)
    logger.info("%s: %d measurements", out.name, n)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "windows",
        nargs="+",
        help="Window output directories or globs (each holding current_composite_*.nc)",
    )
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--prefix", default="rr", help="Filename prefix / platform tag (default: rr)")
    args = p.parse_args()

    dirs: list[Path] = []
    for spec in args.windows:
        matches = glob.glob(spec)
        dirs.extend(Path(m) for m in sorted(matches) if Path(m).is_dir())
    if not dirs:
        raise SystemExit(f"no window directories matched {args.windows}")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    hours = collect(dirs)
    for key in sorted(hours):
        write_hour(key, hours[key], outdir, args.prefix)


if __name__ == "__main__":
    main()
