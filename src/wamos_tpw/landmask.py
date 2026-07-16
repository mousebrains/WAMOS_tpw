"""Static earth-referenced land mask built from stacked radar mosaics.

Hard returns (land, fixed platforms) are persistently bright in earth
coordinates while sea clutter and wave backscatter fluctuate, so the
per-cell temporal MINIMUM intensity across many mosaics separates land
from sea far more reliably than any single image. The mask lives on a
regular latitude/longitude grid and is queried per analysis tile with
the same equirectangular meters-from-center convention the current
pipeline uses, so a tile can be rejected when its land fraction exceeds
a threshold (stationary land echoes otherwise drag dispersion fits
toward zero velocity).

Build a mask from ``wamos files-pipeline`` merged mosaics::

    wamos land-mask 'mosaics/merged_*.nc' -o rota_mask.nc

then apply it during current extraction::

    wamos current ... --land-mask rota_mask.nc
"""

from __future__ import annotations

import functools
import glob as _glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)

__all__ = ["LandMask", "build_from_mosaics", "load_cached"]

_DEG2M = 111_319.5  # meters per degree latitude; matches current.py

_CELL_M_DEFAULT = 30.0
_THRESHOLD_PCT_DEFAULT = 99.0
_MIN_COUNT_DEFAULT = 3


@dataclass
class LandMask:
    """Boolean land grid on regular latitude/longitude cells.

    Attributes:
        lat0: Latitude of the southern edge of the first row (degrees).
        lon0: Longitude of the western edge of the first column (degrees).
        dlat: Cell height in degrees.
        dlon: Cell width in degrees.
        land: 2D bool array (n_lat, n_lon), True where land.
        threshold: Intensity threshold used to build the mask (counts).
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    land: np.ndarray
    threshold: float = float("nan")

    @property
    def n_lat(self) -> int:
        """Number of latitude rows."""
        return self.land.shape[0]

    @property
    def n_lon(self) -> int:
        """Number of longitude columns."""
        return self.land.shape[1]

    def land_fraction(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        center_lat: float,
        center_lon: float,
    ) -> float:
        """Fraction of mask cells inside a tile that are land.

        Args:
            x_min: Tile bound in meters east of (center_lat, center_lon).
            x_max: Tile bound in meters east of (center_lat, center_lon).
            y_min: Tile bound in meters north of (center_lat, center_lon).
            y_max: Tile bound in meters north of (center_lat, center_lon).
            center_lat: Reference latitude of the tile coordinate frame.
            center_lon: Reference longitude of the tile coordinate frame.

        Returns:
            Land fraction in [0, 1]. Cells outside the mask's coverage
            count as sea, and a tile entirely outside coverage returns 0.
        """
        m_per_deg_lon = _DEG2M * np.cos(np.radians(center_lat))
        lat_min = center_lat + y_min / _DEG2M
        lat_max = center_lat + y_max / _DEG2M
        lon_min = center_lon + x_min / m_per_deg_lon
        lon_max = center_lon + x_max / m_per_deg_lon

        i0 = int(np.floor((lat_min - self.lat0) / self.dlat))
        i1 = int(np.ceil((lat_max - self.lat0) / self.dlat))
        j0 = int(np.floor((lon_min - self.lon0) / self.dlon))
        j1 = int(np.ceil((lon_max - self.lon0) / self.dlon))

        n_total = max(i1 - i0, 1) * max(j1 - j0, 1)
        i0c, i1c = max(i0, 0), min(i1, self.n_lat)
        j0c, j1c = max(j0, 0), min(j1, self.n_lon)
        if i0c >= i1c or j0c >= j1c:
            return 0.0
        n_land = int(self.land[i0c:i1c, j0c:j1c].sum())
        return n_land / n_total

    def to_netcdf(self, path: str | Path) -> Path:
        """Write the mask as a CF-1.13 NetCDF file."""
        import xarray as xr

        lat = self.lat0 + (np.arange(self.n_lat) + 0.5) * self.dlat
        lon = self.lon0 + (np.arange(self.n_lon) + 0.5) * self.dlon
        ds = xr.Dataset(
            {
                "land": (
                    ("latitude", "longitude"),
                    self.land.astype(np.int8),
                    {
                        "long_name": "land / persistent hard-return mask",
                        "standard_name": "land_binary_mask",
                        "flag_values": np.array([0, 1], np.int8),
                        "flag_meanings": "sea land",
                    },
                ),
            },
            coords={
                "latitude": (
                    "latitude",
                    lat,
                    {"units": "degrees_north", "standard_name": "latitude"},
                ),
                "longitude": (
                    "longitude",
                    lon,
                    {"units": "degrees_east", "standard_name": "longitude"},
                ),
            },
            attrs={
                "title": "WAMOS static land mask from stacked radar mosaics",
                "source": "wamos land-mask (temporal-minimum intensity threshold)",
                "Conventions": "CF-1.13",
                "intensity_threshold": float(self.threshold),
                "history": f"Created {np.datetime64('now')}",
            },
        )
        path = Path(path)
        encoding = {"land": {"zlib": True, "complevel": 4}}
        ds.to_netcdf(path, encoding=encoding)
        logger.info(
            "Wrote land mask %s (%d x %d cells, %d land)",
            path,
            self.n_lat,
            self.n_lon,
            int(self.land.sum()),
        )
        return path

    @classmethod
    def from_netcdf(cls, path: str | Path) -> LandMask:
        """Load a mask written by :meth:`to_netcdf`."""
        import xarray as xr

        with xr.open_dataset(path) as ds:
            lat = ds.latitude.values
            lon = ds.longitude.values
            land = ds.land.values.astype(bool)
            threshold = float(ds.attrs.get("intensity_threshold", np.nan))
        dlat = float(lat[1] - lat[0]) if len(lat) > 1 else 1e-4
        dlon = float(lon[1] - lon[0]) if len(lon) > 1 else 1e-4
        return cls(
            lat0=float(lat[0]) - dlat / 2,
            lon0=float(lon[0]) - dlon / 2,
            dlat=dlat,
            dlon=dlon,
            land=land,
            threshold=threshold,
        )


@functools.lru_cache(maxsize=4)
def load_cached(path: str) -> LandMask:
    """Load a mask NetCDF once per path (cached across cubes/blocks)."""
    mask = LandMask.from_netcdf(path)
    logger.info(
        "Loaded land mask %s: %d x %d cells, %.1f%% land",
        path,
        mask.n_lat,
        mask.n_lon,
        100.0 * mask.land.mean(),
    )
    return mask


def _mosaic_lat_lon(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel latitude/longitude vectors of a merged mosaic."""
    center_lat = float(ds.attrs["center_latitude"])
    center_lon = float(ds.attrs["center_longitude"])
    lat = center_lat + ds.y.values / _DEG2M
    lon = center_lon + ds.x.values / (_DEG2M * np.cos(np.radians(center_lat)))
    return lat, lon


def build_from_mosaics(
    mosaics: list[str | Path] | list[xr.Dataset],
    cell_m: float = _CELL_M_DEFAULT,
    threshold_pct: float = _THRESHOLD_PCT_DEFAULT,
    threshold: float | None = None,
    min_count: int = _MIN_COUNT_DEFAULT,
) -> LandMask:
    """Build a land mask from merged mosaic files or open datasets.

    Args:
        mosaics: Paths to ``merged_*.nc`` files (from ``wamos
            files-pipeline``) or already-open datasets with the same
            layout (``intensity(y, x)``, meter offsets ``x``/``y``,
            ``center_latitude``/``center_longitude`` attributes).
        cell_m: Mask grid resolution in meters.
        threshold_pct: Percentile of the temporal-minimum stack used as
            the land threshold when ``threshold`` is not given.
        threshold: Absolute intensity threshold overriding
            ``threshold_pct``.
        min_count: Minimum mosaics covering a cell for it to be usable.

    Returns:
        LandMask on a lat/lon grid covering the union of the mosaics.
    """
    import xarray as xr

    def _open(m):
        if isinstance(m, (str, Path)):
            return xr.open_dataset(m)
        return m

    if not mosaics:
        raise ValueError("no mosaics given")

    # Pass 1: geographic bounds
    lat_lo, lat_hi = np.inf, -np.inf
    lon_lo, lon_hi = np.inf, -np.inf
    for m in mosaics:
        ds = _open(m)
        lat, lon = _mosaic_lat_lon(ds)
        lat_lo, lat_hi = min(lat_lo, lat.min()), max(lat_hi, lat.max())
        lon_lo, lon_hi = min(lon_lo, lon.min()), max(lon_hi, lon.max())
        if ds is not m:
            ds.close()

    mid_lat = 0.5 * (lat_lo + lat_hi)
    dlat = cell_m / _DEG2M
    dlon = cell_m / (_DEG2M * np.cos(np.radians(mid_lat)))
    n_lat = int((lat_hi - lat_lo) / dlat) + 1
    n_lon = int((lon_hi - lon_lo) / dlon) + 1
    logger.info(
        "Land mask grid %d x %d cells (%.0f m) from %d mosaics",
        n_lat,
        n_lon,
        cell_m,
        len(mosaics),
    )

    # Pass 2: accumulate per-cell temporal minimum and coverage count
    mn = np.full((n_lat, n_lon), np.inf, np.float32)
    cnt = np.zeros((n_lat, n_lon), np.int32)
    for m in mosaics:
        ds = _open(m)
        lat, lon = _mosaic_lat_lon(ds)
        iy = ((lat - lat_lo) / dlat).astype(np.intp)
        ix = ((lon - lon_lo) / dlon).astype(np.intp)
        val = ds.intensity.values
        gy, gx = np.nonzero(np.isfinite(val))
        np.minimum.at(mn, (iy[gy], ix[gx]), val[gy, gx].astype(np.float32))
        np.add.at(cnt, (iy[gy], ix[gx]), 1)
        if ds is not m:
            ds.close()

    seen = cnt >= min_count
    if not seen.any():
        raise ValueError(f"no mask cell covered by at least {min_count} mosaics")
    if threshold is None:
        threshold = float(np.percentile(mn[seen], threshold_pct))
    land = seen & (mn > threshold)
    logger.info(
        "Land threshold %.0f counts -> %d land cells (%.2f%% of covered)",
        threshold,
        int(land.sum()),
        100.0 * land.sum() / seen.sum(),
    )
    return LandMask(
        lat0=lat_lo,
        lon0=lon_lo,
        dlat=dlat,
        dlon=dlon,
        land=land,
        threshold=threshold,
    )


def _add_arguments(parser) -> None:
    """Add land-mask CLI arguments."""
    parser.add_argument(
        "mosaics",
        nargs="+",
        help="Merged mosaic NetCDF files, directories, or globs (from 'wamos files-pipeline')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output land mask NetCDF filename",
    )
    parser.add_argument(
        "--cell",
        type=float,
        default=_CELL_M_DEFAULT,
        help=f"Mask grid resolution in meters (default: {_CELL_M_DEFAULT:.0f})",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=_THRESHOLD_PCT_DEFAULT,
        help="Percentile of the temporal-minimum stack used as the land "
        f"threshold (default: {_THRESHOLD_PCT_DEFAULT:.0f})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute intensity threshold (overrides --threshold-pct)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=_MIN_COUNT_DEFAULT,
        help=f"Minimum mosaics covering a cell for it to be usable (default: {_MIN_COUNT_DEFAULT})",
    )


def add_subparser(subparsers) -> None:
    """Register the 'land-mask' subcommand."""
    p = subparsers.add_parser(
        "land-mask",
        help="Build a static land mask from stacked mosaics",
        description="Build a static earth-referenced land mask from "
        "'wamos files-pipeline' merged mosaics via a temporal-minimum "
        "intensity threshold. Apply with 'wamos current --land-mask'.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'land-mask' command."""
    files: list[str | Path] = []
    for spec in args.mosaics:
        p = Path(spec)
        if p.is_dir():
            files.extend(sorted(p.glob("merged_*.nc")))
        elif any(ch in spec for ch in "*?["):
            files.extend(sorted(Path(f) for f in _glob.glob(spec)))
        else:
            files.append(p)
    if not files:
        raise SystemExit(f"no mosaic files matched {args.mosaics}")
    logger.info("Building land mask from %d mosaics", len(files))
    mask = build_from_mosaics(
        files,
        cell_m=args.cell,
        threshold_pct=args.threshold_pct,
        threshold=args.threshold,
        min_count=args.min_count,
    )
    mask.to_netcdf(args.output)
