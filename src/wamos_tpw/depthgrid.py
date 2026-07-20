"""Per-tile water depth from a bathymetry grid for finite-depth dispersion.

Over shallow banks the dispersion relation depends on depth to first
order (omega^2 = g k tanh(kh); kh ~ 0.5-1.5 on an 18 m bank for the
energetic wave band), so a single scalar ``current.depth`` is not enough
where depth varies across the scene. A :class:`DepthGrid` loads a
bathymetry NetCDF once and answers per-tile queries with the median
depth over the tile footprint plus a heterogeneity flag for tiles that
straddle steep relief (bank edges), where a single depth cannot
represent the tile.

Supported inputs:

- 2-D curvilinear grids with per-point ``lat``/``lon`` arrays and an
  elevation variable (e.g. the Palau ``Angaur_Peleliu_25m.nc``);
  rebinned to a regular lat/lon grid at native resolution on load.
- 1-D regular lat/lon grids (GEBCO-style ``lat``/``lon``/``elevation``).

Elevation is meters, positive up; depth = -elevation over water.

Usage::

    wamos current ... --depth-grid Angaur_Peleliu_25m.nc
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DepthGrid", "load_cached"]

_DEG2M = 111_319.5  # meters per degree latitude; matches current.py

# A tile is flagged heterogeneous when the shallow side is shallow
# enough for finite-depth effects (kh < ~1.5 for the energetic band)
# while the deep side is a different dispersion regime.
_HETERO_SHALLOW_M = 50.0
_HETERO_RATIO = 2.0


@dataclass
class DepthGrid:
    """Regular lat/lon grid of water depth (meters, positive down).

    Attributes:
        lat0: Latitude of the southern edge of the first row (degrees).
        lon0: Longitude of the western edge of the first column (degrees).
        dlat: Cell height in degrees.
        dlon: Cell width in degrees.
        depth: 2D float array (n_lat, n_lon); NaN where unknown or land.
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    depth: np.ndarray

    def tile_depth(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        center_lat: float,
        center_lon: float,
    ) -> tuple[float, bool]:
        """Median depth over a tile and a heterogeneity flag.

        Args:
            x_min: Tile bound in meters east of (center_lat, center_lon).
            x_max: Tile bound in meters east of (center_lat, center_lon).
            y_min: Tile bound in meters north of (center_lat, center_lon).
            y_max: Tile bound in meters north of (center_lat, center_lon).
            center_lat: Reference latitude of the tile coordinate frame.
            center_lon: Reference longitude of the tile coordinate frame.

        Returns:
            ``(depth_m, heterogeneous)``. ``depth_m`` is the median water
            depth over grid cells inside the tile (positive meters, may
            be ``inf`` when the tile has no bathymetry coverage).
            ``heterogeneous`` is True when the tile straddles relief
            steep enough that one depth misrepresents it (p90/p10 >
            ``_HETERO_RATIO`` with the shallow side above
            ``_HETERO_SHALLOW_M``).
        """
        m_per_deg_lon = _DEG2M * np.cos(np.radians(center_lat))
        i0 = int(np.floor((center_lat + y_min / _DEG2M - self.lat0) / self.dlat))
        i1 = int(np.ceil((center_lat + y_max / _DEG2M - self.lat0) / self.dlat))
        j0 = int(np.floor((center_lon + x_min / m_per_deg_lon - self.lon0) / self.dlon))
        j1 = int(np.ceil((center_lon + x_max / m_per_deg_lon - self.lon0) / self.dlon))
        i0, i1 = max(i0, 0), min(i1, self.depth.shape[0])
        j0, j1 = max(j0, 0), min(j1, self.depth.shape[1])
        if i0 >= i1 or j0 >= j1:
            return float("inf"), False
        d = self.depth[i0:i1, j0:j1]
        d = d[np.isfinite(d) & (d > 0)]
        if d.size < 4:
            return float("inf"), False
        p10, p50, p90 = np.percentile(d, [10, 50, 90])
        hetero = bool(p10 < _HETERO_SHALLOW_M and p90 > _HETERO_RATIO * p10)
        return float(p50), hetero

    @classmethod
    def from_netcdf(cls, path: str) -> DepthGrid:
        """Load a bathymetry NetCDF (2-D curvilinear or 1-D regular)."""
        import xarray as xr

        with xr.open_dataset(path) as ds:
            names = {k.lower(): k for k in list(ds.variables)}
            lat = ds[names.get("lat", names.get("latitude", ""))].values
            lon = ds[names.get("lon", names.get("longitude", ""))].values
            zname = next(
                (names[c] for c in ("z", "elevation", "depth") if c in names),
                None,
            )
            if zname is None:
                raise ValueError(f"{path}: no elevation variable (looked for z/elevation/depth)")
            z = ds[zname].values.astype(float)
            if names.get("depth", "") == zname:
                z = -z  # a 'depth' variable is positive down already

        if lat.ndim == 1:
            dlat = float(np.median(np.diff(lat)))
            dlon = float(np.median(np.diff(lon)))
            if dlat < 0:  # north-to-south grids
                lat, z = lat[::-1], z[::-1]
                dlat = -dlat
            depth = np.where(z < 0, -z, np.nan)
            return cls(
                lat0=float(lat[0]) - dlat / 2,
                lon0=float(lon[0]) - dlon / 2,
                dlat=dlat,
                dlon=dlon,
                depth=depth,
            )

        # 2-D curvilinear: bin points onto a regular grid at native spacing
        good = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(z)
        glat, glon, gz = lat[good], lon[good], z[good]
        if glat.size == 0:
            raise ValueError(f"{path}: no finite bathymetry points")
        # native spacing from typical nearest-row offsets
        spacing_m = (
            np.median(np.abs(np.diff(lat, axis=0))[np.abs(np.diff(lat, axis=0)) > 0]) * _DEG2M
        )
        dlat = spacing_m / _DEG2M
        dlon = spacing_m / (_DEG2M * np.cos(np.radians(float(np.mean(glat)))))
        lat0 = glat.min() - dlat / 2
        lon0 = glon.min() - dlon / 2
        n_lat = int((glat.max() - lat0) / dlat) + 1
        n_lon = int((glon.max() - lon0) / dlon) + 1
        iy = ((glat - lat0) / dlat).astype(np.intp)
        ix = ((glon - lon0) / dlon).astype(np.intp)
        acc = np.zeros((n_lat, n_lon))
        cnt = np.zeros((n_lat, n_lon), np.int32)
        np.add.at(acc, (iy, ix), gz)
        np.add.at(cnt, (iy, ix), 1)
        with np.errstate(invalid="ignore"):
            zmean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        depth = np.where(zmean < 0, -zmean, np.nan)
        logger.info(
            "Depth grid %s: %d x %d cells (%.0f m), %.1f%% covered",
            path,
            n_lat,
            n_lon,
            spacing_m,
            100.0 * (cnt > 0).mean(),
        )
        return cls(lat0=lat0, lon0=lon0, dlat=dlat, dlon=dlon, depth=depth)


@functools.lru_cache(maxsize=4)
def load_cached(path: str) -> DepthGrid:
    """Load a depth grid once per path (cached across cubes/blocks)."""
    grid = DepthGrid.from_netcdf(path)
    logger.info(
        "Loaded depth grid %s: depth %.0f..%.0f m",
        path,
        np.nanmin(grid.depth),
        np.nanmax(grid.depth),
    )
    return grid
