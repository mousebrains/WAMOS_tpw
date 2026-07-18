"""Tests for per-tile depth from bathymetry grids (depthgrid.py)."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from wamos_tpw.depthgrid import DepthGrid  # noqa: E402

_DEG2M = 111_319.5

CENTER_LAT = 6.95
CENTER_LON = 134.2


def _bank_grid(cell_m: float = 50.0, n: int = 200) -> DepthGrid:
    """Regular grid: 18 m bank in the NE quadrant, 500 m water elsewhere."""
    dlat = cell_m / _DEG2M
    dlon = cell_m / (_DEG2M * np.cos(np.radians(CENTER_LAT)))
    depth = np.full((n, n), 500.0)
    depth[n // 2 :, n // 2 :] = 18.0
    return DepthGrid(
        lat0=CENTER_LAT - n / 2 * dlat,
        lon0=CENTER_LON - n / 2 * dlon,
        dlat=dlat,
        dlon=dlon,
        depth=depth,
    )


class TestTileDepth:
    def test_tile_on_bank(self):
        g = _bank_grid()
        d, het = g.tile_depth(1000, 3000, 1000, 3000, CENTER_LAT, CENTER_LON)
        assert d == pytest.approx(18.0)
        assert not het

    def test_tile_in_deep_water(self):
        g = _bank_grid()
        d, het = g.tile_depth(-3000, -1000, -3000, -1000, CENTER_LAT, CENTER_LON)
        assert d == pytest.approx(500.0)
        assert not het

    def test_tile_straddling_bank_edge_flagged(self):
        g = _bank_grid()
        d, het = g.tile_depth(-1000, 1000, -1000, 1000, CENTER_LAT, CENTER_LON)
        assert het
        assert 18.0 <= d <= 500.0

    def test_tile_outside_coverage_returns_inf(self):
        g = _bank_grid()
        d, het = g.tile_depth(50_000, 52_000, 50_000, 52_000, CENTER_LAT, CENTER_LON)
        assert np.isinf(d)
        assert not het

    def test_deep_deep_variation_not_flagged(self):
        """Relief entirely below the shallow threshold is not heterogeneous."""
        g = _bank_grid()
        g.depth[: g.depth.shape[0] // 2, : g.depth.shape[1] // 2] = 2000.0
        d, het = g.tile_depth(-3000, -500, -3000, -500, CENTER_LAT, CENTER_LON)
        assert not het


class TestFromNetcdf:
    def test_regular_1d_grid(self, tmp_path):
        lat = np.linspace(6.9, 7.0, 50)
        lon = np.linspace(134.15, 134.25, 60)
        z = np.full((50, 60), -300.0)
        z[:10, :10] = 20.0  # land
        ds = xr.Dataset(
            {"elevation": (("lat", "lon"), z)},
            coords={"lat": lat, "lon": lon},
        )
        p = tmp_path / "gebco.nc"
        ds.to_netcdf(p)
        g = DepthGrid.from_netcdf(str(p))
        d, _ = g.tile_depth(0, 2000, 0, 2000, 6.95, 134.2)
        assert d == pytest.approx(300.0)
        # land cells are NaN -> excluded
        assert np.isnan(g.depth[:10, :10]).all()

    def test_curvilinear_2d_grid(self, tmp_path):
        ny, nx = 80, 90
        lat2d, lon2d = np.meshgrid(
            np.linspace(6.90, 6.98, ny),
            np.linspace(134.16, 134.24, nx),
            indexing="ij",
        )
        z = np.full((ny, nx), -40.0)
        ds = xr.Dataset(
            {
                "lat": (("y", "x"), lat2d),
                "lon": (("y", "x"), lon2d),
                "Z": (("y", "x"), z),
            }
        )
        p = tmp_path / "curvi.nc"
        ds.to_netcdf(p)
        g = DepthGrid.from_netcdf(str(p))
        d, het = g.tile_depth(-1000, 1000, -1000, 1000, 6.94, 134.20)
        assert d == pytest.approx(40.0, rel=0.01)
        assert not het

    def test_north_to_south_grid(self, tmp_path):
        lat = np.linspace(7.0, 6.9, 50)  # descending
        lon = np.linspace(134.15, 134.25, 60)
        ds = xr.Dataset(
            {"elevation": (("lat", "lon"), np.full((50, 60), -123.0))},
            coords={"lat": lat, "lon": lon},
        )
        p = tmp_path / "n2s.nc"
        ds.to_netcdf(p)
        g = DepthGrid.from_netcdf(str(p))
        d, _ = g.tile_depth(-500, 500, -500, 500, 6.95, 134.2)
        assert d == pytest.approx(123.0)


class TestTileSpecsIntegration:
    def test_per_tile_depth_and_hetero_flag(self, tmp_path):
        from wamos_tpw.current import FrameCube, compute_tile_specs

        g = _bank_grid()
        # persist as a 1-D regular file the loader accepts
        lat = g.lat0 + (np.arange(g.depth.shape[0]) + 0.5) * g.dlat
        lon = g.lon0 + (np.arange(g.depth.shape[1]) + 0.5) * g.dlon
        ds = xr.Dataset(
            {"elevation": (("lat", "lon"), -g.depth)},
            coords={"lat": lat, "lon": lon},
        )
        p = tmp_path / "bank.nc"
        ds.to_netcdf(p)

        n = 200
        spacing = 40.0
        x = (np.arange(n) - n / 2) * spacing
        cube = FrameCube(
            intensity=np.zeros((4, n, n)),
            timestamps=np.arange(4).astype("datetime64[s]"),
            dt=1.5,
            x_centers=x,
            y_centers=x.copy(),
            grid_spacing=spacing,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
        )
        cfg = {
            "current.sub_region_size": 2000.0,
            "current.mask_seam": False,
            "current.depth_grid": str(p),
        }
        tiles = compute_tile_specs(cube, config=cfg)["tiles"]
        bank = [t for t in tiles if t["center_x"] > 1500 and t["center_y"] > 1500]
        deep = [t for t in tiles if t["center_x"] < -1500 and t["center_y"] < -1500]
        edge = [t for t in tiles if abs(t["center_x"]) < 600 and abs(t["center_y"]) < 600]
        assert bank and deep and edge
        assert all(t["depth"] == pytest.approx(18.0) for t in bank)
        assert all(t["depth"] == pytest.approx(500.0) for t in deep)
        assert any(t["depth_hetero"] for t in edge)
        # full coverage: nothing should be flagged missing
        assert not any(t["depth_missing"] for t in tiles)

        # depth_adjust shifts every finite tile depth by the calibrated bias
        cfg_adj = dict(cfg)
        cfg_adj["current.depth_adjust"] = 1.2
        tiles_adj = compute_tile_specs(cube, config=cfg_adj)["tiles"]
        bank_adj = [t for t in tiles_adj if t["center_x"] > 1500 and t["center_y"] > 1500]
        assert all(t["depth"] == pytest.approx(19.2) for t in bank_adj)

    def test_missing_coverage_flagged_not_silent(self, tmp_path):
        """Tiles outside the bathymetry get depth=inf AND depth_missing=True."""
        from wamos_tpw.current import FrameCube, compute_tile_specs

        # tiny grid covering only the north-east corner of the cube
        n_g = 40
        dlat = 50.0 / _DEG2M
        lat = CENTER_LAT + (np.arange(n_g) + 10) * dlat
        lon = CENTER_LON + (np.arange(n_g) + 10) * dlat
        ds = xr.Dataset(
            {"elevation": (("lat", "lon"), np.full((n_g, n_g), -20.0))},
            coords={"lat": lat, "lon": lon},
        )
        p = tmp_path / "corner.nc"
        ds.to_netcdf(p)

        n = 200
        spacing = 40.0
        x = (np.arange(n) - n / 2) * spacing
        cube = FrameCube(
            intensity=np.zeros((4, n, n)),
            timestamps=np.arange(4).astype("datetime64[s]"),
            dt=1.5,
            x_centers=x,
            y_centers=x.copy(),
            grid_spacing=spacing,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
        )
        cfg = {
            "current.sub_region_size": 2000.0,
            "current.mask_seam": False,
            "current.depth_grid": str(p),
        }
        tiles = compute_tile_specs(cube, config=cfg)["tiles"]
        missing = [t for t in tiles if t["depth_missing"]]
        covered = [t for t in tiles if not t["depth_missing"]]
        assert missing, "tiles beyond the grid must be flagged"
        assert all(not np.isfinite(t["depth"]) for t in missing)
        assert covered and all(t["depth"] == pytest.approx(20.0) for t in covered)


class TestExtractorOverride:
    def test_depth_argument_overrides_config(self):
        from wamos_tpw.current import CurrentExtractor, FrameCube

        rng = np.random.default_rng(0)
        n = 32
        cube = FrameCube(
            intensity=rng.normal(size=(8, n, n)),
            timestamps=np.arange(8).astype("datetime64[s]"),
            dt=1.5,
            x_centers=np.arange(n) * 10.0,
            y_centers=np.arange(n) * 10.0,
            grid_spacing=10.0,
            center_lat=CENTER_LAT,
            center_lon=CENTER_LON,
        )
        ex = CurrentExtractor(cube, config={"current.depth": 1000.0}, depth=18.0)
        assert ex._depth == 18.0
        ex2 = CurrentExtractor(cube, config={"current.depth": 1000.0})
        assert ex2._depth == 1000.0
