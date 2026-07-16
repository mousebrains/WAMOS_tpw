"""Tests for the static land mask (landmask.py)."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from wamos_tpw.landmask import LandMask, build_from_mosaics  # noqa: E402

_DEG2M = 111_319.5

CENTER_LAT = 14.1
CENTER_LON = 145.1


def _simple_mask(cell_m: float = 100.0, n: int = 100) -> LandMask:
    """Mask centered on (CENTER_LAT, CENTER_LON); NE quadrant is land."""
    dlat = cell_m / _DEG2M
    dlon = cell_m / (_DEG2M * np.cos(np.radians(CENTER_LAT)))
    land = np.zeros((n, n), bool)
    land[n // 2 :, n // 2 :] = True
    return LandMask(
        lat0=CENTER_LAT - n / 2 * dlat,
        lon0=CENTER_LON - n / 2 * dlon,
        dlat=dlat,
        dlon=dlon,
        land=land,
        threshold=500.0,
    )


class TestLandFraction:
    def test_tile_fully_on_land(self):
        mask = _simple_mask()
        frac = mask.land_fraction(1000, 3000, 1000, 3000, CENTER_LAT, CENTER_LON)
        assert frac == pytest.approx(1.0)

    def test_tile_fully_at_sea(self):
        mask = _simple_mask()
        frac = mask.land_fraction(-3000, -1000, -3000, -1000, CENTER_LAT, CENTER_LON)
        assert frac == pytest.approx(0.0)

    def test_tile_half_on_land(self):
        mask = _simple_mask()
        # Tile straddles the east-west land boundary north of center
        frac = mask.land_fraction(-1000, 1000, 1000, 3000, CENTER_LAT, CENTER_LON)
        assert frac == pytest.approx(0.5, abs=0.05)

    def test_tile_outside_coverage_counts_as_sea(self):
        mask = _simple_mask()
        frac = mask.land_fraction(50_000, 52_000, 50_000, 52_000, CENTER_LAT, CENTER_LON)
        assert frac == 0.0

    def test_tile_partially_outside_coverage(self):
        mask = _simple_mask()
        # Extends beyond the northern edge; outside cells count as sea,
        # so the fraction must be strictly between 0 and 1.
        frac = mask.land_fraction(1000, 3000, 4000, 20_000, CENTER_LAT, CENTER_LON)
        assert 0.0 < frac < 1.0


class TestNetcdfRoundTrip:
    def test_round_trip(self, tmp_path):
        mask = _simple_mask()
        path = tmp_path / "mask.nc"
        mask.to_netcdf(path)
        back = LandMask.from_netcdf(path)
        assert back.land.shape == mask.land.shape
        np.testing.assert_array_equal(back.land, mask.land)
        assert back.threshold == pytest.approx(mask.threshold)
        assert back.lat0 == pytest.approx(mask.lat0, abs=1e-9)
        assert back.dlon == pytest.approx(mask.dlon, rel=1e-6)
        # Query behaves identically after the round trip
        args = (1000, 3000, 1000, 3000, CENTER_LAT, CENTER_LON)
        assert back.land_fraction(*args) == pytest.approx(mask.land_fraction(*args))


def _synthetic_mosaic(rng, bright_block: bool) -> xr.Dataset:
    """A 100x100 mosaic, 20 m pixels, optional persistent bright block."""
    n = 100
    spacing = 20.0
    x = (np.arange(n) - n / 2) * spacing
    intensity = rng.uniform(0, 200, (n, n))
    if bright_block:
        intensity[70:90, 70:90] = rng.uniform(800, 1000, (20, 20))
    return xr.Dataset(
        {"intensity": (("y", "x"), intensity)},
        coords={"x": x, "y": x},
        attrs={"center_latitude": CENTER_LAT, "center_longitude": CENTER_LON},
    )


class TestBuildFromMosaics:
    def test_persistent_block_becomes_land(self):
        rng = np.random.default_rng(0)
        mosaics = [_synthetic_mosaic(rng, bright_block=True) for _ in range(6)]
        # Absolute threshold between the sea (<200) and block (>800) levels;
        # the percentile default is scene-dependent and tested separately.
        mask = build_from_mosaics(mosaics, cell_m=40.0, threshold=500.0)
        assert mask.land.any()
        # The block sits NE of center: meters (400..800, 400..800)
        frac_block = mask.land_fraction(500, 700, 500, 700, CENTER_LAT, CENTER_LON)
        frac_sea = mask.land_fraction(-900, -100, -900, -100, CENTER_LAT, CENTER_LON)
        assert frac_block > 0.8
        assert frac_sea == pytest.approx(0.0, abs=0.02)

    def test_fluctuating_bright_spots_stay_sea(self):
        rng = np.random.default_rng(1)
        # Bright block only in one mosaic out of six: the temporal min kills it
        mosaics = [_synthetic_mosaic(rng, bright_block=(i == 0)) for i in range(6)]
        mask = build_from_mosaics(mosaics, cell_m=40.0, threshold=500.0)
        assert not mask.land.any()

    def test_absolute_threshold_override(self):
        rng = np.random.default_rng(2)
        mosaics = [_synthetic_mosaic(rng, bright_block=True) for _ in range(6)]
        mask = build_from_mosaics(mosaics, cell_m=40.0, threshold=500.0)
        assert mask.threshold == 500.0
        assert mask.land.any()

    def test_dilation_grows_mask(self):
        rng = np.random.default_rng(3)
        mosaics = [_synthetic_mosaic(rng, bright_block=True) for _ in range(6)]
        m0 = build_from_mosaics(mosaics, cell_m=40.0, threshold=500.0, dilate=0)
        m2 = build_from_mosaics(mosaics, cell_m=40.0, threshold=500.0, dilate=2)
        assert m2.land.sum() > m0.land.sum()
        # dilation only adds cells, never removes
        assert not (m0.land & ~m2.land).any()

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            build_from_mosaics([])


class TestComputeTilesIntegration:
    def test_land_tiles_masked(self, tmp_path):
        """compute_tiles rejects tiles overlapping the mask's land."""
        from wamos_tpw.current import FrameCube, compute_tile_specs

        mask = _simple_mask()
        path = tmp_path / "mask.nc"
        mask.to_netcdf(path)

        n = 200
        spacing = 40.0
        x = (np.arange(n) - n / 2) * spacing  # +-4000 m
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
            "current.land_mask": str(path),
        }
        tiles = compute_tile_specs(cube, config=cfg)["tiles"]
        ne = [t for t in tiles if t["center_x"] > 500 and t["center_y"] > 500]
        sw = [t for t in tiles if t["center_x"] < -500 and t["center_y"] < -500]
        assert ne and sw
        assert all(t["masked"] for t in ne)
        assert not any(t["masked"] for t in sw)
