#! /usr/bin/env python3
#
# Tests for temporal compositing of surface current maps
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Tests for wamos_tpw.current_composite."""

from __future__ import annotations

import numpy as np
import pytest

from wamos_tpw.current import CurrentEstimate, CurrentMap
from wamos_tpw.current_composite import composite_current_maps, write_composite_netcdf


def make_map(
    estimates: list[CurrentEstimate],
    center_lat: float = 32.75,
    center_lon: float = -117.25,
    start: str = "2022-04-05T14:00:00",
    end: str = "2022-04-05T14:01:00",
) -> CurrentMap:
    """Build a minimal CurrentMap carrying the given estimates."""
    n = max(1, len(estimates))
    return CurrentMap(
        ux=np.full((1, n), np.nan),
        uy=np.full((1, n), np.nan),
        speed=np.full((1, n), np.nan),
        direction=np.full((1, n), np.nan),
        snr=np.full((1, n), np.nan),
        tile_x_centers=np.arange(n, dtype=float),
        tile_y_centers=np.zeros(1),
        depth=np.inf,
        center_lat=center_lat,
        center_lon=center_lon,
        start_time=np.datetime64(start),
        end_time=np.datetime64(end),
        estimates=estimates,
    )


def est(
    ux: float,
    uy: float,
    ux_err: float = 0.1,
    uy_err: float = 0.1,
    center_x: float = 0.0,
    center_y: float = 0.0,
    snr: float = 10.0,
) -> CurrentEstimate:
    return CurrentEstimate(
        ux=ux,
        uy=uy,
        speed=float(np.hypot(ux, uy)),
        direction=0.0,
        snr=snr,
        depth=np.inf,
        center_x=center_x,
        center_y=center_y,
        ux_err=ux_err,
        uy_err=uy_err,
    )


class TestCompositeMath:
    """Inverse-variance weighting and chi-square behavior."""

    def test_weighted_mean_and_error(self):
        """Two co-located estimates combine by inverse variance."""
        e1, e2 = 0.1, 0.2  # formal errors
        floor = 0.0
        maps = [
            make_map([est(1.0, -0.5, e1, e1)]),
            make_map([est(2.0, -1.0, e2, e2)]),
        ]
        comp = composite_current_maps(maps, grid_spacing=1000.0, min_obs=2, err_floor=floor)
        assert comp is not None

        w1, w2 = 1 / e1**2, 1 / e2**2
        expected_ux = (w1 * 1.0 + w2 * 2.0) / (w1 + w2)
        expected_err = np.sqrt(1 / (w1 + w2))

        populated = comp.n_obs == 2
        assert populated.sum() == 1
        np.testing.assert_allclose(comp.ux[populated][0], expected_ux, rtol=1e-12)
        np.testing.assert_allclose(comp.ux_err[populated][0], expected_err, rtol=1e-12)

    def test_chi2_zero_for_identical(self):
        """Identical estimates give chi2 ~ 0."""
        maps = [make_map([est(0.5, -0.3)]) for _ in range(3)]
        comp = composite_current_maps(maps, grid_spacing=1000.0)
        populated = comp.n_obs == 3
        np.testing.assert_allclose(comp.chi2[populated], 0.0, atol=1e-20)

    def test_chi2_large_for_inconsistent(self):
        """Estimates that disagree far beyond their errors give chi2 >> 1."""
        maps = [
            make_map([est(0.0, 0.0, 0.01, 0.01)]),
            make_map([est(1.0, 1.0, 0.01, 0.01)]),
        ]
        comp = composite_current_maps(maps, grid_spacing=1000.0, err_floor=0.0)
        populated = comp.n_obs == 2
        assert np.all(comp.chi2[populated] > 100.0)

    def test_min_obs_gate(self):
        """Cells with fewer than min_obs estimates are NaN."""
        maps = [make_map([est(0.5, -0.3)])]
        comp = composite_current_maps(maps, grid_spacing=1000.0, min_obs=2)
        assert np.all(np.isnan(comp.ux))
        assert comp.n_obs.max() == 1

    def test_non_finite_errors_dropped(self):
        """Estimates whose LS errors are NaN carry no weight."""
        maps = [
            make_map([est(0.5, -0.3, 0.1, 0.1)]),
            make_map([est(99.0, 99.0, float("nan"), float("nan"))]),
        ]
        comp = composite_current_maps(maps, grid_spacing=1000.0, min_obs=1)
        populated = comp.n_obs >= 1
        assert populated.sum() == 1
        np.testing.assert_allclose(comp.ux[populated][0], 0.5, rtol=1e-12)

    def test_min_snr_filter(self):
        """Low-SNR estimates are excluded."""
        maps = [
            make_map([est(0.5, -0.3, snr=10.0)]),
            make_map([est(99.0, 99.0, snr=0.1)]),
        ]
        comp = composite_current_maps(maps, grid_spacing=1000.0, min_obs=1, min_snr=1.0)
        populated = comp.n_obs >= 1
        np.testing.assert_allclose(comp.ux[populated][0], 0.5, rtol=1e-12)

    def test_empty_input(self):
        assert composite_current_maps([]) is None

    def test_no_usable_estimates(self):
        maps = [make_map([est(0.5, -0.3, float("nan"), float("nan"))])]
        assert composite_current_maps(maps, grid_spacing=1000.0) is None


class TestCompositeGeoreferencing:
    """Spatial binning across maps with different centers."""

    def test_separate_locations_separate_cells(self):
        """Estimates 2 km apart land in different composite cells."""
        maps = [
            make_map([est(0.5, 0.0, center_x=0.0)]),
            make_map([est(1.5, 0.0, center_x=2000.0)]),
        ]
        comp = composite_current_maps(maps, grid_spacing=1000.0, min_obs=1)
        populated = comp.n_obs >= 1
        assert populated.sum() == 2
        values = np.sort(comp.ux[populated])
        np.testing.assert_allclose(values, [0.5, 1.5], rtol=1e-12)

    def test_shifted_map_centers_colocate(self):
        """The same earth location from maps with different cube centers
        composites into one cell."""
        # Map 2's center is ~500 m east of map 1's: estimates at
        # center_x=+250 (map 1) and center_x=-250 (map 2) are co-located
        from wamos_tpw.grid import _DEG2M

        lat = 32.75
        dlon = 500.0 / (_DEG2M * np.cos(np.deg2rad(lat)))
        maps = [
            make_map([est(1.0, 0.0, center_x=250.0)], center_lon=-117.25),
            make_map([est(2.0, 0.0, center_x=-250.0)], center_lon=-117.25 + dlon),
        ]
        comp = composite_current_maps(maps, grid_spacing=1000.0, min_obs=2)
        populated = comp.n_obs == 2
        assert populated.sum() == 1


class TestCompositeNetcdf:
    """NetCDF output round trip."""

    def test_write_and_read(self, tmp_path):
        xr = pytest.importorskip("xarray")

        maps = [make_map([est(0.5, -0.3)]) for _ in range(2)]
        comp = composite_current_maps(maps, grid_spacing=1000.0)
        path = write_composite_netcdf(comp, str(tmp_path))
        assert path

        ds = xr.open_dataset(path)
        try:
            for var in ("ux", "uy", "ux_err", "uy_err", "n_obs", "chi2"):
                assert var in ds
            assert ds.attrs["n_input_maps"] == 2
        finally:
            ds.close()
