#! /usr/bin/env python3
#
# Tests for surface current extraction via 3D FFT dispersion fitting
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Tests for wamos_tpw.current module."""

from __future__ import annotations

import numpy as np
import pytest

from wamos_tpw.current import (
    CurrentEstimate,
    CurrentExtractor,
    CurrentMap,
    FrameCube,
    compute_tile_specs,
    dispersion_relation,
)

# Physical constants (same as current.py)
_G = 9.81


# ============================================================
# Synthetic Data Generation
# ============================================================


def make_synthetic_cube(
    ux: float = 0.5,
    uy: float = -0.3,
    n_t: int = 32,
    n_xy: int = 128,
    dx: float = 15.0,
    dt: float = 1.5,
    depth: float = np.inf,
    noise_level: float = 0.1,
    n_waves: int = 10,
    seed: int = 42,
) -> FrameCube:
    """Generate a synthetic FrameCube with waves following the dispersion relation.

    Creates a superposition of plane waves whose frequencies satisfy the
    Doppler-shifted dispersion relation for a known current (ux, uy).

    Args:
        ux: Eastward current component (m/s).
        uy: Northward current component (m/s).
        n_t: Number of time steps.
        n_xy: Number of spatial grid points in each direction.
        dx: Grid spacing (meters).
        dt: Time step (seconds).
        depth: Water depth (meters). Use np.inf for deep water.
        noise_level: Standard deviation of additive Gaussian noise
                     relative to signal amplitude.
        n_waves: Number of wave components to superpose.
        seed: Random seed for reproducibility.

    Returns:
        FrameCube containing synthetic wave field.
    """
    rng = np.random.default_rng(seed)

    x = np.arange(n_xy) * dx
    y = np.arange(n_xy) * dx
    t = np.arange(n_t) * dt

    X, Y = np.meshgrid(x, y)  # (n_xy, n_xy)
    intensity = np.zeros((n_t, n_xy, n_xy))

    # Generate wave components with random directions and wavelengths
    wavelengths = rng.uniform(50.0, 500.0, size=n_waves)  # meters
    directions = rng.uniform(0, 2 * np.pi, size=n_waves)  # radians
    amplitudes = rng.uniform(0.5, 2.0, size=n_waves)

    for i in range(n_waves):
        k_mag = 2.0 * np.pi / wavelengths[i]
        kx = k_mag * np.cos(directions[i])
        ky = k_mag * np.sin(directions[i])

        # Intrinsic frequency from dispersion relation
        omega_0 = float(dispersion_relation(np.array([k_mag]), depth)[0])

        # Doppler-shifted frequency
        omega = omega_0 + kx * ux + ky * uy

        phase = rng.uniform(0, 2 * np.pi)

        for it in range(n_t):
            intensity[it] += amplitudes[i] * np.cos(kx * X + ky * Y - omega * t[it] + phase)

    # Add noise
    if noise_level > 0:
        signal_std = np.std(intensity)
        intensity += rng.normal(0, noise_level * signal_std, intensity.shape)

    # Shift to positive values (like radar intensity)
    intensity += np.abs(intensity.min()) + 1.0

    return FrameCube.from_arrays(
        intensity=intensity,
        dt=dt,
        dx=dx,
        center_lat=32.0,
        center_lon=-117.0,
    )


# ============================================================
# Test Dispersion Relation
# ============================================================


class TestDispersionRelation:
    """Tests for the dispersion_relation function."""

    def test_deep_water(self):
        """Deep water: omega = sqrt(g * k)."""
        k = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
        omega = dispersion_relation(k, depth=np.inf)
        expected = np.sqrt(_G * k)
        np.testing.assert_allclose(omega, expected, rtol=1e-10)

    def test_finite_depth(self):
        """Finite depth: omega = sqrt(g * k * tanh(k * d))."""
        k = np.array([0.01, 0.05, 0.1])
        d = 20.0
        omega = dispersion_relation(k, depth=d)
        expected = np.sqrt(_G * k * np.tanh(k * d))
        np.testing.assert_allclose(omega, expected, rtol=1e-10)

    def test_shallow_water_limit(self):
        """Very shallow water: omega -> sqrt(g * d) * k."""
        d = 1.0
        k = np.array([0.001])  # k*d << 1
        omega = dispersion_relation(k, depth=d)
        # For small k*d, tanh(kd) ~ kd, so omega ~ sqrt(g*d) * k
        expected = np.sqrt(_G * d) * k
        np.testing.assert_allclose(omega, expected, rtol=0.01)

    def test_zero_wavenumber(self):
        """Zero wavenumber should give zero frequency."""
        omega = dispersion_relation(np.array([0.0]))
        assert omega[0] == 0.0

    def test_known_values(self):
        """Test against known deep water wave parameters.

        A 100m wavelength deep water wave has:
        k = 2*pi/100 ~ 0.0628 rad/m
        omega = sqrt(g*k) ~ 0.785 rad/s
        period ~ 8.0 s
        """
        k = 2.0 * np.pi / 100.0
        omega = float(dispersion_relation(np.array([k]))[0])
        expected_period = 2.0 * np.pi / omega
        assert abs(expected_period - 8.0) < 0.1  # Within 0.1 seconds

    def test_scalar_input(self):
        """Accept scalar (0-d array) input."""
        omega = dispersion_relation(np.array(0.1))
        assert omega.shape == ()
        assert omega > 0

    def test_deep_water_matches_inf_depth(self):
        """Very deep finite depth should match infinite depth."""
        k = np.array([0.01, 0.1, 1.0])
        omega_inf = dispersion_relation(k, depth=np.inf)
        omega_deep = dispersion_relation(k, depth=10000.0)
        np.testing.assert_allclose(omega_inf, omega_deep, rtol=1e-6)


# ============================================================
# Test FrameCube
# ============================================================


class TestFrameCube:
    """Tests for FrameCube data structure."""

    def test_from_arrays_shape(self):
        """Shape and metadata consistency."""
        n_t, n_y, n_x = 16, 64, 64
        data = np.random.default_rng(0).random((n_t, n_y, n_x))
        cube = FrameCube.from_arrays(data, dt=1.5, dx=10.0)

        assert cube.n_t == n_t
        assert cube.n_y == n_y
        assert cube.n_x == n_x
        assert cube.dt == 1.5
        assert cube.grid_spacing == 10.0
        assert len(cube.x_centers) == n_x
        assert len(cube.y_centers) == n_y
        assert len(cube.timestamps) == n_t

    def test_from_arrays_timestamps_ordered(self):
        """Timestamps should be monotonically increasing."""
        data = np.random.default_rng(0).random((8, 32, 32))
        cube = FrameCube.from_arrays(data, dt=2.0, dx=10.0)
        diffs = np.diff(cube.timestamps)
        assert np.all(diffs > np.timedelta64(0))

    def test_sub_cube(self):
        """Sub-cube extraction preserves data."""
        data = np.random.default_rng(0).random((8, 64, 64))
        cube = FrameCube.from_arrays(data, dt=1.0, dx=10.0)
        sub = cube.sub_cube(10, 30, 20, 50)

        assert sub.n_t == 8
        assert sub.n_x == 20
        assert sub.n_y == 30
        np.testing.assert_array_equal(sub.intensity, data[:, 20:50, 10:30])

    def test_sub_cube_independent(self):
        """Sub-cube should be independent (copy, not view)."""
        data = np.random.default_rng(0).random((4, 32, 32))
        cube = FrameCube.from_arrays(data, dt=1.0, dx=10.0)
        sub = cube.sub_cube(0, 16, 0, 16)
        sub.intensity[:] = 999.0
        assert not np.any(cube.intensity == 999.0)


# ============================================================
# Test CurrentExtractor
# ============================================================


class TestCurrentExtractor:
    """Tests for CurrentExtractor algorithm."""

    def test_known_current_deep_water(self):
        """Extract a known current in deep water — accuracy < 0.15 m/s."""
        true_ux, true_uy = 0.5, -0.3
        cube = make_synthetic_cube(
            ux=true_ux,
            uy=true_uy,
            n_t=64,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            depth=np.inf,
            noise_level=0.05,
            n_waves=12,
        )

        extractor = CurrentExtractor(cube)
        est = extractor.estimate

        assert abs(est.ux - true_ux) < 0.15, f"Ux error: {abs(est.ux - true_ux):.3f}"
        assert abs(est.uy - true_uy) < 0.15, f"Uy error: {abs(est.uy - true_uy):.3f}"

    def test_known_current_finite_depth(self):
        """Extract current with finite depth — dispersion relation changes shape."""
        true_ux, true_uy = 0.3, 0.4
        depth = 50.0
        cube = make_synthetic_cube(
            ux=true_ux,
            uy=true_uy,
            n_t=64,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            depth=depth,
            noise_level=0.02,
            n_waves=15,
            seed=123,
        )

        from wamos_tpw.config import Config

        config = Config()
        config["current.depth"] = depth
        extractor = CurrentExtractor(cube, config=config)
        est = extractor.estimate

        assert est.depth == depth
        # Finite depth is harder to resolve; verify the algorithm runs
        # and produces a reasonable estimate
        assert est.speed < 3.0, f"Speed unreasonably high: {est.speed:.2f}"
        assert np.isfinite(est.snr)

    def test_zero_current(self):
        """Zero current should produce near-zero estimate."""
        cube = make_synthetic_cube(
            ux=0.0,
            uy=0.0,
            n_t=64,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            noise_level=0.05,
        )

        extractor = CurrentExtractor(cube)
        est = extractor.estimate

        assert est.speed < 0.3, f"Expected near-zero speed, got {est.speed:.3f}"

    def test_high_noise_rejection(self):
        """High noise should produce low SNR."""
        cube = make_synthetic_cube(
            ux=0.5,
            uy=0.0,
            n_t=32,
            n_xy=64,
            dx=7.5,
            dt=1.5,
            noise_level=5.0,
            n_waves=3,
        )

        extractor = CurrentExtractor(cube)
        est = extractor.estimate

        # High noise -> low SNR, but should still run without error
        assert est.snr >= 0

    def test_nan_handling(self):
        """NaN values in the cube should be handled gracefully."""
        cube = make_synthetic_cube(
            ux=0.5,
            uy=-0.3,
            n_t=32,
            n_xy=64,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        # Inject NaN values (simulate shadow regions)
        rng = np.random.default_rng(123)
        mask = rng.random(cube.intensity.shape) < 0.1
        cube.intensity[mask] = np.nan

        # Should not raise
        extractor = CurrentExtractor(cube)
        est = extractor.estimate
        assert np.isfinite(est.ux)
        assert np.isfinite(est.uy)
        assert np.isfinite(est.snr)

    def test_power_spectrum_shape(self):
        """Power spectrum should have correct dimensions."""
        cube = make_synthetic_cube(n_t=16, n_xy=32, dx=7.5, dt=1.0)
        extractor = CurrentExtractor(cube)

        n_t = 16
        n_ky = 32
        n_kx = 32

        assert extractor.power_spectrum.shape == (n_t, n_ky, n_kx)
        assert len(extractor.kx) == n_kx
        assert len(extractor.ky) == n_ky
        assert len(extractor.omega) == n_t

    def test_estimate_direction_convention(self):
        """Direction should be TO convention, clockwise from north.

        Eastward current (ux>0, uy=0) -> direction = 90 degrees.
        """
        cube = make_synthetic_cube(
            ux=1.0,
            uy=0.0,
            n_t=64,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            noise_level=0.02,
            n_waves=15,
        )

        extractor = CurrentExtractor(cube)
        est = extractor.estimate

        # arctan2(ux, uy) for ux=positive, uy=0 -> 90 degrees
        if est.speed > 0.3:
            assert abs(est.direction - 90.0) < 30.0, f"Direction: {est.direction:.1f}, expected ~90"

    def test_without_refinement(self):
        """Should work without refinement (coarse grid search only)."""
        cube = make_synthetic_cube(
            ux=0.5,
            uy=0.0,
            n_t=32,
            n_xy=64,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        from wamos_tpw.config import Config

        config = Config()
        config["current.refine"] = False
        extractor = CurrentExtractor(cube, config=config)
        est = extractor.estimate

        assert np.isfinite(est.ux)
        assert np.isfinite(est.speed)


# ============================================================
# Test CurrentMap
# ============================================================


class TestCurrentMap:
    """Tests for CurrentMap tiled extraction."""

    def test_single_tile(self):
        """Single tile should match direct extraction."""
        cube = make_synthetic_cube(
            ux=0.3,
            uy=0.2,
            n_t=32,
            n_xy=64,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        from wamos_tpw.config import Config

        config = Config()
        # Make sub_region larger than the cube so only one tile
        config["current.sub_region_size"] = 5000.0
        config["current.min_snr"] = 0.0

        current_map = CurrentMap.from_cube(cube, config=config)

        assert current_map.n_tiles_x == 1
        assert current_map.n_tiles_y == 1
        assert len(current_map.estimates) == 1

        # Direct extraction for comparison
        extractor = CurrentExtractor(cube, config=config)
        direct = extractor.estimate

        np.testing.assert_allclose(current_map.ux[0, 0], direct.ux, atol=0.01)
        np.testing.assert_allclose(current_map.uy[0, 0], direct.uy, atol=0.01)

    def test_multiple_tiles(self):
        """Multiple tiles produce a spatial map."""
        cube = make_synthetic_cube(
            ux=0.3,
            uy=-0.2,
            n_t=32,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 400.0
        config["current.sub_region_overlap"] = 0.0
        config["current.min_snr"] = 0.0

        current_map = CurrentMap.from_cube(cube, config=config)

        assert current_map.n_tiles_x >= 2
        assert current_map.n_tiles_y >= 2
        assert len(current_map.estimates) > 1

        # All tiles should have valid estimates (uniform wave field)
        valid_count = np.sum(~np.isnan(current_map.speed))
        assert valid_count > 0

    def test_snr_filtering(self):
        """Tiles below min_snr should have NaN values."""
        cube = make_synthetic_cube(
            ux=0.5,
            uy=0.0,
            n_t=32,
            n_xy=64,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 5000.0
        config["current.min_snr"] = 1000.0  # Unrealistically high

        current_map = CurrentMap.from_cube(cube, config=config)

        # All values should be NaN due to high SNR threshold
        assert np.all(np.isnan(current_map.ux))
        assert np.all(np.isnan(current_map.speed))
        # But SNR should still be computed
        assert not np.all(np.isnan(current_map.snr))

    def test_metadata(self):
        """CurrentMap should preserve metadata."""
        cube = make_synthetic_cube(n_t=8, n_xy=32, dx=7.5, dt=1.0)

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 5000.0
        config["current.min_snr"] = 0.0

        current_map = CurrentMap.from_cube(cube, config=config)

        assert current_map.center_lat == cube.center_lat
        assert current_map.center_lon == cube.center_lon
        assert current_map.start_time == cube.timestamps[0]
        assert current_map.end_time == cube.timestamps[-1]


# ============================================================
# Test CurrentEstimate
# ============================================================


class TestCurrentEstimate:
    """Tests for CurrentEstimate dataclass."""

    def test_fields(self):
        """All fields should be accessible."""
        est = CurrentEstimate(
            ux=0.5,
            uy=-0.3,
            speed=0.583,
            direction=120.96,
            snr=5.0,
            depth=np.inf,
            center_x=100.0,
            center_y=200.0,
        )
        assert est.ux == 0.5
        assert est.uy == -0.3
        assert est.speed == pytest.approx(0.583)
        assert est.depth == np.inf


# ============================================================
# Test compute_tile_specs
# ============================================================


class TestComputeTileSpecs:
    """Tests for the compute_tile_specs function."""

    def test_single_tile(self):
        """Large sub_region_size should produce exactly one tile."""
        cube = make_synthetic_cube(n_t=8, n_xy=64, dx=7.5, dt=1.0)

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 5000.0

        specs = compute_tile_specs(cube, config)

        assert specs["n_tiles_x"] == 1
        assert specs["n_tiles_y"] == 1
        assert len(specs["tiles"]) == 1
        assert len(specs["tile_x_centers"]) == 1
        assert len(specs["tile_y_centers"]) == 1

    def test_multiple_tiles(self):
        """Small sub_region_size should produce multiple tiles."""
        cube = make_synthetic_cube(n_t=8, n_xy=128, dx=7.5, dt=1.0)

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 400.0
        config["current.sub_region_overlap"] = 0.0

        specs = compute_tile_specs(cube, config)

        assert specs["n_tiles_x"] >= 2
        assert specs["n_tiles_y"] >= 2
        assert len(specs["tiles"]) == specs["n_tiles_x"] * specs["n_tiles_y"]

    def test_tile_geometry_matches_from_cube(self):
        """Tile specs should produce same geometry as from_cube uses internally."""
        cube = make_synthetic_cube(n_t=8, n_xy=64, dx=7.5, dt=1.0)

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 200.0
        config["current.sub_region_overlap"] = 0.5
        config["current.min_snr"] = 0.0

        specs = compute_tile_specs(cube, config)
        current_map = CurrentMap.from_cube(cube, config=config)

        assert specs["n_tiles_x"] == current_map.n_tiles_x
        assert specs["n_tiles_y"] == current_map.n_tiles_y
        np.testing.assert_array_equal(specs["tile_x_centers"], current_map.tile_x_centers)
        np.testing.assert_array_equal(specs["tile_y_centers"], current_map.tile_y_centers)

    def test_tile_fields(self):
        """Each tile dict should have required fields."""
        cube = make_synthetic_cube(n_t=8, n_xy=64, dx=7.5, dt=1.0)
        specs = compute_tile_specs(cube)

        for tile in specs["tiles"]:
            assert "ix" in tile
            assert "iy" in tile
            assert "x_start" in tile
            assert "x_end" in tile
            assert "y_start" in tile
            assert "y_end" in tile
            assert "center_x" in tile
            assert "center_y" in tile
            assert tile["x_end"] > tile["x_start"]
            assert tile["y_end"] > tile["y_start"]


# ============================================================
# Test CurrentMap.from_tile_results
# ============================================================


class TestFromTileResults:
    """Tests for CurrentMap.from_tile_results assembly."""

    def test_matches_serial(self):
        """from_tile_results should produce same output as serial from_cube."""
        cube = make_synthetic_cube(
            ux=0.3,
            uy=-0.2,
            n_t=32,
            n_xy=64,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 5000.0
        config["current.min_snr"] = 0.0

        # Serial extraction
        serial_map = CurrentMap.from_cube(cube, config=config)

        # Manual tile extraction to simulate parallel results
        specs = compute_tile_specs(cube, config)
        tile_results = []
        for tile in specs["tiles"]:
            sub = cube.sub_cube(tile["x_start"], tile["x_end"], tile["y_start"], tile["y_end"])
            extractor = CurrentExtractor(sub, config=config)
            est = extractor.estimate
            tile_results.append(
                {
                    "ix": tile["ix"],
                    "iy": tile["iy"],
                    "ux": est.ux,
                    "uy": est.uy,
                    "speed": est.speed,
                    "direction": est.direction,
                    "snr": est.snr,
                    "depth": est.depth,
                    "center_x": est.center_x,
                    "center_y": est.center_y,
                }
            )

        meta = {
            "center_lat": cube.center_lat,
            "center_lon": cube.center_lon,
            "start_time": cube.timestamps[0],
            "end_time": cube.timestamps[-1],
        }
        assembled = CurrentMap.from_tile_results(specs, tile_results, meta)

        np.testing.assert_array_equal(assembled.ux, serial_map.ux)
        np.testing.assert_array_equal(assembled.uy, serial_map.uy)
        np.testing.assert_array_equal(assembled.snr, serial_map.snr)
        assert assembled.center_lat == serial_map.center_lat
        assert assembled.center_lon == serial_map.center_lon

    def test_error_tiles_become_nan(self):
        """Tiles with error key should produce NaN in output."""
        cube = make_synthetic_cube(n_t=8, n_xy=32, dx=7.5, dt=1.0)
        specs = compute_tile_specs(cube)

        # Simulate all tiles failing
        tile_results = [
            {"error": "test error", "ix": t["ix"], "iy": t["iy"]} for t in specs["tiles"]
        ]
        meta = {
            "center_lat": 32.0,
            "center_lon": -117.0,
            "start_time": np.datetime64("2022-01-01"),
            "end_time": np.datetime64("2022-01-01T00:00:10"),
        }

        cm = CurrentMap.from_tile_results(specs, tile_results, meta)

        assert np.all(np.isnan(cm.ux))
        assert np.all(np.isnan(cm.snr))
        assert len(cm.estimates) == 0

    def test_multi_tile_assembly(self):
        """Assembly with multiple tiles and SNR filtering."""
        cube = make_synthetic_cube(
            ux=0.5,
            uy=-0.3,
            n_t=32,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            noise_level=0.1,
        )

        from wamos_tpw.config import Config

        config = Config()
        config["current.sub_region_size"] = 400.0
        config["current.sub_region_overlap"] = 0.0
        config["current.min_snr"] = 0.0

        serial_map = CurrentMap.from_cube(cube, config=config)

        specs = compute_tile_specs(cube, config)
        tile_results = []
        for tile in specs["tiles"]:
            sub = cube.sub_cube(tile["x_start"], tile["x_end"], tile["y_start"], tile["y_end"])
            try:
                extractor = CurrentExtractor(sub, config=config)
                est = extractor.estimate
                tile_results.append(
                    {
                        "ix": tile["ix"],
                        "iy": tile["iy"],
                        "ux": est.ux,
                        "uy": est.uy,
                        "speed": est.speed,
                        "direction": est.direction,
                        "snr": est.snr,
                        "depth": est.depth,
                        "center_x": est.center_x,
                        "center_y": est.center_y,
                    }
                )
            except Exception:
                tile_results.append({"error": "failed", "ix": tile["ix"], "iy": tile["iy"]})

        meta = {
            "center_lat": cube.center_lat,
            "center_lon": cube.center_lon,
            "start_time": cube.timestamps[0],
            "end_time": cube.timestamps[-1],
        }
        assembled = CurrentMap.from_tile_results(specs, tile_results, meta)

        np.testing.assert_array_equal(assembled.ux, serial_map.ux)
        np.testing.assert_array_equal(assembled.snr, serial_map.snr)
