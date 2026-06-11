#! /usr/bin/env python3
#
# Tests for synthetic cubes with prescribed spatially varying currents
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Tests for wamos_tpw.synthetic."""

from __future__ import annotations

import numpy as np

from wamos_tpw.synthetic import (
    current_front,
    gaussian_eddy,
    linear_shear,
    make_current_cube,
    sinusoidal_current,
    uniform_current,
)


class TestFeatureFields:
    """Current feature field functions."""

    def test_uniform(self):
        u = uniform_current(0.5, -0.3)
        x = np.linspace(-1000, 1000, 5)
        ux, uy = u(x, x)
        np.testing.assert_allclose(ux, 0.5)
        np.testing.assert_allclose(uy, -0.3)

    def test_linear_shear(self):
        u = linear_shear(dudx=1e-4, ux0=0.1)
        x = np.array([0.0, 1000.0])
        y = np.zeros(2)
        ux, uy = u(x, y)
        np.testing.assert_allclose(ux, [0.1, 0.2])
        np.testing.assert_allclose(uy, 0.0)

    def test_sinusoid_along_vs_across(self):
        """'along' is parallel to the variation direction, 'across' normal."""
        u_along = sinusoidal_current(0.4, 2000.0, direction_deg=90.0, component="along")
        u_across = sinusoidal_current(0.4, 2000.0, direction_deg=90.0, component="across")

        x = np.array([500.0])  # quarter wavelength: sin = 1
        y = np.array([0.0])
        ux_a, uy_a = u_along(x, y)
        ux_c, uy_c = u_across(x, y)

        np.testing.assert_allclose(ux_a, 0.4, atol=1e-12)
        np.testing.assert_allclose(uy_a, 0.0, atol=1e-12)
        np.testing.assert_allclose(ux_c, 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(uy_c), 0.4, atol=1e-12)

    def test_eddy_peak_at_radius(self):
        u = gaussian_eddy(peak_speed=0.5, radius=1000.0)
        x = np.array([1000.0, 0.0])
        y = np.array([0.0, 0.0])
        ux, uy = u(x, y)
        speed = np.hypot(ux, uy)
        np.testing.assert_allclose(speed[0], 0.5, rtol=1e-12)  # at r = R
        np.testing.assert_allclose(speed[1], 0.0, atol=1e-12)  # at center

    def test_eddy_azimuthal(self):
        """Velocity is perpendicular to the radius vector."""
        u = gaussian_eddy(peak_speed=0.5, radius=1000.0)
        x = np.array([800.0])
        y = np.array([600.0])
        ux, uy = u(x, y)
        radial = ux * x + uy * y
        np.testing.assert_allclose(radial, 0.0, atol=1e-9)

    def test_front_limits(self):
        u = current_front((0.1, 0.0), (0.6, -0.2), front_x=0.0, width=200.0)
        x = np.array([-5000.0, 5000.0])
        y = np.zeros(2)
        ux, uy = u(x, y)
        np.testing.assert_allclose(ux, [0.1, 0.6], atol=1e-6)
        np.testing.assert_allclose(uy, [0.0, -0.2], atol=1e-6)


class TestCubeSynthesis:
    """Cube generation correctness."""

    def test_phasor_recurrence_matches_direct(self):
        """The phasor-recurrence synthesis equals direct cosine evaluation."""
        n_t, n_xy, dx, dt = 6, 16, 15.0, 1.5
        u = linear_shear(dudx=2e-4, ux0=0.3)
        seed = 7

        cube = make_current_cube(
            u, n_t=n_t, n_xy=n_xy, dx=dx, dt=dt, n_waves=5, noise_level=0.0, seed=seed
        )

        # Re-derive the same wave parameters with the same RNG sequence
        rng = np.random.default_rng(seed)
        coords = (np.arange(n_xy) - (n_xy - 1) / 2) * dx
        xg, yg = np.meshgrid(coords, coords)
        ux_f, uy_f = u(xg, yg)

        k = np.exp(rng.uniform(np.log(2 * np.pi / 600.0), np.log(2 * np.pi / 60.0), 5))
        theta = np.deg2rad(60.0) + rng.normal(0, 0.8, 5)
        amp = k**-1.25
        amp /= amp.max()
        amp *= rng.rayleigh(1.0, 5)
        kx = k * np.sin(theta)
        ky = k * np.cos(theta)
        omega0 = np.sqrt(9.81 * k)
        phase0 = rng.uniform(0, 2 * np.pi, 5)

        direct = np.zeros((n_t, n_xy, n_xy))
        for i in range(5):
            omega_local = omega0[i] + kx[i] * ux_f + ky[i] * uy_f
            for it in range(n_t):
                direct[it] += amp[i] * np.cos(
                    kx[i] * xg + ky[i] * yg + phase0[i] - omega_local * it * dt
                )
        direct += np.abs(direct.min()) + 1.0

        np.testing.assert_allclose(cube.intensity, direct, atol=1e-9)

    def test_uniform_current_recovered(self):
        """A uniform prescribed current is recovered by the extractor."""
        from wamos_tpw.current import CurrentExtractor

        true_ux, true_uy = 0.5, -0.3
        cube = make_current_cube(
            uniform_current(true_ux, true_uy),
            n_t=64,
            n_xy=128,
            dx=7.5,
            dt=1.5,
            n_waves=200,
            wavelength_range=(40.0, 600.0),
            seed=1,
        )
        est = CurrentExtractor(cube).estimate
        assert abs(est.ux - true_ux) < 0.15, f"ux={est.ux:.3f}"
        assert abs(est.uy - true_uy) < 0.15, f"uy={est.uy:.3f}"

    def test_metadata(self):
        cube = make_current_cube(uniform_current(0, 0), n_t=4, n_xy=32, dx=10.0, dt=2.0)
        assert cube.n_t == 4
        assert cube.n_x == 32
        assert cube.dt == 2.0
        assert cube.grid_spacing == 10.0
        # Domain centered on zero
        np.testing.assert_allclose(cube.x_centers.mean(), 0.0, atol=1e-9)


class TestResolutionRegression:
    """Slow regression: the multi-scale field inversion resolves a current
    feature that single large tiles cannot."""

    import pytest

    @pytest.mark.slow
    def test_sinusoid_recovery_beats_tiles(self):
        from wamos_tpw.config import Config
        from wamos_tpw.current import CurrentExtractor, compute_multiscale_tile_specs
        from wamos_tpw.current_field import solve_current_field

        wavelength = 1800.0
        amp = 0.3
        u_func = sinusoidal_current(amp, wavelength, 90.0, "across")

        cube = make_current_cube(
            u_func,
            n_t=64,
            n_xy=384,
            dx=15.0,
            dt=1.5,
            n_waves=200,
            seed=3,
        )

        config = Config()
        config["current.window_sizes"] = [1800.0, 900.0]
        config["current.sub_region_overlap"] = 0.5

        specs = compute_multiscale_tile_specs(cube, config)
        estimates = []
        for tile in specs["tiles"]:
            if tile["masked"]:
                continue
            sub = cube.sub_cube(tile["x_start"], tile["x_end"], tile["y_start"], tile["y_end"])
            try:
                estimates.append(CurrentExtractor(sub, config=config).estimate)
            except Exception:
                continue

        fld = solve_current_field(estimates, field_spacing=225.0, correlation_length=450.0)
        assert fld is not None

        covered = fld.coverage > 0
        xg, _ = np.meshgrid(fld.x_centers, fld.y_centers)
        tux, tuy = u_func(xg, np.zeros_like(xg))

        # Pattern transfer of the uy component (the sinusoid is across)
        ta = tuy[covered] - tuy[covered].mean()
        ra = fld.uy[covered] - fld.uy[covered].mean()
        transfer = float(np.sum(ra * ta) / np.sum(ta**2))

        # 1800 m windows alone see a near-zero boxcar response at 1800 m
        # wavelength; the joint inversion with 900 m windows must recover
        # a substantial fraction of the amplitude.
        assert transfer > 0.4, f"transfer {transfer:.2f}"
