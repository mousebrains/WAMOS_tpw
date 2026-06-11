#! /usr/bin/env python3
#
# Tests for joint regularized current field inversion
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Tests for wamos_tpw.current_field."""

from __future__ import annotations

import numpy as np
import pytest

from wamos_tpw.current import CurrentEstimate
from wamos_tpw.current_field import CurrentField, solve_current_field, write_field_netcdf


def obs(
    ux: float,
    uy: float,
    center_x: float,
    center_y: float,
    window_size: float,
    err: float = 0.05,
    cov: float = 0.0,
    snr: float = 10.0,
) -> CurrentEstimate:
    """Build a window observation for the inversion."""
    return CurrentEstimate(
        ux=ux,
        uy=uy,
        speed=float(np.hypot(ux, uy)),
        direction=0.0,
        snr=snr,
        depth=np.inf,
        center_x=center_x,
        center_y=center_y,
        ux_err=err,
        uy_err=err,
        cov_uxuy=cov,
        window_size=window_size,
    )


def make_tiled_obs(
    u_func,
    window_size: float,
    domain: float = 4000.0,
    stride: float | None = None,
    err: float = 0.05,
    noise: float = 0.0,
    seed: int = 0,
) -> list[CurrentEstimate]:
    """Tile a square domain with window observations of a current field.

    Each observation's value is the Hann^2-weighted footprint average of
    ``u_func`` (matching both the inversion's forward model and the
    empirical sensing kernel of the dispersion estimator), optionally
    perturbed by Gaussian noise of standard deviation ``noise``.
    """
    if stride is None:
        stride = window_size / 2
    half_dom = domain / 2
    centers = np.arange(-half_dom + window_size / 2, half_dom - window_size / 2 + 1, stride)
    rng = np.random.default_rng(seed)

    out = []
    s = np.linspace(-window_size / 2, window_size / 2, 17)
    sx, sy = np.meshgrid(s, s)
    w = (np.cos(np.pi * sx / window_size) * np.cos(np.pi * sy / window_size)) ** 4
    w /= w.sum()
    for cy in centers:
        for cx in centers:
            ux, uy = u_func(cx + sx, cy + sy)
            out.append(
                obs(
                    float(np.sum(w * ux)) + rng.normal(0, noise),
                    float(np.sum(w * uy)) + rng.normal(0, noise),
                    cx,
                    cy,
                    window_size,
                    err,
                )
            )
    return out


class TestUniformField:
    """Uniform currents must be recovered (nearly) exactly."""

    def test_single_scale(self):
        def u(x, y):
            return 0.5 * np.ones_like(x), -0.3 * np.ones_like(x)

        estimates = make_tiled_obs(u, window_size=2000.0)
        fld = solve_current_field(estimates, field_spacing=250.0)
        assert fld is not None

        covered = fld.coverage > 0
        np.testing.assert_allclose(fld.ux[covered], 0.5, atol=0.01)
        np.testing.assert_allclose(fld.uy[covered], -0.3, atol=0.01)
        assert fld.chi2 < 1.0

    def test_multi_scale(self):
        def u(x, y):
            return 0.5 * np.ones_like(x), -0.3 * np.ones_like(x)

        estimates = make_tiled_obs(u, window_size=2000.0) + make_tiled_obs(u, window_size=1000.0)
        fld = solve_current_field(estimates, field_spacing=250.0)
        covered = fld.coverage > 0
        np.testing.assert_allclose(fld.ux[covered], 0.5, atol=0.01)
        np.testing.assert_allclose(fld.uy[covered], -0.3, atol=0.01)


class TestVaryingField:
    """Spatially varying currents are resolved beyond a single window size."""

    def test_linear_shear(self):
        """A linear shear is exactly representable and footprint-consistent."""
        g = 2e-4  # 0.2 m/s per km

        def u(x, y):
            return g * x, -0.5 * g * y

        estimates = make_tiled_obs(u, window_size=1000.0, domain=6000.0)
        fld = solve_current_field(estimates, field_spacing=250.0, correlation_length=500.0)
        covered = fld.coverage > 0

        xg, yg = np.meshgrid(fld.x_centers, fld.y_centers)
        true_ux = g * xg
        true_uy = -0.5 * g * yg
        err = np.abs(fld.ux - true_ux)[covered]
        assert np.median(err) < 0.02, f"median shear error {np.median(err):.3f}"
        err_y = np.abs(fld.uy - true_uy)[covered]
        assert np.median(err_y) < 0.02

    def test_sinusoid_better_resolved_with_small_windows(self):
        """A 1.3 km sinusoid is invisible to 2 km windows at 1 km stride
        (below their Nyquist sampling, and Hann^2 response only 0.37) but
        well resolved when 1 km windows at 500 m stride contribute.

        The wavelength is chosen incommensurate with the window strides so
        the observations do not all land on the sinusoid's nodes. The
        smoothness prior is set to allow gradients of the feature's scale
        (sigma_prior / L_c = 0.002 per m ~ the feature's peak gradient);
        the prior is the resolution knob of the inversion.
        """
        wavelength = 1300.0
        amp = 0.4
        noise = 0.05
        phase = 0.4

        def u(x, y):
            s = amp * np.sin(2 * np.pi * x / wavelength + phase)
            return s, np.zeros_like(x)

        big = make_tiled_obs(u, window_size=2000.0, domain=8000.0, noise=noise, seed=1)
        small = make_tiled_obs(u, window_size=1000.0, domain=8000.0, noise=noise, seed=2)

        kwargs = {
            "field_spacing": 250.0,
            "correlation_length": 250.0,
            "sigma_prior": 0.5,
        }
        fld_big = solve_current_field(big, **kwargs)
        fld_both = solve_current_field(big + small, **kwargs)

        def recovered_amplitude(fld: CurrentField) -> float:
            covered = fld.coverage > 0
            xg, _ = np.meshgrid(fld.x_centers, fld.y_centers)
            ref = np.sin(2 * np.pi * xg / wavelength + phase)
            num = np.sum(fld.ux[covered] * ref[covered])
            den = np.sum(ref[covered] ** 2)
            return float(num / den)

        a_big = recovered_amplitude(fld_big)
        a_both = recovered_amplitude(fld_both)

        assert abs(a_big) < 0.15 * amp, f"big={a_big:.3f}"
        assert a_both > 0.6 * amp, f"both={a_both:.3f}"


class TestWeightsAndErrors:
    """Covariance handling and posterior errors."""

    def test_posterior_errors_shrink_with_more_obs(self):
        def u(x, y):
            return 0.5 * np.ones_like(x), np.zeros_like(x)

        one = make_tiled_obs(u, window_size=1000.0, domain=2000.0, stride=1000.0)
        many = one * 5  # five independent repeats of the same observations

        fld_one = solve_current_field(one, field_spacing=250.0)
        fld_many = solve_current_field(many, field_spacing=250.0)

        covered = fld_one.coverage > 0
        assert np.all(fld_many.ux_err[covered] < fld_one.ux_err[covered])

    def test_low_weight_outlier_has_little_effect(self):
        def u(x, y):
            return 0.5 * np.ones_like(x), np.zeros_like(x)

        good = make_tiled_obs(u, window_size=1000.0, domain=2000.0, err=0.02)
        outlier = [obs(3.0, 3.0, 0.0, 0.0, 1000.0, err=1.0)]

        fld = solve_current_field(good + outlier, field_spacing=250.0)
        covered = fld.coverage > 0
        np.testing.assert_allclose(fld.ux[covered], 0.5, atol=0.02)

    def test_non_finite_estimates_dropped(self):
        good = [obs(0.5, -0.3, 0.0, 0.0, 1000.0)]
        bad = [obs(9.0, 9.0, 0.0, 0.0, 1000.0, err=float("nan"))]
        fld = solve_current_field(good + bad, field_spacing=250.0)
        assert fld.n_obs == 1
        covered = fld.coverage > 0
        np.testing.assert_allclose(fld.ux[covered], 0.5, atol=0.02)

    def test_no_usable_estimates(self):
        bad = [obs(9.0, 9.0, 0.0, 0.0, 1000.0, err=float("nan"))]
        assert solve_current_field(bad) is None

    def test_zero_window_size_dropped(self):
        bad = [obs(0.5, -0.3, 0.0, 0.0, 0.0)]
        assert solve_current_field(bad) is None


class TestSmoothness:
    """Regularization behavior."""

    def test_smoother_field_with_longer_correlation(self):
        rng = np.random.default_rng(0)

        def u(x, y):
            return 0.5 * np.ones_like(x), np.zeros_like(x)

        estimates = make_tiled_obs(u, window_size=1000.0, domain=6000.0, err=0.05)
        # Perturb observations with noise
        noisy = [
            obs(
                e.ux + rng.normal(0, 0.1),
                e.uy + rng.normal(0, 0.1),
                e.center_x,
                e.center_y,
                e.window_size,
                err=0.1,
            )
            for e in estimates
        ]

        fld_tight = solve_current_field(noisy, field_spacing=250.0, correlation_length=250.0)
        fld_smooth = solve_current_field(noisy, field_spacing=250.0, correlation_length=2000.0)

        def roughness(fld: CurrentField) -> float:
            return float(np.nanmean(np.abs(np.diff(fld.ux, axis=1))))

        assert roughness(fld_smooth) < roughness(fld_tight)


def make_directional_cube(
    ux: float,
    uy: float,
    n_t: int = 64,
    n_xy: int = 128,
    dx: float = 7.5,
    dt: float = 1.5,
    n_waves: int = 300,
    seed: int = 0,
    noise_level: float = 0.2,
    mean_dir_deg: float = 60.0,
    spread: float = 0.8,
):
    """Continuous directional wave spectrum cube (realistic sea surface).

    Unlike the sparse 12-wave generator in test_current.py, this
    approximates a continuous red spectrum with directional spreading —
    the regime real radar data lives in.
    """
    from wamos_tpw.current import FrameCube

    rng = np.random.default_rng(seed)
    x = np.arange(n_xy) * dx
    xg, yg = np.meshgrid(x, x)
    t = np.arange(n_t) * dt
    k = np.exp(rng.uniform(np.log(2 * np.pi / 600), np.log(2 * np.pi / 40), n_waves))
    th = np.deg2rad(mean_dir_deg) + rng.normal(0, spread, n_waves)
    amp = k**-1.25
    amp /= amp.max()
    amp *= rng.rayleigh(1.0, n_waves)
    kx = k * np.cos(th)
    ky = k * np.sin(th)
    om = np.sqrt(9.81 * k) + kx * ux + ky * uy
    ph = rng.uniform(0, 2 * np.pi, n_waves)

    intensity = np.zeros((n_t, n_xy, n_xy))
    for i in range(n_waves):
        phase_xy = kx[i] * xg + ky[i] * yg + ph[i]
        for it in range(n_t):
            intensity[it] += amp[i] * np.cos(phase_xy - om[i] * t[it])
    intensity += rng.normal(0, noise_level * intensity.std(), intensity.shape)
    intensity += abs(intensity.min()) + 1.0
    return FrameCube.from_arrays(intensity, dt=dt, dx=dx)


class TestEndToEnd:
    """Synthetic cube -> multi-scale extraction -> field inversion."""

    def test_uniform_current_cube_to_field(self):
        from wamos_tpw.config import Config
        from wamos_tpw.current import CurrentExtractor, compute_multiscale_tile_specs

        true_ux, true_uy = 0.5, -0.3
        cube = make_directional_cube(true_ux, true_uy, seed=0)

        config = Config()
        config["current.window_sizes"] = [950.0, 480.0]
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

        assert len(estimates) > 3
        assert len({e.window_size for e in estimates}) == 2

        fld = solve_current_field(estimates, field_spacing=120.0)
        assert fld is not None
        covered = fld.coverage > 0
        # Median over the field: uniform current recovered
        assert abs(np.median(fld.ux[covered]) - true_ux) < 0.15
        assert abs(np.median(fld.uy[covered]) - true_uy) < 0.15
        # Robust weighting keeps the data misfit sane
        assert fld.chi2 < 10.0


class TestFieldNetcdf:
    """NetCDF output round trip."""

    def test_write_and_read(self, tmp_path):
        xr = pytest.importorskip("xarray")

        estimates = [obs(0.5, -0.3, 0.0, 0.0, 1000.0)]
        fld = solve_current_field(
            estimates,
            field_spacing=250.0,
            start_time=np.datetime64("2022-04-05T14:00:00"),
            end_time=np.datetime64("2022-04-05T14:01:00"),
        )
        path = write_field_netcdf(fld, str(tmp_path))
        assert path

        ds = xr.open_dataset(path)
        try:
            for var in ("ux", "uy", "ux_err", "uy_err", "coverage"):
                assert var in ds
            assert ds.attrs["n_observations"] == 1
        finally:
            ds.close()
