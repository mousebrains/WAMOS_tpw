#! /usr/bin/env python3
#
# Synthetic radar intensity cubes with prescribed, spatially varying currents
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Synthetic radar intensity cubes with prescribed currents.

Provides wave-field cubes whose Doppler shifts follow a *prescribed,
spatially varying* current field U(x, y), for measuring the spatial
resolution and accuracy of the current-extraction pipeline (which feature
scales are recovered, at what amplitude, with what bias).

The wave field uses the **local plane-wave (frozen Doppler) approximation**:
each spectral component keeps its wavevector k everywhere but oscillates
with the locally Doppler-shifted frequency,

    eta(x, t) = sum_i A_i cos(k_i . x - [omega_0(k_i) + k_i . U(x)] t + phi_i)

This is exactly the signal model the dispersion-based estimator assumes
locally, so recovered-vs-true comparisons measure the *estimator and
inversion* response, not wave-current interaction physics. It is valid for
currents varying slowly relative to a wavelength and for record lengths
short enough that ray refraction is negligible; it does not include
amplitude modulation by current gradients, wave imaging nonlinearity, or
the shadowing/tilt modulation of real radar backscatter.

Current feature fields (each returns ``u_func(x, y) -> (ux, uy)``):

- :func:`uniform_current`
- :func:`linear_shear`
- :func:`sinusoidal_current`
- :func:`gaussian_eddy`
- :func:`current_front`
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from wamos_tpw.current import FrameCube

logger = logging.getLogger(__name__)

__all__ = [
    "current_front",
    "gaussian_eddy",
    "linear_shear",
    "make_current_cube",
    "sinusoidal_current",
    "uniform_current",
]

_G = 9.81

UFunc = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


# ============================================================
# Current feature fields
# ============================================================


def uniform_current(ux: float, uy: float) -> UFunc:
    """Constant current everywhere."""

    def u(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(x, ux, dtype=float), np.full_like(y, uy, dtype=float)

    return u


def linear_shear(
    dudx: float = 0.0,
    dudy: float = 0.0,
    dvdx: float = 0.0,
    dvdy: float = 0.0,
    ux0: float = 0.0,
    uy0: float = 0.0,
) -> UFunc:
    """Linear shear: ux = ux0 + dudx*x + dudy*y, uy likewise (gradients in 1/s)."""

    def u(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ux0 + dudx * x + dudy * y, uy0 + dvdx * x + dvdy * y

    return u


def sinusoidal_current(
    amplitude: float,
    wavelength: float,
    direction_deg: float = 90.0,
    component: str = "along",
) -> UFunc:
    """Sinusoidal current variation at one spatial wavelength.

    Args:
        amplitude: Peak current (m/s).
        wavelength: Spatial wavelength of the variation (m).
        direction_deg: Compass direction of the variation's wavenumber
            (90 = varies along east-west).
        component: ``"along"`` makes the current vector parallel to the
            variation direction (convergent/divergent pattern);
            ``"across"`` makes it perpendicular (shear pattern).
    """
    b = np.deg2rad(direction_deg)
    kx = np.sin(b) * 2 * np.pi / wavelength
    ky = np.cos(b) * 2 * np.pi / wavelength
    if component == "along":
        ex, ey = np.sin(b), np.cos(b)
    else:
        ex, ey = np.cos(b), -np.sin(b)

    def u(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = amplitude * np.sin(kx * x + ky * y)
        return ex * s, ey * s

    return u


def gaussian_eddy(
    peak_speed: float,
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    sense: float = 1.0,
) -> UFunc:
    """Gaussian vortex: azimuthal speed v(r) = peak * (r/R) * exp(0.5*(1-(r/R)^2)).

    Solid-body-like core inside radius R, decaying outside; ``sense`` +1
    is counterclockwise.
    """

    def u(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx_ = x - center_x
        dy_ = y - center_y
        r = np.sqrt(dx_**2 + dy_**2)
        rr = r / radius
        v = peak_speed * rr * np.exp(0.5 * (1.0 - rr**2))
        with np.errstate(invalid="ignore", divide="ignore"):
            ex = np.where(r > 0, -dy_ / r, 0.0)
            ey = np.where(r > 0, dx_ / r, 0.0)
        return sense * v * ex, sense * v * ey

    return u


def current_front(
    u_left: tuple[float, float],
    u_right: tuple[float, float],
    front_x: float = 0.0,
    width: float = 500.0,
) -> UFunc:
    """Smooth tanh front between two uniform currents along x."""

    def u(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = 0.5 * (1.0 + np.tanh((x - front_x) / (width / 2)))
        ux = u_left[0] + (u_right[0] - u_left[0]) * s
        uy = u_left[1] + (u_right[1] - u_left[1]) * s
        return ux, uy

    return u


# ============================================================
# Cube synthesis
# ============================================================


def make_current_cube(
    u_func: UFunc,
    n_t: int = 64,
    n_xy: int = 512,
    dx: float = 15.0,
    dt: float = 1.5,
    n_waves: int = 200,
    wavelength_range: tuple[float, float] = (60.0, 600.0),
    mean_dir_deg: float = 60.0,
    spread: float = 0.8,
    spectral_slope: float = -1.25,
    noise_level: float = 0.2,
    seed: int = 0,
) -> FrameCube:
    """Synthesize a wave cube over a prescribed current field U(x, y).

    A continuous directional spectrum (``n_waves`` components with
    amplitudes ~ k**spectral_slope, Rayleigh-distributed, directions
    normally spread about ``mean_dir_deg``) propagates over the current:
    each component oscillates at its locally Doppler-shifted frequency
    (local plane-wave approximation; see module docstring).

    Uses a per-frame complex phasor recurrence so the cost is one complex
    multiply per (wave, frame, pixel) instead of a transcendental, making
    ~5e9-sample syntheses practical.

    Args:
        u_func: Current field; called with 2D x, y meshes (meters,
            centered on the domain), returns (ux, uy) meshes (m/s).
        n_t: Number of frames.
        n_xy: Grid points per side.
        dx: Grid spacing (m).
        dt: Frame interval (s).
        n_waves: Number of spectral components.
        wavelength_range: (min, max) component wavelengths (m).
        mean_dir_deg: Mean propagation direction (compass, deg).
        spread: Directional spread (radians std).
        spectral_slope: Amplitude power-law exponent in k.
        noise_level: Additive white noise std relative to signal std.
        seed: Random seed.

    Returns:
        FrameCube with the synthetic intensity (positive-shifted).
    """
    rng = np.random.default_rng(seed)

    coords = (np.arange(n_xy) - (n_xy - 1) / 2) * dx
    xg, yg = np.meshgrid(coords, coords)
    ux_field, uy_field = u_func(xg, yg)

    k = np.exp(
        rng.uniform(
            np.log(2 * np.pi / wavelength_range[1]),
            np.log(2 * np.pi / wavelength_range[0]),
            n_waves,
        )
    )
    theta = np.deg2rad(mean_dir_deg) + rng.normal(0, spread, n_waves)
    amp = k**spectral_slope
    amp /= amp.max()
    amp *= rng.rayleigh(1.0, n_waves)
    kx = k * np.sin(theta)  # compass convention: direction -> (sin, cos)
    ky = k * np.cos(theta)
    omega0 = np.sqrt(_G * k)
    phase0 = rng.uniform(0, 2 * np.pi, n_waves)

    intensity = np.zeros((n_t, n_xy, n_xy))

    for i in range(n_waves):
        # Locally Doppler-shifted frequency field
        omega_local = omega0[i] + kx[i] * ux_field + ky[i] * uy_field
        # Phasor at t=0 and its per-frame rotation
        phasor = amp[i] * np.exp(1j * (kx[i] * xg + ky[i] * yg + phase0[i]))
        step = np.exp(-1j * omega_local * dt)
        for it in range(n_t):
            intensity[it] += phasor.real
            phasor *= step

    if noise_level > 0:
        intensity += rng.normal(0, noise_level * intensity.std(), intensity.shape)
    intensity += np.abs(intensity.min()) + 1.0

    logger.debug(
        "Synthesized cube: %d frames, %dx%d at %.1f m, %d waves",
        n_t,
        n_xy,
        n_xy,
        dx,
        n_waves,
    )

    return FrameCube.from_arrays(intensity, dt=dt, dx=dx)
