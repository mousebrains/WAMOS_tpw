"""Alias-aware dispersion fitting (current.alias_m) on undersampled cubes.

Angaur-like sampling: dt ~ 5 s (Nyquist period 10 s) folds all wind-sea
components (deep-water wavelengths < 156 m) onto alias branches. With
alias_m = 1 the extractor models one Nyquist fold; because direction
and frequency negate together under folding, folded peaks constrain
the same (Ux, Uy) as the true branch.
"""

import numpy as np
import pytest

from wamos_tpw.current import CurrentExtractor
from wamos_tpw.synthetic import make_current_cube, uniform_current

DT = 5.0
UX, UY = 0.40, -0.20


def _extract(cube, alias_m):
    cfg = {
        "current.depth": 5000.0,
        "current.k_max": 2.0 * np.pi / 50.0,
        "current.alias_m": alias_m,
        "current.search_radius": 1.5,
    }
    ex = CurrentExtractor(cube, config=cfg)
    return ex.estimate


def _err(est):
    return float(np.hypot(est.ux - UX, est.uy - UY))


@pytest.fixture(scope="module")
def mixed_cube():
    """Swell (unaliased) + wind sea (aliased): the realistic Angaur mix."""
    return make_current_cube(
        uniform_current(UX, UY),
        n_t=128,
        n_xy=128,
        dx=20.0,
        dt=DT,
        n_waves=120,
        wavelength_range=(55.0, 400.0),
        mean_dir_deg=90.0,
        spread=0.6,
        noise_level=0.3,
        seed=11,
    )


@pytest.fixture(scope="module")
def allwind_cube():
    """Every component beyond Nyquist: no true-branch energy at all."""
    return make_current_cube(
        uniform_current(UX, UY),
        n_t=128,
        n_xy=128,
        dx=20.0,
        dt=DT,
        n_waves=120,
        wavelength_range=(55.0, 140.0),
        mean_dir_deg=90.0,
        spread=0.6,
        noise_level=0.3,
        seed=12,
    )


class TestAliasAware:
    def test_mixed_spectrum_alias_improves(self, mixed_cube):
        e0 = _err(_extract(mixed_cube, 0))
        e1 = _err(_extract(mixed_cube, 1))
        assert e1 < 0.10, f"alias_m=1 error {e1:.3f}"
        assert e1 <= e0 + 0.02, f"alias fit worse: {e1:.3f} vs {e0:.3f}"

    def test_allwind_spectrum_requires_alias(self, allwind_cube):
        est1 = _extract(allwind_cube, 1)
        e1 = _err(est1)
        assert e1 < 0.15, f"alias_m=1 error {e1:.3f} on all-aliased spectrum"
        assert est1.n_ls_points >= 6

    def test_legacy_unchanged_when_unaliased(self):
        cube = make_current_cube(
            uniform_current(UX, UY),
            n_t=96,
            n_xy=128,
            dx=15.0,
            dt=1.5,
            n_waves=100,
            wavelength_range=(60.0, 400.0),
            mean_dir_deg=60.0,
            noise_level=0.3,
            seed=3,
        )
        cfg0 = {"current.depth": 5000.0}
        cfg1 = {"current.depth": 5000.0, "current.alias_m": 1}
        est0 = CurrentExtractor(cube, config=cfg0).estimate
        est1 = CurrentExtractor(cube, config=cfg1).estimate
        # well-sampled cube: alias branches are empty, results must agree
        assert np.hypot(est0.ux - est1.ux, est0.uy - est1.uy) < 0.05
        assert np.hypot(est0.ux - UX, est0.uy - UY) < 0.10
