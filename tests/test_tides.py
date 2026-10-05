"""Tests for tidal harmonic analysis (tides.py)."""

import numpy as np
import pytest

from wamos_tpw.tides import (
    CONDITION_WARN,
    CONSTITUENT_SPEEDS,
    DEFAULT_EPOCH,
    TidalFit,
    harmonic_fit,
    rayleigh_days,
    unresolvable_pairs,
)

EPOCH = DEFAULT_EPOCH


def _synth(
    days: float,
    parts: dict[str, tuple[float, float]],
    step_min: float = 10.0,
    mean: float = 18.0,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic elevation record with known constituents."""
    n = int(days * 24 * 60 / step_min)
    t = EPOCH + (np.arange(n) * step_min * 60).astype("timedelta64[s]")
    th = (t - EPOCH) / np.timedelta64(1, "h")
    y = np.full(n, mean, dtype=float)
    for name, (amp, pha) in parts.items():
        w = np.radians(CONSTITUENT_SPEEDS[name])
        y += amp * np.cos(w * th - np.radians(pha))
    if noise:
        y += np.random.default_rng(seed).normal(0.0, noise, n)
    return t, y


# --------------------------------------------------------------- Rayleigh ---


def test_rayleigh_known_pairs() -> None:
    """K1/P1 and S2/K2 need half a year; M2/S2 needs a fortnight."""
    assert rayleigh_days("K1", "P1") == pytest.approx(182.6, rel=1e-2)
    assert rayleigh_days("S2", "K2") == pytest.approx(182.6, rel=1e-2)
    assert rayleigh_days("M2", "S2") == pytest.approx(14.77, rel=1e-2)
    assert rayleigh_days("K1", "O1") == pytest.approx(13.66, rel=1e-2)
    assert rayleigh_days("M2", "M2") == float("inf")


def test_unresolvable_pairs_sorted_by_requirement() -> None:
    bad = unresolvable_pairs(["M2", "S2", "K1", "P1"], duration_days=29.0)
    assert ("K1", "P1", pytest.approx(182.6, rel=1e-2)) in [
        (a, b, pytest.approx(d, rel=1e-2)) for a, b, d in bad
    ]
    assert bad[0][2] >= bad[-1][2]
    # M2/S2 IS resolvable in 29 days, so it must not be listed
    assert not any({a, b} == {"M2", "S2"} for a, b, _ in bad)


# ------------------------------------------------------------------- fit ---


def test_recovers_known_amplitudes_and_phases() -> None:
    truth = {"M2": (0.50, 168.0), "S2": (0.18, 200.0), "K1": (0.24, 90.0), "O1": (0.17, 300.0)}
    t, y = _synth(29.0, truth)
    fit = harmonic_fit(t, y, ("M2", "S2", "K1", "O1"))
    for name, (amp, pha) in truth.items():
        assert fit.amplitude(name) == pytest.approx(amp, abs=2e-3)
        assert fit.phase(name) == pytest.approx(pha, abs=1.0)
    assert fit.variance_explained > 0.999
    assert fit.n_used == len(t)


def test_predict_reproduces_the_record() -> None:
    t, y = _synth(29.0, {"M2": (0.5, 168.0), "K1": (0.24, 90.0)})
    fit = harmonic_fit(t, y, ("M2", "K1"))
    assert np.allclose(fit.predict(t), y, atol=1e-6)


def test_predict_includes_mean_so_it_can_correct_a_dem() -> None:
    t, y = _synth(29.0, {"M2": (0.5, 0.0)}, mean=18.4)
    fit = harmonic_fit(t, y, ("M2",))
    assert float(np.mean(fit.predict(t))) == pytest.approx(18.4, abs=1e-3)


def test_nans_are_dropped() -> None:
    t, y = _synth(29.0, {"M2": (0.5, 168.0)})
    y[::13] = np.nan
    fit = harmonic_fit(t, y, ("M2",))
    assert fit.amplitude("M2") == pytest.approx(0.5, abs=2e-3)
    assert fit.n_used == int(np.isfinite(y).sum())


# ------------------------------------------------ the guard that matters ---


def test_unresolvable_constituents_are_refused() -> None:
    """A 29-day record must not silently fit K1 and P1 together."""
    t, y = _synth(29.0, {"K1": (0.24, 90.0), "P1": (0.08, 80.0)})
    with pytest.raises(ValueError, match="cannot separate"):
        harmonic_fit(t, y, ("M2", "K1", "P1"))


def test_pairwise_rayleigh_is_conservative() -> None:
    """K1/P1 alone survives 29 days: Rayleigh is necessary, not sufficient.

    They still drift 57 degrees apart over the deployment, so with ordinary
    noise least squares recovers both. The guard is still right to refuse by
    default, but this records why the pair alone is not the failure mode.
    """
    t, y = _synth(29.0, {"K1": (0.24, 90.0), "P1": (0.08, 80.0)}, noise=0.04)
    fit = harmonic_fit(t, y, ("K1", "P1"), allow_unresolvable=True)
    assert fit.amplitude("P1") == pytest.approx(0.08, abs=0.02)
    assert fit.condition < CONDITION_WARN


def test_three_way_diurnal_degeneracy_corrupts_amplitudes() -> None:
    """S1 alongside K1 and P1 is what actually breaks the fit.

    The residual stays small -- which reads as success -- while P1 comes out
    several times its true amplitude. The condition number is what exposes it.
    """
    truth = {
        "K1": (0.24, 90.0),
        "O1": (0.17, 300.0),
        "P1": (0.08, 80.0),
        "S1": (0.02, 10.0),
        "M2": (0.50, 168.0),
    }
    err_ok, err_bad = [], []
    for seed in range(12):
        t, y = _synth(29.0, truth, noise=0.04, seed=seed)
        ok = harmonic_fit(t, y, ("K1", "O1", "P1", "M2"), allow_unresolvable=True)
        bad = harmonic_fit(t, y, ("K1", "O1", "P1", "S1", "M2"), allow_unresolvable=True)
        err_ok.append(abs(ok.amplitude("P1") - 0.08))
        err_bad.append(abs(bad.amplitude("P1") - 0.08))

    # Conditioning is deterministic -- it does not depend on the noise draw.
    assert bad.condition > 10 * ok.condition
    assert bad.condition > CONDITION_WARN
    assert ok.condition < CONDITION_WARN

    # Amplitude damage is statistical: the median error inflates several-fold,
    # but a single record can land close by luck, which is exactly why the
    # residual and a one-off amplitude are not adequate diagnostics.
    assert float(np.median(err_bad)) > 3.0 * float(np.median(err_ok))

    # the trap: the corrupted fit does NOT look worse by residual
    assert bad.residual_std <= ok.residual_std * 1.05


def test_inference_carries_p1_without_a_free_parameter() -> None:
    """Imposing P1 from K1 recovers both from a short record."""
    ratio, lag = 0.331, -10.0
    t, y = _synth(29.0, {"K1": (0.24, 90.0), "P1": (0.24 * ratio, 90.0 + lag)})
    fit = harmonic_fit(t, y, ("K1",), infer={"P1": ("K1", ratio, lag)})
    assert fit.amplitude("K1") == pytest.approx(0.24, abs=5e-3)
    assert fit.amplitude("P1") == pytest.approx(0.24 * ratio, abs=5e-3)
    assert fit.phase("P1") == pytest.approx((90.0 + lag) % 360.0, abs=2.0)
    assert "P1" in fit.inferred
    assert fit.variance_explained > 0.999


def test_infer_requires_the_major_to_be_fitted() -> None:
    t, y = _synth(29.0, {"K1": (0.24, 90.0)})
    with pytest.raises(ValueError, match="is not fitted"):
        harmonic_fit(t, y, ("M2",), infer={"P1": ("K1", 0.331, -10.0)})


# ------------------------------------------------------- phase referencing ---


def test_phase_depends_on_epoch_and_is_reported() -> None:
    """Two records starting at different times give the same phase."""
    t, y = _synth(29.0, {"M2": (0.5, 168.0)})
    late = slice(1000, None)
    a = harmonic_fit(t, y, ("M2",))
    b = harmonic_fit(t[late], y[late], ("M2",))
    assert a.phase("M2") == pytest.approx(b.phase("M2"), abs=0.5)
    assert a.epoch == b.epoch == EPOCH


def test_custom_epoch_shifts_phase_predictably() -> None:
    t, y = _synth(29.0, {"M2": (0.5, 0.0)})
    other = EPOCH + np.timedelta64(6, "h")
    a = harmonic_fit(t, y, ("M2",))
    b = harmonic_fit(t, y, ("M2",), epoch=other)
    # y = A cos(w*th - phi); moving the epoch 6 h later replaces th by th+6,
    # so the reported phase decreases by 6*w.
    shift = (-CONSTITUENT_SPEEDS["M2"] * 6.0) % 360.0
    assert (b.phase("M2") - a.phase("M2")) % 360.0 == pytest.approx(shift, abs=1.0)


# ------------------------------------------------------------ validation ---


def test_shape_mismatch_raises() -> None:
    t, y = _synth(5.0, {"M2": (0.5, 0.0)})
    with pytest.raises(ValueError, match="differ in shape"):
        harmonic_fit(t, y[:-1], ("M2",))


def test_unknown_constituent_raises() -> None:
    t, y = _synth(5.0, {"M2": (0.5, 0.0)})
    with pytest.raises(KeyError, match="NOPE"):
        harmonic_fit(t, y, ("M2", "NOPE"))


def test_too_few_finite_samples_raises() -> None:
    t, y = _synth(5.0, {"M2": (0.5, 0.0)})
    y[:] = np.nan
    with pytest.raises(ValueError, match="at least 3 finite"):
        harmonic_fit(t, y, ("M2",))


def test_trend_is_recovered() -> None:
    t, y = _synth(29.0, {"M2": (0.5, 0.0)})
    th = (t - EPOCH) / np.timedelta64(1, "h")
    fit = harmonic_fit(t, y - 0.0038 * th, ("M2",))
    assert fit.trend == pytest.approx(-0.0038, abs=1e-5)


def test_returns_tidalfit_dataclass() -> None:
    t, y = _synth(29.0, {"M2": (0.5, 0.0)})
    assert isinstance(harmonic_fit(t, y, ("M2",)), TidalFit)
