"""Tidal harmonic analysis of sea-surface elevation records.

Supports the depth-error mitigation ladder in the Thompson-2023 plan (T2b):
finite-depth dispersion needs *instantaneous* water depth, and bank depth
varies by ~±1 m (±5 %) over the tidal cycle. This module turns a bottom-
pressure or tide-gauge record into tidal constituents, and predicts elevation
at arbitrary times so a DEM depth can be corrected to the moment of a radar
scan.

Two failure modes dominate this analysis, and both are guarded here rather
than left to the caller:

**Unresolvable constituents.** Two constituents separated by ``ds`` degrees
per hour need a record of at least ``360/ds`` hours before least squares can
separate them cleanly (the Rayleigh criterion). K1 and P1 need 183 days; S2
and K2 likewise. :func:`unresolvable_pairs` reports the conflicts and
:func:`harmonic_fit` refuses them unless ``allow_unresolvable=True``.

That pairwise test is necessary but *not sufficient*, and the distinction
matters. Rayleigh is conservative: with well-behaved noise a 29-day record
recovers K1 and P1 to a few percent, because they still drift 57 degrees
apart over the deployment. What actually destroys the fit is stacking
several mutually close lines. On the 2023 Palau gauges, adding S1 (24.000 h)
alongside K1 (23.934 h) and P1 (24.066 h) made it a three-way degeneracy and
P1 came out at 22-76 cm against an expected ~8 cm (up to 9x, and 3x K1 itself
where the ratio should be ~0.33) -- while the residual *fell* by 25 %, which
reads as success. The design-matrix condition number rises ~26x with that
third line, from ~2.9e3 to ~7.7e4; in synthetic tests at the same noise level
the median P1 error grows ~6x, though on any single record it can land close
by luck. :attr:`TidalFit.condition` is therefore the reliable indicator, not
the residual, and a warning fires above :data:`CONDITION_WARN`.

The correct treatment for a short record is *inference*: take the amplitude
ratio and phase lag of the minor constituent from a long reference record --
for Palau, the Malakal Harbor gauge at Koror -- and impose it. See
``infer`` in :func:`harmonic_fit`.

**Phase referencing.** Phases are only comparable between instruments when
they share an epoch. :class:`TidalFit` always carries the epoch it was
computed against, and :data:`DEFAULT_EPOCH` is used unless one is given.

A note on preparing the input, since it is not this module's job but it
determines whether the answer means anything: decimate a high-rate pressure
record by *block-averaging*, never by striding. At 18 m depth an 8 s wave
still reaches the bottom at ~58 % amplitude, so 0.5 m of swell puts ~25 cm
into bottom pressure; striding to 1 min folds all of it into the tidal band.
On the 2023 Palau gauges this inflated the post-fit residual from 4 cm to
14 cm and produced a physically impossible 149-degree M2 phase spread across
an 11 km array.

Usage::

    from wamos_tpw.tides import harmonic_fit, DEFAULT_CONSTITUENTS

    fit = harmonic_fit(times, elevation)
    print(fit.amplitude("M2"), fit.phase("M2"))
    depth_now = dem_depth + fit.predict(scan_times)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CONDITION_WARN",
    "CONSTITUENT_SPEEDS",
    "DEFAULT_CONSTITUENTS",
    "DEFAULT_EPOCH",
    "TidalFit",
    "harmonic_fit",
    "rayleigh_days",
    "unresolvable_pairs",
]

#: Constituent speeds in degrees per mean solar hour (Doodson).
CONSTITUENT_SPEEDS: dict[str, float] = {
    # semidiurnal
    "M2": 28.9841042,
    "S2": 30.0000000,
    "N2": 28.4397295,
    "K2": 30.0821373,
    "NU2": 28.5125831,
    "MU2": 27.9682084,
    "L2": 29.5284789,
    "T2": 29.9589333,
    "2N2": 27.8953548,
    # diurnal
    "K1": 15.0410686,
    "O1": 13.9430356,
    "P1": 14.9589314,
    "Q1": 13.3986609,
    "S1": 15.0000000,
    "J1": 15.5854433,
    # shallow-water overtides and compound tides
    "M4": 57.9682084,
    "MS4": 58.9841042,
    "MN4": 57.4238337,
    "M6": 86.9523127,
    "2MS6": 87.9682084,
    "M8": 115.9364169,
    # long period
    "Mf": 1.0980331,
    "Mm": 0.5443747,
    "MSf": 1.0158958,
    "Sa": 0.0410686,
    "Ssa": 0.0821373,
}

#: Constituents a ~29 day record can carry without inference.
#:
#: Deliberately excludes MSf: it sits 0.082 deg/h from Mf, the same separation
#: as K1/P1, so the two need 183 days. Including both is the identical trap.
#: N2 needs 27.6 days against M2, so a 26-day record (the 2023 Hydrographer
#: Bank deployments) must drop it too -- :func:`harmonic_fit` will say so.
DEFAULT_CONSTITUENTS: tuple[str, ...] = (
    "M2",
    "S2",
    "N2",
    "K1",
    "O1",
    "M4",
    "MS4",
    "M6",
    "Mf",
)

#: Phase reference shared by every fit unless overridden.
DEFAULT_EPOCH = np.datetime64("2023-05-01T00:00:00", "s")

#: Design-matrix condition number above which amplitudes are not trustworthy.
#: Calibrated on the 2023 Palau gauges: ~8e2 for a clean set, ~3e3 with one
#: unresolvable pair (a few percent error), ~8e4 with a three-way diurnal
#: degeneracy (several hundred percent error).
CONDITION_WARN = 1.0e4


def rayleigh_days(name_a: str, name_b: str) -> float:
    """Record length needed to separate two constituents by least squares.

    Args:
        name_a: Constituent name, a key of :data:`CONSTITUENT_SPEEDS`.
        name_b: Constituent name, a key of :data:`CONSTITUENT_SPEEDS`.

    Returns:
        Days of continuous record required, ``inf`` when the two speeds are
        identical (never separable).

    Raises:
        KeyError: If either name is not a known constituent.
    """
    ds = abs(CONSTITUENT_SPEEDS[name_a] - CONSTITUENT_SPEEDS[name_b])
    if ds == 0.0:
        return float("inf")
    return 360.0 / ds / 24.0


def unresolvable_pairs(
    constituents: tuple[str, ...] | list[str],
    duration_days: float,
) -> list[tuple[str, str, float]]:
    """Constituent pairs the record is too short to separate.

    Args:
        constituents: Names to be fitted together.
        duration_days: Length of the record in days.

    Returns:
        ``(name_a, name_b, days_needed)`` for every pair whose Rayleigh
        requirement exceeds ``duration_days``, longest requirement first.
    """
    out: list[tuple[str, str, float]] = []
    names = list(constituents)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            need = rayleigh_days(a, b)
            if need > duration_days:
                out.append((a, b, need))
    out.sort(key=lambda t: -t[2])
    return out


@dataclass
class TidalFit:
    """Result of a least-squares tidal harmonic fit.

    Attributes:
        constituents: Names fitted, in the order supplied.
        amplitudes: Amplitude of each constituent in the units of the input
            (meters for an elevation record), keyed by name.
        phases: Greenwich-style phase lag in degrees relative to ``epoch``,
            keyed by name. Comparable between instruments only when they
            share an epoch.
        epoch: Phase reference.
        mean: Fitted constant term.
        trend: Fitted linear term per hour, ``0.0`` when not fitted.
        residual_std: Standard deviation of observation minus fit.
        variance_explained: Fraction of the input variance the fit removes.
        n_used: Number of finite samples the fit used.
        duration_days: Span of the record.
        condition: Condition number of the design matrix. Below ~1e3 the
            amplitudes are well determined; above :data:`CONDITION_WARN` they
            are not, even if the residual looks small.
        inferred: Constituents imposed by inference rather than fitted.
    """

    constituents: tuple[str, ...]
    amplitudes: dict[str, float]
    phases: dict[str, float]
    epoch: np.datetime64
    mean: float
    trend: float
    residual_std: float
    variance_explained: float
    n_used: int
    duration_days: float
    condition: float = float("nan")
    inferred: tuple[str, ...] = field(default=())

    def amplitude(self, name: str) -> float:
        """Amplitude of one constituent."""
        return self.amplitudes[name]

    def phase(self, name: str) -> float:
        """Phase lag of one constituent, degrees relative to ``self.epoch``."""
        return self.phases[name]

    def predict(self, times: np.ndarray) -> np.ndarray:
        """Predict elevation at arbitrary times.

        Args:
            times: ``datetime64`` array.

        Returns:
            Predicted elevation, same shape as ``times``, including the mean
            and trend terms so it can be added directly to a DEM depth.
        """
        th = (times - self.epoch) / np.timedelta64(1, "h")
        out = np.full(np.shape(th), self.mean, dtype=float) + self.trend * th
        for name in self.constituents:
            w = np.radians(CONSTITUENT_SPEEDS[name])
            out = out + self.amplitudes[name] * np.cos(w * th - np.radians(self.phases[name]))
        return out


def harmonic_fit(
    times: np.ndarray,
    elevation: np.ndarray,
    constituents: tuple[str, ...] | list[str] = DEFAULT_CONSTITUENTS,
    *,
    epoch: np.datetime64 | None = None,
    trend: bool = True,
    infer: dict[str, tuple[str, float, float]] | None = None,
    allow_unresolvable: bool = False,
) -> TidalFit:
    """Least-squares tidal harmonic analysis of an elevation record.

    Args:
        times: ``datetime64`` array of sample times. Need not be regular.
        elevation: Sea-surface elevation, same length as ``times``. NaNs are
            dropped.
        constituents: Names to fit. The default set is what a 25-30 day
            deployment supports without inference.
        epoch: Phase reference; :data:`DEFAULT_EPOCH` when omitted. Use the
            same epoch for every instrument in an array or the phases cannot
            be compared.
        trend: Fit a linear drift term alongside the mean.
        infer: Constituents to impose rather than fit, as
            ``{minor: (major, amplitude_ratio, phase_lag_deg)}``. Ratios come
            from a long reference record -- for Palau, Malakal Harbor. The
            minor constituent is added to the design matrix with its
            amplitude and phase locked to the fitted major, which is how a
            short record carries K1/P1 and S2/K2 without overfitting.
        allow_unresolvable: Fit constituents the record is too short to
            separate. Off by default because the failure is silent: least
            squares splits energy between the pair, reduces the residual, and
            returns amplitudes that are badly wrong.

    Returns:
        The fitted :class:`TidalFit`.

    Raises:
        ValueError: If the inputs disagree in length, fewer than three finite
            samples remain, or the record cannot resolve the requested
            constituents and ``allow_unresolvable`` is False.
        KeyError: If a constituent name is unknown.
    """
    times = np.asarray(times)
    elevation = np.asarray(elevation, dtype=float)
    if times.shape != elevation.shape:
        raise ValueError(f"times and elevation differ in shape: {times.shape} vs {elevation.shape}")
    names = tuple(constituents)
    for name in names:
        if name not in CONSTITUENT_SPEEDS:
            raise KeyError(f"unknown tidal constituent {name!r}")

    ok = np.isfinite(elevation)
    if ok.sum() < 3:
        raise ValueError(f"need at least 3 finite samples, got {int(ok.sum())}")

    ep = DEFAULT_EPOCH if epoch is None else np.datetime64(epoch)
    th = (times - ep) / np.timedelta64(1, "h")
    duration_days = float((th.max() - th.min()) / 24.0)

    conflicts = unresolvable_pairs(names, duration_days)
    if conflicts and not allow_unresolvable:
        worst = ", ".join(f"{a}/{b} needs {d:.0f} d" for a, b, d in conflicts[:4])
        raise ValueError(
            f"record is {duration_days:.1f} days; cannot separate {worst}. "
            "Drop one of each pair, or impose it with infer=..., or pass "
            "allow_unresolvable=True if you accept that the split between "
            "them is arbitrary."
        )
    if conflicts:
        logger.warning(
            "fitting %d unresolvable constituent pair(s) in a %.1f day record; "
            "individual amplitudes are not trustworthy",
            len(conflicts),
            duration_days,
        )

    infer = infer or {}
    for minor, (major, _ratio, _lag) in infer.items():
        if minor not in CONSTITUENT_SPEEDS:
            raise KeyError(f"unknown tidal constituent {minor!r}")
        if major not in names:
            raise ValueError(f"cannot infer {minor!r} from {major!r}: {major!r} is not fitted")

    # Design matrix. Each fitted constituent contributes cos/sin columns; an
    # inferred minor rides on its major's columns with a fixed rotation, so it
    # adds no free parameters.
    cols: list[np.ndarray] = [np.ones_like(th)]
    if trend:
        cols.append(th)
    for name in names:
        w = np.radians(CONSTITUENT_SPEEDS[name])
        c = np.cos(w * th)
        s = np.sin(w * th)
        for minor, (major, ratio, lag) in infer.items():
            if major != name:
                continue
            wm = np.radians(CONSTITUENT_SPEEDS[minor])
            lg = np.radians(lag)
            c = c + ratio * np.cos(wm * th - lg)
            s = s + ratio * np.sin(wm * th - lg)
        cols.extend((c, s))

    design = np.column_stack(cols)
    condition = float(np.linalg.cond(design[ok]))
    if condition > CONDITION_WARN:
        logger.warning(
            "design-matrix condition number %.3g exceeds %.3g: individual "
            "amplitudes are unreliable even though the residual may look "
            "small. Drop near-degenerate constituents (S1 alongside K1/P1 is "
            "the usual culprit) or impose them with infer=...",
            condition,
            CONDITION_WARN,
        )
    coef, *_ = np.linalg.lstsq(design[ok], elevation[ok], rcond=None)
    fitted = design @ coef
    residual = elevation - fitted

    amplitudes: dict[str, float] = {}
    phases: dict[str, float] = {}
    base = 2 if trend else 1
    for i, name in enumerate(names):
        c, s = coef[base + 2 * i], coef[base + 2 * i + 1]
        amplitudes[name] = float(np.hypot(c, s))
        phases[name] = float(np.degrees(np.arctan2(s, c)) % 360.0)
    for minor, (major, ratio, lag) in infer.items():
        amplitudes[minor] = amplitudes[major] * ratio
        phases[minor] = float((phases[major] + lag) % 360.0)

    var_in = float(np.nanvar(elevation[ok]))
    return TidalFit(
        constituents=names + tuple(infer),
        amplitudes=amplitudes,
        phases=phases,
        epoch=ep,
        mean=float(coef[0]),
        trend=float(coef[1]) if trend else 0.0,
        residual_std=float(np.nanstd(residual[ok])),
        variance_explained=(float(1.0 - np.nanvar(residual[ok]) / var_in) if var_in > 0 else 0.0),
        n_used=int(ok.sum()),
        duration_days=duration_days,
        condition=condition,
        inferred=tuple(infer),
    )
