#! /usr/bin/env python3
#
# Empirical spatial-resolution study of the current-extraction pipeline
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Measure the spatial resolution of the current-extraction pipeline.

Synthesizes wave cubes over prescribed current fields (uniform, shear,
sinusoids at several wavelengths, an eddy, a front) using the local
plane-wave model in :mod:`wamos_tpw.synthetic`, runs several pipeline
variants on each, and measures how faithfully each variant recovers the
prescribed field:

- **amplitude transfer**: regression of the recovered anomaly pattern
  onto the true pattern (1 = perfect, 0 = feature invisible),
- **rmse / bias** over covered cells,
- **error calibration**: median(|error| / reported 1-sigma error).

Variants:

- ``tiles-2km``: independent 2 km windows (the classical approach);
  output at the 1 km tile stride.
- ``field-2+1``: joint inversion of 2 km + 1 km windows on a 250 m grid.
- ``field-2+1+0.5``: adds 500 m windows.
- ``field-weak-prior``: same windows with a prior allowing 4x steeper
  gradients (the smoothness prior is the resolution knob).

Writes ``docs/resolution_validation.md`` and figures under
``docs/figures/resolution/``.

Usage::

    python3 tools/resolution_study.py [--quick] [--out docs]
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wamos_tpw.config import Config
from wamos_tpw.current import CurrentExtractor, compute_multiscale_tile_specs
from wamos_tpw.current_field import solve_current_field
from wamos_tpw.synthetic import (
    UFunc,
    current_front,
    gaussian_eddy,
    linear_shear,
    make_current_cube,
    sinusoidal_current,
    uniform_current,
)

logger = logging.getLogger("resolution_study")


# ============================================================
# Extraction variants
# ============================================================


def extract_estimates(cube, window_sizes: list[float], overlap: float = 0.5):
    """Run CurrentExtractor over all (multi-scale) windows of a cube."""
    config = Config()
    config["current.window_sizes"] = list(window_sizes)
    config["current.sub_region_overlap"] = overlap

    specs = compute_multiscale_tile_specs(cube, config)
    tiles = [t for t in specs["tiles"] if not t.get("masked")]

    def run(tile):
        sub = cube.sub_cube(tile["x_start"], tile["x_end"], tile["y_start"], tile["y_end"])
        try:
            return CurrentExtractor(sub, config=config).estimate
        except Exception:
            logger.debug("Window extraction failed", exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(run, tiles))

    return [r for r in results if r is not None]


def variant_tiles(cube, window: float):
    """Classical independent tiles: estimates at tile centers."""
    ests = extract_estimates(cube, [window])
    xs = np.array([e.center_x for e in ests])
    ys = np.array([e.center_y for e in ests])
    ux = np.array([e.ux for e in ests])
    uy = np.array([e.uy for e in ests])
    ex = np.array([e.ux_err for e in ests])
    ey = np.array([e.uy_err for e in ests])
    ok = np.isfinite(ux) & np.isfinite(uy) & np.isfinite(ex) & np.isfinite(ey)
    return {
        "x": xs[ok],
        "y": ys[ok],
        "ux": ux[ok],
        "uy": uy[ok],
        "ux_err": ex[ok],
        "uy_err": ey[ok],
        "footprint": np.full(int(ok.sum()), window),
    }


def variant_field(
    cube,
    window_sizes: list[float],
    field_spacing: float,
    corr_len: float,
    sigma_prior: float = 0.3,
):
    """Joint inversion of multi-scale windows."""
    ests = extract_estimates(cube, window_sizes)
    fld = solve_current_field(
        ests,
        field_spacing=field_spacing,
        correlation_length=corr_len,
        sigma_prior=sigma_prior,
    )
    if fld is None:
        return None
    xg, yg = np.meshgrid(fld.x_centers, fld.y_centers)
    covered = fld.coverage > 0
    return {
        "x": xg[covered],
        "y": yg[covered],
        "ux": fld.ux[covered],
        "uy": fld.uy[covered],
        "ux_err": fld.ux_err[covered],
        "uy_err": fld.uy_err[covered],
        "footprint": np.full(int(covered.sum()), field_spacing),
        "field": fld,
    }


# ============================================================
# Metrics
# ============================================================


def footprint_true(u_func: UFunc, x: np.ndarray, y: np.ndarray, footprint: np.ndarray):
    """True current footprint-averaged at each output location.

    The fair reference for a tile estimate is the average of the true
    field over the tile, not the point value at its center.
    """
    s = np.linspace(-0.5, 0.5, 7)
    sx, sy = np.meshgrid(s, s)
    tux = np.empty_like(x)
    tuy = np.empty_like(y)
    for i in range(len(x)):
        ux_i, uy_i = u_func(x[i] + sx * footprint[i], y[i] + sy * footprint[i])
        tux[i] = np.mean(ux_i)
        tuy[i] = np.mean(uy_i)
    return tux, tuy


def metrics(result: dict, u_func: UFunc) -> dict:
    """Compute transfer, rmse, bias, and calibration for one variant run."""
    x, y = result["x"], result["y"]
    # Point-value truth (what we want to know); footprint-average truth is
    # the fair target for transfer of the *estimator*, point truth measures
    # what the user actually gets.
    tux, tuy = u_func(x, y)

    rux = result["ux"]
    ruy = result["uy"]

    err = np.concatenate([rux - tux, ruy - tuy])
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = (float(np.mean(rux - tux)), float(np.mean(ruy - tuy)))

    # Pattern (anomaly) transfer: regress recovered anomaly on true anomaly
    ta = np.concatenate([tux - tux.mean(), tuy - tuy.mean()])
    ra = np.concatenate([rux - rux.mean(), ruy - ruy.mean()])
    denom = float(np.sum(ta**2))
    transfer = float(np.sum(ra * ta) / denom) if denom > 1e-12 else float("nan")

    rep = np.concatenate([result["ux_err"], result["uy_err"]])
    with np.errstate(invalid="ignore", divide="ignore"):
        calib = float(np.nanmedian(np.abs(err) / rep))

    return {
        "rmse": rmse,
        "bias": bias,
        "transfer": transfer,
        "calibration": calib,
        "n": len(x),
    }


# ============================================================
# Study definition
# ============================================================


def build_features(quick: bool) -> list[tuple[str, UFunc, str]]:
    """(name, u_func, description) for each prescribed current field."""
    feats = [
        ("uniform", uniform_current(0.3, -0.2), "Uniform 0.36 m/s"),
        (
            "shear",
            linear_shear(dudx=1e-4, ux0=0.0),
            "Linear shear dUx/dx = 1e-4 /s (0.77 m/s across domain)",
        ),
        (
            "sin-4km",
            sinusoidal_current(0.3, 4000.0, 90.0, "across"),
            "Sinusoidal shear, 4 km wavelength, 0.3 m/s amplitude",
        ),
        (
            "sin-2km",
            sinusoidal_current(0.3, 2000.0, 90.0, "across"),
            "Sinusoidal shear, 2 km wavelength, 0.3 m/s amplitude",
        ),
        (
            "sin-1km",
            sinusoidal_current(0.3, 1000.0, 90.0, "across"),
            "Sinusoidal shear, 1 km wavelength, 0.3 m/s amplitude",
        ),
        (
            "eddy",
            gaussian_eddy(0.5, 1500.0),
            "Gaussian eddy, 0.5 m/s peak, 1.5 km radius",
        ),
        (
            "front",
            current_front((0.0, 0.0), (0.5, 0.0), front_x=0.0, width=500.0),
            "Current front, 0.5 m/s step over 500 m",
        ),
    ]
    if quick:
        feats = [feats[0], feats[3]]
    return feats


def run_study(out_dir: Path, quick: bool) -> None:
    n_xy = 256 if quick else 512
    n_t = 32 if quick else 64
    dx = 15.0
    domain = n_xy * dx

    variants = [
        ("tiles-2km", lambda cube: variant_tiles(cube, 2000.0)),
        (
            "field-2+1",
            lambda cube: variant_field(cube, [2000.0, 1000.0], 250.0, 500.0),
        ),
        (
            "field-2+1+0.5",
            lambda cube: variant_field(cube, [2000.0, 1000.0, 500.0], 250.0, 500.0),
        ),
        (
            # The smoothness prior is the resolution knob: this variant
            # allows gradients 4x steeper (sigma_prior/L_c = 0.5/250 vs
            # 0.3/500), trading noise suppression for fine-scale response
            "field-weak-prior",
            lambda cube: variant_field(
                cube, [2000.0, 1000.0, 500.0], 250.0, 250.0, sigma_prior=0.5
            ),
        ),
    ]

    fig_dir = out_dir / "figures" / "resolution"
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(quick)
    rows: list[dict] = []
    fields_for_plots: dict[tuple[str, str], dict] = {}

    for fname, u_func, desc in features:
        t0 = time.perf_counter()
        cube = make_current_cube(u_func, n_t=n_t, n_xy=n_xy, dx=dx, seed=11)
        logger.info("Synthesized %s in %.1fs", fname, time.perf_counter() - t0)

        for vname, vfunc in variants:
            t0 = time.perf_counter()
            result = vfunc(cube)
            if result is None or len(result["x"]) == 0:
                logger.warning("%s / %s produced no output", fname, vname)
                continue
            m = metrics(result, u_func)
            m.update({"feature": fname, "variant": vname, "desc": desc})
            rows.append(m)
            fields_for_plots[(fname, vname)] = result
            logger.info(
                "%s / %s: transfer=%.2f rmse=%.3f n=%d (%.1fs)",
                fname,
                vname,
                m["transfer"],
                m["rmse"],
                m["n"],
                time.perf_counter() - t0,
            )

        del cube

    _plot_transfer_curve(rows, fig_dir)
    _plot_feature_maps(features, fields_for_plots, fig_dir, domain)
    _write_report(rows, features, out_dir, n_xy, n_t, dx, quick)


# ============================================================
# Plots and report
# ============================================================


def _plot_transfer_curve(rows: list[dict], fig_dir: Path) -> None:
    """Amplitude transfer vs feature wavelength per variant."""
    sin_rows = [r for r in rows if r["feature"].startswith("sin-")]
    if not sin_rows:
        return

    wavelengths = {"sin-1km": 1000, "sin-2km": 2000, "sin-4km": 4000}
    fig, ax = plt.subplots(figsize=(7, 5))
    for vname in sorted({r["variant"] for r in sin_rows}):
        pts = sorted(
            (wavelengths[r["feature"]], r["transfer"]) for r in sin_rows if r["variant"] == vname
        )
        if pts:
            ax.plot(*zip(*pts, strict=True), "o-", label=vname)

    ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Current feature wavelength (m)")
    ax.set_ylabel("Amplitude transfer (1 = perfect)")
    ax.set_title("Spatial transfer function of pipeline variants")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "transfer_function.png", dpi=150)
    plt.close(fig)


def _plot_feature_maps(features, fields_for_plots, fig_dir: Path, domain: float) -> None:
    """True vs recovered ux maps for each feature (field variants only)."""
    for fname, u_func, _desc in features:
        keys = [k for k in fields_for_plots if k[0] == fname and "field" in k[1]]
        if not keys:
            continue
        result = fields_for_plots[keys[-1]]
        fld = result.get("field")
        if fld is None:
            continue

        xg, yg = np.meshgrid(fld.x_centers, fld.y_centers)
        tux, _ = u_func(xg, yg)
        vmax = float(np.nanmax(np.abs(tux))) or 0.1

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        im0 = axes[0].pcolormesh(
            fld.x_centers, fld.y_centers, tux, cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[0].set_title(f"{fname}: true ux")
        rec = np.where(fld.coverage > 0, fld.ux, np.nan)
        axes[1].pcolormesh(fld.x_centers, fld.y_centers, rec, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"recovered ux ({keys[-1][1]})")
        diff = rec - tux
        axes[2].pcolormesh(
            fld.x_centers,
            fld.y_centers,
            diff,
            cmap="RdBu_r",
            vmin=-vmax / 2,
            vmax=vmax / 2,
        )
        axes[2].set_title("recovered - true")
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        fig.colorbar(im0, ax=axes, shrink=0.8, label="ux (m/s)")
        fig.savefig(fig_dir / f"feature_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _write_report(rows, features, out_dir: Path, n_xy: int, n_t: int, dx: float, quick: bool):
    path = out_dir / "resolution_validation.md"
    domain_km = n_xy * dx / 1000

    lines = [
        "# Spatial Resolution Validation",
        "",
        "Empirical resolution measurements of the current-extraction pipeline",
        "variants on synthetic wave cubes with **prescribed, spatially varying",
        "currents** (local plane-wave model, `wamos_tpw.synthetic`). Generated",
        f"by `tools/resolution_study.py`{' (--quick)' if quick else ''}:",
        f"{domain_km:.1f} km domain, {dx:.0f} m grid, {n_t} frames at 1.5 s,",
        "200 wave components (60-600 m wavelengths, directional spread 0.8 rad).",
        "",
        "## Variants",
        "",
        "- **tiles-2km** — independent 2 km windows at 1 km stride (classical",
        "  WaMoS-style tiling); output at tile centers.",
        "- **field-2+1** — joint regularized inversion of 2 km + 1 km windows",
        "  on a 250 m grid (correlation length 500 m).",
        "- **field-2+1+0.5** — adds 500 m windows.",
        "- **field-weak-prior** — same windows, prior allowing 4x steeper",
        "  gradients (correlation length 250 m, sigma_prior 0.5 m/s): the",
        "  smoothness prior is the resolution knob, trading noise",
        "  suppression for fine-scale response.",
        "",
        "## Metrics",
        "",
        "- **transfer** — regression of recovered anomaly onto true anomaly",
        "  (1 = feature fully recovered, 0 = invisible).",
        "- **rmse** — over both components at covered output cells (m/s).",
        "- **bias** — mean (ux, uy) error (m/s).",
        "- **calib** — median(|error| / reported 1-sigma); 1 = perfectly",
        "  calibrated errors, >1 = overconfident.",
        "",
        "## Results",
        "",
    ]

    for fname, _u, desc in features:
        feat_rows = [r for r in rows if r["feature"] == fname]
        if not feat_rows:
            continue
        lines += [f"### {fname} — {desc}", ""]
        lines += [
            "| variant | transfer | rmse (m/s) | bias ux/uy (m/s) | calib | n |",
            "|---|---|---|---|---|---|",
        ]
        for r in feat_rows:
            lines.append(
                f"| {r['variant']} | {r['transfer']:.2f} | {r['rmse']:.3f} | "
                f"{r['bias'][0]:+.3f} / {r['bias'][1]:+.3f} | "
                f"{r['calibration']:.1f} | {r['n']} |"
            )
        lines.append("")

    lines += [
        "## Figures",
        "",
        "![Transfer function](figures/resolution/transfer_function.png)",
        "",
    ]
    for fname, _u, _d in features:
        fig = out_dir / "figures" / "resolution" / f"feature_{fname}.png"
        if fig.exists():
            lines.append(f"![{fname}](figures/resolution/feature_{fname}.png)")
            lines.append("")

    lines += [
        "## Caveats",
        "",
        "- The synthetic model is the local plane-wave approximation: it",
        "  measures the estimator + inversion response, not wave-current",
        "  interaction physics, radar imaging nonlinearity, or shadowing.",
        "- A wave group traverses c_g x T (~0.3-0.8 km here) during a block,",
        "  which the frozen model does not include; real-data resolution at",
        "  short feature scales will be somewhat poorer.",
        "- `calib` > 1 means the reported errors are optimistic by that",
        "  factor; use it to scale error-based thresholds on real data.",
        "",
    ]

    path.write_text("\n".join(lines))
    logger.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument(
        "--quick", action="store_true", help="Small domain, two features (smoke test)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    run_study(args.out, args.quick)
    logger.info("Study complete in %.1f min", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
