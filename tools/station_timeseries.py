#! /usr/bin/env python3
#
# Station time series, hodograph, and map from composite current files
#
# Jun-2026, Pat Welch, pat@mousebrains.com

"""Build a current time series from ``current_composite_*.nc`` files.

For a station occupation processed with ``wamos current
--composite-minutes N``, this produces:

- a time series of the spatially averaged (inverse-variance weighted)
  current with scatter-based error bars,
- a mean + semidiurnal (12.42 h) harmonic fit,
- a hodograph (ux vs uy trajectory, colored by time) — tidal rotation
  shows as a closed clockwise loop in the Northern Hemisphere,
- a quiver map of one mid-record composite, optionally with a target
  position marked.

Usage::

    python3 tools/station_timeseries.py <dir> [--target LAT,LON] [--prefix name]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_DEG2M = 111_319.5
_M2_HOURS = 12.42


def composite_series(directory: str, min_cells: int = 3, min_obs: int = 5):
    """Per-composite weighted-mean current with scatter-based errors."""
    rows = []
    for f in sorted(glob.glob(f"{directory}/current_composite_*.nc")):
        ds = xr.open_dataset(f)
        n = ds["n_obs"].values
        ux, uy = ds["ux"].values, ds["uy"].values
        ex, ey = ds["ux_err"].values, ds["uy_err"].values
        good = np.isfinite(ux) & (n >= min_obs)
        if good.sum() < min_cells:
            ds.close()
            continue
        wx = 1.0 / np.clip(ex[good], 0.005, None) ** 2
        wy = 1.0 / np.clip(ey[good], 0.005, None) ** 2
        t0 = np.datetime64(ds["time_start"].values)
        t1 = np.datetime64(ds["time_end"].values)
        rows.append(
            {
                "t": t0 + (t1 - t0) / 2,
                "ux": float(np.sum(wx * ux[good]) / np.sum(wx)),
                "uy": float(np.sum(wy * uy[good]) / np.sum(wy)),
                "sx": float(np.std(ux[good]) / np.sqrt(good.sum())),
                "sy": float(np.std(uy[good]) / np.sqrt(good.sum())),
            }
        )
        ds.close()
    return rows


def harmonic_fit(rows) -> None:
    """Mean + single semidiurnal harmonic; prints amplitudes and residual."""
    t0 = rows[0]["t"]
    th = np.array([(r["t"] - t0) / np.timedelta64(1, "h") for r in rows])
    ux = np.array([r["ux"] for r in rows])
    uy = np.array([r["uy"] for r in rows])
    om = 2 * np.pi / _M2_HOURS
    a = np.column_stack([np.ones_like(th), np.cos(om * th), np.sin(om * th)])
    cx, *_ = np.linalg.lstsq(a, ux, rcond=None)
    cy, *_ = np.linalg.lstsq(a, uy, rcond=None)
    resid = np.hypot(ux - a @ cx, uy - a @ cy)
    speed = float(np.hypot(cx[0], cy[0]))
    toward = float(np.degrees(np.arctan2(cx[0], cy[0])) % 360)
    print(f"mean current: ({cx[0]:+.3f}, {cy[0]:+.3f}) m/s = {speed:.2f} m/s toward {toward:.0f}")
    print(
        f"semidiurnal amplitude: ux {np.hypot(cx[1], cx[2]):.3f}, "
        f"uy {np.hypot(cy[1], cy[2]):.3f} m/s"
    )
    print(f"residual about mean+M2 fit: median {np.median(resid):.3f} m/s")


def plot_series(rows, out: Path, title: str) -> None:
    t = [r["t"].astype("datetime64[s]").astype("O") for r in rows]
    ux = np.array([r["ux"] for r in rows])
    uy = np.array([r["uy"] for r in rows])
    sx = np.array([r["sx"] for r in rows])
    sy = np.array([r["sy"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].errorbar(t, ux, yerr=sx, fmt="o-", capsize=3, label="ux (east)")
    axes[0].errorbar(t, uy, yerr=sy, fmt="s-", capsize=3, label="uy (north)")
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_ylabel("current (m/s)")
    axes[0].legend()
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    spd = np.hypot(ux, uy)
    direc = np.degrees(np.arctan2(ux, uy)) % 360
    ax2 = axes[1]
    ax2.plot(t, spd, "o-", color="tab:blue")
    ax2.set_ylabel("speed (m/s)", color="tab:blue")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(t, direc, "^--", color="tab:red", alpha=0.7)
    ax3.set_ylabel("direction toward (deg)", color="tab:red")
    ax3.set_ylim(0, 360)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_hodograph(rows, out: Path, title: str) -> None:
    t = np.array([r["t"] for r in rows])
    ux = np.array([r["ux"] for r in rows])
    uy = np.array([r["uy"] for r in rows])
    hours = (t - t[0]) / np.timedelta64(1, "h")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    sc = ax.scatter(ux, uy, c=hours, cmap="twilight", s=60, zorder=3)
    ax.plot(ux, uy, "-", color="gray", alpha=0.5, linewidth=1)
    for i in range(0, len(rows), max(1, len(rows) // 5)):
        stamp = str(t[i].astype("datetime64[m]")).split("T")[1]
        ax.annotate(stamp, (ux[i], uy[i]), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    plt.colorbar(sc, ax=ax, label="hours since start")
    ax.set_xlabel("ux east (m/s)")
    ax.set_ylabel("uy north (m/s)")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_map(directory: str, out: Path, target: tuple[float, float] | None) -> None:
    files = sorted(glob.glob(f"{directory}/current_composite_*.nc"))
    if not files:
        return
    f = files[len(files) // 2]
    ds = xr.open_dataset(f)
    n = ds["n_obs"].values
    ux, uy = ds["ux"].values, ds["uy"].values
    good = np.isfinite(ux) & (n >= 5)
    xg, yg = np.meshgrid(ds["x"].values, ds["y"].values)

    fig, ax = plt.subplots(figsize=(8, 7))
    spd = np.hypot(ux, uy)
    q = ax.quiver(
        xg[good],
        yg[good],
        ux[good],
        uy[good],
        spd[good],
        cmap="viridis",
        scale=3e-4,
        scale_units="xy",
        width=0.006,
    )
    ax.quiverkey(q, 0.85, 0.92, 0.2, "0.2 m/s", labelpos="E")
    plt.colorbar(q, ax=ax, label="speed (m/s)")
    if target is not None:
        ref_lat = float(ds.attrs["reference_latitude"])
        ref_lon = float(ds.attrs["reference_longitude"])
        tx = (target[1] - ref_lon) * _DEG2M * np.cos(np.deg2rad(ref_lat))
        ty = (target[0] - ref_lat) * _DEG2M
        ax.plot(tx, ty, "r*", markersize=16, label="target")
        ax.legend(loc="lower left")
    ax.set_xlabel("m east of reference")
    ax.set_ylabel("m north of reference")
    ax.set_title(f"15-min composite, {f.split('/')[-1][18:37]} UTC")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory with current_composite_*.nc files")
    parser.add_argument("--target", type=str, default=None, help="Mark LAT,LON on the map")
    parser.add_argument("--prefix", type=str, default="station", help="Output file prefix")
    parser.add_argument("--title", type=str, default="Station current time series")
    args = parser.parse_args()

    rows = composite_series(args.directory)
    print(f"{len(rows)} usable composites in {args.directory}")
    if len(rows) < 3:
        raise SystemExit("Not enough composites for a time series")

    out = Path(args.directory)
    harmonic_fit(rows)
    plot_series(rows, out / f"{args.prefix}_timeseries.png", args.title)
    plot_hodograph(rows, out / f"{args.prefix}_hodograph.png", args.title)
    target = None
    if args.target:
        lat, lon = (float(v) for v in args.target.split(","))
        target = (lat, lon)
    plot_map(args.directory, out / f"{args.prefix}_map.png", target)
    print(f"wrote {args.prefix}_timeseries/hodograph/map.png to {out}")


if __name__ == "__main__":
    main()
