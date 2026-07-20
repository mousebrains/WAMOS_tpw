"""Render Bjorn-style radar scene frames: range-corrected backscatter in
greyscale + current quiver, per radar and joint, with nav/wind polar
insets, bathymetry, mooring/PS markers, and drifter/wave-glider tracks.

Inputs:
  scene_data/shore_scene.npz          shore bg + CurrentMap + nav/wind
  scene_data/merged_*.nc              ship mosaic (local-meter grid)
  bank_sweep_*/current_composite_*    ship currents (15-min composites)
  scene_data/tracks.npz               drifter + wave glider tracks

Movie hook: render_frame(npz, mosaic, composite, out, title, t).
Tracks are drawn automatically for frame time t (trail + head marker).
"""

import glob

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/pat/Desktop/WAMOS/thompson2023"
MOSAIC = f"{BASE}/scene_data/merged_2023-05-19_15-00-46_to_2023-05-19_15-01-13.nc"
COMPOSITE = sorted(glob.glob(f"{BASE}/bank_sweep_20230519_full/current_composite_2023-05-19_15-00*.nc"))[0]
S = np.load(f"{BASE}/scene_data/shore_scene.npz")
try:
    TRACKS = dict(np.load(f"{BASE}/scene_data/tracks.npz"))
except FileNotFoundError:
    TRACKS = {}
T_FRAME = np.datetime64("2023-05-19T15:00")
TITLE_T = "2023-05-19 15:00-15:15 UTC"
TRAIL_HOURS = 2.0   # track tail length (h)
STAGE = (134.086, 134.269, 6.865, 7.025)  # display box (Pat 2026-07-19):
        # ship coverage outside the Angaur footprint stays visible

TOWER = (6.91677, 134.14840)
# pressure sensors (bank RBRs + C05 + Peleliu/Angaur wave gauges)
PS_SITES = [
    (6.92086, 134.19281), (6.93202, 134.19677), (6.92919, 134.20148),
    (6.93516, 134.20261), (6.93017, 134.19942), (6.9302, 134.1994),
    (6.98328, 134.21843), (6.97068, 134.22392), (6.97760, 134.23270),
    (6.92293, 134.13513), (6.92155, 134.14603), (6.91228, 134.15533),
    (6.90633, 134.15307),
]
QUIVER_KW = dict(scale=6.0, scale_units="inches", width=0.004, zorder=5)
DRIFTER_COLORS = ["orangered", "magenta", "gold", "tomato", "orchid",
                  "darkorange", "hotpink", "salmon"]
GLIDER_COLORS = {"wg_sven": "springgreen", "wg_emily": "turquoise",
                 "wg_ole": "lawngreen", "wg_ragnar": "aquamarine"}


def ship_background():
    """Merged mosaic intensity on lon/lat (x/y are LOCAL meters about
    the attrs center).

    The pipeline output is ALREADY deramped + dewinded — a SIGNED
    residual field (negative on the weak-signal side). Display it
    as-is: no re-normalization, no log, and NEVER mask <=0 (doing so
    amputated the negative half-plane and masqueraded as a "missing
    forward sector" until Pat caught it). NaN = genuinely no deposits.
    """
    ds = xr.open_dataset(MOSAIC)
    clat = float(ds.attrs["center_latitude"])
    clon = float(ds.attrs["center_longitude"])
    deg2m = 111_319.5
    lon = clon + ds["x"].values / (deg2m * np.cos(np.radians(clat)))
    lat = clat + ds["y"].values / deg2m
    LO, LA = np.meshgrid(lon, lat)
    resid = ds["intensity"].values.astype(float)
    X, Y = np.meshgrid(ds["x"].values, ds["y"].values)
    r = np.hypot(X, Y)
    resid[r < 150] = np.nan          # own-ship near field
    # aft shadow is masked PER SCAN in the pipeline (yaml shadow config),
    # so merged windows keep the astern strip filled by earlier scans as
    # the ship moves (Pat); no display-time cone mask.
    hd = float(ds.attrs.get("mean_ship_heading_deg", np.nan))
    rel = (np.degrees(np.arctan2(X, Y)) % 360 - hd) % 360
    # display flattening: remove the LEFTOVER look-angle profile
    # (order-2 dewind cannot capture strong non-sinusoidal up/downwind
    # asymmetry; without this the ship patch is light on one beam and
    # dark on the other, occluding shore structure in joint frames).
    # Median in (relative-azimuth x range) bins, subtracted; wave and
    # feature texture (deviations from the profile) survive.
    if np.isfinite(hd):
        naz, dr_m, r_max = 48, 400.0, 4400.0
        nr = int(r_max / dr_m)
        azb = np.clip((rel / (360.0 / naz)).astype(int), 0, naz - 1)
        rb = np.clip((r / dr_m).astype(int), 0, nr - 1)
        prof = np.full((naz, nr), np.nan)
        fin_m = np.isfinite(resid)
        key = (azb * nr + rb)[fin_m]
        vals = resid[fin_m]
        order = np.argsort(key, kind="stable")
        ks, vs = key[order], vals[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(ks)]])
        for s0, e0 in zip(starts, ends):
            if e0 - s0 >= 20:
                prof[ks[s0] // nr, ks[s0] % nr] = np.median(vs[s0:e0])
        # fill gaps along range, smooth in az (wrap) and range
        from scipy.ndimage import uniform_filter1d

        for a in range(naz):
            row = prof[a]
            good = np.isfinite(row)
            if good.sum() >= 2:
                prof[a] = np.interp(np.arange(nr), np.flatnonzero(good), row[good])
            elif good.any():
                prof[a] = row[good][0]
            else:
                prof[a] = 0.0
        prof = uniform_filter1d(prof, 5, axis=0, mode="wrap")
        prof = uniform_filter1d(prof, 3, axis=1, mode="nearest")
        # bilinear sample per pixel
        az_f = np.clip(rel / (360.0 / naz) - 0.5, 0, naz - 1)
        r_f = np.clip(r / dr_m - 0.5, 0, nr - 1)
        a0 = np.floor(az_f).astype(int); a1 = np.minimum(a0 + 1, naz - 1)
        r0 = np.floor(r_f).astype(int); r1 = np.minimum(r0 + 1, nr - 1)
        wa = az_f - a0; wr = r_f - r0
        prof_px = (prof[a0, r0] * (1 - wa) * (1 - wr) + prof[a1, r0] * wa * (1 - wr)
                   + prof[a0, r1] * (1 - wa) * wr + prof[a1, r1] * wa * wr)
        resid = resid - prof_px

    # light NaN-aware smoothing for 30-s snapshots (folds barely need it)
    from scipy.ndimage import uniform_filter

    filled = np.where(np.isfinite(resid), resid, 0.0)
    w = np.isfinite(resid).astype(float)
    sm = uniform_filter(filled, 3)
    sw = uniform_filter(w, 3)
    return LO, LA, np.where(sw > 0.4, sm / np.maximum(sw, 1e-9), np.nan)


def ship_currents():
    ds = xr.open_dataset(COMPOSITE)
    la, lo = ds["latitude"].values, ds["longitude"].values
    LO, LA = np.meshgrid(lo, la) if la.ndim == 1 else (lo, la)
    return LO, LA, ds["ux"].values, ds["uy"].values


def shore_bg_log():
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(np.where(S["bg"] > 0, S["bg"], np.nan))


def shore_quiver_mask():
    ux = np.where(S["snr"] >= 1.5, S["ux"], np.nan)
    uy = np.where(S["snr"] >= 1.5, S["uy"], np.nan)
    return ux, uy


FIXED_CLIM = (-0.8, 1.0)   # shore radar log10-ratio greyscale
FIXED_CLIM_SHIP = (-45.0, 60.0)  # ship residual counts, display-flattened
                           # (fixed across frames: no flicker, no cross-radar
                           # brightness mismatch)


def grey(ax, lon, lat, img, pmin=2, pmax=99, clim=None):
    if clim is None:
        v = img[np.isfinite(img)]
        clim = (np.percentile(v, pmin), np.percentile(v, pmax))
    ax.pcolormesh(lon, lat, img, cmap="gray", vmin=clim[0], vmax=clim[1],
                  shading="auto", zorder=1)


def add_bathy(ax):
    """25-m isobath, registered: DEM 23.8-m level (+1.2 m bias) shifted
    dN +90 m / dE -90 m into the radar/GPS frame."""
    from wamos_tpw.depthgrid import load_cached

    g = load_cached("/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/Angaur_Peleliu_25m.nc")
    deg2m = 111_319.5
    lat = g.lat0 + (np.arange(g.depth.shape[0]) + 0.5) * g.dlat + 90.0 / deg2m
    lon = g.lon0 + (np.arange(g.depth.shape[1]) + 0.5) * g.dlon - 90.0 / (
        deg2m * np.cos(np.radians(6.93))
    )
    # mask no-data: the DEM is only ~54% covered and its NaNs mark BOTH
    # land AND unsurveyed ocean — filling them with 0 painted phantom
    # "isobaths" along survey boundaries in deep water (Pat caught it).
    z = np.ma.masked_invalid(g.depth)
    ax.contour(lon, lat, z, levels=[23.8], colors="orange", linewidths=1.1, zorder=4)
    ax.plot([], [], color="orange", lw=1.1, label="25 m isobath (registered)")


def add_marks(ax, ship=True):
    ax.plot(TOWER[1], TOWER[0], "^", color="gold", ms=11, mec="k",
            label="Angaur tower", zorder=6)
    ps = np.array(PS_SITES)
    ax.plot(ps[:, 1], ps[:, 0], "o", color="white", ms=4.5, mec="k",
            mew=0.8, ls="none", label="PS", zorder=6)
    if ship:
        ax.plot(float(S["ship_lon"]), float(S["ship_lat"]), "o", color="cyan",
                ms=9, mec="k", label="R/V Thompson", zorder=6)


def add_tracks(ax, t=None):
    """Drifter + wave glider trails up to frame time t (+ head marker)."""
    if not TRACKS:
        return
    t = np.datetime64(t if t is not None else T_FRAME)
    t_unix = t.astype("datetime64[s]").astype(float)
    t0 = t_unix - TRAIL_HOURS * 3600
    di = 0
    seen_d = seen_g = False
    seen_v = False
    for name in sorted(TRACKS):
        arr = TRACKS[name]
        tt, la, lo = arr[0], arr[1], arr[2]
        uu = arr[3] if arr.shape[0] > 3 else np.full_like(tt, np.nan)
        vv = arr[4] if arr.shape[0] > 3 else np.full_like(tt, np.nan)
        m = (tt >= t0) & (tt <= t_unix)
        if m.sum() < 2:
            continue
        if name.startswith("wg_"):
            color = GLIDER_COLORS.get(name, "springgreen")
            label = None if seen_g else "wave gliders"
            seen_g = True
        else:
            color = DRIFTER_COLORS[di % len(DRIFTER_COLORS)]
            di += 1
            label = None if seen_d else "drifters"
            seen_d = True
        ax.plot(lo[m], la[m], "-", color=color, lw=1.4, alpha=0.9, zorder=6)
        ax.plot(lo[m][-1], la[m][-1], "o", color=color, ms=6, mec="k",
                mew=0.8, label=label, zorder=7)
        # drifter's OWN velocity (GPS-derived) at the same quiver scale
        # as the radar currents -> direct on-screen validation
        uh = np.nanmedian(uu[m][-10:]); vh = np.nanmedian(vv[m][-10:])
        if np.isfinite(uh) and np.isfinite(vh):
            ax.quiver(lo[m][-1], la[m][-1], uh, vh, color="black",
                      label=None if seen_v else "drifter velocity (GPS)",
                      **{**QUIVER_KW, "zorder": 8})
            seen_v = True


_TIDE = None


def _tide():
    global _TIDE
    if _TIDE is None:
        import csv
        t, h = [], []
        with open(f"{BASE}/scene_data/malakal_tide.csv") as f:
            for row in csv.reader(f):
                try:
                    t.append(np.datetime64(row[0].replace(" ", "T")))
                    h.append(float(row[1]))
                except (ValueError, IndexError):
                    continue
        _TIDE = (np.array(t), np.array(h))
    return _TIDE


def add_tide_strip(fig, t=None):
    """Malakal Harbor tide (NOAA 1770000, MSL) with a cursor at frame time."""
    tt, hh = _tide()
    t = np.datetime64(t if t is not None else T_FRAME)
    m = (tt >= t - np.timedelta64(36, "h")) & (tt <= t + np.timedelta64(36, "h"))
    if m.sum() < 10:
        return
    ax = fig.add_axes([0.30, 0.928, 0.40, 0.048])
    ax.plot(tt[m], hh[m], "-", color="steelblue", lw=1.0)
    ax.axvline(t, color="crimson", lw=1.4)
    ax.set_ylabel("m", fontsize=6)
    ax.set_title(f"Malakal tide (MSL) — {str(t)[:16]} UTC", fontsize=7, pad=2)
    ax.set_xticks(ax.get_xticks()[::2])
    ax.tick_params(labelsize=5)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def finish(ax, extent, title):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(extent[2:]))))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              framealpha=0.95, borderaxespad=0)


def nav_insets(fig):
    for pos, ang, label in (
        ([0.035, 0.875, 0.095, 0.095], float(S["heading"]),
         f"ship {float(S['sog']):.1f} m/s\nhdg {float(S['heading']):.0f}°"),
        ([0.885, 0.875, 0.095, 0.095], (float(S["wdir"]) + 180) % 360,
         f"wind {float(S['wspd']):.1f} m/s\nfrom {float(S['wdir']):.0f}°"),
    ):
        ax = fig.add_axes(pos, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_facecolor("#f2f2f2")
        ax.patch.set_alpha(0.85)
        ax.annotate("", xy=(np.radians(ang), 1.0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="crimson"))
        ax.set_rlim(0, 1.15)
        ax.set_rticks([])
        ax.set_thetagrids([0, 90, 180, 270], ["N", "E", "S", "W"], fontsize=6)
        ax.set_title(label, fontsize=7, pad=6)


def render_ship(out=None, title=None, t=None, extent=None):
    lon, lat, img = ship_background()
    LO, LA, ux, uy = ship_currents()
    fig, ax = plt.subplots(figsize=(12.5, 10))
    grey(ax, lon, lat, img, clim=FIXED_CLIM_SHIP)
    q = ax.quiver(LO, LA, ux, uy, color="crimson", **QUIVER_KW)
    ax.quiverkey(q, 0.87, 1.02, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    add_bathy(ax)
    add_marks(ax)
    add_tracks(ax, t)
    if extent is None:
        fin = np.isfinite(img)
        extent = (np.nanmin(lon[fin]), np.nanmax(lon[fin]),
                  np.nanmin(lat[fin]), np.nanmax(lat[fin]))
    finish(ax, extent, title or "R/V Thompson WAMOS — backscatter + currents")
    nav_insets(fig)
    add_tide_strip(fig, t)
    out = out or f"{BASE}/scene_ship.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def render_shore(out=None, title=None, t=None):
    fig, ax = plt.subplots(figsize=(12.5, 10))
    grey(ax, S["bg_lon"], S["bg_lat"], shore_bg_log(), clim=FIXED_CLIM)
    QLO, QLA = np.meshgrid(S["q_lon"], S["q_lat"])
    ux_d, uy_d = shore_quiver_mask()
    q = ax.quiver(QLO, QLA, ux_d, uy_d, color="dodgerblue", **QUIVER_KW)
    ax.quiverkey(q, 0.87, 1.02, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    add_bathy(ax)
    add_marks(ax)
    add_tracks(ax, t)
    finish(ax, STAGE, title or "CSS Angaur (shore radar) — backscatter + currents")
    add_tide_strip(fig, t)
    out = out or f"{BASE}/scene_shore.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def render_joint(out=None, title=None, t=None, tracks=None):
    lon_s, lat_s, img_s = ship_background()
    LO, LA, ux_s, uy_s = ship_currents()
    fig, ax = plt.subplots(figsize=(13.5, 10))
    grey(ax, S["bg_lon"], S["bg_lat"], shore_bg_log(), clim=FIXED_CLIM)
    grey(ax, lon_s, lat_s, img_s, clim=FIXED_CLIM_SHIP)
    QLO, QLA = np.meshgrid(S["q_lon"], S["q_lat"])
    ux_d, uy_d = shore_quiver_mask()
    q1 = ax.quiver(QLO, QLA, ux_d, uy_d, color="dodgerblue",
                   label="shore currents", **QUIVER_KW)
    ax.quiver(LO, LA, ux_s, uy_s, color="crimson", label="ship currents", **QUIVER_KW)
    ax.quiverkey(q1, 0.87, 1.02, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    add_bathy(ax)
    add_marks(ax)
    add_tracks(ax, t)
    lo0, lo1, la0, la1 = STAGE
    finish(ax, (lo0, lo1, la0, la1),
           title or "Joint scene — ship (red) + shore (blue) currents")
    nav_insets(fig)
    add_tide_strip(fig, t)
    out = out or f"{BASE}/scene_joint.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def render_frame(npz_path, mosaic_path, composite_path, out_png, title, t=None):
    """Movie hook: render one joint frame from per-time inputs."""
    global S, MOSAIC, COMPOSITE
    S = np.load(npz_path)
    MOSAIC = str(mosaic_path)
    COMPOSITE = str(composite_path)
    render_joint(out=str(out_png), title=title, t=t)


if __name__ == "__main__":
    render_ship()
    render_shore()
    render_joint()
