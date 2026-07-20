"""Prepare the heavy inputs for a radar scene frame (May 19 15:00 demo).

Computes and caches to scene_data/shore_scene.npz:
  - shore background: time-mean CSSAngaur intensity on an earth grid,
    range-corrected (divided by the per-range-bin median so the
    range fall-off flattens out)
  - shore currents: CurrentMap.from_cube with the full pipeline
    safeguards (960-m Hann tiles, per-tile depth + 1.2 m, land mask,
    hetero inflation)
  - nav/wind window means for the insets

The renderer (scene_render.py) is then fast — the same split a movie
loop needs (prepare per frame time, render per frame).
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(message)s")
from wamos_tpw.cordc import FixedStation
from wamos_tpw.cordc_cube import build_cube
from wamos_tpw.current import CurrentMap

BASE = Path("/Users/pat/Desktop/WAMOS/thompson2023")
import sys
_t0 = sys.argv[1] if len(sys.argv) > 1 else "2023-05-19T15:00:00"
T0 = np.datetime64(_t0)
T1 = T0 + np.timedelta64(15, "m")
OUT_NPZ = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "scene_data/shore_scene.npz")
# wide domain: Angaur island+reef, bank, channel, Peleliu SW reef/harbor
CENTER = (6.945, 134.18)   # shifted west: Angaur west annulus in frame
HALF = 8000.0                    # m
SPACING = 20.0                   # m
DEG2M = 111_319.5

# ---- shore cube -----------------------------------------------------
station = FixedStation.from_yaml("css_angaur_2023")
_dt = str(T0)
_day, _hour = _dt[8:10], _dt[11:13]
D = Path(f"/Volumes/SeaChest/ARCTERX/2023/CSSAngaur/2023/05/{_day}/{_hour}")
_key = _dt[:4] + _dt[5:7] + _dt[8:10] + _dt[11:13] + _dt[14:16]
files = [f for f in sorted(D.glob("*.pol")) if f.name[:12] >= _key][:128]
if len(files) < 128:  # spill into the next hour
    _nh = f"{int(_hour)+1:02d}"
    D2 = Path(f"/Volumes/SeaChest/ARCTERX/2023/CSSAngaur/2023/05/{_day}/{_nh}")
    if D2.exists():
        files += sorted(D2.glob("*.pol"))[:128 - len(files)]
print(f"{len(files)} shore sweeps from {files[0].name}")
cube = build_cube(files, station, center_lat=CENTER[0], center_lon=CENTER[1],
                  half_size=HALF, grid_spacing=SPACING)

# background: time mean, range-corrected
mean_int = np.nanmean(cube.intensity, axis=0)
gx, gy = np.meshgrid(cube.x_centers, cube.y_centers)
tow_x = (station.longitude - CENTER[1]) * DEG2M * np.cos(np.radians(CENTER[0]))
tow_y = (station.latitude - CENTER[0]) * DEG2M
rng = np.hypot(gx - tow_x, gy - tow_y)
# smooth range-correction: per-bin medians interpolated continuously in
# range, so no stair-step rings radiate from the antenna
rbin = (rng / 250.0).astype(int)
centers, meds = [], []
for b in np.unique(rbin):
    m = (rbin == b) & np.isfinite(mean_int)
    if m.sum() < 50:
        continue
    med = np.median(mean_int[m])
    if med > 0:
        centers.append((b + 0.5) * 250.0)
        meds.append(med)
prof = np.interp(rng, np.array(centers), np.array(meds))
corr = np.where(np.isfinite(mean_int) & (prof > 0), mean_int / prof, np.nan)
lon2d = CENTER[1] + gx / (DEG2M * np.cos(np.radians(CENTER[0])))
lat2d = CENTER[0] + gy / DEG2M

# ---- shore currents through the real pipeline machinery -------------
cfg = {
    "current.sub_region_size": 960.0,
    "current.sub_region_overlap": 0.5,
    "current.min_snr": 1.2,
    "current.mask_seam": False,
    "current.land_mask": str(BASE / "palau_landmask.nc"),
    "current.max_land_fraction": 0.05,
    "current.depth_grid": "/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/Angaur_Peleliu_25m.nc",
    "current.depth_adjust": 1.2,
}
cm = CurrentMap.from_cube(cube, config=cfg)
good = np.isfinite(cm.ux)
print(f"shore CurrentMap: {good.sum()}/{cm.ux.size} tiles accepted")
q_lon = CENTER[1] + cm.tile_x_centers / (DEG2M * np.cos(np.radians(CENTER[0])))
q_lat = CENTER[0] + cm.tile_y_centers / DEG2M

# ---- nav / wind window means ----------------------------------------
gy_ds = xr.open_dataset("/Users/pat/Desktop/WAMOS/ship_tn417/gyro_sperry.nc")
sel = (gy_ds["time"].values >= T0) & (gy_ds["time"].values < T1)
h = np.radians(gy_ds["heading"].values[sel])
heading = float(np.degrees(np.arctan2(np.mean(np.sin(h)), np.mean(np.cos(h)))) % 360)
gp = xr.open_dataset("/Users/pat/Desktop/WAMOS/ship_tn417/gps_abxtwo.nc")
sel = (gp["time"].values >= T0) & (gp["time"].values < T1)
sog = float(np.nanmedian(gp["sog"].values[sel]))
ship_lat = float(np.nanmedian(gp["latitude"].values[sel]))
ship_lon = float(np.nanmedian(gp["longitude"].values[sel]))
md = xr.open_dataset(str(BASE / "scene_data/melded_cache.nc"))
sel = (md["time"].values >= T0) & (md["time"].values < T1)
wspd = float(np.nanmedian(md["windSpeedSonic"].values[sel]))
wd = np.radians(md["windDirectionSonic"].values[sel])
wdir = float(np.degrees(np.arctan2(np.nanmean(np.sin(wd)), np.nanmean(np.cos(wd)))) % 360)
print(f"nav: heading {heading:.0f}, sog {sog:.1f} m/s; wind {wspd:.1f} m/s from {wdir:.0f}")

np.savez_compressed(
    OUT_NPZ,
    bg=corr, bg_lon=lon2d, bg_lat=lat2d,
    q_lon=q_lon, q_lat=q_lat, ux=cm.ux, uy=cm.uy, snr=cm.snr,
    heading=heading, sog=sog, ship_lat=ship_lat, ship_lon=ship_lon,
    wspd=wspd, wdir=wdir,
    t0=str(T0), t1=str(T1),
)
print(f"wrote {OUT_NPZ}")
