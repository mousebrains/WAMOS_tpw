"""Build tracks.npz: all drifter + wave glider tracks for scene overlays.

Drifters: Drifter GPS/mwb{ID}d{NN}_gps_timeseries.mat (datenum, bursts).
Gliders:  any NetCDF under wavegliders/*/ with time+lat+lon.
Stored per platform: t (unix s), lat, lon — decimated to ~1 min.
"""
from pathlib import Path
import numpy as np
import scipy.io as sio
import xarray as xr

W = Path("/Volumes/SeaChest/ARCTERX/2023/Wake")
out = {}

for f in sorted((W / "Drifter GPS").glob("mwb*_gps_timeseries.mat")):
    m = sio.loadmat(f, squeeze_me=True)
    t = (np.asarray(m["time"], float).T.ravel() - 719529.0) * 86400.0
    la = np.asarray(m["lat"], float).T.ravel()
    lo = np.asarray(m["lon"], float).T.ravel()
    u = np.asarray(m["u"], float).T.ravel()
    v = np.asarray(m["v"], float).T.ravel()
    ok = np.isfinite(t) & np.isfinite(la) & np.isfinite(lo) & (la != 0)
    t, la, lo, u, v = t[ok], la[ok], lo[ok], u[ok], v[ok]
    o = np.argsort(t); t, la, lo, u, v = t[o], la[o], lo[o], u[o], v[o]
    keep = np.concatenate([[True], np.diff(t) > 55])   # ~1-min decimation
    name = f.stem.replace("_gps_timeseries", "")
    out[name] = np.vstack([t[keep], la[keep], lo[keep], u[keep], v[keep]])
    print(f"{name}: {keep.sum()} pts, "
          f"{np.datetime64(int(t[0]),'s')} .. {np.datetime64(int(t[-1]),'s')}")

for gdir in sorted((W / "wavegliders").iterdir()):
    if not gdir.is_dir():
        continue
    for nc in sorted(gdir.glob("*.nc")):
        try:
            ds = xr.open_dataset(nc)
        except Exception:
            continue
        names = {v.lower(): v for v in list(ds.variables)}
        if "time" in names and "lat" in names and "lon" in names:
            t = ds[names["time"]].values
            if np.issubdtype(t.dtype, np.datetime64):
                t = t.astype("datetime64[s]").astype(float)
            la = np.asarray(ds[names["lat"]].values, float).ravel()
            lo = np.asarray(ds[names["lon"]].values, float).ravel()
            n = min(len(t), len(la), len(lo))
            t, la, lo = t[:n], la[:n], lo[:n]
            ok = np.isfinite(t) & np.isfinite(la) & np.isfinite(lo)
            key = f"wg_{gdir.name}"
            if key in out or ok.sum() < 10:
                continue
            o = np.argsort(t[ok])
            nanv = np.full(ok.sum(), np.nan)
            out[key] = np.vstack([t[ok][o], la[ok][o], lo[ok][o], nanv, nanv])
            print(f"{key}: {ok.sum()} pts from {nc.name}, "
                  f"{np.datetime64(int(t[ok].min()),'s')} .. {np.datetime64(int(t[ok].max()),'s')}")
            break

np.savez_compressed("scene_data/tracks.npz", **out)
print(f"saved {len(out)} platforms -> scene_data/tracks.npz")
