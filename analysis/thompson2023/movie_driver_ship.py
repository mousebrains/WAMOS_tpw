"""Ship-only movie driver (no shore radar): mosaic background + ship
composites + tracks + tide strip + per-frame nav/wind insets.

Usage: movie_driver_ship.py 2023-05-13T22:00 2023-05-18T00:00 60 out.mp4
"""

import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

BASE = Path("/Users/pat/Desktop/WAMOS/thompson2023")
VENV = "/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python"
SRC = "/Users/pat/Desktop/WAMOS/wamos_tpw/src"
COMPOSITE_DIRS = [*sorted((BASE / "bank_sweep_may").glob("day*")),
                  BASE / "bank_sweep_20230519_full"]
MDIR = BASE / "scene_data/movie"
FRAMES = MDIR / "frames_ship"


def _open_retry(path, tries=4, wait=15):
    import time
    for i in range(tries):
        try:
            ds = xr.open_dataset(path)
            ds.load()          # materialize into memory
            ds.close()
            return ds
        except Exception as e:  # noqa: BLE001 transient SMB/netCDF failures
            print(f"open {path} failed ({e}); retry {i+1}/{tries}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"cannot open {path} after {tries} tries")


_gy = _open_retry("/Users/pat/Desktop/WAMOS/ship_tn417/gyro_sperry.nc")
_gp = _open_retry("/Users/pat/Desktop/WAMOS/ship_tn417/gps_abxtwo.nc")
_md = _open_retry(str(BASE / "scene_data/melded_cache.nc"))



def nav_at(t):
    t0, t1 = t, t + np.timedelta64(15, "m")
    s = (_gy["time"].values >= t0) & (_gy["time"].values < t1)
    h = np.radians(_gy["heading"].values[s])
    heading = float(np.degrees(np.arctan2(np.mean(np.sin(h)), np.mean(np.cos(h)))) % 360)
    s = (_gp["time"].values >= t0) & (_gp["time"].values < t1)
    sog = float(np.nanmedian(_gp["sog"].values[s]))
    lat = float(np.nanmedian(_gp["latitude"].values[s]))
    lon = float(np.nanmedian(_gp["longitude"].values[s]))
    s = (_md["time"].values >= t0) & (_md["time"].values < t1)
    wspd = float(np.nanmedian(_md["windSpeedSonic"].values[s]))
    wd = np.radians(_md["windDirectionSonic"].values[s])
    wdir = float(np.degrees(np.arctan2(np.nanmean(np.sin(wd)), np.nanmean(np.cos(wd)))) % 360)
    return {"heading": heading, "sog": sog, "ship_lat": lat, "ship_lon": lon,
            "wspd": wspd, "wdir": wdir}


def ensure_mosaic(t):
    ts = str(t)
    tag = f"{ts[:10]}_{ts[11:13]}-{ts[14:16]}"
    hit = sorted(MDIR.glob(f"mosaics_v2/merged_{tag}-0*.nc"))
    if hit:
        return hit[0]
    (MDIR / "mosaics_v2").mkdir(parents=True, exist_ok=True)
    t1 = t + np.timedelta64(2, "m")
    fmt = lambda x: str(x).replace("-", "").replace(":", "")[:15]
    r = subprocess.run([VENV, "-m", "wamos_tpw.cli", "files-pipeline", fmt(t), fmt(t1),
                        "/Volumes/SeaChest/ARCTERX/2023/WAMOS/POLAR",
                        "--config", str(BASE / "tn2023_wamos.yaml"),
                        "--ship-data", "/Users/pat/Desktop/WAMOS/ship_tn417",
                        "--window", "30", "--interpolate", "--grid-spacing", "25", "--format", "netcdf",
                        "-o", str(MDIR / "mosaics_v2"), "--no-progress"],
                       env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
                       capture_output=True)
    hit = sorted(MDIR.glob(f"mosaics_v2/merged_{tag}-0*.nc"))
    return hit[0] if hit else None


def find_composite(t):
    ts = str(t)
    tag = f"{ts[:10]}_{ts[11:13]}-{ts[14:16]}"
    for d in COMPOSITE_DIRS:
        hit = sorted(glob.glob(str(d / f"current_composite_{tag}*.nc")))
        if hit:
            return hit[0]
    return None


def main():
    t0 = np.datetime64(sys.argv[1])
    t1 = np.datetime64(sys.argv[2])
    step_s = int(float(sys.argv[3]) * 60)
    mp4 = BASE / (sys.argv[4] if len(sys.argv) > 4 else "scene_movie_ship.mp4")
    fixed = "--fixed" in sys.argv
    # geographically anchored frame: bank + channel + Peleliu SW coast
    EXTENT = (134.08, 134.28, 6.85, 7.08) if fixed else None
    FRAMES.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE))
    import scene_render

    t, n = t0, 0
    while t < t1:
        label = str(t)[:16]
        try:
            mosaic = ensure_mosaic(t)
            comp = find_composite(t)
            if mosaic is None or comp is None:
                print(f"{label}: missing {'mosaic' if mosaic is None else 'composite'}, skip",
                      flush=True)
                t += np.timedelta64(step_s, "s")
                continue
            scene_render.S = nav_at(t)
            scene_render.MOSAIC = str(mosaic)
            scene_render.COMPOSITE = str(comp)
            out = FRAMES / f"frame_{n:04d}.png"
            scene_render.render_ship(out=str(out),
                                     title="R/V Thompson WAMOS — backscatter + currents",
                                     t=t, extent=EXTENT)
            print(f"{label}: frame {n:04d} done", flush=True)
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"{label}: FAILED {e}", flush=True)
        t += np.timedelta64(step_s, "s")
    if n >= 2:
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-framerate", "4",
                        "-i", str(FRAMES / "frame_%04d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
                       check=True, capture_output=True)
        print(f"MOVIE DONE: {mp4} ({n} frames)", flush=True)
    else:
        print(f"only {n} frames, no movie", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
