"""Movie driver: render joint radar-scene frames at a fixed cadence and
assemble with ffmpeg.

Per frame time t:
  1. shore npz   <- scene_prepare.py t (skipped if cached)
  2. ship mosaic <- wamos files-pipeline [t, t+2min] (skipped if cached)
  3. ship currents <- nearest 15-min composite from the sweep dirs
  4. scene_render.render_frame(...) -> frames/frame_NNNN.png
Then: ffmpeg -> movie mp4.

Usage: movie_driver.py 2023-05-19T14:00 2023-05-19T16:00 [step_min]
"""

import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path("/Users/pat/Desktop/WAMOS/thompson2023")
VENV = "/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python"
SRC = "/Users/pat/Desktop/WAMOS/wamos_tpw/src"
COMPOSITE_DIRS = [BASE / "bank_sweep_20230519_full", BASE / "bank_sweep_20230519",
                  *sorted((BASE / "bank_sweep_may").glob("day*"))]
MDIR = BASE / "scene_data/movie"
FRAMES = MDIR / "frames"


def ensure_npz(t):
    # shore currents refresh on a 30-min grid (tidal scale); frames in
    # between reuse the covering npz
    t = t.astype("datetime64[s]")
    tn = (t.astype(int) // 1800) * 1800
    t = np.datetime64(int(tn), "s").astype("datetime64[m]")
    out = MDIR / f"shore_W8000_{str(t).replace(':', '')}.npz"
    if out.exists():
        return out
    subprocess.run([VENV, str(BASE / "scene_prepare.py"), str(t), str(out)],
                   env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"}, check=True,
                   capture_output=True)
    return out


def ensure_mosaic(t):
    """15-min fold background (v2 merge: gap fill), cached in folds_v2."""
    ts = str(t)
    tag = f"{ts[:10]}_{ts[11:13]}-{ts[14:16]}"
    hit = sorted(MDIR.glob(f"folds_v2/merged_{tag}-*.nc"))
    if hit:
        return hit[0]
    (MDIR / "folds_v2").mkdir(parents=True, exist_ok=True)
    t1 = t + np.timedelta64(15, "m")
    fmt = lambda x: str(x).replace("-", "").replace(":", "")[:15]
    subprocess.run([VENV, "-m", "wamos_tpw.cli", "files-pipeline", fmt(t), fmt(t1),
                    "/Volumes/SeaChest/ARCTERX/2023/WAMOS/POLAR",
                    "--config", str(BASE / "tn2023_wamos.yaml"),
                    "--ship-data", "/Users/pat/Desktop/WAMOS/ship_tn417",
                    "--window", "900", "--interpolate", "--format", "netcdf",
                    "-o", str(MDIR / "folds_v2"), "--no-progress"],
                   env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"}, check=True,
                   capture_output=True)
    hit = sorted(MDIR.glob(f"folds_v2/merged_{tag}-*.nc"))
    return hit[0] if hit else None


def find_composite(t):
    t = t.astype("datetime64[s]")
    tn = (t.astype(int) // 900) * 900
    ts = str(np.datetime64(int(tn), "s"))
    tag = f"{ts[:10]}_{ts[11:13]}-{ts[14:16]}"
    for d in COMPOSITE_DIRS:
        hit = sorted(glob.glob(str(d / f"current_composite_{tag}*.nc")))
        if hit:
            return hit[0]
    return None


def main():
    t0 = np.datetime64(sys.argv[1])
    t1 = np.datetime64(sys.argv[2])
    step_s = int(float(sys.argv[3]) * 60) if len(sys.argv) > 3 else 900
    n0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # resume: seed frame counter
    FRAMES.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE))
    import scene_render

    t, n = t0, n0
    while t < t1:
        label = str(t)[:16]
        try:
            npz = ensure_npz(t)
            mosaic = ensure_mosaic(t)
            comp = find_composite(t)
            if mosaic is None or comp is None:
                print(f"{label}: missing {'mosaic' if mosaic is None else 'composite'}, skip",
                      flush=True)
                t += np.timedelta64(step_s, "s"); continue
            out = FRAMES / f"frame_{n:04d}.png"
            scene_render.render_frame(npz, mosaic, comp, out,
                                      f"Ship (red) + shore (blue) currents — {label} UTC", t=t)
            print(f"{label}: frame {n:04d} done", flush=True)
            n += 1
        except Exception as e:  # noqa: BLE001 keep the reel going
            print(f"{label}: FAILED {e}", flush=True)
        t += np.timedelta64(step_s, "s")
    if n >= 2:
        mp4 = BASE / (sys.argv[4] if len(sys.argv) > 4 else "scene_movie.mp4")
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-framerate", "3",
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
