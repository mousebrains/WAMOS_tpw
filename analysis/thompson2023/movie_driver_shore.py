"""Pure Angaur (shore radar only) reel: folded backscatter + shore
currents + tracks + tide, no ship products. Shares the shore npz cache
with the joint driver.

Usage: movie_driver_shore.py 2023-05-18T05:00 2023-05-23T00:00 60 ANGAUR.mp4
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path("/Users/pat/Desktop/WAMOS/thompson2023")
VENV = "/Users/pat/.local/pipx/venvs/wamos-tpw/bin/python"
SRC = "/Users/pat/Desktop/WAMOS/wamos_tpw/src"
MDIR = BASE / "scene_data/movie"
FRAMES = MDIR / "frames_shore"


def ensure_npz(t):
    # shore currents on a 30-min grid; frames between reuse the cover
    t = t.astype("datetime64[s]")
    tn = (t.astype(int) // 1800) * 1800
    t = np.datetime64(int(tn), "s").astype("datetime64[m]")
    out = MDIR / f"shore_W8000_{str(t).replace(':', '')}.npz"
    if out.exists():
        return out
    r = subprocess.run([VENV, str(BASE / "scene_prepare.py"), str(t), str(out)],
                       env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
                       capture_output=True)
    return out if out.exists() else None


def main():
    t0 = np.datetime64(sys.argv[1]); t1 = np.datetime64(sys.argv[2])
    step_s = int(float(sys.argv[3]) * 60)
    mp4 = BASE / (sys.argv[4] if len(sys.argv) > 4 else "ANGAUR.mp4")
    FRAMES.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE))
    import scene_render

    t, n = t0, 0
    while t < t1:
        label = str(t)[:16]
        try:
            npz = ensure_npz(t)
            if npz is None:
                print(f"{label}: no shore data, skip", flush=True)
                t += np.timedelta64(step_s, "s"); continue
            scene_render.S = np.load(npz)
            out = FRAMES / f"frame_{n:04d}.png"
            scene_render.render_shore(out=str(out),
                                      title="CSS Angaur — backscatter + currents", t=t)
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
