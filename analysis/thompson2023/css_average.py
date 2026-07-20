"""Accumulate a long-time-average CSSAngaur intensity map in polar coords.

At a fixed station, intermittent reef breakers integrate into a
continuous bright band over hours — revealing the reef line (and the
Peleliu harbor breakthrough as a notch) for encoder-zero calibration.

Grid: azimuth encoder (8192 counts -> AZ_BINS) x 884 range cells.
Writes mean and bright-fraction (frac of frames > threshold) arrays.

Crash-safe: the accumulators are checkpointed atomically every
CKPT_EVERY frames to css_average.ckpt.npz, recording which files have
been folded in. Re-running resumes from the checkpoint and skips those
files, so an interrupted run loses at most CKPT_EVERY frames of work.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

AZ_BINS = 1024                 # ~0.35 deg per bin
THRESH = 60                    # 8-bit counts; breakers/land >> sea mean (~35)
CKPT_EVERY = 1000              # checkpoint cadence (frames) ~2 min at 8 files/s
OUT = Path("/Users/pat/Desktop/WAMOS/thompson2023/css_average.npz")
CKPT = Path("/Users/pat/Desktop/WAMOS/thompson2023/css_average.ckpt.npz")


def decode(fn: Path):
    raw = fn.read_bytes()
    i_nl = raw.find(b"\n", raw.find(b"EOH"))
    payload = np.frombuffer(raw[i_nl + 1 + 10:], dtype=np.uint8)
    n = len(payload) // 886
    a = payload[: n * 886].reshape(n, 886)
    az = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)) >> 3  # 8192 -> 1024
    return az, a[:, 2:]


def save_ckpt(acc, cnt, bright, done):
    """Atomically write the checkpoint (temp file + rename)."""
    tmp = CKPT.with_suffix(".tmp.npz")
    np.savez(tmp, acc=acc, cnt=cnt, bright=bright,
             done=np.array(sorted(done), dtype=object), az_bins=AZ_BINS,
             thresh=THRESH)
    os.replace(tmp, CKPT)


def main(day_dirs, step):
    acc = np.zeros((AZ_BINS, 884), np.float64)
    cnt = np.zeros((AZ_BINS, 884), np.int64)
    bright = np.zeros((AZ_BINS, 884), np.int64)
    done: set[str] = set()

    if CKPT.exists():
        c = np.load(CKPT, allow_pickle=True)
        acc, cnt, bright = c["acc"], c["cnt"], c["bright"]
        done = set(map(str, c["done"]))
        print(f"resumed from {CKPT.name}: {len(done)} frames already folded in")

    files = []
    for d in day_dirs:
        files.extend(sorted(Path(d).rglob("*.pol")))
    files = files[::step]
    todo = [fn for fn in files if str(fn) not in done]
    print(f"{len(files)} frames total, {len(todo)} remaining", flush=True)

    t0 = time.time()
    since_ckpt = 0
    for i, fn in enumerate(todo):
        try:
            az, data = decode(fn)
        except Exception as e:  # noqa: BLE001 — a bad sweep must not abort the run
            print(f"skip {fn.name}: {e}", flush=True)
            done.add(str(fn))
            continue
        np.add.at(acc, az, data.astype(np.float64))
        np.add.at(cnt, az, 1)
        np.add.at(bright, az, (data > THRESH).astype(np.int64))
        done.add(str(fn))
        since_ckpt += 1
        if since_ckpt >= CKPT_EVERY:
            save_ckpt(acc, cnt, bright, done)
            since_ckpt = 0
        if i % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / rate / 60 if rate else 0
            print(f"  {i}/{len(todo)}  {rate:.1f} f/s  ETA {eta:.0f} min", flush=True)

    save_ckpt(acc, cnt, bright, done)
    with np.errstate(invalid="ignore"):
        mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        bfrac = np.where(cnt > 0, bright / np.maximum(cnt, 1), np.nan)
    np.savez_compressed(OUT, mean=mean, bfrac=bfrac, n=cnt, az_bins=AZ_BINS)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    days = sys.argv[1:-1] or ["/Volumes/SeaChest/ARCTERX/2023/CSSAngaur/2023/05/20"]
    step = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 6
    main(days, step)
