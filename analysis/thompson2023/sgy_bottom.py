"""Batch bottom-picker for the TN417 Knudsen 3.5 kHz SEG-Y echograms.

The Knudsen's own digitized depths (.kea) lock onto bottom multiples in
<25 m water, so over Hydrographer Bank we re-pick the bottom from the
raw traces: first arrival exceeding a fraction of the trace maximum
after a transmit-ringdown blank. Depth convention matches the .kea
(draft 5.6 m included, 1500 m/s assumed) so the empirically calibrated
transfer depth_true = depth_kea_equiv / 0.960 applies downstream.

Writes one row per trace to sgy_bottom.csv (t_unix, depth_kea_equiv,
qual = peak/median). Resumable: already-processed files are skipped.
"""

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

D = Path("/Volumes/SeaChest/ARCTERX/2023/R2R/TN417/singlebeam_knudsen3260/TN417/156934/data")
OUT = Path("/Users/pat/Desktop/WAMOS/thompson2023/sgy_bottom.csv")
DRAFT = 5.6          # m, matches the .kea depth convention
C_HALF = 750.0       # 1500 m/s two-way, matches the .kea convention
BLANK_M = 10.5        # skip transmit ringdown (below transducer)
PICK_FRAC = 0.5      # first sample above this fraction of trace max



def pick_trace(d, delay_ms, dt_us):
    """Reverb-end then first-dominant-arrival pick.

    Returns (depth_kea_equiv, quality, clip_frac) or (None, 0, clip_frac).
    clip_frac = fraction of clipped samples 10-20 m below transducer -
    high values mean the bottom may hide inside receiver saturation
    (the <25 m bank case; drop those picks downstream).
    """
    ds = dt_us * 1e-6 * C_HALF
    i10, i20 = int(10 / ds), int(20 / ds)
    clip_frac = float((d[i10:i20] >= 32000).mean()) if delay_ms == 0 else 0.0
    if delay_ms == 0:
        run = max(1, int(2 / ds))
        below = d[i10:] < 0.05 * 32768
        c = np.convolve(below.astype(int), np.ones(run, int), "valid")
        j = np.flatnonzero(c == run)
        if len(j) == 0:
            return None, 0.0, clip_frac
        start = i10 + int(j[0])
    else:
        start = 0
    seg = d[start:]
    if seg.size < 10:
        return None, 0.0, clip_frac
    mx = int(seg.max())
    med = max(float(np.median(seg)), 1.0)
    if mx < 8 * med:
        return None, 0.0, clip_frac
    idx = start + int(np.argmax(seg > 0.5 * mx))
    depth = DRAFT + (delay_ms / 1000 + idx * dt_us * 1e-6) * C_HALF
    return depth, mx / med, clip_frac

def process(fn: Path, writer) -> int:
    raw = np.frombuffer(fn.read_bytes(), dtype=">i2")
    if len(raw) < 1920:
        return 0
    ns = int(raw[1610])            # binary header bytes 3221-3222 -> word 1610
    dt_us = int(raw[1608])         # bytes 3217-3218 -> word 1608
    fmt = int(raw[1612])           # bytes 3225-3226 -> word 1612
    if fmt != 3 or ns <= 0 or dt_us <= 0:
        raise ValueError(f"{fn.name}: unexpected format {fmt}/{ns}/{dt_us}")
    body = raw[1800:]
    stride = 120 + ns
    n = len(body) // stride
    tr = body[: n * stride].reshape(n, stride)
    hdr, data = tr[:, :120], np.abs(tr[:, 120:].astype(np.int32))
    delay_ms = hdr[:, 54].astype(np.int32)                     # bytes 109-110
    yr, day = hdr[:, 78].astype(int), hdr[:, 79].astype(int)   # bytes 157-160
    hh, mm, ss = (hdr[:, 80].astype(int), hdr[:, 81].astype(int), hdr[:, 82].astype(int))
    rows = 0
    for i in range(n):
        depth, qual, clip = pick_trace(data[i], int(delay_ms[i]), dt_us)
        if depth is None:
            continue
        try:
            t = datetime(int(yr[i]), 1, 1, tzinfo=timezone.utc).timestamp() \
                + (int(day[i]) - 1) * 86400 + int(hh[i]) * 3600 + int(mm[i]) * 60 + int(ss[i])
        except ValueError:
            continue
        writer.writerow((fn.name, f"{t:.0f}", f"{depth:.2f}", f"{qual:.0f}", f"{clip:.2f}"))
        rows += 1
    return rows


def main() -> None:
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r[0] for r in csv.reader(f) if r}
    files = sorted(D.glob("*.sgy"))
    todo = [f for f in files if f.name not in done]
    print(f"{len(files)} sgy files, {len(todo)} to do", flush=True)
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if not done:
        w.writerow(("file", "t", "depth", "qual", "clip"))
    t0 = time.time()
    for i, fn in enumerate(todo):
        try:
            n = process(fn, w)
        except Exception as e:  # noqa: BLE001 keep the batch going
            print(f"skip {fn.name}: {e}", flush=True)
            continue
        f.flush()
        if i % 20 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i}/{len(todo)}  {rate:.1f} files/s  ETA {(len(todo)-i-1)/rate/60:.0f} min",
                  flush=True)
    f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
