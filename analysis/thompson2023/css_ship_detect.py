"""Detect the R/V Thompson on every (stepped) Angaur frame over May 18-23
and write one row per detection to ship_detections.csv.

The ship is a saturated (I=252) compact echo. For each frame we predict
(encoder word, range cell) from the 1 Hz TN417 GPS, take the intensity
centroid of the compact bright blob in a box around it, and record the
raw observation (encoder word_c, range cell_c) plus GPS truth and the
ship velocity vector. The calibration itself is fit downstream by
css_ship_calib.py from these raw observations, so nothing here bakes in
a theta0 or cell size.

Crash-safe: rows are appended and flushed; a re-run reads back the
'file' column and skips frames already recorded. Squalls are recorded
too (compact=0) but flagged so the fit can drop them.

Usage:  css_ship_detect.py [smoke_limit]
"""

import csv
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

TOWER = (6.91677, 134.14840)
KM = 111_319.5
PRED_THETA0 = 19.0          # box-centering ONLY (ship-absolute prelim); fit is independent
NOMINAL_CELL = 7408.0 / 496.0
SWEEP_LEN = 884
BRIGHT = 200                # ship saturates at 252
BOX_AZ_DEG = 8.0            # +/- search half-width in bearing
BOX_R_CELL = 100           # +/- search half-width in range cells
MAX_BLOB, MAX_AZ, MAX_R = 300, 20, 25   # compactness gate (squall/reef rejection)
STEP = 6                    # ~1 frame / 30 s; dense enough, 6x fewer slow SeaChest reads
DAYS = [18, 19, 20, 21, 22, 23]
BASE = Path("/Volumes/SeaChest/ARCTERX/2023/CSSAngaur/2023/05")
GPS = "/Users/pat/Desktop/WAMOS/ship_tn417/gps_abxtwo.nc"
OUT = Path("/Users/pat/Desktop/WAMOS/thompson2023/ship_detections.csv")
HDR = ("file,t,lat,lon,sog,vx,vy,rng_true,brg_true,cell_m,word_c,range_c,"
       "rng_obs,brg_obs,nblob,az_ext,r_ext,compact\n")

_num = r"([-\d.]+)"


def cell_size(raw):
    head = raw[: raw.find(b"EOH")].decode("latin1", "ignore")
    mr = re.search(rf"RANGE\s+{_num}", head)
    ms = re.search(rf"SCALE\s+{_num}", head)
    if mr and ms and float(ms.group(1)) > 0:
        return float(mr.group(1)) * 1000.0 / float(ms.group(1))
    return NOMINAL_CELL


def decode(fn):
    raw = fn.read_bytes()
    cm = cell_size(raw)
    i_nl = raw.find(b"\n", raw.find(b"EOH"))
    payload = np.frombuffer(raw[i_nl + 1 + 10:], dtype=np.uint8)
    n = len(payload) // (SWEEP_LEN + 2)
    a = payload[: n * (SWEEP_LEN + 2)].reshape(n, SWEEP_LEN + 2)
    word = a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)
    return word, a[:, 2:], cm


def fname_ts(fn):
    s = fn.name[:14]
    return int(datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]),
                        int(s[10:12]), int(s[12:14]), tzinfo=timezone.utc).timestamp())


g = xr.open_dataset(GPS)
gt = g["time"].values.astype("datetime64[s]").astype(np.int64)
gla, glo, gsog = g["latitude"].values, g["longitude"].values, g["sog"].values


def ship_at(ts):
    i = np.searchsorted(gt, ts)
    if i <= 0 or i >= len(gt):
        return None
    f = (ts - gt[i - 1]) / (gt[i] - gt[i - 1])
    return (gla[i - 1] + f * (gla[i] - gla[i - 1]),
            glo[i - 1] + f * (glo[i] - glo[i - 1]), float(gsog[i]))


def enu(la, lo):
    return ((lo - TOWER[1]) * KM * np.cos(np.radians(TOWER[0])), (la - TOWER[0]) * KM)


def vel(ts):
    a, b = ship_at(ts - 5), ship_at(ts + 5)
    if a is None or b is None:
        return (np.nan, np.nan)
    ax, ay = enu(a[0], a[1])
    bx, by = enu(b[0], b[1])
    return ((bx - ax) / 10.0, (by - ay) / 10.0)


def main(limit):
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r["file"] for r in csv.DictReader(f)}
        print(f"resume: {len(done)} rows already recorded", flush=True)
    fout = open(OUT, "a")
    if not done:
        fout.write(HDR)

    files = []
    for d in DAYS:
        files.extend(sorted((BASE / f"{d:02d}").rglob("*.pol")))
    files = files[::STEP]
    todo = [f for f in files if f.name not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(files)} stepped frames, {len(todo)} to do (step={STEP})", flush=True)

    t0 = time.time()
    nrec = 0
    for i, fn in enumerate(todo):
        ts = fname_ts(fn)
        s = ship_at(ts)
        if s is None:
            continue
        la, lo, sog = s
        ex, ey = enu(la, lo)
        rng_true = np.hypot(ex, ey)
        if rng_true < 500 or rng_true > 13600:      # out of radar window (pre-decode skip)
            continue
        brg_true = np.degrees(np.arctan2(ex, ey)) % 360
        try:
            word, data, cm = decode(fn)
        except Exception:                            # noqa: BLE001 a bad frame must not stop the run
            continue
        if rng_true > (SWEEP_LEN - 1) * cm:
            continue
        w_pred = ((brg_true - PRED_THETA0) % 360) / 360 * 8192
        r_pred = rng_true / cm
        dword = (word - w_pred + 4096) % 8192 - 4096
        rsel = np.where(np.abs(dword) < BOX_AZ_DEG / 360 * 8192)[0]
        rlo = int(max(0, r_pred - BOX_R_CELL))
        rhi = int(min(SWEEP_LEN, r_pred + BOX_R_CELL))
        if len(rsel) == 0 or rhi <= rlo:
            continue
        block = data[rsel, rlo:rhi].astype(float)
        bright = block >= BRIGHT
        nblob = int(bright.sum())
        if nblob == 0:
            continue
        bi, bj = np.nonzero(bright)
        az_ext = int(bi.max() - bi.min() + 1)
        r_ext = int(bj.max() - bj.min() + 1)
        compact = int(nblob <= MAX_BLOB and az_ext <= MAX_AZ and r_ext <= MAX_R)
        w_blob = word[rsel[bi]].astype(float)
        w_un = ((w_blob - w_pred + 4096) % 8192 - 4096) + w_pred
        word_c = w_un.mean() % 8192
        range_c = rlo + bj.mean() + 0.5
        rng_obs = range_c * cm
        brg_obs = (word_c / 8192 * 360 + PRED_THETA0) % 360
        vx, vy = vel(ts)
        fout.write(f"{fn.name},{ts},{la:.7f},{lo:.7f},{sog:.3f},{vx:.3f},{vy:.3f},"
                   f"{rng_true:.1f},{brg_true:.3f},{cm:.4f},{word_c:.2f},{range_c:.2f},"
                   f"{rng_obs:.1f},{brg_obs:.3f},{nblob},{az_ext},{r_ext},{compact}\n")
        nrec += 1
        if i % 500 == 0:
            fout.flush()
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / rate / 60 if rate else 0
            print(f"  {i}/{len(todo)}  {rate:.1f} f/s  {nrec} det  ETA {eta:.0f} min", flush=True)
    fout.flush()
    fout.close()
    print(f"done: {nrec} new detections -> {OUT}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
