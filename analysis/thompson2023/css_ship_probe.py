"""Prototype: does the Thompson's radar echo sit where its GPS says, on
the Angaur scans? Measures observed-minus-GPS residual in bearing and
range under the baseline theta0 / nominal cell size — the residual is
the angle+range calibration.

Ship = a bright compact echo (Pat: 'very visible'). For each frame we
predict (encoder word, range bin) from the 1 Hz GPS and look for the
brightest cell in a box around it.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xarray as xr

TOWER = (6.91677, 134.14840)
KM = 111_319.5
CELL_M = 7408.0 / 496.0          # 14.935 m nominal
THETA0 = 24.2                    # baseline encoder zero (deg true)
SWEEP_LEN = 884
DAYDIR = Path("/Volumes/SeaChest/ARCTERX/2023/CSSAngaur/2023/05/20")


def decode(fn):
    raw = fn.read_bytes()
    i_nl = raw.find(b"\n", raw.find(b"EOH"))
    payload = np.frombuffer(raw[i_nl + 1 + 10:], dtype=np.uint8)
    n = len(payload) // (SWEEP_LEN + 2)
    a = payload[: n * (SWEEP_LEN + 2)].reshape(n, SWEEP_LEN + 2)
    word = a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)   # full 13-bit
    return word, a[:, 2:]


def fname_time(fn):
    s = fn.name[:14]
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]),
                    int(s[10:12]), int(s[12:14]), tzinfo=timezone.utc)


# ship GPS interpolators (1 Hz)
g = xr.open_dataset("/Users/pat/Desktop/WAMOS/ship_tn417/gps_abxtwo.nc")
gt = g["time"].values.astype("datetime64[s]").astype(np.int64)
gla, glo = g["latitude"].values, g["longitude"].values
gsog = g["sog"].values


def ship_at(ts):
    i = np.searchsorted(gt, ts)
    if i <= 0 or i >= len(gt):
        return None
    f = (ts - gt[i - 1]) / (gt[i] - gt[i - 1])
    la = gla[i - 1] + f * (gla[i] - gla[i - 1])
    lo = glo[i - 1] + f * (glo[i] - glo[i - 1])
    return la, lo, gsog[i]


# pick a slow, mid-range window on May 20
files = sorted(DAYDIR.rglob("*.pol"))
print(f"{len(files)} frames on 05/20")
cand = []
for fn in files:
    ts = int(fname_time(fn).timestamp())
    s = ship_at(ts)
    if s is None:
        continue
    la, lo, sog = s
    dx = (lo - TOWER[1]) * KM * np.cos(np.radians(TOWER[0]))
    dy = (la - TOWER[0]) * KM
    rng = np.hypot(dx, dy)
    if 3000 < rng < 9000 and abs(sog) < 1.2:
        cand.append((fn, ts, la, lo, rng))
print(f"{len(cand)} frames in slow (sog<1.2) mid-range (3-9km) window")
if not cand:
    sys.exit("no candidate window")

BRIGHT = 200    # ship saturates at 252; threshold well above sea/rain background
MAX_BLOB = 300  # cells
MAX_AZ = 20     # radials: a ship at these ranges subtends ~10-15; wider = squall/reef
MAX_R = 25      # range cells

res = []
for fn, ts, la, lo, rng_true in cand[::max(1, len(cand) // 120)]:
    word, data = decode(fn)
    dx = (lo - TOWER[1]) * KM * np.cos(np.radians(TOWER[0]))
    dy = (la - TOWER[0]) * KM
    brg_true = np.degrees(np.arctan2(dx, dy)) % 360
    w_pred = ((brg_true - THETA0) % 360) / 360 * 8192
    r_pred = rng_true / CELL_M
    dword = (word - w_pred + 4096) % 8192 - 4096
    rsel = np.where(np.abs(dword) < 137)[0]          # +/- ~6 deg
    rlo, rhi = int(max(0, r_pred - 80)), int(min(SWEEP_LEN, r_pred + 80))
    if len(rsel) == 0 or rhi <= rlo:
        continue
    block = data[rsel, rlo:rhi].astype(float)
    bright = block >= BRIGHT
    nblob = int(bright.sum())
    if nblob == 0:
        continue
    bi, bj = np.nonzero(bright)                       # bi -> radial in rsel, bj -> range
    az_ext = int(bi.max() - bi.min() + 1)
    r_ext = int(bj.max() - bj.min() + 1)
    compact = nblob <= MAX_BLOB and az_ext <= MAX_AZ and r_ext <= MAX_R
    w_blob = word[rsel[bi]].astype(float)
    w_un = ((w_blob - w_pred + 4096) % 8192 - 4096) + w_pred   # unwrap near prediction
    word_c = w_un.mean()
    range_c = rlo + bj.mean() + 0.5
    rng_obs = range_c * CELL_M
    brg_obs = (word_c / 8192 * 360 + THETA0) % 360
    d_rng = rng_obs - rng_true
    d_brg = ((brg_obs - brg_true + 180) % 360) - 180
    res.append((d_rng, d_brg, brg_true, rng_true, az_ext, compact))

res = np.array(res)
comp = res[res[:, 5] == 1]
print(f"{len(res)} detections, {len(comp)} compact (nblob<={MAX_BLOB}, azext<={MAX_AZ})")
db, dr, bt = comp[:, 1], comp[:, 0], comp[:, 2]
# robust (median) to shrug off any remaining contamination
print(f"\ncompact-detection residuals (obs - GPS):")
print(f"  bearing: mean {db.mean():+.2f}  median {np.median(db):+.2f}  std {db.std():.2f} deg  (n={len(comp)})")
print(f"  range  : mean {dr.mean():+.0f}  median {np.median(dr):+.0f}  std {dr.std():.0f} m")
# is the bearing offset flat across bearing? (constant = encoder zero; slope = other)
A = np.vstack([bt, np.ones_like(bt)]).T
slope, icpt = np.linalg.lstsq(A, db, rcond=None)[0]
print(f"  bearing residual vs brg_true: slope {slope:+.4f} deg/deg, intercept {icpt:+.2f} "
      f"(flat => pure encoder zero)")
print(f"\n  => theta0 (ship-absolute) = {THETA0} - {np.median(db):+.2f} = "
      f"{THETA0 - np.median(db):.2f} deg true   [reef fit was 24.2, GSHHS 26.2]")
print(f"     cell size nominal 14.935 m looks {'OK' if abs(np.median(dr))<50 else 'OFF'} "
      f"(median range residual {np.median(dr):+.0f} m)")
