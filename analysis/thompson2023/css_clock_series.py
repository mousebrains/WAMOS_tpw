"""Time-varying Angaur recorder clock offset from the ship detections.

Geometry frozen at the pooled joint-fit values (theta0, cell, offset).
For each detection, search the actual 1 Hz GPS track for the time t*
whose position is closest to the radar-observed position:

    dt_i = t* - t_filename - intra_sweep(word)

This is exact through turns (no v*dt linearization) and yields a
cross-track miss distance as QC. Only detections whose position pins
time well (ship speed > MIN_SPD) are used.

Output: hourly-median clock series (CSV + PNG) + drift statistics.
"""

import csv
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THETA0 = 18.836
CELL = 15.0003
OFFSET = -36.8
ROT_S = 2.5                     # 24 rpm
MIN_SPD = 1.0                   # m/s
SEARCH = np.arange(-30.0, 120.0, 0.25)   # dt candidates (s) around t_filename
MAX_MISS = 60.0                 # m — reject detections whose best match is worse
TOWER = (6.91677, 134.14840)
KM = 111_319.5
CSV_IN = "/Users/pat/Desktop/WAMOS/thompson2023/ship_detections.csv"
CSV_OUT = "/Users/pat/Desktop/WAMOS/thompson2023/angaur_clock_series.csv"
PNG = "/Users/pat/Desktop/WAMOS/thompson2023/angaur_clock_series.png"

rows = [r for r in csv.DictReader(open(CSV_IN)) if r["compact"] == "1"]
d = {k: np.array([float(r[k]) for r in rows]) for k in
     ("t", "sog", "word_c", "range_c")}

# radar-observed ENU position (tower origin) under the frozen calibration
brg = np.radians((d["word_c"] / 8192 * 360 + THETA0) % 360)
rng = CELL * d["range_c"] + OFFSET
obs_x, obs_y = rng * np.sin(brg), rng * np.cos(brg)

# 1 Hz GPS track as ENU interpolants
g = xr.open_dataset("/Users/pat/Desktop/WAMOS/ship_tn417/gps_abxtwo.nc")
gt = g["time"].values.astype("datetime64[s]").astype(np.int64).astype(float)
gx = (g["longitude"].values - TOWER[1]) * KM * np.cos(np.radians(TOWER[0]))
gy = (g["latitude"].values - TOWER[0]) * KM

ok = d["sog"] > MIN_SPD
intra = (d["word_c"] / 8192) * ROT_S
dt_clock = np.full(len(rows), np.nan)
miss = np.full(len(rows), np.nan)
idx = np.where(ok)[0]
for i in idx:
    cand_t = d["t"][i] + SEARCH
    cx = np.interp(cand_t, gt, gx)
    cy = np.interp(cand_t, gt, gy)
    dist = np.hypot(cx - obs_x[i], cy - obs_y[i])
    j = dist.argmin()
    if dist[j] < MAX_MISS and 0 < j < len(SEARCH) - 1:
        dt_clock[i] = SEARCH[j] - intra[i]
        miss[i] = dist[j]

good = np.isfinite(dt_clock)
print(f"{ok.sum()} moving detections, {good.sum()} matched on track "
      f"(median miss {np.nanmedian(miss):.1f} m)")

# hourly median series
t = d["t"]
hr = np.floor(t / 3600).astype(int)
rows_out = []
for h in np.unique(hr[good]):
    s = good & (hr == h)
    if s.sum() < 5:
        continue
    med = np.median(dt_clock[s])
    mad = 1.4826 * np.median(np.abs(dt_clock[s] - med))
    rows_out.append((h * 3600 + 1800, med, mad / np.sqrt(s.sum()), int(s.sum()), mad))
arr = np.array(rows_out)
with open(CSV_OUT, "w") as f:
    f.write("t_mid_unix,clock_offset_s,se_s,n,mad_s\n")
    for r in rows_out:
        f.write(f"{r[0]:.0f},{r[1]:.3f},{r[2]:.3f},{r[3]},{r[4]:.3f}\n")
print(f"{len(rows_out)} hourly points -> {CSV_OUT}")
print(f"offset span {arr[:,1].min():.1f}..{arr[:,1].max():.1f} s, "
      f"within-hour MAD median {np.median(arr[:,4]):.2f} s")
dr = np.diff(arr[:, 1]) / np.diff(arr[:, 0]) * 86400
print(f"hour-to-hour drift: median {np.median(dr):+.1f} s/day, "
      f"IQR {np.percentile(dr,25):+.1f}..{np.percentile(dr,75):+.1f} s/day")

fig, axs = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1], sharex=True)
tt = [datetime.fromtimestamp(x, timezone.utc) for x in arr[:, 0]]
axs[0].errorbar(tt, arr[:, 1], yerr=arr[:, 2], fmt="o", ms=3, lw=0.8, capsize=2)
axs[0].axhline(18, color="r", lw=0.8, ls="--", label="GPS-UTC leap seconds (18 s)")
axs[0].set_ylabel("recorder clock - GPS time (s)")
axs[0].set_title("CSSAngaur recorder clock offset (hourly median, track-matched)")
axs[0].grid(alpha=0.3)
axs[0].legend()
axs[1].plot(tt, arr[:, 3], "o-", ms=3, lw=0.6)
axs[1].set_ylabel("n/hr")
axs[1].grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PNG, dpi=120)
print("saved", PNG)
