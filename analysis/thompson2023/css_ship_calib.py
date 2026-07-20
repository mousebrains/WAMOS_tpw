"""Fit the Angaur tower calibration from ship_detections.csv:

  1. ANGLE   theta0 = true bearing - raw encoder bearing (a constant).
  2. RANGE   robust regress rng_true vs range cell -> cell size (slope)
             + range offset (intercept).
  3. CLOCK   residual position vector = velocity vector * dt + const;
             ship TURNS rotate the velocity so dt separates from the
             constant offset. Joint LS over [res_x; res_y].

Runs on the partial or complete CSV. Compact (non-squall) detections
only. First-order constants; secondary structure (tower position,
GPS-antenna aspect) is reported but not removed, per plan.
"""

import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/Users/pat/Desktop/WAMOS/thompson2023/ship_detections.csv"
OUT = "/Users/pat/Desktop/WAMOS/thompson2023/ship_calib.png"

rows = [r for r in csv.DictReader(open(CSV)) if r["compact"] == "1"]
d = {k: np.array([float(r[k]) for r in rows]) for k in
     ("t", "sog", "vx", "vy", "rng_true", "brg_true", "cell_m", "word_c", "range_c")}
n = len(rows)
print(f"{n} compact detections  ({d['t'].min():.0f}..{d['t'].max():.0f} unix, "
      f"{(d['t'].max()-d['t'].min())/3600:.0f} h span)")

# ---------- JOINT Gauss-Newton fit: theta0, cell, offset, dt ----------
# obs_pos(theta0,cell,offset) must equal gps_pos(t_file) + v*dt   (dt = t_obs - t_file).
# A clock skew displaces the predicted ship by v*dt, biasing BOTH angle and
# range if fit separately, so solve all four together.
raw_brg = d["word_c"] / 8192 * 360
rc = d["range_c"]
gps_x = d["rng_true"] * np.sin(np.radians(d["brg_true"]))
gps_y = d["rng_true"] * np.cos(np.radians(d["brg_true"]))
vok = np.isfinite(d["vx"]) & np.isfinite(d["vy"])
vx, vy = np.nan_to_num(d["vx"]), np.nan_to_num(d["vy"])
DEG = np.pi / 180

theta0, cell, offset, dt = np.median((d["brg_true"] - raw_brg + 180) % 360 - 180), \
    np.median(d["cell_m"]), 0.0, 0.0
mask = vok.copy()
for it in range(6):
    brg = np.radians((raw_brg + theta0) % 360)
    rng = cell * rc + offset
    ox, oy = rng * np.sin(brg), rng * np.cos(brg)
    Rx = ox - gps_x - vx * dt
    Ry = oy - gps_y - vy * dt
    # rows: [dtheta0(deg) dcell doffset ddt]
    Jx = np.stack([rng * np.cos(brg) * DEG,  rc * np.sin(brg), np.sin(brg), -vx], 1)
    Jy = np.stack([-rng * np.sin(brg) * DEG, rc * np.cos(brg), np.cos(brg), -vy], 1)
    J = np.concatenate([Jx[mask], Jy[mask]])
    r = np.concatenate([Rx[mask], Ry[mask]])
    step, *_ = np.linalg.lstsq(J, -r, rcond=None)
    theta0 += step[0]; cell += step[1]; offset += step[2]; dt += step[3]
    # robust: re-mask on 3.5-MAD of residual magnitude
    resid = np.hypot(Rx, Ry)
    m = np.median(resid[vok]); s = 1.4826 * np.median(np.abs(resid[vok] - m))
    mask = vok & (resid < m + 3.5 * s)

# uncertainties from final linearization
sigma = np.sqrt(np.sum(r ** 2) / (len(r) - 4))
cov = sigma ** 2 * np.linalg.inv(J.T @ J)
se = np.sqrt(np.diag(cov))
resid = np.hypot(Rx, Ry)
th = (d["brg_true"] - raw_brg + 180) % 360 - 180   # for the plot / flatness

print(f"\n[JOINT FIT]  {mask.sum()} used, {sigma:.0f} m rms residual")
print(f"  theta0      = {theta0:8.3f} +/- {se[0]:.3f} deg true  (encoder zero, +ve = east of N)")
print(f"  cell size   = {cell:8.4f} +/- {se[1]:.4f} m/cell   (header {np.median(d['cell_m']):.4f})")
print(f"  range offset= {offset:8.1f} +/- {se[2]:.1f} m")
print(f"  clock skew  = {dt:8.2f} +/- {se[3]:.2f} s  (t_obs - t_filename)")
# tower-position check: is theta0 flat vs bearing AFTER the joint fit?
slope = np.polyfit(d["brg_true"][mask], th[mask], 1)[0]
print(f"  theta0 flatness vs bearing: slope {slope:+.4f} deg/deg (secondary; ignore for now)")
# named for the plots below
cell_fit, off_fit = cell, offset
res_x, res_y = (Rx + vx * dt), (Ry + vy * dt)   # residual before removing v*dt (for clock panel)
spd = np.hypot(vx, vy); cx = cy = 0.0

# ---------- plots ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
ax[0, 0].scatter(d["brg_true"], th, s=4, alpha=.4)
ax[0, 0].axhline(theta0, color="r", lw=1)
ax[0, 0].set(xlabel="true bearing (deg)", ylabel="theta0 estimate (deg)",
             title=f"ANGLE: theta0 = {theta0:.2f} deg (flat = encoder zero)")
ax[0, 1].scatter(d["range_c"], d["rng_true"], s=4, alpha=.4)
xx = np.array([d["range_c"].min(), d["range_c"].max()])
ax[0, 1].plot(xx, cell_fit * xx + off_fit, "r", lw=1)
ax[0, 1].set(xlabel="range cell", ylabel="GPS range (m)",
             title=f"RANGE: {cell_fit:.3f} m/cell, offset {off_fit:+.0f} m")
ax[1, 0].scatter(vx[mask], res_x[mask], s=4, alpha=.4, label="east")
ax[1, 0].scatter(vy[mask], res_y[mask], s=4, alpha=.4, label="north")
vv = np.array([vx[mask].min(), vx[mask].max()])
ax[1, 0].plot(vv, dt * vv, "r", lw=1)
ax[1, 0].set(xlabel="velocity component (m/s)", ylabel="obs-gps residual (m)",
             title=f"CLOCK: dt = {dt:+.2f} s (slope)")
ax[1, 0].legend()
sc = ax[1, 1].scatter(gps_x / 1000, gps_y / 1000, c=th, s=6, cmap="coolwarm",
                      vmin=theta0 - 1, vmax=theta0 + 1)
plt.colorbar(sc, ax=ax[1, 1], label="theta0 (deg)")
ax[1, 1].plot(0, 0, "k*", ms=12)
ax[1, 1].set(xlabel="east (km)", ylabel="north (km)", title="detections (color=theta0)")
ax[1, 1].set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print(f"\nsaved {OUT}")
