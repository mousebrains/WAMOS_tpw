"""Fit the CSSAngaur encoder-zero orientation from the reef breaker band.

Projects DEM <5 m cells into (true bearing, range) from the tower and
finds the rotation theta0 (encoder zero relative to true north) that
maximizes the observed long-average bright fraction along the predicted
reef line. The Peleliu harbor breakthrough (a notch in the band at
exactly known coordinates) makes the optimum sharp.

Tower: 6.91677 N, 134.14840 E, antenna ~37.2 m ASL (Pat).
Cell size: RANGE/SCALE = 7408/496 = 14.935 m (Furuno API).
Harbor gap: 6.9842963 N, 134.2197885 E (Pat).
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wamos_tpw.depthgrid import DepthGrid

TOWER = (6.91677, 134.14840)
HARBOR = (6.9842963203036375, 134.21978846163344)
CELL_M = 7408.0 / 496.0
NPZ = "/Users/pat/Desktop/WAMOS/thompson2023/css_average.npz"
OUT = "/Users/pat/Desktop/WAMOS/thompson2023/css_georeg.png"

d = np.load(NPZ)
bfrac = d["bfrac"]
az_bins = int(d["az_bins"])
n_rng = bfrac.shape[1]

g = DepthGrid.from_netcdf("/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/Angaur_Peleliu_25m.nc")
glat = g.lat0 + (np.arange(g.depth.shape[0]) + 0.5) * g.dlat
glon = g.lon0 + (np.arange(g.depth.shape[1]) + 0.5) * g.dlon
iy, ix = np.nonzero(np.isfinite(g.depth) & (g.depth < 5))
km = 111_319.5
dx = (glon[ix] - TOWER[1]) * km * np.cos(np.radians(TOWER[0]))
dy = (glat[iy] - TOWER[0]) * km
rng = np.hypot(dx, dy)
brg = np.degrees(np.arctan2(dx, dy)) % 360
keep = (rng > 500) & (rng < (n_rng - 1) * CELL_M)
rng, brg = rng[keep], brg[keep]
ri = (rng / CELL_M).astype(int)
print(f"{len(rng)} DEM shallow cells inside radar window")

# score(theta0): mean bright fraction at predicted (az, range) cells
thetas = np.arange(0, 360, 0.2)
score = np.zeros(len(thetas))
b = np.nan_to_num(bfrac, nan=0.0)
for k, t0 in enumerate(thetas):
    ai = (((brg - t0) % 360) / 360 * az_bins).astype(int) % az_bins
    score[k] = b[ai, ri].mean()
best = thetas[score.argmax()]
print(f"best encoder zero theta0 = {best:.1f} deg true "
      f"(score {score.max():.3f} vs median {np.median(score):.3f})")

# harbor-gap check: bearing/range of the gap under the fitted theta0
hdx = (HARBOR[1] - TOWER[1]) * km * np.cos(np.radians(TOWER[0]))
hdy = (HARBOR[0] - TOWER[0]) * km
h_rng = np.hypot(hdx, hdy)
h_brg = np.degrees(np.arctan2(hdx, hdy)) % 360
print(f"harbor gap: true bearing {h_brg:.2f} deg, range {h_rng:.0f} m "
      f"-> az bin {(((h_brg - best) % 360) / 360 * az_bins):.0f}, "
      f"range bin {h_rng / CELL_M:.0f}")

fig, axs = plt.subplots(1, 2, figsize=(17, 8))
ax = axs[0]
ax.plot(thetas, score, lw=0.8)
ax.axvline(best, color="r", lw=1, label=f"theta0 = {best:.1f} deg")
ax.set_xlabel("encoder zero (deg true)")
ax.set_ylabel("mean bright fraction on predicted reef cells")
ax.legend()
ax.grid(alpha=0.3)

# earth-projected bright-fraction map under fitted theta0 + DEM overlay
ax = axs[1]
az_axis = (np.arange(az_bins) + 0.5) / az_bins * 360 + best
r_axis = (np.arange(n_rng) + 0.5) * CELL_M
A, R = np.meshgrid(np.radians(az_axis), r_axis, indexing="ij")
X = TOWER[1] + (R * np.sin(A)) / (km * np.cos(np.radians(TOWER[0])))
Y = TOWER[0] + (R * np.cos(A)) / km
sub = slice(None, None, 2)
pc = ax.pcolormesh(X[sub, sub], Y[sub, sub], bfrac[sub, sub],
                   cmap="inferno", vmin=0, vmax=0.6, shading="auto")
plt.colorbar(pc, ax=ax, label="bright fraction")
ax.contour(glon, glat, (np.isfinite(g.depth) & (g.depth < 5)).astype(float),
           levels=[0.5], colors="c", linewidths=0.7)
ax.plot(HARBOR[1], HARBOR[0], "g^", ms=11, label="harbor gap")
ax.plot(TOWER[1], TOWER[0], "w*", ms=13, label="tower")
ax.set_xlim(134.07, 134.30)
ax.set_ylim(6.85, 7.06)
ax.set_aspect(1 / np.cos(np.radians(6.95)))
ax.legend(loc="lower right")
ax.set_title(f"3-day bright fraction, earth-projected (theta0 = {best:.1f} deg)")
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("saved", OUT)
