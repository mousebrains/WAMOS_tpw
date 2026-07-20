"""AR7: anchor the CSSAngaur encoder-zero on the tide-invariant SHORELINE
rather than the breaker band, and measure the breaker-band bias.

The Angaur_Peleliu DEM is bathymetry-only — Z is clipped at the
waterline (<10 cells above 0 m), so the islands are the NaN region and
'hard shore' = finite water (Z<0) adjacent to that NaN land. Wave
breaking forms on the seaward reef face (depth ~1.3*Hs, i.e. the 2-5 m
band) and migrates with tide/Hs, so the baseline css_georeg.py target
(depth<5 m, weighted toward the 3-5 m outer reef) biases theta0 along
the shore. Here we fit theta0 to:

  - the shoreline boundary (hard shore, tide-invariant)   [AR7 anchor]
  - narrow depth bands 0-1 / 1-3 / 3-5 m                   [drift probe]
  - depth<5 m (reproduces the baseline)

and report how theta0 drifts from the waterline out to the reef. That
drift is the breaker-band bias; adopt the shoreline value.
"""

import numpy as np
import xarray as xr
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOWER = (6.91677, 134.14840)
HARBOR = (6.9842963203036375, 134.21978846163344)
CELL_M = 7408.0 / 496.0
KM = 111_319.5
DEM = "/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/Angaur_Peleliu_25m.nc"
NPZ = "/Users/pat/Desktop/WAMOS/thompson2023/css_average.npz"
OUT = "/Users/pat/Desktop/WAMOS/thompson2023/css_georeg_ar7.png"

d = np.load(NPZ)
bfrac = d["bfrac"]
az_bins = int(d["az_bins"])
n_rng = bfrac.shape[1]
b = np.nan_to_num(bfrac, nan=0.0)
thetas = np.arange(0, 360, 0.2)


def project(lat, lon):
    dx = (lon - TOWER[1]) * KM * np.cos(np.radians(TOWER[0]))
    dy = (lat - TOWER[0]) * KM
    rng = np.hypot(dx, dy)
    brg = np.degrees(np.arctan2(dx, dy)) % 360
    keep = (rng > 500) & (rng < (n_rng - 1) * CELL_M)
    return rng[keep], brg[keep]


def fit(rng, brg, label):
    if len(rng) < 5:
        print(f"  {label:14s}: only {len(rng)} cells in window — SKIP")
        return None, None
    ri = (rng / CELL_M).astype(int)
    score = np.empty(len(thetas))
    for k, t0 in enumerate(thetas):
        ai = (((brg - t0) % 360) / 360 * az_bins).astype(int) % az_bins
        score[k] = b[ai, ri].mean()
    best = thetas[score.argmax()]
    print(f"  {label:14s}: theta0 = {best:6.1f} deg  (n={len(rng):4d}, "
          f"peak {score.max():.3f} vs median {np.median(score):.3f})")
    return best, score


ds = xr.open_dataset(DEM)
lat = ds["lat"].values
lon = ds["lon"].values
Z = ds["Z"].values
water = np.isfinite(Z) & (Z < 0)
depth = np.where(water, -Z, np.nan)
land = ~np.isfinite(Z)  # islands are the NaN region (DEM clipped at waterline)
# shoreline = shallow water cells touching the NaN land region
shore = water & ndimage.binary_dilation(land) & (depth < 50)

print("targets in radar window:")
results = {}
for label, mask in [
    ("shoreline", shore),
    ("depth 0-1 m", water & (depth > 0) & (depth <= 1)),
    ("depth 1-3 m", water & (depth > 1) & (depth <= 3)),
    ("depth 3-5 m", water & (depth > 3) & (depth <= 5)),
    ("depth<5 (base)", water & (depth > 0) & (depth < 5)),
]:
    rng, brg = project(lat[mask], lon[mask])
    results[label] = fit(rng, brg, label)

t_shore = results["shoreline"][0]
t_base = results["depth<5 (base)"][0]
if t_shore is not None and t_base is not None:
    dbias = ((t_base - t_shore + 180) % 360) - 180
    print(f"\nbreaker-band bias (baseline - shoreline) = {dbias:+.1f} deg")
    print(f"ADOPT theta0 = {t_shore:.1f} deg true (shoreline-anchored)")

# harbor-gap check under the shoreline theta0
best = t_shore if t_shore is not None else t_base
hdx = (HARBOR[1] - TOWER[1]) * KM * np.cos(np.radians(TOWER[0]))
hdy = (HARBOR[0] - TOWER[0]) * KM
h_rng, h_brg = np.hypot(hdx, hdy), np.degrees(np.arctan2(hdx, hdy)) % 360
print(f"harbor gap: bearing {h_brg:.2f} deg, range {h_rng:.0f} m -> "
      f"az bin {(((h_brg - best) % 360) / 360 * az_bins):.0f}, "
      f"range bin {h_rng / CELL_M:.0f}")

# ---- figure: score curves (left) + earth map under shoreline theta0 (right)
fig, axs = plt.subplots(1, 2, figsize=(17, 8))
ax = axs[0]
for label in ("shoreline", "depth 0-1 m", "depth 1-3 m", "depth 3-5 m", "depth<5 (base)"):
    best_l, score_l = results[label]
    if score_l is not None:
        ax.plot(thetas, score_l, lw=0.9, label=f"{label} ({best_l:.1f})")
if t_shore is not None:
    ax.axvline(t_shore, color="k", lw=1, ls="--")
ax.set_xlabel("encoder zero (deg true)")
ax.set_ylabel("mean bright fraction on target cells")
ax.set_title("theta0 by target: waterline vs breaker zone")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axs[1]
az_axis = (np.arange(az_bins) + 0.5) / az_bins * 360 + best
r_axis = (np.arange(n_rng) + 0.5) * CELL_M
A, R = np.meshgrid(np.radians(az_axis), r_axis, indexing="ij")
X = TOWER[1] + (R * np.sin(A)) / (KM * np.cos(np.radians(TOWER[0])))
Y = TOWER[0] + (R * np.cos(A)) / KM
sub = slice(None, None, 2)
pc = ax.pcolormesh(X[sub, sub], Y[sub, sub], bfrac[sub, sub],
                   cmap="inferno", vmin=0, vmax=0.6, shading="auto")
plt.colorbar(pc, ax=ax, label="bright fraction")
ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="c", linewidths=0.6)
ax.contour(lon, lat, np.nan_to_num(depth, nan=999), levels=[5], colors="lime",
           linewidths=0.5, linestyles=":")
ax.plot(HARBOR[1], HARBOR[0], "g^", ms=11, label="harbor gap")
ax.plot(TOWER[1], TOWER[0], "w*", ms=13, label="tower")
ax.set_xlim(134.07, 134.30)
ax.set_ylim(6.85, 7.06)
ax.set_aspect(1 / np.cos(np.radians(6.95)))
ax.legend(loc="lower right")
ax.set_title(f"bright fraction, shoreline-anchored (theta0 = {best:.1f} deg)\n"
             "cyan = coastline (land/NaN edge), green dotted = 5 m contour")
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("saved", OUT)
