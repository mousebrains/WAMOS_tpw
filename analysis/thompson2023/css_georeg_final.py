"""Final CSSAngaur encoder-zero: anchored on the Peleliu hard-land coast
(independent GSHHS coastline.mat), cross-checked by the harbor-gap notch.

Result of the AR7 investigation: the baseline breaker-zone fit (24.2 deg)
and the independent Peleliu hard-land fit (26.2 deg) agree to ~2 deg, so
the breaker-band bias is small. The degenerate near-tower Angaur coast
and the bathy-NaN edge are NOT usable anchors (they smear the fit). We
adopt the Peleliu hard-land value and verify the harbor gap lands in a
dark notch flanked by bright coast.
"""

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOWER = (6.91677, 134.14840)
HARBOR = (6.9842963203036375, 134.21978846163344)
CELL_M = 7408.0 / 496.0
KM = 111_319.5
THETA0 = 26.2  # Peleliu hard-land anchored (deg true)

d = np.load("css_average.npz")
bfrac = d["bfrac"]
az_bins = int(d["az_bins"])
n_rng = bfrac.shape[1]

# --- harbor-gap notch check
hdx = (HARBOR[1] - TOWER[1]) * KM * np.cos(np.radians(TOWER[0]))
hdy = (HARBOR[0] - TOWER[0]) * KM
h_rng = np.hypot(hdx, hdy)
h_brg = np.degrees(np.arctan2(hdx, hdy)) % 360
h_az = int(((h_brg - THETA0) % 360) / 360 * az_bins) % az_bins
h_ri = int(h_rng / CELL_M)
prof = np.nanmean(bfrac[:, h_ri - 4:h_ri + 5], axis=1)  # az profile at harbor range
win = np.arange(h_az - 12, h_az + 13) % az_bins
print(f"harbor gap: bearing {h_brg:.1f} deg -> az bin {h_az}, range bin {h_ri} ({h_rng:.0f} m)")
print(f"  bright fraction at harbor az {h_az}: {prof[h_az]:.3f}")
print(f"  flanking coast (az {h_az-8}, {h_az+8}): {prof[(h_az-8)%az_bins]:.3f}, {prof[(h_az+8)%az_bins]:.3f}")
print(f"  is harbor az a local min in +/-12 bins? {'YES' if prof[h_az]==prof[win].min() else 'no'}")

# --- coastline for overlay
cl = sio.loadmat("/Volumes/SeaChest/ARCTERX/2023/Wake/bathy/coastline.mat")["coastline"]
clat = np.concatenate([np.asarray(cl["Lat"][i, 0]).ravel() for i in range(cl.shape[0])])
clon = np.concatenate([np.asarray(cl["Lon"][i, 0]).ravel() for i in range(cl.shape[0])])

# --- earth-projected map under THETA0, zoomed on Peleliu
az_axis = (np.arange(az_bins) + 0.5) / az_bins * 360 + THETA0
r_axis = (np.arange(n_rng) + 0.5) * CELL_M
A, R = np.meshgrid(np.radians(az_axis), r_axis, indexing="ij")
X = TOWER[1] + (R * np.sin(A)) / (KM * np.cos(np.radians(TOWER[0])))
Y = TOWER[0] + (R * np.cos(A)) / KM

fig, ax = plt.subplots(figsize=(10, 9))
pc = ax.pcolormesh(X, Y, bfrac, cmap="inferno", vmin=0, vmax=0.6, shading="auto")
plt.colorbar(pc, ax=ax, label="bright fraction")
ax.plot(clon, clat, ".", color="cyan", ms=1.0, label="GSHHS coastline")
# predicted harbor bearing ray
ray_r = np.array([500, (n_rng - 1) * CELL_M])
ax.plot(TOWER[1] + ray_r * np.sin(np.radians(h_brg)) / (KM * np.cos(np.radians(TOWER[0]))),
        TOWER[0] + ray_r * np.cos(np.radians(h_brg)) / KM,
        "g--", lw=0.8, label="harbor bearing")
ax.plot(HARBOR[1], HARBOR[0], "g^", ms=12, label="harbor gap")
ax.plot(TOWER[1], TOWER[0], "w*", ms=14, label="tower")
ax.set_xlim(134.14, 134.28)
ax.set_ylim(6.94, 7.04)
ax.set_aspect(1 / np.cos(np.radians(6.99)))
ax.legend(loc="lower left", fontsize=9)
ax.set_title(f"CSSAngaur bright fraction, Peleliu-anchored theta0 = {THETA0} deg true\n"
             "(baseline breaker-zone fit was 24.2 deg -> ~2 deg breaker bias)")
fig.tight_layout()
fig.savefig("css_georeg_final.png", dpi=130)
print("saved css_georeg_final.png")
