"""Write css_average.npz from the latest checkpoint.

Lets an interrupted / early-stopped css_average.py run feed
css_georeg.py without re-processing: the checkpoint holds the raw
accumulators, so mean/bfrac are just derived and saved in the final
format css_georeg.py expects.
"""

from pathlib import Path

import numpy as np

CKPT = Path("/Users/pat/Desktop/WAMOS/thompson2023/css_average.ckpt.npz")
OUT = Path("/Users/pat/Desktop/WAMOS/thompson2023/css_average.npz")

c = np.load(CKPT, allow_pickle=True)
acc, cnt, bright = c["acc"], c["cnt"], c["bright"]
az_bins = int(c["az_bins"])
n_frames = len(c["done"])
with np.errstate(invalid="ignore"):
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    bfrac = np.where(cnt > 0, bright / np.maximum(cnt, 1), np.nan)
np.savez_compressed(OUT, mean=mean, bfrac=bfrac, n=cnt, az_bins=az_bins)
print(f"finalized {OUT} from {n_frames} frames "
      f"(max bright fraction {np.nanmax(bfrac):.3f})")
