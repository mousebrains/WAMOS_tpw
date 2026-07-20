"""Extract per-ping digitized depths from TN417 Knudsen .kea files and
join ship position from the 1 Hz C-Nav GPS (the Knudsen had no nav
feed — its Lat/Long fields are zeroed).

$PKEL99 columns of interest: ddmmyyyy, hhmmss.sss, Depth (m), Valid,
Draft (m). Output: singlebeam_depths.csv with t, lat, lon, depth,
valid, draft + a quick QC report vs the calibrated bank truth.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xarray as xr

D = Path("/Volumes/SeaChest/ARCTERX/2023/R2R/TN417/singlebeam_knudsen3260/TN417/156934/data")
OUT = Path("/Users/pat/Desktop/WAMOS/thompson2023/singlebeam_depths.csv")
BANK = dict(lat0=6.916, lat1=6.941, lon0=134.186, lon1=134.209)  # RBR box

recs = []
for fn in sorted(D.glob("*.kea")):
    for line in fn.read_text(errors="replace").splitlines():
        if not line.startswith("$PKEL99,") or ",RecNum," in line:
            continue
        p = line.split(",")
        try:
            ts = datetime.strptime(p[3] + p[4][:6], "%d%m%Y%H%M%S").replace(
                tzinfo=timezone.utc
            ).timestamp() + float(p[4][6:] or 0)
            depth = float(p[7])
            valid = int(p[8])
            draft = float(p[9])
        except (ValueError, IndexError):
            continue
        recs.append((ts, depth, valid, draft))
a = np.array(recs)
print(f"{len(a)} pings from {len(list(D.glob('*.kea')))} kea files, "
      f"{datetime.fromtimestamp(a[:,0].min(), timezone.utc):%m-%d} .. "
      f"{datetime.fromtimestamp(a[:,0].max(), timezone.utc):%m-%d}")
print(f"valid flag counts: {dict(zip(*map(list, np.unique(a[:,2], return_counts=True))))}")
print(f"depth==0 pings: {(a[:,1]==0).sum()}  draft values: {np.unique(a[:,3])}")

g = xr.open_dataset("/Users/pat/Desktop/WAMOS/ship_tn417/gps_abxtwo.nc")
gt = g["time"].values.astype("datetime64[s]").astype(np.int64).astype(float)
gla, glo = g["latitude"].values, g["longitude"].values
lat = np.interp(a[:, 0], gt, gla)
lon = np.interp(a[:, 0], gt, glo)
in_nav = (a[:, 0] >= gt[0]) & (a[:, 0] <= gt[-1])

with open(OUT, "w") as f:
    f.write("t,lat,lon,depth,valid,draft\n")
    for i in np.where(in_nav)[0]:
        f.write(f"{a[i,0]:.3f},{lat[i]:.7f},{lon[i]:.7f},{a[i,1]:.2f},{int(a[i,2])},{a[i,3]:.2f}\n")
print(f"wrote {in_nav.sum()} nav-joined pings -> {OUT}")

# QC: pings over the RBR bank box, where truth is 19-21 m (sensor level)
bank = in_nav & (lat > BANK["lat0"]) & (lat < BANK["lat1"]) & \
       (lon > BANK["lon0"]) & (lon < BANK["lon1"]) & (a[:, 1] > 0)
db = a[bank, 1]
if len(db):
    print(f"\nbank-box pings: {len(db)}; depth median {np.median(db):.2f} m "
          f"(p10 {np.percentile(db,10):.2f}, p90 {np.percentile(db,90):.2f})")
    print("RBR truth at sensors: 18.9-21.2 m (bed ~+0.5-1 m deeper); "
          "DEM+1.2 m prediction ~19-21 m")
