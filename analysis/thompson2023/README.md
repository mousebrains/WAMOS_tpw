# Thompson 2023 (TN417, Palau) — analysis scripts and calibrations

Campaign analysis for the May 2023 Hydrographer Bank / Angaur work:
calibration of the CSS Angaur shore radar and the R/V Thompson WAMOS,
depth-truth validation, and the movie/scene production drivers.

These scripts are point-in-time analysis records: they carry absolute
paths to the local data layout (`~/Desktop/WAMOS/...`,
`/Volumes/SeaChest/ARCTERX/...`) and expect the `wamos_tpw` package
from this repo (branch `thompson2023/depth-aware` or later).

## Key artifacts
- `tn2023_wamos.yaml` — tower config: measured `time_shift: -1.00 s`
  (hard-returns fine sweep) and the measured aft shadow cone
  (rel 140-220 deg, per-scan masking).
- `css_ship_detect.py` / `css_ship_calib.py` — ship-as-GPS-target
  calibration of the Angaur radar (theta0 18.836 deg, cell 15.0003 m,
  range offset -36.8 m, clock ~+23 s wandering +/-4 s/day).
- `ship_detections.csv`, `angaur_clock_series.csv` — the calibration
  evidence; `angaur_radial_words.csv` — CORDC format documentation.
- `scene_prepare.py` / `scene_render.py` / `movie_driver*.py` — the
  approved frame language and reel drivers (sliding 15-min windows,
  drifter Doppler arrows, Malakal tide strip).
- `PLAN.md` — living plan incl. the adversarial review (AR1-AR13).

Large derived files (folds, npz caches, detection CSVs from the
singlebeam work, movies) are intentionally NOT committed — all are
regenerable from the raw archives via these scripts.
