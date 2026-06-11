# Real-Data Findings: Revelle 2022 and Thompson 2025

First real-data shakedown of the rebuilt current-extraction pipeline
(review + phases 1–3) on two ships and radar installations:

- **R/V Roger Revelle**, 2022-04-05 14:00–14:15 UTC, off Wedding Cake
  (Mt. Taipingot), Rota (14.09°N 145.13°E, ARCTERX). One frame per `.pol.zst` file, 1.42 s
  rotation, 2514 × 1552 polar bins (~7 km range), ship nearly stationary
  (0.5 m/s). Ship-instrument NetCDFs available (`--ship-data`).
- **R/V Thomas G. Thompson**, 2025-01-16 20:00–20:15 UTC, near Saipan
  (15.40°N 145.44°E). One frame per `.pol` file, 1.5 s rotation,
  2142 × 752 bins (~3.5 km range), **underway at 4.2 m/s**. No ship
  NetCDFs — per-frame header metadata only.

38 analysis blocks per ship (32 frames, 50 % overlap, 2 km windows,
`--min-snr 0` to capture full distributions), plus 15-minute composites.

## Headline results

| | Revelle 2022 | Thompson 2025 |
|---|---|---|
| median current (gated) | 0.12 m/s @ 081° | 0.55 m/s @ 258° |
| block-to-block std (ux, uy) | (0.15, 0.23) m/s | (0.11, 0.08) m/s |
| usable tiles per block | ~18 | ~9 |
| usable range | ≲ 3 km | ≲ 3 km (full disc) |

The Thompson current is physically sensible — westward at ~0.5 m/s at
15°N in the Marianas is the North Equatorial Current — and is stable
block to block **even with the ship underway at 8 knots**, which
exercises the motion compensation end to end. The Revelle window shows
weak (0.1–0.3 m/s) currents off Rota; at that amplitude single blocks
are noise-dominated and the 15-minute composites are the usable product.

## Finding 1: wave signal is usable only to ~3 km range

Per-tile vector scatter (same earth cell across blocks) vs range from
the radar, Revelle, SNR ≥ 1.2:

| range ring | scatter (m/s) | median SNR |
|---|---|---|
| 0–2 km | 0.51 | 3.6 |
| 2–3 km | 0.84 | 2.6 |
| 3–4 km | 1.30 | 1.7 |
| 4–5 km | 1.85 | 1.4 |
| 5–7 km | 2.6 (≈ search radius: pure noise) | 1.4 |

Tiles beyond ~3–4 km return vectors pinned at the search radius. The
Thompson, whose disc only reaches 3.5 km, is mostly inside the good
zone (0–2 km scatter: 0.16 m/s).

**Action (implemented):** `current.max_tile_range` /
`--max-tile-range` masks tiles beyond a configured distance from the
radar (uses the ship track already carried by the cube). Recommended
value for these installations: **3000 m**.

## Finding 2: data-driven SNR thresholds

Threshold sweep against per-cell temporal scatter:

- **Thompson:** razor-sharp quality transition at SNR ≈ 1.05
  (scatter 1.03 m/s below → 0.24 m/s above). **min_snr ≈ 1.1–1.2**
  keeps ~50 % of tiles at ~0.18 m/s scatter.
- **Revelle:** no sharp transition — SNR degrades continuously with
  range; after range gating to 3 km, SNR ≥ 1.2 behaves comparably.

The redefined SNR scale (analysis-band, post-review) sits at ~1.0–2.5
on real data versus 5–20 on synthetics. The shipped default
`min_snr = 1.5` is usable but conservative; **1.1–1.2 with the 3 km
range gate** is the better operating point for these datasets.

## Finding 3: formal error calibration on real data

With quality gating, per-tile temporal scatter is ~10× the formal
least-squares errors (Thompson gated: scatter 0.18 m/s vs formal
0.017 m/s). On synthetics the factor was 5–10×; real data lands at the
top of that range (correlated spectral leakage plus real environmental
variability). **Treat reported `ux_err`/`uy_err` as relative weights;
multiply by ~10 for absolute 1-sigma values**, or rely on the composite
`chi2` per cell, which measures exactly this. Ungated composites show
chi2 ~ 700–1400 because unfiltered garbage tiles dominate; gate before
compositing.

## Finding 4: dataset and configuration caveats

- **Thompson tower config missing**: the loader warns
  `Tower 'r/v tommy thompson' not found in config`. The Revelle entry
  carries calibrated biases (theta +10°, range −6 m, shadow-edge
  angles); the Thompson runs with zero offsets, so its **absolute
  directions may be rotated by an unknown constant** — calibrate
  against the ADCP (first bin ~8 m; note the wave-derived current is
  depth-weighted over ~1/(2k) ≈ 2–25 m for the energetic band, and
  includes a fraction of wind/Stokes drift).
- **2025 directory layout**: days 16–19 live under `2025/01/DD/HH` but
  days 20–24 are misfiled as `2025/DD/HH` (month level missing).
  `Filenames` discovery cannot see the misfiled days; moving them under
  `2025/01/` makes them discoverable.
- **Land contamination (Revelle)**: Rota is inside the radar range;
  land tiles currently rely on SNR/FOM gating to fail. A coastline or
  static-intensity land mask would be a clean future improvement, but
  the 3 km range gate already removes most affected tiles at this site.
- **Interference stripes**: both ships' spectra show horizontal bands
  (energy at all k at fixed omega ~ 0.5–0.7 rad/s), likely RF
  interference from other radars; the omega-localized bands overlap the
  dispersion shell only at specific k and appear to be handled by the
  robust fit, but worth monitoring.

## Station demonstration: 9 hours off Wedding Cake, Rota

The Revelle occupied 14.1177°N 145.1136°E (off Mt. Taipingot, "Wedding
Cake", Rota) from 2022-04-05 00:11 to 09:26 UTC. Processing the full
occupation (~23,000 files, 37 fifteen-minute composites, gated
configuration below) resolves a coherent **clockwise-rotating tidal
current** of 0.1–0.3 m/s superposed on a weak mean flow:

- mean current 0.14 m/s toward 022°,
- semidiurnal harmonic amplitude ~0.13 m/s (ux) / 0.07 m/s (uy),
- residual about the mean+M2 fit: 0.064 m/s median,
- per-composite scatter-based errors ~0.04 m/s — the point-to-point
  smoothness of the series confirms the error scale.

The rotation sense (clockwise) and period (most of a cycle in 9 h) are
consistent with semidiurnal tidal currents; the inertial period at 14°N
(~49 h) is too long to contribute. Wall time: ~50 minutes on the M4 Max
for 9.2 hours of data — roughly 11x real time at one block every 23 s.

## Recommended starting configuration for bulk reprocessing

```bash
wamos current <stime> <etime> /path/to/POLAR \
    --ship-data <dir-if-available> -o <out> \
    --min-snr 1.1 --max-tile-range 3000 \
    --composite-minutes 15
```

Composites of ~14 gated estimates per cell give ~0.05 m/s precision;
`chi2` in the composite files flags cells (and periods) where the
formal precision is not met.
