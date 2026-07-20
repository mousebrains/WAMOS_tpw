# Plan: Thompson 2023 (Palau / Hydrographer Bank) WAMOS processing

*Drafted 2026-07-16 while the 2022 April sweep runs. Same product goals
as the 2022/2025 work: currents / roughness / waves NetCDF (CF-1.13,
Bjorn-style layout), validation against in-situ truth, overlay movie.*

## Status & learnings (updated 2026-07-17)

Grounded in the code (branch `thompson2023/depth-aware`) and the T0/T1
work since drafting. Supersedes stale phrasings below; see the
Adversarial review at the end.

**Done / measured:**
- **T0 inventory — COMPLETE. Nav SOLVED via R2R TN417** (Palau
  2023-04-29 → Guam 2023-06-11): POS MV-320, 2× Sperry gyro, C-Nav,
  parsed by new `instruments/scs.py` → `gyro_sperry.nc` (heading from
  `$__HDT`), `gps_abxtwo.nc` (lat/lon; **sog/cog derived by np.gradient
  of GGA**), `wind_bridge.nc` (relative wind). The pre-made
  `ThompsonSCS/*.nc` carry **no** heading/cog/sog. Nav gap **Apr 27–28**
  (pre-TN417 → frame-metadata GYROC+GPS fallback).
- **T1 timing — MEASURED.** 2023 PPS metadata offset −1490 ms
  (std 2.5 ms); the provisional −0.44 s cross-cruise mapping was
  **WRONG**; hard returns on Palau **turn** windows converge
  **−0.9..−1.0 s** (steady legs flat — time_shift only separates during
  turns). Fine-step (0.02 s) refine + a `tn2023_wamos.yaml` tower entry
  pending. Lesson: **always calibrate time_shift directly per cruise.**
- **T1 land mask — BUILT.** `palau_landmask.nc` (majority-vote +
  dilation + DEM-shallow union via `wamos land-mask --shallow-grid`),
  georegistered vs the 25 m bathy (`palau_mask_georeg.png`).
- **T2 depth-aware dispersion — IMPLEMENTED.** `depthgrid.py` +
  `current.py::dispersion_relation` = ω=√(g·k·tanh(k·h)); `--depth-grid`
  gives each tile a **FIXED median depth**. `--composite-minutes`,
  `--field --window-sizes`, `tools/composites_to_trajectory.py` all in
  place.
- **CSSAngaur** format fully decoded; encoder-zero georeg **IN PROGRESS**
  (`css_average.py` running, crash-safe + detached, over 05/18–20).

**Assumed by the plan below but NOT yet built:**
- **Joint depth+current inversion (T2b.3)** — does not exist; depth is
  never a free parameter (→ AR5).
- **Flagged-tile uncertainty inflation (T2b.4)** — the `depth_hetero`
  flag is computed then **silently dropped**; it inflates/masks nothing
  (→ AR3).
- **CSSAngaur package handler (C1)** — `polarfile.py` is uint16-only;
  the 8-bit radial decode lives only in throwaway scripts (→ AR13).

## Dataset

- **WAMOS**: `/Volumes/SeaChest/ARCTERX/2023/WAMOS/POLAR/2023/`
  *(leg mapping VERIFIED against .pol GPS, 2026-07-16 — differs from
  first recollections):*
  - **Apr 04–09**: ship at 41°S off New Zealand transiting NW —
    pre-cruise repositioning, NOT ARCTERX. Exclude (or keep only as
    open-ocean bonus data).
  - **Apr 27**: no GPS fix in sampled frames (arrival day; needs a
    data-quality check before use).
  - **Apr 28 – May 05**: work period 1 — Koror area, then a tight
    pattern over Hydrographer Bank / Angaur (6.9–7.0°N,
    134.13–134.26°E) May 2–4, departing May 5. (The "bow-tie".)
  - **May 05–13 gap**: in port (the mechanical problem).
  - **May 13 – 23**: work period 2 — same bank area for ten days,
    mostly ~1 m/s slow-pattern/station work (the "less tight" group).
  - **The Palau→Guam Interior transit has NO WAMOS coverage** — data
    ends May 23 with the ship still at Palau.
  - Tower `R/V Tommy Thompson`, 975 range samples ≈ 3.6 km slant range,
    1.49 s rotation, uncompressed `.pol` ~4 MB/frame (similar to 2025).
- **Ship data**: `2023/Wake/ThompsonSCS/*.nc` (gps, bowMET, …) — already
  NetCDF; check compatibility with `--ship-data` (our loader expects
  revelle-style variable names; likely needs a small adapter).
  Also `PAVS150/`, `GPS/`, `timing/` dirs to inventory.
- **Bathymetry**: `2023/Wake/bathy/Angaur_Peleliu_25m.nc` (25 m!),
  `coastline.mat`, `angaur.dat`/`kayangel.dat` outlines,
  plus `/Volumes/SeaChest/ARCTERX/Bathymetry`.
  **Hydrographer Bank ~18 m deep; drops off Rota-fast near land.**

## Ground truth (richest of any leg)

| source | what it gives |
|---|---|
| `Bank Seaspider/` | Sig1000 ADCP + **2 pressure sensors (Hs truth!)** + 2 MicroRiders + CTD |
| `Bank ADCP/` | bottom-mounted RDI Workhorse, upward-looking, near the Seaspider |
| `sig1000/` | second bottom-mounted Sig1000 on the bank |
| `Pressure Sensors/` | additional bottom pressure (waves/tides) |
| `MiniWaveBuoys/` (IDs 458, 727, 742, 764, 780, …) | drifting buoys: directional waves + surface currents; **mostly drogued ~2 m but depths varied and are poorly documented** |

## Phases

**T0 — inventory (cheap, can run during 2022 sweeps):** map POLAR dates
to ship positions (which dates = bank pattern vs loose pattern vs
transits); inventory instrument formats (Seaspider `*_HBM_clipped.nc`,
sig1000 `processed/`, buoy per-ID dirs); ship-data adapter for
ThompsonSCS; PPS timing statistics (2023 recording-software version —
compare against 2022-Mar/2022-Apr/2025 conventions).

**T1 — calibration:**
- **Direction offset: CARRIES OVER FROM 2025** — Pat confirms the
  Thompson WAMOS tower was not reinstalled between 2023 and 2025
  (Bjorn's Doppler was a separate 2025-only install on the tower).
  The 2025 CSTARS cross-check measured −2.5° [−11, +5] ≈ 0, so adopt 0
  and *verify* (not fit) against MiniWaveBuoy drift + bank ADCP
  near-surface bins with the CSTARS-arbitration machinery.
- **Timing**: per-cruise recording-software conventions are now a known
  hazard (2022 March vs April differed by 620 ms) → run `wamos
  pps-timing` on 2023 vs 2025 data first; Palau/Peleliu/Angaur hard
  returns allow a `wamos hard-returns` check as well.
- **Land mask**: build from mosaics as for Rota (majority-vote,
  near-range); georegistration against the 25-m bathy (−5 m rule again).

**T2 — depth-aware dispersion (the main NEW development):**
Over the bank kh ≈ 0.5–1.5 for the energetic wave band — finite-depth
dispersion is first-order, not a refinement. Per-tile depth from the
bathy grid is IMPLEMENTED (`--depth-grid`, commit e63647d).

**T2b — depth error handling (added after Pat's caution 2026-07-17:
the DEM is degraded by reefs/steep relief here).** Sensitivity: in the
shallow limit c ≈ √(gh), so current bias ≈ (c/2)·(Δh/h) along the wave
direction — 2 m error on the 18 m bank ≈ 0.7 m/s. Mitigation ladder:
1. **Tide**: bank depth varies ±~1 m (±5 %) over the tidal cycle —
   take instantaneous depth = DEM + tidal elevation from the
   Seaspider/bottom-pressure records (they measure it directly).
   *Backup/extension (Pat 2026-07-17): the Malakal Harbor tide gauge
   (Koror; NOAA/UHSLC station) tracks the bank tides closely — use it
   as the continuous elevation reference outside the mooring
   deployment windows and as an independent phase check.*
2. **Point truth**: pressure sensors give exact depth at the mooring
   sites — first check of the DEM right where validation happens.
3. **Radar bathymetry inversion (the robust answer)**: with enough
   wavenumber spread, depth and current are JOINTLY estimable from the
   dispersion surface (depth bends the curve shape in |k|; current
   tilts it in k·U — classic X-band depth retrieval). Add per-tile
   depth as an optional free parameter over the bank; compare fitted
   vs DEM depth as a diagnostic AND a science product (updated bank
   bathymetry). Validate the joint fit against the pressure-sensor
   depths + bank-ADCP currents before trusting either.
4. Inflate flagged-tile uncertainty wherever fitted-vs-DEM depth
   disagreement is large.

**T3 — currents + validation** *(design refined with Pat 2026-07-16 —
the bank is ~1 km wide × 3–4 km long, at our resolution limit):*
- Over the bank, run the **joint field inversion**
  (`--field --window-sizes 2000,1000`): the resolution study showed
  2-km tiles at 1-km stride ALIAS 1-km features (transfer −2.3) while
  the field inversion recovers ~0.9 — this is its designed use case.
  Its known caveat (small windows on narrowband swell: ~0.2 m/s
  collective bias on synthetics) gets its first real-world test here.
- **Bottom ADCPs (Seaspider Sig1000 + Workhorse) = depth-kernel
  validation**: fixed, in-footprint, full vertical profile (~17 m to
  near-surface). Compare WAMOS against the exp(2kz)-weighted ADCP
  profile using the actually-fitted wavenumbers — the first direct
  test of the depth-weighting physics, and the decisive separation of
  the ~2% U10 downwind estimator bias from real measured shear
  (closes the wamos-downwind-bias question).
- **Drifters (MiniWaveBuoys, ~2 m drogue) = spatial-structure
  validation**: trajectories through the scene test the field
  inversion's PATTERN (wake shear, bank acceleration), which no fixed
  point can.
- Tidal analysis over the bank with bottom-pressure tides as phase
  reference. Products to `2023/WAMOS/products/` mirroring 2022.

**T4 — waves + Hs calibration:** pressure-sensor Hs (depth-attenuation
corrected) + buoy spectra = **direct in-situ MTF/Hs calibration** — far
better than ERA5; makes 2023 the calibration anchor for the waves
product everywhere.

**T5 — products + movie:** same machinery, per-leg extents.

**Interior 2023 (Palau → Guam)**: NOT covered by WAMOS (verified from
GPS — recordings end May 23 at Palau). `2023/Interior/` in-situ data
stands alone. The Apr 4–9 New Zealand transit recordings are the only
open-ocean bonus material, if ever wanted.

## Open questions for Pat

1. MiniWaveBuoys drogue depths: any partial deployment log? (Matters
   for interpreting buoy-vs-WAMOS current offsets: 2 m vs deeper
   changes the expected wind-drift difference.)
3. Whose processing to trust for Seaspider/sig1000 (`processed/`,
   `matFiles/`, `tpw/`)?
4. Ship nav preference: ThompsonSCS gps.nc vs PAVS150 (POS MV?).

## 2025 additions registered (from Pat, same message)

- 2025 Wake: PEARL drifters are **essentially undrogued** (windage-
  heavy — use cautiously as surface-current truth); Scripps mini wave
  buoys at `2025/Wake/wavebuoy` → **in-situ Hs calibration for the 2025
  waves product** (no ERA5 needed there); Marianas Bathymetry for land
  mask/georegistration.
- 2025 Interior (Guam→Taiwan, rarely near land): hundreds of drifters
  exist — Pat will dig them up. (WAMOS .pol for that leg still not
  located on SeaChest.)

---

# Plan addendum: CSSAngaur land-based radar (2026-07-16)

`/Volumes/SeaChest/ARCTERX/2023/CSSAngaur/2023/05/{07,09,18..30}` —
land tower on Angaur covering the Angaur–Peleliu channel; overlaps the
ship's second bank period (May 13–23) and the bank instrument array.

## Format (reverse-engineered from the data + Pat's specs table)

- Radar: **Furuno FAR-3220BB** chart radar, 9.41 GHz, 25 kW, HH-pol,
  2.4 m array, 0.95° beam, **24 rpm**, 0.3 µs pulse, PRF 1500 Hz,
  **15 m/range cell**. FAR-3000 internal processing (clutter controls,
  interference rejection, echo averaging, echo stretch, noise
  rejection) sits ahead of recording — affects backscatter statistics.
- Recorder: CORDC "NAVNET SAMPLE" header (OWNER CORDC); `RANGE 7.408`
  km is the 4-NM display setting, NOT the digitized window
  (884 × 15 m = 13.3 km recorded); `SFREQ/SDRNG = −1` (unrecorded).
- Payload **fully decoded 2026-07-16**: after `EOH\n` a 10-character
  ASCII payload byte count (no trailing newline!), then per radial
  **[uint16 LE azimuth word][884 × uint8 intensity]**. The azimuth
  word is a **13-bit encoder (8192 counts/rev)**, az = w/8192 × 360°,
  perfectly monotonic within every sweep. Radial count varies per
  sweep (2274–2289), so per-radial azimuths are required — no uniform
  spacing assumption. One file every ~5 s = every OTHER rotation.
- **Tower (Pat 2026-07-16): 6.91677° N, 134.14840° E; tower 100 ft on
  a 22 ft base → antenna ≈ 37.2 m above MSL.** Encoder zero vs true
  north still to be calibrated from the Peleliu/Angaur coastline;
  tower wind-sway monitored the same way (per-sweep bearing offset
  time series from land echoes).

## Key caveats

1. **5-s sampling (every other rotation)**: Nyquist period 10 s — the
   wind sea (5–9 s) is aliased; dispersion fitting must model aliasing
   or rely on swell; currents still recoverable (CORDC does this) but
   needs its own validation. Worth asking whether every-rotation data
   exists anywhere.
2. **8-bit dynamic range** + chart-radar preprocessing → Hs/MTF
   calibration is radar-specific; pressure sensors on the bank are the
   truth source and are IN VIEW of this radar.
3. Fixed station: no nav needed, but need **tower lat/lon + antenna
   height** (not in header) — ask Pat.

## Work items

- C1: format handler in polarfile.py (CORDC variant: 8-bit payload,
  radial prefix, header defaults from specs) + fixed-station mode
  (position/height/orientation from config instead of NMEA).
- C2: bearing calibration + sway monitoring from land echoes;
  georegistration vs 25-m bathy.
- C3: channel currents (tides!) validated against the bank ADCPs;
  ship-vs-shore WAMOS intercomparison May 13–23.
- C4: waves vs Seaspider pressure sensors (aliasing-aware).

## Format questions — ALL RESOLVED (2026-07-16)

Recorder source at `CSSAngaur/navnet-sample-master` (Furuno NavNet
sample SDK + CORDC recorder) closes everything:

- `angle` = "relative antenna angle (0 to 8191)" (radar.h) — the
  13-bit encoder word we decoded, RELATIVE TO THE HEADING-LINE MARK,
  i.e. the tower mounting orientation (calibrate vs coastline).
- **`SCALE` = "echo data position corresponding to present range"** —
  the sample index at the selected radar range. So per-file
  **cell size = RANGE_m / SCALE**: 7408/496 = 14.94 m (matches the
  15.00 m spec); May 07 with RANGE=12 km → 24.2 m cells, 21.4 km
  window — May 07 IS usable with its own geometry, no extra
  calibration needed.
- `FIFO` = sweep_len = 884 samples actually delivered per spoke.
- Every-other-sweep recording was a tower-PC performance limit
  (pre-2025) — no full-rate 2023 data exists; 10-s Nyquist stands.

Tower: 6.91677° N, 134.14840° E, antenna ≈ 37.2 m above MSL (Pat).
Still open (nice-to-have): the paper the specs table came from.

---

# Adversarial review (2026-07-17)

Stress-testing the plan against what the code actually does (audited on
branch `thompson2023/depth-aware`) and what T0/T1 taught us. Ordered by
impact.

## Structural (highest impact)

**AR1 — "Self-consistent" ≠ correct: the DEM is a common-mode single
point of failure.** The same `Angaur_Peleliu_25m.nc` does three jobs at
once — per-tile **depth**, **land-mask** source (DEM-shallow union), and
**georeg reference** for BOTH the Angaur encoder-zero fit and the ship
land mask. A horizontal DEM mis-registration (plausible for a reef DEM)
is quietly absorbed by all three: theta0, the ship mask, and the tile
depths all become "consistent with the DEM" and jointly wrong in earth
coordinates. Angaur↔Thompson↔DEM agreement is necessary but NOT
sufficient. **Action:** anchor to something external before trusting
theta0 or tile depths — cross-check the DEM coastline against the
independent `coastline.mat` (different provenance/date; confirmed
present) and against GPS-surveyed mooring positions vs their DEM depth
(`Bank Seaspider`/`Bank ADCP`/`Pressure Sensors` sit on the bank).

**AR2 — Silent deep-water fallback punches holes exactly over the
reef.** *(FIXED 2026-07-17, commit 6423f53: depth_missing flag + warning
+ error inflation.)* Tiles with no bathy coverage get `depth = inf` and revert
silently to deep-water dispersion (`current.py`). On a degraded/patchy
reef DEM, a data hole on the shallowest ground turns the finite-depth
correction OFF where it matters most — unflagged. **Action:** treat
missing depth as interpolate-or-reject-and-flag, never a silent
deep-water tile; log missing-depth fraction per scene.

**AR3 — The uncertainty-inflation safety net (T2b.4) is dead code.**
*(FIXED 2026-07-17, commit 6423f53: depth_hetero/depth_missing now reach
the extractor; flagged tiles get ux_err/uy_err x2. Justified by the
singlebeam result: 25-60 m flank truth is unobtainable — chirp saturates.)*
`depth_hetero` is computed in `tile_depth` and stored in the tile dict,
but never serialized into `task_data`, so it inflates nothing and masks
nothing. The plan leans on it to down-weight steep/uncertain tiles; that
mechanism does not exist. **Action:** wire `depth_hetero` (and a
missing-depth flag) through into an uncertainty multiplier, or strike
the claim.

**AR4 — Error budget vs signal: the headline product (bank current
field) may sit below the noise floor; the robust wins are undersold.**
Stack the terms over the bank: field-inversion narrowband bias
~0.2 m/s (small Hann windows on swell, documented ~0.9 recovery) + a 2 m
DEM depth error ≈ **0.7 m/s** in the swell band (AR6) + the carried-over
direction CI of ±11° (an 11°-rotated current injects a large cross-track
component). Likely bank currents (tidal / bank-accelerated) are
~0.1–0.4 m/s → the error budget can **exceed** the signal. Meanwhile the
two bulletproof 2023 wins — **in-situ Hs/MTF calibration** from the
bottom pressure sensors, and the **ADCP depth-kernel validation** that
settles the downwind-bias question — depend on none of the above.
**Action:** build the quantitative error budget first; promote T4
(waves/Hs) and the ADCP-profile test to primary deliverables; treat the
bank current field as exploratory until the budget closes.

## Method & calibration

**AR5 — The "robust answer" (joint depth+current inversion) is unbuilt
AND least reliable where it's needed.** It needs wavenumber spread to
separate depth (curve shape in |k|) from current (tilt in k·U). Over a
~1 km bank the wind sea is short-range/low-SNR (ship usable to ~3 km)
and, for CSSAngaur, aliased (10 s Nyquist); a swell-dominated narrowband
spectrum makes depth↔current ill-conditioned exactly there. **Action:**
demote from "robust answer" to an experimental cross-check /
bathymetry-science bonus; make **measured tidal depth from bottom
pressure (T2b.1) + point-truth DEM correction (T2b.2)** the primary
depth defense — both direct, no spectral conditioning needed.

**AR6 — The 0.7 m/s depth sensitivity is swell-specific, not uniform
over "kh 0.5–1.5."** For T≈14 s swell over 18 m, kh≈0.37 (near the
shallow limit) → a 2 m error gives ~0.66 m/s (∂U/∂h worked through the
finite-depth relation); for 6 s wind sea, kh≈2 → nearly
depth-insensitive. The current fit's leverage comes from the long-swell
end, so depth error bites the current estimate hard. **Action:**
stratify depth-sensitivity and validation by wave band. The same kh
contrast is the only thing that could make AR5's joint fit conditionable
— it needs both bands with SNR.

**AR7 — Encoder-zero should anchor on hard land, not the breaker band.**
`css_georeg.py` fits theta0 by maximizing bright fraction on DEM <5 m
cells, but depth-limited breaking sits on the seaward reef face
(depth ≈ 1.3·Hs, the ~2–3 m contour) and migrates with tide and Hs → an
along-shore bias in theta0. **Action:** fit theta0 primarily to
tide-invariant geometric returns — the **above-water Angaur/Peleliu
shoreline** (hard land, already in the average) and the **harbor-gap
notch** — using the diffuse breaker band only as corroboration.

**AR8 — time_shift and direction offset are degenerate off-turns.** The
−0.9..−1.0 s shift was fit on turn windows because a constant-heading
leg makes a timing error indistinguishable from a bearing offset.
**Action:** confirm the turn-window fits cleanly separated the two (a
direction error must not hide inside time_shift); note bank
station-keeping is quasi-steady → do not re-fit either there.

## Data & scope reality checks

**AR9 — Gate on geometry first: is the bank inside the usable ~3 km wave
annulus during each science window?** Ship WAMOS waves are usable only
to ~3 km. If the ship stood off the bank/moorings beyond that in the
bow-tie (May 2–4) or station work (May 13–23), there is no usable bank
wave signal there. **Action:** compute tower→bank and tower→each mooring
range per window before any processing — cheaply gates all of T3/T4.

**AR10 — Drifter validation is magnitude-uninterpretable, so the field
inversion's spatial recovery gets only qualitative truth.** Undocumented
MiniWaveBuoy drogue depths (open-Q #1) mean buoy currents mix windage +
~2 m drift unknowably, and the fixed ADCPs can't test spatial pattern.
**Action:** frame drifter-vs-WAMOS as structure/topology agreement
(shear sign, convergence), not a quantitative transfer; first check
whether enough drifters cross the ~1 km footprint in-window to trace
structure at all.

**AR11 — SCS cog/sog are np.gradient of GGA → noisy at station-keeping
speed,** which is the bank regime; bad COG corrupts ship-motion
compensation. **Action:** drive heading/motion from POS MV +
`gyro_sperry.nc`; treat GGA-derived COG as unreliable below ~1 m/s.

**AR12 — `composites_to_trajectory.py` hard-codes Revelle/ARCTERX
platform attrs;** `--prefix` changes only the filename. 2023 Thompson
trajectory products would be mislabeled as Revelle. **Action:**
parameterize platform/project before generating 2023 products.

**AR13 — CSSAngaur C1 is not built.** `polarfile.py` is uint16-only; the
8-bit radial decode exists only in throwaway scripts — enough for the
reflection/geometry study, not for running CSSAngaur through the
current/wave pipeline (C3/C4). **Action:** decide explicitly — build the
fixed-station handler in `polarfile.py`, or descope CSSAngaur to the
standalone geometry/reflection + tide cross-check the scripts enable.

## Open questions this review surfaces

- DEM absolute georeference: does `coastline.mat` agree with the DEM
  <5 m contour to within a tile? (AR1)
- Expected bank-current magnitude — any prior (charts, TN417 shipboard
  ADCP over the bank) to compare against the ~0.2–0.7 m/s error budget?
  (AR4)
- Drifter density over the bank during May 13–23. (AR10)
