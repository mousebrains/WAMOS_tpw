# WG + Thompson ADCP assessment — 2026-07-21 (session 1)

## Inventory
- 4 gliders: emily(sv3-125*), ole(sv3-253), ragnar(sv3-1101), sven(sv3-251)
  `wavegliders/<name>/HB_sv3-*_adcp_data.nc`: 10-min ensembles, 50 bins,
  z = 4.25..102.25 m (dz 2 m), Evel/Nvel (absolute, ENU) + lat/lon/sog/cog.
  (*raw dir names sv3-125 "stallion" = emily's vehicle — confirm mapping.)
- sven also has `adcp_sven_arcterx_final.nc`: 2-min, Evel/Nvel_std (~0.30
  m/s per sample), N_adcp, ADCP-vs-GPS platform velocities.
- Raw telemetry (METOC/SMC_BBB) present for ragnar/sven/stallion; NONE for ole.
- Co-located: metbuoy_*.dat (MET), mwb8*.nc (directional wave buoys ON the
  gliders — Pat's "directional wave packages"), CTD for ole/sven.
- Thompson: CODAS wh300 (bins from 9.1 m) + os75.

## Quality findings
- HB vs sven-final: dmed <= 5 mm/s, MAD ~3 cm/s, r 0.94-0.97 all depths ->
  same processing lineage; motion compensation already applied and the
  10-min ensembles average wave-band motion down. NOT independent accuracy.
- Platform noise floor from emily<->sven sub-km encounters: ~2-3 cm/s per
  10-min 9-25 m band mean. Excellent.
- Cross-cal <2 km all pairs pooled: |dU|med ~ 0.10 m/s — dominated by REAL
  bank gradients at 1-2 km separations, not instrument error.
- ragnar "bias" REFUTED at <1 km: event medians sign-flip. No constant
  heading bias resolvable yet (upper bound unclear, ~0.1 m/s).
- ole: 41% missing; event-median swings -0.5..+0.24 m/s vs sven -> suspect;
  no raw telemetry available to reprocess. Down-weight until understood.
- emily final sample(s) = recovery contamination (dV ~ 2.5 m/s at ship).
  TRIM deployment/recovery edges on all platforms.

## Next steps
1. Structure function |dV|^2 vs separation from ALL coincidences ->
   extrapolate to 0 km -> per-pair offsets + noise, gradient-corrected.
2. Network least-squares platform offsets (TGT reference) using
   gradient-aware weighting.
3. ole deep-dive: time-resolved QC, correlate bad events with sea state
   (metbuoy/mwb) and heading; decide keep/flag/reject.
4. Edge-trim rule (first/last ~1 h of each deployment leg).
5. THEN: X-band UV validation vs vetted ADCP ensemble (ship exp(2kz)
   kernel per plan; shore tiles at WG positions).

## Bank ADCPs (session 1, cont.)
- C05 Workhorse (Bank ADCP/C05_2023_proc.mat): 2022-12-03 -> 2023-06-02,
  dt 12 min (not 20), inst depth 18.2 m, 43 up-looking bins dz 0.5 m.
  error_vel available.
- Sig1000 (sig1000/processed/sig1000_DT_2min.mat, v7.3): 2023-05-07 ->
  2023-06-02, 2-min means of 16-Hz bursts, P~18.8 dbar, 43 bins.
  config Declination = 0.0 -> magnetic reference (Palau ~+0.6 deg, minor).
- **HEADING SKEW (Pat knew): WH = 0.985 * R(+21.09 deg) * Sig** (complex
  transfer fit, n=2854, 4-14 m band, speed>0.15 weighted, outlier-cut).
  After rotation: dE,dN medians ~0.000, MAD 0.023-0.028 m/s -> the two
  moorings become a single 2-3 cm/s truth node. Iron anchor under C05 =
  suspected hard-iron cause -> Sig assumed true, WH corrected by -21.09;
  ARBITRATE absolute orientation vs Thompson wh300 bank passes (todo).
  Also todo: skew stability over record (first split botched a datenum).

## Session 1 steps 1-3 results (network.py)
1. SKEW STABLE: WH=c*SIG angle +20.2..+21.7 deg (gain .96-.98) in every
   5-day chunk May 07-Jun 02 -> single constant correction valid:
   **WH_corr = R(-21.09 deg) * WH** (rotation only; gain within noise).
2. ARBITRATION (externals <1.5 km of HBM, 9-15 m band):
   vs SIG: ole -3.2, sven +8.1, TGT +5.4 deg (n=53/138/103)
   vs WH:  ole -25.2, sven -13.3, TGT -17.1 deg
   -> **SIG holds true north to within ~5 deg; WH is the rotated one**
   (confirms Pat). TGT-implied SIG absolute offset ~ -5 deg CW but
   gradient-contaminated at 1.5 km; treat SIG frame as truth pending a
   dedicated loiter refit. (SIG declination unapplied: +0.6 deg, minor.)
3. STRUCTURE FUNCTION (all 7 platforms, WH corrected, 9-15 m):
   r<0.25 km: 0.033 m/s per component (instrument floor, WH-SIG)
   0.25-0.5: 0.080 | 0.5-1: 0.138 | 1-2: 0.186 | 2-3: 0.204
   3-5: 0.241 | 5-8: 0.271 | 8-12: 0.331
   -> bank flow decorrelates HARD: ~0.14 m/s real difference at 1 km.
   CONSEQUENCES: (a) platform offsets must be fit with gradient-aware
   weights (r<0.5 km only, or model D(r)); (b) X-band validation needs
   footprint matching <~0.5 km or the comparison measures the OCEAN.
   ole again inconsistent (r<1 km: +0.14 E vs moorings, -0.14 E vs
   sven) -> sign-flipping offsets = data quality, not bias; quarantine.

## Pointing-offset network solution (pointing_table.py)
Reference TGT wh300 = 0 (Applanix/CODAS). Offsets deg CCW of true;
correct by rotating velocities -offset. Bootstrap SEs x sqrt(3).
  TGT +0.00 (ref) | sven +0.31 +-1.9 | emily -1.44 +-2.5
  SIG -6.57 +-1.7 | WH +14.39 +-1.7 (skew 21 = 14.4+6.6 decomposed)
  ragnar -18.11 +-3.6 (two independent links agree)
  ole -7.89 +-2.6 BUT links internally inconsistent -> quarantined
Median pairwise residual 2.6 deg (16 links).
**RETRACTION: earlier "ragnar bias refuted" was WRONG — sign-flipping
event dV is the signature OF a rotation under reversing tidal flow;
complex-transfer is the right estimator. ragnar = -18 deg compass
rotation, correct by +18.1.**
Corrections adopted for the truth ensemble: SIG +6.6, WH -14.4,
ragnar +18.1, sven/emily/TGT none, ole excluded pending deep-dive.
> **SUPERSEDED 2026-10-05 (Pat): trust the Thompson cross-check for the Sig
> heading.** Hydrographer-Analysis `notes/10` section 2 compares the wh300 with the
> Sig1000 within 0.3 km of the frame (160 pairs, 9-17 m, 6-min means): direction
> ADCP - Sig +1.6 deg (MAD 2.1), the same sign as the network's 6.6 deg but
> smaller. The network value predates that check and is gradient-contaminated at
> 1.5 km (step 2 above).
> - Adopted: SIG +1.6, so WH -(21.09 - 1.6) = -19.5 (absolute
>   W exp(-i 19.5 deg); C05 axis 118.5/298.5 deg).
> - truth_ensemble.npz and the comparisons below were built with SIG +6.6 / WH
>   -14.4 and have NOT been rerun.
> - Consequence to check: the shore radar's bank-node angle (+0.3 deg against the
>   +6.6-rotated Sig, snr 2-3) becomes ~+5 deg against a +1.6-rotated Sig. Whether
>   that is radar pointing or residual Sig heading is open.

## ole deep-dive verdict (session 1)
- Daily health looks normal: vstd 0.032 = sven's 0.033; missingness
  uniform ~40% every day (duty cycle, not episodic).
- BUT per-event transfer vs references: |gain| ~1.0 at flow <=0.5 m/s,
  0.36-0.47 at 1.1-1.3 m/s (strong bank jets) -> SPEED-DEPENDENT
  UNDERREAD (~50-60% at 1+ m/s); May 21+ additional degradation
  (gain 0.5-0.8, angle wandering to -27 deg). Candidate mechanism:
  platform-velocity removal failing under strong drift.
- VERDICT: EXCLUDE ole from the truth ensemble entirely (nonlinear in
  the regime that matters). Its rotation "offset" is meaningless.
- Its MET + wave buoy (mwb824d02: Hs/Tp/Dp) remain usable.

## Truth ensemble (final, session 1)
  TGT wh300 (ref), sven (+0), emily (-0; both ~2-3 cm/s),
  ragnar rotated +18.1 deg, SIG rotated +6.6, WH rotated -14.4
  (bank pair merges to one 2-3 cm/s node). Edge-trim WG legs.
  [2026-10-05: SIG +6.6 / WH -14.4 superseded by SIG +1.6 / WH -19.5 (Thompson
  cross-check, Pat); the ensemble file has not been rebuilt.]
Next: build merged vetted ensemble file (common format, corrections
applied) -> design footprint-matched X-band comparison (<0.5 km).

## First X-band vs bank-node comparison (xband_bank_compare.py + diag)
Truth = ensemble SIG (rotated) 2-8 m band over the radar window.
SHORE (547 tile-matched 30-min windows at HBM):
- No time lag (flat lag scan, peak at 0).
- SNR is the QC axis: snr 1.5-2 (n=260): |c|=0.59 ang +27 (noise-
  dominated); snr 2-3 (n=139): |c|=0.887 ang +0.3; snr>=3 (n=20):
  |c|=0.954 ang -9.5.
  -> WITH snr>=2 the Angaur radar tracks the rotated Sig truth with
  ~zero angular bias and gain 0.89-0.95 (some dilution from window
  scatter). VALIDATES radar geometry AND the Sig +6.6 rotation.
  [2026-10-05: this is the one piece of evidence FOR the 6.6 deg Sig offset. Pat
  adopted the Thompson cross-check (+1.6 deg) instead; against a +1.6-rotated Sig
  this angle becomes ~+5 deg. Radar pointing vs Sig heading not yet separated;
  see the SUPERSEDED note at the truth-ensemble corrections.]
  [2026-10-05, later: neither. Pat: radar pointing is calibrated independently
  on reef breaks and hard targets. Hydrographer-Analysis code/sig_heading_sweep.py
  (notes/12 section 8) shows the wh300-vs-Sig angle grows with distance and, beyond
  0.3 km, REVERSES SIGN with flow direction: -9 to -26 deg in ESE flow, +5 to +6
  in WNW flow. That is flow steering by the bank, not a compass error. Within 0.3 km
  it is +1.3 to +1.6 deg. The network's 6.6 and this bank-node angle are footprint
  effects in WNW-dominated samples (May 13-23 was a westward regime).
  TEST: split this comparison by flow direction; the angle should flip sign.]
  Prescription: shore current QC = snr>=2 for validation/products.
SHIP (274 composite windows w/ cells <0.5 km of HBM):
- Naive all-cell comparison BAD (dU +0.27, |c|=0.25) — NOT a radar
  verdict: no ux_err/n_obs filtering and no ship-distance cut yet
  (May-19 golden-window result was +0.02 with ship <=1.4 km). TODO:
  redo with ux_err filter + ship-HBM range stratification (the known
  range-dependent-scatter question, now with a fixed truth node).

## Round 2 (xband_compare2.py)
A. SHIP vs bank node, err<0.15 & n_obs>=3, by ship-HBM range:
   0-1.5 km: n=89 dU +0.144 dV +0.083 MAD 0.26 |c|=0.46 ang -9
   1.5-2.5:  n=80 dU +0.370 MAD 0.57 |c|=0.24
   2.5-4.0:  n=96 dU +0.384 MAD 0.70 |c|=0.12
   -> RANGE COLLAPSE quantified: beyond ~1.5 km the composite cells at
   HBM are noise-dominated (formal errors underestimate). Even <1.5 km
   dU +0.14 vs the May-19 golden +0.02 — window population differs
   (all of May 13-23, all winds); reconcile vs golden subset + wind
   stratification next.
B. SHORE tiles vs WG truth along tracks (snr>=2, <0.5 km):
   MAD 0.78-1.19 m/s everywhere, transfer angle drifts -25..-35 deg
   at 6-13 km range. vs the BANK-NODE result (gain 0.89, ang 0, much
   smaller scatter) -> shore current validity is NOT footprint-wide;
   snr>=2 alone does NOT flag bad regions. Next: 2D error map (bin
   pairs by tile x,y), azimuth/wind stratification; suspect aliasing
   x look-direction and weak wave contrast off-bank.
H0 discipline: neither branch is yet a radar verdict; both expose
comparison-population effects to resolve first.

## QC mask verdict (shore_qc_mask.npz built, NOT wired to renderer)
>=80%-shallow gate leaves EXACTLY 1 valid tile of 1024 (the HBM tile,
0.82) — bank top too narrow for 960-m footprints on a 480-m grid;
corrupted neighbors measured at 0.68-0.76 frac confirm the threshold
cannot be relaxed. Channel 40-60 m tiles = effectively deep (alias
regime). CONCLUSION: Angaur current FIELD not defensible as extracted;
validated product = bank-node time SERIES. Field paths (ranked):
(1) bank-aligned elongated windows (~1.5x0.5 km rotated along crest)
-> along-bank transect; (2) h(x,y)-aware dispersion fit in-window;
(3) alias-aware two-branch fit (unlocks deep). Backscatter products
unaffected. Ship tiles over the bank deserve the same uniformity
audit.

## Alias batch full results (alias_batch.csv, 697 pairs, May 18-30)
Deep water by tower range (m=0 -> m=1):
  1.5-5 km (n=72):  MAD .62 ang+55  ->  MAD .18 ang -7.6 |c|=.80  UNLOCKED
  5-6.5   (n=204):  MAD .45 ang+58  ->  MAD .91 ang -7.0        direction fixed, amp noisy
  6.5-8   (n=162):  MAD .61 ang+67  ->  MAD .82 ang -2.3        same
  8-12.5  (n=239):  rotated garbage ->  broken (|c|~.2)          needs aniso weighting
KEY: legacy deep estimates were SYSTEMATICALLY rotated ~+60 deg
(the uv-reel chaos anatomy). Alias mode fixes direction <8 km.
Early "5-8 km MAD 0.137" was the May 18-19 subset — benefit is
CONDITION-DEPENDENT (wind sea vs azimuthal smearing); stratify the
CSV vs palau_obs winds next. QC metrics (uxe/nls/snr) do NOT
discriminate in alias mode — QC must be physics-based (predicted
azimuthal MTF at range). DEM cached locally (worker-spawn SMB
collisions fixed); batch checkpoint/resume worked as designed.
NEXT DEV: anisotropic k-weighting (range x beamwidth); wind
stratification of 5-8 km scatter; then bank-node + ship-side alias
reruns.

## Wind stratification -> 2025-ready QC recipe (session 2)
- Station wave fields (waveHs/Tp/Dp) FROZEN at hourly cadence during
  May 2023 (0.9 m / 11.3 s constant) -> unusable for conditioning;
  ask Sean about their update cadence. U10 is the dynamic knob.
- U10 gate monotone: err 1.23 (U10<3) -> 0.39 (U10>=7, station scale).
- May 18-19 "golden subset" dissolved: it was SVEN; residual structure
  is SECTOR-dependent (static): look 120-300 deg (island shadow + W
  obstruction arc) err 1.0-1.4; clean arc 300->120 through N err
  0.15-0.62 wind-gated. emily(295)-vs-sven(300) anomaly = 5 deg of
  geometry across the arc boundary.
- QC RECIPE (transfers to 2025 with Angaur MET only): (1) r<=8 km,
  (2) look bearing in ~300..120 arc, (3) station U10>=5 for marginal
  sectors. Yields med err 0.15-0.43 over ~180 deg arc to 8 km
  (includes truth noise + real <0.5 km ocean variability).
- 2025: 2.5-s sampling halves folding (alias_m still useful for
  T<5 s); grazing limit (8 km) and obstruction arc persist.

## Ship-side threads (session 2)
- Ship aliasing: dt=1.49 s -> Nyquist T~3 s; wind sea fully resolved.
  No ship alias rerun needed (alias_m matters only for Angaur-class
  undersampling).
- THREE-TIER SHEAR LADDER (footprint-matched, good geometry):
  (c) drifter(0-2m)-ADCP(4-10m) = +0.34 m/s ~ 2.5% U10 (n=11) REAL
  (a) radar-ADCP = +0.12 ~ 0.9% U10 (n=19)
  (b) radar-drifter = -0.46 (n=28)
  drifter > radar > ADCP -> radar senses intermediate effective depth
  in a genuinely sheared layer: the "downwind bias" reframes as
  depth-of-measurement physics (provisional: small n, closure
  sign-consistent factor ~2, drogue depths undocumented).
  Next: more triplets, drogue bookkeeping, exp(2kz) kernel profile.

## Incoming data (Pat, 2026-07-21)
Pat is retrieving the REMAINING wave-glider data from other time
periods — expands the validation space, especially for 2025 (which
currently has no in-situ truth beyond Angaur MET). When it lands:
run each new set through the same ladder — pointing-network fit
(rotation/gain vs references), speed-dependent-gain check (the ole
failure), edge trims — before admitting to the ensemble; note the
2025 Wake `_rt_` products are real-time motion-corrected and may
need the precision reprocessing pass Pat planned. All machinery
reusable: build_ensemble.py, pointing_table.py, structure function,
checkpointed batch pattern.

## Bank-node alias rerun (bank_alias_batch.csv, 548 windows)
alias m=1 + snr top-quartile: gain 0.890 ang -1.4 MAD 0.49 (n=137) —
reproduces the validated tile result; + U10>=5: MAD 0.392, offsets
<=0.07 (n=111). Alias snr still RANKS quality despite dilution
(quartile gate works). VERDICT: single estimator config (alias_m=1,
MTF weighting, snr-quartile + wind gate) serves bank AND deep domains.

## Decorrelation floor -> 2025 validation footprint (2026-08-01, Pat + pressure-array work)
Derived from the session-1 step-3 structure function (7 platforms, 9-15 m)
against sigma_field = 0.42 m/s/comp (from C05 depth-avg: east .555, north .213).

  sep km   D(r)   rho   var expl   |   sep km   D(r)   rho   var expl
   1.5    0.186  0.90     81%      |    6.5    0.271  0.79     63%
   4.0    0.241  0.84     70%      |   10.0    0.331  0.69     48%

**A perfect radar cannot agree with a mooring better than the ocean
decorrelates.** vs an assumed 0.15 m/s radar error the crossover is at
**~0.9 km**: inside that, radar error dominates; outside, decorrelation does.
0.3 m/s scatter at 6 km is the EXPECTED result for a flawless radar.

GEOMETRY (Angaur tower 6.91677/134.14840): C05 5.83 km, HBM 5.78 km (C05-HBM
0.05 km), An2 0.59 km, Peleliu Pe2 10.26 km from tower / 5.23 from C05.
-> the tiles nearest the tower (best radar performance) are ~6 km from C05 and
CANNOT be tightly validated against it. C05-validatable tiles sit at mid-range.

CONSEQUENCES FOR 2025 (C05 is the only current mooring):
1. Validate within ~1 km of C05; beyond that the scatter plot measures the
   ocean, not the instrument.
2. Quote the decorrelation floor beside every comparison, matched to
   tile-mooring separation. Omitting it understates skill systematically
   with range.
3. **Characterise radar skill vs range on the 2023 7-platform ensemble NOW**,
   then carry that characterisation into 2025. This is a 2023 task whose
   entire value is in 2025 — easy to leave until too late.
4. Single-mooring reach is good for REGIME/SIGN out to ~10 km, marginal for
   magnitude. Adequate for the eddy shedding-direction question (below).

RELATED, from the ARCTERX pressure array (see
~/Downloads/netcdf/PRESSURE_ANALYSIS.md):
- Pressure-gradient currents DO NOT WORK here: predicted M2..O1 amplitudes
  2-7x measured with mostly wrong phases (HB gradient SNR 1.0). Failure is
  BIAS not noise -> cannot be averaged away. Root cause: only 17%/10% of the
  measured E/N current is phase-locked tide; bottom pressure sees only the
  barotropic mode.
- What the 12-gauge array DOES resolve: co-tidal chart. M2 phase gradient
  0.761 +/- 0.102 deg/km, propagating 314 deg at 11 m/s (7.5 sigma); S2 311
  deg / 9 m/s; O1 328 deg / 6 m/s. Amplitude gradient NOT resolved
  (1.4 +/- 1.8 mm/km). **11 m/s implies h_eff ~11 m** (bank tops, sqrt(g*19)
  =14 m/s) not the 1500 m channel (121 m/s) -> the gauges sense a
  shallow-water-controlled wave over the reef complex, which is probably why
  the deep-barotropic momentum balance fails.
  > **CORRECTED 2026-10-05 — the co-tidal numbers above are WITHDRAWN.** The
  > ARCTERX-Peleliu-tip repo retracted them on 2026-08-03 (`PRESSURE_ANALYSIS.md`
  > section 8.1). With all twelve gauges the M2 phase gradient is 0.230 +- 0.071
  > deg/km, propagating toward 297 deg at 35 m/s (3.2 sigma). S2 and K1 are not
  > resolved; O1 is 1.5 sigma. The 0.761 deg/km came from a short baseline:
  > Angaur-only fits give 0.758 deg/km and scale as 1/baseline. 35 m/s implies
  > h_eff ~125 m, between the bank tops (14 m/s) and the channel (121 m/s), so the
  > "shallow-water-controlled wave over the reef complex" reading above does not
  > follow. The forward-model route (next bullet) stands.
- Legitimate sea-level -> current route is a FORWARD barotropic model
  constrained by the co-tidal chart + bathymetry, not an inversion.
- C05 rotation independently reproduced: gain 0.983, angle +21.00 deg vs
  Sig1000 (n=2958) against the 0.985/+21.09 here. Absolute correction
  W*exp(-i*14.4deg) applied. E/W event labels are insensitive to it (0.0% of
  ensembles change sign) — the rotation matters for the axis bearing, not the
  classification.
- C05 shedding-direction lead: subtidal flow sets the SIGN (73% agreement),
  tide supplies 61% of along-axis variance. May 2023 regimes W 14-23,
  E 24-27, W 28-30, E 31. Drifter eddy sits in a 7 h westward event
  05-22 04:12-11:12 peak -115 cm/s. 141 events >1 h in
  c05_high_flow_events.csv (81 E, 60 W over 181 d). Testable against radar.
- C05 gotchas: 12-min ensembles (not 2); adcpr.pressure is deca-pascals with
  4294967286 fill; depth has 0.000 for 181 out-of-water ensembles (mask >5 m).

