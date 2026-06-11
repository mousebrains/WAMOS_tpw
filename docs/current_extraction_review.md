# Surface Current Extraction — Scientific Review (June 2026)

A review of the 3D-FFT dispersion-fitting current extraction
(`current.py`, `current_pipeline.py`, `current_tasks.py`) prompted by
unsatisfactory current vectors from the ARCTERX X-band data. This
document records what was found, what was changed, the validation
evidence, and the issues that remain open.

## Summary of the root cause

The dominant problem was **estimator precision, not a sign or geometry
bug**. The frame conventions (FFT sign, Doppler branches, y-axis
orientation, TO-direction convention) all check out, and the synthetic
tests confirmed them. But the original estimator — coarse shell-energy
maximization over a 0.1 m/s grid followed by Nelder-Mead polish
restricted to ±0.15 m/s of the coarse maximum — is fundamentally limited
by the omega bin width:

    delta_U (one omega bin) = d_omega / k

With 32–64 frames at a ~1.5–2.5 s rotation period, `d_omega` is
0.065–0.13 rad/s, and the energetic wavenumbers are k = 0.01–0.13 rad/m,
so **one frequency bin corresponds to 0.5–10 m/s of current**. On
synthetic data with known currents the original code produced errors of:

| Spectrum type                    | median error | max error |
|----------------------------------|--------------|-----------|
| 12 discrete waves (sparse)       | 0.19 m/s     | 0.49 m/s  |
| 300-wave directional continuum   | 0.79 m/s     | 3.16 m/s  |

The continuum case — the realistic one — was catastrophic: when barely
resolved long swell dominates the variance (λ comparable to the tile
size), the shell-energy surface is nearly flat along the swell direction
and its maximum lands far from the true current.

## What was changed (this branch)

### 1. Least-squares dispersion fit replaces Nelder-Mead refinement

`CurrentExtractor._refine()` now implements the literature-standard
approach (Young et al. 1985; Senet et al. 2001): locate spectral peaks
near the predicted dispersion shell, interpolate their coordinates to
sub-bin accuracy, and solve a weighted linear least-squares problem

    omega_0(k_obs) - omega_peak = kx*Ux + ky*Uy

Key details, each of which was found to be necessary in testing:

- **Sub-bin interpolation in omega AND in (kx, ky)** (parabolic).
  Without sub-bin k, a wave whose true k falls between grid columns is
  assigned the bin-center k and its intrinsic frequency is wrong by up
  to `c_g * d_k / 2` — at k = 0.015 rad/m that alone is ~0.6 m/s of
  velocity error.
- **Inverse-variance weights** `w = P_peak / sigma_omega^2` with
  `sigma_omega^2 = (0.3 d_omega)^2 + (0.3 c_g d_k)^2`. The group-velocity
  term downweights low-k columns, where frequency errors map into huge
  velocity errors. Pure power weighting (the naive choice) concentrates
  weight exactly where the velocity information is worst, because radar
  image spectra are red.
- **Velocity-space sigma-clipping**: residuals are converted to velocity
  units (r / |k|) before the 2.5-sigma outlier test. Frequency-space
  clipping leaves in low-k leakage points whose frequency residuals look
  innocuous but are ruinous in velocity.
- **k-scaled peak-search windows** (± min(3, 1.5 m/s × k / d_omega)
  bins) so low-k columns cannot lock onto spectral leakage one or two
  bins away.
- Three outer iterations re-center the search windows; this gives the
  fit a large basin of attraction (it recovered from coarse errors of
  3 m/s in testing).
- The fit produces **formal 1-sigma uncertainties** (`ux_err`,
  `uy_err`), the number of peaks used, and the rms frequency residual,
  all propagated into `CurrentEstimate`, `CurrentMap`, and the NetCDF
  output. Caveat: the formal errors assume independent peaks; with
  correlated leakage the true error is several times larger (observed
  ~5–10x on synthetics). Treat them as relative quality, not absolute.

Validation after the change (same synthetic suites):

| Spectrum type                    | median error | max error |
|----------------------------------|--------------|-----------|
| 12 discrete waves (sparse)       | 0.085 m/s    | 0.17 m/s  |
| 300-wave directional continuum   | 0.10 m/s     | 0.14 m/s  |
| continuum + drifting static bkg  | ~0.11 m/s    | 0.14 m/s  |

### 2. Coarse grid search objective normalized per shell bin

The raw shell-energy sum was biased: the number of shell bins inside the
temporal Nyquist varies with the candidate (Ux, Uy), so raw sums favor
candidates that merely sample more bins. The search surface is now the
*mean* power per shell bin, with candidates rejected when they retain
fewer than `current.min_shell_fraction` (default 0.25) of the possible
bins. The coarse search only needs to initialize the LS fit, so residual
coarse bias is tolerable.

### 3. Spectral band limits

- **`current.k_max`** (default `pi / (2 dx)`, i.e. wavelengths ≥ 4 grid
  cells): bins near the spatial Nyquist are dominated by speckle,
  gridding artifacts, and temporally aliased short waves; they
  previously entered the fit whenever a large Doppler shift brought
  their predicted frequency back under Nyquist.
- **`current.omega_min_factor`** (default 2, i.e. |omega| ≥ 2 d_omega):
  the residual static pattern and its window leakage concentrate
  enormous power near omega = 0; shells were attracted to that ridge.
- The fit objective additionally subtracts each (kx, ky) column's median
  power over omega (a per-column noise floor), so the broadband pedestal
  cannot pull the fit.

### 4. SNR redefined over the analysis set

The SNR denominator previously included the entire spectrum — the DC
plane, the static-pattern ridge, and k bins outside the annulus — so its
value was deflated by an arbitrary, scene-dependent factor and the
`min_snr` threshold was not meaningful. Noise is now estimated only from
bins the shell could actually occupy (valid k annulus,
omega_min ≤ |omega| < Nyquist). Pure noise still gives SNR ≈ 1 (tested),
but **absolute SNR values are larger than before — `current.min_snr`
needs retuning on real data** (the default 1.5 is now conservative).

### 5. NaN / missing-frame handling

Previously NaN pixels were filled with the per-frame spatial mean and
all-NaN frames with 0.0; after temporal-mean removal those became large
artifact slabs (broadband spectral contamination). Now the per-pixel
temporal mean is computed over valid samples only and all missing
samples become exactly zero anomaly. Pixels never seen (outside the
radar disc, permanent shadow) contribute nothing.

### 6. Smaller correctness fixes

- Grid binning used `astype(int32)` (truncation toward zero), so
  coordinates in (-spacing, 0) were wrongly binned into row/column 0
  instead of being rejected. All projection and remap paths (NumPy,
  Numba, CuPy) now use floor.
- `FrameCube.from_frame_dicts` warns when frame intervals are non-uniform
  by more than 10% (the FFT assumes uniform sampling; missing frames or
  rotation-period jitter smear the omega axis).

## Open issues, in suggested priority order

1. **Antenna-rotation seam (scan-time) within tiles.** Each frame is
   treated as an instant, but pixels are scanned over a full rotation
   (the metadata timestamp is end-of-rotation; radial 0 starts one
   rotation earlier). For a tile away from the seam this is a constant
   time offset (harmless for the fit). Tiles crossed by the seam radial
   contain a one-rotation time discontinuity, and tiles containing the
   radar position mix all azimuths. The seam azimuth is fixed in the
   ship frame, so it sweeps in earth coordinates as the ship turns.
   Suggested: carry per-radial times (already computed in
   `FrameInterpolator.times`) into the projection, store a per-pixel
   mean scan-time offset, and mask tiles whose internal scan-time spread
   exceeds a fraction of the rotation period.

2. **Double resampling with drifting grid origins.** Polar samples are
   binned to a per-frame grid (origin tied to the ship position), then
   remapped to the common cube grid — two nearest-neighbor quantizations
   whose relative phase drifts with the ship. This raises the noise
   floor at high k. Suggested: project polar samples directly onto the
   common-cube grid in the current pipeline, or snap per-frame grid
   origins to a fixed global lattice (requires a shared equirectangular
   reference rather than per-frame `ref_lat/ref_lon`).

3. **Threshold retuning.** `min_snr` (1.5) and `min_fom` (3.0) predate
   the SNR redefinition. Re-derive them from real-data distributions
   (e.g. histogram of tile SNR over a day with known-quiet conditions);
   also consider gating on the new `ux_err`/`uy_err` and `n_ls_points`
   instead of FOM.

4. **Validation against independent measurements.** Compare retrieved
   vectors against the Revelle's ADCP (shallowest bin), GPS drift during
   station-keeping, or drifters. Note the wave-derived current is an
   effective velocity weighted over ~1/(2k) of depth (a few meters to
   tens of meters for these wavenumbers) and includes a fraction of the
   Stokes drift and wind drift — expect systematic differences from an
   ADCP bin at depth.

5. **Heading accuracy.** A gyro bias rotates the whole vector field; a
   slowly varying heading error masquerades as a time-varying current at
   long range (3 km × 0.1°/min ≈ 0.09 m/s of apparent azimuthal motion).
   Worth quantifying the gyro/GPS-heading discrepancy statistics from
   the instrument NetCDF files.

6. **Block length.** 32 frames (~1 minute) is the WaMoS convention, but
   accuracy scales directly with record length through d_omega. If
   stationarity allows, 64-frame blocks halve d_omega and were used in
   the validation here to good effect. Consider making 64 the default
   when memory permits.

7. **Finite-depth c_g in the weight model.** The LS weights use the
   deep-water group velocity as a scale; for shallow ARCTERX sites with
   `current.depth` set, the weights are slightly suboptimal (the fit
   itself uses the exact finite-depth dispersion). Low priority.

## Validation provenance

All numbers above come from `tests/test_current.py` plus ad-hoc
synthetic runs (12-wave sparse spectra, 300-wave directional continuum
with red slope and additive noise, drifting static background fields,
all-NaN frame injection) executed on this branch; see the adversarial
tests `test_static_pattern_with_drift`, `test_all_nan_frames`,
`test_pure_noise_snr_near_one`, and `test_k_max_config`.
