# `wamos` -- Core Processing Commands

Pipeline commands for processing radar frames into motion-corrected composite images.

---

## `wamos files-pipeline`

Merge frames into motion-corrected composite images. This is the primary processing command: it loads polar files in a time range, applies per-frame corrections (deramp, destreak, dewind, bearing), interpolates ship motion onto each radial using PPS timing, projects onto a UTM grid, and merges frames within sliding time windows. Output formats include NetCDF, PNG, GeoTIFF, KML/KMZ, and MP4 movies.

```bash
wamos files-pipeline 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --ship-data ./output/ -o ./results/ \
    --mp4 movie.mp4 --kml --geotiff
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time (see Timestamp Formats in README) |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files (`YYYY/MM/DD/HH/` structure) |
| `--config`, `-c` | YAML configuration file |
| `--window` | Window duration (default: `60s`). Accepts: `60`, `60s`, `1.5m`, `1h`, `0.5d` |
| `--overlap` | Overlap fraction between windows (default: `0.5` = 50%) |
| `--min-frames` | Minimum frames per window (default: `5`) |
| `--resolution-scale` | Grid resolution multiplier (default: `1.0`) |
| `--grid-spacing` | Grid cell size in meters (default: auto from range resolution) |
| `--interpolate` | Fill NaN gaps using nearest neighbor interpolation |
| `--output-dir`, `-o` | Output directory for merged images |
| `--format` | Output format: `netcdf`, `png`, or `both` (default: `netcdf`) |
| `--mp4` | Generate MP4 movie file (requires ffmpeg) |
| `--fps` | Frames per second for MP4 (default: `2.0`) |
| `--geotiff` | Write georeferenced GeoTIFF files |
| `--kml` | Generate KML file with ground overlays |
| `--kmz` | Generate self-contained KMZ file |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--tolerance` | Time tolerance multiplier (default: `1.2`) |
| `--timing`, `-t` | Show timing statistics |
| `--plot` | Show interactive viewer |
| `--ship-data` | Directory with instrument NetCDF files (from `revelle`) |
| `--streaming` | Use streaming file discovery |
| `--progress` / `--no-progress` | Show/hide progress bars (default: show) |
| `--memory-stats` | Show detailed memory usage statistics |
| `--max-windows` | Maximum number of windows to process |
| `--pending-multiplier` | Multiplier for n_workers to set max in-flight file loads (default: `3.0`) |
| `--max-queued-merges` | Max windows queued for merge thread (default: `4`) |

**Output:** Per-window NetCDF files (`merged_*.nc`), PNG images, GeoTIFF files, KML/KMZ overlays, and/or MP4 movie.

Source: `src/wamos_tpw/files_pipeline.py`

---

## `wamos stream-pipeline`

Streaming merge pipeline. Same processing as `files-pipeline` but discovers and processes files incrementally, starting immediately as files are found. Ideal for large datasets or near-real-time processing.

```bash
wamos stream-pipeline 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --ship-data ./output/ -o ./results/
```

**Arguments:** Same as `files-pipeline` except always uses streaming file discovery (no `--streaming` flag needed).

Source: `src/wamos_tpw/streaming_pipeline.py`

---

## `wamos frame-pipeline`

Process single frames through the full per-frame pipeline. Used for testing and benchmarking the frame processing stages: deramp, destreak, dewind, bearing calculation, and optional projection. Can compare executor types (thread pool, process pool, priority pool).

```bash
# Benchmark with all executor types
wamos frame-pipeline 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --all-executors --timing

# View frame in earth coordinates
wamos frame-pipeline 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --view earth --ship-data ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |
| `--frame`, `-f` | Frame index to process (default: `0`) |
| `--timing`, `-t` | Show timing statistics |
| `--save`, `-s` | Save intermediate results |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--plot`, `-p` | Display diagnostic plots for each pipeline stage |
| `--polar` | Display polar plots of Raw, Destreaked, Deramped, Dewinded |
| `--view` | View final intensity: `polar`, `ship`, or `earth` |
| `--ship-data` | Directory with instrument NetCDF files |
| `--radar-height` | Radar height above water in meters |
| `--progress` / `--no-progress` | Show/hide progress bars (default: show) |
| `--threadpool` | Use ThreadPoolExecutor |
| `--processpool` | Use ProcessPoolExecutor |
| `--prioritypool` | Use PriorityProcessExecutor |
| `--all-executors` | Benchmark all executor types |

Source: `src/wamos_tpw/frame_pipeline.py`

---

## `wamos interpolator`

Test per-radial metadata interpolation between frames. Uses PPS pulse timing to assign sub-second timestamps to each radial within a frame, then interpolates ship position, heading, and motion from instrument data at each radial's timestamp. Uses triplet collection (previous, current, next frames) for smooth interpolation. Can optionally project dewinded intensity onto a UTM grid.

```bash
# Test interpolation with projection and viewer
wamos interpolator 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --ship-data ./output/ --project --plot

# Write per-frame NetCDF files
wamos interpolator 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --ship-data ./output/ --project -o ./frames/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |
| `--tolerance` | Time tolerance multiplier (default: `1.2`) |
| `--timing`, `-t` | Show timing statistics |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--project`, `-p` | Project dewinded intensity onto UTM grid |
| `--plot` | Plot the projected intensity (requires `--project`) |
| `--netcdf-dir`, `-o` | Output directory for per-frame NetCDF files (requires `--project`) |
| `--ship-data` | Directory with instrument NetCDF files |
| `--grid-spacing` | Grid cell size in meters (default: auto) |
| `--progress` / `--no-progress` | Show/hide progress bars (default: show) |

Source: `src/wamos_tpw/interpolator.py`

---

## `wamos current`

Extract surface current vectors from sequential radar frames via 3D FFT
dispersion fitting: blocks of frames become space-time cubes, analysis
windows are fitted with a Doppler-shifted dispersion shell (coarse search +
sub-bin least-squares refinement with formal uncertainties), and tiles are
assembled into current maps. Optional temporal compositing and a joint
multi-scale field inversion for higher spatial resolution.

```bash
# Per-block tile maps
wamos current 2022040514 2022040515 /path/to/POLAR \
    --ship-data ./output/ -o ./currents/

# 15-minute inverse-variance composites on top of the block maps
wamos current 2022040514 2022040515 /path/to/POLAR \
    --ship-data ./output/ -o ./currents/ --composite-minutes 15

# High-resolution field inversion from 2 km + 1 km windows
wamos current 2022040514 2022040515 /path/to/POLAR \
    --ship-data ./output/ -o ./currents/ --field --window-sizes 2000,1000
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |
| `--depth` | Water depth in meters (default: deep water) |
| `--block-frames` | Frames per analysis block (default: `32`) |
| `--block-overlap` | Overlap between blocks (default: `0.5`) |
| `--sub-region-size` | Analysis window side length in meters (default: `2000`) |
| `--search-radius` | Maximum current speed searched, m/s (default: `3.0`) |
| `--min-snr` | Minimum SNR to accept an estimate (default: `1.5`) |
| `--composite-minutes` | Write inverse-variance composites over windows of this many minutes |
| `--field` | Joint regularized field inversion per block (`current_field_*.nc`) |
| `--window-sizes` | Comma-separated window sizes in meters, e.g. `2000,1000` |
| `--field-spacing` | Field inversion cell size in meters (default: `250`) |
| `--field-correlation-length` | Smoothness prior length scale in meters (default: `1000`) |
| `--field-sigma-prior` | Expected current variation over the correlation length, m/s (default: `0.3`) |
| `--output-dir`, `-o` | Output directory |
| `--format` | `netcdf`, `png`, or `both` (default: `netcdf`) |
| `--ship-data` | Directory with instrument NetCDF files |
| `--grid-spacing` | Grid cell size in meters (default: auto) |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--plot` | Show diagnostic plots |

Notes:

- Tiles containing the radar or crossed by the antenna rotation seam are
  masked (`current.mask_seam`, default on).
- The smoothness prior is the field inversion's resolution knob:
  `sigma_prior / correlation_length` is the largest current gradient the
  prior allows without penalty. Resolving 0.3 m/s features at 1 km
  requires roughly `--field-correlation-length 250 --field-sigma-prior 0.5`.
- See `docs/resolution_validation.md` for measured transfer functions of
  the variants and `docs/current_extraction_review.md` for the algorithm
  review.

Source: `src/wamos_tpw/current_pipeline.py`

---

## `wamos cube-diag`

Whole-cube diagnostic for current extraction: 2x2 figure with time-averaged
intensity plus sub-cube current vectors, kx-omega and ky-omega spectrum
slices with the fitted dispersion curves, and the (Ux, Uy) search surface.

```bash
wamos cube-diag 2022040514 2022040515 /path/to/POLAR \
    --ship-data ./output/ -o ./diag/
```

Source: `src/wamos_tpw/current_pipeline.py` (`run_cube_diag`)
