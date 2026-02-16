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
