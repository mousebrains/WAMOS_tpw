# `wamos` -- Visualization Commands

Commands for viewing radar data and generating movies.

---

## `wamos view`

Interactive intensity viewer. View raw radar data in polar, ship, or earth coordinates with navigation controls. Supports multiple plot types: intensity, individual bit fields, and cross-frame bit analysis.

```bash
# View intensity in polar coordinates
wamos view 2022040514 2022040515 /path/to/POLAR --plot-intensity

# View in earth coordinates with config
wamos view 2022040514 2022040515 /path/to/POLAR --plot-intensity --view earth --config config.yaml

# Plot bit fields across frames
wamos view 2022040514 2022040515 /path/to/POLAR --plot-bits-across
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time (see Timestamp Formats in README) |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files (`YYYY/MM/DD/HH/` structure) |
| `--groupby`, `-g` | Groupby frequency for file batching (default: `h` for hourly) |
| `--workers`, `-w` | Number of workers |
| `--load` | Actually load the files (not just discover) |
| `--plot-intensity` | Plot intensity (bottom 12 bits) for each frame |
| `--plot-bits` | Plot top 4 bits per frame (uses `--distance-bins`) |
| `--plot-bits-detail` | Plot detailed view of top 4 bits per distance bin |
| `--plot-bits-across` | Plot top 4 bits across all frames with boundaries |
| `--distance-bins` | Distance bins to plot: `N`, `:N`, `M:N` (default: `:21`) |
| `--output-dir`, `-o` | Output directory for saved plots |
| `--max-frames` | Maximum frames to plot (default: `10`) |
| `--cmap` | Colormap for intensity plots (default: `viridis`) |
| `--dpi` | DPI for saved plots (default: `150`) |
| `--view` | Initial view type: `polar`, `ship`, or `earth` (default: `polar`) |
| `--radar-height` | Radar height above water in meters |
| `--config`, `-c` | YAML configuration file |

**Viewer controls:**

- **Navigation**: Left/Right arrow keys or Prev/Next buttons (wraps around)
- **Views**: Press `1` for Polar, `2` for Ship coordinates, `3` for Earth coordinates

Source: `src/wamos_tpw/files.py`

---

## `wamos stitch`

Combine images and movies into larger outputs. Provides four sub-commands for post-processing pipeline outputs.

### `wamos stitch images-to-movie`

Create an MP4 movie from NetCDF output files.

```bash
wamos stitch images-to-movie ./results/ movie.mp4 --fps 4 --cmap plasma
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing NetCDF files |
| `output` | Output MP4 file path |
| `--pattern` | Glob pattern for NetCDF files (default: `merged_*.nc`) |
| `--fps` | Frames per second (default: `2.0`) |
| `--cmap` | Colormap name (default: `viridis`) |
| `--dpi` | Output resolution (default: `150`) |
| `--no-range-rings` | Disable range ring overlays |
| `--no-inset` | Disable ship/wind inset diagram |
| `--max-files` | Maximum number of files to process |

### `wamos stitch images-to-kml`

Create a KML file with ground overlays from NetCDF output files.

```bash
wamos stitch images-to-kml ./results/ output.kml
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing NetCDF files |
| `output` | Output KML file path |
| `--pattern` | Glob pattern (default: `merged_*.nc`) |
| `--max-files` | Maximum number of files to process |

### `wamos stitch images-to-kmz`

Create a self-contained KMZ file from NetCDF output files.

```bash
wamos stitch images-to-kmz ./results/ output.kmz
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_dir` | Directory containing NetCDF files |
| `output` | Output KMZ file path |
| `--pattern` | Glob pattern (default: `merged_*.nc`) |
| `--max-files` | Maximum number of files to process |

### `wamos stitch movies`

Concatenate MP4 files into a single movie.

```bash
wamos stitch movies combined.mp4 --input-dir ./results/ --pattern "*.mp4"
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `output` | Output MP4 file path |
| `input_files` | Input MP4 files (positional, optional) |
| `--input-dir` | Directory containing MP4 files |
| `--pattern` | Glob pattern (default: `*.mp4`) |
| `--reencode` | Re-encode video (slower but more compatible) |

Source: `src/wamos_tpw/stitch.py`
