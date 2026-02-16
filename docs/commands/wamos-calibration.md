# `wamos` -- Calibration Tools

Tools for calibrating timing between radar data and ship instruments. Accurate timing is critical for motion correction -- even small offsets cause systematic position errors when projecting radar data onto earth coordinates.

---

## `wamos timeshift`

Estimate the time shift between radar timestamps and GPS time. Compares radar PPS (Pulse Per Second) pulse timing against GPS second boundaries to compute a per-frame timing offset. The estimated `time_shift` value should be set in the YAML configuration file for use by the processing pipeline.

**Prerequisites:** Requires output from both `wamos metadata` (radar metadata NetCDF) and `revelle gps` (GPS NetCDF).

```bash
# Generate required inputs first
wamos metadata 2022040514 2022040515 /path/to/POLAR -o polar_metadata.nc
revelle gps /path/to/serialinstruments/ -o ./output/

# Estimate time shift
wamos timeshift polar_metadata.nc ./output/gps_abxtwo.nc --plot -o timeshift.nc
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `metadata_nc` | Path to metadata NetCDF file (from `wamos metadata`) |
| `ship_data_nc` | Path to ship GPS NetCDF file (from `revelle gps`) |
| `--output`, `-o` | Output NetCDF file with estimated time shifts |
| `--plot` | Generate diagnostic plots |

**Output:** Estimated time shift in seconds and optional NetCDF file with per-frame timing data.

Source: `src/wamos_tpw/timeshift.py`

---

## `wamos pps-timing`

Analyze PPS pulse timing statistics across multiple frames. Extracts PPS pulse indices and timestamps from all frames in a time range, computes timing intervals and drift statistics, and writes results to NetCDF. Useful for assessing radar clock stability and PPS signal quality.

```bash
wamos pps-timing 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml -o pps_timing.nc --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time (see Timestamp Formats in README) |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--output`, `-o` | Output NetCDF file for PPS timing data |
| `--plot` | Generate diagnostic plots |
| `--max-frames` | Maximum frames to process |
| `--progress` / `--no-progress` | Show/hide progress bar (default: show) |

**Output:** PPS timing statistics and optional NetCDF file with per-frame pulse data.

Source: `src/wamos_tpw/pps_timing.py`

---

## `wamos hard-returns`

Detect persistent high-intensity returns from ship structure. Sweeps timing offsets to find the value that maximizes spatial concentration of stable, high-intensity targets (masts, antennas, railings). These "hard returns" appear at consistent angular positions across frames and can be used to refine the bearing offset.

```bash
wamos hard-returns 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml -o hard_returns.nc --threshold 3500
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--output`, `-o` | Output NetCDF file for hard return masks |
| `--threshold` | Intensity threshold for detection (default: auto) |
| `--min-persistence` | Minimum fraction of frames a target must persist (default: `0.5`) |
| `--max-range` | Maximum range to analyze in meters (default: all) |
| `--max-frames` | Maximum frames to process |
| `--progress` / `--no-progress` | Show/hide progress bar (default: show) |
| `--plot` / `--no-plot` | Generate/suppress diagnostic plots (default: show) |

**Output:** Hard return mask with angular distribution and range profiles, optional NetCDF file and diagnostic plots.

Source: `src/wamos_tpw/hard_returns.py`
