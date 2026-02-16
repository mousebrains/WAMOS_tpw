# `wamos` -- File Discovery and Inspection

Commands for discovering, listing, and inspecting POLAR radar files.

---

## `wamos list`

List and discover POLAR files in a time range. Scans the directory tree (`YYYY/MM/DD/HH/`) for `.pol` files (including compressed variants) whose timestamps fall within the specified window. Uses parallel directory scanning when there are 4 or more hour directories.

```bash
wamos list 2022040514 2022040515 /path/to/POLAR
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time (see Timestamp Formats in README) |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files (`YYYY/MM/DD/HH/` structure) |
| `--workers`, `-w` | Number of worker processes (default: CPU count) |

**Output:** Prints discovered file paths and total count.

Source: `src/wamos_tpw/filenames.py`

---

## `wamos parse`

Parse a single POLAR file and display its structure. Shows file-level information, the ASCII header (optionally), and first-frame statistics including timestamp, array shape, intensity range, and bit field info.

```bash
wamos parse /path/to/file.pol.gz --show-header
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | One or more polar file(s) to parse |
| `--show-header` | Show full ASCII header |
| `--config` | Path to configuration file |

**Output:** File info, header content, frame metadata (timestamp, shape, intensity statistics, PPS/bearing bit counts).

Source: `src/wamos_tpw/polarfile.py`

---

## `wamos list-frames`

List frame timestamps and repeat times for every frame in a time interval. Useful for inspecting frame cadence and identifying gaps in coverage.

```bash
wamos list-frames 2022040514 2022040515 /path/to/POLAR
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |

**Output:** Per-frame timestamp and repeat_time, with inter-frame gaps.

Source: `src/wamos_tpw/list_frames.py`

---

## `wamos stream-list`

Streaming file discovery. Discovers files incrementally and shows progress as new files are found. Uses a context manager to start/stop the discovery process.

```bash
wamos stream-list 2022040514 2022040515 /path/to/POLAR
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--workers`, `-w` | Number of workers |

**Output:** Incremental file discovery with batch progress updates.

Source: `src/wamos_tpw/streaming_filenames.py`

---

## `wamos metadata`

Extract frame metadata to a CF-compliant NetCDF file. Processes all frames in a time range using parallel workers and writes metadata fields (latitude, longitude, heading, ship speed, wind, PPS info) to a single NetCDF file. The output file is used by `wamos timeshift` for timing calibration.

```bash
wamos metadata 2022040514 2022040515 /path/to/POLAR -o polar_metadata.nc
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `stime` | Start time |
| `etime` | End time |
| `polar_path` | Root directory of POLAR files |
| `--config`, `-c` | YAML configuration file |
| `--output`, `-o` | Output NetCDF file path (default: `polar_metadata.nc`) |
| `--workers`, `-w` | Number of parallel workers (default: auto) |
| `--progress` / `--no-progress` | Show/hide progress bar (default: show) |

**Output:** `polar_metadata.nc` with per-frame metadata variables.

Source: `src/wamos_tpw/metadata.py`
