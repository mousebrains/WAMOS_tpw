# `revelle` -- Ship Instrument Data

Parse R/V Roger Revelle ship instrument log files into CF-1.13 compliant NetCDF files. **Run this before `wamos`** to generate the instrument NetCDF files needed for per-radial interpolation onto radar beams.

Source: `src/wamos_tpw/instruments/cli.py`

---

## `revelle all`

Process all instruments at once from a cruise data directory.

Expects the directory to contain `serialinstruments/` and `met/data/` subdirectories. Runs all five instrument parsers sequentially and then pre-builds the ShipData `.npy` memmap cache used by worker processes during radar processing.

```bash
revelle all /Volumes/SeaChest/ARCTERX/2022/Wake/cruise/data/ -o ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Cruise data directory (containing `serialinstruments/` and `met/data/`) |
| `--output-dir`, `-o` | Output directory for NetCDF files (default: `.`) |

**Output files:** `gps_abxtwo.nc`, `gyro_sperry.nc`, `mru_phins.nc`, `wind_bridge.nc`, `met_revelle.nc`

Source: `src/wamos_tpw/instruments/cli.py`

---

## `revelle gps`

Parse GPS ABX-Two dual-antenna log files.

Input files have the format `gps_abxtwo_rr_rx2_navbho-YYYY-MM-DD.log` (~1.55 GB/day, ~2 Hz). Lines are grouped by timestamp proximity (<250 ms gap). Within each group, values are extracted from all available NMEA sentence types: GPGGA (position, altitude, fix quality), GPRMC (speed/course over ground), GNGST (position error estimates), GPHDT (true heading), and PASHR,ATT (heading, roll, pitch).

```bash
revelle gps /path/to/serialinstruments/ -o ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Log file or directory of GPS files |
| `--output-dir`, `-o` | Output directory (default: `.`) |
| `--glob`, `-g` | Glob pattern for file matching (default: `gps_abxtwo_rr_rx2_navbho-*`) |

**Output:** `gps_abxtwo.nc` with variables: latitude, longitude, altitude, heading, roll, pitch, sog, cog, n_satellites, hdop, lat_error, lon_error, alt_error, fix_quality.

Source: `src/wamos_tpw/instruments/gps.py`

---

## `revelle gyro`

Parse Sperry gyrocompass heading log files.

Input files have the format `gyro_sperry_rr_heading-YYYY-MM-DD.log` (~575 KB/day, ~5 Hz). Each line contains an ISO 8601 timestamp and a `$HEHDT` sentence with true heading. Lines with `$PPLAN` sentences are skipped.

```bash
revelle gyro /path/to/serialinstruments/ -o ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Log file or directory of gyro files |
| `--output-dir`, `-o` | Output directory (default: `.`) |
| `--glob`, `-g` | Glob pattern for file matching (default: `gyro_sperry_rr_heading-*`) |

**Output:** `gyro_sperry.nc` with variable: heading.

Source: `src/wamos_tpw/instruments/gyro.py`

---

## `revelle mru`

Parse PHINS-III MRU (Motion Reference Unit) log files.

Input files have the format `mru_phinsiii_rr_navbho-YYYY-MM-DD.log` (~1.2 MB/day, ~2 Hz). Uses a multi-line grouped format where only the first line of each group has a timestamp prefix; subsequent lines lack timestamps. A new group starts when a line has a timestamp prefix.

Extracts from: PHGGA (position), PHVTG (speed/course), HEHDT (heading), PASHR (heading, roll, pitch, heave).

```bash
revelle mru /path/to/serialinstruments/ -o ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Log file or directory of MRU files |
| `--output-dir`, `-o` | Output directory (default: `.`) |
| `--glob`, `-g` | Glob pattern for file matching (default: `mru_phinsiii_rr_navbho-*`) |

**Output:** `mru_phins.nc` with variables: latitude, longitude, altitude, heading, roll, pitch, heave, sog, cog.

Source: `src/wamos_tpw/instruments/mru.py`

---

## `revelle wind`

Parse RM Young wind bridge log files.

Input files have the format `wind_bridge_rr-YYYY-MM-DD.log` (~86 KB/day, 1 Hz). Each line contains an ISO 8601 timestamp and a `$WIMWV` sentence with relative wind direction and speed. Only records with status `A` (valid) are kept.

```bash
revelle wind /path/to/serialinstruments/ -o ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | Log file or directory of wind files |
| `--output-dir`, `-o` | Output directory (default: `.`) |
| `--glob`, `-g` | Glob pattern for file matching (default: `wind_bridge_rr-*`) |

**Output:** `wind_bridge.nc` with variables: relative_wind_direction, relative_wind_speed (m/s, converted from knots).

Source: `src/wamos_tpw/instruments/wind.py`

---

## `revelle met`

Parse R/V Revelle MET system data files.

Input files have the format `YYMMDD.MET` (~5-67 MB/day, 1 Hz). These are space-delimited table files with 4 header lines. The time column is HHMMSS (integer seconds since midnight) and the date is extracted from header line 2. Missing values (`-99.0`) are converted to NaN. Wind speeds are converted from knots to m/s.

```bash
revelle met /path/to/met/data/ -o ./output/
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input` | MET file or directory of MET files |
| `--output-dir`, `-o` | Output directory (default: `.`) |
| `--glob`, `-g` | Glob pattern for file matching (default: `*.MET`) |

**Output:** `met_revelle.nc` with variables: wind_speed, wind_direction, wind_speed_2, wind_direction_2, true_wind_speed, true_wind_index, latitude, longitude, heading, course, speed.

Source: `src/wamos_tpw/instruments/met.py`
