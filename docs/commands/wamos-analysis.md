# `wamos` -- Analysis Tools

Standalone tools for inspecting and analyzing individual radar frames. These are useful for debugging, calibration, and understanding the data before running the full pipeline.

---

## `wamos theta`

Calculate radar beam angle (theta) from bit 13 transitions. Theta is the raw angular position of each radial in the frame's internal coordinate system. Bit 13 alternates between 0 (even degree) and 1 (odd degree), and transitions are used to determine the fractional-degree position of each radial.

```bash
wamos theta /path/to/file.pol --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | Polar file to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot theta vs radial index |

**Output:** Theta range, sample values, and optional diagnostic plot.

Source: `src/wamos_tpw/theta.py`

---

## `wamos bearing`

Convert radar theta to earth heading. Applies corrections for antenna-to-bow offset (BO2RA), heading delta (HDGDL), and gyrocompass offset (GYROC) from the configuration file. Produces bearings in both ship-relative and earth-referenced coordinate systems.

```bash
wamos bearing /path/to/file.pol --config config.yaml --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | Polar file to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot bearing angles |

**Output:** Ship heading, bearing range, sample values in ship and earth coordinates.

Source: `src/wamos_tpw/bearing.py`

---

## `wamos range`

Calculate slant and ground range from frame metadata. Uses the speed of light in air (`c_air = c_vacuum / 1.000273`) and the sampling frequency to compute range resolution, then builds range arrays from sample delay to maximum range. Ground range is computed as `sqrt(slant_range^2 - radar_height^2)`.

```bash
wamos range /path/to/file.pol --config config.yaml --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | Polar file to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot range profiles |

**Output:** Slant range, ground range, range resolution, and optional range profile plot.

Source: `src/wamos_tpw/range.py`

---

## `wamos shadow`

Detect shadow regions in radar data. Identifies angular sectors blocked by ship superstructure or other obstructions by finding persistent low-intensity regions. Requires destreaking first to remove radial artifacts that could mask true shadow regions.

```bash
wamos shadow /path/to/file.pol --config config.yaml --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | Polar file to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot shadow detection results |

**Output:** Theta bias, shadow region statistics, and optional diagnostic plot.

Source: `src/wamos_tpw/shadow.py`

---

## `wamos deramp`

Remove range-dependent intensity fall-off from radar data. Radar intensity naturally decreases with range due to signal spreading. Deramping estimates and removes this trend so that near-range and far-range intensities are comparable.

```bash
wamos deramp /path/to/file.pol --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | Polar file to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot before/after deramping |

**Output:** Intensity statistics and optional before/after comparison plot.

Source: `src/wamos_tpw/deramp.py`

---

## `wamos destreak`

Remove radial streak artifacts from radar data. Streaks are azimuthally narrow, radially persistent intensity anomalies caused by interference or hardware artifacts. Destreaking identifies and removes these per-radial biases.

```bash
wamos destreak /path/to/file.pol --plot
wamos destreak file1.pol file2.pol file3.pol  # Multiple files
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `polar_files` | One or more polar files to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot before/after destreaking |

**Output:** Intensity statistics and optional before/after comparison plot.

Source: `src/wamos_tpw/destreak.py`

---

## `wamos dewind`

Remove wind-direction-dependent intensity bias from radar data. Sea surface roughness varies with wind direction relative to the radar beam, causing upwind/downwind intensity modulation. Dewinding estimates and removes this azimuthal bias.

```bash
wamos dewind /path/to/file.pol --config config.yaml --plot
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | Polar file to process |
| `--config`, `-c` | YAML configuration file |
| `--frame` | Frame index (default: `0`) |
| `--plot` | Plot before/after dewinding |

**Output:** Intensity statistics and optional before/after comparison plot.

Source: `src/wamos_tpw/dewind.py`

---

## `wamos PPS`

Extract and inspect GPS Pulse Per Second (PPS) timing signals from radar data. PPS pulses are encoded in bit 12 of the raw data and provide sub-second timing for each radial. This tool shows pulse locations and counts for each frame.

```bash
wamos PPS /path/to/file.pol --plot
wamos PPS file1.pol file2.pol  # Multiple files
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `filename` | One or more polar files to process |
| `--plot` | Plot PPS pulse locations |

**Output:** PPS pulse counts and radial indices for each frame.

Source: `src/wamos_tpw/pps.py`

---

## `wamos config`

Load and display WAMOS configuration settings. Shows the resolved configuration from a YAML file, or the default configuration if no file is specified. Useful for verifying that configuration values are being parsed correctly.

```bash
wamos config config.yaml
wamos config  # Show defaults
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `config` | YAML configuration file (optional; shows defaults if omitted) |

**Output:** Pretty-printed configuration dictionary.

Source: `src/wamos_tpw/config.py`
