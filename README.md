# wamos_tpw

[![CI](https://github.com/mousebrains/WAMOS_tpw/actions/workflows/ci.yml/badge.svg)](https://github.com/mousebrains/WAMOS_tpw/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mousebrains/WAMOS_tpw/branch/main/graph/badge.svg)](https://codecov.io/gh/mousebrains/WAMOS_tpw)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WAMOS (Wave and Meteorological Observation System) marine radar data processing pipeline.

Loads, parses, processes, and visualizes radar scan data from `.pol` files (supports `.gz`, `.bz2`, `.xz`, `.lzma` compression).

## Installation

```bash
pip install wamos_tpw
# or
pipx install wamos_tpw
```

For development:

```bash
git clone https://github.com/mousebrains/WAMOS_tpw.git
cd WAMOS_tpw
pip install -e ".[dev]"
```

## Quick Start

The project provides two CLIs: **`revelle`** for ship instrument data and **`wamos`** for radar processing. Run `revelle` first to generate instrument NetCDF files, then `wamos` for radar processing.

```bash
# 1. Parse all ship instruments into NetCDF (run first)
revelle all /path/to/cruise/data/ -o ./output/

# 2. List available POLAR files
wamos list 2022040514 2022040515 /path/to/POLAR

# 3. View raw intensity data
wamos view 2022040514 2022040515 /path/to/POLAR --plot-intensity

# 4. Process and merge frames into earth-referenced composites
wamos files-pipeline 2022040514 2022040515 /path/to/POLAR \
    --config config.yaml --ship-data ./output/ -o ./results/ --mp4 movie.mp4
```

## Timestamp Formats

All commands that accept start/end times support these formats:

| Format | Example | Resolution |
|--------|---------|------------|
| `YYYY` | `2022` | Year |
| `YYYYMMDD` | `20220405` | Day |
| `YYYYMMDDHH` | `2022040514` | Hour |
| `YYYYMMDDHHmm` | `202204051430` | Minute |
| `YYYYMMDDHHmmss` | `20220405143000` | Second |
| ISO 8601 | `2022-04-05T14:30:00` | Any |

## CLI Commands

### `revelle` -- Ship Instrument Data

Parse R/V Roger Revelle ship instrument log files into CF-1.13 NetCDF files. **Run this before `wamos`** to generate the instrument NetCDF files needed for interpolation onto radar beams.

| Command | Description | Details |
|---------|-------------|---------|
| [`revelle all`](docs/commands/revelle.md#revelle-all) | Process all instruments at once | Parses GPS, gyro, MRU, wind, and MET from a cruise data directory |
| [`revelle gps`](docs/commands/revelle.md#revelle-gps) | Parse GPS data to NetCDF | Trimble ABX-Two dual-antenna GPS (GPGGA, GPRMC, GNGST, GPHDT, PASHR) |
| [`revelle gyro`](docs/commands/revelle.md#revelle-gyro) | Parse gyrocompass data to NetCDF | Sperry Marine gyrocompass heading (HEHDT) |
| [`revelle mru`](docs/commands/revelle.md#revelle-mru) | Parse MRU data to NetCDF | iXBlue PHINS-III inertial navigation (PHGGA, PHVTG, HEHDT, PASHR) |
| [`revelle wind`](docs/commands/revelle.md#revelle-wind) | Parse wind bridge data to NetCDF | RM Young wind sensor (WIMWV) |
| [`revelle met`](docs/commands/revelle.md#revelle-met) | Parse MET system data to NetCDF | Ship MET system (space-delimited table format) |

### `wamos` -- Radar Data Processing

#### Core Processing

| Command | Description | Details |
|---------|-------------|---------|
| [`wamos files-pipeline`](docs/commands/wamos-processing.md#wamos-files-pipeline) | Merge frames into motion-corrected composite images | Full processing pipeline with windowed merging, output to NetCDF/PNG/MP4/GeoTIFF/KML |
| [`wamos stream-pipeline`](docs/commands/wamos-processing.md#wamos-stream-pipeline) | Streaming merge pipeline | Same as `files-pipeline` but discovers and processes files incrementally |
| [`wamos frame-pipeline`](docs/commands/wamos-processing.md#wamos-frame-pipeline) | Process single frames through the pipeline | Per-frame processing: deramp, destreak, bearing, projection |
| [`wamos interpolator`](docs/commands/wamos-processing.md#wamos-interpolator) | Test frame interpolation/extrapolation | Per-radial timestamp and position interpolation using PPS timing |

#### File Discovery and Inspection

| Command | Description | Details |
|---------|-------------|---------|
| [`wamos list`](docs/commands/wamos-files.md#wamos-list) | List POLAR files in time range | Discover `.pol` files matching a time window |
| [`wamos parse`](docs/commands/wamos-files.md#wamos-parse) | Parse single POLAR file | Display header, metadata, and frame info from `.pol` files |
| [`wamos list-frames`](docs/commands/wamos-files.md#wamos-list-frames) | List frame timestamps and repeat times | Print timestamp and repeat_time for every frame in a time interval |
| [`wamos stream-list`](docs/commands/wamos-files.md#wamos-stream-list) | Streaming file discovery | Discover files incrementally and show progress |
| [`wamos metadata`](docs/commands/wamos-files.md#wamos-metadata) | Extract frame metadata to NetCDF | Extract metadata from polar files into a CF-compliant NetCDF file |

#### Visualization

| Command | Description | Details |
|---------|-------------|---------|
| [`wamos view`](docs/commands/wamos-visualization.md#wamos-view) | Interactive intensity viewer | View raw radar data in polar, ship, or earth coordinates |
| [`wamos stitch`](docs/commands/wamos-visualization.md#wamos-stitch) | Combine images/movies into larger outputs | Stitch NetCDF files into movies/KMZ, or concatenate MP4 files |

#### Analysis Tools

| Command | Description | Details |
|---------|-------------|---------|
| [`wamos theta`](docs/commands/wamos-analysis.md#wamos-theta) | Calculate theta angles for a frame | Compute radar beam angle from bit 13 transitions |
| [`wamos bearing`](docs/commands/wamos-analysis.md#wamos-bearing) | Calculate bearing from theta | Convert radar theta to earth heading |
| [`wamos range`](docs/commands/wamos-analysis.md#wamos-range) | Calculate range values for a frame | Compute slant and ground range from frame metadata |
| [`wamos shadow`](docs/commands/wamos-analysis.md#wamos-shadow) | Detect shadow regions in radar data | Find and display blocked angular sectors |
| [`wamos deramp`](docs/commands/wamos-analysis.md#wamos-deramp) | Standalone deramp tool | Remove range-dependent intensity fall-off |
| [`wamos destreak`](docs/commands/wamos-analysis.md#wamos-destreak) | Standalone destreak tool | Remove radial streak artifacts |
| [`wamos dewind`](docs/commands/wamos-analysis.md#wamos-dewind) | Standalone dewind tool | Remove wind-dependent intensity modulation |
| [`wamos PPS`](docs/commands/wamos-analysis.md#wamos-pps) | Standalone PPS tool | Inspect Pulse Per Second timing signals |
| [`wamos config`](docs/commands/wamos-analysis.md#wamos-config) | Show/validate configuration | Load and display a YAML configuration file |

#### Calibration

| Command | Description | Details |
|---------|-------------|---------|
| [`wamos timeshift`](docs/commands/wamos-calibration.md#wamos-timeshift) | Calculate time shift between radar and GPS | Compute timing lag for config `time_shift` parameter |
| [`wamos pps-timing`](docs/commands/wamos-calibration.md#wamos-pps-timing) | Compute PPS-derived timestamps | First-radial timestamps using triplet PPS interpolation |
| [`wamos hard-returns`](docs/commands/wamos-calibration.md#wamos-hard-returns) | Find timing offset from hard returns | Sweep offsets to maximize spatial concentration of stable targets |

## Viewer Controls

When using interactive viewers (`wamos view`, `wamos files-pipeline --plot`):

- **Navigation**: Left/Right arrow keys or Prev/Next buttons
- **Views**: Press `1` for Polar, `2` for Ship coordinates, `3` for Earth coordinates

## Data Flow

```
.pol files -> PolarFile -> Frame -> FramePipeline (per-frame processing)
                                        |
                                   Theta/Bearing
                                   Deramp/Destreak
                                   Shadow detection
                                        |
                                   Interpolator (ship motion, PPS timing)
                                        |
                                   FilesPipeline (time-windowed merging)
                                        |
                                   Output: NetCDF, PNG, MP4, GeoTIFF, KML/KMZ
```

## Directory Structure

POLAR files are expected in the structure `YYYY/MM/DD/HH/YYYYMMDDHHmmss*.pol*`.

## Documentation

- [Architecture](docs/architecture.md) -- Data flow, class relationships, and processing pipelines
- [Configuration](docs/configuration.md) -- YAML config format and options
- [Examples](docs/examples.md) -- Usage examples and workflows
- [Performance](docs/performance.md) -- Optimization and parallelization
- [Deployment](docs/deployment.md) -- Installation and deployment

### Command Reference

- [`revelle` commands](docs/commands/revelle.md) -- Ship instrument parsers
- [`wamos` processing](docs/commands/wamos-processing.md) -- Core processing pipeline commands
- [`wamos` files](docs/commands/wamos-files.md) -- File discovery and inspection
- [`wamos` visualization](docs/commands/wamos-visualization.md) -- Viewers and movie generation
- [`wamos` analysis](docs/commands/wamos-analysis.md) -- Standalone analysis tools
- [`wamos` calibration](docs/commands/wamos-calibration.md) -- Timing calibration tools

## Author

Pat Welch, pat@mousebrains.com

With help from [Anthropic's Claude](https://www.anthropic.com/).

Derived from code developed at [CORDC at SIO](https://cordc.ucsd.edu).

## License

MIT License - see [LICENSE](LICENSE) for details.
