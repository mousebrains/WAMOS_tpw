# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WAMOS (Wave and Meteorological Observation System) marine radar data processing pipeline. Loads, parses, processes, and visualizes radar scan data from `.pol` files (supports `.gz`, `.bz2`, `.xz`, `.lzma` compression).

## Build and Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/test_frame.py

# Run with coverage
pytest --cov=wamos_tpw

# Type checking
mypy src/wamos_tpw

# Linting
ruff check src/wamos_tpw
```

## Architecture

### Data Flow
```
Filenames -> PolarFile -> Frame -> Theta/Bearing -> IntensityViewer
                               |
                     ProcessedFrames -> ProcessedViewer
                               |
                            Combine -> Movie generation
```

### Core Classes

- **Filenames** (`filenames.py`): File discovery with time-based filtering. Expects `YYYY/MM/DD/HH/YYYYMMDDHHmmss*.pol*` structure.
- **PolarFile** (`polarfile.py`): Parses `.pol` files - ASCII header, frame metadata, binary uint16 data blocks.
- **Frame** (`frame.py`): Radar scan data as uint16 array `(n_bearings, n_distances)`. Bottom 12 bits = intensity (0-4095), bit 12 = PPS, bit 13 = bearing pulse.
- **Theta/Bearing** (`bearing.py`): Calculates radar beam angle from bit 13 transitions. Converts to ship/earth reference frames using BO2RA, HDGDL, GYROC.
- **WamosConfig** (`config.py`): YAML configuration loader for tower-specific settings.
- **Files** (`files.py`): High-level interface combining Filenames with time-based group iteration. Includes `IntensityViewer`.
- **ProcessedFrames** (`processed.py`): Extends Files with processing: `process()` -> `refine_theta()` -> `destreak_frames()`. Includes `ProcessedViewer`.
- **Combine** (`combine.py`): Combines frames into earth-referenced images with ship motion compensation.
- **Deramp** (`deramp.py`): Removes range-dependent intensity fall-off.
- **Destreak** (`destreak.py`): Removes radial streak artifacts.

### Ship Instrument Parsers (`instruments/`)

- **nmea** (`instruments/nmea.py`): Shared NMEA parsing — timestamp splitting, checksum validation, lat/lon conversion.
- **netcdf_writer** (`instruments/netcdf_writer.py`): Shared CF-1.13 NetCDF writer with zlib compression.
- **gps** (`instruments/gps.py`): GPS ABX-Two dual-antenna — groups multi-sentence epochs by timestamp proximity.
- **gyro** (`instruments/gyro.py`): Sperry gyrocompass heading.
- **mru** (`instruments/mru.py`): PHINS-III MRU — multi-line groups with single timestamp prefix.
- **wind** (`instruments/wind.py`): RM Young wind bridge — relative wind direction and speed.
- **met** (`instruments/met.py`): Ship MET system — space-delimited table format.

### Key Technical Details

- **Range**: Uses c_air = c_vacuum / 1.000273. Ground range = sqrt(slant_range^2 - radar_height^2).
- **Bearing**: Bit 13 = 0 (even degree), bit 13 = 1 (odd degree). All bearings normalized to [0, 360).
- **Radar height priority**: CLI arg > config file > frame metadata > WINDH.

## CLI Entry Points

### `revelle` — Ship Instrument Data (run first)

Parse R/V Roger Revelle ship instrument log files into CF-1.13 NetCDF files. **Run this before `wamos`** to generate the instrument NetCDF files needed for interpolation onto radar beams.

```bash
# Process all instruments at once from cruise data directory
revelle all /Volumes/SeaChest/ARCTERX/2022/Wake/cruise/data/ -o ./output/

# Or process individual instruments
revelle gps /path/to/serialinstruments/ -o ./output/
revelle gyro /path/to/serialinstruments/ -o ./output/
revelle mru /path/to/serialinstruments/ -o ./output/
revelle wind /path/to/serialinstruments/ -o ./output/
revelle met /path/to/met/data/ -o ./output/
```

Source: `src/wamos_tpw/instruments/cli.py`. Instrument parsers live in `src/wamos_tpw/instruments/`.

### `wamos` — Radar Data Processing

All radar commands via `wamos` CLI (`src/wamos_tpw/cli.py`). Each module registers its subparser via `add_subparser()`:

```bash
wamos list|parse|view|process|combine|bearing|timestamp|config|deramp|destreak
```

## Viewer Controls

Navigation: Arrow keys or Prev/Next buttons. Views: 1=Polar, 2=Ship, 3=Earth.
