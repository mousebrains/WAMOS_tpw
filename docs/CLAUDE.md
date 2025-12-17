# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WAMOS (Wave and Meteorological Observation System) polar radar data processing pipeline. Loads, parses, and visualizes marine radar scan data from `.pol` files (optionally compressed).

## Architecture

See [architecture.md](architecture.md) for detailed class descriptions and data flow.

## Configuration

See [configuration.md](configuration.md) for YAML config format and options.

## Running the Tools

After installation (`pip install wamos_tpw` or `pipx install wamos_tpw`), use the unified `wamos` command:

```bash
# List available commands
wamos --help

# File discovery
wamos list 2022040400 2022040600 /path/to/POLAR

# Parse single polar file
wamos parse /path/to/file.pol.gz --show-header

# Interactive intensity viewer
wamos view 2022040400 2022040600 /path/to/POLAR --plot-intensity

# View in different coordinate systems (polar/ship/earth)
wamos view 2022040400 2022040600 /path/to/POLAR --plot-intensity --view ship

# Bearing analysis
wamos bearing 2022040400 2022040600 /path/to/POLAR --plot all

# Timestamp analysis
wamos timestamp 2022040400 2022040600 /path/to/POLAR

# Processed frames with interactive viewer
wamos process 2022040400 2022040600 /path/to/POLAR --plot --view polar

# Combine frames into earth coordinate image
wamos combine 2022040400 2022040600 /path/to/POLAR --process --plot

# Generate movie
wamos combine 2022040400 2022040600 /path/to/POLAR --process --movie output.mp4
```

**Viewer Controls:**
- Navigation: Left/Right arrow keys or Prev/Next buttons (wraps around)
- Views: 1=Polar, 2=Ship coords, 3=Earth coords

## Key Technical Details

**Range Calculations:**
- Uses speed of light in air: c_air = c_vacuum / n_air where n_air = 1.000273 (20C, 50% RH)
- SFREQ in metadata is MHz (multiply by 1e6 for Hz)
- Ground range = sqrt(slant_range^2 - radar_height^2)

**Bearing Calculation:**
- Bit 13 = 0: even degree, Bit 13 = 1: odd degree
- All bearings normalized to [0, 360)
- Shadow region alignment refines initial estimate

**Radar Height Priority:** CLI argument > config file > frame metadata (radar_height) > wind sensor height (WINDH)

## Dependencies

- numpy (core data handling)
- scipy (image processing - despiking)
- matplotlib (visualization)
- pyyaml (configuration)
- pandas (time series handling)
