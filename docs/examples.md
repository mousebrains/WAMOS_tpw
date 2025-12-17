# Examples

## Quick Start

After installation:

```bash
pip install wamos_tpw
# or
pipx install wamos_tpw
```

## File Discovery

List available POLAR files in a time range:

```bash
wamos list 2022040400 2022040600 /path/to/POLAR
```

With parallel scanning and grouping by hour:

```bash
wamos list 2022040400 2022040600 /path/to/POLAR --workers 4 --groupby h
```

## Viewing Raw Data

Interactive intensity viewer:

```bash
wamos view 2022040400 2022040600 /path/to/POLAR --plot-intensity
```

Different coordinate systems:
- Polar coordinates (default): `--view polar`
- Ship coordinates (X=starboard, Y=bow): `--view ship`
- Earth coordinates (X=East, Y=North): `--view earth`

```bash
wamos view 2022040400 2022040600 /path/to/POLAR --plot-intensity --view ship
```

## Processing Data

Process frames with deramp and destreak corrections:

```bash
wamos process 2022040400 2022040600 /path/to/POLAR --plot
```

With configuration file:

```bash
wamos process 2022040400 2022040600 /path/to/POLAR --config wamos_config.yaml --plot
```

## Combining Frames

Combine multiple frames into earth-referenced images:

```bash
wamos combine 2022040400 2022040600 /path/to/POLAR --process --plot
```

Generate a movie:

```bash
wamos combine 2022040400 2022040600 /path/to/POLAR --process --movie output.mp4 --fps 10
```

## Analysis Tools

### Bearing Analysis

Analyze and plot bearing calculations:

```bash
wamos bearing 2022040400 2022040600 /path/to/POLAR --plot all
```

### Timestamp Analysis

Analyze timing signals:

```bash
wamos timestamp 2022040400 2022040600 /path/to/POLAR
```

### Standalone Deramp

Test deramping on a single file:

```bash
wamos deramp /path/to/file.pol --plot --quantile 0.10
```

### Standalone Destreak

Test destreaking on a single file:

```bash
wamos destreak /path/to/file.pol --plot
```

## Python API

### Basic Usage

```python
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.frame import Frame

# Load a polar file
pf = PolarFile('/path/to/file.pol.gz')
frame = pf.frame()

# Access intensity data
intensity = frame.intensity  # (n_bearings, n_distances) uint16 array

# Get range values
slant_range = frame.slant_range()  # meters
ground_range = frame.ground_range(radar_height=25.0)  # meters
```

### Processing Pipeline

```python
from wamos_tpw.config import WamosConfig
from wamos_tpw.processed import ProcessedFrames

config = WamosConfig('wamos_config.yaml')

with ProcessedFrames(
    stime='2022040400',
    etime='2022040600',
    polar_path='/path/to/POLAR',
    groupby='h',
    config=config
) as pframes:
    for period, frames in pframes.itergroups():
        frames = list(frames)
        corrected = pframes.process_group(frames)
        for frame, corr in zip(frames, corrected):
            frame.corrected_intensity = corr
```

### Combining Frames

```python
from wamos_tpw.combine import Combine

# After processing frames...
combine = Combine(frames, config, radar_height=25.0)

# Get earth coordinates
x_earth, y_earth = combine.xy_earth_all()
lat, lon = combine.latlon_all()
intensity = combine.intensity_all()

# Ship track
ship_lat, ship_lon = combine.ship_track()

# Grid for plotting
x_edges, y_edges, gridded = combine.grid_parallel()
```

## Viewer Controls

In interactive viewers:

- **Navigation:** Left/Right arrow keys, or Prev/Next buttons
- **Views:** Press 1 (Polar), 2 (Ship), 3 (Earth)
- **Alternative keys:** p/b (previous), n/f (next)
