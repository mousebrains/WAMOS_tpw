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

### Single File Pipeline

Process all frames in a single file:

```bash
wamos file-pipeline 2022040400 2022040600 /path/to/POLAR --timing
```

### Multi-File Pipeline

Process multiple files with time windowing and merge:

```bash
wamos files-pipeline 2022040400 2022040600 /path/to/POLAR \
    --window 60 --overlap 0.5 --output-dir /tmp/output
```

With KML output for Google Earth:

```bash
wamos files-pipeline 2022040400 2022040600 /path/to/POLAR \
    --window 60 --output-dir /tmp/output --kml
```

With KMZ output (packaged KML with images):

```bash
wamos files-pipeline 2022040400 2022040600 /path/to/POLAR \
    --window 60 --output-dir /tmp/output --kmz
```

## Analysis Tools

### Bearing Analysis

Analyze and plot bearing calculations:

```bash
wamos bearing 2022040400 2022040600 /path/to/POLAR --plot all
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

### List Frame Timestamps

List timestamps for all frames in a time range:

```bash
wamos list-frames 2022040400 2022040600 /path/to/POLAR
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
from wamos_tpw.config import Config
from wamos_tpw.file_pipeline import FilePipeline

config = Config('wamos_config.yaml')

# Process a single file
fp = FilePipeline('/path/to/file.pol', config=config, qTiming=True)

for frame_pipeline in fp.frames:
    # Access processed intensity
    intensity = frame_pipeline.intensity
    # Access timing information
    timings = frame_pipeline.timings
```

### Multi-File Processing

```python
from wamos_tpw.files_pipeline import FilesMergePipeline, TimeWindowConfig
from wamos_tpw.filenames import Filenames

# Get list of files
filenames = Filenames('2022040400', '2022040600', '/path/to/POLAR')
files = list(filenames)

# Configure time windows
window_config = TimeWindowConfig(
    window_seconds=60.0,
    overlap_fraction=0.5,
    min_frames_per_window=5
)

# Process and merge
pipeline = FilesMergePipeline(files, window_config=window_config)
for merged in pipeline.iter_merged():
    print(f"Merged {merged.n_frames} frames")
    print(f"Time: {merged.start_time} to {merged.end_time}")
    # Access merged intensity grid
    intensity = merged.intensity
```

## Viewer Controls

In interactive viewers:

- **Navigation:** Left/Right arrow keys, or Prev/Next buttons
- **Views:** Press 1 (Polar), 2 (Ship), 3 (Earth)
- **Alternative keys:** p/b (previous), n/f (next)
