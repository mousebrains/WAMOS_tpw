# Configuration

The `wamos_tpw` package uses YAML configuration files for tower-specific settings.

## Configuration File Format

```yaml
tower: "TOWER_A"

radar:
  height: 25.0  # meters above water

shadow:
  center: 180.0  # degrees from bow (center of shadow region)
  width: 90.0    # total width in degrees

offsets:
  compass: 0.0      # compass offset correction (degrees)
  bow_to_radar: 0.0 # bow-to-radar angle (BO2RA)
  heading_delay: 0.0 # heading delay (HDGDL)
```

## Configuration Options

### `tower`
Identifier for the radar installation. Used for logging and diagnostics.

### `radar.height`
Radar antenna height above water surface in meters. Used for:
- Ground range calculation from slant range
- Earth coordinate transformation

**Priority order:**
1. CLI `--radar-height` argument
2. Config file `radar.height`
3. Frame metadata `radar_height`
4. Wind sensor height (WINDH) from metadata

### `shadow.center`
Center bearing of the shadow region in degrees from bow (0-360). The shadow region is typically caused by ship superstructure blocking the radar beam.

### `shadow.width`
Total width of the shadow region in degrees. The shadow extends from `center - width/2` to `center + width/2`.

Used for:
- Excluding shadow region from theta refinement
- Excluding shadow region from deramp profile calculation

### `offsets.compass`
Compass offset correction in degrees. Applied to convert compass heading to true heading.

### `offsets.bow_to_radar`
Bow-to-radar angle (BO2RA) in degrees. Rotation from ship bow to radar zero angle.

### `offsets.heading_delay`
Heading delay (HDGDL) in degrees. Compensates for latency in heading measurement.

## Using Configuration

Pass the configuration file path to any command:

```bash
wamos process 2022040400 2022040600 /path/to/POLAR --config wamos_config.yaml
wamos combine 2022040400 2022040600 /path/to/POLAR --config wamos_config.yaml --process
```

Or in Python:

```python
from wamos_tpw.config import WamosConfig
from wamos_tpw.processed import ProcessedFrames

config = WamosConfig('wamos_config.yaml')

with ProcessedFrames(
    stime='2022040400',
    etime='2022040600',
    polar_path='/path/to/POLAR',
    config=config
) as pframes:
    for period, frames in pframes.itergroups():
        corrected = pframes.process_group(frames)
```

## Default Values

If no configuration file is provided, defaults are:
- `radar.height`: From frame metadata
- `shadow.center`: None (no shadow masking)
- `shadow.width`: 0 (no shadow masking)
- All offsets: 0.0
