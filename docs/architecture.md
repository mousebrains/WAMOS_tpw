# Architecture

## Data Flow

```
Filenames -> PolarFile -> Frame -> Theta/Bearing -> IntensityViewer
                               |                  (inherits BaseViewer)
                     ProcessedFrames -> ProcessedViewer
                     (extends Files)              (inherits BaseViewer)
                               |
                            Combine -> Movie generation

plotting.py provides: BaseViewer, quantile_limits, calc_bin_edges, format_nav_title,
                      add_crosshairs, add_range_rings, sort_polar_data
```

## Core Classes

### Filenames (filenames.py)
File discovery with time-based filtering. Expects directory structure `YYYY/MM/DD/HH/YYYYMMDDHHmmss*.pol*`. Supports parallel directory scanning and groupby operations.

### PolarFile (polarfile.py)
Parses `.pol` files (supports `.gz`, `.bz2`, `.xz`, `.lzma` compression). Extracts ASCII header, frame metadata section, and binary uint16 data blocks.

### Frame (frame.py)
Contains radar scan data as uint16 array `(n_bearings, n_distances)`. Data encoding:
- Bottom 12 bits: Radar intensity (0-4095)
- Bit 12: PPS (Pulse Per Second)
- Bit 13: Bearing pulse (used for angle calculation)
- Bits 14-15: Reserved

### Theta (bearing.py)
Calculates radar beam angle from bit 13 transitions. Refines estimate using shadow region alignment.

### Bearing (bearing.py)
Converts theta to ship/earth reference frames using:
- BO2RA (bow-to-radar)
- HDGDL (heading delay)
- GYROC (gyro compass)

Provides cartesian x/y coordinates.

### WamosConfig (config.py)
YAML configuration loader for tower-specific settings (shadow region, offsets, radar height).

### Files (files.py)
High-level interface combining Filenames loading with time-based group iteration. Includes `IntensityViewer` for interactive plotting.

### Timestamp (timestamp.py)
Calculates precise timing for each radial using 1-second timing signal encoded in bit 12, bin 18. Estimates lat/lon positions using ship speed/heading.

### ProcessedFrames (processed.py)
Extends Files with configuration and processing capabilities:
- `process()`: Main processing pipeline - calls refine_theta then destreak_frames for each itergroup. Supports parallel processing when diagnostics are disabled.
- `refine_theta()`: Shadow-based angle refinement
- `deramp_frames()`: Range-dependent intensity fall-off correction
- `destreak_frames()`: Radial streak artifact removal
- `ProcessedViewer`: Interactive visualization with polar, ship, and earth coordinate views.

### Combine (combine.py)
Combines multiple radar frames into earth-referenced images with ship motion compensation:
- `xy_earth()`: Get x/y earth coordinates for frames
- `latlon()`: Get lat/lon coordinates
- `ship_track()`: Continuous ship position during scan
- `grid_parallel_rotated()`: Fast parallel gridding aligned to ship track
- `save_frame()`: Non-interactive frame saving for movie generation

### Deramp (deramp.py)
Removes range-dependent intensity fall-off:
- Calculates quantile intensity profile as function of range (excluding shadow)
- Applies smoothing and subtracts from data

### Destreak (destreak.py)
Removes radial streak artifacts from radar data:
- Detects streaks using derivative analysis
- Applies interpolation to replace streak pixels

## Plotting Classes (plotting.py)

### BaseViewer
Abstract base class for interactive frame viewers:
- Navigation: `_on_prev()`, `_on_next()` with wraparound
- Keyboard handling: Arrow keys, p/n/b/f for navigation, 1/2/3 for views
- Button setup: `_add_nav_buttons()`, `_add_view_buttons()`, `_connect_keyboard()`
- Abstract methods: `_draw_plot()`, `_update_title()`, `_get_frame()`

### IntensityViewer (files.py)
Three view modes inheriting BaseViewer:
- Polar: bearing vs ground distance
- Ship: +X=starboard, +Y=bow
- Earth: +X=East, +Y=North

### ProcessedViewer (processed.py)
Three view modes inheriting BaseViewer:
- Polar: bearing vs ground distance
- Ship: +X=starboard, +Y=bow
- Earth: +X=East, +Y=North

## Plotting Utilities (plotting.py)

- `quantile_limits(data, low_pct, high_pct)`: Calculate colorbar limits from percentiles
- `calc_bin_edges(centers)`: Calculate bin edges from centers for pcolormesh
- `format_nav_title(frame)`: Format ship/wind navigation info for titles
- `add_crosshairs(ax)`: Add crosshairs at origin for coordinate plots
- `add_range_rings(ax, max_range, interval)`: Add range rings to coordinate plots
- `sort_polar_data(bearing, data)`: Sort bearing and reorder data rows for monotonic pcolormesh input
