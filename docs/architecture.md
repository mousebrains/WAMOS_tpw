# Architecture

This document describes the architecture of the `wamos_tpw` package, including data flow, class relationships, and processing pipelines.

## High-Level Data Flow

```
                              WAMOS Radar Processing Pipeline
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  ┌─────────────┐    ┌───────────┐    ┌─────────┐    ┌──────────────────┐   │
    │  │  .pol Files │───▶│ PolarFile │───▶│  Frame  │───▶│  FramePipeline   │   │
    │  │  (on disk)  │    │  Parser   │    │ Objects │    │  (per-frame)     │   │
    │  └─────────────┘    └───────────┘    └────┬────┘    └────────┬─────────┘   │
    │                                           │                  │             │
    │                                           ▼                  ▼             │
    │                                    ┌──────────────┐   ┌─────────────┐      │
    │                                    │ Theta/Shadow │   │ Deramp +    │      │
    │                                    │ Detection    │   │ Destreak    │      │
    │                                    └──────┬───────┘   └──────┬──────┘      │
    │                                           │                  │             │
    │                                           └────────┬─────────┘             │
    │                                                    │                       │
    │                                                    ▼                       │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────────┐               │
    │  │ Time Window │◀───│ FilesPipeline │◀───│  Interpolator  │               │
    │  │   Groups    │    │   (multi-file)│    │ (ship motion)  │               │
    │  └──────┬──────┘    └───────────────┘    └────────────────┘               │
    │         │                                                                  │
    │         ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────┐              │
    │  │                    MergedImage                          │              │
    │  │  ┌───────────┐  ┌──────────┐  ┌─────────────────────┐  │              │
    │  │  │ Intensity │  │ UTM Grid │  │ Metadata (heading,  │  │              │
    │  │  │  2D Array │  │  Edges   │  │ speed, wind, time)  │  │              │
    │  │  └───────────┘  └──────────┘  └─────────────────────┘  │              │
    │  └──────────────────────────┬──────────────────────────────┘              │
    │                             │                                              │
    │         ┌───────────────────┼───────────────────┐                         │
    │         ▼                   ▼                   ▼                         │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
    │  │   NetCDF    │    │  GeoTIFF    │    │  KML/KMZ    │                   │
    │  │   Export    │    │   Export    │    │   Export    │                   │
    │  └─────────────┘    └─────────────┘    └─────────────┘                   │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
                           Module Dependencies
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │                        CLI (cli.py)                         │
    │                             │                               │
    │         ┌───────────────────┼───────────────────┐          │
    │         ▼                   ▼                   ▼          │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │ files.py    │    │files_pipeline│   │ plotting.py │     │
    │  │ (Files,     │    │(FilesMerge- │    │ (viewers)   │     │
    │  │  grouping)  │    │ Pipeline)   │    │             │     │
    │  └──────┬──────┘    └──────┬──────┘    └─────────────┘     │
    │         │                  │                               │
    │         │    ┌─────────────┼─────────────┐                 │
    │         │    ▼             ▼             ▼                 │
    │         │  ┌─────┐   ┌──────────┐   ┌──────────┐          │
    │         │  │grid │   │ window   │   │merged_   │          │
    │         │  │.py  │   │ .py      │   │viewer.py │          │
    │         │  └─────┘   └──────────┘   └──────────┘          │
    │         │                                                  │
    │         ▼                                                  │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │frame_       │    │interpolator │    │ priority_   │     │
    │  │pipeline.py  │◀───│.py          │───▶│ executor.py │     │
    │  └──────┬──────┘    └─────────────┘    └─────────────┘     │
    │         │                                                  │
    │    ┌────┴────┬─────────────┬─────────────┐                │
    │    ▼         ▼             ▼             ▼                │
    │ ┌──────┐ ┌────────┐  ┌──────────┐  ┌──────────┐          │
    │ │deramp│ │destreak│  │ theta.py │  │ shadow   │          │
    │ │.py   │ │.py     │  │          │  │ .py      │          │
    │ └──────┘ └────────┘  └────┬─────┘  └──────────┘          │
    │                           │                               │
    │    ┌──────────────────────┼──────────────────────┐       │
    │    ▼                      ▼                      ▼       │
    │ ┌──────────┐       ┌───────────┐          ┌──────────┐   │
    │ │bearing.py│       │ frame.py  │          │ range.py │   │
    │ └────┬─────┘       └─────┬─────┘          └──────────┘   │
    │      │                   │                               │
    │      │    ┌──────────────┴──────────────┐               │
    │      ▼    ▼                             ▼               │
    │  ┌────────────┐               ┌──────────────┐          │
    │  │polarfile.py│               │  config.py   │          │
    │  └─────┬──────┘               └──────────────┘          │
    │        │                                                 │
    │        ▼                                                 │
    │  ┌─────────────┐                                        │
    │  │filenames.py │                                        │
    │  └─────────────┘                                        │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
```

## Processing Pipeline Stages

```
                        Single Frame Processing
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Stage 1: Parse              Stage 2: Geometry              │
    │  ┌─────────────────┐        ┌─────────────────┐            │
    │  │ Binary .pol     │        │ Calculate       │            │
    │  │ ─────────────── │        │ ─────────────── │            │
    │  │ • Header parse  │───────▶│ • Theta angles  │            │
    │  │ • Frame extract │        │ • Shadow mask   │            │
    │  │ • Metadata read │        │ • Ground range  │            │
    │  └─────────────────┘        └────────┬────────┘            │
    │                                      │                      │
    │                                      ▼                      │
    │  Stage 3: Corrections        Stage 4: Cleanup               │
    │  ┌─────────────────┐        ┌─────────────────┐            │
    │  │ Deramp          │        │ Destreak        │            │
    │  │ ─────────────── │        │ ─────────────── │            │
    │  │ • Range profile │───────▶│ • Streak detect │            │
    │  │ • Polynomial fit│        │ • Interpolation │            │
    │  │ • Subtract trend│        │ • Gap fill      │            │
    │  └─────────────────┘        └────────┬────────┘            │
    │                                      │                      │
    │                                      ▼                      │
    │                             ┌─────────────────┐            │
    │                             │ Processed Frame │            │
    │                             │ ─────────────── │            │
    │                             │ • Clean intensity│            │
    │                             │ • Valid mask    │            │
    │                             │ • Geometry data │            │
    │                             └─────────────────┘            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

```
                      Multi-Frame Merge Pipeline
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              Time Window Creation                    │   │
    │  │                                                      │   │
    │  │  File timestamps:  ──●────●────●────●────●────●──   │   │
    │  │                                                      │   │
    │  │  Window 1:         [═══════════]                     │   │
    │  │  Window 2:              [═══════════]                │   │
    │  │  Window 3:                   [═══════════]           │   │
    │  │                        ↑                             │   │
    │  │                    50% overlap                       │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                             │                               │
    │                             ▼                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │            Parallel Frame Processing                 │   │
    │  │                                                      │   │
    │  │   Worker 1: [Frame A] ──▶ [Process] ──▶ [Result]    │   │
    │  │   Worker 2: [Frame B] ──▶ [Process] ──▶ [Result]    │   │
    │  │   Worker 3: [Frame C] ──▶ [Process] ──▶ [Result]    │   │
    │  │   Worker N: [Frame D] ──▶ [Process] ──▶ [Result]    │   │
    │  │                                                      │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                             │                               │
    │                             ▼                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              Ship Motion Interpolation               │   │
    │  │                                                      │   │
    │  │    Frame N-1    Frame N     Frame N+1               │   │
    │  │       ●───────────●───────────●                     │   │
    │  │                   │                                  │   │
    │  │          Interpolate position                        │   │
    │  │          per radial using                            │   │
    │  │          triplet timestamps                          │   │
    │  │                                                      │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                             │                               │
    │                             ▼                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              UTM Grid Projection                     │   │
    │  │                                                      │   │
    │  │   Radar (polar)          Earth (UTM grid)           │   │
    │  │       ╲   │   ╱              ┌───┬───┬───┐          │   │
    │  │        ╲  │  ╱               │   │ ● │   │          │   │
    │  │         ╲ │ ╱                ├───┼───┼───┤          │   │
    │  │    ──────●──────  ═════▶    │ ● │ ● │ ● │          │   │
    │  │         ╱ │ ╲                ├───┼───┼───┤          │   │
    │  │        ╱  │  ╲               │   │ ● │   │          │   │
    │  │       ╱   │   ╲              └───┴───┴───┘          │   │
    │  │                                                      │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                             │                               │
    │                             ▼                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              Window Accumulation                     │   │
    │  │                                                      │   │
    │  │   Frame 1    Frame 2    Frame N                     │   │
    │  │   ┌─────┐    ┌─────┐    ┌─────┐                     │   │
    │  │   │░░░░░│ +  │▒▒▒▒▒│ +  │▓▓▓▓▓│  ═▶  Averaged      │   │
    │  │   │░░░░░│    │▒▒▒▒▒│    │▓▓▓▓▓│      MergedImage   │   │
    │  │   └─────┘    └─────┘    └─────┘                     │   │
    │  │                                                      │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

## Class Relationships

### Core Data Classes

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Core Data Classes                        │
    │                                                             │
    │  ┌─────────────┐         ┌─────────────┐                   │
    │  │  Filenames  │────────▶│  PolarFile  │                   │
    │  │ ─────────── │  finds  │ ─────────── │                   │
    │  │ stime, etime│         │ header      │                   │
    │  │ polar_path  │         │ metadata    │                   │
    │  │ pattern     │         │ frames[]    │                   │
    │  └─────────────┘         └──────┬──────┘                   │
    │                                 │ contains                  │
    │                                 ▼                           │
    │                          ┌─────────────┐                   │
    │                          │    Frame    │                   │
    │                          │ ─────────── │                   │
    │                          │ raw: uint16 │                   │
    │                          │ timestamp   │                   │
    │                          │ metadata    │                   │
    │                          │ config      │                   │
    │                          └──────┬──────┘                   │
    │                                 │ used by                   │
    │              ┌──────────────────┼──────────────────┐       │
    │              ▼                  ▼                  ▼       │
    │       ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
    │       │    Theta    │   │   Deramp    │   │  Destreak   │ │
    │       │ ─────────── │   │ ─────────── │   │ ─────────── │ │
    │       │ theta[]     │   │ intensity   │   │ intensity   │ │
    │       │ sorted_idx  │   │ profile     │   │ streak_mask │ │
    │       └─────────────┘   └─────────────┘   └─────────────┘ │
    │              │                                             │
    │              ▼                                             │
    │       ┌─────────────┐   ┌─────────────┐                   │
    │       │   Shadow    │   │   Bearing   │                   │
    │       │ ─────────── │   │ ─────────── │                   │
    │       │ indices     │   │ heading_ship│                   │
    │       │ thetas      │   │ heading_earth                   │
    │       │ theta_bias  │   │ xy_ship     │                   │
    │       └─────────────┘   │ xy_earth    │                   │
    │                         └─────────────┘                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### Pipeline Classes

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Pipeline Classes                         │
    │                                                             │
    │  ┌───────────────────┐                                     │
    │  │   FramePipeline   │  Single frame processing            │
    │  │ ───────────────── │                                     │
    │  │ frame: Frame      │                                     │
    │  │ theta: Theta      │                                     │
    │  │ shadow: Shadow    │                                     │
    │  │ deramp: Deramp    │                                     │
    │  │ destreak: Destreak│                                     │
    │  │ bearing: Bearing  │                                     │
    │  └─────────┬─────────┘                                     │
    │            │ used by                                        │
    │            ▼                                                │
    │  ┌───────────────────┐                                     │
    │  │ FilesMergePipeline│  Multi-file merge                   │
    │  │ ───────────────── │                                     │
    │  │ files: list[str]  │                                     │
    │  │ config: Config    │                                     │
    │  │ window_config     │                                     │
    │  │ n_workers: int    │                                     │
    │  │ ───────────────── │                                     │
    │  │ iter_merged()     │──────▶ yields MergedImage           │
    │  └─────────┬─────────┘                                     │
    │            │ uses                                           │
    │    ┌───────┴───────┬──────────────┐                        │
    │    ▼               ▼              ▼                        │
    │ ┌──────────┐ ┌───────────┐ ┌──────────────────┐           │
    │ │  grid.py │ │ window.py │ │priority_executor │           │
    │ │ ──────── │ │ ───────── │ │ ──────────────── │           │
    │ │ compute_ │ │ create_   │ │ PriorityProcess  │           │
    │ │ common_  │ │ time_     │ │ Executor         │           │
    │ │ grid()   │ │ windows() │ │ TripletCollector │           │
    │ │ project_ │ │ Window-   │ │ SharedMemory     │           │
    │ │ frame()  │ │ Accum.    │ │ Manager          │           │
    │ │ remap_() │ │           │ │                  │           │
    │ └──────────┘ └───────────┘ └──────────────────┘           │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### Output Classes

```
    ┌─────────────────────────────────────────────────────────────┐
    │                     Output Classes                          │
    │                                                             │
    │  ┌───────────────────────────────────────────────────┐     │
    │  │                   MergedImage                      │     │
    │  │ ───────────────────────────────────────────────── │     │
    │  │ intensity: np.ndarray      # 2D averaged data     │     │
    │  │ x_edges: np.ndarray        # Grid x edges (m)     │     │
    │  │ y_edges: np.ndarray        # Grid y edges (m)     │     │
    │  │ start_time: datetime64     # Window start         │     │
    │  │ end_time: datetime64       # Window end           │     │
    │  │ n_frames: int              # Frames merged        │     │
    │  │ utm_zone: int              # UTM zone number      │     │
    │  │ hemisphere: str            # 'north' or 'south'   │     │
    │  │ center_lat: float          # Grid center lat      │     │
    │  │ center_lon: float          # Grid center lon      │     │
    │  │ grid_spacing: float        # Cell size (m)        │     │
    │  │ mean_heading: float        # Avg ship heading     │     │
    │  │ mean_ship_speed: float     # Avg ship speed       │     │
    │  │ mean_wind_speed: float     # Avg wind speed       │     │
    │  │ mean_wind_direction: float # Avg wind direction   │     │
    │  │ window_index: int          # Window sequence num  │     │
    │  └───────────────────────────────────────────────────┘     │
    │                          │                                  │
    │          ┌───────────────┼───────────────┐                 │
    │          ▼               ▼               ▼                 │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
    │  │write_merged │ │write_geotiff│ │ write_kml   │          │
    │  │_netcdf()    │ │()           │ │ write_kmz   │          │
    │  │             │ │             │ │ write_mp4   │          │
    │  └─────────────┘ └─────────────┘ └─────────────┘          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

## Coordinate Systems

```
                         Coordinate Systems
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  1. POLAR (Radar Native)      2. SHIP (Vessel-Relative)    │
    │                                                             │
    │         θ = 0°                        +Y (Bow)              │
    │           │                              │                  │
    │           │                              │                  │
    │     ────●────  ◀── radar          ────●────                │
    │         / \                        │     │                  │
    │        /   \                       │     │                  │
    │       θ     r                    +X      (Starboard)        │
    │                                                             │
    │  θ: Beam angle (deg)           Origin: Radar position      │
    │  r: Ground range (m)           +X: Starboard               │
    │                                +Y: Bow                      │
    │                                                             │
    │  3. EARTH (Geographic)         4. UTM (Projected)          │
    │                                                             │
    │         +Y (North)                    +Y (Northing)         │
    │           │                              │                  │
    │           │                              │                  │
    │     ────●────                      ────●────                │
    │         │                              │                    │
    │         │                              │                    │
    │       +X (East)                      +X (Easting)           │
    │                                                             │
    │  Origin: Ship position          Origin: Grid center         │
    │  Heading rotates to north       Units: meters               │
    │  Units: meters                  Zone-specific projection    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

## Viewer Inheritance

```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Viewer Class Hierarchy                   │
    │                                                             │
    │                   ┌─────────────────┐                      │
    │                   │   BaseViewer    │  (plotting.py)       │
    │                   │ ─────────────── │                      │
    │                   │ _on_prev()      │                      │
    │                   │ _on_next()      │                      │
    │                   │ _on_key()       │                      │
    │                   │ _add_nav_btns() │                      │
    │                   │ ─────────────── │                      │
    │                   │ _draw_plot()    │◀── abstract          │
    │                   │ _update_title() │◀── abstract          │
    │                   │ _get_frame()    │◀── abstract          │
    │                   └────────┬────────┘                      │
    │                            │                                │
    │            ┌───────────────┼───────────────┐               │
    │            ▼               ▼               ▼               │
    │    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
    │    │Intensity-   │ │Processed-   │ │ Merged-     │        │
    │    │Viewer       │ │Viewer       │ │ Viewer      │        │
    │    │(files.py)   │ │(processed)  │ │(merged_     │        │
    │    │             │ │             │ │ viewer.py)  │        │
    │    │─────────────│ │─────────────│ │─────────────│        │
    │    │ Views:      │ │ Views:      │ │ Features:   │        │
    │    │ • Polar     │ │ • Polar     │ │ • Play/Stop │        │
    │    │ • Ship      │ │ • Ship      │ │ • Animation │        │
    │    │ • Earth     │ │ • Earth     │ │ • Ship/Wind │        │
    │    │             │ │             │ │   insets    │        │
    │    └─────────────┘ └─────────────┘ └─────────────┘        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

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

### Theta (theta.py)
Calculates radar beam angle from 12-bit counter encoded in distance bins 18-20. Distributes fractional degrees within each run using interpolation.

### Shadow (shadow.py)
Detects shadow regions blocked by ship structure. Uses edge detection to refine shadow boundaries and optionally adjusts theta bias.

### Bearing (bearing.py)
Converts theta to ship/earth reference frames using:
- BO2RA (bow-to-radar)
- HDGDL (heading delay)
- GYROC (gyro compass)

Provides cartesian x/y coordinates.

### Config (config.py)
YAML configuration loader for tower-specific settings (shadow region, offsets, radar height).

### Files (files.py)
High-level interface combining Filenames loading with time-based group iteration. Includes `IntensityViewer` for interactive plotting.

### Deramp (deramp.py)
Removes range-dependent intensity fall-off:
- Calculates quantile intensity profile as function of range (excluding shadow)
- Applies smoothing and subtracts from data

### Destreak (destreak.py)
Removes radial streak artifacts from radar data:
- Detects streaks using derivative analysis
- Applies interpolation to replace streak pixels

---

## Pipeline Classes

### FramePipeline (frame_pipeline.py)
Single frame processing pipeline combining deramp and destreak operations.

### FilesMergePipeline (files_pipeline.py)
Multi-file processing pipeline with time windowing and parallel execution:
- Groups frames into overlapping time windows
- Processes windows in parallel using PriorityProcessExecutor
- Outputs merged earth-referenced images

### Grid Module (grid.py)
UTM grid computation and projection:
- `compute_common_grid()`: Create UTM grid covering all frames
- `project_frame_to_common_grid()`: Project polar frame to UTM
- `remap_to_common_grid()`: Remap between different grid resolutions

### Window Module (window.py)
Time window creation and accumulation:
- `create_time_windows()`: Create overlapping windows from timestamps
- `WindowAccumulator`: Accumulate frames onto common grid

### Interpolator (interpolator.py)
Multi-frame interpolation with ship motion correction:
- Collects frames in temporal triplets
- Interpolates ship position per radial
- Projects to common UTM grid

### MergedImage (merged_image.py)
Dataclass containing merged radar image with metadata:
- intensity: 2D averaged intensity array
- x_edges, y_edges: Grid edges in meters
- UTM zone and hemisphere
- Mean heading, ship speed, wind data

---

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

---

## Plotting Utilities (plotting.py)

- `quantile_limits(data, low_pct, high_pct)`: Calculate colorbar limits from percentiles
- `calc_bin_edges(centers)`: Calculate bin edges from centers for pcolormesh
- `format_nav_title(frame)`: Format ship/wind navigation info for titles
- `add_crosshairs(ax)`: Add crosshairs at origin for coordinate plots
- `add_range_rings(ax, max_range, interval)`: Add range rings to coordinate plots
- `sort_polar_data(bearing, data)`: Sort bearing and reorder data rows for monotonic pcolormesh input

---

## Projection (projection.py)

UTM projection and coordinate transformation:
- `get_utm_zone()`: Determine UTM zone from longitude
- `get_utm_epsg()`: Get EPSG code for UTM zone
- `create_utm_transformer()`: Create pyproj transformer
- `transform_to_utm()`: Convert lat/lon to UTM coordinates

---

## Priority Executor (priority_executor.py)

Multi-process executor with priority scheduling:
- Supports high/medium/low priority task queues
- Manages worker processes with graceful shutdown
- Includes TripletCollector for frame grouping
- SharedMemoryManager for inter-process data sharing
