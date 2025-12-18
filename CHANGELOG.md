# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Decomposed large methods in `bearing.py` for better readability
- Complete API documentation with Sphinx
- Performance documentation and benchmarking guidance
- Integration test for full pipeline

### Changed
- More specific type ignore comments throughout codebase

## [0.1.0] - 2025-12

Initial release of the wamos_tpw package.

### Added
- **Core Data Structures**
  - `Frame` class for single radar scan data with lazy property evaluation
  - `PolarFile` parser supporting .pol files with compression (.gz, .bz2, .xz, .lzma)
  - Custom exception hierarchy (`WamosError`, `PolarFileError`, `ConfigError`, etc.)

- **Coordinate Systems**
  - `Theta` class for radar beam angle calculation from bit 13 transitions
  - `Bearing` class for ship/earth coordinate transformations
  - Shadow region detection and bearing refinement

- **Data Processing**
  - `Deramp` for range-dependent intensity correction
  - `Destreak` for radial streak artifact removal
  - Circular theta handling for seamless 360deg wraparound

- **File Management**
  - `Filenames` for time-based file discovery with glob patterns
  - `Files` and `ProcessedFrames` for high-level iteration
  - Support for `YYYY/MM/DD/HH/YYYYMMDDHHmmss*.pol*` directory structure

- **Visualization**
  - `Combine` for earth-referenced image composition
  - Movie generation with ffmpeg integration
  - Interactive viewers for polar/ship/earth coordinate systems
  - Parallel frame processing with ProcessPoolExecutor

- **Configuration**
  - YAML-based configuration with validation
  - Dataclass-based config structure with `__post_init__` validation
  - Hierarchical settings (shadow, offsets, radar, theta_refinement, etc.)

- **CLI**
  - Unified `wamos` command with subcommands
  - Commands: list, parse, view, process, combine, bearing, timestamp, config, deramp, destreak
  - Comprehensive argument parsing with validation

- **Development**
  - Full test suite (234 tests) with pytest
  - Type hints with mypy validation
  - Ruff linting and formatting
  - Pre-commit hooks for code quality
  - CI/CD with GitHub Actions (Linux, macOS, Windows)
  - Codecov integration for coverage reporting

### Technical Details
- Python 3.13+ required
- Uses speed of light in air: c_air = c_vacuum / 1.000273
- Bit 13 encoding: 0 = even degree, 1 = odd degree
- All bearings normalized to [0, 360)
- Radar height priority: CLI > config > metadata > wind sensor height

[Unreleased]: https://github.com/mousebrains/WAMOS_tpw/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mousebrains/WAMOS_tpw/releases/tag/v0.1.0
