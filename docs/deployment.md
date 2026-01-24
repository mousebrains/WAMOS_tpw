# Deployment Guide

This guide covers installation, configuration, and production deployment of the `wamos_tpw` package.

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 100 GB | 1+ TB SSD |
| Network | 100 Mbps | 1 Gbps |

Processing is CPU-bound and benefits from multiple cores for parallel frame processing.

### Software

- **Python**: 3.13 or later
- **Operating System**: Linux (recommended), macOS, or Windows
- **FFmpeg**: Required for movie generation (`wamos combine --movie`)

### Python Dependencies

Core dependencies (installed automatically):
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7
- pyyaml >= 6.0
- pandas >= 2.0
- pyproj >= 3.6
- tqdm >= 4.65
- zstandard >= 0.20
- opencv-python >= 4.8

Optional dependencies:
- xarray, netCDF4, zarr (for NetCDF/Zarr export)

## Installation

### From PyPI (Recommended)

```bash
pip install wamos_tpw
```

Or with isolated environment using pipx:

```bash
pipx install wamos_tpw
```

### From Source

```bash
git clone https://github.com/mousebrains/WAMOS_tpw.git
cd WAMOS_tpw
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

With all optional dependencies:

```bash
pip install -e ".[all]"
```

### Verify Installation

```bash
wamos --help
wamos --version
```

## Data Directory Structure

WAMOS expects polar files organized in a time-based directory structure:

```
/path/to/POLAR/
├── 2022/
│   ├── 04/
│   │   ├── 04/
│   │   │   ├── 00/
│   │   │   │   ├── 20220404000015_TOWER.pol.gz
│   │   │   │   ├── 20220404000115_TOWER.pol.gz
│   │   │   │   └── ...
│   │   │   ├── 01/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── 05/
│   │       └── ...
│   └── ...
└── ...
```

Pattern: `YYYY/MM/DD/HH/YYYYMMDDHHmmss*.pol*`

Supported compression formats: `.gz`, `.bz2`, `.xz`, `.lzma`, `.zst`

## Configuration

### Tower Configuration File

Create a YAML configuration file for each radar tower:

```yaml
# wamos_config.yaml
tower: "TOWER_A"

radar:
  height: 25.0  # meters above water

shadow:
  center: 180.0  # degrees from bow
  width: 90.0    # total width in degrees

offsets:
  compass: 0.0
  bow_to_radar: 0.0
  heading_delay: 0.0
```

See [Configuration](configuration.md) for detailed option descriptions.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WAMOS_CONFIG` | Default config file path | None |
| `WAMOS_POLAR_PATH` | Default polar data path | None |
| `WAMOS_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |

## Production Deployment

### Systemd Service (Linux)

Create a systemd service for continuous processing:

```ini
# /etc/systemd/system/wamos-processor.service
[Unit]
Description=WAMOS Radar Data Processor
After=network.target

[Service]
Type=simple
User=wamos
Group=wamos
WorkingDirectory=/opt/wamos
ExecStart=/opt/wamos/venv/bin/wamos files-pipeline \
    --polar-path /data/POLAR \
    --output /data/processed \
    --config /etc/wamos/config.yaml \
    --window 60 \
    --continuous
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable wamos-processor
sudo systemctl start wamos-processor
sudo systemctl status wamos-processor
```

### Cron Job

For periodic batch processing:

```bash
# /etc/cron.d/wamos
# Process last hour of data every hour
0 * * * * wamos /opt/wamos/venv/bin/wamos files-pipeline \
    $(date -d '1 hour ago' +\%Y\%m\%d\%H00) \
    $(date +\%Y\%m\%d\%H00) \
    /data/POLAR \
    --config /etc/wamos/config.yaml \
    --output /data/processed \
    >> /var/log/wamos/processor.log 2>&1
```

### Docker Deployment

Dockerfile:

```dockerfile
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install wamos_tpw
RUN pip install --no-cache-dir wamos_tpw

# Create non-root user
RUN useradd -m -s /bin/bash wamos
USER wamos
WORKDIR /home/wamos

# Default command
ENTRYPOINT ["wamos"]
CMD ["--help"]
```

Build and run:

```bash
docker build -t wamos_tpw .

# Interactive viewer (requires X11 forwarding)
docker run -it --rm \
    -v /data/POLAR:/data/POLAR:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    wamos_tpw view 2022040400 2022040600 /data/POLAR --plot-intensity

# Batch processing
docker run --rm \
    -v /data/POLAR:/data/POLAR:ro \
    -v /data/output:/output \
    wamos_tpw files-pipeline 2022040400 2022040600 /data/POLAR \
    --output /output --netcdf
```

Docker Compose for multi-container setup:

```yaml
# docker-compose.yml
version: '3.8'

services:
  wamos-processor:
    build: .
    volumes:
      - /data/POLAR:/data/POLAR:ro
      - /data/output:/output
      - ./config.yaml:/etc/wamos/config.yaml:ro
    command: >
      files-pipeline
      --polar-path /data/POLAR
      --output /output
      --config /etc/wamos/config.yaml
      --window 60
      --continuous
    restart: unless-stopped

  wamos-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - /data/output:/data/output:ro
    # Add your API server command here
```

## Output Formats

### KML Export

Generate Google Earth overlays:

```bash
wamos files-pipeline 2022040400 2022040600 /data/POLAR --kml --output /data/kml/
```

### NetCDF Export

Export to NetCDF for scientific analysis:

```bash
pip install "wamos_tpw[export]"
wamos files-pipeline 2022040400 2022040600 /data/POLAR --netcdf --output /data/nc/
```

### Movie Generation

Requires FFmpeg:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Generate movie
wamos combine 2022040400 2022040600 /data/POLAR --process --movie output.mp4
```

## Monitoring and Logging

### Log Configuration

Set log level via environment or CLI:

```bash
export WAMOS_LOG_LEVEL=DEBUG
wamos files-pipeline ...

# Or per-command
wamos files-pipeline ... --log-level DEBUG
```

### Log Output

Logs include:
- File discovery and parsing progress
- Processing timing information
- Error details with file context

Example log output:

```
2024-01-15 10:30:00 INFO  Found 1234 polar files in time range
2024-01-15 10:30:01 INFO  Processing window 2024-01-15 10:00 - 10:01
2024-01-15 10:30:05 INFO  Merged 60 frames, output: merged_20240115100000.nc
```

### Health Checks

For Docker/Kubernetes deployments:

```bash
# Check if wamos is responsive
wamos --version && echo "healthy" || echo "unhealthy"
```

## Troubleshooting

### Common Issues

**"No polar files found"**
- Verify directory structure matches `YYYY/MM/DD/HH/` pattern
- Check file permissions
- Verify time range format (YYYYMMDDHHmm or "YYYY-MM-DD HH:MM")

**Memory errors during processing**
- Reduce `--window` size for files-pipeline
- Process smaller time ranges
- Increase system swap space

**Slow processing**
- Enable parallel processing with `--workers N`
- Use SSD storage for polar files
- Consider pre-decompressing frequently accessed files

**FFmpeg errors for movie generation**
- Ensure FFmpeg is installed and in PATH
- Check output directory write permissions
- Verify sufficient disk space

### Debug Mode

Enable detailed debugging:

```bash
wamos files-pipeline ... --log-level DEBUG --timing
```

### Getting Help

- GitHub Issues: https://github.com/mousebrains/WAMOS_tpw/issues
- Documentation: https://github.com/mousebrains/WAMOS_tpw#readme

## Security Considerations

### File Permissions

```bash
# Restrict config file access (may contain paths)
chmod 600 /etc/wamos/config.yaml

# Data directories
chmod 755 /data/POLAR
chmod 755 /data/output
```

### Running as Non-Root

Always run the processor as a non-root user:

```bash
# Create dedicated user
sudo useradd -r -s /bin/false wamos

# Set ownership
sudo chown -R wamos:wamos /data/output
```

### Network Security

If exposing any API endpoints:
- Use HTTPS/TLS
- Implement authentication
- Restrict to internal networks where possible
