# wamos_tpw

WAMOS (Wave and Meteorological Observation System) marine radar data processing pipeline.

Loads, parses, processes, and visualizes radar scan data from `.pol` files.

## Installation

```bash
pip install wamos_tpw
# or
pipx install wamos_tpw
```

## Quick Start

```bash
# List available POLAR files
wamos list 2022040400 2022040600 /path/to/POLAR

# View raw intensity data
wamos view 2022040400 2022040600 /path/to/POLAR --plot-intensity

# Process frames (deramp + destreak)
wamos process 2022040400 2022040600 /path/to/POLAR --plot

# Combine frames into earth-referenced image
wamos combine 2022040400 2022040600 /path/to/POLAR --process --plot

# Generate movie
wamos combine 2022040400 2022040600 /path/to/POLAR --process --movie output.mp4
```

## Available Commands

| Command | Description |
|---------|-------------|
| `wamos list` | Discover POLAR files in time range |
| `wamos parse` | Parse single POLAR file |
| `wamos view` | Interactive intensity viewer |
| `wamos process` | Process frames with viewer |
| `wamos combine` | Combine frames to earth coordinates |
| `wamos bearing` | Bearing analysis and plotting |
| `wamos timestamp` | Timestamp analysis |
| `wamos config` | Show/validate configuration |
| `wamos deramp` | Standalone deramp tool |
| `wamos destreak` | Standalone destreak tool |

## Documentation

Full documentation available in the `docs/` directory:
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)

## Author

Pat Welch, pat@mousebrains.com

With help from [Anthropic's Claude](https://www.anthropic.com/).

Derived from code developed at [CORDC at SIO](https://cordc.ucsd.edu).

## License

MIT
