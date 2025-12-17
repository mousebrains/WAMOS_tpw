#! /usr/bin/env python3
#
# WamosConfig - Generic configuration class for WAMOS processing
# Loads from YAML files and provides settings for all processing modules
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ShadowConfig:
    """Configuration for radar shadow region."""
    center: float = 180.0      # Degrees from bow (aft)
    width: float = 90.0        # Total width in degrees (±45°)

    @property
    def start(self) -> float:
        """Start angle of shadow region (degrees)."""
        return (self.center - self.width / 2) % 360

    @property
    def end(self) -> float:
        """End angle of shadow region (degrees)."""
        return (self.center + self.width / 2) % 360


@dataclass
class OffsetsConfig:
    """Configuration for compass and mounting offsets."""
    compass: float = 0.0           # Compass offset (CMPOFF)
    bow_to_radar: float = 0.0      # Bow to radar angle (BO2RA)
    heading_delay: float = 0.0     # Heading delay (HDGDL)


@dataclass
class ThetaRefinementConfig:
    """Configuration for theta/bearing refinement using shadow region."""
    enabled: bool = True
    search_range: float = 55.0      # Degrees to search around expected shadow
    min_frames: int = 3             # Minimum frames for refinement
    intensity_threshold: float = 0.2  # Fraction of max for shadow detection


@dataclass
class RadarConfig:
    """Configuration for radar physical parameters."""
    height: float | None = None     # Height above water (meters)
    tower: str = "UNKNOWN"          # Tower identifier


@dataclass
class PlottingConfig:
    """Configuration for plotting."""
    cmap: str = "viridis"
    intensity_vmin: float = 0.0
    intensity_vmax: float = 4095.0
    dpi: int = 150


@dataclass
class DestreakConfig:
    """Configuration for destreaking algorithm."""
    min_streak_length: int = 10       # Minimum contiguous flagged bins required
    threshold_sigma: float = 7.5      # Number of one-sided standard deviations for threshold


class WamosConfig:
    """
    Generic configuration class for WAMOS processing.

    Loads settings from YAML files and provides configuration for all
    processing modules including bearing calculation, plotting, and
    data processing.

    Example YAML structure:
        tower: "TOWER_A"

        radar:
          height: 25.0

        shadow:
          center: 180.0
          width: 90.0

        offsets:
          compass: 0.0
          bow_to_radar: 0.0
          heading_delay: 0.0

        theta_refinement:
          enabled: true
          search_range: 10.0
          min_frames: 3
          intensity_threshold: 0.2

        plotting:
          cmap: "viridis"
          intensity_vmin: 0
          intensity_vmax: 4095

        destreak:
          min_streak_length: 10
          threshold_sigma: 7.5

    Example:
        >>> config = WamosConfig('radar_config.yaml')
        >>> print(config.shadow.center)
        180.0
        >>> print(config.radar.height)
        25.0
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration, optionally loading from YAML file.

        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        # Initialize with defaults
        self.tower: str = "UNKNOWN"
        self.shadow = ShadowConfig()
        self.offsets = OffsetsConfig()
        self.theta_refinement = ThetaRefinementConfig()
        self.radar = RadarConfig()
        self.plotting = PlottingConfig()
        self.destreak = DestreakConfig()

        # Additional arbitrary settings
        self._extra: dict[str, Any] = {}

        if config_path is not None:
            self.load(config_path)

    def load(self, config_path: str | Path) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            return

        # Tower identification
        self.tower = config.get('tower', self.tower)

        # Radar parameters
        radar = config.get('radar', {})
        self.radar.height = radar.get('height', self.radar.height)
        self.radar.tower = config.get('tower', self.radar.tower)

        # Shadow region
        shadow = config.get('shadow', {})
        self.shadow.center = shadow.get('center', self.shadow.center)
        self.shadow.width = shadow.get('width', self.shadow.width)

        # Compass/mounting offsets
        offsets = config.get('offsets', {})
        self.offsets.compass = offsets.get('compass', self.offsets.compass)
        self.offsets.bow_to_radar = offsets.get('bow_to_radar', self.offsets.bow_to_radar)
        self.offsets.heading_delay = offsets.get('heading_delay', self.offsets.heading_delay)

        # Theta refinement parameters
        refine = config.get('theta_refinement', {})
        self.theta_refinement.enabled = refine.get('enabled', self.theta_refinement.enabled)
        self.theta_refinement.search_range = refine.get('search_range',
                                                         self.theta_refinement.search_range)
        self.theta_refinement.min_frames = refine.get('min_frames',
                                                       self.theta_refinement.min_frames)
        self.theta_refinement.intensity_threshold = refine.get('intensity_threshold',
                                                                self.theta_refinement.intensity_threshold)

        # Plotting configuration
        plotting = config.get('plotting', {})
        self.plotting.cmap = plotting.get('cmap', self.plotting.cmap)
        self.plotting.intensity_vmin = plotting.get('intensity_vmin', self.plotting.intensity_vmin)
        self.plotting.intensity_vmax = plotting.get('intensity_vmax', self.plotting.intensity_vmax)
        self.plotting.dpi = plotting.get('dpi', self.plotting.dpi)

        # Destreak configuration
        destreak = config.get('destreak', {})
        self.destreak.min_streak_length = destreak.get('min_streak_length',
                                                        self.destreak.min_streak_length)
        self.destreak.threshold_sigma = destreak.get('threshold_sigma',
                                                      self.destreak.threshold_sigma)

        # Store any extra configuration sections
        known_sections = {'tower', 'radar', 'shadow', 'offsets', 'theta_refinement', 'plotting',
                          'destreak'}
        for key, value in config.items():
            if key not in known_sections:
                self._extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an arbitrary configuration value.

        Args:
            key: Configuration key (can use dot notation for nested access)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Check extra config first
        if key in self._extra:
            return self._extra[key]

        # Support dot notation for nested access
        parts = key.split('.')
        obj: Any = self

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __repr__(self) -> str:
        return (
            f"WamosConfig(tower={self.tower!r}, "
            f"shadow={self.shadow.center}±{self.shadow.width/2}°, "
            f"radar_height={self.radar.height})"
        )

    def __enter__(self) -> 'WamosConfig':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        pass


SAMPLE_CONFIG = """# WAMOS Processing Configuration
# Tower-specific settings

tower: "TOWER_A"

radar:
  height: 25.0  # Height above water in meters

shadow:
  center: 180.0  # Degrees from bow (180 = aft)
  width: 90.0    # Total width (±45°)

offsets:
  compass: 0.0        # Compass offset (CMPOFF)
  bow_to_radar: 0.0   # Bow to radar angle (BO2RA)
  heading_delay: 0.0  # Heading delay (HDGDL)

theta_refinement:
  enabled: true
  search_range: 10.0       # Degrees to search around expected shadow
  min_frames: 3            # Minimum frames for refinement
  intensity_threshold: 0.2 # Fraction of max for shadow detection

plotting:
  cmap: "viridis"
  intensity_vmin: 0
  intensity_vmax: 4095
  dpi: 150

destreak:
  min_streak_length: 10   # Minimum contiguous flagged bins required
  threshold_sigma: 7.5    # Number of one-sided standard deviations for threshold
"""


def add_subparser(subparsers) -> None:
    """Register the 'config' subcommand."""
    p = subparsers.add_parser(
        'config',
        help='Show/validate configuration',
        description="Test WAMOS configuration loading"
    )
    p.add_argument("config", nargs="?", type=str, default=None,
                   help="YAML configuration file")
    p.add_argument("--create-sample", action="store_true",
                   help="Create a sample configuration file")
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'config' command."""
    if args.create_sample:
        output_file = Path("wamos_config.yaml")
        output_file.write_text(SAMPLE_CONFIG)
        print(f"Created sample configuration: {output_file}")
        return

    # Load and display configuration
    config = WamosConfig(args.config)
    print(f"Configuration: {config}")
    print()
    print(f"Tower: {config.tower}")
    print(f"Radar height: {config.radar.height}")
    print(f"Shadow region: {config.shadow.start:.1f}° to {config.shadow.end:.1f}°")
    print(f"Shadow center: {config.shadow.center}°")
    print(f"Shadow width: {config.shadow.width}°")
    print()
    print("Offsets:")
    print(f"  Compass: {config.offsets.compass}°")
    print(f"  Bow to radar: {config.offsets.bow_to_radar}°")
    print(f"  Heading delay: {config.offsets.heading_delay}°")
    print()
    print("Theta refinement:")
    print(f"  Enabled: {config.theta_refinement.enabled}")
    print(f"  Search range: {config.theta_refinement.search_range}°")
    print(f"  Min frames: {config.theta_refinement.min_frames}")
    print(f"  Intensity threshold: {config.theta_refinement.intensity_threshold}")


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Test WAMOS configuration loading")
    parser.add_argument("config", nargs="?", type=str, default=None,
                        help="YAML configuration file")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create a sample configuration file")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
