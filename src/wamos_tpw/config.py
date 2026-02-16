#! /usr/bin/env python3
#
# Config - Generic configuration class for WAMOS processing
# Loads from YAML files and provides settings for all processing modules
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import pprint
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from wamos_tpw.exceptions import ConfigError

# Default config file name within package data
_DEFAULT_CONFIG = "default_wamos.yaml"

logger = logging.getLogger(__name__)

__all__ = ["Config", "ConfigSchema", "NullConfig"]


# =============================================================================
# Configuration Schema (for validation)
# =============================================================================


@dataclass
class OffsetsSchema:
    """Schema for coordinate offset parameters."""

    bow_to_radar: float = 0.0  # BO2RA: Angle from bow to radar beam (degrees)
    heading_delay: float = 0.0  # HDGDL: Heading delay correction (degrees)
    compass: float = 0.0  # Compass offset correction (degrees)


@dataclass
class BiasSchema:
    """Schema for bias parameters."""

    range: float = 0.0  # Range bias in meters
    theta: float = 0.0  # Theta bias in degrees


@dataclass
class ShadowRegionSchema:
    """Schema for a shadow region definition."""

    LHS: float = 0.0  # Left-hand side angle (degrees)
    RHS: float = 0.0  # Right-hand side angle (degrees)


@dataclass
class ShadowSchema:
    """Schema for shadow detection parameters."""

    range_fraction: float = 0.1  # Fraction of range bins for shadow detection
    theta_refinement: bool = False  # Use shadow edges to refine theta
    angle_range: float = 10.0  # +/- degrees to search around expected angles
    # Named shadow regions (e.g., "aft", "forward") are dynamic


@dataclass
class ThetaRefinementSchema:
    """Schema for theta refinement parameters."""

    enabled: bool = True  # Enable theta refinement
    min_frames: int = 3  # Minimum frames needed


@dataclass
class DerampSchema:
    """Schema for deramp parameters."""

    order: int = 4  # Polynomial order for fit


@dataclass
class DewindSchema:
    """Schema for dewind parameters."""

    order: int = 2  # Polynomial order for fit


@dataclass
class ConfigSchema:
    """
    Schema defining valid configuration structure and types.

    Used for validation at load time to catch configuration errors early.
    """

    time_shift: float = 0.0  # Timestamp correction in seconds (added to frame timestamps)
    theta_refinement: ThetaRefinementSchema = field(default_factory=ThetaRefinementSchema)
    shadow: ShadowSchema = field(default_factory=ShadowSchema)
    bias: BiasSchema = field(default_factory=BiasSchema)
    offsets: OffsetsSchema = field(default_factory=OffsetsSchema)
    deramp: DerampSchema = field(default_factory=DerampSchema)
    dewind: DewindSchema = field(default_factory=DewindSchema)


# Type mapping for validation
# Note: YAML doesn't distinguish int/float for whole numbers, so numeric types accept both
_TYPE_MAP = {
    "time_shift": (int, float),
    "theta_refinement.enabled": bool,
    "theta_refinement.min_frames": int,
    "shadow.range_fraction": (int, float),
    "shadow.theta_refinement": bool,
    "shadow.angle_range": (int, float),
    "bias.range": (int, float),
    "bias.theta": (int, float),
    "offsets.bow_to_radar": (int, float),
    "offsets.heading_delay": (int, float),
    "offsets.compass": (int, float),
    "deramp.order": int,
    "dewind.order": int,
}

# Value constraints
_CONSTRAINTS = {
    "shadow.range_fraction": lambda v: 0.0 < v <= 1.0,
    "theta_refinement.min_frames": lambda v: v >= 1,
    "deramp.order": lambda v: v >= 1,
    "dewind.order": lambda v: v >= 1,
}


class Config:
    """
    Configuration class for WAMOS processing.

    Provides:
    - YAML configuration loading from files or package defaults
    - Dot-notation access for nested keys (e.g., config["shadow.range_fraction"])
    - Attribute-style access (e.g., config.shadow.range_fraction)
    - Optional validation against schema with type checking

    Example::

        config = Config("tower_config.yaml")
        config.validate()  # Raises ConfigError if invalid

        # Access values
        range_frac = config["shadow.range_fraction"]
        deramp_order = config.deramp.order
    """

    def __init__(
        self,
        filename: str | Path | None = None,
        *,
        validate: bool = False,
    ):
        """
        Initialize configuration, optionally loading from YAML file.

        Args:
            filename: Path to YAML config file. If None, loads the default
                     configuration from package data.
            validate: If True, validate configuration after loading.
                     Raises ConfigError if validation fails.
        """
        if filename:
            filename = Path(filename)
            self._filename = str(filename.name)
            self._config = self._load(filename)
        else:
            self._filename = _DEFAULT_CONFIG
            self._config = self._load_default()

        if validate:
            self.validate()

    def _load(self, filename: Path) -> dict:
        """
        Load configuration from YAML file.

        Args:
            filename: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not filename.exists():
            raise FileNotFoundError(f"Config file not found: {filename}")

        with open(filename) as f:
            return yaml.safe_load(f) or {}

    def _load_default(self) -> dict:
        """Load the default configuration from package data."""
        config_file = files("wamos_tpw.data").joinpath(_DEFAULT_CONFIG)
        return yaml.safe_load(config_file.read_text()) or {}

    def validate(self, section: str | None = None) -> list[str]:
        """
        Validate configuration against schema.

        Checks:
        - Type correctness for known parameters
        - Value constraints (e.g., range_fraction must be 0 < x <= 1)

        Args:
            section: Optional section name to validate (e.g., "global", "roger revelle").
                    If None, validates all sections.

        Returns:
            List of warning messages for minor issues (e.g., unknown keys)

        Raises:
            ConfigError: If validation fails with type or constraint errors
        """
        errors = []
        warnings = []

        # Determine which sections to validate
        if section:
            sections = {section: self._config.get(section, {})}
        else:
            sections = self._config

        for section_name, section_config in sections.items():
            if not isinstance(section_config, dict):
                continue

            prefix = f"[{section_name}] " if section_name else ""

            # Validate known keys
            for key_path, expected_type in _TYPE_MAP.items():
                try:
                    value = self._get_nested(section_config, key_path)
                    if value is None:
                        continue  # Optional key not present

                    # Type check
                    if not isinstance(value, expected_type):
                        type_names = (
                            expected_type.__name__
                            if isinstance(expected_type, type)
                            else " or ".join(t.__name__ for t in expected_type)
                        )
                        errors.append(
                            f"{prefix}{key_path}: expected {type_names}, "
                            f"got {type(value).__name__} ({value!r})"
                        )
                        continue

                    # Constraint check
                    if key_path in _CONSTRAINTS:
                        constraint = _CONSTRAINTS[key_path]
                        if not constraint(value):
                            errors.append(f"{prefix}{key_path}: invalid value {value!r}")

                except (KeyError, TypeError):
                    pass  # Key not present, which is OK

        if errors:
            error_msg = f"Configuration validation failed for {self._filename}:\n"
            error_msg += "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(error_msg)

        return warnings

    def _get_nested(self, config: dict, key_path: str) -> Any:
        """Get a nested value from config using dot notation."""
        obj = config
        for key in key_path.split("."):
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]
        return obj

    def keys(self):
        """Return configuration keys."""
        return self._config.keys()

    def get(self, key, default: Any = None) -> Any:
        """Get configuration value with default."""
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        keys = key.split(".")  # Support nested keys with dot notation
        obj = self._config
        for k in keys:
            if not isinstance(obj, dict) or k not in obj:
                raise KeyError(f"{self._filename} no key '{key}'")
            obj = obj[k]
        return self._child_config(obj) if isinstance(obj, dict) else obj

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation for nested keys."""
        keys = key.split(".")
        obj = self._config
        for k in keys[:-1]:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        obj[keys[-1]] = value

    def _child_config(self, obj: dict) -> Config:
        """Create a child Config object from a nested dictionary."""
        child = Config()
        child._filename = self._filename
        child._config = obj
        return child

    def __len__(self) -> int:
        """Return number of configuration items."""
        return len(self._config)

    def __bool__(self) -> bool:
        """Check if configuration is non-empty."""
        return bool(self._config)

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains a key."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getattr__(self, name: str) -> Any:
        """Attribute-style access to configuration (e.g., config.theta_refinement)."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from e

    def update(self, rhs: dict | Config) -> None:
        """Update configuration with another dictionary."""
        if isinstance(rhs, Config):
            rhs = rhs._config
        self._config.update(rhs)

    def items(self):
        """Return configuration items."""
        return self._config.items()

    def values(self):
        """Return configuration values."""
        return self._config.values()

    def __iter__(self):
        """Iterate over configuration keys."""
        return iter(self._config)

    def __repr__(self) -> str:
        name = self._filename or "None"
        return f"{type(self).__name__}({name}):\n{pprint.pformat(self._config)}"


class NullConfig(Config):
    """
    Null object pattern for Config - provides safe defaults without file loading.

    Use NullConfig when a Config object is expected but no actual configuration
    is available. All get() calls return the default value, and __getitem__ raises
    KeyError for any key.

    This eliminates the need for `if config is not None` checks throughout the code.

    Example::

        def process(frame, config: Config | None = None):
            # Old pattern (error-prone):
            # if config and "shadow" in config:
            #     ...

            # New pattern (type-safe):
            cfg = config or NullConfig()
            shadow_frac = cfg.get("shadow.range_fraction", 0.1)  # Always works

    See Also
    --------
    Config : Full configuration with file loading and validation.
    """

    _instance: NullConfig | None = None

    def __new__(cls) -> NullConfig:
        """Singleton pattern - only one NullConfig instance needed."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize empty NullConfig (no file loading)."""
        # Avoid re-initialization in singleton
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._filename = "NullConfig"
        self._config: dict = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Always return default value."""
        return default

    def __getitem__(self, key: str) -> Any:
        """Raise KeyError for any key (no config available)."""
        raise KeyError(f"NullConfig has no key '{key}'")

    def __contains__(self, key: str) -> bool:
        """NullConfig contains no keys."""
        return False

    def __bool__(self) -> bool:
        """NullConfig is falsy (no configuration)."""
        return False

    def validate(self, section: str | None = None) -> list[str]:
        """NullConfig always validates (no config to validate)."""
        return []

    def __repr__(self) -> str:
        return "NullConfig()"


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("config", nargs="?", type=str, default=None, help="YAML configuration file")


def add_subparser(subparsers) -> None:
    """Register the 'config' subcommand."""
    p = subparsers.add_parser(
        "config", help="Show/validate configuration", description="Test WAMOS configuration loading"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'config' command."""
    # Load configuration
    try:
        config = Config(args.config)
    except (OSError, yaml.YAMLError, ConfigError):
        logging.exception("Failed to load configuration: %s", args)
        return

    # Normal display mode
    logging.info("Configuration: %s", config)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Test WAMOS configuration loading")

if __name__ == "__main__":
    main()
