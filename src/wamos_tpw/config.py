#! /usr/bin/env python3
#
# Config - Generic configuration class for WAMOS processing
# Loads from YAML files and provides settings for all processing modules
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
import pprint
from typing import Any

import yaml

# Default config file name within package data
_DEFAULT_CONFIG = "default_wamos.yaml"


class Config:
    """
    Configuration class for WAMOS processing.
    """

    def __init__(self, filename: str | Path | None = None):
        """
        Initialize configuration, optionally loading from YAML file.

        Args:
            filename: Path to YAML config file. If None, loads the default
                     configuration from package data.
        """
        if filename:
            filename = Path(filename)
            self._filename = str(filename.name)
            self._config = self._load(filename)
        else:
            self._filename = _DEFAULT_CONFIG
            self._config = self._load_default()

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
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

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
    except Exception:
        logging.exception("Failed to load configuration: %s", args)
        return

    # Normal display mode
    logging.info("Configuration: %s", config)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Test WAMOS configuration loading")

# Backward compatibility alias
WamosConfig = Config


if __name__ == "__main__":
    main()
