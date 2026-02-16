#!/usr/bin/env python3
"""
WAMOS radar data processing CLI.

Master command that provides access to all WAMOS processing tools via subcommands.
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

from wamos_tpw.exceptions import ConfigError, PolarFileError, ValidationError, WamosError
from wamos_tpw.logging_config import add_logging_arguments, setup_logging

logger = logging.getLogger(__name__)


def _validate_path(path: str, must_exist: bool = True, is_file: bool = False) -> Path:
    """
    Validate a path argument.

    Args:
        path: Path string to validate
        must_exist: If True, path must exist
        is_file: If True, path must be a file (not directory)

    Returns:
        Path object

    Raises:
        ValidationError: If path validation fails
    """
    p = Path(path)
    if must_exist and not p.exists():
        raise ValidationError(f"Path does not exist: {path}", parameter="path")
    if is_file and p.exists() and not p.is_file():
        raise ValidationError(f"Not a file: {path}", parameter="path")
    return p


def _handle_error(error: Exception) -> None:
    """Log user-friendly error message and exit."""
    if isinstance(error, ConfigError):
        logger.error(f"Configuration error: {error}")
    elif isinstance(error, PolarFileError):
        logger.error(f"Polar file error: {error}")
    elif isinstance(error, ValidationError):
        logger.error(f"Validation error: {error}")
    elif isinstance(error, WamosError):
        logger.error(f"Error: {error}")
    elif isinstance(error, FileNotFoundError):
        logger.error(f"File not found: {error}")
    elif isinstance(error, ValueError):
        logger.error(f"Invalid value: {error}")
    else:
        logger.error(f"Unexpected error: {error}")
    sys.exit(1)


def main() -> None:
    """Main entry point for the wamos CLI."""
    parser = argparse.ArgumentParser(
        prog="wamos",
        description="WAMOS marine radar data processing tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wamos files-pipeline "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR --plot
  wamos file-pipeline "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR
  wamos view "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR --plot-intensity
  wamos list "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR
  wamos parse /path/to/file.pol --show-header
""",
    )

    # Global options
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be done without executing"
    )
    add_logging_arguments(parser)

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available processing commands",
        metavar="COMMAND",
    )

    # Register subcommands from each module
    from wamos_tpw import (
        bearing,
        config,
        deramp,
        destreak,
        dewind,
        filenames,
        files,
        files_pipeline,
        frame_pipeline,
        interpolator,
        list_frames,
        polarfile,
        pps,
        shadow,
        stitch,
        streaming_filenames,
        streaming_pipeline,
        theta,
    )
    from wamos_tpw import range as range_module  # Avoid shadowing builtin

    files.add_subparser(subparsers)  # wamos view
    bearing.add_subparser(subparsers)  # wamos bearing
    theta.add_subparser(subparsers)  # wamos theta
    range_module.add_subparser(subparsers)  # wamos range
    filenames.add_subparser(subparsers)  # wamos list
    polarfile.add_subparser(subparsers)  # wamos parse
    config.add_subparser(subparsers)  # wamos config
    deramp.add_subparser(subparsers)  # wamos deramp
    destreak.add_subparser(subparsers)  # wamos destreak
    shadow.add_subparser(subparsers)  # wamos shadow
    dewind.add_subparser(subparsers)  # wamos dewind
    pps.add_subparser(subparsers)  # wamos PPS
    frame_pipeline.add_subparser(subparsers)  # wamos frame-pipeline
    files_pipeline.add_subparser(subparsers)  # wamos files-pipeline
    interpolator.add_subparser(subparsers)  # wamos interpolator
    list_frames.add_subparser(subparsers)  # wamos list-frames
    streaming_filenames.add_subparser(subparsers)  # wamos stream-list
    streaming_pipeline.add_subparser(subparsers)  # wamos stream-pipeline
    stitch.add_subparser(subparsers)  # wamos stitch

    # Optional subcommands (not yet committed to the repository)
    for _mod_name in ("metadata", "timeshift", "pps_timing", "hard_returns"):
        try:
            _mod = importlib.import_module(f"wamos_tpw.{_mod_name}")
            _mod.add_subparser(subparsers)
        except ImportError:
            pass

    # Parse arguments
    args = parser.parse_args()

    # Configure logging based on flags
    setup_logging(args)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Handle dry-run mode
    if args.dry_run:
        logger.info("Dry-run mode enabled")
        logger.info(f"[DRY-RUN] Would execute: wamos {args.command}")
        logger.info(f"[DRY-RUN] Arguments: {vars(args)}")
        sys.exit(0)

    # Call the command's function with error handling
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except (WamosError, FileNotFoundError, ValueError) as e:
        _handle_error(e)
    except Exception as e:
        # Unexpected error - show full traceback for debugging
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
