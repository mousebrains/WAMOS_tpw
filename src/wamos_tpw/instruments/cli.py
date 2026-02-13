#!/usr/bin/env python3
"""
R/V Roger Revelle ship instrument data CLI.

Parses ship instrument log files into CF-1.13 compliant NetCDF files.
Run this before the wamos radar processing pipeline to generate the
instrument NetCDF files needed for interpolation onto radar beams.
"""

import argparse
import logging
import sys

from wamos_tpw.logging_config import add_logging_arguments, setup_logging


logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the revelle CLI."""
    parser = argparse.ArgumentParser(
        prog="revelle",
        description="R/V Roger Revelle ship instrument data processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run this tool first to generate CF-1.13 NetCDF files from ship instrument
log files. The output NetCDF files are then used by the wamos radar
processing pipeline for interpolation onto radar beams.

Examples:
  revelle gps /path/to/serialinstruments/ -o ./output/
  revelle gyro /path/to/serialinstruments/ -o ./output/
  revelle mru /path/to/serialinstruments/ -o ./output/
  revelle wind /path/to/serialinstruments/ -o ./output/
  revelle met /path/to/met/data/ -o ./output/
  revelle all /path/to/cruise/data/ -o ./output/
""",
    )

    add_logging_arguments(parser)

    subparsers = parser.add_subparsers(
        dest="command",
        title="instruments",
        description="Available instrument parsers",
        metavar="INSTRUMENT",
    )

    # Register subcommands from each instrument module
    from wamos_tpw.instruments import gps, gyro, met, mru, wind

    gps.add_subparser(subparsers)
    gyro.add_subparser(subparsers)
    mru.add_subparser(subparsers)
    wind.add_subparser(subparsers)
    met.add_subparser(subparsers)

    # "all" subcommand to process every instrument at once
    p_all = subparsers.add_parser(
        "all",
        help="Process all instruments at once",
        description=(
            "Parse all ship instrument data from a cruise data directory. "
            "Expects serialinstruments/ and met/data/ subdirectories."
        ),
    )
    p_all.add_argument(
        "input",
        type=str,
        help="Cruise data directory (containing serialinstruments/ and met/data/)",
    )
    p_all.add_argument(
        "--output-dir", "-o", type=str, default=".", help="Output directory"
    )
    p_all.set_defaults(func=_run_all)

    # Parse arguments
    args = parser.parse_args()

    # Configure logging based on flags
    setup_logging(args)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Call the command's function with error handling
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


def _run_all(args) -> None:
    """Process all instruments from a cruise data directory."""
    from pathlib import Path

    from wamos_tpw.instruments.gps import parse_gps_directory
    from wamos_tpw.instruments.gyro import parse_gyro_directory
    from wamos_tpw.instruments.met import parse_met_directory
    from wamos_tpw.instruments.mru import parse_mru_directory
    from wamos_tpw.instruments.wind import parse_wind_directory

    cruise_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    serial_dir = cruise_dir / "serialinstruments"
    met_dir = cruise_dir / "met" / "data"

    if not serial_dir.is_dir():
        raise FileNotFoundError(f"Serial instruments directory not found: {serial_dir}")
    if not met_dir.is_dir():
        raise FileNotFoundError(f"MET data directory not found: {met_dir}")

    parsers = [
        ("GPS", lambda: parse_gps_directory(serial_dir, output_dir)),
        ("Gyro", lambda: parse_gyro_directory(serial_dir, output_dir)),
        ("MRU", lambda: parse_mru_directory(serial_dir, output_dir)),
        ("Wind", lambda: parse_wind_directory(serial_dir, output_dir)),
        ("MET", lambda: parse_met_directory(met_dir, output_dir)),
    ]

    for name, parse_fn in parsers:
        logger.info("Processing %s...", name)
        try:
            result = parse_fn()
            logger.info("%s -> %s", name, result)
        except Exception:
            logger.exception("Failed to process %s", name)


if __name__ == "__main__":
    main()
