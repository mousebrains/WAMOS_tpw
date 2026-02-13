"""Sperry gyrocompass data parser.

Input files: ``gyro_sperry_rr_heading-YYYY-MM-DD.log`` (~575 KB/day, ~5 Hz)
Each line: ``ISO8601Z $HEHDT,heading,T*XX`` (skip ``$PPLAN`` lines)

Output: ``gyro_sperry.nc`` with heading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wamos_tpw.instruments.nmea import parse_log_timestamp, to_datetime64, validate_checksum

__all__ = ["parse_gyro_file", "parse_gyro_directory"]

logger = logging.getLogger(__name__)

DEFAULT_GLOB = "gyro_sperry_rr_heading-*"
OUTPUT_FILENAME = "gyro_sperry.nc"


def parse_gyro_file(filepath: Path) -> dict[str, list]:
    """Parse a single gyro log file.

    Returns:
        Dict with keys: time, heading.
    """
    records: dict[str, list] = {"time": [], "heading": []}

    with open(filepath) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ts, sentence = parse_log_timestamp(line)
            except ValueError:
                logger.debug("Skipping malformed line %d in %s", line_no, filepath)
                continue

            # Skip non-HEHDT sentences (e.g. $PPLAN)
            if not sentence.startswith("$HEHDT"):
                continue

            if not validate_checksum(sentence):
                logger.debug("Bad checksum at line %d in %s", line_no, filepath)
                continue

            # $HEHDT,heading,T*XX
            body = sentence.split("*")[0]
            fields = body.split(",")
            if len(fields) < 3:
                logger.debug("Short HEHDT at line %d in %s", line_no, filepath)
                continue

            try:
                heading = float(fields[1])
            except ValueError:
                logger.debug("Bad heading value at line %d in %s", line_no, filepath)
                continue

            records["time"].append(to_datetime64(ts))
            records["heading"].append(heading)

    logger.info("Parsed %d records from %s", len(records["time"]), filepath)
    return records


def parse_gyro_directory(
    input_path: Path,
    output_dir: Path,
    glob_pattern: str = DEFAULT_GLOB,
) -> Path:
    """Parse all gyro files in a directory and write a single NetCDF.

    Args:
        input_path: Directory containing gyro log files, or a single file.
        output_dir: Directory for the output NetCDF file.
        glob_pattern: Glob pattern for matching gyro files.

    Returns:
        Path to the output NetCDF file.
    """
    from wamos_tpw.instruments.netcdf_writer import write_cf_netcdf

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(
                f"No gyro files matching '{glob_pattern}' in {input_path}"
            )

    all_records: dict[str, list] = {"time": [], "heading": []}

    for f in files:
        records = parse_gyro_file(f)
        for key in all_records:
            all_records[key].extend(records[key])

    time = np.array(all_records["time"], dtype="datetime64[ns]")
    variables = {
        "heading": (
            np.array(all_records["heading"], dtype=np.float64),
            {
                "standard_name": "platform_azimuth_angle",
                "long_name": "Gyrocompass heading",
                "units": "degrees",
            },
        ),
    }

    global_attrs = {
        "title": "R/V Roger Revelle Sperry gyrocompass data",
        "source": "Sperry Marine gyrocompass",
    }

    output_path = Path(output_dir) / OUTPUT_FILENAME
    return write_cf_netcdf(output_path, time, variables, global_attrs)


# --- CLI integration ---


def _add_arguments(parser) -> None:
    parser.add_argument("input", type=str, help="Log file or directory of gyro files")
    parser.add_argument(
        "--output-dir", "-o", type=str, default=".", help="Output directory"
    )
    parser.add_argument(
        "--glob", "-g", type=str, default=DEFAULT_GLOB,
        help="Glob pattern for file matching in directory mode",
    )


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "gyro",
        help="Parse gyrocompass data to NetCDF",
        description="Parse Sperry gyrocompass log files to CF-1.13 NetCDF.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result = parse_gyro_directory(input_path, output_dir, glob_pattern=args.glob)
    logger.info("Output: %s", result)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Parse gyrocompass data to NetCDF")

if __name__ == "__main__":
    main()
