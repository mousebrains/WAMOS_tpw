"""Wind bridge (RM Young) data parser.

Input files: ``wind_bridge_rr-YYYY-MM-DD.log`` (~86 KB/day, 1 Hz)
Each line: ``ISO8601Z $WIMWV,angle,R,speed,N,A``

Output: ``wind_bridge.nc`` with relative_wind_direction and relative_wind_speed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wamos_tpw.instruments.nmea import KNOTS_TO_MS, parse_log_timestamp, to_datetime64

__all__ = ["parse_wind_file", "parse_wind_directory"]

logger = logging.getLogger(__name__)

DEFAULT_GLOB = "wind_bridge_rr-*"
OUTPUT_FILENAME = "wind_bridge.nc"


def parse_wind_file(filepath: Path) -> dict[str, list]:
    """Parse a single wind bridge log file.

    Returns:
        Dict with keys: time, relative_wind_direction, relative_wind_speed.
    """
    records: dict[str, list] = {
        "time": [],
        "relative_wind_direction": [],
        "relative_wind_speed": [],
    }

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

            # $WIMWV,angle,R,speed,N,A
            if not sentence.startswith("$WIMWV"):
                continue

            # Strip checksum if present
            body = sentence.split("*")[0]
            fields = body.split(",")
            if len(fields) < 6:
                logger.debug("Short WIMWV at line %d in %s", line_no, filepath)
                continue

            # fields: $WIMWV, angle, R, speed, N, A
            status = fields[5]
            if status != "A":
                continue

            try:
                angle = float(fields[1])
                speed_knots = float(fields[3])
            except ValueError:
                logger.debug("Bad numeric values at line %d in %s", line_no, filepath)
                continue

            records["time"].append(to_datetime64(ts))
            records["relative_wind_direction"].append(angle)
            records["relative_wind_speed"].append(speed_knots * KNOTS_TO_MS)

    logger.info("Parsed %d records from %s", len(records["time"]), filepath)
    return records


def parse_wind_directory(
    input_path: Path,
    output_dir: Path,
    glob_pattern: str = DEFAULT_GLOB,
) -> Path:
    """Parse all wind files in a directory and write a single NetCDF.

    Args:
        input_path: Directory containing wind log files, or a single file.
        output_dir: Directory for the output NetCDF file.
        glob_pattern: Glob pattern for matching wind files.

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
                f"No wind files matching '{glob_pattern}' in {input_path}"
            )

    all_records: dict[str, list] = {
        "time": [],
        "relative_wind_direction": [],
        "relative_wind_speed": [],
    }

    for f in files:
        records = parse_wind_file(f)
        for key in all_records:
            all_records[key].extend(records[key])

    time = np.array(all_records["time"], dtype="datetime64[ns]")
    variables = {
        "relative_wind_direction": (
            np.array(all_records["relative_wind_direction"], dtype=np.float64),
            {
                "long_name": "Wind direction relative to ship heading",
                "units": "degrees",
            },
        ),
        "relative_wind_speed": (
            np.array(all_records["relative_wind_speed"], dtype=np.float64),
            {
                "standard_name": "wind_speed",
                "long_name": "Relative wind speed",
                "units": "m s-1",
            },
        ),
    }

    global_attrs = {
        "title": "R/V Roger Revelle wind bridge (RM Young) data",
        "source": "RM Young wind sensor, bridge installation",
    }

    output_path = Path(output_dir) / OUTPUT_FILENAME
    return write_cf_netcdf(output_path, time, variables, global_attrs)


# --- CLI integration ---


def _add_arguments(parser) -> None:
    parser.add_argument("input", type=str, help="Log file or directory of wind files")
    parser.add_argument(
        "--output-dir", "-o", type=str, default=".", help="Output directory"
    )
    parser.add_argument(
        "--glob", "-g", type=str, default=DEFAULT_GLOB,
        help="Glob pattern for file matching in directory mode",
    )


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "wind",
        help="Parse wind bridge data to NetCDF",
        description="Parse RM Young wind bridge log files to CF-1.13 NetCDF.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result = parse_wind_directory(input_path, output_dir, glob_pattern=args.glob)
    logger.info("Output: %s", result)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Parse wind bridge data to NetCDF")

if __name__ == "__main__":
    main()
