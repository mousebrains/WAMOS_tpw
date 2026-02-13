"""R/V Revelle MET system data parser.

Input files: ``YYMMDD.MET`` in ``.../met/data/`` (~5-67 MB/day, 1 Hz)

Format: 4 header lines (# comments), then space-delimited columns.
Time column is HHMMSS (seconds since midnight). Date from header line 2.
Missing values are ``-99.0`` / ``-99.00``.

Output: ``met_revelle.nc`` with wind, navigation, and selected variables.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wamos_tpw.instruments.nmea import KNOTS_TO_MS

__all__ = ["parse_met_file", "parse_met_directory"]

logger = logging.getLogger(__name__)

DEFAULT_GLOB = "*.MET"
OUTPUT_FILENAME = "met_revelle.nc"

# Columns to extract and their output variable mapping
# (column_name, var_name, attrs, convert_knots)
_COLUMN_MAP = [
    (
        "WS",
        "wind_speed",
        {"standard_name": "wind_speed", "long_name": "Wind speed", "units": "m s-1"},
        True,
    ),
    (
        "WD",
        "wind_direction",
        {"standard_name": "wind_from_direction", "long_name": "Wind direction", "units": "degrees"},
        False,
    ),
    ("WS-2", "wind_speed_2", {"long_name": "Wind speed (sensor 2)", "units": "m s-1"}, True),
    (
        "WD-2",
        "wind_direction_2",
        {"long_name": "Wind direction (sensor 2)", "units": "degrees"},
        False,
    ),
    ("TW", "true_wind_speed", {"long_name": "True wind speed", "units": "m s-1"}, True),
    ("TI", "true_wind_index", {"long_name": "True wind quality index", "units": "1"}, False),
    (
        "LA",
        "latitude",
        {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"},
        False,
    ),
    (
        "LO",
        "longitude",
        {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"},
        False,
    ),
    (
        "GY",
        "heading",
        {
            "standard_name": "platform_azimuth_angle",
            "long_name": "Gyro heading",
            "units": "degrees",
        },
        False,
    ),
    (
        "CR",
        "course",
        {"standard_name": "platform_course", "long_name": "Course", "units": "degrees"},
        False,
    ),
    (
        "SP",
        "speed",
        {
            "standard_name": "platform_speed_wrt_ground",
            "long_name": "Speed over ground",
            "units": "m s-1",
        },
        True,
    ),
]

_MISSING_VALUES = {-99.0, -99.00}


def _parse_met_date(header_line: str) -> datetime:
    """Parse the date from MET header line 2.

    Format: ``# Sat 02-Apr-22  00:00:00``
    """
    # Strip leading '# ' and day-of-week
    parts = header_line.lstrip("#").strip().split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot parse MET date from: {header_line!r}")
    date_time_str = parts[1].strip()
    # Parse: 02-Apr-22  00:00:00
    return datetime.strptime(date_time_str, "%d-%b-%y  %H:%M:%S").replace(tzinfo=timezone.utc)


def _hhmmss_to_datetime64(time_val: int, base_date: datetime) -> np.datetime64:
    """Convert HHMMSS integer to datetime64[ns]."""
    hours = time_val // 10000
    minutes = (time_val % 10000) // 100
    seconds = time_val % 100
    dt = base_date.replace(hour=hours, minute=minutes, second=seconds, tzinfo=None)
    return np.datetime64(dt.isoformat(), "ns")


def parse_met_file(filepath: Path) -> dict[str, list]:
    """Parse a single MET data file.

    Returns:
        Dict with keys: time, plus all variable names from _COLUMN_MAP.
    """
    var_names = [entry[1] for entry in _COLUMN_MAP]
    records: dict[str, list] = {"time": []}
    for name in var_names:
        records[name] = []

    with open(filepath) as f:
        lines = f.readlines()

    if len(lines) < 5:
        logger.warning("MET file too short: %s", filepath)
        return records

    # Line 2 (index 1): date
    try:
        base_date = _parse_met_date(lines[1])
    except ValueError:
        logger.error("Cannot parse date from %s: %s", filepath, lines[1].strip())
        return records

    # Line 4 (index 3): column headers
    header_line = lines[3].lstrip("#").strip()
    col_names = header_line.split()

    # Build column index mapping
    col_indices: dict[str, int] = {}
    for col_name, var_name, _, _ in _COLUMN_MAP:
        try:
            col_indices[var_name] = col_names.index(col_name)
        except ValueError:
            logger.warning("Column '%s' not found in %s", col_name, filepath)

    time_idx = col_names.index("Time") if "Time" in col_names else 0

    # Parse data lines
    for line_no, line in enumerate(lines[4:], 5):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) < len(col_names):
            logger.debug("Short line %d in %s", line_no, filepath)
            continue

        try:
            time_val = int(fields[time_idx])
            ts = _hhmmss_to_datetime64(time_val, base_date)
        except (ValueError, IndexError):
            logger.debug("Bad time at line %d in %s", line_no, filepath)
            continue

        records["time"].append(ts)

        for col_name, var_name, _, convert_knots in _COLUMN_MAP:
            idx = col_indices.get(var_name)
            if idx is None or idx >= len(fields):
                records[var_name].append(np.nan)
                continue
            try:
                val = float(fields[idx])
                if val in _MISSING_VALUES:
                    val = np.nan
                elif convert_knots:
                    val *= KNOTS_TO_MS
                records[var_name].append(val)
            except ValueError:
                records[var_name].append(np.nan)

    logger.info("Parsed %d records from %s", len(records["time"]), filepath)
    return records


def parse_met_directory(
    input_path: Path,
    output_dir: Path,
    glob_pattern: str = DEFAULT_GLOB,
) -> Path:
    """Parse all MET files in a directory and write a single NetCDF.

    Args:
        input_path: Directory containing MET data files, or a single file.
        output_dir: Directory for the output NetCDF file.
        glob_pattern: Glob pattern for matching MET files.

    Returns:
        Path to the output NetCDF file.
    """
    from wamos_tpw.instruments.netcdf_writer import write_cf_netcdf

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(f"No MET files matching '{glob_pattern}' in {input_path}")

    var_names = [entry[1] for entry in _COLUMN_MAP]
    all_records: dict[str, list] = {"time": []}
    for name in var_names:
        all_records[name] = []

    for f in files:
        records = parse_met_file(f)
        for key in all_records:
            all_records[key].extend(records[key])

    time = np.array(all_records["time"], dtype="datetime64[ns]")

    # Build variables dict from _COLUMN_MAP
    variables = {}
    for _, var_name, attrs, _ in _COLUMN_MAP:
        variables[var_name] = (
            np.array(all_records[var_name], dtype=np.float64),
            dict(attrs),
        )

    global_attrs = {
        "title": "R/V Roger Revelle MET system data",
        "source": "R/V Roger Revelle shipboard MET system",
    }

    output_path = Path(output_dir) / OUTPUT_FILENAME
    return write_cf_netcdf(output_path, time, variables, global_attrs)


# --- CLI integration ---


def _add_arguments(parser) -> None:
    parser.add_argument("input", type=str, help="MET file or directory of MET files")
    parser.add_argument("--output-dir", "-o", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        default=DEFAULT_GLOB,
        help="Glob pattern for file matching in directory mode",
    )


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "met",
        help="Parse MET system data to NetCDF",
        description="Parse R/V Revelle MET system data files to CF-1.13 NetCDF.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result = parse_met_directory(input_path, output_dir, glob_pattern=args.glob)
    logger.info("Output: %s", result)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Parse MET data to NetCDF")

if __name__ == "__main__":
    main()
