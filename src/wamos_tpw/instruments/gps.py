"""GPS ABX-Two dual-antenna data parser.

Input files: ``gps_abxtwo_rr_rx2_navbho-YYYY-MM-DD.log`` (~1.55 GB/day, ~2 Hz)

Sentences are grouped by timestamp proximity (<250 ms gap). Each group may
contain: GPGGA, GPRMC, GPVTG, GNGST, GPHDT, PASHR,ATT.

Output: ``gps_abxtwo.nc``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wamos_tpw.instruments.nmea import (
    KNOTS_TO_MS,
    parse_latlon,
    parse_log_timestamp,
    to_datetime64,
    validate_checksum,
)

__all__ = ["parse_gps_file", "parse_gps_directory"]

logger = logging.getLogger(__name__)

DEFAULT_GLOB = "gps_abxtwo_rr_rx2_navbho-*"
OUTPUT_FILENAME = "gps_abxtwo.nc"

# Maximum time gap (seconds) between lines in the same epoch group
_GROUP_GAP_S = 0.250

# Variable names in output order
_VAR_NAMES = [
    "latitude",
    "longitude",
    "altitude",
    "heading",
    "roll",
    "pitch",
    "sog",
    "cog",
    "n_satellites",
    "hdop",
    "lat_error",
    "lon_error",
    "alt_error",
    "fix_quality",
]


def _new_record() -> dict:
    return {k: np.nan for k in _VAR_NAMES}


def _parse_gpgga(fields: list[str], rec: dict) -> None:
    """Extract position, altitude, fix quality from GPGGA."""
    # $GPGGA,time,lat,N/S,lon,E/W,quality,nsat,hdop,alt,M,geoid,M,,
    if len(fields) < 15:
        return
    try:
        if fields[2] and fields[4]:
            rec["latitude"] = parse_latlon(fields[2], fields[3])
            rec["longitude"] = parse_latlon(fields[4], fields[5])
        if fields[6]:
            rec["fix_quality"] = float(fields[6])
        if fields[7]:
            rec["n_satellites"] = float(fields[7])
        if fields[8]:
            rec["hdop"] = float(fields[8])
        if fields[9]:
            rec["altitude"] = float(fields[9])
    except (ValueError, IndexError):
        pass


def _parse_gprmc(fields: list[str], rec: dict) -> None:
    """Extract SOG, COG from GPRMC."""
    # $GPRMC,time,status,lat,N/S,lon,E/W,sog_knots,cog,date,magvar,E/W,mode
    if len(fields) < 10:
        return
    try:
        if fields[7]:
            rec["sog"] = float(fields[7]) * KNOTS_TO_MS
        if fields[8]:
            rec["cog"] = float(fields[8])
    except (ValueError, IndexError):
        pass


def _parse_gngst(fields: list[str], rec: dict) -> None:
    """Extract position error estimates from GNGST."""
    # $GNGST,time,rms,smj,smn,orient,lat_err,lon_err,alt_err
    if len(fields) < 9:
        return
    try:
        if fields[6]:
            rec["lat_error"] = float(fields[6])
        if fields[7]:
            rec["lon_error"] = float(fields[7])
        if fields[8]:
            rec["alt_error"] = float(fields[8])
    except (ValueError, IndexError):
        pass


def _parse_gphdt(fields: list[str], rec: dict) -> None:
    """Extract true heading from GPHDT."""
    # $GPHDT,heading,T
    if len(fields) < 3:
        return
    try:
        if fields[1]:
            rec["heading"] = float(fields[1])
    except (ValueError, IndexError):
        pass


def _parse_pashr_att(fields: list[str], rec: dict) -> None:
    """Extract heading, roll, pitch from PASHR,ATT."""
    # $PASHR,ATT,time,heading,roll,pitch,heading_rms,roll_rms,flag
    if len(fields) < 6:
        return
    try:
        if fields[3]:
            rec["heading"] = float(fields[3])
        if fields[4]:
            rec["roll"] = float(fields[4])
        if fields[5]:
            rec["pitch"] = float(fields[5])
    except (ValueError, IndexError):
        pass


def _process_group(lines: list[tuple], records: dict[str, list]) -> None:
    """Process a group of timestamped NMEA sentences into one record."""
    if not lines:
        return

    # Use the earliest timestamp in the group
    ts = lines[0][0]
    rec = _new_record()

    for _, sentence in lines:
        if not validate_checksum(sentence):
            continue
        body = sentence.split("*")[0]
        fields = body.split(",")
        tag = fields[0]

        if tag == "$GPGGA":
            _parse_gpgga(fields, rec)
        elif tag == "$GPRMC":
            _parse_gprmc(fields, rec)
        elif tag == "$GNGST":
            _parse_gngst(fields, rec)
        elif tag == "$GPHDT":
            _parse_gphdt(fields, rec)
        elif tag == "$PASHR" and len(fields) > 1 and fields[1] == "ATT":
            _parse_pashr_att(fields, rec)

    # Only emit record if we got at least latitude
    if np.isnan(rec["latitude"]):
        return

    records["time"].append(to_datetime64(ts))
    for key in _VAR_NAMES:
        records[key].append(rec[key])


def parse_gps_file(filepath: Path) -> dict[str, list]:
    """Parse a single GPS log file.

    Lines are grouped by timestamp proximity (<250 ms gap). Within each group,
    values are extracted from all available sentence types.

    Returns:
        Dict with keys: time, latitude, longitude, altitude, heading, roll,
        pitch, sog, cog, n_satellites, hdop, lat_error, lon_error, alt_error,
        fix_quality.
    """
    records: dict[str, list] = {"time": []}
    for key in _VAR_NAMES:
        records[key] = []

    group: list[tuple] = []
    prev_ts = None

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

            # Check if this line starts a new group
            if prev_ts is not None:
                gap = (ts - prev_ts).total_seconds()
                if gap > _GROUP_GAP_S:
                    _process_group(group, records)
                    group = []

            group.append((ts, sentence))
            prev_ts = ts

    # Process final group
    _process_group(group, records)

    logger.info("Parsed %d records from %s", len(records["time"]), filepath)
    return records


def parse_gps_directory(
    input_path: Path,
    output_dir: Path,
    glob_pattern: str = DEFAULT_GLOB,
) -> Path:
    """Parse all GPS files in a directory and write a single NetCDF.

    Args:
        input_path: Directory containing GPS log files, or a single file.
        output_dir: Directory for the output NetCDF file.
        glob_pattern: Glob pattern for matching GPS files.

    Returns:
        Path to the output NetCDF file.
    """
    from wamos_tpw.instruments.netcdf_writer import write_cf_netcdf

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(f"No GPS files matching '{glob_pattern}' in {input_path}")

    all_records: dict[str, list] = {"time": []}
    for key in _VAR_NAMES:
        all_records[key] = []

    for f in files:
        records = parse_gps_file(f)
        for key in all_records:
            all_records[key].extend(records[key])

    time = np.array(all_records["time"], dtype="datetime64[ns]")

    variables = {
        "latitude": (
            np.array(all_records["latitude"], dtype=np.float64),
            {"standard_name": "latitude", "long_name": "Latitude", "units": "degrees_north"},
        ),
        "longitude": (
            np.array(all_records["longitude"], dtype=np.float64),
            {"standard_name": "longitude", "long_name": "Longitude", "units": "degrees_east"},
        ),
        "altitude": (
            np.array(all_records["altitude"], dtype=np.float64),
            {"standard_name": "altitude", "long_name": "Altitude above geoid", "units": "m"},
        ),
        "heading": (
            np.array(all_records["heading"], dtype=np.float64),
            {
                "standard_name": "platform_azimuth_angle",
                "long_name": "True heading",
                "units": "degrees",
            },
        ),
        "roll": (
            np.array(all_records["roll"], dtype=np.float64),
            {"standard_name": "platform_roll_angle", "long_name": "Roll", "units": "degrees"},
        ),
        "pitch": (
            np.array(all_records["pitch"], dtype=np.float64),
            {"standard_name": "platform_pitch_angle", "long_name": "Pitch", "units": "degrees"},
        ),
        "sog": (
            np.array(all_records["sog"], dtype=np.float64),
            {
                "standard_name": "platform_speed_wrt_ground",
                "long_name": "Speed over ground",
                "units": "m s-1",
            },
        ),
        "cog": (
            np.array(all_records["cog"], dtype=np.float64),
            {
                "standard_name": "platform_course",
                "long_name": "Course over ground",
                "units": "degrees",
            },
        ),
        "n_satellites": (
            np.array(all_records["n_satellites"], dtype=np.float64),
            {"long_name": "Number of satellites", "units": "1"},
        ),
        "hdop": (
            np.array(all_records["hdop"], dtype=np.float64),
            {"long_name": "Horizontal dilution of precision", "units": "1"},
        ),
        "lat_error": (
            np.array(all_records["lat_error"], dtype=np.float64),
            {"long_name": "Latitude error (1-sigma)", "units": "m"},
        ),
        "lon_error": (
            np.array(all_records["lon_error"], dtype=np.float64),
            {"long_name": "Longitude error (1-sigma)", "units": "m"},
        ),
        "alt_error": (
            np.array(all_records["alt_error"], dtype=np.float64),
            {"long_name": "Altitude error (1-sigma)", "units": "m"},
        ),
        "fix_quality": (
            np.array(all_records["fix_quality"], dtype=np.float64),
            {"long_name": "GPS fix quality indicator", "units": "1"},
        ),
    }

    global_attrs = {
        "title": "R/V Roger Revelle GPS ABX-Two dual-antenna data",
        "source": "Trimble ABX-Two dual-antenna GPS, rx2, navbho",
    }

    output_path = Path(output_dir) / OUTPUT_FILENAME
    return write_cf_netcdf(output_path, time, variables, global_attrs)


# --- CLI integration ---


def _add_arguments(parser) -> None:
    parser.add_argument("input", type=str, help="Log file or directory of GPS files")
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
        "gps",
        help="Parse GPS data to NetCDF",
        description="Parse GPS ABX-Two log files to CF-1.13 NetCDF.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result = parse_gps_directory(input_path, output_dir, glob_pattern=args.glob)
    logger.info("Output: %s", result)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Parse GPS data to NetCDF")

if __name__ == "__main__":
    main()
