"""PHINS-III MRU (Motion Reference Unit) data parser.

Input files: ``mru_phinsiii_rr_navbho-YYYY-MM-DD.log`` (~1.2 MB/day, ~2 Hz)

Multi-line grouped format: only the first line of each group has a timestamp
prefix. Subsequent lines lack timestamps. A new group starts when a line has
a timestamp prefix.

Extracts from: PHGGA (position), PHVTG (speed/course), HEHDT (heading),
PASHR (heading, roll, pitch, heave).

Output: ``mru_phins.nc``
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

__all__ = ["parse_mru_file", "parse_mru_directory"]

logger = logging.getLogger(__name__)

DEFAULT_GLOB = "mru_phinsiii_rr_navbho-*"
OUTPUT_FILENAME = "mru_phins.nc"

_VAR_NAMES = [
    "latitude",
    "longitude",
    "altitude",
    "heading",
    "roll",
    "pitch",
    "heave",
    "sog",
    "cog",
]


def _new_record() -> dict:
    return {k: np.nan for k in _VAR_NAMES}


def _has_timestamp(line: str) -> bool:
    """Check if a line starts with an ISO 8601 timestamp."""
    # Timestamps look like: 2022-04-02T00:00:00.005222Z
    return len(line) > 24 and line[4] == "-" and line[10] == "T"


def _parse_phgga(fields: list[str], rec: dict) -> None:
    """Extract position and altitude from PHGGA."""
    # $PHGGA,time,lat,N/S,lon,E/W,quality,nsat,hdop,alt,M,geoid,M,age,stn
    if len(fields) < 10:
        return
    try:
        if fields[2] and fields[4]:
            rec["latitude"] = parse_latlon(fields[2], fields[3])
            rec["longitude"] = parse_latlon(fields[4], fields[5])
        if fields[9]:
            rec["altitude"] = float(fields[9])
    except (ValueError, IndexError):
        pass


def _parse_phvtg(fields: list[str], rec: dict) -> None:
    """Extract COG and SOG from PHVTG."""
    # $PHVTG,cog_true,T,cog_mag,M,sog_knots,N,sog_kph,K,mode
    if len(fields) < 6:
        return
    try:
        if fields[1]:
            rec["cog"] = float(fields[1])
        if fields[5]:
            rec["sog"] = float(fields[5]) * KNOTS_TO_MS
    except (ValueError, IndexError):
        pass


def _parse_hehdt(fields: list[str], rec: dict) -> None:
    """Extract true heading from HEHDT."""
    # $HEHDT,heading,T
    if len(fields) < 3:
        return
    try:
        if fields[1]:
            rec["heading"] = float(fields[1])
    except (ValueError, IndexError):
        pass


def _parse_pashr(fields: list[str], rec: dict) -> None:
    """Extract heading, roll, pitch, heave from PASHR.

    MRU format: $PASHR,time,heading,T,roll,pitch,heave,...
    """
    if len(fields) < 7:
        return
    try:
        if fields[2]:
            rec["heading"] = float(fields[2])
        if fields[4]:
            rec["roll"] = float(fields[4])
        if fields[5]:
            rec["pitch"] = float(fields[5])
        if fields[6]:
            rec["heave"] = float(fields[6])
    except (ValueError, IndexError):
        pass


def _process_group(ts, sentences: list[str], records: dict[str, list]) -> None:
    """Process a group of MRU sentences into one record."""
    if ts is None or not sentences:
        return

    rec = _new_record()

    for sentence in sentences:
        if not validate_checksum(sentence):
            continue
        body = sentence.split("*")[0]
        fields = body.split(",")
        tag = fields[0]

        if tag == "$PHGGA":
            _parse_phgga(fields, rec)
        elif tag == "$PHVTG":
            _parse_phvtg(fields, rec)
        elif tag == "$HEHDT":
            _parse_hehdt(fields, rec)
        elif tag == "$PASHR":
            _parse_pashr(fields, rec)

    # Only emit if we have at least heading
    if np.isnan(rec["heading"]):
        return

    records["time"].append(to_datetime64(ts))
    for key in _VAR_NAMES:
        records[key].append(rec[key])


def parse_mru_file(filepath: Path) -> dict[str, list]:
    """Parse a single MRU log file.

    Groups lines by timestamp: a new group starts when a line has a timestamp
    prefix. Lines without timestamps are appended to the current group.

    Returns:
        Dict with keys: time, latitude, longitude, altitude, heading, roll,
        pitch, heave, sog, cog.
    """
    records: dict[str, list] = {"time": []}
    for key in _VAR_NAMES:
        records[key] = []

    current_ts = None
    current_sentences: list[str] = []

    with open(filepath) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if _has_timestamp(line):
                # Flush previous group
                _process_group(current_ts, current_sentences, records)

                # Start new group
                try:
                    current_ts, sentence = parse_log_timestamp(line)
                except ValueError:
                    logger.debug("Bad timestamp at line %d in %s", line_no, filepath)
                    current_ts = None
                    current_sentences = []
                    continue
                current_sentences = [sentence]
            else:
                # Continuation line (no timestamp) — add to current group
                if line.startswith("$"):
                    current_sentences.append(line)

    # Flush final group
    _process_group(current_ts, current_sentences, records)

    logger.info("Parsed %d records from %s", len(records["time"]), filepath)
    return records


def parse_mru_directory(
    input_path: Path,
    output_dir: Path,
    glob_pattern: str = DEFAULT_GLOB,
) -> Path:
    """Parse all MRU files in a directory and write a single NetCDF.

    Args:
        input_path: Directory containing MRU log files, or a single file.
        output_dir: Directory for the output NetCDF file.
        glob_pattern: Glob pattern for matching MRU files.

    Returns:
        Path to the output NetCDF file.
    """
    from wamos_tpw.instruments.netcdf_writer import write_cf_netcdf

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(f"No MRU files matching '{glob_pattern}' in {input_path}")

    all_records: dict[str, list] = {"time": []}
    for key in _VAR_NAMES:
        all_records[key] = []

    for f in files:
        records = parse_mru_file(f)
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
                "long_name": "PHINS heading",
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
        "heave": (
            np.array(all_records["heave"], dtype=np.float64),
            {"long_name": "Heave", "units": "m"},
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
    }

    global_attrs = {
        "title": "R/V Roger Revelle PHINS-III MRU data",
        "source": "iXBlue PHINS-III inertial navigation system",
    }

    output_path = Path(output_dir) / OUTPUT_FILENAME
    return write_cf_netcdf(output_path, time, variables, global_attrs)


# --- CLI integration ---


def _add_arguments(parser) -> None:
    parser.add_argument("input", type=str, help="Log file or directory of MRU files")
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
        "mru",
        help="Parse MRU data to NetCDF",
        description="Parse PHINS-III MRU log files to CF-1.13 NetCDF.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    result = parse_mru_directory(input_path, output_dir, glob_pattern=args.glob)
    logger.info("Output: %s", result)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Parse MRU data to NetCDF")

if __name__ == "__main__":
    main()
