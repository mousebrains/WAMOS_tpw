"""NOAA SCS raw NMEA log parser (Thompson / R2R format).

The Thompson's Shipboard Computer System (and R2R bags derived from it)
log one NMEA sentence per line with a comma-separated US-format
timestamp prefix::

    05/20/2023,00:00:02.130,$GPGGA,000002.00,0649.936971,N,...
    04/29/2023,00:35:51.014,$HEHDT,306.68,T*14
    06/06/2023,00:00:01.723,$WIMWV,037,R,013,N,A*25

This module parses the streams the radar pipeline needs — gyro heading
(HDT), GNSS position (GGA, with SOG/COG derived from positions), and
relative wind (MWV) — and writes NetCDFs with the filenames and
variable names :class:`~wamos_tpw.instruments.ship_data.ShipData`
expects (``gyro_sperry.nc``, ``gps_abxtwo.nc``, ``wind_bridge.nc``), so
``--ship-data`` works unchanged. The true instrument is recorded in
each file's global attributes. No attitude stream is produced (R2R
archives only the POS MV GGA); ``ShipData`` tolerates the missing file.

CLI::

    revelle scs /path/to/R2R/TN417 -o ./ship_tn417/
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from wamos_tpw.instruments.nmea import KNOTS_TO_MS, parse_latlon, validate_checksum

__all__ = ["parse_scs_line", "parse_scs_directory"]

logger = logging.getLogger(__name__)

GYRO_GLOB = "GYRO-01-HDT-RAW_*"
GPS_GLOB = "CNAV3050-GGA-RAW_*"
WIND_GLOB = "BRIDGE-WIND-STBD_*"

_KN_TO_MS = KNOTS_TO_MS


def parse_scs_line(line: str) -> tuple[np.datetime64, str]:
    """Split an SCS log line into timestamp and NMEA sentence.

    Args:
        line: ``MM/DD/YYYY,HH:MM:SS.sss,$SENTENCE...``

    Returns:
        (timestamp, sentence) tuple.

    Raises:
        ValueError: If the line does not have the SCS prefix layout.
    """
    parts = line.split(",", 2)
    if len(parts) != 3 or not parts[2].startswith("$"):
        raise ValueError(f"Not an SCS line: {line!r}")
    ts = datetime.strptime(f"{parts[0]} {parts[1]}", "%m/%d/%Y %H:%M:%S.%f")
    return np.datetime64(ts, "ns"), parts[2]


def _iter_sentences(files: list[Path], prefix: str):
    """Yield (time, fields) for checksum-valid sentences of one type."""
    for filepath in files:
        with open(filepath, errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ts, sentence = parse_scs_line(line)
                except ValueError:
                    logger.debug("Malformed line %d in %s", line_no, filepath)
                    continue
                if sentence[3:6] != prefix:
                    continue
                if not validate_checksum(sentence):
                    logger.debug("Bad checksum line %d in %s", line_no, filepath)
                    continue
                yield ts, sentence.split("*")[0].split(",")


def _parse_hdt(files: list[Path]) -> dict[str, list]:
    rec: dict[str, list] = {"time": [], "heading": []}
    for ts, f in _iter_sentences(files, "HDT"):
        if len(f) < 3:
            continue
        try:
            rec["heading"].append(float(f[1]))
        except ValueError:
            continue
        rec["time"].append(ts)
    return rec


def _parse_gga(files: list[Path]) -> dict[str, list]:
    rec: dict[str, list] = {"time": [], "latitude": [], "longitude": []}
    for ts, f in _iter_sentences(files, "GGA"):
        # $__GGA,hhmmss,lat,N,lon,E,fix,nsat,hdop,alt,M,...
        if len(f) < 7 or not f[2] or not f[4] or f[6] == "0":
            continue
        try:
            lat = parse_latlon(f[2], f[3])
            lon = parse_latlon(f[4], f[5])
        except (ValueError, IndexError):
            continue
        rec["time"].append(ts)
        rec["latitude"].append(lat)
        rec["longitude"].append(lon)
    return rec


def _parse_mwv(files: list[Path]) -> dict[str, list]:
    rec: dict[str, list] = {"time": [], "relative_wind_direction": [], "relative_wind_speed": []}
    for ts, f in _iter_sentences(files, "MWV"):
        # $WIMWV,dir,R,speed,N,A — keep relative, valid sentences only
        if len(f) < 6 or f[2] != "R" or f[5] != "A":
            continue
        try:
            wd = float(f[1])
            ws = float(f[3]) * (_KN_TO_MS if f[4] == "N" else 1.0)
        except ValueError:
            continue
        rec["time"].append(ts)
        rec["relative_wind_direction"].append(wd)
        rec["relative_wind_speed"].append(ws)
    return rec


def _derive_sog_cog(
    time: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Speed/course over ground from consecutive GGA positions (centered)."""
    t = time.astype("datetime64[ns]").astype(np.int64) / 1e9
    dt = np.gradient(t)
    dt[dt <= 0] = np.nan
    m_per_deg = 111_319.5
    dx = np.gradient(lon) * m_per_deg * np.cos(np.radians(lat))
    dy = np.gradient(lat) * m_per_deg
    sog = np.hypot(dx, dy) / dt
    cog = (np.degrees(np.arctan2(dx, dy))) % 360
    return sog, cog


def parse_scs_directory(
    input_path: Path,
    output_dir: Path,
    gyro_glob: str = GYRO_GLOB,
    gps_glob: str = GPS_GLOB,
    wind_glob: str = WIND_GLOB,
) -> list[Path]:
    """Parse SCS logs under a tree into ShipData-compatible NetCDFs.

    Args:
        input_path: Directory searched recursively for SCS ``.Raw`` logs.
        output_dir: Directory for output NetCDFs.
        gyro_glob: Filename pattern for the heading stream.
        gps_glob: Filename pattern for the position stream.
        wind_glob: Filename pattern for the relative-wind stream.

    Returns:
        List of written NetCDF paths (streams with no files are skipped).
    """
    from wamos_tpw.instruments.netcdf_writer import write_cf_netcdf

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _files(pattern: str) -> list[Path]:
        return sorted(input_path.rglob(pattern))

    gyro_files = _files(gyro_glob)
    if gyro_files:
        rec = _parse_hdt(gyro_files)
        logger.info("Gyro: %d records from %d files", len(rec["time"]), len(gyro_files))
        written.append(
            write_cf_netcdf(
                output_dir / "gyro_sperry.nc",
                np.array(rec["time"], dtype="datetime64[ns]"),
                {
                    "heading": (
                        np.array(rec["heading"], dtype=np.float64),
                        {
                            "standard_name": "platform_azimuth_angle",
                            "long_name": "Gyrocompass heading",
                            "units": "degrees",
                        },
                    )
                },
                {
                    "title": "Gyrocompass heading from SCS HDT logs",
                    "source": f"SCS stream {gyro_glob} (Sperry NAVIGAT 3000)",
                },
            )
        )

    gps_files = _files(gps_glob)
    if gps_files:
        rec = _parse_gga(gps_files)
        logger.info("GPS: %d records from %d files", len(rec["time"]), len(gps_files))
        time = np.array(rec["time"], dtype="datetime64[ns]")
        lat = np.array(rec["latitude"], dtype=np.float64)
        lon = np.array(rec["longitude"], dtype=np.float64)
        sog, cog = _derive_sog_cog(time, lat, lon)
        written.append(
            write_cf_netcdf(
                output_dir / "gps_abxtwo.nc",
                time,
                {
                    "latitude": (
                        lat,
                        {"standard_name": "latitude", "units": "degrees_north"},
                    ),
                    "longitude": (
                        lon,
                        {"standard_name": "longitude", "units": "degrees_east"},
                    ),
                    "sog": (
                        sog,
                        {
                            "standard_name": "platform_speed_wrt_ground",
                            "long_name": "Speed over ground (derived from GGA positions)",
                            "units": "m s-1",
                        },
                    ),
                    "cog": (
                        cog,
                        {
                            "standard_name": "platform_course",
                            "long_name": "Course over ground (derived from GGA positions)",
                            "units": "degrees",
                        },
                    ),
                },
                {
                    "title": "GNSS position from SCS GGA logs",
                    "source": f"SCS stream {gps_glob}",
                    "comment": "sog/cog derived from consecutive GGA positions",
                },
            )
        )

    wind_files = _files(wind_glob)
    if wind_files:
        rec = _parse_mwv(wind_files)
        logger.info("Wind: %d records from %d files", len(rec["time"]), len(wind_files))
        written.append(
            write_cf_netcdf(
                output_dir / "wind_bridge.nc",
                np.array(rec["time"], dtype="datetime64[ns]"),
                {
                    "relative_wind_direction": (
                        np.array(rec["relative_wind_direction"], dtype=np.float64),
                        {
                            "long_name": "Wind direction relative to ship heading",
                            "units": "degrees",
                        },
                    ),
                    "relative_wind_speed": (
                        np.array(rec["relative_wind_speed"], dtype=np.float64),
                        {"long_name": "Relative wind speed", "units": "m s-1"},
                    ),
                },
                {
                    "title": "Relative wind from SCS MWV logs",
                    "source": f"SCS stream {wind_glob}",
                },
            )
        )

    if not written:
        raise FileNotFoundError(f"No SCS streams found under {input_path}")
    return written


def add_subparser(subparsers) -> None:
    """Register the 'scs' subcommand on the revelle CLI."""
    p = subparsers.add_parser(
        "scs",
        help="Parse NOAA SCS raw NMEA logs (Thompson / R2R bags)",
        description=(
            "Parse SCS-format NMEA logs (comma-separated US-date timestamp "
            "prefix) into ShipData-compatible NetCDFs: gyro heading (HDT), "
            "GNSS position with derived SOG/COG (GGA), relative wind (MWV). "
            "The input tree is searched recursively, so an R2R fileset "
            "directory or an SCS NAV directory both work."
        ),
    )
    p.add_argument("input", type=str, help="Directory searched recursively for SCS .Raw logs")
    p.add_argument("--output-dir", "-o", type=str, default=".", help="Output directory")
    p.add_argument(
        "--gyro-glob",
        type=str,
        default=GYRO_GLOB,
        help=f"Heading stream filename pattern (default: {GYRO_GLOB})",
    )
    p.add_argument(
        "--gps-glob",
        type=str,
        default=GPS_GLOB,
        help=f"Position stream filename pattern (default: {GPS_GLOB})",
    )
    p.add_argument(
        "--wind-glob",
        type=str,
        default=WIND_GLOB,
        help=f"Relative-wind stream filename pattern (default: {WIND_GLOB})",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    """Execute the 'scs' subcommand."""
    written = parse_scs_directory(
        Path(args.input),
        Path(args.output_dir),
        gyro_glob=args.gyro_glob,
        gps_glob=args.gps_glob,
        wind_glob=args.wind_glob,
    )
    for p in written:
        logger.info("Wrote %s", p)
