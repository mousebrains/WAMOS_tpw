"""Shared NMEA parsing utilities for ship instrument log files.

Each log file line has the format:
    ISO8601Z $SENTENCE,...*XX

where the ISO timestamp is followed by a space and an NMEA sentence.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "parse_log_timestamp",
    "validate_checksum",
    "parse_latlon",
    "parse_time_hhmmss",
    "to_datetime64",
    "KNOTS_TO_MS",
]

KNOTS_TO_MS = 0.514444


def to_datetime64(dt: datetime) -> np.datetime64:
    """Convert a datetime to numpy datetime64[ns], stripping timezone info.

    numpy datetime64 does not support timezone-aware datetimes. This helper
    strips the tzinfo (assumed UTC) before conversion to avoid warnings.
    """
    import numpy

    return numpy.datetime64(dt.replace(tzinfo=None).isoformat(), "ns")


def parse_log_timestamp(line: str) -> tuple[datetime, str]:
    """Split a log line into (datetime, NMEA sentence).

    Expected format: ``2022-04-02T00:00:00.003358Z $GPGGA,...``

    Returns:
        Tuple of (UTC datetime with microseconds, raw sentence string).

    Raises:
        ValueError: If the line cannot be split into timestamp + sentence.
    """
    parts = line.split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Cannot split line into timestamp and sentence: {line!r}")
    ts_str, sentence = parts
    # Parse ISO 8601 timestamp with fractional seconds
    # Format: 2022-04-02T00:00:00.003358Z
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt, sentence.strip()


def validate_checksum(sentence: str) -> bool:
    """Validate NMEA checksum.

    The checksum is the XOR of all characters between ``$`` and ``*``.

    Args:
        sentence: NMEA sentence starting with ``$`` and ending with ``*XX``.

    Returns:
        True if the checksum is valid, or if no checksum is present.
    """
    if "*" not in sentence:
        return True
    # Strip any leading $ or !
    body = sentence.lstrip("$!")
    payload, checksum_str = body.rsplit("*", 1)
    try:
        expected = int(checksum_str, 16)
    except ValueError:
        return False
    computed = 0
    for ch in payload:
        computed ^= ord(ch)
    return computed == expected


def parse_latlon(value: str, hemisphere: str) -> float:
    """Convert NMEA lat/lon to signed decimal degrees.

    NMEA format is ``DDMM.MMMMMM`` for latitude or ``DDDMM.MMMMMM`` for
    longitude. The hemisphere character (N/S/E/W) determines the sign.

    Args:
        value: Coordinate string in NMEA format.
        hemisphere: One of ``N``, ``S``, ``E``, ``W``.

    Returns:
        Signed decimal degrees (negative for S and W).
    """
    # Find the decimal point, then the degrees are everything before MM.MMM
    dot_idx = value.index(".")
    # Degrees are all characters before the last 2 digits before the dot
    degrees = int(value[: dot_idx - 2])
    minutes = float(value[dot_idx - 2 :])
    result = degrees + minutes / 60.0
    if hemisphere in ("S", "W"):
        result = -result
    return result


def parse_time_hhmmss(time_str: str, day: date) -> datetime:
    """Convert HHMMSS.SS time string and a date to a UTC datetime.

    Args:
        time_str: Time in ``HHMMSS.SS`` format (fractional seconds optional).
        day: The date to combine with the parsed time.

    Returns:
        UTC datetime with microsecond precision.
    """
    if "." in time_str:
        whole, frac = time_str.split(".")
        microseconds = int(frac.ljust(6, "0")[:6])
    else:
        whole = time_str
        microseconds = 0

    whole = whole.zfill(6)
    hours = int(whole[0:2])
    minutes = int(whole[2:4])
    seconds = int(whole[4:6])

    return datetime(
        day.year,
        day.month,
        day.day,
        hours,
        minutes,
        seconds,
        microseconds,
        tzinfo=timezone.utc,
    )
