#! /usr/bin/env python3
#
# CordcFile - parser for CORDC "NAVNET SAMPLE" fixed-station radar files
# (CSS Angaur, ARCTERX 2023)
#
# Jul-2026, Pat Welch, pat@mousebrains.com

"""
CORDC NAVNET-SAMPLE fixed-station radar file parser.

The CORDC recorder on the Angaur tower (Furuno FAR-3220BB chart radar)
writes one file per recorded antenna rotation (every other rotation,
~5 s cadence):

- ASCII header of ``KEY  VALUE   CC comment`` lines (CRLF) ending with
  an ``EOH`` line.
- A 10-character ASCII payload byte count (no trailing newline).
- Per radial: ``[uint16 LE azimuth word][FIFO x uint8 intensity]``.
  The azimuth word is a 13-bit encoder (8192 counts/rev) relative to
  the antenna heading-line mark; the radial count varies per sweep, so
  per-radial azimuths are required.

Calibration (station geometry, encoder zero, true cell size, range
offset, recorder clock offset) comes from a :class:`FixedStation`,
normally loaded from packaged YAML (see ``data/css_angaur_2023.yaml``,
measured from GPS-tracked ship transits). The header's RANGE/SCALE
ratio is close to, but NOT, the true range cell size — prefer the
station's calibrated ``cell_m``.

See Also
--------
- PolarFile: the shipboard WAMOS uint16 format parser
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

__all__ = ["CordcFile", "FixedStation"]

ENCODER_COUNTS = 8192  # 13-bit azimuth encoder, counts per revolution
_RADIAL_PREFIX = 2  # bytes of azimuth word ahead of each radial
_COUNT_CHARS = 10  # ASCII payload byte count after the EOH line
_HEADER_LINE = re.compile(r"^(\w+)\s+(\S+)")
_FRAME_LINE = re.compile(r"^F\d+\s+(\d{2}-\d{2}-\d{4})\s+(\d{2}:\d{2}:\d{2})")


@dataclass(frozen=True)
class FixedStation:
    """Fixed-station geometry and calibration for a CORDC radar."""

    name: str
    latitude: float  # deg N
    longitude: float  # deg E
    antenna_height_m: float  # above MSL
    theta0_deg: float  # encoder zero: true = word/8192*360 + theta0
    cell_m: float | None = None  # None -> header RANGE/SCALE fallback
    range_offset_m: float = 0.0  # range = cell_m*index + range_offset_m
    clock_offset_s: float = 0.0  # t_true = t_file + clock + intra-sweep

    @classmethod
    def from_yaml(cls, source: str | Path) -> FixedStation:
        """Load a station from a YAML path or a packaged station name.

        Args:
            source: Filesystem path, or a bare name like
                ``"css_angaur_2023"`` resolved from package data.
        """
        path = Path(source)
        if path.suffix in (".yaml", ".yml") and path.exists():
            text = path.read_text()
        else:
            text = files("wamos_tpw.data").joinpath(f"{source}.yaml").read_text()
        raw = yaml.safe_load(text)
        fields = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**fields)


class CordcFile:
    """One CORDC NAVNET-SAMPLE sweep (a single antenna rotation)."""

    def __init__(self, filename: str | Path, station: FixedStation | None = None) -> None:
        self.filename = Path(filename)
        self.station = station
        raw = self.filename.read_bytes()
        self.header = self._parse_header(raw)
        self.words, self.intensity = self._parse_payload(raw)
        logger.debug("%s: %d radials x %d cells", self.filename.name, *self.intensity.shape)

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_header(raw: bytes) -> dict[str, float | str]:
        """Parse ``KEY  VALUE`` header lines up to the EOH marker."""
        i_eoh = raw.find(b"EOH")
        if i_eoh < 0:
            raise ValueError("no EOH marker: not a CORDC NAVNET-SAMPLE file")
        header: dict[str, float | str] = {}
        for line in raw[:i_eoh].decode("latin1").splitlines():
            m = _FRAME_LINE.match(line)
            if m:  # F0001  MM-DD-YYYY HH:MM:SS RPT
                header["frame_time"] = datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%m-%d-%Y %H:%M:%S"
                ).replace(tzinfo=UTC)
                continue
            m = _HEADER_LINE.match(line)
            if not m or m.group(1) == "CC":
                continue
            key, value = m.group(1), m.group(2)
            try:
                header[key] = float(value)
            except ValueError:
                header[key] = value
        return header

    def _parse_payload(self, raw: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Decode the radials after the EOH line.

        Returns:
            ``(words, intensity)`` — the per-radial 13-bit encoder words
            ``(n_radials,)`` int32 and intensities ``(n_radials,
            sweep_len)`` uint8.
        """
        i_nl = raw.find(b"\n", raw.find(b"EOH"))
        start = i_nl + 1 + _COUNT_CHARS
        try:
            nbytes = int(raw[i_nl + 1 : start])
        except ValueError as e:
            raise ValueError(f"{self.filename}: bad payload byte count") from e
        payload = np.frombuffer(raw, dtype=np.uint8, offset=start)
        sweep_len = int(self.header.get("FIFO", 884))
        stride = _RADIAL_PREFIX + sweep_len
        n = min(nbytes, len(payload)) // stride
        if n == 0:
            raise ValueError(f"{self.filename}: empty payload")
        if nbytes > len(payload):
            logger.warning(
                "%s: payload count %d exceeds file (%d); truncated read",
                self.filename.name,
                nbytes,
                len(payload),
            )
        a = payload[: n * stride].reshape(n, stride)
        words = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)) % ENCODER_COUNTS
        return words, a[:, _RADIAL_PREFIX:]

    # ------------------------------------------------------------------
    # calibrated views (require a station)
    # ------------------------------------------------------------------

    def _require_station(self) -> FixedStation:
        if self.station is None:
            raise ValueError("a FixedStation is required for calibrated output")
        return self.station

    @property
    def cell_m(self) -> float:
        """Calibrated range cell size, falling back to header RANGE/SCALE."""
        if self.station is not None and self.station.cell_m is not None:
            return self.station.cell_m
        range_km = float(self.header["RANGE"])
        scale = float(self.header["SCALE"])
        if scale <= 0:
            raise ValueError(f"{self.filename}: no usable SCALE in header")
        return range_km * 1000.0 / scale

    @property
    def bearings(self) -> np.ndarray:
        """Per-radial true bearings (deg, [0, 360))."""
        station = self._require_station()
        return (self.words / ENCODER_COUNTS * 360.0 + station.theta0_deg) % 360.0

    @property
    def ranges(self) -> np.ndarray:
        """Range to each cell center (m from the antenna)."""
        station = self._require_station()
        n_cells = self.intensity.shape[1]
        return self.cell_m * (np.arange(n_cells) + 0.5) + station.range_offset_m

    @property
    def file_time(self) -> datetime:
        """Uncorrected sweep start time (header frame line, else filename)."""
        if "frame_time" in self.header:
            ft = self.header["frame_time"]
            assert isinstance(ft, datetime)
            return ft
        s = self.filename.name[:14]
        return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=UTC)

    @property
    def times(self) -> np.ndarray:
        """Per-radial UTC timestamps (datetime64[ms]), clock-corrected.

        ``t_true = t_file + clock_offset + word/8192 * RPT`` — sweeps
        start at encoder word 0, so the word is also the intra-sweep
        scan phase. The recorder PC clock wanders ±4 s day-to-day; for
        sub-5 s work use an hourly clock series in place of the
        station's constant offset.
        """
        station = self._require_station()
        rpt = float(self.header.get("RPT", 2.5))
        base = np.datetime64(self.file_time.replace(tzinfo=None), "ms")
        offset_ms = (station.clock_offset_s + self.words / ENCODER_COUNTS * rpt) * 1000.0
        return base + offset_ms.astype("timedelta64[ms]")
