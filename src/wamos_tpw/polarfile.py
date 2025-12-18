#! /usr/bin/env python3
#
# PolarFile class for parsing WAMOS polar files
# Handles compressed files, header parsing, and binary data extraction
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import bz2
import gzip
import logging
import lzma
import re
from pathlib import Path
from typing import BinaryIO, Iterator, Any

import numpy as np

from wamos_tpw.frame import Frame, FrameMetadata, KNOTS_TO_MS
from wamos_tpw.filenames import _extract_file_timestamp


class PolarFile:
    """
    WAMOS polar file parser.

    Parses .pol files (optionally compressed) containing radar scan data.
    Supports .gz, .bz2, .xz, and .lzma compression.

    File format:
        - ASCII header with key-value pairs until EOH marker
        - Frame data section with per-frame metadata
        - Binary data blocks: 10-byte ASCII length + uint16 little-endian data

    Example:
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> print(f"Found {len(pf)} frames")
        >>> for frame in pf:
        ...     process(frame.intensity)

        >>> # Or get first frame directly
        >>> frame = pf.frame()
    """

    # Header parsing patterns
    _RE_KEY_VALUE = re.compile(rb"^\s*([A-Z0-9_]+)\s+(.+?)(?:\s+CC.*)?$")
    _RE_LATLON = re.compile(rb"^(\d+)\xb0(\d+[.]\d+)\s+([EWNS])\s*$")
    _RE_DATE = re.compile(rb"^(\d{2})-(\d{2})-(\d{4})\s*$")
    _RE_TIME = re.compile(rb"^(\d{2}):(\d{2}):(\d{2})(?:[.](\d{3}))?\s*$")
    _RE_FRAME_START = re.compile(rb"^\s*CC\s+[*]+\s+START\s+FRAMEDATA\s+SECTION")
    _RE_FRAME_STOP = re.compile(rb"^\s*CC\s+[*]+\s+STOP\s+FRAMEDATA\s+SECTION")
    _RE_FRAME_LINE = re.compile(rb"^\s*F(\d+)\s+(.*)$")

    _LENGTH_FIELD_SIZE = 10  # Bytes for ASCII length field

    def __init__(self, filepath: str | Path) -> None:
        """
        Initialize and parse a WAMOS polar file.

        Args:
            filepath: Path to .pol file (supports .gz, .bz2, .xz, .lzma)
        """
        self._filepath = Path(filepath)
        self._header: dict[str, Any] = {}
        self._frame_metadata: list[FrameMetadata] = []
        self._frames: list[Frame] = []

        self._parse()

    def _parse(self) -> None:
        """Parse the polar file."""
        opener = self._get_opener()

        with opener(str(self._filepath), "rb") as fp:
            # Parse header
            header_lines, frame_lines = self._read_header(fp)
            self._header = self._parse_header(header_lines)
            self._frame_metadata = self._parse_frame_section(frame_lines)

            # Parse binary data
            self._parse_data(fp)

    def _get_opener(self):
        """Get the appropriate file opener based on extension."""
        suffix = self._filepath.suffix.lower()
        name = self._filepath.name.lower()

        if suffix == ".gz" or name.endswith(".pol.gz"):
            return gzip.open
        elif suffix == ".bz2" or name.endswith(".pol.bz2"):
            return bz2.open
        elif suffix in {".xz", ".lzma"} or name.endswith(".pol.xz"):
            return lzma.open
        else:
            return open

    def _read_header(self, fp: BinaryIO) -> tuple[list[bytes], list[bytes]]:
        """Read header lines until EOH marker, separating frame section."""
        header_lines = []
        frame_lines = []
        in_frame_section = False

        for line in fp:
            stripped = line.strip()

            # Check for end of header
            if stripped.startswith(b"EOH"):
                break

            # Track frame section
            if self._RE_FRAME_START.match(stripped):
                in_frame_section = True
                continue
            if self._RE_FRAME_STOP.match(stripped):
                in_frame_section = False
                continue

            if in_frame_section:
                frame_lines.append(stripped)
            else:
                header_lines.append(stripped)

        return header_lines, frame_lines

    def _parse_header(self, lines: list[bytes]) -> dict[str, Any]:
        """Parse header key-value pairs."""
        header = {}

        for line in lines:
            # Skip pure comments
            if line.startswith(b"CC"):
                continue

            match = self._RE_KEY_VALUE.match(line)
            if match:
                key = match.group(1).decode("utf-8", errors="ignore")
                value = self._parse_value(match.group(2).strip())
                header[key] = value

        return header

    def _parse_frame_section(self, lines: list[bytes]) -> list[FrameMetadata]:
        """Parse frame metadata from frame section lines."""
        metadata_list = []

        for line in lines:
            # Skip comment lines (header rows)
            if line.startswith(b"CC"):
                continue

            match = self._RE_FRAME_LINE.match(line)
            if match:
                frame_idx = int(match.group(1))
                frame_data = match.group(2).split()

                metadata = self._build_frame_metadata(frame_idx, frame_data)
                metadata_list.append(metadata)

        # If no frame section, create single metadata from header
        if not metadata_list:
            metadata_list.append(self._build_metadata_from_header())

        return metadata_list

    def _build_frame_metadata(self, frame_idx: int, parts: list[bytes]) -> FrameMetadata:
        """Build FrameMetadata from frame line parts."""
        # Frame line format (typical):
        # F0001 MM-DD-YYYY HH:MM:SS.mmm GYROC SHIPS RPT LAT LONG SHIPR WINDS WINDR P_DEP SPTWL SPTWT WATSP

        metadata = FrameMetadata(
            timestamp=np.datetime64("NaT"),
            filename=self._filepath.name,
            frame_index=frame_idx,
            samples_in_range=self._header.get("FIFO", 0),
            sample_delay_range=self._header.get("SDRNG", 0.0),
            sampling_frequency=self._header.get("SFREQ", 0.0),
            repeat_time=self._header.get("RPT", 0.0),
            data_bits=self._header.get("DABIT", 12),
            noise_floor=self._header.get("NSFLR", 0),
            wind_sensor_height=self._header.get("WINDH"),
        )

        # Parse frame-specific fields
        try:
            if len(parts) >= 2:
                # Date and time
                date_match = self._RE_DATE.match(parts[0])
                time_match = self._RE_TIME.match(parts[1])

                if date_match and time_match:
                    year = int(date_match.group(3))
                    month = int(date_match.group(1))
                    day = int(date_match.group(2))
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    second = int(time_match.group(3))
                    ms = int(time_match.group(4)) if time_match.group(4) else 0

                    metadata.timestamp = np.datetime64(
                        f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{ms:03d}"
                    )

            if len(parts) >= 3:
                metadata.heading = float(parts[2])
            if len(parts) >= 4:
                metadata.ship_speed = float(parts[3]) * KNOTS_TO_MS  # Convert knots to m/s
            if len(parts) >= 7:  # LAT is typically parts[5:7] (deg + dir)
                metadata.latitude = self._parse_latlon(parts[5], parts[6])
            if len(parts) >= 9:  # LONG is typically parts[7:9]
                metadata.longitude = self._parse_latlon(parts[7], parts[8])
            if len(parts) >= 10:
                metadata.ship_course = float(parts[9])
            if len(parts) >= 11:
                metadata.wind_speed = float(parts[10])
            if len(parts) >= 12:
                metadata.wind_direction = float(parts[11])

        except (ValueError, IndexError):
            pass

        return metadata

    def _build_metadata_from_header(self) -> FrameMetadata:
        """Build FrameMetadata from header when no frame section exists."""
        # Try to get timestamp from filename
        timestamp = _extract_file_timestamp(str(self._filepath))
        if timestamp is None:
            timestamp = np.datetime64("NaT")

        # Convert ship speed from knots to m/s
        ships_knots = self._header.get("SHIPS")
        ship_speed_ms = ships_knots * KNOTS_TO_MS if ships_knots is not None else None

        return FrameMetadata(
            timestamp=timestamp,
            filename=self._filepath.name,
            frame_index=0,
            latitude=self._header.get("LAT"),
            longitude=self._header.get("LONG"),
            heading=self._header.get("GYROC"),
            ship_speed=ship_speed_ms,
            ship_course=self._header.get("SHIPR"),
            samples_in_range=self._header.get("FIFO", 0),
            sample_delay_range=self._header.get("SDRNG", 0.0),
            sampling_frequency=self._header.get("SFREQ", 0.0),
            repeat_time=self._header.get("RPT", 0.0),
            data_bits=self._header.get("DABIT", 12),
            noise_floor=self._header.get("NSFLR", 0),
            wind_speed=self._header.get("WINDS"),
            wind_direction=self._header.get("WINDR"),
            wind_sensor_height=self._header.get("WINDH"),
        )

    def _parse_data(self, fp: BinaryIO) -> None:
        """Parse binary data blocks into Frame objects."""
        n_samples = self._header.get("FIFO", 0)

        for idx, metadata in enumerate(self._frame_metadata):
            # Read length field
            length_bytes = fp.read(self._LENGTH_FIELD_SIZE)
            if len(length_bytes) != self._LENGTH_FIELD_SIZE:
                logging.debug(f"End of file at frame {idx}")
                break

            try:
                length = int(length_bytes.decode("utf-8", errors="ignore").strip())
            except ValueError:
                logging.warning(f"{self._filepath}: Invalid length field at frame {idx}")
                break

            # Read binary data
            buffer = fp.read(length)
            if len(buffer) != length:
                logging.warning(
                    f"{self._filepath}: Frame {idx}: expected {length} bytes, got {len(buffer)}"
                )
                break

            # Parse as uint16 little-endian
            data = np.frombuffer(buffer, dtype="<H")

            # Reshape if we know samples_in_range
            if n_samples > 0:
                n_radials = data.size // n_samples
                if n_radials * n_samples == data.size:
                    data = data.reshape((n_radials, n_samples))
                else:
                    # Truncate to fit
                    usable = n_radials * n_samples
                    logging.warning(
                        f"{self._filepath}: Frame {idx}: truncating {data.size - usable} values"
                    )
                    data = data[:usable].reshape((n_radials, n_samples))

            frame = Frame(data, metadata)
            self._frames.append(frame)

    def _parse_value(self, value: bytes) -> Any:
        """Parse a value from the header."""
        # Try lat/lon
        match = self._RE_LATLON.match(value)
        if match:
            deg = int(match.group(1))
            minutes = float(match.group(2))
            direction = match.group(3).decode()
            result = deg + minutes / 60.0
            if direction in ("S", "W"):
                result = -result
            return result

        # Try date
        match = self._RE_DATE.match(value)
        if match:
            return np.datetime64(
                f"{match.group(3).decode()}-{match.group(1).decode()}-{match.group(2).decode()}"
            )

        # Try time
        match = self._RE_TIME.match(value)
        if match:
            h, m, s = match.group(1), match.group(2), match.group(3)
            ms = match.group(4) or b"000"
            return (
                np.timedelta64(int(h), "h")
                + np.timedelta64(int(m), "m")
                + np.timedelta64(int(s), "s")
                + np.timedelta64(int(ms), "ms")
            )

        # Try number
        value_str = value.decode("utf-8", errors="ignore").strip()
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            return value_str

    def _parse_latlon(self, deg_part: bytes, dir_part: bytes) -> float | None:
        """Parse latitude/longitude from separate degree and direction parts."""
        try:
            # Format: "018°34.213" "N" or similar
            combined = deg_part + b" " + dir_part
            match = self._RE_LATLON.match(combined)
            if match:
                deg = int(match.group(1))
                minutes = float(match.group(2))
                direction = match.group(3).decode()
                result = deg + minutes / 60.0
                if direction in ("S", "W"):
                    result = -result
                return result
        except (ValueError, AttributeError):
            pass
        return None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def filepath(self) -> Path:
        """Return the file path."""
        return self._filepath

    @property
    def header(self) -> dict[str, Any]:
        """Return the parsed header dictionary."""
        return self._header

    @property
    def frames(self) -> list[Frame]:
        """Return all parsed frames."""
        return self._frames

    def frame(self, index: int = 0) -> Frame:
        """
        Get a specific frame by index.

        Args:
            index: Frame index (default: 0 for first frame)

        Returns:
            Frame object

        Raises:
            IndexError: If index is out of range
        """
        return self._frames[index]

    def __len__(self) -> int:
        """Return the number of frames in the file."""
        return len(self._frames)

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames."""
        return iter(self._frames)

    def __getitem__(self, index: int) -> Frame:
        """Get frame by index."""
        return self._frames[index]

    def __bool__(self) -> bool:
        """Return True if any frames were parsed."""
        return len(self._frames) > 0

    def __repr__(self) -> str:
        return f"PolarFile('{self._filepath.name}', frames={len(self)})"

    def __str__(self) -> str:
        lines = [f"PolarFile: {self._filepath.name}"]
        lines.append(f"  Frames: {len(self)}")
        if self._frames:
            lines.append(f"  Shape: {self._frames[0].shape}")
            lines.append(f"  First timestamp: {self._frames[0].timestamp}")
        if "FIFO" in self._header:
            lines.append(f"  Samples in range: {self._header['FIFO']}")
        return "\n".join(lines)


# Module-level loader function for multiprocessing compatibility
def load_polar_file(filepath: str) -> Frame | None:
    """
    Load a polar file and return its first frame.

    This is a module-level function for use with multiprocessing.

    Args:
        filepath: Path to polar file

    Returns:
        First Frame from the file, or None on error
    """
    try:
        pf = PolarFile(filepath)
        if pf:
            return pf.frame()
        return None
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None


def load_polar_file_all(filepath: str) -> list[Frame]:
    """
    Load a polar file and return all frames.

    Args:
        filepath: Path to polar file

    Returns:
        List of Frame objects (empty on error)
    """
    try:
        pf = PolarFile(filepath)
        return pf.frames
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return []


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", nargs="+", help="Polar file(s) to parse")
    parser.add_argument("--show-header", action="store_true", help="Show full header")


def add_subparser(subparsers) -> None:
    """Register the 'parse' subcommand."""
    p = subparsers.add_parser(
        "parse",
        help="Parse single POLAR file",
        description="Parse WAMOS polar file and display information",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'parse' command."""
    for filepath in args.filename:
        logging.info(f"{'=' * 60}")
        logging.info(f"Parsing: {filepath}")
        logging.info("=" * 60)

        try:
            pf = PolarFile(filepath)
            logging.info(f"{pf}")

            if args.show_header:
                logging.info("Header:")
                for key, value in sorted(pf.header.items()):
                    logging.info(f"  {key}: {value}")

            if pf:
                frame = pf.frame()
                logging.info("First frame:")
                logging.info(f"  Timestamp: {frame.timestamp}")
                logging.info(f"  Shape: {frame.shape}")
                logging.info(
                    f"  Intensity range: [{frame.intensity.min()}, {frame.intensity.max()}]"
                )
                logging.info(f"  Bit12 (PPS) any: {frame.bit12.any()}")
                logging.info(f"  Bit13 (bearing) any: {frame.bit13.any()}")

        except Exception as e:
            logging.error(f"Failed to parse {filepath}: {e}")


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Parse WAMOS polar file")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
