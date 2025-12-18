#! /usr/bin/env python3
#
# Parse a WAMOS polar file and extract data frames with metadata.
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations
import logging
import numpy as np
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator
from functools import partial

def _extract_file_timestamp(filepath: str) -> np.datetime64 | None:
    """
    Extract timestamp from a filename.

    Expects filename format: YYYYMMDDHHmmss*.pol*
    Returns None if timestamp cannot be extracted.
    """
    name = os.path.basename(filepath)
    if len(name) < 14:
        return None
    timestamp_str = name[:14]
    if not timestamp_str.isdigit():
        return None
    try:
        return np.datetime64(
            f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}"
            f"T{timestamp_str[8:10]}:{timestamp_str[10:12]}:{timestamp_str[12:14]}"
        )
    except ValueError:
        return None


def _scan_hour_directory(dir_path: str, stime_ns: int, etime_ns: int) -> list[str]:
    """
    Scan a single hour directory for matching .pol files.

    This is a module-level function for pickling compatibility with multiprocessing.
    Times are passed as nanoseconds since epoch for efficient comparison.
    """
    if not os.path.isdir(dir_path):
        return []

    matching_files = []
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                # Check for .pol in filename
                if '.pol' not in name:
                    continue
                # Extract timestamp from filename (first 14 chars: YYYYMMDDHHmmss)
                if len(name) < 14:
                    continue
                timestamp_str = name[:14]
                if not timestamp_str.isdigit():
                    continue
                try:
                    # Parse timestamp efficiently
                    file_time = np.datetime64(
                        f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]}"
                        f"T{timestamp_str[8:10]}:{timestamp_str[10:12]}:{timestamp_str[12:14]}"
                    )
                    file_ns = file_time.astype('datetime64[ns]').astype(np.int64)
                    if stime_ns <= file_ns <= etime_ns:
                        matching_files.append(entry.path)
                except (ValueError, OverflowError):
                    continue
    except OSError:
        pass

    return matching_files


class Filenames:
    """
    Extract filenames between two timestamps.
    This assumes a directory structure, YYYY/MM/DD/HH/YYYYMMDDHHmmss*.pol*

    Uses multiprocessing for GIL-free parallel directory scanning.
    """
    # Minimum number of hour directories before using parallel processing
    _PARALLEL_THRESHOLD = 4

    def __init__(self,
                 stime: np.datetime64,
                 etime: np.datetime64,
                 polar_path: str,
                 workers: int | None = None
                 ) -> None:
        self.stime = np.datetime64(stime, 'ns')  # type: ignore[call-overload]
        self.etime = np.datetime64(etime, 'ns')  # type: ignore[call-overload]
        if self.stime > self.etime:
            raise ValueError(f"Start time ({self.stime}) must be <= end time ({self.etime})")
        self.polar_path = Path(polar_path)
        self.workers = workers
        self._files: list[str] | None = None

    def _generate_hour_directories(self) -> Iterator[str]:
        """
        Generate all hour-level directory paths that could contain files
        within the time range. This efficiently prunes the search space.
        """
        # Round stime down to the hour, etime up to include the full hour
        current = np.datetime64(self.stime, 'h')
        end_hour = np.datetime64(self.etime, 'h')

        while current <= end_hour:
            # Convert to string and parse: format is "YYYY-MM-DDTHH"
            dt_str = str(current)
            parts = dt_str.replace('T', '-').split('-')
            year_s, month_s, day_s, hour_s = parts[0], parts[1], parts[2], parts[3]

            dir_path = self.polar_path / year_s / month_s / day_s / hour_s
            yield str(dir_path)

            current = current + np.timedelta64(1, 'h')

    def _find_files_parallel(self, hour_dirs: list[str]) -> list[str]:
        """Find files using ProcessPoolExecutor for GIL-free parallelism."""
        if not hour_dirs:
            return []

        # Convert times to nanoseconds for efficient integer comparison
        stime_ns = self.stime.astype(np.int64)
        etime_ns = self.etime.astype(np.int64)

        # Create partial function with time bounds
        scan_func = partial(_scan_hour_directory, stime_ns=stime_ns, etime_ns=etime_ns)

        # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
        all_files = []
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = executor.map(scan_func, hour_dirs)
            for file_list in results:
                all_files.extend(file_list)

        return sorted(all_files)

    def _find_files_sequential(self, hour_dirs: list[str]) -> list[str]:
        """Find files sequentially (for small ranges or debugging)."""
        if not hour_dirs:
            return []

        stime_ns = self.stime.astype(np.int64)
        etime_ns = self.etime.astype(np.int64)

        all_files = []
        for dir_path in hour_dirs:
            all_files.extend(_scan_hour_directory(dir_path, stime_ns, etime_ns))

        return sorted(all_files)

    @property
    def files(self) -> list[str]:
        """Lazily compute and cache the list of matching files."""
        if self._files is None:
            hour_dirs = list(self._generate_hour_directories())
            # Use sequential for small ranges or when workers=1 (process spawn overhead)
            if self.workers == 1 or len(hour_dirs) < self._PARALLEL_THRESHOLD:
                self._files = self._find_files_sequential(hour_dirs)
            else:
                self._files = self._find_files_parallel(hour_dirs)
        return self._files

    def __iter__(self) -> Iterator[str]:
        """Iterate over matching filenames."""
        return iter(self.files)

    def __len__(self) -> int:
        """Return the number of matching files."""
        return len(self.files)

    def __bool__(self) -> bool:
        """Return True if any matching files were found."""
        return len(self.files) > 0

    def __repr__(self) -> str:
        return f"Filenames(stime={self.stime}, etime={self.etime}, path={self.polar_path}, count={len(self)})"

    def __str__(self) -> str:
        return '\n'.join(self.files)

    def __enter__(self) -> 'Filenames':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        pass

    @staticmethod
    def _parse_freq(freq: str) -> tuple[int, str]:
        """
        Parse frequency string into multiplier and unit.

        Args:
            freq: Frequency string like 'h', '12m', '30s', '2D', etc.

        Returns:
            Tuple of (multiplier, numpy_unit)

        Examples:
            'h'    -> (1, 'h')
            '12m'  -> (12, 'm')
            '30s'  -> (30, 's')
            '2D'   -> (2, 'D')
            '6hour' -> (6, 'h')
        """
        freq = freq.strip()

        # Unit aliases mapping to numpy datetime64 units
        unit_map = {
            'year': 'Y', 'years': 'Y', 'Y': 'Y',
            'month': 'M', 'months': 'M', 'M': 'M',
            'day': 'D', 'days': 'D', 'D': 'D',
            'hour': 'h', 'hours': 'h', 'h': 'h', 'H': 'h',
            'minute': 'm', 'minutes': 'm', 'min': 'm', 'm': 'm', 'T': 'm',
            'second': 's', 'seconds': 's', 'sec': 's', 's': 's', 'S': 's',
        }

        # Try to parse as "number + unit" (e.g., "12m", "30s", "2D")
        match = re.match(r'^(\d+)\s*([a-zA-Z]+)$', freq)
        if match:
            multiplier = int(match.group(1))
            unit_str = match.group(2)
        else:
            # No number prefix, assume multiplier of 1
            multiplier = 1
            unit_str = freq

        if unit_str not in unit_map:
            raise ValueError(
                f"Unknown frequency unit '{unit_str}'. "
                f"Use: Y/year, M/month, D/day, h/hour, m/min/minute, s/sec/second. "
                f"Optionally prefix with a number (e.g., '12m', '30s', '6h')"
            )

        return multiplier, unit_map[unit_str]

    def groupby(self, freq: str = 'h') -> dict[np.datetime64, list[str]]:
        """
        Group files by time frequency.

        This method groups all matching files into time buckets, enabling
        parallel processing of each group independently.

        Args:
            freq: Time frequency for grouping. Supports:
                - Simple units: 'Y', 'M', 'D', 'h', 'm', 's'
                - Named units: 'year', 'month', 'day', 'hour', 'minute', 'second'
                - Arbitrary periods: '12m' (12 minutes), '30s', '6h', '2D', etc.

        Returns:
            Dict mapping period start times to lists of file paths.
            Keys are sorted chronologically.

        Examples:
            >>> groups = filenames.groupby('D')      # Daily
            >>> groups = filenames.groupby('h')      # Hourly
            >>> groups = filenames.groupby('12m')    # Every 12 minutes
            >>> groups = filenames.groupby('30s')    # Every 30 seconds
            >>> groups = filenames.groupby('6h')     # Every 6 hours

            >>> # Process groups in parallel
            >>> with ProcessPoolExecutor() as executor:
            ...     results = executor.map(process_files, groups.values())
        """
        multiplier, unit = self._parse_freq(freq)

        # For multiplier > 1, use fixed-duration windowing from stime
        if multiplier > 1:
            delta = np.timedelta64(multiplier, unit)  # type: ignore[call-overload]
            return self.time_chunks(delta)

        # For multiplier == 1, use calendar-aligned boundaries
        groups: dict[np.datetime64, list[str]] = {}

        for filepath in self.files:
            ts = _extract_file_timestamp(filepath)
            if ts is None:
                continue
            # Truncate to the specified frequency (calendar-aligned)
            period = np.datetime64(ts, unit)  # type: ignore[call-overload]
            if period not in groups:
                groups[period] = []
            groups[period].append(filepath)

        # Return sorted by period
        return dict(sorted(groups.items()))

    def itergroups(self, freq: str = 'h') -> Iterator[tuple[np.datetime64, list[str]]]:
        """
        Iterate over file groups by time frequency.

        Memory-efficient iterator version of groupby(). Yields groups
        one at a time for streaming processing.

        Args:
            freq: Time frequency - supports 'h', 'D', '12m', '30s', etc.
                  (see groupby() for full options)

        Yields:
            Tuples of (period_start, list of files in that period)

        Example:
            >>> for period, files in filenames.itergroups('12m'):
            ...     print(f"{period}: {len(files)} files")
            ...     process_files(files)
        """
        yield from self.groupby(freq).items()

    def chunks(self, n: int) -> list[list[str]]:
        """
        Split files into n roughly equal-sized chunks.

        Useful for distributing work across a fixed number of workers
        regardless of time boundaries.

        Args:
            n: Number of chunks to create

        Returns:
            List of n lists, each containing a portion of the files.
            Files maintain their sorted order within chunks.

        Example:
            >>> chunks = filenames.chunks(4)  # Split into 4 chunks
            >>> with ProcessPoolExecutor(max_workers=4) as executor:
            ...     results = list(executor.map(process_files, chunks))
        """
        if n <= 0:
            raise ValueError(f"Number of chunks must be positive, got {n}")

        files = self.files
        if not files:
            return [[] for _ in range(n)]

        # Distribute files as evenly as possible
        chunk_size, remainder = divmod(len(files), n)
        chunks = []
        start = 0
        for i in range(n):
            # First 'remainder' chunks get one extra file
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(files[start:end])
            start = end

        return chunks

    def iterchunks(self, n: int) -> Iterator[list[str]]:
        """
        Iterate over n file chunks.

        Args:
            n: Number of chunks

        Yields:
            Lists of files, one chunk at a time
        """
        yield from self.chunks(n)

    def time_chunks(self, delta: np.timedelta64) -> dict[np.datetime64, list[str]]:
        """
        Group files into fixed-duration time windows.

        Unlike groupby() which uses calendar boundaries (e.g., hourly at :00),
        this method creates windows of exactly the specified duration starting
        from stime.

        Args:
            delta: Duration of each chunk as np.timedelta64
                   e.g., np.timedelta64(30, 'm') for 30-minute windows

        Returns:
            Dict mapping window start times to lists of file paths.

        Example:
            >>> # 30-minute windows
            >>> windows = filenames.time_chunks(np.timedelta64(30, 'm'))
            >>> # 6-hour windows
            >>> windows = filenames.time_chunks(np.timedelta64(6, 'h'))
        """
        if delta <= np.timedelta64(0):
            raise ValueError(f"Time delta must be positive, got {delta}")

        groups: dict[np.datetime64, list[str]] = {}
        delta_ns = delta.astype('timedelta64[ns]').astype(np.int64)
        stime_ns = self.stime.astype(np.int64)

        for filepath in self.files:
            ts = _extract_file_timestamp(filepath)
            if ts is None:
                continue

            # Calculate which window this file belongs to
            ts_ns = ts.astype('datetime64[ns]').astype(np.int64)
            window_idx = (ts_ns - stime_ns) // delta_ns
            window_start_ns = stime_ns + (window_idx * delta_ns)
            # Convert int64 nanoseconds back to datetime64
            window_start = window_start_ns.astype('datetime64[ns]')

            if window_start not in groups:
                groups[window_start] = []
            groups[window_start].append(filepath)

        return dict(sorted(groups.items()))


def _parse_timestamp(ts: str) -> np.datetime64:
    """
    Parse a timestamp string in various formats to np.datetime64.

    Supported formats:
        Compact (no separators):
            YYYY                    -> YYYY-01-01T00:00:00
            YYYYMM                  -> YYYY-MM-01T00:00:00
            YYYYMMDD                -> YYYY-MM-DDT00:00:00
            YYYYMMDDHH              -> YYYY-MM-DDTHH:00:00
            YYYYMMDDHHmm            -> YYYY-MM-DDTHH:mm:00
            YYYYMMDDHHmmss          -> YYYY-MM-DDTHH:mm:ss
            YYYYMMDDHHmmssSSS       -> YYYY-MM-DDTHH:mm:ss.SSS (milliseconds)
            YYYYMMDDHHmmssSSSSSS    -> YYYY-MM-DDTHH:mm:ss.SSSSSS (microseconds)

        Compact with T separator:
            YYYYMMDDTHH             -> YYYY-MM-DDTHH:00:00
            YYYYMMDDTHHmm           -> YYYY-MM-DDTHH:mm:00
            YYYYMMDDTHHmmss         -> YYYY-MM-DDTHH:mm:ss

        ISO formats (with separators):
            YYYY-MM-DD
            YYYY-MM-DDTHH
            YYYY-MM-DDTHH:mm
            YYYY-MM-DDTHH:mm:ss
            YYYY-MM-DDTHH:mm:ss.SSS (fractional seconds)
            YYYY-MM-DD HH:mm:ss (space separator)

    Args:
        ts: Timestamp string in any of the supported formats

    Returns:
        np.datetime64 in nanosecond precision

    Raises:
        ValueError: If the timestamp format is not recognized
    """
    ts = ts.strip()

    # Check if it's already in ISO format (contains dashes)
    if '-' in ts:
        # Normalize space separator to T
        ts = ts.replace(' ', 'T')
        try:
            return np.datetime64(ts)
        except ValueError:
            raise ValueError(f"Invalid ISO timestamp format: {ts}")

    # Handle compact format with T separator (e.g., 20220405T1400)
    if 'T' in ts:
        parts = ts.split('T')
        if len(parts) == 2:
            date_part, time_part = parts
            if date_part.isdigit() and time_part.isdigit():
                # Combine and parse as compact format
                ts = date_part + time_part
            else:
                raise ValueError(f"Invalid compact timestamp with T separator: {ts}")
        else:
            raise ValueError(f"Invalid timestamp format: {ts}")

    # Handle compact formats (digits only)
    if not ts.isdigit():
        raise ValueError(f"Compact timestamp must be all digits, got: {ts}")

    n = len(ts)

    # Map length to components: (year, month, day, hour, minute, second, fractional)
    if n == 4:      # YYYY
        iso = f"{ts}-01-01T00:00:00"
    elif n == 6:    # YYYYMM
        iso = f"{ts[:4]}-{ts[4:6]}-01T00:00:00"
    elif n == 8:    # YYYYMMDD
        iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T00:00:00"
    elif n == 10:   # YYYYMMDDHH
        iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:00:00"
    elif n == 12:   # YYYYMMDDHHmm
        iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:00"
    elif n == 14:   # YYYYMMDDHHmmss
        iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}"
    elif n == 17:   # YYYYMMDDHHmmssSSS (milliseconds)
        iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}.{ts[14:17]}"
    elif n == 20:   # YYYYMMDDHHmmssSSSSSS (microseconds)
        iso = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}.{ts[14:20]}"
    else:
        raise ValueError(
            f"Unrecognized compact timestamp length {n}. "
            f"Expected 4 (YYYY), 6 (YYYYMM), 8 (YYYYMMDD), 10 (YYYYMMDDHH), "
            f"12 (YYYYMMDDHHmm), 14 (YYYYMMDDHHmmss), 17 (with ms), or 20 (with us). "
            f"Got: {ts}"
        )

    try:
        return np.datetime64(iso)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp values in '{ts}': {e}")


def _timestamp_type(ts: str) -> np.datetime64:
    """
    Argparse type function for timestamp validation.

    Validates and converts timestamp strings to np.datetime64.
    Used as the 'type' argument in argparse.add_argument().

    Args:
        ts: Timestamp string

    Returns:
        np.datetime64

    Raises:
        argparse.ArgumentTypeError: If timestamp format is invalid
    """
    from argparse import ArgumentTypeError
    try:
        return _parse_timestamp(ts)
    except ValueError as e:
        raise ArgumentTypeError(str(e))


def _directory_type(path: str) -> Path:
    """
    Argparse type function for directory validation.

    Validates that the path exists and is a directory.

    Args:
        path: Path string

    Returns:
        Path object

    Raises:
        argparse.ArgumentTypeError: If path doesn't exist or isn't a directory
    """
    from argparse import ArgumentTypeError
    p = Path(path)
    if not p.exists():
        raise ArgumentTypeError(f"Directory does not exist: {path}")
    if not p.is_dir():
        raise ArgumentTypeError(f"Path is not a directory: {path}")
    return p


def add_common_arguments(parser) -> None:
    """
    Add stime, etime, and polar_path arguments to an argument parser.

    Adds standardized positional arguments with type validation:
    - stime: Start time (validated and converted to np.datetime64)
    - etime: End time (validated and converted to np.datetime64)
    - polar_path: Directory path (validated to exist and be a directory)

    Supported timestamp formats:
        Compact: YYYYMMDD, YYYYMMDDHHmm, YYYYMMDDHHmmss
        Compact with T: YYYYMMDDTHHmm, YYYYMMDDTHHmmss
        ISO: YYYY-MM-DD, YYYY-MM-DDTHH:mm, YYYY-MM-DDTHH:mm:ss

    Args:
        parser: argparse.ArgumentParser or subparser to add arguments to
    """
    parser.add_argument(
        "stime",
        type=_timestamp_type,
        help="Start time (YYYYMMDD, YYYYMMDDTHHmm, or ISO format)"
    )
    parser.add_argument(
        "etime",
        type=_timestamp_type,
        help="End time (YYYYMMDD, YYYYMMDDTHHmm, or ISO format)"
    )
    parser.add_argument(
        "polar_path",
        type=_directory_type,
        help="Path to directory containing polar files"
    )


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    add_common_arguments(parser)
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")


def add_subparser(subparsers) -> None:
    """Register the 'list' subcommand."""
    p = subparsers.add_parser(
        'list',
        help='List/discover POLAR files in time range',
        description="Extract WAMOS polar data files between two timestamps.",
        epilog="Timestamp formats: YYYY, YYYYMM, YYYYMMDD, YYYYMMDDHH, YYYYMMDDHHmm, "
               "YYYYMMDDHHmmss, YYYYMMDDHHmmssSSS (ms), YYYYMMDDHHmmssSSSSSS (us), "
               "or ISO format (YYYY-MM-DD, YYYY-MM-DDTHH:mm:ss, etc.)"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'list' command."""
    import time

    t0 = time.time()
    filenames = Filenames(
        args.stime,
        args.etime,
        args.polar_path,
        workers=args.workers
    )
    # Access .files to trigger the search
    _ = filenames.files
    t1 = time.time()

    logging.info(f"Found {len(filenames)} files in {t1-t0:.3f}s")


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(
        description="Extract WAMOS polar data files between two timestamps.",
        epilog="Timestamp formats: YYYY, YYYYMM, YYYYMMDD, YYYYMMDDHH, YYYYMMDDHHmm, "
               "YYYYMMDDHHmmss, YYYYMMDDHHmmssSSS (ms), YYYYMMDDHHmmssSSSSSS (us), "
               "or ISO format (YYYY-MM-DD, YYYY-MM-DDTHH:mm:ss, etc.)"
    )
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
