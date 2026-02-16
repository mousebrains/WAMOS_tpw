#!/usr/bin/env python3
"""
Extract GPS data from WAMOS polar file metadata and save to CF-compliant NetCDF.

Extracts latitude, longitude, timestamp (UTC), speed, and heading from polar
file metadata, filtering to a minimum time granularity.

Uses a parallelized pipeline with block-based processing for efficient
handling of large datasets (e.g., month-long archives).

Usage:
    python gps_harvest.py 20220405T0000 20220405T2359 /path/to/POLAR -o output.nc --dt 60

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.filenames import _extract_file_timestamp, add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def extract_file_timestamp(filepath: str) -> datetime | None:
    """
    Extract timestamp from a polar filename and return as datetime.

    Args:
        filepath: Path to polar file

    Returns:
        datetime object or None if parsing fails
    """
    ts = _extract_file_timestamp(filepath)
    if ts is None:
        return None
    # Convert np.datetime64 to datetime with UTC timezone
    dt = ts.astype("datetime64[us]").astype("M8[us]").item()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


@dataclass
class GPSPoint:
    """Single GPS observation."""

    time: datetime
    lat: float
    lon: float
    speed: float
    heading: float
    source: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract GPS data from WAMOS polar files to NetCDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("gps_track.nc"),
        help="Output NetCDF file path",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=60.0,
        help="Minimum time step between observations (seconds)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for file reading",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1000,
        help="Number of files to process per block",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for NetCDF file (default: auto-generated)",
    )
    parser.add_argument(
        "--ship",
        type=str,
        default=None,
        help="Ship/platform name (default: from first file metadata)",
    )

    return parser.parse_args()


def print_progress(
    files_processed: int,
    total_files: int,
    points_extracted: int,
    points_written: int,
    width: int = 40,
) -> None:
    """Print a progress bar that updates in place."""
    if total_files == 0:
        return

    fraction = files_processed / total_files
    filled = int(width * fraction)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    percent = fraction * 100

    print(
        f"\r[{bar}] {files_processed:>{len(str(total_files))}}/{total_files} "
        f"({percent:.1f}%) extracted: {points_extracted}, written: {points_written}",
        end="",
        flush=True,
    )


def prune_files_by_timestamp(
    files: list[str],
    dt_seconds: float,
    show_progress: bool = True,
) -> list[str]:
    """
    Pre-filter files based on filename timestamps to reduce I/O.

    Selects files that are at least dt_seconds apart based on the timestamp
    encoded in the filename (first 14 chars: YYYYMMDDHHmmss).

    Args:
        files: List of file paths (assumed sorted by time)
        dt_seconds: Minimum time step between observations
        show_progress: Show progress for large file counts

    Returns:
        Pruned list of file paths
    """
    if not files or dt_seconds <= 0:
        return files

    n_files = len(files)
    progress_interval = max(1, n_files // 100)  # Update every 1%

    pruned: list[str] = []
    last_ts: datetime | None = None

    for i, fn in enumerate(files):
        # Show progress for large file counts
        if show_progress and n_files > 10000 and i % progress_interval == 0:
            pct = 100 * i / n_files
            print(f"\rPruning by filename: {i:,}/{n_files:,} ({pct:.0f}%)", end="", flush=True)

        ts = extract_file_timestamp(fn)
        if ts is None:
            # Can't parse timestamp, keep the file
            pruned.append(fn)
            continue

        if last_ts is None:
            pruned.append(fn)
            last_ts = ts
        else:
            delta = (ts - last_ts).total_seconds()
            if delta >= dt_seconds:
                pruned.append(fn)
                last_ts = ts

    # Clear progress line
    if show_progress and n_files > 10000:
        print("\r" + " " * 60 + "\r", end="", flush=True)

    return pruned


def read_gps_from_file(fn: str) -> GPSPoint | None:
    """
    Read GPS data from a single polar file.

    Args:
        fn: Path to polar file

    Returns:
        GPSPoint or None if data unavailable
    """
    try:
        frame = PolarFile(fn).frame()
        meta = frame.metadata

        # Get timestamp - convert from np.datetime64 to datetime
        ts = frame.timestamp
        if ts is None:
            return None

        # Convert np.datetime64 to datetime with UTC timezone
        frame_dt = ts.astype("datetime64[us]").astype(datetime)
        if frame_dt.tzinfo is None:
            frame_dt = frame_dt.replace(tzinfo=UTC)

        # Get GPS data
        lat = meta.latitude
        lon = meta.longitude

        if lat is None or lon is None:
            return None

        speed = meta.ship_speed
        heading = meta.heading

        return GPSPoint(
            time=frame_dt,
            lat=float(lat),
            lon=float(lon),
            speed=float(speed) if speed is not None else np.nan,
            heading=float(heading) if heading is not None else np.nan,
            source=Path(fn).name,
        )
    except Exception:
        logger.debug("Error reading %s", fn, exc_info=True)
        return None


def process_block(
    files: list[str],
    workers: int,
) -> tuple[list[GPSPoint], str | None]:
    """
    Process a block of files in parallel.

    Args:
        files: List of file paths to process
        workers: Number of parallel workers

    Returns:
        Tuple of (list of GPSPoints, ship_name or None)
    """
    points: list[GPSPoint] = []
    ship_name: str | None = None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(read_gps_from_file, fn): fn for fn in files}

        for future in as_completed(futures):
            point = future.result()
            if point is not None:
                points.append(point)

                # Get ship name from first file with valid metadata
                if ship_name is None:
                    try:
                        frame = PolarFile(futures[future]).frame()
                        ship_name = frame.metadata.attrs.get("TOWER")
                    except Exception:
                        pass

    # Sort by time
    points.sort(key=lambda p: p.time)

    return points, ship_name


def filter_by_dt(
    points: list[GPSPoint],
    dt_seconds: float,
    last_time: datetime | None = None,
) -> tuple[list[GPSPoint], datetime | None]:
    """
    Filter points to maintain minimum time spacing.

    Args:
        points: List of GPSPoints (must be sorted by time)
        dt_seconds: Minimum time step between observations
        last_time: Last accepted time from previous block

    Returns:
        Tuple of (filtered points, last accepted time)
    """
    filtered: list[GPSPoint] = []

    for point in points:
        if last_time is None:
            filtered.append(point)
            last_time = point.time
        else:
            delta = (point.time - last_time).total_seconds()
            if delta >= dt_seconds:
                filtered.append(point)
                last_time = point.time

    return filtered, last_time


class NetCDFWriter:
    """
    Incremental NetCDF writer for GPS data.

    Uses unlimited time dimension for efficient block-based writing.
    """

    def __init__(
        self,
        output_path: Path,
        title: str | None = None,
        ship_name: str | None = None,
    ):
        """
        Initialize the NetCDF writer.

        Args:
            output_path: Output file path
            title: Optional title for the file
            ship_name: Optional ship/platform name
        """
        try:
            import netCDF4 as nc  # type: ignore[import-untyped]
        except ImportError:
            logger.error("netCDF4 package required. Install with: pip install netCDF4")
            raise

        self.output_path = output_path
        self.title = title
        self.ship_name = ship_name or "Unknown"
        self.nc = nc
        self.ds = None
        self.n_written = 0
        self.first_time: datetime | None = None
        self.last_time: datetime | None = None

        # Reference time for CF conventions
        self.time_units = "seconds since 1970-01-01T00:00:00Z"
        self.calendar = "proleptic_gregorian"
        self.epoch = datetime(1970, 1, 1, tzinfo=UTC)

    def _create_file(self) -> None:
        """Create the NetCDF file with dimensions and variables."""
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.ds = self.nc.Dataset(self.output_path, "w", format="NETCDF4")

        # Global attributes (CF-1.8 compliant)
        self.ds.Conventions = "CF-1.8"
        self.ds.title = self.title or f"GPS track for {self.ship_name}"
        self.ds.institution = "WAMOS radar system"
        self.ds.source = "WAMOS polar file metadata"
        self.ds.history = f"Created {datetime.now(UTC).isoformat()} by gps_harvest.py"
        self.ds.references = "https://github.com/OceanWaveS/wamos"
        self.ds.platform = self.ship_name
        self.ds.featureType = "trajectory"

        # Create unlimited time dimension
        self.ds.createDimension("time", None)

        # Time variable
        time_var = self.ds.createVariable("time", "f8", ("time",))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = self.time_units
        time_var.calendar = self.calendar
        time_var.axis = "T"

        # Latitude variable
        lat_var = self.ds.createVariable("latitude", "f8", ("time",))
        lat_var.long_name = "latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var.axis = "Y"
        lat_var.valid_min = -90.0
        lat_var.valid_max = 90.0

        # Longitude variable
        lon_var = self.ds.createVariable("longitude", "f8", ("time",))
        lon_var.long_name = "longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var.axis = "X"
        lon_var.valid_min = -180.0
        lon_var.valid_max = 180.0

        # Speed variable
        speed_var = self.ds.createVariable(
            "speed",
            "f4",
            ("time",),
            fill_value=np.float32(-9999.0),
        )
        speed_var.long_name = "platform speed over ground"
        speed_var.standard_name = "platform_speed_wrt_ground"
        speed_var.units = "m s-1"
        speed_var.valid_min = 0.0
        speed_var.valid_max = 50.0

        # Heading variable
        heading_var = self.ds.createVariable(
            "heading",
            "f4",
            ("time",),
            fill_value=np.float32(-9999.0),
        )
        heading_var.long_name = "platform heading"
        heading_var.standard_name = "platform_course"
        heading_var.units = "degrees"
        heading_var.valid_min = 0.0
        heading_var.valid_max = 360.0
        heading_var.comment = "Heading in degrees clockwise from true north"

    def write_block(self, points: list[GPSPoint]) -> int:
        """
        Write a block of GPS points to the NetCDF file.

        Args:
            points: List of GPSPoints to write

        Returns:
            Number of points written
        """
        if not points:
            return 0

        # Create file on first write
        if self.ds is None:
            self._create_file()

        n = len(points)
        start_idx = self.n_written
        end_idx = start_idx + n

        # Convert times to numeric values
        time_values = np.array(
            [(p.time - self.epoch).total_seconds() for p in points], dtype=np.float64
        )

        # Extract arrays
        lat_values = np.array([p.lat for p in points], dtype=np.float64)
        lon_values = np.array([p.lon for p in points], dtype=np.float64)

        speed_values = np.array([p.speed for p in points], dtype=np.float32)
        speed_values[np.isnan(speed_values)] = -9999.0

        heading_values = np.array([p.heading for p in points], dtype=np.float32)
        heading_values[np.isnan(heading_values)] = -9999.0

        # Write to NetCDF
        self.ds.variables["time"][start_idx:end_idx] = time_values
        self.ds.variables["latitude"][start_idx:end_idx] = lat_values
        self.ds.variables["longitude"][start_idx:end_idx] = lon_values
        self.ds.variables["speed"][start_idx:end_idx] = speed_values
        self.ds.variables["heading"][start_idx:end_idx] = heading_values

        # Flush to disk
        self.ds.sync()

        self.n_written += n

        # Track time range
        if self.first_time is None:
            self.first_time = points[0].time
        self.last_time = points[-1].time

        return n

    def update_ship_name(self, ship_name: str) -> None:
        """Update ship name if not already set."""
        if self.ship_name == "Unknown" and ship_name:
            self.ship_name = ship_name
            if self.ds is not None:
                self.ds.platform = ship_name

    def finalize(self) -> None:
        """Finalize the NetCDF file with updated metadata."""
        if self.ds is None:
            return

        # Update title with time range
        if self.first_time and self.last_time:
            t0 = self.first_time.strftime("%Y-%m-%d %H:%M")
            t1 = self.last_time.strftime("%Y-%m-%d %H:%M")
            if self.title is None:
                self.ds.title = f"GPS track for {self.ship_name} from {t0} to {t1} UTC"

        self.ds.close()
        self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args)

    # Get file list
    print("Finding polar files...", flush=True)
    all_files = list(Filenames(args.stime, args.etime, args.polar_path))
    if not all_files:
        logger.error("No polar files found in time range")
        return 1

    n_all = len(all_files)
    print(f"Found {n_all:,} polar files", flush=True)

    # Pre-filter by filename timestamp to reduce I/O
    files = prune_files_by_timestamp(all_files, args.dt)
    n_files = len(files)
    reduction = 100 * (1 - n_files / n_all) if n_all > 0 else 0
    print(
        f"Pruned to {n_files:,} files by filename timestamp "
        f"(dt >= {args.dt:.1f} s, {reduction:.1f}% reduction)",
        flush=True,
    )

    # Process in blocks
    block_size = args.block_size
    n_blocks = (n_files + block_size - 1) // block_size

    print(
        f"Processing in {n_blocks} blocks of up to {block_size} files each "
        f"with {args.workers} workers",
        flush=True,
    )

    files_processed = 0
    points_extracted = 0
    points_written = 0
    last_time: datetime | None = None

    # Show initial progress bar
    print_progress(0, n_files, 0, 0)

    t0 = time.perf_counter()
    with NetCDFWriter(args.output, title=args.title, ship_name=args.ship) as writer:
        for block_idx in range(n_blocks):
            # Get files for this block
            start = block_idx * block_size
            end = min(start + block_size, n_files)
            block_files = files[start:end]

            # Process block in parallel
            points, ship_name = process_block(block_files, args.workers)

            # Update ship name
            if ship_name:
                writer.update_ship_name(ship_name)

            # Filter by dt
            filtered, last_time = filter_by_dt(points, args.dt, last_time)

            # Write to NetCDF
            n_written = writer.write_block(filtered)

            # Update counters
            files_processed += len(block_files)
            points_extracted += len(points)
            points_written += n_written

            # Update progress
            print_progress(files_processed, n_files, points_extracted, points_written)
    elapsed = time.perf_counter() - t0

    # Final newline after progress bar
    print()
    fps = files_processed / elapsed if elapsed > 0 else 0
    print(f"Processed {files_processed} files in {elapsed:.2f}s ({fps:.1f} files/sec)")

    if points_written == 0:
        print("ERROR: No GPS data extracted", flush=True)
        return 1

    print(
        f"Extracted {points_extracted:,} points, wrote {points_written:,} observations "
        f"(dt >= {args.dt:.1f} s)",
        flush=True,
    )
    print(f"Output: {args.output}", flush=True)

    # Report peak memory usage
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes; on Linux it's in KB
    if sys.platform == "darwin":
        peak_mem_mb = peak_mem / (1024 * 1024)
    else:
        peak_mem_mb = peak_mem / 1024
    print(f"Peak memory: {peak_mem_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
