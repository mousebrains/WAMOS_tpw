#! /usr/bin/env python3
#
# Triplet-based PPS timing: first-radial timestamps using FrameInterpolator
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Compute first-radial timestamps using the triplet FrameInterpolator approach.

For each frame, builds a triplet (prev, current, next) and runs
FrameInterpolator to compute per-radial timestamps using PPS pulses
from all three frames.  Records the timestamp of radial 0 as the
first-radial time and computes the offset from the metadata timestamp.

The parallel phase extracts lightweight metadata + PPS indices from
each file.  The sequential phase walks the sorted frames with a sliding
window of three, constructing FrameProxy objects and running
FrameInterpolator on each triplet.

Usage::

    wamos pps-timing 2022-04-05 2022-04-06 /path/to/POLAR -o pps_timing.nc
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MASK_BIT12 = np.uint16(0x1000)
_PPS_RANGE_BIN = 0


# ---------------------------------------------------------------
# Worker function (must be at module level for ProcessPoolExecutor)
# ---------------------------------------------------------------


def _parse_one_file(filepath: str, config_dict: dict | None) -> list[dict]:
    """Parse lightweight metadata + PPS indices from a single polar file.

    Returns a list of dicts, one per frame, with metadata fields and the
    full PPS indices array needed by FrameInterpolator.
    """
    from wamos_tpw.config import Config
    from wamos_tpw.polarfile import PolarFile

    config = Config()
    if config_dict:
        config._config = config_dict

    results = []
    try:
        pf = PolarFile(filepath, metadata_only=False, max_frames=None, config=config)
        for frame in pf:
            md = frame.metadata
            n_bearings = frame.shape[0]

            # Extract PPS indices from bit 12, range bin 0
            bit12_col = (frame.raw[:, _PPS_RANGE_BIN] & _MASK_BIT12) != 0
            pps_indices = np.where(bit12_col)[0].astype(np.int32)

            results.append({
                "filepath": filepath,
                "timestamp": md.timestamp,
                "repeat_time": md.repeat_time,
                "n_bearings": n_bearings,
                "pps_indices": pps_indices,
                "latitude": md.latitude,
                "longitude": md.longitude,
                "heading": md.heading,
                "ship_speed": md.ship_speed,
                "ship_course": md.ship_course,
                "wind_speed": md.wind_speed,
                "wind_direction": md.wind_direction,
            })
    except Exception as e:
        logger.warning("Error reading %s: %s", filepath, e)

    return results


# ---------------------------------------------------------------
# Triplet processing
# ---------------------------------------------------------------


def _build_frame_data(record: dict):
    """Build a FrameData object from a parsed record dict."""
    from wamos_tpw.interpolator import FrameData, FrameInterpolator

    return FrameData(
        filepath=record["filepath"],
        file_index=0,
        frame_index=0,
        timestamp=record["timestamp"],
        repeat_time=record["repeat_time"] or FrameInterpolator._DEFAULT_REPEAT_TIME,
        latitude=record["latitude"],
        longitude=record["longitude"],
        heading=record["heading"],
        ship_speed=record["ship_speed"],
        wind_speed=record["wind_speed"],
        wind_direction=record["wind_direction"],
        n_bearings=record["n_bearings"],
        n_distances=0,
        pps_indices=record["pps_indices"],
    )


def _process_triplets(all_records: list[dict], tolerance: float) -> list[dict]:
    """Walk sorted records with a sliding window of 3, run FrameInterpolator.

    Returns a list of result dicts with first_radial_time and timing info.
    """
    from wamos_tpw.interpolator import FrameInterpolator
    from wamos_tpw.interpolator_tasks import FrameProxy

    results = []
    n = len(all_records)

    for i in range(n):
        prev_fd = _build_frame_data(all_records[i - 1]) if i > 0 else None
        curr_fd = _build_frame_data(all_records[i])
        next_fd = _build_frame_data(all_records[i + 1]) if i < n - 1 else None

        prev_proxy = FrameProxy(prev_fd) if prev_fd is not None else None
        curr_proxy = FrameProxy(curr_fd)
        next_proxy = FrameProxy(next_fd) if next_fd is not None else None

        interp = FrameInterpolator(prev_proxy, curr_proxy, next_proxy, tolerance=tolerance)

        first_radial_time = interp.times[0]
        offset_ns = (first_radial_time - curr_fd.timestamp) / np.timedelta64(1, "ns")
        offset_ms = offset_ns / 1e6

        results.append({
            "timestamp": curr_fd.timestamp,
            "first_radial_time": first_radial_time,
            "time_offset_ms": offset_ms,
            "pps_count": len(all_records[i]["pps_indices"]),
            "timing_method": interp.timing_method,
            "n_bearings": curr_fd.n_bearings,
            "repeat_time": curr_fd.repeat_time,
            "latitude": curr_fd.latitude,
            "longitude": curr_fd.longitude,
            "heading": curr_fd.heading,
            "ship_speed": curr_fd.ship_speed,
            "wind_speed": curr_fd.wind_speed,
            "wind_direction": curr_fd.wind_direction,
        })

    return results


# ---------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------


def extract_pps_timing(
    stime,
    etime,
    polar_path: str,
    output: str,
    config=None,
    tolerance: float = 1.2,
    workers: int | None = None,
    progress: bool = True,
) -> str:
    """Extract triplet-based PPS timing from all polar files to NetCDF.

    Args:
        stime: Start time
        etime: End time
        polar_path: Polar file directory
        output: Output NetCDF file path
        config: Optional Config object
        tolerance: Time tolerance multiplier for FrameInterpolator
        workers: Number of parallel workers (None = auto)
        progress: Show progress bar

    Returns:
        Path to the output NetCDF file.
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    from wamos_tpw.filenames import Filenames

    filenames = Filenames(stime, etime, polar_path)
    files = list(filenames)

    if not files:
        logging.warning(
            "No files found in %s for time range %s to %s",
            polar_path, stime, etime,
        )
        return ""

    logging.info("Found %d files to process", len(files))

    if workers is None:
        workers = min(len(files), os.cpu_count() or 1)

    config_dict = config._config if config else None

    # Phase 1: Parallel extraction of metadata + PPS
    all_records: list[dict] = []
    t0 = time.perf_counter()

    if workers <= 1:
        try:
            from tqdm import tqdm
            file_iter = tqdm(files, desc="Parsing", unit="file", disable=not progress)
        except ImportError:
            file_iter = files

        for filepath in file_iter:
            all_records.extend(_parse_one_file(filepath, config_dict))
    else:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(files), desc="Parsing", unit="file", disable=not progress)
        except ImportError:
            pbar = None

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_parse_one_file, fp, config_dict): fp
                for fp in files
            }
            for future in as_completed(futures):
                records = future.result()
                all_records.extend(records)
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

    elapsed_parse = time.perf_counter() - t0
    logging.info(
        "Parsed %d frames from %d files in %.1fs (%.0f files/sec)",
        len(all_records), len(files), elapsed_parse,
        len(files) / elapsed_parse if elapsed_parse > 0 else 0,
    )

    if not all_records:
        logging.warning("No valid frame data found")
        return ""

    # Sort by timestamp
    all_records.sort(key=lambda r: r["timestamp"])

    # Phase 2: Sequential triplet processing
    t1 = time.perf_counter()
    results = _process_triplets(all_records, tolerance)
    elapsed_triplet = time.perf_counter() - t1
    logging.info(
        "Triplet processing: %d frames in %.1fs (%.0f frames/sec)",
        len(results), elapsed_triplet,
        len(results) / elapsed_triplet if elapsed_triplet > 0 else 0,
    )

    # Build arrays
    n = len(results)
    time_arr = np.array([r["timestamp"] for r in results], dtype="datetime64[ns]")
    first_radial_arr = np.array(
        [r["first_radial_time"] for r in results], dtype="datetime64[ns]"
    )

    def _float_array(key):
        return np.array(
            [r[key] if r[key] is not None else np.nan for r in results],
            dtype=np.float64,
        )

    def _int_array(key):
        return np.array([r[key] for r in results], dtype=np.int32)

    # Encode timing method as int: 0=linear, N=PPS(N)
    def _parse_timing(method_str):
        if method_str.startswith("PPS("):
            return int(method_str[4:-1])
        return 0

    timing_method_arr = np.array(
        [_parse_timing(r["timing_method"]) for r in results], dtype=np.int32
    )

    # Write NetCDF
    import xarray as xr

    ds = xr.Dataset(
        data_vars={
            "first_radial_time": (
                "time", first_radial_arr,
                {"long_name": "Triplet PPS-derived timestamp of first radial",
                 "standard_name": "time"},
            ),
            "time_offset_ms": (
                "time", _float_array("time_offset_ms"),
                {"long_name": "First radial time minus metadata timestamp",
                 "units": "ms"},
            ),
            "pps_count": (
                "time", _int_array("pps_count"),
                {"long_name": "Number of PPS pulses in frame", "units": "1"},
            ),
            "timing_method": (
                "time", timing_method_arr,
                {"long_name": "Timing method (0=linear, N=PPS pulse count)",
                 "units": "1"},
            ),
            "latitude": (
                "time", _float_array("latitude"),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                "time", _float_array("longitude"),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "heading": (
                "time", _float_array("heading"),
                {"standard_name": "platform_azimuth_angle", "units": "degrees"},
            ),
            "ship_speed": (
                "time", _float_array("ship_speed"),
                {"long_name": "Ship speed", "units": "m s-1"},
            ),
            "wind_speed": (
                "time", _float_array("wind_speed"),
                {"long_name": "Wind speed", "units": "m s-1"},
            ),
            "wind_direction": (
                "time", _float_array("wind_direction"),
                {"long_name": "Wind direction", "units": "degrees"},
            ),
            "repeat_time": (
                "time", _float_array("repeat_time"),
                {"long_name": "Frame repeat time", "units": "s"},
            ),
            "n_bearings": (
                "time", _int_array("n_bearings"),
                {"long_name": "Number of radials in frame", "units": "1"},
            ),
        },
        coords={
            "time": (
                "time", time_arr,
                {"standard_name": "time", "axis": "T"},
            ),
        },
        attrs={
            "title": "WAMOS triplet PPS timing metadata",
            "Conventions": "CF-1.8",
            "source": "wamos pps-timing",
            "history": f"Created {np.datetime64('now')}",
        },
    )

    # Compression encoding
    encoding: dict[str, dict[str, object]] = {}
    for var in ds.data_vars:
        if ds[var].dtype.kind == "M":  # datetime
            encoding[var] = {
                "units": "seconds since 1970-01-01T00:00:00Z",
                "calendar": "standard",
                "dtype": "float64",
                "_FillValue": None,
            }
        else:
            encoding[var] = {"zlib": True, "complevel": 4}
    encoding["time"] = {
        "units": "seconds since 1970-01-01T00:00:00Z",
        "calendar": "standard",
        "dtype": "float64",
        "_FillValue": None,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, encoding=encoding)
    logging.info("Wrote %s (%d records)", output_path, n)

    # Print summary
    offsets = _float_array("time_offset_ms")
    n_pps = int(np.sum(timing_method_arr > 0))
    n_linear = int(np.sum(timing_method_arr == 0))
    print(f"\n{'='*60}")
    print(f"PPS timing (triplet): {n} frames ({n_pps} PPS, {n_linear} linear)")
    print(f"{'='*60}")
    print("  Time offset (first_radial - metadata) in ms:")
    print(f"    mean   = {np.nanmean(offsets):+.1f}")
    print(f"    median = {np.nanmedian(offsets):+.1f}")
    print(f"    std    = {np.nanstd(offsets):.1f}")
    print(f"    min    = {np.nanmin(offsets):+.1f}")
    print(f"    max    = {np.nanmax(offsets):+.1f}")
    for q in [1, 25, 75, 99]:
        print(f"    {q:3d}%   = {np.nanpercentile(offsets, q):+.1f}")
    print(f"  Wrote: {output_path}")

    return str(output_path)


# ---------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument(
        "--output", "-o", type=str, default="pps_timing.nc",
        help="Output NetCDF file path (default: pps_timing.nc)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.2,
        help="Time tolerance multiplier for triplet matching (default: 1.2)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Number of parallel workers (default: auto)",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress", dest="progress", action="store_true", default=True,
        help="Show progress bar (default)",
    )
    progress_group.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="Hide progress bar",
    )


def add_subparser(subparsers) -> None:
    """Register the 'pps-timing' subcommand."""
    p = subparsers.add_parser(
        "pps-timing",
        help="Compute first-radial timestamps using triplet PPS timing",
        description=(
            "Compute PPS-derived first-radial timestamps using the triplet "
            "FrameInterpolator approach (prev/current/next frames) and write "
            "results to a CF-compliant NetCDF file."
        ),
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'pps-timing' command."""
    from wamos_tpw.config import Config

    config = Config(args.config) if args.config else Config()
    extract_pps_timing(
        args.stime,
        args.etime,
        args.polar_path,
        args.output,
        config=config,
        tolerance=args.tolerance,
        workers=args.workers,
        progress=args.progress,
    )


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(
    _add_arguments, run,
    "Compute first-radial timestamps using triplet PPS timing",
)

if __name__ == "__main__":
    main()
