#! /usr/bin/env python3
#
# Extract polar file metadata to NetCDF
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Extract frame metadata from polar files into a CF-compliant NetCDF file.

Parses all .pol files in a time range and writes the per-frame metadata
fields plus PPS pulse information to a single NetCDF file.  Parallelized
across files using ProcessPoolExecutor.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Worker function (must be at module level for ProcessPoolExecutor)
# ---------------------------------------------------------------


_MASK_BIT12 = np.uint16(0x1000)
_PPS_RANGE_BIN = 0


def _parse_one_file(filepath: str, config_dict: dict | None) -> list[dict]:
    """Parse metadata + PPS from a single polar file (runs in worker process).

    Returns a list of dicts, one per frame, with all FrameMetadata fields
    plus PPS pulse count and index of the first PPS pulse.
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
            # Extract PPS indices from bit 12, range bin 0
            bit12_col = (frame.raw[:, _PPS_RANGE_BIN] & _MASK_BIT12) != 0
            pps_indices = np.where(bit12_col)[0]
            results.append(
                {
                    "filepath": filepath,
                    "timestamp": md.timestamp,
                    "repeat_time": md.repeat_time,
                    "latitude": md.latitude,
                    "longitude": md.longitude,
                    "heading": md.heading,
                    "ship_speed": md.ship_speed,
                    "ship_course": md.ship_course,
                    "wind_speed": md.wind_speed,
                    "wind_direction": md.wind_direction,
                    "samples_in_range": md.samples_in_range,
                    "sampling_frequency": md.sampling_frequency,
                    "sample_delay_range": md.sample_delay_range,
                    "n_bearings": frame.n_bearings,
                    "pps_count": len(pps_indices),
                    "first_pps_index": int(pps_indices[0]) if len(pps_indices) > 0 else -1,
                }
            )
    except Exception as e:
        logger.warning("Error reading %s: %s", filepath, e)

    return results


# ---------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------


def extract_metadata(
    stime,
    etime,
    polar_path: str,
    output: str,
    config=None,
    workers: int | None = None,
    progress: bool = True,
) -> str:
    """Extract metadata from all polar files in a time range to NetCDF.

    Args:
        stime: Start time
        etime: End time
        polar_path: Polar file directory
        output: Output NetCDF file path
        config: Optional Config object
        workers: Number of parallel workers (None = auto)
        progress: Show progress bar

    Returns:
        Path to the output NetCDF file.
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np

    from wamos_tpw.filenames import Filenames

    filenames = Filenames(stime, etime, polar_path)
    files = list(filenames)

    if not files:
        logging.warning(
            "No files found in %s for time range %s to %s",
            polar_path,
            stime,
            etime,
        )
        return ""

    logging.info("Found %d files to process", len(files))

    if workers is None:
        workers = min(len(files), os.cpu_count() or 1)

    config_dict = config._config if config else None

    all_records: list[dict] = []
    t0 = time.perf_counter()

    if workers <= 1:
        # Sequential fallback
        try:
            from tqdm import tqdm

            file_iter = tqdm(files, desc="Extracting", unit="file", disable=not progress)
        except ImportError:
            file_iter = files

        for filepath in file_iter:
            all_records.extend(_parse_one_file(filepath, config_dict))
    else:
        # Parallel extraction
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(files), desc="Extracting", unit="file", disable=not progress)
        except ImportError:
            pbar = None

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_parse_one_file, fp, config_dict): fp for fp in files
            }
            for future in as_completed(futures):
                records = future.result()
                all_records.extend(records)
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

    elapsed = time.perf_counter() - t0
    logging.info(
        "Extracted %d frames from %d files in %.1fs (%.0f files/sec)",
        len(all_records),
        len(files),
        elapsed,
        len(files) / elapsed if elapsed > 0 else 0,
    )

    if not all_records:
        logging.warning("No valid frame metadata found")
        return ""

    # Sort by timestamp
    all_records.sort(key=lambda r: r["timestamp"])

    # Build arrays
    n = len(all_records)
    time_arr = np.array([r["timestamp"] for r in all_records], dtype="datetime64[ns]")

    def _float_array(key):
        return np.array(
            [r[key] if r[key] is not None else np.nan for r in all_records],
            dtype=np.float64,
        )

    def _int_array(key):
        return np.array([r[key] for r in all_records], dtype=np.int32)

    # Write NetCDF
    import xarray as xr

    ds = xr.Dataset(
        data_vars={
            "latitude": (
                "time",
                _float_array("latitude"),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": (
                "time",
                _float_array("longitude"),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "heading": (
                "time",
                _float_array("heading"),
                {"standard_name": "platform_azimuth_angle", "units": "degrees"},
            ),
            "ship_speed": (
                "time",
                _float_array("ship_speed"),
                {
                    "standard_name": "platform_speed_wrt_ground",
                    "long_name": "Ship speed",
                    "units": "m s-1",
                },
            ),
            "ship_course": (
                "time",
                _float_array("ship_course"),
                {"standard_name": "platform_course", "units": "degrees"},
            ),
            "wind_speed": (
                "time",
                _float_array("wind_speed"),
                {"long_name": "Wind speed", "units": "m s-1"},
            ),
            "wind_direction": (
                "time",
                _float_array("wind_direction"),
                {"long_name": "Wind direction", "units": "degrees"},
            ),
            "repeat_time": (
                "time",
                _float_array("repeat_time"),
                {"long_name": "Frame repeat time", "units": "s"},
            ),
            "samples_in_range": (
                "time",
                _int_array("samples_in_range"),
                {"long_name": "Range samples per radial", "units": "1"},
            ),
            "sampling_frequency": (
                "time",
                _float_array("sampling_frequency"),
                {"long_name": "ADC sampling frequency", "units": "Hz"},
            ),
            "sample_delay_range": (
                "time",
                _float_array("sample_delay_range"),
                {"long_name": "Sample delay range", "units": "m"},
            ),
            "n_bearings": (
                "time",
                _int_array("n_bearings"),
                {"long_name": "Number of radials in frame", "units": "1"},
            ),
            "pps_count": (
                "time",
                _int_array("pps_count"),
                {"long_name": "Number of PPS pulses in frame", "units": "1"},
            ),
            "first_pps_index": (
                "time",
                _int_array("first_pps_index"),
                {"long_name": "Radial index of first PPS pulse (-1 if none)",
                 "units": "1"},
            ),
        },
        coords={
            "time": (
                "time",
                time_arr,
                {"standard_name": "time", "axis": "T"},
            ),
        },
        attrs={
            "title": "WAMOS polar file frame metadata",
            "Conventions": "CF-1.8",
            "source": "wamos metadata",
            "history": f"Created {np.datetime64('now')}",
        },
    )

    # Compression encoding
    encoding: dict[str, dict[str, object]] = {}
    for var in ds.data_vars:
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
        "--output",
        "-o",
        type=str,
        default="polar_metadata.nc",
        help="Output NetCDF file path (default: polar_metadata.nc)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show progress bar (default)",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Hide progress bar",
    )


def add_subparser(subparsers) -> None:
    """Register the 'metadata' subcommand."""
    p = subparsers.add_parser(
        "metadata",
        help="Extract frame metadata to NetCDF",
        description="Extract metadata from polar files into a CF-compliant NetCDF file",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'metadata' command."""
    from wamos_tpw.config import Config

    config = Config(args.config) if args.config else Config()
    extract_metadata(
        args.stime,
        args.etime,
        args.polar_path,
        args.output,
        config=config,
        workers=args.workers,
        progress=args.progress,
    )


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Extract polar file metadata to NetCDF")

if __name__ == "__main__":
    main()
