#! /usr/bin/env python3
#
# Calculate time shift between polar file metadata and ship GPS positions
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Compute the time lag between radar frame lat/lon and ship GPS lat/lon.

Divides the data into 1-hour chunks and finds the optimal time shift for
each chunk via grid search (minimizing RMS position difference).  Reports
per-hour and overall statistics.

Usage::

    # First extract metadata (fast, parallelized):
    wamos metadata 20220401 20220414 ~/POLAR -o polar_metadata.nc

    # Then compute the time shift:
    wamos timeshift polar_metadata.nc ~/Desktop/WAMOS/ship/
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEG2M = 111_319.5  # meters per degree of latitude


def _rms_at_lag(
    pol_time_s: np.ndarray,
    pol_lat: np.ndarray,
    pol_lon: np.ndarray,
    gps_time_s: np.ndarray,
    gps_lat: np.ndarray,
    gps_lon: np.ndarray,
    m_per_deg_lon: float,
    lag: float,
) -> float:
    """Compute RMS position error at a given lag (seconds)."""
    shifted = pol_time_s + lag
    lat_i = np.interp(shifted, gps_time_s, gps_lat)
    lon_i = np.interp(shifted, gps_time_s, gps_lon)
    dlat = (pol_lat - lat_i) * _DEG2M
    dlon = (pol_lon - lon_i) * m_per_deg_lon
    return float(np.sqrt(np.mean(dlat**2 + dlon**2)))


def find_best_lag(
    pol_time_s: np.ndarray,
    pol_lat: np.ndarray,
    pol_lon: np.ndarray,
    gps_time_s: np.ndarray,
    gps_lat: np.ndarray,
    gps_lon: np.ndarray,
    coarse_range: float = 30.0,
    coarse_step: float = 0.1,
    fine_range: float = 1.0,
    fine_step: float = 0.01,
) -> tuple[float, float]:
    """Find optimal lag via coarse+fine grid search.

    Returns:
        (best_lag_seconds, rms_at_best_lag_meters)
    """
    if len(pol_time_s) < 2:
        return 0.0, float("nan")

    m_per_deg_lon = _DEG2M * np.cos(np.deg2rad(np.mean(pol_lat)))

    # Coarse scan
    lags = np.arange(-coarse_range, coarse_range + coarse_step / 2, coarse_step)
    rms = np.array(
        [
            _rms_at_lag(
                pol_time_s, pol_lat, pol_lon, gps_time_s, gps_lat, gps_lon, m_per_deg_lon, lag
            )
            for lag in lags
        ]
    )
    best = lags[np.argmin(rms)]

    # Fine scan around best
    fine_lags = np.arange(best - fine_range, best + fine_range + fine_step / 2, fine_step)
    fine_rms = np.array(
        [
            _rms_at_lag(
                pol_time_s, pol_lat, pol_lon, gps_time_s, gps_lat, gps_lon, m_per_deg_lon, lag
            )
            for lag in fine_lags
        ]
    )
    fine_best_idx = np.argmin(fine_rms)
    return float(fine_lags[fine_best_idx]), float(fine_rms[fine_best_idx])


def compute_timeshift(
    metadata_nc: str,
    ship_data_dir: str,
    chunk_hours: float = 1.0,
) -> dict:
    """Compute time shift between polar metadata and GPS in hourly chunks.

    Args:
        metadata_nc: Path to polar metadata NetCDF (from ``wamos metadata``).
        ship_data_dir: Directory containing instrument NetCDFs (needs ``gps_abxtwo.nc``).
        chunk_hours: Size of each time chunk in hours.

    Returns:
        Dict with keys: overall_lag, overall_rms, chunks (list of per-chunk results).
    """
    from pathlib import Path

    import xarray as xr

    # Load polar metadata
    ds_pol = xr.open_dataset(metadata_nc)
    pol_time = ds_pol["time"].values.astype("datetime64[ns]")
    pol_lat = ds_pol["latitude"].values.astype(np.float64)
    pol_lon = ds_pol["longitude"].values.astype(np.float64)
    ds_pol.close()

    # Filter NaNs
    valid = np.isfinite(pol_lat) & np.isfinite(pol_lon)
    pol_time = pol_time[valid]
    pol_lat = pol_lat[valid]
    pol_lon = pol_lon[valid]
    pol_time_s = pol_time.astype(np.int64) / 1e9

    logger.info("Polar metadata: %d valid frames", len(pol_time_s))

    # Load GPS
    gps_path = Path(ship_data_dir) / "gps_abxtwo.nc"
    ds_gps = xr.open_dataset(gps_path)
    gps_time_ns = ds_gps["time"].values.astype("datetime64[ns]").astype(np.int64)
    gps_time_s = gps_time_ns / 1e9
    gps_lat = ds_gps["latitude"].values.astype(np.float64)
    gps_lon = ds_gps["longitude"].values.astype(np.float64)
    ds_gps.close()

    gps_valid = np.isfinite(gps_lat) & np.isfinite(gps_lon)
    gps_time_s = gps_time_s[gps_valid]
    gps_lat = gps_lat[gps_valid]
    gps_lon = gps_lon[gps_valid]

    logger.info("GPS data: %d valid records", len(gps_time_s))

    # Overall lag
    overall_lag, overall_rms = find_best_lag(
        pol_time_s, pol_lat, pol_lon, gps_time_s, gps_lat, gps_lon
    )

    # Zero-lag RMS for comparison
    m_per_deg_lon = _DEG2M * np.cos(np.deg2rad(np.mean(pol_lat)))
    zero_rms = _rms_at_lag(
        pol_time_s, pol_lat, pol_lon, gps_time_s, gps_lat, gps_lon, m_per_deg_lon, 0.0
    )

    # Chunk by hour
    chunk_ns = int(chunk_hours * 3600e9)
    t_start = pol_time[0].astype(np.int64)
    t_end = pol_time[-1].astype(np.int64)

    chunks = []
    t = t_start
    while t < t_end:
        t_next = t + chunk_ns
        mask = (pol_time.astype(np.int64) >= t) & (pol_time.astype(np.int64) < t_next)
        n = int(np.sum(mask))

        if n >= 10:
            chunk_lag, chunk_rms = find_best_lag(
                pol_time_s[mask],
                pol_lat[mask],
                pol_lon[mask],
                gps_time_s,
                gps_lat,
                gps_lon,
            )
        else:
            chunk_lag, chunk_rms = float("nan"), float("nan")

        chunk_time = np.datetime64(int(t), "ns")
        chunks.append(
            {
                "start": chunk_time,
                "n_frames": n,
                "lag": chunk_lag,
                "rms": chunk_rms,
            }
        )
        t = t_next

    return {
        "overall_lag": overall_lag,
        "overall_rms": overall_rms,
        "zero_rms": zero_rms,
        "n_frames": len(pol_time_s),
        "chunks": chunks,
    }


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument(
        "metadata_nc",
        type=str,
        help="Polar metadata NetCDF file (from 'wamos metadata')",
    )
    parser.add_argument(
        "ship_data_dir",
        type=str,
        help="Directory containing instrument NetCDFs (needs gps_abxtwo.nc)",
    )
    parser.add_argument(
        "--chunk-hours",
        type=float,
        default=1.0,
        help="Size of each time chunk in hours (default: 1.0)",
    )


def add_subparser(subparsers) -> None:
    """Register the 'timeshift' subcommand."""
    p = subparsers.add_parser(
        "timeshift",
        help="Calculate time shift between radar and GPS positions",
        description=(
            "Compute the time lag between polar file lat/lon and ship GPS positions. "
            "Reports per-hour and overall statistics. Use the result to set "
            "'time_shift' in your config YAML."
        ),
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'timeshift' command."""
    result = compute_timeshift(
        args.metadata_nc,
        args.ship_data_dir,
        chunk_hours=args.chunk_hours,
    )

    # Print results
    print(f"\n{'='*72}")
    print(f"Time shift analysis: {result['n_frames']} frames")
    print(f"{'='*72}")
    print(f"  Overall best lag:  {result['overall_lag']:+.2f} s")
    print(f"  RMS at best lag:   {result['overall_rms']:.2f} m")
    print(f"  RMS at zero lag:   {result['zero_rms']:.2f} m")
    print()

    print(f"{'Start (UTC)':<26s} {'Frames':>7s} {'Lag (s)':>10s} {'RMS (m)':>10s}")
    print("-" * 56)
    lags = []
    for chunk in result["chunks"]:
        ts = np.datetime_as_string(chunk["start"], unit="s")
        lag = chunk["lag"]
        rms = chunk["rms"]
        n = chunk["n_frames"]
        if np.isfinite(lag):
            lags.append(lag)
            print(f"{ts:<26s} {n:>7d} {lag:>+10.2f} {rms:>10.2f}")
        else:
            print(f"{ts:<26s} {n:>7d} {'N/A':>10s} {'N/A':>10s}")

    if lags:
        lags_arr = np.array(lags)
        print("-" * 56)
        print(
            f"{'Lag statistics:':<26s} {'':>7s} "
            f"mean={np.mean(lags_arr):+.2f}  std={np.std(lags_arr):.2f}  "
            f"range=[{np.min(lags_arr):+.2f}, {np.max(lags_arr):+.2f}]"
        )

    print("\nSuggested config setting:")
    print(f"  time_shift: {result['overall_lag']:.2f}  # seconds")


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(
    _add_arguments,
    run,
    "Calculate time shift between radar and GPS positions",
)

if __name__ == "__main__":
    main()
