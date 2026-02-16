#! /usr/bin/env python3
#
# Hard return timing offset analysis
#
# Sweeps timing offsets to find the one that maximizes spatial concentration
# of bright pixels across frames — i.e., the offset that makes hard returns
# (land, platforms, buoys) most stable in earth coordinates.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Find the timing offset that stabilizes hard returns in earth coordinates.

Hard returns should appear at stable earth coordinates across frames.
When per-radial timestamps used for heading interpolation are off by dt,
the heading error is ``dH/dt * dt``.  This creates position errors
proportional to ``range * dH/dt * dt``.  Because ``dH/dt`` varies as the
ship turns, the error varies frame-to-frame, causing hard returns to wobble.

This module sweeps candidate timing offsets and finds the one that maximizes
spatial concentration (L2 norm of grid counts) of bright pixels — the offset
that makes hard returns most stable.

Usage::

    wamos hard-returns 20220402T0415 20220402T0500 ~/Desktop/WAMOS/POLAR \\
        --ship-data ~/Desktop/WAMOS/ship
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MASK_BIT12 = np.uint16(0x1000)
_PPS_RANGE_BIN = 0
_DEG2M = 111_319.5  # meters per degree of latitude
_DEG2RAD = np.pi / 180.0


# ---------------------------------------------------------------
# Worker function (must be at module level for ProcessPoolExecutor)
# ---------------------------------------------------------------


def _process_one_file(
    filepath: str,
    config_dict: dict | None,
    threshold_pct: float,
) -> list[dict]:
    """Run FramePipeline on each frame, extract bright pixels + metadata.

    Returns a list of dicts, one per frame, containing:
    - metadata fields needed for FrameInterpolator triplets
    - theta_array, ground_range for projection
    - sparse bright pixels (bearing_idx, range_idx) above threshold
    """
    from wamos_tpw.config import Config
    from wamos_tpw.frame_pipeline import FramePipeline
    from wamos_tpw.interpolator import FrameInterpolator
    from wamos_tpw.polarfile import PolarFile

    config = Config()
    if config_dict:
        config._config = config_dict

    results = []
    try:
        pf = PolarFile(filepath, config=config)
        for frame in pf:
            fp = FramePipeline(frame, config=config)
            md = fp.metadata
            n_bearings = fp.n_bearings

            # Extract PPS indices from bit 12, range bin 0
            bit12_col = (frame.raw[:, _PPS_RANGE_BIN] & _MASK_BIT12) != 0
            pps_indices = np.where(bit12_col)[0].astype(np.int32)

            # Extract bright pixels above percentile threshold
            intensity = fp.final_intensity
            valid_mask = ~np.isnan(intensity)
            if np.any(valid_mask):
                threshold = np.percentile(intensity[valid_mask], threshold_pct)
                bright_mask = valid_mask & (intensity >= threshold)
                bright_bi, bright_ri = np.where(bright_mask)
            else:
                bright_bi = np.array([], dtype=np.int32)
                bright_ri = np.array([], dtype=np.int32)

            results.append({
                "filepath": filepath,
                "timestamp": md.timestamp,
                "repeat_time": md.repeat_time or FrameInterpolator._DEFAULT_REPEAT_TIME,
                "n_bearings": n_bearings,
                "pps_indices": pps_indices,
                "latitude": md.latitude,
                "longitude": md.longitude,
                "heading": md.heading,
                "ship_speed": md.ship_speed,
                "wind_speed": md.wind_speed,
                "wind_direction": md.wind_direction,
                "theta_array": fp.theta_array.copy(),
                "ground_range": fp.ground_range.copy(),
                "bright_bi": bright_bi.astype(np.int32),
                "bright_ri": bright_ri.astype(np.int32),
            })
    except Exception as e:
        logger.warning("Error reading %s: %s", filepath, e)

    return results


# ---------------------------------------------------------------
# Triplet processing helpers
# ---------------------------------------------------------------


def _build_frame_data(record: dict):
    """Build a FrameData object from a parsed record dict."""
    from wamos_tpw.interpolator import FrameData

    return FrameData(
        filepath=record["filepath"],
        file_index=0,
        frame_index=0,
        timestamp=record["timestamp"],
        repeat_time=record["repeat_time"],
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


# ---------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------


def find_hard_return_offset(
    stime,
    etime,
    polar_path: str,
    ship_data_dir: str,
    config=None,
    threshold_pct: float = 97.0,
    sweep_range: float = 2.0,
    sweep_step: float = 0.02,
    grid_spacing: float = 20.0,
    workers: int | None = None,
    progress: bool = True,
    plot: bool = True,
) -> dict:
    """Sweep timing offsets and find the one that stabilizes hard returns.

    Args:
        stime: Start time
        etime: End time
        polar_path: Polar file directory
        ship_data_dir: Directory containing instrument NetCDFs
        config: Optional Config object
        threshold_pct: Bright pixel percentile threshold (default 97)
        sweep_range: Sweep range in seconds (default +/-2.0)
        sweep_step: Sweep step in seconds (default 0.02)
        grid_spacing: Grid cell size in meters (default 20)
        workers: Number of parallel workers (None = auto)
        progress: Show progress bar
        plot: Show result plots

    Returns:
        Dict with best_offset, offsets, metrics, and statistics.
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import Path

    from wamos_tpw.filenames import Filenames
    from wamos_tpw.instruments.ship_data import ShipData
    from wamos_tpw.interpolator import FrameInterpolator
    from wamos_tpw.interpolator_tasks import FrameProxy

    # ------------------------------------------------------------------
    # Phase 1: Parallel frame processing
    # ------------------------------------------------------------------
    filenames = Filenames(stime, etime, polar_path)
    files = list(filenames)

    if not files:
        logger.warning(
            "No files found in %s for time range %s to %s",
            polar_path, stime, etime,
        )
        return {"best_offset": 0.0, "offsets": np.array([]), "metrics": np.array([])}

    logger.info("Found %d files to process", len(files))

    if workers is None:
        workers = min(len(files), os.cpu_count() or 1)

    config_dict = config._config if config else None

    all_records: list[dict] = []
    t0 = time.perf_counter()

    if workers <= 1:
        try:
            from tqdm import tqdm
            file_iter = tqdm(files, desc="Parsing", unit="file", disable=not progress)
        except ImportError:
            file_iter = files

        for filepath in file_iter:
            all_records.extend(_process_one_file(filepath, config_dict, threshold_pct))
    else:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(files), desc="Parsing", unit="file", disable=not progress)
        except ImportError:
            pbar = None

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_one_file, fp, config_dict, threshold_pct): fp
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
    logger.info(
        "Parsed %d frames from %d files in %.1fs (%.0f files/sec)",
        len(all_records), len(files), elapsed_parse,
        len(files) / elapsed_parse if elapsed_parse > 0 else 0,
    )

    if not all_records:
        logger.warning("No valid frame data found")
        return {"best_offset": 0.0, "offsets": np.array([]), "metrics": np.array([])}

    # Sort by timestamp
    all_records.sort(key=lambda r: r["timestamp"])
    n_frames = len(all_records)

    total_bright = sum(len(r["bright_bi"]) for r in all_records)
    logger.info("Total bright pixels across all frames: %d", total_bright)

    if total_bright == 0:
        logger.warning("No bright pixels found — try lowering --threshold")
        return {"best_offset": 0.0, "offsets": np.array([]), "metrics": np.array([])}

    # ------------------------------------------------------------------
    # Phase 2: Sequential triplet timing + heading interpolation
    # ------------------------------------------------------------------
    t1 = time.perf_counter()

    # Pre-build ship data cache in main process
    sd = ShipData(Path(ship_data_dir))
    logger.info("Ship data: %s", sd)

    # Heading interpolation delta for dH/dt (100ms)
    dt_deriv = np.timedelta64(100, "ms")

    # Walk sorted frames with sliding window of 3
    frame_bright_data = []  # One entry per frame with bright pixels

    try:
        from tqdm import tqdm
        triplet_iter = tqdm(range(n_frames), desc="Interpolating", unit="frame", disable=not progress)
    except ImportError:
        triplet_iter = range(n_frames)

    for i in triplet_iter:
        rec = all_records[i]
        if len(rec["bright_bi"]) == 0:
            continue

        # Build triplet
        prev_fd = _build_frame_data(all_records[i - 1]) if i > 0 else None
        curr_fd = _build_frame_data(rec)
        next_fd = _build_frame_data(all_records[i + 1]) if i < n_frames - 1 else None

        prev_proxy = FrameProxy(prev_fd) if prev_fd is not None else None
        curr_proxy = FrameProxy(curr_fd)
        next_proxy = FrameProxy(next_fd) if next_fd is not None else None

        interp = FrameInterpolator(prev_proxy, curr_proxy, next_proxy)
        times = interp.times  # per-radial timestamps

        # Interpolate heading at per-radial times using high-freq ship data
        headings = sd.interpolate(times, "heading")
        if headings is None:
            headings = interp.headings

        # Central-difference dH/dt with angular wrapping via sin/cos
        times_plus = times + dt_deriv
        times_minus = times - dt_deriv
        h_plus = sd.interpolate(times_plus, "heading")
        h_minus = sd.interpolate(times_minus, "heading")
        if h_plus is not None and h_minus is not None:
            # Angular difference using sin/cos to handle wrapping
            dh = np.rad2deg(
                np.arctan2(
                    np.sin(np.deg2rad(h_plus) - np.deg2rad(h_minus)),
                    np.cos(np.deg2rad(h_plus) - np.deg2rad(h_minus)),
                )
            )
            dt_s = 0.2  # 200ms total span
            dH_dt = dh / dt_s  # degrees/second
        else:
            dH_dt = np.zeros(len(times))

        # Interpolate lat/lon at per-radial times
        lats = sd.interpolate(times, "latitude")
        lons = sd.interpolate(times, "longitude")
        if lats is None:
            lats = interp.latitudes
        if lons is None:
            lons = interp.longitudes

        frame_bright_data.append({
            "theta": rec["theta_array"],
            "ground_range": rec["ground_range"],
            "headings": headings,
            "dH_dt": dH_dt,
            "lats": lats,
            "lons": lons,
            "bright_bi": rec["bright_bi"],
            "bright_ri": rec["bright_ri"],
        })

    elapsed_interp = time.perf_counter() - t1
    logger.info(
        "Interpolated %d frames with bright pixels in %.1fs",
        len(frame_bright_data), elapsed_interp,
    )

    if not frame_bright_data:
        logger.warning("No frames with bright pixels after interpolation")
        return {"best_offset": 0.0, "offsets": np.array([]), "metrics": np.array([])}

    # ------------------------------------------------------------------
    # Phase 3: Pre-compute base coordinates + derivatives
    # ------------------------------------------------------------------
    t2 = time.perf_counter()

    # Use mean latitude for equirectangular projection
    all_lats_mean = np.mean([np.mean(fd["lats"]) for fd in frame_bright_data])
    m_per_deg_lon = _DEG2M * np.cos(np.deg2rad(all_lats_mean))

    # Reference point for projection
    ref_lat = all_lats_mean
    ref_lon = np.mean([np.mean(fd["lons"]) for fd in frame_bright_data])

    # Pre-allocate flat arrays
    x_base_list = []
    y_base_list = []
    dx_ddt_list = []
    dy_ddt_list = []

    for fd in frame_bright_data:
        bi = fd["bright_bi"]
        ri = fd["bright_ri"]
        theta = fd["theta"]
        gr = fd["ground_range"]
        headings = fd["headings"]
        dH_dt = fd["dH_dt"]
        lats = fd["lats"]
        lons = fd["lons"]

        # Ship position in meters relative to reference
        ship_x = (lons[bi] - ref_lon) * m_per_deg_lon
        ship_y = (lats[bi] - ref_lat) * _DEG2M

        # Earth bearing for each bright pixel
        earth_bearing_deg = (theta[bi] + headings[bi]) % 360
        earth_bearing_rad = earth_bearing_deg * _DEG2RAD

        sin_b = np.sin(earth_bearing_rad)
        cos_b = np.cos(earth_bearing_rad)
        r = gr[ri]

        # Base earth coordinates at zero offset
        x = ship_x + r * sin_b
        y = ship_y + r * cos_b

        # Derivatives: how x,y change per second of timing offset
        # dH_dt is in degrees/second; convert to radians/second
        dH_dt_rad = dH_dt[bi] * _DEG2RAD
        dx = r * cos_b * dH_dt_rad
        dy = -r * sin_b * dH_dt_rad

        x_base_list.append(x)
        y_base_list.append(y)
        dx_ddt_list.append(dx)
        dy_ddt_list.append(dy)

    x_base = np.concatenate(x_base_list)
    y_base = np.concatenate(y_base_list)
    dx_ddt = np.concatenate(dx_ddt_list)
    dy_ddt = np.concatenate(dy_ddt_list)

    n_points = len(x_base)
    elapsed_precomp = time.perf_counter() - t2
    logger.info(
        "Pre-computed %d bright pixel positions in %.2fs",
        n_points, elapsed_precomp,
    )

    # ------------------------------------------------------------------
    # Phase 4: Vectorized sweep
    # ------------------------------------------------------------------
    t3 = time.perf_counter()

    offsets = np.arange(-sweep_range, sweep_range + sweep_step / 2, sweep_step)
    n_offsets = len(offsets)

    # Compute grid extent from zero-offset positions with padding
    x_pad = max(np.abs(dx_ddt).max() * sweep_range, grid_spacing * 10) if n_points > 0 else 100
    y_pad = max(np.abs(dy_ddt).max() * sweep_range, grid_spacing * 10) if n_points > 0 else 100
    x_min = x_base.min() - x_pad
    x_max = x_base.max() + x_pad
    y_min = y_base.min() - y_pad
    y_max = y_base.max() + y_pad

    n_gx = int(np.ceil((x_max - x_min) / grid_spacing))
    n_gy = int(np.ceil((y_max - y_min) / grid_spacing))
    grid_size = n_gx * n_gy

    logger.info(
        "Sweep: %d offsets x %d points, grid %d x %d (%.0fm spacing)",
        n_offsets, n_points, n_gx, n_gy, grid_spacing,
    )

    inv_spacing = 1.0 / grid_spacing
    metrics = np.empty(n_offsets, dtype=np.float64)

    try:
        from tqdm import tqdm
        sweep_iter = tqdm(range(n_offsets), desc="Sweeping", unit="offset", disable=not progress)
    except ImportError:
        sweep_iter = range(n_offsets)

    for k in sweep_iter:
        dt = offsets[k]
        x = x_base + dx_ddt * dt
        y = y_base + dy_ddt * dt

        xi = ((x - x_min) * inv_spacing).astype(np.int32)
        yi = ((y - y_min) * inv_spacing).astype(np.int32)

        valid = (xi >= 0) & (xi < n_gx) & (yi >= 0) & (yi < n_gy)
        linear_idx = yi[valid] * n_gx + xi[valid]

        counts = np.bincount(linear_idx, minlength=grid_size)
        metrics[k] = np.sum(counts.astype(np.float64) ** 2)  # L2 concentration

    elapsed_sweep = time.perf_counter() - t3
    logger.info("Sweep completed in %.1fs", elapsed_sweep)

    best_idx = int(np.argmax(metrics))
    best_offset = float(offsets[best_idx])

    # ------------------------------------------------------------------
    # Phase 5: Report + visualize
    # ------------------------------------------------------------------
    zero_idx = int(np.argmin(np.abs(offsets)))
    improvement = (metrics[best_idx] / metrics[zero_idx] - 1) * 100 if metrics[zero_idx] > 0 else 0

    print(f"\n{'='*60}")
    print("Hard return timing offset analysis")
    print(f"{'='*60}")
    print(f"  Frames:           {n_frames}")
    print(f"  Bright pixels:    {n_points} (>{threshold_pct:.0f}th percentile)")
    print(f"  Sweep range:      [{-sweep_range:+.3f}, {+sweep_range:+.3f}] s")
    print(f"  Sweep step:       {sweep_step:.3f} s ({n_offsets} offsets)")
    print(f"  Grid spacing:     {grid_spacing:.1f} m ({n_gx} x {n_gy})")
    print(f"  Best offset:      {best_offset:+.3f} s")
    print(f"  Metric at best:   {metrics[best_idx]:.0f}")
    print(f"  Metric at zero:   {metrics[zero_idx]:.0f}")
    print(f"  Improvement:      {improvement:+.1f}%")
    print(f"{'='*60}")

    result = {
        "best_offset": best_offset,
        "best_metric": float(metrics[best_idx]),
        "zero_metric": float(metrics[zero_idx]),
        "improvement_pct": improvement,
        "offsets": offsets,
        "metrics": metrics,
        "n_frames": n_frames,
        "n_points": n_points,
        "grid_spacing": grid_spacing,
        "n_gx": n_gx,
        "n_gy": n_gy,
        # Needed for density map plotting
        "x_base": x_base,
        "y_base": y_base,
        "dx_ddt": dx_ddt,
        "dy_ddt": dy_ddt,
        "x_min": x_min,
        "y_min": y_min,
    }

    if plot:
        _plot_results(result)

    return result


def _plot_results(result: dict) -> None:
    """Plot metric curve and side-by-side density maps."""
    import matplotlib.pyplot as plt

    offsets = result["offsets"]
    metrics = result["metrics"]
    best_offset = result["best_offset"]
    grid_spacing = result["grid_spacing"]
    n_gx = result["n_gx"]
    n_gy = result["n_gy"]
    x_base = result["x_base"]
    y_base = result["y_base"]
    dx_ddt = result["dx_ddt"]
    dy_ddt = result["dy_ddt"]
    x_min = result["x_min"]
    y_min = result["y_min"]
    inv_spacing = 1.0 / grid_spacing
    grid_size = n_gx * n_gy

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Metric curve
    ax = axes[0]
    ax.plot(offsets, metrics, "b-", linewidth=1)
    ax.axvline(best_offset, color="r", linestyle="--", linewidth=1, label=f"Best: {best_offset:+.3f} s")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Timing offset (s)")
    ax.set_ylabel("L2 concentration metric")
    ax.set_title("Concentration vs. timing offset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Density map at zero offset
    ax = axes[1]
    xi = ((x_base - x_min) * inv_spacing).astype(np.int32)
    yi = ((y_base - y_min) * inv_spacing).astype(np.int32)
    valid = (xi >= 0) & (xi < n_gx) & (yi >= 0) & (yi < n_gy)
    counts_zero = np.bincount(yi[valid] * n_gx + xi[valid], minlength=grid_size).reshape(n_gy, n_gx)
    vmax = max(1, counts_zero.max())
    density_zero = counts_zero.astype(np.float64)
    density_zero[density_zero == 0] = np.nan
    ax.imshow(
        density_zero, origin="lower", aspect="auto",
        vmin=0, vmax=vmax, cmap="hot",
    )
    ax.set_title("Density at zero offset")
    ax.set_xlabel("x grid index")
    ax.set_ylabel("y grid index")

    # 3. Density map at best offset
    ax = axes[2]
    x = x_base + dx_ddt * best_offset
    y = y_base + dy_ddt * best_offset
    xi = ((x - x_min) * inv_spacing).astype(np.int32)
    yi = ((y - y_min) * inv_spacing).astype(np.int32)
    valid = (xi >= 0) & (xi < n_gx) & (yi >= 0) & (yi < n_gy)
    counts_best = np.bincount(yi[valid] * n_gx + xi[valid], minlength=grid_size).reshape(n_gy, n_gx)
    density_best = counts_best.astype(np.float64)
    density_best[density_best == 0] = np.nan
    ax.imshow(
        density_best, origin="lower", aspect="auto",
        vmin=0, vmax=vmax, cmap="hot",
    )
    ax.set_title(f"Density at best offset ({best_offset:+.3f} s)")
    ax.set_xlabel("x grid index")
    ax.set_ylabel("y grid index")

    fig.suptitle(
        f"Hard return analysis: {result['n_frames']} frames, "
        f"{result['n_points']} bright pixels",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument(
        "--ship-data",
        type=str,
        required=True,
        help="Directory with instrument NetCDF files (from revelle CLI)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=97.0,
        help="Bright pixel percentile threshold (default: 97)",
    )
    parser.add_argument(
        "--sweep-range",
        type=float,
        default=2.0,
        help="Sweep range in +/- seconds (default: 2.0)",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.02,
        help="Sweep step in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=20.0,
        help="Grid cell size in meters (default: 20)",
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
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plot", action="store_true", default=True,
        help="Show result plots (default)",
    )
    plot_group.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Suppress plots",
    )


def add_subparser(subparsers) -> None:
    """Register the 'hard-returns' subcommand."""
    p = subparsers.add_parser(
        "hard-returns",
        help="Find timing offset that stabilizes hard returns",
        description=(
            "Sweep timing offsets to find the one that maximizes spatial "
            "concentration of bright pixels across frames.  Hard returns "
            "(land, platforms, buoys) should be stable in earth coordinates; "
            "the optimal offset corrects for heading interpolation timing errors."
        ),
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'hard-returns' command."""
    from wamos_tpw.config import Config

    config = Config(args.config) if args.config else Config()
    find_hard_return_offset(
        args.stime,
        args.etime,
        args.polar_path,
        args.ship_data,
        config=config,
        threshold_pct=args.threshold,
        sweep_range=args.sweep_range,
        sweep_step=args.sweep_step,
        grid_spacing=args.grid_spacing,
        workers=args.workers,
        progress=args.progress,
        plot=args.plot,
    )


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(
    _add_arguments, run,
    "Find timing offset that stabilizes hard returns",
)

if __name__ == "__main__":
    main()
