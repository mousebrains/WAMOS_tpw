#!/usr/bin/env python3
"""
Diagnostic: compare pipeline stages side-by-side with projected output.

Shows 5 panels: raw intensity, destreaked, deramped, dewinded, projected.
First 4 are polar plots; the 5th is the earth-referenced grid projection.

Usage:
    python scripts/projection_check.py 20220405T0400 20220405T040001 ~/Desktop/WAMOS/POLAR
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile
from wamos_tpw.config import Config
from wamos_tpw.filenames import add_common_arguments
from wamos_tpw.frame_pipeline import FramePipeline
from wamos_tpw.interpolator import FrameInterpolator


def polar_mesh(intensity, theta_deg, range_m):
    """Build pcolormesh arrays for a polar plot sorted by theta."""
    order = np.argsort(theta_deg)
    theta_sorted = np.deg2rad(theta_deg[order])
    intensity_sorted = intensity[order, :]
    # Close the circle
    theta_closed = np.append(theta_sorted, theta_sorted[0] + 2 * np.pi)
    intensity_closed = np.vstack([intensity_sorted, intensity_sorted[0:1, :]])
    t_mesh, r_mesh = np.meshgrid(theta_closed, range_m, indexing="ij")
    return t_mesh, r_mesh, intensity_closed


def project_to_grid(intensity, theta_deg, headings, ground_range, ship_x, ship_y,
                    grid_spacing, x_min, y_min, n_x, n_y):
    """Equirectangular projection onto a north-up grid."""
    earth_bearing_rad = np.deg2rad((theta_deg + headings) % 360)
    sin_b = np.sin(earth_bearing_rad)
    cos_b = np.cos(earth_bearing_rad)

    inv_sp = 1.0 / grid_spacing
    x_idx = ((np.outer(sin_b, ground_range) + ship_x[:, None] - x_min) * inv_sp).astype(np.int32).ravel()
    y_idx = ((np.outer(cos_b, ground_range) + ship_y[:, None] - y_min) * inv_sp).astype(np.int32).ravel()
    vals = intensity.ravel()

    valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y) & ~np.isnan(vals)
    if not np.any(valid):
        return np.full((n_y, n_x), np.nan)

    linear = y_idx[valid] * n_x + x_idx[valid]
    grid_size = n_x * n_y
    s = np.bincount(linear, weights=vals[valid], minlength=grid_size).reshape(n_y, n_x)
    c = np.bincount(linear, minlength=grid_size).reshape(n_y, n_x)
    with np.errstate(invalid="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Projection diagnostic viewer")
    add_common_arguments(parser)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--frame", type=int, default=0, help="Frame index within first file")
    args = parser.parse_args()

    config = Config(args.config) if args.config else Config()
    filenames = Filenames(args.stime, args.etime, args.polar_path)
    files = list(filenames)
    if not files:
        print("No files found"); return 1

    # Load two consecutive files for interpolation
    pf0 = PolarFile(files[0], config=config)
    frame0 = pf0.frame(args.frame)
    fp0 = FramePipeline(frame0, config=config, qSave=True)

    fp1 = None
    if len(files) > 1:
        pf1 = PolarFile(files[1], config=config)
        fp1 = FramePipeline(pf1.frame(0), config=config)

    # Run interpolator
    interp = FrameInterpolator(None, fp0, fp1)

    # Collect pipeline stages
    raw = (frame0.intensity[:, :fp0.n_distances] & 0x0FFF).astype(np.float32)
    destreaked = fp0.destreak.intensity.astype(np.float32)
    deramped = fp0.deramp.intensity.astype(np.float32)
    dewinded = fp0.final_intensity  # processed (deramped + dewinded)
    theta_deg = fp0.theta_array
    ground_range = fp0.ground_range

    # Build projection grid
    latitudes = interp.latitudes
    longitudes = interp.longitudes
    headings = interp.headings

    ref_lat = float(np.mean(latitudes))
    ref_lon = float(np.mean(longitudes))
    DEG2M = 111_319.5
    m_per_deg_lon = DEG2M * np.cos(np.deg2rad(ref_lat))
    ship_x = (longitudes - ref_lon) * m_per_deg_lon
    ship_y = (latitudes - ref_lat) * DEG2M

    max_range = float(ground_range[-1]) * 1.1
    range_res = float(ground_range[1] - ground_range[0]) if len(ground_range) > 1 else 10.0
    angular_width = float(ground_range[-1]) * 2 * np.pi / fp0.n_bearings
    grid_spacing = max(range_res, angular_width)

    x_min = ship_x.min() - max_range
    x_max = ship_x.max() + max_range
    y_min = ship_y.min() - max_range
    y_max = ship_y.max() + max_range
    n_x = int(np.ceil((x_max - x_min) / grid_spacing))
    n_y = int(np.ceil((y_max - y_min) / grid_spacing))

    projected = project_to_grid(
        dewinded, theta_deg, headings, ground_range,
        ship_x, ship_y, grid_spacing, x_min, y_min, n_x, n_y)

    # --- Plot ---
    fig, axes = plt.subplots(1, 5, figsize=(24, 5),
                             subplot_kw={"projection": "polar"})

    stages = [
        ("Raw (12-bit)", raw),
        ("Destreaked", destreaked),
        ("Deramped", deramped),
        ("Dewinded", dewinded),
    ]

    for ax, (title, data) in zip(axes[:4], stages):
        vmin, vmax = np.nanpercentile(data, [1, 99])
        t_mesh, r_mesh, d_closed = polar_mesh(data, theta_deg, ground_range)
        ax.pcolormesh(t_mesh, r_mesh, d_closed, shading="auto", cmap="viridis",
                      vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    # Projected panel — replace polar with Cartesian
    axes[4].remove()
    ax_proj = fig.add_subplot(1, 5, 5)
    x_edges = np.linspace(x_min, x_min + n_x * grid_spacing, n_x + 1)
    y_edges = np.linspace(y_min, y_min + n_y * grid_spacing, n_y + 1)
    x_center = (x_edges[0] + x_edges[-1]) / 2
    y_center = (y_edges[0] + y_edges[-1]) / 2
    extent = [(x_edges[0] - x_center), (x_edges[-1] - x_center),
              (y_edges[0] - y_center), (y_edges[-1] - y_center)]
    vmin, vmax = np.nanpercentile(projected, [1, 99])
    ax_proj.imshow(projected, origin="lower", extent=extent, aspect="equal",
                   cmap="viridis", vmin=vmin, vmax=vmax)
    ax_proj.set_title("Projected", fontsize=10)
    ax_proj.set_xlabel("East (m)")
    ax_proj.set_ylabel("North (m)")

    ts = np.datetime_as_string(fp0.metadata.timestamp, unit="s")
    fig.suptitle(f"{files[0]}  frame={args.frame}  {ts}\n"
                 f"method={interp.method}  timing={interp.timing_method}  "
                 f"heading={fp0.metadata.heading:.1f}°",
                 fontsize=11)
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
