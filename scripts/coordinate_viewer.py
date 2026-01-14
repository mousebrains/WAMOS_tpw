#!/usr/bin/env python3
"""
Interactive viewer for WAMOS radar data in multiple coordinate systems.

Displays frame-by-frame intensity data with navigation controls and
multiple coordinate system views:
    1. Raw intensity (radial vs range bin)
    2. Ship-relative Cartesian (x=starboard, y=forward)
    3. Ship-relative polar (beam angle)
    4. Earth coordinates (longitude, latitude)

Features:
    - Next/Play/Previous navigation buttons
    - Interpolated ship position, heading, speed, and wind at each radial
    - Real-time coordinate transformation

Usage:
    python coordinate_viewer.py 20220405 20220406 /path/to/POLAR

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFrame  # noqa: E402
from wamos_tpw.args import add_time_range_arguments  # noqa: E402
from wamos_tpw.coordinates import CoordinateTransformer  # noqa: E402
from wamos_tpw.destreak import DestreakFrame  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402
from wamos_tpw.pps import WamosPPS  # noqa: E402
from wamos_tpw.shadow import ShadowConfig  # noqa: E402
from wamos_tpw.theta_calc import WamosTheta  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame_data(
    fn: str,
    destreak_config: dict | None = None,
    shadow_config: ShadowConfig | None = None,
) -> dict | None:
    """
    Load a single frame with all required data for coordinate transformation.

    Args:
        fn: Filename to load
        destreak_config: Optional destreak configuration
        shadow_config: Optional shadow detection configuration

    Returns:
        Dictionary with frame data, or None on error
    """
    try:
        polar_frame = PolarFrame(fn)
        frame = DestreakFrame(polar_frame, destreak_config)

        wpps = WamosPPS(frame)
        wtheta = WamosTheta(frame, shadow_config=shadow_config)

        # Extract metadata for coordinate transformer
        metadata = frame.metadata

        return {
            "filename": fn,
            "timestamp": wpps.timestamp,
            "n_bearings": frame.n_bearings,
            "n_ranges": frame.n_ranges,
            "intensity": frame.intensity,
            "wtheta": wtheta,
            "wpps": wpps,
            # Range data from frame
            "slant_ranges": frame.slant_ranges,
            "horizontal_ranges": frame.horizontal_ranges,
            "range_resolution": frame.range_resolution,
            "range_offset": frame.range_offset,
            "radar_height": frame.radar_height,
            # Navigation metadata
            "frame_lat": metadata.get("frame_lat", metadata.get("lat", 0.0)),
            "frame_lon": metadata.get("frame_lon", metadata.get("lon", 0.0)),
            "frame_gyroc": metadata.get("frame_gyroc", metadata.get("GYROC", 0.0)),
            "frame_ships": metadata.get("frame_ships", metadata.get("SHIPS", 0.0)),
            "frame_winds": metadata.get("frame_winds", metadata.get("WINDS", 0.0)),
            "frame_windr": metadata.get("frame_windr", metadata.get("WINDR", 0.0)),
            "frame_rpt": metadata.get("frame_rpt", metadata.get("RPT", 1.5)),
            "frame_windh": metadata.get("frame_windh", metadata.get("WINDH", 0.0)),
        }
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def print_progress(current: int, total: int, width: int = 40, prefix: str = "Progress") -> None:
    """Print a progress bar that updates in place."""
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - 1)
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()


class CoordinateViewer:
    """
    Interactive viewer for radar data in multiple coordinate systems.

    Displays three panels:
        1. Raw intensity (theta vs range in meters)
        2. Ship Polar (beam angle vs range in meters)
        3. Earth coordinates (longitude vs latitude)
    """

    def __init__(
        self,
        results: list[dict],
        transformer: CoordinateTransformer,
        range_start: int = 0,
        n_ranges: int = 100,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Initialize the viewer.

        Args:
            results: List of frame data dictionaries
            transformer: CoordinateTransformer for the frames
            range_start: Starting range bin for display
            n_ranges: Number of range bins to display
            vmin: Minimum intensity for colormap
            vmax: Maximum intensity for colormap
        """
        self.results = results
        self.transformer = transformer
        self.range_start = range_start
        self.n_ranges = n_ranges
        self.vmin = vmin
        self.vmax = vmax

        self.current_idx = 0
        self.n_frames = len(results)
        self.playing = False
        self.timer: object | None = None

        # Set up the figure
        self._setup_figure()
        self._update_display()

    def _setup_figure(self) -> None:
        """Set up the matplotlib figure with three subplots and navigation."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.fig = plt.figure(figsize=(18, 6))

        # Create 1x3 grid of subplots
        self.ax_raw = self.fig.add_subplot(1, 3, 1)
        self.ax_ship_polar = self.fig.add_subplot(1, 3, 2, projection="polar")
        self.ax_earth = self.fig.add_subplot(1, 3, 3)

        # Add navigation buttons
        ax_prev = self.fig.add_axes((0.3, 0.02, 0.1, 0.04))
        ax_play = self.fig.add_axes((0.45, 0.02, 0.1, 0.04))
        ax_next = self.fig.add_axes((0.6, 0.02, 0.1, 0.04))

        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, "Next")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play)
        self.btn_next.on_clicked(self._on_next)

        # Add info text area
        self.info_text = self.fig.text(0.02, 0.02, "", fontsize=9, fontfamily="monospace")

        self.fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3, wspace=0.25)

    def _on_prev(self, event: object) -> None:
        """Handle Previous button click."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_display()

    def _on_next(self, event: object) -> None:
        """Handle Next button click."""
        if self.current_idx < self.n_frames - 1:
            self.current_idx += 1
            self._update_display()

    def _on_play(self, event: object) -> None:
        """Handle Play/Stop button click."""
        import matplotlib.pyplot as plt

        if self.playing:
            self.playing = False
            self.btn_play.label.set_text("Play")
            if self.timer is not None:
                timer = self.timer
                if hasattr(timer, "stop"):
                    timer.stop()  # type: ignore[union-attr]
                self.timer = None
        else:
            self.playing = True
            self.btn_play.label.set_text("Stop")
            self.timer = self.fig.canvas.new_timer(interval=500)
            if self.timer is not None:
                timer = self.timer
                if hasattr(timer, "add_callback"):
                    timer.add_callback(self._on_timer)  # type: ignore[union-attr]
                if hasattr(timer, "start"):
                    timer.start()  # type: ignore[union-attr]
        plt.draw()

    def _on_timer(self) -> None:
        """Handle timer tick for animation."""
        if not self.playing:
            return

        if self.current_idx < self.n_frames - 1:
            self.current_idx += 1
            self._update_display()
        else:
            # Stop at end
            self._on_play(None)

    def _update_display(self) -> None:
        """Update all three plots for the current frame."""
        import matplotlib.pyplot as plt

        frame = self.results[self.current_idx]
        intensity = frame["intensity"]
        wtheta = frame["wtheta"]

        # Clip to available range bins
        actual_end = min(self.range_start + self.n_ranges, intensity.shape[1])
        intensity_slice = intensity[:, self.range_start : actual_end]

        # Get range arrays from frame (slant and horizontal)
        slant_ranges = frame["slant_ranges"][self.range_start : actual_end]
        horizontal_ranges = frame["horizontal_ranges"][self.range_start : actual_end]

        # Get navigation data
        nav = self.transformer.get_frame_navigation(self.current_idx)
        meta = self.transformer.get_frame_metadata(self.current_idx)

        # Get coordinates (use horizontal ranges for Earth coordinates)
        earth_lon, earth_lat = self.transformer.get_earth_coords(
            self.current_idx, horizontal_ranges
        )

        # Clear all axes
        self.ax_raw.clear()
        self.ax_ship_polar.clear()
        self.ax_earth.clear()

        # Determine color limits
        vmin = self.vmin if self.vmin is not None else np.percentile(intensity_slice, 2)
        vmax = self.vmax if self.vmax is not None else np.percentile(intensity_slice, 98)

        # Maximum range for ring calculations (use horizontal for Earth, slant for radar display)
        max_slant_range = float(np.max(slant_ranges))
        max_horizontal_range = float(np.max(horizontal_ranges))

        # 1. Raw intensity plot (theta vs slant range in meters)
        # Sort by theta for proper pcolormesh display
        theta_values = wtheta.theta
        sort_idx = np.argsort(theta_values)
        theta_sorted = theta_values[sort_idx]
        intensity_sorted = intensity_slice[sort_idx, :]

        self.ax_raw.pcolormesh(
            slant_ranges,
            theta_sorted,
            intensity_sorted,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self.ax_raw.set_xlabel("Slant Range (m)")
        self.ax_raw.set_ylabel("Theta (°)")
        self.ax_raw.set_title("Raw Intensity")

        # 2. Ship Polar plot (beam angle vs slant range) with 500m range rings
        beam_rad = np.deg2rad(nav.beam_angle)
        r_mesh, theta_mesh = np.meshgrid(slant_ranges, beam_rad)
        self.ax_ship_polar.pcolormesh(
            theta_mesh,
            r_mesh,
            intensity_slice,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self.ax_ship_polar.set_theta_zero_location("N")  # type: ignore[attr-defined]
        self.ax_ship_polar.set_theta_direction(-1)  # type: ignore[attr-defined]
        self.ax_ship_polar.set_title("Ship Polar (Beam Angle)")

        # Add 500m range rings (slant range)
        ring_interval = 500.0
        ring_radii = np.arange(ring_interval, max_slant_range + ring_interval, ring_interval)
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        for r in ring_radii:
            self.ax_ship_polar.plot(
                theta_circle, np.full_like(theta_circle, r), "w--", linewidth=0.5, alpha=0.5
            )
        # Set radial ticks at 500m intervals
        self.ax_ship_polar.set_rticks(ring_radii)  # type: ignore[attr-defined]

        # 3. Earth coordinates plot (lon, lat) with 500m range rings
        self.ax_earth.pcolormesh(
            earth_lon,
            earth_lat,
            intensity_slice,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        self.ax_earth.set_xlabel("Longitude")
        self.ax_earth.set_ylabel("Latitude")
        self.ax_earth.set_title("Earth Coordinates")
        self.ax_earth.set_aspect("equal", adjustable="box")

        # Mark ship position
        ship_lon = meta["lon"]
        ship_lat = meta["lat"]
        self.ax_earth.plot(ship_lon, ship_lat, "r^", markersize=10, label="Ship")

        # Add 500m range rings in Earth coordinates (use horizontal range)
        # Convert meters to degrees (approximate at this latitude)
        meters_per_deg_lat = 111320.0  # meters per degree latitude
        meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(ship_lat))

        horizontal_ring_radii = np.arange(
            ring_interval, max_horizontal_range + ring_interval, ring_interval
        )
        for r in horizontal_ring_radii:
            # Draw circle at radius r meters (horizontal distance)
            circle_theta = np.linspace(0, 2 * np.pi, 100)
            circle_lon = ship_lon + (r * np.cos(circle_theta)) / meters_per_deg_lon
            circle_lat = ship_lat + (r * np.sin(circle_theta)) / meters_per_deg_lat
            self.ax_earth.plot(circle_lon, circle_lat, "w--", linewidth=0.5, alpha=0.5)

        # Update info text
        ts_str = str(meta["timestamp"])[:26] if meta["timestamp"] is not None else "N/A"
        radar_height = frame["radar_height"]
        info = (
            f"Frame {self.current_idx + 1}/{self.n_frames}  |  {ts_str}\n"
            f"Pos: {meta['lat']:.4f}°N, {meta['lon']:.4f}°E  |  "
            f"Hdg: {meta['heading']:.1f}°  |  Speed: {meta['ship_speed']:.1f} m/s\n"
            f"Wind: {meta['wind_speed']:.1f} m/s from {meta['wind_dir']:.1f}°  |  "
            f"Radar height: {radar_height:.1f} m"
        )
        self.info_text.set_text(info)

        # Update figure title
        self.fig.suptitle(
            f"WAMOS Coordinate Viewer - Frame {self.current_idx + 1}/{self.n_frames}",
            fontsize=12,
        )

        plt.draw()

    def show(self) -> None:
        """Display the viewer."""
        import matplotlib.pyplot as plt

        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for WAMOS radar data in multiple coordinate systems"
    )

    add_time_range_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker threads (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=21,
        help="Starting range bin (default: 21, after timing bins)",
    )
    parser.add_argument(
        "--n-ranges",
        type=int,
        default=491,
        help="Number of range bins to display (default: 491, full range from bin 21)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum intensity for colormap",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum intensity for colormap",
    )
    parser.add_argument(
        "--destreak-min-length",
        type=int,
        default=4,
        help="Minimum consecutive streak length (default: 4)",
    )
    parser.add_argument(
        "--destreak-k-sigma",
        type=float,
        default=5.0,
        help="MAD threshold multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--destreak-neighbor-size",
        type=int,
        default=5,
        help="Number of neighbors for local statistics (default: 5)",
    )

    # Shadow detection arguments
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Enable shadow region detection for theta adjustment",
    )
    parser.add_argument(
        "--shadow-search",
        type=str,
        default="140-220",
        help="Theta range to search for shadow region (default: 140-220)",
    )
    parser.add_argument(
        "--shadow-bin-size",
        type=float,
        default=0.5,
        help="Theta bin size in degrees for shadow detection (default: 0.5)",
    )
    parser.add_argument(
        "--shadow-min-width",
        type=float,
        default=50.0,
        help="Minimum shadow width in degrees to accept (default: 50.0)",
    )

    # Theta adjustment arguments
    parser.add_argument(
        "--adjust",
        type=str,
        choices=["none", "shift", "shift_scale"],
        default="none",
        help="Theta adjustment mode: none, shift, or shift_scale (default: none)",
    )

    args = parser.parse_args()
    setup_logging(args)

    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        logger.info("Loading %d files with %d workers", n_files, args.workers or os.cpu_count())

        destreak_config = {
            "min_length": args.destreak_min_length,
            "k_sigma": args.destreak_k_sigma,
            "neighbor_size": args.destreak_neighbor_size,
        }

        # Build shadow config if shadow detection or adjustment is requested
        shadow_config: ShadowConfig | None = None
        if args.shadow or args.adjust != "none":
            # Parse shadow search range
            try:
                lo, hi = args.shadow_search.strip().split("-")
                search_min = float(lo)
                search_max = float(hi)
            except ValueError:
                logger.error("Invalid shadow-search format: %s", args.shadow_search)
                return 1

            shadow_config = ShadowConfig(
                search_min=search_min,
                search_max=search_max,
                bin_size=args.shadow_bin_size,
                min_width=args.shadow_min_width,
            )

        # Load frames in parallel
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(load_frame_data, fn, destreak_config, shadow_config): fn
                for fn in files
            }

            for i, future in enumerate(as_completed(futures)):
                print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)

        n_loaded = len(results)
        print(f"Successfully loaded {n_loaded} of {n_files} frames")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        # Sort by timestamp
        def sort_key(x: dict) -> np.datetime64:
            ts = x["timestamp"]
            return ts if ts is not None else np.datetime64(0, "ns")

        results.sort(key=sort_key)

        # Update PPS with adjacent frame context
        print("Updating timing with multi-frame context...")
        for i, curr in enumerate(results):
            prev = results[i - 1] if i > 0 else None
            nxt = results[i + 1] if i < len(results) - 1 else None
            curr["wpps"].update(
                prev["wpps"] if prev else None,
                nxt["wpps"] if nxt else None,
            )

        # Shadow detection and theta adjustment
        if args.shadow or args.adjust != "none":
            # Shadow detection was already performed during frame loading
            # Collect the detected edges
            print("Collecting shadow regions...")
            leading_edges: list[float] = []
            trailing_edges: list[float] = []

            for r in results:
                wtheta = r["wtheta"]
                shadow = wtheta.shadow_region

                if shadow.is_valid:
                    leading_edges.append(shadow.leading)  # type: ignore[arg-type]
                    trailing_edges.append(shadow.trailing)  # type: ignore[arg-type]

            n_detected = len(leading_edges)
            print(f"  Detected shadow in {n_detected} of {len(results)} frames")

            if leading_edges:
                print(
                    f"  Leading edge: {np.median(leading_edges):.1f}° "
                    f"(range: {np.min(leading_edges):.1f}° - {np.max(leading_edges):.1f}°)"
                )
            if trailing_edges:
                print(
                    f"  Trailing edge: {np.median(trailing_edges):.1f}° "
                    f"(range: {np.min(trailing_edges):.1f}° - {np.max(trailing_edges):.1f}°)"
                )

            # Apply theta adjustment if requested
            if args.adjust != "none" and leading_edges and trailing_edges:
                median_leading = float(np.median(leading_edges))
                median_trailing = float(np.median(trailing_edges))

                print(f"\nApplying theta adjustment (mode: {args.adjust})...")
                print(f"  Target leading edge:  {median_leading:.2f}°")
                print(f"  Target trailing edge: {median_trailing:.2f}°")

                n_adjusted = 0
                shifts: list[float] = []
                scales: list[float] = []

                for r in results:
                    wtheta = r["wtheta"]

                    if wtheta.has_shadow:
                        wtheta.adjust_to_shadow(
                            target_leading=median_leading,
                            target_trailing=median_trailing,
                            mode=args.adjust,
                        )
                        shifts.append(wtheta.shift)
                        scales.append(wtheta.scale)
                        n_adjusted += 1

                print(f"  Adjusted {n_adjusted} frames")
                if shifts:
                    print(f"  Shift: mean={np.mean(shifts):.3f}°, std={np.std(shifts):.3f}°")
                if scales and args.adjust == "shift_scale":
                    print(f"  Scale: mean={np.mean(scales):.6f}, std={np.std(scales):.6f}")

        # Create coordinate transformer
        print("Creating coordinate transformer...")
        transformer = CoordinateTransformer(results)

        # Create and show viewer
        print("Launching viewer...")
        viewer = CoordinateViewer(
            results=results,
            transformer=transformer,
            range_start=args.range_start,
            n_ranges=args.n_ranges,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        viewer.show()

        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
