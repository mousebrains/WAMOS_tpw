#! /usr/bin/env python3
#
# Frame interpolation/extrapolation for per-radial metadata
#
# Uses PPS pulses from adjacent frames to calculate accurate timestamps,
# falling back to linear interpolation when PPS is unavailable.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.frame_pipeline import FramePipeline


logger = logging.getLogger(__name__)


class FrameInterpolator:
    """
    Compute interpolated/extrapolated per-radial metadata for a frame.

    Uses a triplet approach (previous, current, next) for:
    - Timestamps: Uses PPS pulses as anchors when available, otherwise linear
    - Position: Interpolates/extrapolates latitude and longitude

    PPS-based timing:
    - PPS pulses occur at whole seconds (GPS-synchronized)
    - Uses PPS from prev, current, and next frames to build timing model
    - Falls back to linear model (start_time + repeat_time) when no PPS

    Position interpolation:
    - If current and next are within time tolerance: forward interpolation
    - Else if previous and current are within tolerance: backward extrapolation
    """

    def __init__(
        self,
        prev: FramePipeline | None,
        current: FramePipeline,
        next_frame: FramePipeline | None,
        tolerance: float = 1.2,
    ) -> None:
        """
        Initialize frame interpolator.

        Args:
            prev: Previous frame (can be None for first frame)
            current: Current frame to compute per-radial values for
            next_frame: Next frame (can be None for last frame)
            tolerance: Multiplier for repeat_time to accept pair (1.2 = 20% margin)

        Raises:
            ValueError: If neither interpolation nor extrapolation is possible
        """
        self._prev = prev
        self._current = current
        self._next = next_frame
        self._tolerance = tolerance
        self._method: str = "none"
        self._timing_method: str = "linear"  # or "pps"

        meta_curr = current.metadata
        repeat_time = meta_curr.repeat_time or 1.5
        max_dt = repeat_time * tolerance

        self._repeat_time = repeat_time
        self._n_radials = current.n_bearings

        # Try forward interpolation first (current + next)
        can_interpolate = False
        if next_frame is not None:
            dt_forward = self._time_delta_seconds(meta_curr.timestamp, next_frame.metadata.timestamp)
            if dt_forward > 0 and dt_forward <= max_dt:
                can_interpolate = True
                self._dt = dt_forward
                self._method = "interpolate"

        # Fall back to backward extrapolation (prev + current)
        can_extrapolate = False
        if not can_interpolate and prev is not None:
            dt_backward = self._time_delta_seconds(prev.metadata.timestamp, meta_curr.timestamp)
            if dt_backward > 0 and dt_backward <= max_dt:
                can_extrapolate = True
                self._dt = dt_backward
                self._method = "extrapolate"

        if not can_interpolate and not can_extrapolate:
            raise ValueError(
                f"Cannot interpolate or extrapolate for frame at {meta_curr.timestamp}: "
                f"no valid adjacent frame within tolerance {max_dt:.2f}s"
            )

        # Compute timestamps using PPS or linear fallback
        self._compute_timestamps()

        # Compute positions based on method
        if self._method == "interpolate":
            self._compute_interpolated_positions()
        else:
            self._compute_extrapolated_positions()

    def _time_delta_seconds(self, t0: np.datetime64, t1: np.datetime64) -> float:
        """Return time difference in seconds."""
        return (t1 - t0) / np.timedelta64(1, 's')

    def _get_pps_anchors(self, frame: FramePipeline) -> list[tuple[int, np.datetime64]]:
        """
        Get PPS timing anchors from a frame.

        Returns list of (radial_index, whole_second_timestamp) tuples.
        """
        if frame is None or frame.pps is None:
            return []

        pps = frame.pps
        if not pps:
            return []

        pps_indices = pps.indices
        if len(pps_indices) == 0:
            return []

        meta = frame.metadata
        start_time = meta.timestamp
        repeat_time = meta.repeat_time or 1.5
        n_radials = frame.n_bearings

        anchors = []
        for idx in pps_indices:
            # Estimate time at this radial using linear model
            fraction = idx / n_radials
            estimated_ns = start_time + np.timedelta64(int(fraction * repeat_time * 1e9), 'ns')

            # Round to nearest whole second (PPS occurs at whole seconds)
            # Convert to seconds since epoch, round, convert back
            estimated_s = estimated_ns.astype('datetime64[s]')
            # Check if we should round up or down
            remainder_ns = (estimated_ns - estimated_s) / np.timedelta64(1, 'ns')
            if remainder_ns >= 0.5e9:
                whole_second = estimated_s + np.timedelta64(1, 's')
            else:
                whole_second = estimated_s

            anchors.append((idx, whole_second))

        return anchors

    def _compute_timestamps(self) -> None:
        """Compute per-radial timestamps using PPS anchors or linear fallback."""
        meta_curr = self._current.metadata
        n_radials = self._n_radials

        # Collect PPS anchors from all available frames
        all_anchors = []

        # Get anchors from previous frame (offset indices to be relative to current)
        if self._prev is not None:
            prev_anchors = self._get_pps_anchors(self._prev)
            # Previous frame's radials come before current frame
            # Offset by -n_radials_prev
            n_prev = self._prev.n_bearings
            for idx, ts in prev_anchors:
                all_anchors.append((idx - n_prev, ts))

        # Get anchors from current frame
        curr_anchors = self._get_pps_anchors(self._current)
        all_anchors.extend(curr_anchors)

        # Get anchors from next frame (offset indices)
        if self._next is not None:
            next_anchors = self._get_pps_anchors(self._next)
            for idx, ts in next_anchors:
                all_anchors.append((idx + n_radials, ts))

        if len(all_anchors) >= 2:
            # Use PPS anchors to build timing model
            self._timing_method = "pps"
            self._times = self._interpolate_from_pps(all_anchors, n_radials)
        elif len(all_anchors) == 1:
            # Single PPS anchor - use it with repeat_time for rate
            self._timing_method = "pps"
            idx, ts = all_anchors[0]
            # Rate: radials per nanosecond
            rate_ns = self._repeat_time * 1e9 / n_radials
            radial_indices = np.arange(n_radials)
            offsets_ns = (radial_indices - idx) * rate_ns
            self._times = ts + offsets_ns.astype('timedelta64[ns]')
        else:
            # No PPS anchors - fall back to linear model
            self._timing_method = "linear"
            self._times = self._compute_linear_timestamps()

    def _interpolate_from_pps(
        self, anchors: list[tuple[int, np.datetime64]], n_radials: int
    ) -> np.ndarray:
        """
        Interpolate timestamps from PPS anchors.

        Uses linear interpolation between anchor points.
        """
        # Sort anchors by index
        anchors = sorted(anchors, key=lambda x: x[0])

        # Convert to arrays for interpolation
        anchor_indices = np.array([a[0] for a in anchors])
        anchor_times_ns = np.array([
            (a[1] - np.datetime64(0, 'ns')) / np.timedelta64(1, 'ns')
            for a in anchors
        ])

        # Interpolate for all radial indices
        radial_indices = np.arange(n_radials)
        interpolated_ns = np.interp(radial_indices, anchor_indices, anchor_times_ns)

        # Convert back to datetime64
        return np.datetime64(0, 'ns') + interpolated_ns.astype('timedelta64[ns]')

    def _compute_linear_timestamps(self) -> np.ndarray:
        """Compute timestamps using linear model from start_time and repeat_time."""
        meta = self._current.metadata
        n_radials = self._n_radials
        radial_fractions = np.linspace(0, 1, n_radials, endpoint=False)
        return meta.timestamp + (radial_fractions * self._repeat_time * 1e9).astype('timedelta64[ns]')

    def _compute_interpolated_positions(self) -> None:
        """Compute interpolated positions and headings using current and next frames."""
        meta_curr = self._current.metadata
        meta_next = self._next.metadata
        n_radials = self._n_radials

        radial_fractions = np.linspace(0, 1, n_radials, endpoint=False)

        # Position scale: how much of the inter-frame motion applies to one frame
        position_scale = self._repeat_time / self._dt if self._dt > 0 else 1.0

        # Interpolate position
        lat0 = meta_curr.latitude or 0.0
        lon0 = meta_curr.longitude or 0.0
        lat1 = meta_next.latitude or 0.0
        lon1 = meta_next.longitude or 0.0

        self._latitudes = lat0 + radial_fractions * (lat1 - lat0) * position_scale
        self._longitudes = self._interpolate_longitude(lon0, lon1, radial_fractions, position_scale)

        # Interpolate heading (circular)
        hdg0 = meta_curr.heading or 0.0
        hdg1 = meta_next.heading or 0.0
        self._headings = self._interpolate_angle(hdg0, hdg1, radial_fractions, position_scale)

    def _compute_extrapolated_positions(self) -> None:
        """Compute extrapolated positions and headings using previous and current frames."""
        meta_prev = self._prev.metadata
        meta_curr = self._current.metadata
        n_radials = self._n_radials

        radial_fractions = np.linspace(0, 1, n_radials, endpoint=False)

        # Rate of change from prev to current, then project forward
        rate_scale = self._repeat_time / self._dt if self._dt > 0 else 1.0

        # Extrapolate position using rate from prev->current
        lat_prev = meta_prev.latitude or 0.0
        lon_prev = meta_prev.longitude or 0.0
        lat_curr = meta_curr.latitude or 0.0
        lon_curr = meta_curr.longitude or 0.0

        lat_rate = lat_curr - lat_prev
        self._latitudes = lat_curr + radial_fractions * lat_rate * rate_scale

        # Handle longitude carefully for rate calculation
        lon_diff = lon_curr - lon_prev
        if abs(lon_diff) > 180:
            lon_diff = lon_diff - 360 if lon_diff > 0 else lon_diff + 360
        lon_rate = lon_diff
        self._longitudes = lon_curr + radial_fractions * lon_rate * rate_scale
        self._longitudes = ((self._longitudes + 180) % 360) - 180

        # Extrapolate heading (circular)
        hdg_prev = meta_prev.heading or 0.0
        hdg_curr = meta_curr.heading or 0.0
        hdg_diff = hdg_curr - hdg_prev
        if abs(hdg_diff) > 180:
            hdg_diff = hdg_diff - 360 if hdg_diff > 0 else hdg_diff + 360
        self._headings = (hdg_curr + radial_fractions * hdg_diff * rate_scale) % 360

    def _interpolate_longitude(
        self, lon0: float, lon1: float, fractions: np.ndarray, scale: float
    ) -> np.ndarray:
        """Interpolate longitude handling date line wrap-around."""
        lon_diff = lon1 - lon0
        if abs(lon_diff) > 180:
            lon_diff = lon_diff - 360 if lon_diff > 0 else lon_diff + 360
        result = lon0 + fractions * lon_diff * scale
        return ((result + 180) % 360) - 180

    def _interpolate_angle(
        self, angle0: float, angle1: float, fractions: np.ndarray, scale: float
    ) -> np.ndarray:
        """Interpolate angles (0-360) handling wrap-around."""
        angle_diff = angle1 - angle0
        if abs(angle_diff) > 180:
            angle_diff = angle_diff - 360 if angle_diff > 0 else angle_diff + 360
        result = angle0 + fractions * angle_diff * scale
        return result % 360

    @property
    def method(self) -> str:
        """Return the position method used: 'interpolate' or 'extrapolate'."""
        return self._method

    @property
    def timing_method(self) -> str:
        """Return the timing method used: 'pps' or 'linear'."""
        return self._timing_method

    @property
    def frame(self) -> FramePipeline:
        """Return the current frame."""
        return self._current

    @property
    def prev_frame(self) -> FramePipeline | None:
        """Return the previous frame (may be None)."""
        return self._prev

    @property
    def next_frame(self) -> FramePipeline | None:
        """Return the next frame (may be None)."""
        return self._next

    @property
    def times(self) -> np.ndarray:
        """Return per-radial timestamps."""
        return self._times

    @property
    def latitudes(self) -> np.ndarray:
        """Return interpolated/extrapolated latitudes for each radial."""
        return self._latitudes

    @property
    def longitudes(self) -> np.ndarray:
        """Return interpolated/extrapolated longitudes for each radial."""
        return self._longitudes

    @property
    def headings(self) -> np.ndarray:
        """Return interpolated/extrapolated ship headings for each radial (degrees, 0-360)."""
        return self._headings

    @property
    def time_delta(self) -> float:
        """Return time delta used for position interpolation/extrapolation in seconds."""
        return self._dt

    @property
    def repeat_time(self) -> float:
        """Return the current frame's repeat time in seconds."""
        return self._repeat_time

    def __repr__(self) -> str:
        return (
            f"<FrameInterpolator position={self._method} timing={self._timing_method} "
            f"dt={self._dt:.2f}s n_radials={len(self._times)}>"
        )


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", nargs="+", help="Polar file(s) to process")
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument("--tolerance", "-t", type=float, default=1.2,
                        help="Time tolerance multiplier (default: 1.2)")


def add_subparser(subparsers) -> None:
    """Register the 'interpolator' subcommand."""
    p = subparsers.add_parser(
        "interpolator",
        help="Test frame interpolation/extrapolation",
        description="Test per-radial metadata interpolation between frames",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'interpolator' command."""
    from wamos_tpw.config import Config
    from wamos_tpw.file_pipeline import FilePipeline

    config = Config(args.config) if args.config else Config()

    # Collect all frames from all files
    all_frames: list[FramePipeline] = []
    for filename in args.filename:
        fp = FilePipeline(filename, config=config)
        if fp:
            all_frames.extend(fp.frames)

    if not all_frames:
        logging.warning("No frames found in %s", args.filename)
        return

    logging.info("Loaded %d frames from %d file(s)", len(all_frames), len(args.filename))

    # Test interpolation with triplet approach
    n_interpolated = 0
    n_extrapolated = 0
    n_pps = 0
    n_linear = 0
    n_skipped = 0

    for i in range(len(all_frames)):
        prev_frame = all_frames[i - 1] if i > 0 else None
        current_frame = all_frames[i]
        next_frame = all_frames[i + 1] if i + 1 < len(all_frames) else None

        try:
            interp = FrameInterpolator(
                prev_frame, current_frame, next_frame,
                tolerance=args.tolerance,
            )
            if interp.method == "interpolate":
                n_interpolated += 1
            else:
                n_extrapolated += 1

            if interp.timing_method == "pps":
                n_pps += 1
            else:
                n_linear += 1

            if i < 3 or i >= len(all_frames) - 3:
                # Show first/last radial times
                t0 = interp.times[0]
                t1 = interp.times[-1]
                logging.info(
                    "Frame %d: pos=%s timing=%s dt=%.2fs times=[%s..%s]",
                    i, interp.method, interp.timing_method, interp.time_delta,
                    np.datetime_as_string(t0, unit='ms'),
                    np.datetime_as_string(t1, unit='ms'),
                )
        except ValueError as e:
            n_skipped += 1
            logging.warning("Frame %d: skipped - %s", i, e)

    logging.info(
        "Summary: position: %d interpolated, %d extrapolated; "
        "timing: %d pps, %d linear; %d skipped",
        n_interpolated, n_extrapolated, n_pps, n_linear, n_skipped,
    )


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test frame interpolation/extrapolation")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
