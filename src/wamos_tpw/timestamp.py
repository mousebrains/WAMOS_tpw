#! /usr/bin/env python3
#
# Timestamp class for estimating precise timing and position for each radial beam
# Uses timing signals encoded in specific bit/bin combinations to anchor timing
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from wamos_tpw.config import WamosConfig
from wamos_tpw.frame import Frame


class TimingSignalExtractor:
    """
    Extract and analyze timing signals from WAMOS polar data.

    The WAMOS radar encodes timing information in specific bit/bin combinations:
    - Bit 12, Bin 18: 1 Hz (1 second period) - primary reference
    - Bit 13, Bin 19: 8 Hz (1/8 second period)
    - Bit 12, Bin 19: 16 Hz (1/16 second period)
    - Bit 14, Bin 19: 4 Hz (1/4 second period)

    These signals allow sub-millisecond timing precision.
    """

    # Timing signal definitions: (bit, bin, period_seconds)
    TIMING_SIGNALS = {
        'signal_1s': (12, 18, 1.0),           # 1 Hz - 1 second
        'signal_250ms': (14, 19, 0.25),       # 4 Hz - 1/4 second
        'signal_125ms': (13, 19, 0.125),      # 8 Hz - 1/8 second
        'signal_62p5ms': (12, 19, 0.0625),    # 16 Hz - 1/16 second
    }

    def __init__(self, data: np.ndarray):
        """
        Initialize timing signal extractor.

        Args:
            data: Raw uint16 data array (n_bearings, n_distances)
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        self._data = data
        self._n_radials = data.shape[0]
        self._n_distances = data.shape[1]

    def extract_timing_bit(self, bit: int, bin_idx: int) -> np.ndarray | None:
        """
        Extract a timing bit from specified distance bin.

        Args:
            bit: Bit number (0-15, where 0 is LSB)
            bin_idx: Distance bin number (0-indexed)

        Returns:
            Boolean array of bit values for each radial, or None if bin out of range
        """
        if bin_idx >= self._n_distances:
            return None

        mask = 1 << bit
        return (self._data[:, bin_idx] & mask) != 0

    def find_transitions(self, signal: np.ndarray) -> np.ndarray:
        """
        Find indices where signal transitions occur.

        Args:
            signal: Boolean array

        Returns:
            Array of radial indices where transitions occur (0->1 or 1->0)
        """
        if signal is None or len(signal) == 0:
            return np.array([], dtype=int)

        changes = np.diff(signal.astype(int)) != 0
        return np.where(changes)[0] + 1

    def get_1s_reference(self) -> int | None:
        """
        Find the radial index where the 1-second timing signal transitions.

        Returns:
            Radial index of first 1s transition, or None if not found
        """
        bit, bin_idx, _ = self.TIMING_SIGNALS['signal_1s']
        signal = self.extract_timing_bit(bit, bin_idx)
        if signal is None:
            return None

        transitions = self.find_transitions(signal)
        if len(transitions) == 0:
            return None

        return int(transitions[0])


class Timestamp:
    """
    Calculate precise timing and position for each radial beam across multiple frames.

    Uses timing signals encoded in the data to anchor timing to the 1-second
    reference, then estimates position using ship speed and heading.

    For the last frame, positions are calculated by projecting backward from
    the frame's recorded position using ship motion.

    Example:
        >>> config = WamosConfig('radar_config.yaml')
        >>> timestamp = Timestamp(frames, config)
        >>> times = timestamp.times  # Absolute times for all radials
        >>> lat, lon = timestamp.position_for_frame(0)  # Positions for frame 0
    """

    # Earth radius in meters (WGS84 mean radius)
    _EARTH_RADIUS = 6371000.0

    def __init__(self,
                 frames: list[Frame],
                 config: WamosConfig | None = None):
        """
        Initialize Timestamp calculator for a set of contiguous frames.

        Args:
            frames: List of contiguous Frame objects (sorted by time)
            config: WamosConfig object (uses defaults if None)
        """
        if not frames:
            raise ValueError("At least one frame is required")

        self._frames = frames
        self._config = config or WamosConfig()

        # Calculated values
        self._times_per_frame: list[np.ndarray] | None = None
        self._frame_start_times: np.ndarray | None = None

        # Calculate timing
        self._calculate_times()

    def _calculate_times(self) -> None:
        """Calculate times for all radials in all frames."""
        self._times_per_frame = []
        self._frame_start_times = np.zeros(len(self._frames))

        cumulative_time = 0.0

        for i, frame in enumerate(self._frames):
            # Get frame duration from RPT (repeat time)
            rpt = frame.metadata.repeat_time
            if rpt <= 0:
                rpt = 1.5  # Default frame duration

            self._frame_start_times[i] = cumulative_time

            # Calculate radial times for this frame
            times = self._calculate_frame_times(frame, cumulative_time, rpt)
            self._times_per_frame.append(times)

            # Advance cumulative time by frame duration
            cumulative_time += rpt

    def _calculate_frame_times(self,
                               frame: Frame,
                               frame_start: float,
                               frame_duration: float) -> np.ndarray:
        """
        Calculate times for each radial in a frame.

        Uses the 1-second timing signal to anchor the timing, then assumes
        uniform radial rate within the frame.

        Args:
            frame: Frame object
            frame_start: Start time of frame (seconds from first frame)
            frame_duration: Frame duration (RPT) in seconds

        Returns:
            Array of times for each radial (seconds from first frame start)
        """
        n_radials = frame.n_bearings

        # Try to extract timing signals
        extractor = TimingSignalExtractor(frame.raw)
        ref_radial = extractor.get_1s_reference()

        if ref_radial is None:
            # No timing reference - use uniform distribution
            return frame_start + np.linspace(0, frame_duration, n_radials, endpoint=False)

        # Calculate time step (uniform assumption)
        dt = frame_duration / n_radials

        # The 1s transition marks where time crosses 1.0 second within the frame
        # Radial (ref_radial - 1) has time < frame_start + 1.0s
        # Radial (ref_radial) has time >= frame_start + 1.0s
        # Place boundary at midpoint
        time_0 = frame_start + 1.0 - (ref_radial - 0.5) * dt

        # Calculate all times with uniform spacing
        times = time_0 + np.arange(n_radials) * dt

        return times

    def _calculate_position(self,
                            frame_idx: int,
                            radial_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate lat/lon for each radial based on ship motion.

        For the last frame, uses the frame's recorded position and works backward.
        For earlier frames, works forward from first frame.

        Args:
            frame_idx: Frame index
            radial_times: Times for each radial in this frame

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        frame = self._frames[frame_idx]
        meta = frame.metadata
        n_radials = len(radial_times)

        # Get ship position and motion
        base_lat = meta.latitude or 0.0
        base_lon = meta.longitude or 0.0

        # Ship speed in m/s (already converted at parse time)
        ship_speed = meta.ship_speed or 0.0

        # Ship heading (course over ground)
        ship_heading = meta.ship_course if meta.ship_course is not None else (meta.heading or 0.0)

        if ship_speed <= 0:
            # No motion - all radials at same position
            return np.full(n_radials, base_lat), np.full(n_radials, base_lon)

        # For last frame, position is recorded at frame time, so work backward
        # For other frames, position is at frame start
        is_last_frame = (frame_idx == len(self._frames) - 1)

        if is_last_frame:
            # Reference time is the frame timestamp (approximately end of frame)
            frame_end_time = self._frame_start_times[frame_idx] + frame.metadata.repeat_time
            time_deltas = radial_times - frame_end_time  # Negative values (backward)
        else:
            # Reference time is frame start
            frame_start_time = self._frame_start_times[frame_idx]
            time_deltas = radial_times - frame_start_time

        # Calculate displacement in meters
        displacement = ship_speed * time_deltas

        # Convert heading to radians (0 = North, 90 = East)
        heading_rad = np.deg2rad(ship_heading)

        # Calculate North/East displacement
        delta_north = displacement * np.cos(heading_rad)
        delta_east = displacement * np.sin(heading_rad)

        # Convert to lat/lon changes
        # At latitude phi: 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(phi)
        meters_per_deg_lat = np.pi * self._EARTH_RADIUS / 180.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(base_lat))

        delta_lat = delta_north / meters_per_deg_lat
        delta_lon = delta_east / meters_per_deg_lon if meters_per_deg_lon > 0 else 0.0

        lat = base_lat + delta_lat
        lon = base_lon + delta_lon

        return lat, lon

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def frames(self) -> list[Frame]:
        """Return the frames."""
        return self._frames

    @property
    def times(self) -> np.ndarray:
        """
        Get times for all radials across all frames.

        Returns:
            Array of times in seconds from the start of the first frame
        """
        return np.concatenate(self._times_per_frame)

    @property
    def times_per_frame(self) -> list[np.ndarray]:
        """
        Get times separated by frame.

        Returns:
            List of arrays, one per frame
        """
        return self._times_per_frame

    def times_for_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get times for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Array of times in seconds from the start of the first frame
        """
        return self._times_per_frame[frame_idx]

    def absolute_times_for_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get absolute datetime64 times for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Array of np.datetime64 timestamps
        """
        # Get base timestamp from frame
        base_time = self._frames[frame_idx].timestamp

        # Get relative times
        relative_times = self._times_per_frame[frame_idx]

        # Offset from frame start
        frame_start = self._frame_start_times[frame_idx]
        offsets_from_base = relative_times - frame_start

        # Convert to timedelta64 and add to base
        offsets_ns = (offsets_from_base * 1e9).astype('int64')
        return base_time + offsets_ns.astype('timedelta64[ns]')

    @property
    def frame_start_times(self) -> np.ndarray:
        """Get the start time of each frame in seconds."""
        return self._frame_start_times

    def position_for_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get estimated lat/lon for each radial in a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (latitude, longitude) arrays in degrees
        """
        times = self._times_per_frame[frame_idx]
        return self._calculate_position(frame_idx, times)

    def position_for_radial(self, frame_idx: int, radial_idx: int) -> Tuple[float, float]:
        """
        Get estimated lat/lon for a specific radial.

        Args:
            frame_idx: Frame index
            radial_idx: Radial index within frame

        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        lat, lon = self.position_for_frame(frame_idx)
        return float(lat[radial_idx]), float(lon[radial_idx])

    def time_step(self, frame_idx: int) -> float:
        """
        Get the average time step between radials for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Average time step in seconds
        """
        times = self._times_per_frame[frame_idx]
        if len(times) < 2:
            return 0.0
        return float(np.mean(np.diff(times)))

    def __len__(self) -> int:
        """Return total number of radials across all frames."""
        return sum(len(t) for t in self._times_per_frame)

    def __repr__(self) -> str:
        total_duration = self._frame_start_times[-1] + self._frames[-1].metadata.repeat_time
        return (
            f"Timestamp(frames={len(self._frames)}, "
            f"radials={len(self)}, "
            f"duration={total_duration:.2f}s)"
        )

    def __enter__(self) -> 'Timestamp':
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        pass


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments
    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML configuration file")
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Maximum frames to process (default: 10)")


def add_subparser(subparsers) -> None:
    """Register the 'timestamp' subcommand."""
    p = subparsers.add_parser(
        'timestamp',
        help='Timestamp analysis',
        description="Calculate radial timing from polar files"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'timestamp' command."""
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.polarfile import load_polar_file

    # Find files (args.stime/etime already parsed by argparse)
    filenames = Filenames(args.stime, args.etime, args.polar_path)
    logging.info(f"Found {len(filenames)} files")

    if not filenames:
        logging.warning("No files found")
        return

    # Load frames
    logging.info(f"Loading up to {args.max_frames} frames...")
    frames = []
    for filepath in filenames.files[:args.max_frames]:
        frame = load_polar_file(filepath)
        if frame is not None:
            frames.append(frame)

    logging.info(f"Loaded {len(frames)} frames")

    if not frames:
        logging.warning("No valid frames")
        return

    # Load config
    config = WamosConfig(args.config) if args.config else WamosConfig()

    # Calculate timing
    timestamp = Timestamp(frames, config)
    logging.info(f"{timestamp}")

    # Show timing statistics
    logging.info("Timing Statistics:")
    total_radials = len(timestamp)
    logging.info(f"  Total radials: {total_radials}")
    logging.info(f"  Frame start times: {timestamp.frame_start_times}")

    for i in range(min(3, len(frames))):
        times = timestamp.times_for_frame(i)
        dt = timestamp.time_step(i)
        logging.info(f"  Frame {i}:")
        logging.info(f"    Radials: {len(times)}")
        logging.info(f"    Time range: {times[0]:.4f}s to {times[-1]:.4f}s")
        logging.info(f"    Mean step: {dt*1000:.3f} ms")

        # Show position
        lat, lon = timestamp.position_for_frame(i)
        logging.info(f"    Position range: ({lat[0]:.6f}, {lon[0]:.6f}) to ({lat[-1]:.6f}, {lon[-1]:.6f})")

        # Show absolute times
        abs_times = timestamp.absolute_times_for_frame(i)
        logging.info(f"    Absolute times: {abs_times[0]} to {abs_times[-1]}")


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Calculate radial timing from polar files")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
