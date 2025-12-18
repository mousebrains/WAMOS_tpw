"""Tests for Timestamp class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.timestamp import Timestamp, TimingSignalExtractor
from wamos_tpw.polarfile import PolarFile


class TestTimingSignalExtractor:
    """Tests for TimingSignalExtractor class."""

    def test_extractor_basic(self, single_polar_file: Path):
        """Test basic signal extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        extractor = TimingSignalExtractor(frame.raw)

        # Should not raise
        signal = extractor.extract_timing_bit(12, 18)
        if signal is not None:
            assert signal.dtype == bool
            assert len(signal) == frame.n_bearings

    def test_extractor_invalid_bin(self, single_polar_file: Path):
        """Test extraction with invalid bin index."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        extractor = TimingSignalExtractor(frame.raw)

        # Invalid bin should return None
        result = extractor.extract_timing_bit(12, 99999)
        assert result is None

    def test_extractor_1d_raises(self):
        """Test that 1D array raises ValueError."""
        data_1d = np.zeros(100, dtype=np.uint16)
        with pytest.raises(ValueError, match="Expected 2D array"):
            TimingSignalExtractor(data_1d)

    def test_find_transitions(self, single_polar_file: Path):
        """Test transition detection."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        extractor = TimingSignalExtractor(frame.raw)

        # Create a simple test signal
        test_signal = np.array([False, False, True, True, False])
        transitions = extractor.find_transitions(test_signal)

        # Should find transitions at indices 2 and 4
        assert len(transitions) == 2
        assert 2 in transitions
        assert 4 in transitions

    def test_find_transitions_empty(self, single_polar_file: Path):
        """Test transition detection with empty signal."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        extractor = TimingSignalExtractor(frame.raw)

        transitions = extractor.find_transitions(np.array([]))
        assert len(transitions) == 0

        transitions = extractor.find_transitions(None)
        assert len(transitions) == 0


class TestTimestamp:
    """Tests for Timestamp class."""

    def test_timestamp_single_frame(self, single_polar_file: Path):
        """Test timestamp calculation with single frame."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)

        assert timestamp.frames is frames
        assert len(timestamp.times) == frames[0].n_bearings

    def test_timestamp_multiple_frames(self, april_polar_files: list[Path]):
        """Test timestamp calculation with multiple frames."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 2:
            pytest.skip("Need at least 2 frames")

        timestamp = Timestamp(frames)

        # Total times should match total radials
        expected_total = sum(f.n_bearings for f in frames)
        assert len(timestamp.times) == expected_total

        # Per-frame times should be available
        assert len(timestamp.times_per_frame) == len(frames)

    def test_timestamp_empty_frames_raises(self):
        """Test that empty frames list raises ValueError."""
        with pytest.raises(ValueError, match="At least one frame"):
            Timestamp([])

    def test_times_monotonic_within_frame(self, single_polar_file: Path):
        """Test that times are monotonically increasing within a frame."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)
        times = timestamp.times_for_frame(0)

        # Times should be monotonically increasing
        diffs = np.diff(times)
        assert np.all(diffs >= 0), "Times should be monotonically increasing"

    def test_time_step(self, single_polar_file: Path):
        """Test time step calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)
        dt = timestamp.time_step(0)

        # Time step should be positive and reasonable
        assert dt > 0
        assert dt < 0.1  # Less than 100ms per radial

    def test_frame_start_times(self, april_polar_files: list[Path]):
        """Test frame start times are properly computed."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 2:
            pytest.skip("Need at least 2 frames")

        timestamp = Timestamp(frames)
        start_times = timestamp.frame_start_times

        # Should have one start time per frame
        assert len(start_times) == len(frames)

        # Should be monotonically increasing
        assert np.all(np.diff(start_times) >= 0)

        # First frame starts at 0
        assert start_times[0] == 0.0

    def test_position_for_frame(self, single_polar_file: Path):
        """Test position calculation for a frame."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)
        lat, lon = timestamp.position_for_frame(0)

        # Should have one position per radial
        assert len(lat) == frames[0].n_bearings
        assert len(lon) == frames[0].n_bearings

        # Latitudes should be in valid range
        assert np.all(np.abs(lat) <= 90)

        # Longitudes should be in valid range
        assert np.all(np.abs(lon) <= 180)

    def test_position_for_radial(self, single_polar_file: Path):
        """Test position for specific radial."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)
        lat, lon = timestamp.position_for_radial(0, 0)

        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_absolute_times(self, single_polar_file: Path):
        """Test absolute datetime conversion."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)
        abs_times = timestamp.absolute_times_for_frame(0)

        # Should be datetime64 array
        assert abs_times.dtype.kind == "M"  # datetime type
        assert len(abs_times) == frames[0].n_bearings

    def test_timestamp_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)
        repr_str = repr(timestamp)

        assert "Timestamp(" in repr_str
        assert "frames=" in repr_str
        assert "radials=" in repr_str
        assert "duration=" in repr_str

    def test_timestamp_len(self, single_polar_file: Path):
        """Test __len__ returns total radials."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        timestamp = Timestamp(frames)

        assert len(timestamp) == frames[0].n_bearings
