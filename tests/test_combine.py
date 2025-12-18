"""Tests for Combine class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.combine import Combine
from wamos_tpw.polarfile import PolarFile


class TestCombine:
    """Tests for Combine class (earth coordinate combination)."""

    def test_combine_basic(self, single_polar_file: Path):
        """Test basic Combine creation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames, radar_height=25.0)

        assert len(combine) == 1
        assert combine.frames is frames

    def test_combine_empty_frames_raises(self):
        """Test that empty frames list raises ValueError."""
        with pytest.raises(ValueError, match="At least one frame"):
            Combine([])

    def test_xy_earth(self, single_polar_file: Path):
        """Test earth coordinate calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        combine = Combine(frames, radar_height=25.0)
        x, y = combine.xy_earth(0)

        # Should be 2D arrays
        assert x.shape == (frame.n_bearings, frame.n_distances)
        assert y.shape == (frame.n_bearings, frame.n_distances)

    def test_xy_earth_all(self, single_polar_file: Path):
        """Test concatenated earth coordinates."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        combine = Combine(frames, radar_height=25.0)
        x, y = combine.xy_earth_all()

        # Should be 1D arrays with total pixels
        expected_pixels = frame.n_bearings * frame.n_distances
        assert x.shape == (expected_pixels,)
        assert y.shape == (expected_pixels,)

    def test_latlon(self, single_polar_file: Path):
        """Test lat/lon coordinate calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        combine = Combine(frames, radar_height=25.0)
        lat, lon = combine.latlon(0)

        # Should be 2D arrays
        assert lat.shape == (frame.n_bearings, frame.n_distances)
        assert lon.shape == (frame.n_bearings, frame.n_distances)

        # Values should be in valid ranges
        assert np.all(np.abs(lat) <= 90)
        assert np.all(np.abs(lon) <= 180)

    def test_latlon_all(self, single_polar_file: Path):
        """Test concatenated lat/lon coordinates."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        combine = Combine(frames, radar_height=25.0)
        lat, lon = combine.latlon_all()

        expected_pixels = frame.n_bearings * frame.n_distances
        assert lat.shape == (expected_pixels,)
        assert lon.shape == (expected_pixels,)

    def test_intensity_all(self, single_polar_file: Path):
        """Test concatenated intensity values."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        # Set corrected_intensity on frame to simulate processed data
        frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames)
        intensity = combine.intensity_all()

        expected_pixels = frame.n_bearings * frame.n_distances
        assert intensity.shape == (expected_pixels,)

    def test_ship_track_xy(self, single_polar_file: Path):
        """Test ship track calculation in x/y coordinates."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        x, y = combine.ship_track_xy()

        # One position per radial
        assert len(x) == frames[0].n_bearings
        assert len(y) == frames[0].n_bearings

    def test_ship_track_latlon(self, single_polar_file: Path):
        """Test ship track calculation in lat/lon."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        lat, lon = combine.ship_track()

        # One position per radial
        assert len(lat) == frames[0].n_bearings
        assert len(lon) == frames[0].n_bearings

        # Values in valid range
        assert np.all(np.abs(lat) <= 90)
        assert np.all(np.abs(lon) <= 180)

    def test_travel_distance(self, single_polar_file: Path):
        """Test travel distance calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        travel = combine.travel_distance()

        assert "total_m" in travel
        assert "x_m" in travel
        assert "y_m" in travel
        assert "duration_s" in travel
        assert "speed_m_s" in travel

        # Values should be reasonable
        assert travel["total_m"] >= 0
        assert travel["duration_s"] >= 0
        assert travel["speed_m_s"] >= 0

    def test_frame_velocities(self, single_polar_file: Path):
        """Test per-frame velocity calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        vel = combine.frame_velocities()

        assert "speeds" in vel
        assert "headings" in vel
        assert "speed_mean" in vel
        assert "speed_std" in vel
        assert "heading_mean" in vel
        assert "heading_std" in vel

        # Mean values should be reasonable
        assert vel["speed_mean"] >= 0
        assert 0 <= vel["heading_mean"] < 360

    def test_compute_grid_bounds(self, single_polar_file: Path):
        """Test grid bounds calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames, radar_height=25.0)
        x_min, x_max, y_min, y_max = combine.compute_grid_bounds()

        # Bounds should form a valid rectangle
        assert x_min < x_max
        assert y_min < y_max

        # Radar range should be reflected in bounds (several km)
        x_range = x_max - x_min
        y_range = y_max - y_min
        assert x_range > 1000  # At least 1km
        assert y_range > 1000

    def test_reference_position(self, single_polar_file: Path):
        """Test reference position property."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        ref_lat, ref_lon = combine.reference_position

        # Should be valid coordinates
        assert -90 <= ref_lat <= 90
        assert -180 <= ref_lon <= 180

    def test_combine_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        repr_str = repr(combine)

        assert "Combine(" in repr_str
        assert "frames=" in repr_str
        assert "duration=" in repr_str

    def test_compute_track_angle(self, single_polar_file: Path):
        """Test track angle calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        combine = Combine(frames)
        angle = combine.compute_track_angle()

        # Should be a valid angle in radians
        assert isinstance(angle, float)
        assert -np.pi <= angle <= np.pi


class TestCombineMultipleFrames:
    """Tests for Combine with multiple frames."""

    def test_combine_two_frames(self, april_polar_files: list[Path]):
        """Test combining two frames."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 2:
            pytest.skip("Need at least 2 frames")

        combine = Combine(frames, radar_height=25.0)

        assert len(combine) == 2

        # Should have coordinates for both frames
        x0, y0 = combine.xy_earth(0)
        x1, y1 = combine.xy_earth(1)

        assert x0.shape[0] == frames[0].n_bearings
        assert x1.shape[0] == frames[1].n_bearings

    def test_ship_track_continuous(self, april_polar_files: list[Path]):
        """Test that ship track is continuous across frames."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 2:
            pytest.skip("Need at least 2 frames")

        combine = Combine(frames)
        x, y = combine.ship_track_xy()

        # Total positions should be sum of all radials
        expected = sum(f.n_bearings for f in frames)
        assert len(x) == expected

        # Track should be relatively continuous (no huge jumps)
        dx = np.diff(x)
        dy = np.diff(y)
        max_step = np.sqrt(dx**2 + dy**2).max()
        # Max step should be reasonable (depends on speed, but < 100m per radial)
        assert max_step < 100
