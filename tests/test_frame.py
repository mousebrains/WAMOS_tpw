"""Tests for Frame class."""

import numpy as np
from pathlib import Path

from wamos_tpw.polarfile import PolarFile


class TestFrame:
    """Tests for Frame class."""

    def test_intensity_extraction(self, single_polar_file: Path):
        """Test intensity data extraction (bottom 12 bits)."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        intensity = frame.intensity
        assert intensity.dtype == np.uint16
        assert intensity.max() <= 4095  # 12-bit max
        assert intensity.min() >= 0

    def test_pps_extraction(self, single_polar_file: Path):
        """Test PPS (bit 12) extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        pps = frame.pps
        assert pps.dtype == bool
        assert pps.shape == frame.intensity.shape

    def test_bearing_pulse_extraction(self, single_polar_file: Path):
        """Test bearing pulse (bit 13) extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        bp = frame.bearing_pulse
        assert bp.dtype == bool
        assert bp.shape == frame.intensity.shape

    def test_slant_range(self, single_polar_file: Path):
        """Test slant range calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        slant = frame.slant_range()
        assert len(slant) == frame.n_distances
        assert slant[0] >= 0
        assert slant[-1] > slant[0]  # Should be monotonically increasing

    def test_ground_range(self, single_polar_file: Path):
        """Test ground range calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        radar_height = 25.0  # meters
        ground = frame.ground_range(radar_height)
        slant = frame.slant_range()

        assert len(ground) == frame.n_distances
        # Ground range should be less than or equal to slant range
        assert np.all(ground <= slant)

    def test_frame_repr(self, single_polar_file: Path):
        """Test frame string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        repr_str = repr(frame)
        assert "Frame(" in repr_str
        assert str(frame.n_bearings) in repr_str
        assert str(frame.n_distances) in repr_str
