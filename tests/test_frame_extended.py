"""Extended tests for Frame class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.polarfile import PolarFile


class TestFrameProperties:
    """Tests for Frame properties."""

    def test_n_bearings(self, single_polar_file: Path):
        """Test n_bearings property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        assert frame.n_bearings > 0
        assert frame.n_bearings == frame.raw.shape[0]

    def test_n_distances(self, single_polar_file: Path):
        """Test n_distances property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        assert frame.n_distances > 0
        assert frame.n_distances == frame.raw.shape[1]

    def test_shape(self, single_polar_file: Path):
        """Test shape property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        assert frame.shape == (frame.n_bearings, frame.n_distances)
        assert frame.shape == frame.raw.shape

    def test_timestamp(self, single_polar_file: Path):
        """Test timestamp property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        assert frame.timestamp is not None
        assert isinstance(frame.timestamp, np.datetime64)

    def test_metadata(self, single_polar_file: Path):
        """Test metadata property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        meta = frame.metadata
        assert meta is not None
        # Check some expected metadata attributes
        assert hasattr(meta, "heading")
        assert hasattr(meta, "latitude")
        assert hasattr(meta, "longitude")

    def test_raw_property(self, single_polar_file: Path):
        """Test raw property returns uint16 array."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        raw = frame.raw
        assert raw.dtype == np.uint16
        assert raw.ndim == 2


class TestFrameBitExtraction:
    """Tests for Frame bit extraction methods."""

    def test_intensity(self, single_polar_file: Path):
        """Test intensity extraction (bottom 12 bits)."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        intensity = frame.intensity
        assert intensity.dtype == np.uint16
        assert intensity.max() <= 4095  # 12-bit max
        assert intensity.min() >= 0

    def test_pps(self, single_polar_file: Path):
        """Test PPS (bit 12) extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        pps = frame.pps
        assert pps.dtype == bool
        assert pps.shape == frame.shape

    def test_bearing_pulse(self, single_polar_file: Path):
        """Test bearing pulse (bit 13) extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        bp = frame.bearing_pulse
        assert bp.dtype == bool
        assert bp.shape == frame.shape

    def test_bit12(self, single_polar_file: Path):
        """Test bit12 extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        bit12 = frame.bit12
        assert bit12.dtype == bool
        assert bit12.shape == frame.shape

    def test_bit13(self, single_polar_file: Path):
        """Test bit13 extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        bit13 = frame.bit13
        assert bit13.dtype == bool
        assert bit13.shape == frame.shape

    def test_bit14(self, single_polar_file: Path):
        """Test bit14 extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        bit14 = frame.bit14
        assert bit14.dtype == bool
        assert bit14.shape == frame.shape

    def test_bit15(self, single_polar_file: Path):
        """Test bit15 extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        bit15 = frame.bit15
        assert bit15.dtype == bool
        assert bit15.shape == frame.shape


class TestFrameRangeCalculations:
    """Tests for Frame range calculation methods."""

    def test_slant_range(self, single_polar_file: Path):
        """Test slant range calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        slant = frame.slant_range()

        assert len(slant) == frame.n_distances
        assert slant[0] >= 0  # First bin is at or beyond radar
        assert np.all(np.diff(slant) > 0)  # Monotonically increasing

    def test_slant_range_with_indices(self, single_polar_file: Path):
        """Test slant range with specific bin indices."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        indices = np.array([0, 10, 50, 100])
        slant = frame.slant_range(bin_indices=indices)

        assert len(slant) == len(indices)
        assert np.all(np.diff(slant) > 0)

    def test_ground_range(self, single_polar_file: Path):
        """Test ground range calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ground = frame.ground_range(radar_height=25.0)
        slant = frame.slant_range()

        assert len(ground) == frame.n_distances
        # Ground range should be <= slant range
        assert np.all(ground <= slant)

    def test_ground_range_without_height_raises(self, single_polar_file: Path):
        """Test ground_range raises if no height available."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        if frame.metadata.radar_height is None:
            with pytest.raises(ValueError, match="radar_height must be provided"):
                frame.ground_range()


class TestFrameRepr:
    """Tests for Frame string representation."""

    def test_repr(self, single_polar_file: Path):
        """Test __repr__ method."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        repr_str = repr(frame)
        assert "Frame(" in repr_str
        assert "timestamp=" in repr_str
        assert "shape=" in repr_str

    def test_str(self, single_polar_file: Path):
        """Test __str__ method."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        str_str = str(frame)
        assert isinstance(str_str, str)


class TestFrameCaching:
    """Tests for Frame property caching."""

    def test_intensity_caching(self, single_polar_file: Path):
        """Test that intensity is cached."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        # First access
        intensity1 = frame.intensity
        # Second access should return same object
        intensity2 = frame.intensity

        assert intensity1 is intensity2

    def test_pps_caching(self, single_polar_file: Path):
        """Test that pps is cached."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        pps1 = frame.pps
        pps2 = frame.pps

        assert pps1 is pps2

    def test_bearing_pulse_caching(self, single_polar_file: Path):
        """Test that bearing_pulse is cached."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        bp1 = frame.bearing_pulse
        bp2 = frame.bearing_pulse

        assert bp1 is bp2
