"""Tests for Destreak class."""

from pathlib import Path

import pytest

from wamos_tpw.destreak import Destreak
from wamos_tpw.polarfile import PolarFile


class TestDestreak:
    """Tests for Destreak class."""

    def test_destreak_single_frame(self, single_polar_file: Path):
        """Test basic Destreak creation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(frame)

        assert ds.intensity is not None
        assert ds.intensity.shape == frame.intensity.shape

    def test_destreak_stats(self, single_polar_file: Path):
        """Test destreak statistics."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(frame)

        assert ds.n_streak_pixels >= 0
        assert 0.0 <= ds.streak_fraction <= 1.0

    def test_streak_mask(self, single_polar_file: Path):
        """Test streak_mask property when save_mask=True."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(frame, save_mask=True)
        mask = ds.streak_mask

        assert mask is not None
        assert mask.dtype == bool
        assert mask.shape == frame.intensity.shape

    def test_streak_mask_none_by_default(self, single_polar_file: Path):
        """Test streak_mask is None when save_mask=False."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(frame, save_mask=False)

        assert ds.streak_mask is None

    def test_destreak_preserves_shape(self, single_polar_file: Path):
        """Test that destreak preserves array shape."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(frame)

        assert ds.intensity.shape == frame.intensity.shape

    def test_frame_required(self):
        """Test that frame is required."""
        with pytest.raises(ValueError, match="frame is required"):
            Destreak(None)


class TestDestreakAlgorithm:
    """Tests for Destreak algorithm correctness."""

    def test_destreak_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(frame)
        repr_str = repr(ds)

        assert "Destreak(" in repr_str
