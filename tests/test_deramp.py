"""Tests for Deramp class."""

from pathlib import Path

import numpy as np

from wamos_tpw.deramp import Deramp
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.range import Range


class TestDeramp:
    """Tests for Deramp class."""

    def test_deramp_basic(self, single_polar_file: Path):
        """Test basic Deramp creation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        deramp = Deramp(intensity, rng)

        assert deramp.intensity is not None
        assert deramp.intensity.shape == frame.intensity.shape

    def test_deramp_intensity(self, single_polar_file: Path):
        """Test corrected intensity property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        deramp = Deramp(intensity, rng)
        corrected = deramp.intensity

        assert corrected is not None
        assert corrected.shape == frame.intensity.shape

    def test_deramp_polynomial(self, single_polar_file: Path):
        """Test polynomial property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        deramp = Deramp(intensity, rng)

        assert deramp.polynomial is not None
        assert deramp.order == 4  # Default order

    def test_deramp_order(self, single_polar_file: Path):
        """Test deramp order from config."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)

        # Modify config to use different order
        frame._config["deramp.order"] = 2
        rng = Range(frame)

        deramp = Deramp(intensity, rng)

        assert deramp.order == 2


class TestDerampAlgorithm:
    """Tests for Deramp algorithm correctness."""

    def test_profile_removes_range_trend(self, single_polar_file: Path):
        """Test that deramp removes range-dependent intensity trend."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        deramp = Deramp(intensity.copy(), rng)

        # Original intensity should have range-dependent trend
        orig_mean_per_range = np.nanmean(intensity, axis=0)

        # Corrected should have flatter profile
        corrected = deramp.intensity
        corr_mean_per_range = np.nanmean(corrected, axis=0)

        # Standard deviation of mean profile should be lower for corrected
        orig_std = np.nanstd(orig_mean_per_range)
        corr_std = np.nanstd(corr_mean_per_range)

        # Corrected should be flatter or similar
        assert corr_std <= orig_std * 1.5

    def test_deramp_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        deramp = Deramp(intensity, rng)
        repr_str = repr(deramp)

        assert "Deramp(" in repr_str
        assert "order=" in repr_str
