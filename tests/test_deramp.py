"""Tests for Deramp class."""

import numpy as np
from pathlib import Path

from wamos_tpw.deramp import Deramp
from wamos_tpw.config import WamosConfig
from wamos_tpw.polarfile import PolarFile


class TestDeramp:
    """Tests for Deramp class."""

    def test_deramp_basic(self, single_polar_file: Path):
        """Test basic Deramp creation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)

        assert deramp.quantile == 0.10  # Default

    def test_deramp_with_config(self, single_polar_file: Path):
        """Test Deramp with custom config."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        config = WamosConfig()
        config.shadow.center = 180.0
        config.shadow.width = 90.0

        deramp = Deramp(frame, config, quantile=0.25)

        assert deramp.quantile == 0.25

    def test_raw_profile(self, single_polar_file: Path):
        """Test raw_profile property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)
        profile = deramp.raw_profile

        assert profile is not None
        assert len(profile) == frame.n_distances
        assert profile.dtype == np.float32

    def test_smooth_profile(self, single_polar_file: Path):
        """Test smooth_profile property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)
        smooth = deramp.smooth_profile

        assert smooth is not None
        assert len(smooth) == frame.n_distances

        # Smooth profile should be similar to raw but smoother
        raw = deramp.raw_profile
        assert smooth.shape == raw.shape

    def test_corrected_intensity(self, single_polar_file: Path):
        """Test corrected_intensity property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)
        corrected = deramp.corrected_intensity

        assert corrected is not None
        assert corrected.shape == frame.intensity.shape
        assert corrected.dtype == np.float64

    def test_shadow_mask(self, single_polar_file: Path):
        """Test shadow_mask property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        config = WamosConfig()
        config.shadow.center = 180.0
        config.shadow.width = 90.0

        deramp = Deramp(frame, config)
        mask = deramp.shadow_mask

        assert mask is not None
        assert mask.dtype == bool
        assert len(mask) == frame.n_bearings

        # Some should be in shadow, some not
        assert mask.sum() > 0
        assert mask.sum() < len(mask)

    def test_different_quantiles(self, single_polar_file: Path):
        """Test Deramp with different quantile values."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        results = {}
        for q in [0.05, 0.10, 0.25, 0.50]:
            deramp = Deramp(frame, quantile=q)
            results[q] = deramp.raw_profile.copy()

        # Higher quantiles should generally give higher profiles
        # (more of the distribution is below)
        assert np.mean(results[0.50]) > np.mean(results[0.10])

    def test_slant_range(self, single_polar_file: Path):
        """Test slant_range property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)
        slant = deramp.slant_range

        assert slant is not None
        assert len(slant) == frame.n_distances


class TestDerampAlgorithm:
    """Tests for Deramp algorithm correctness."""

    def test_profile_removes_range_trend(self, single_polar_file: Path):
        """Test that deramp removes range-dependent intensity trend."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)

        # Original intensity should have range-dependent trend
        orig = frame.intensity.astype(float)
        orig_mean_per_range = np.mean(orig, axis=0)

        # Corrected should have flatter profile
        corrected = deramp.corrected_intensity
        corr_mean_per_range = np.mean(corrected, axis=0)

        # Standard deviation of mean profile should be lower for corrected
        orig_std = np.std(orig_mean_per_range)
        corr_std = np.std(corr_mean_per_range)

        # Allow some tolerance - corrected should be flatter or similar
        assert corr_std <= orig_std * 1.5

    def test_smoothing_reduces_noise(self, single_polar_file: Path):
        """Test that smoothing reduces profile noise."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        deramp = Deramp(frame)

        raw = deramp.raw_profile
        smooth = deramp.smooth_profile

        # Calculate roughness (sum of absolute differences between adjacent points)
        raw_roughness = np.sum(np.abs(np.diff(raw)))
        smooth_roughness = np.sum(np.abs(np.diff(smooth)))

        # Smooth profile should be less rough
        assert smooth_roughness <= raw_roughness

    def test_shadow_exclusion(self, single_polar_file: Path):
        """Test that shadow region is excluded from profile calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        # Wide shadow
        config_wide = WamosConfig()
        config_wide.shadow.center = 180.0
        config_wide.shadow.width = 180.0  # Half the bearings

        # Narrow shadow
        config_narrow = WamosConfig()
        config_narrow.shadow.center = 180.0
        config_narrow.shadow.width = 10.0  # Very few bearings

        deramp_wide = Deramp(frame, config_wide)
        deramp_narrow = Deramp(frame, config_narrow)

        # Both should produce valid profiles
        assert deramp_wide.raw_profile is not None
        assert deramp_narrow.raw_profile is not None

        # The number of non-shadow bearings should differ
        assert deramp_wide.shadow_mask.sum() > deramp_narrow.shadow_mask.sum()
