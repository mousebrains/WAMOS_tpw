"""Tests for Destreak class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.destreak import Destreak
from wamos_tpw.config import WamosConfig
from wamos_tpw.polarfile import PolarFile


class TestDestreak:
    """Tests for Destreak class."""

    def test_destreak_single_frame(self, single_polar_file: Path):
        """Test Destreak with single frame (no neighbors)."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(None, frame, None)

        assert ds.center_frame is frame
        assert ds.corrected_intensity is not None
        assert ds.corrected_intensity.shape == frame.intensity.shape

    def test_destreak_with_neighbors(self, april_polar_files: list[Path]):
        """Test Destreak with neighboring frames."""
        frames = []
        for fp in april_polar_files[:3]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 3:
            pytest.skip("Need at least 3 frames")

        prev_frame = frames[0]
        center_frame = frames[1]
        next_frame = frames[2]

        ds = Destreak(prev_frame, center_frame, next_frame)

        assert ds.center_frame is center_frame
        assert ds.corrected_intensity.shape == center_frame.intensity.shape

    def test_destreak_with_config(self, single_polar_file: Path):
        """Test Destreak with custom config."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        config = WamosConfig()
        config.destreak.min_streak_length = 5
        config.destreak.threshold_sigma = 5.0

        ds = Destreak(None, frame, None, config)

        assert ds.corrected_intensity is not None

    def test_streak_mask(self, single_polar_file: Path):
        """Test streak_mask property."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(None, frame, None)
        mask = ds.streak_mask

        assert mask is not None
        assert mask.dtype == bool
        assert mask.shape == frame.intensity.shape

    def test_corrected_preserves_non_streaks(self, single_polar_file: Path):
        """Test that non-streak pixels are preserved."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ds = Destreak(None, frame, None)

        # Non-streak pixels should be unchanged
        mask = ds.streak_mask
        orig = frame.intensity.astype(float)
        corr = ds.corrected_intensity

        # Where mask is False, values should be similar
        non_streak_orig = orig[~mask]
        non_streak_corr = corr[~mask]

        np.testing.assert_allclose(non_streak_orig, non_streak_corr, rtol=1e-10)

    def test_center_frame_required(self):
        """Test that center_frame is required."""
        with pytest.raises(ValueError, match="center_frame is required"):
            Destreak(None, None, None)


class TestDestreakAlgorithm:
    """Tests for Destreak algorithm correctness."""

    def test_different_threshold_sigma(self, single_polar_file: Path):
        """Test Destreak with different threshold sigma values."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        results = {}
        for sigma in [3.0, 5.0, 7.5, 10.0]:
            config = WamosConfig()
            config.destreak.threshold_sigma = sigma
            ds = Destreak(None, frame, None, config)
            results[sigma] = ds.streak_mask.sum()

        # Higher sigma should detect fewer streaks (more permissive threshold)
        assert results[10.0] <= results[3.0]

    def test_different_min_streak_length(self, single_polar_file: Path):
        """Test Destreak with different min_streak_length values."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        results = {}
        for length in [3, 10, 20, 50]:
            config = WamosConfig()
            config.destreak.min_streak_length = length
            ds = Destreak(None, frame, None, config)
            results[length] = ds.streak_mask.sum()

        # Longer min length should detect fewer streak pixels
        assert results[50] <= results[3]

    def test_with_deramped_intensity(self, single_polar_file: Path):
        """Test Destreak uses deramped_intensity if available."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        # Set deramped_intensity
        frame.deramped_intensity = frame.intensity.astype(float) - 100

        ds = Destreak(None, frame, None)

        # Should use deramped_intensity as input and produce valid output
        assert ds.corrected_intensity is not None
        assert ds.corrected_intensity.shape == frame.intensity.shape
