"""Tests for processing pipeline (deramp, destreak)."""

import pytest
import numpy as np
from pathlib import Path

from wamos_tpw.polarfile import PolarFile
from wamos_tpw.deramp import Deramp
from wamos_tpw.destreak import Destreak, destreak_frame


class TestDeramp:
    """Tests for the Deramp class."""

    def test_deramp_single_frame(self, single_polar_file: Path):
        """Test deramping a single frame."""
        pf = PolarFile(str(single_polar_file))
        frames = pf.frames

        assert len(frames) > 0, "Should have at least one frame"
        frame = frames[0]

        # Create Deramp instance with frame
        deramp = Deramp(frame)

        # Get corrected intensity
        result = deramp.corrected_intensity

        # Result should be same shape as input
        assert result.shape == frame.intensity.shape, "Output shape should match input"

        # Result should be floating point
        assert result.dtype == np.float64, "Output should be float64"

        # Result should be different from input (deramping changes values)
        assert not np.allclose(result, frame.intensity), "Deramped data should differ from input"

    def test_deramp_properties(self, single_polar_file: Path):
        """Test that Deramp computes all expected properties."""
        pf = PolarFile(str(single_polar_file))
        frame = pf.frames[0]

        deramp = Deramp(frame, quantile=0.25)

        # Check quantile is set correctly
        assert deramp.quantile == 0.25

        # Check all properties are computed
        assert deramp.corrected_intensity is not None
        assert deramp.raw_profile is not None
        assert deramp.smooth_profile is not None
        assert deramp.slant_range is not None

        # Profiles should be 1D arrays
        assert len(deramp.raw_profile.shape) == 1
        assert len(deramp.smooth_profile.shape) == 1

    def test_deramp_different_quantiles(self, single_polar_file: Path):
        """Test deramping with different quantile values."""
        pf = PolarFile(str(single_polar_file))
        frame = pf.frames[0]

        deramp_10 = Deramp(frame, quantile=0.10)
        deramp_50 = Deramp(frame, quantile=0.50)

        result_10 = deramp_10.corrected_intensity
        result_50 = deramp_50.corrected_intensity

        # Both should have same shape
        assert result_10.shape == result_50.shape

        # But profiles should differ
        assert not np.allclose(deramp_10.smooth_profile, deramp_50.smooth_profile)


class TestDestreak:
    """Tests for the Destreak class."""

    def test_destreak_single_frame(self, single_polar_file: Path):
        """Test destreaking a single frame (no neighbors)."""
        pf = PolarFile(str(single_polar_file))
        frame = pf.frames[0]

        # Create Destreak with no neighboring frames
        destreak = Destreak(
            prev_frame=None,
            center_frame=frame,
            next_frame=None
        )

        result = destreak.corrected_intensity

        # Result should be same shape
        assert result.shape == frame.intensity.shape, "Output shape should match input"

        # Result should be floating point
        assert result.dtype == np.float64, "Output should be float64"

    def test_destreak_with_neighbors(self, april_polar_files: list[Path]):
        """Test destreaking with neighboring frames."""
        pf = PolarFile(str(april_polar_files[0]))
        frames = pf.frames

        if len(frames) < 3:
            pytest.skip("Need at least 3 frames for neighbor test")

        # Use frames 0, 1, 2 as prev, center, next
        destreak = Destreak(
            prev_frame=frames[0],
            center_frame=frames[1],
            next_frame=frames[2]
        )

        result = destreak.corrected_intensity
        assert result.shape == frames[1].intensity.shape

        # Check mask was computed
        assert destreak.streak_mask is not None
        assert destreak.streak_mask.shape == frames[1].intensity.shape

    def test_destreak_frame_function(self, single_polar_file: Path):
        """Test the destreak_frame convenience function."""
        pf = PolarFile(str(single_polar_file))
        frame = pf.frames[0]

        result = destreak_frame(
            prev_frame=None,
            center_frame=frame,
            next_frame=None
        )

        assert result.shape == frame.intensity.shape
        assert result.dtype == np.float64


class TestProcessingPipeline:
    """Test the combined processing pipeline."""

    def test_deramp_then_destreak(self, single_polar_file: Path):
        """Test deramping followed by destreaking."""
        pf = PolarFile(str(single_polar_file))
        frame = pf.frames[0]

        # Step 1: Deramp
        deramp = Deramp(frame)
        deramped = deramp.corrected_intensity

        # Store deramped result on frame for destreak to use
        frame.deramped_intensity = deramped

        # Step 2: Destreak (will use deramped_intensity if available)
        destreak = Destreak(
            prev_frame=None,
            center_frame=frame,
            next_frame=None
        )
        final = destreak.corrected_intensity

        # Final should be same shape
        assert final.shape == frame.intensity.shape

        # Final should be different from original
        assert not np.allclose(final, frame.intensity)

    def test_multiple_frames_pipeline(self, april_polar_files: list[Path]):
        """Test processing multiple frames through the pipeline."""
        pf = PolarFile(str(april_polar_files[0]))
        frames = pf.frames[:3]  # First 3 frames

        if len(frames) < 3:
            pytest.skip("Need at least 3 frames")

        # Deramp all frames
        for frame in frames:
            deramp = Deramp(frame)
            frame.deramped_intensity = deramp.corrected_intensity

        # Destreak middle frame with neighbors
        destreak = Destreak(
            prev_frame=frames[0],
            center_frame=frames[1],
            next_frame=frames[2]
        )
        result = destreak.corrected_intensity

        assert result.shape == frames[1].intensity.shape
        assert result.dtype == np.float64
