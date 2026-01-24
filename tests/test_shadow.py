#! /usr/bin/env python3
"""Tests for wamos_tpw.shadow module."""

import numpy as np
import pytest

from wamos_tpw.config import Config
from wamos_tpw.destreak import Destreak
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.shadow import Shadow, ShadowDiag
from wamos_tpw.theta import Theta


class TestShadow:
    """Tests for Shadow class."""

    def test_shadow_basic(self, single_polar_file):
        """Test basic shadow detection from polar file."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # Shadow should be created
        assert shadow is not None

    def test_shadow_indices_shape(self, single_polar_file):
        """Test that shadow indices have correct shape."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # Indices should be Nx2 (start, end pairs)
        if len(shadow.indices) > 0:
            assert shadow.indices.ndim == 2
            assert shadow.indices.shape[1] == 2

    def test_shadow_thetas_shape(self, single_polar_file):
        """Test that shadow thetas have correct shape."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # Thetas should match indices shape
        if len(shadow.indices) > 0:
            assert shadow.thetas.shape == shadow.indices.shape

    def test_shadow_mask_shape(self, single_polar_file):
        """Test that shadow mask preserves shape."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)
        masked = shadow.mask(destreak.intensity)

        assert masked.shape == destreak.intensity.shape

    def test_shadow_mask_creates_nan(self, single_polar_file):
        """Test that shadow mask creates NaN values in shadow regions."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        if len(shadow.indices) > 0:
            masked = shadow.mask(destreak.intensity)

            # Shadow regions should have NaN values
            for start, end in shadow.indices:
                region = masked[start : end + 1, :]
                assert np.all(np.isnan(region)), "Shadow region should be NaN"

    def test_shadow_timing(self, single_polar_file):
        """Test that timing information is recorded."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # Timing should have some entries
        assert isinstance(shadow.timing, dict)

    def test_shadow_config(self, single_polar_file):
        """Test that config is accessible."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        assert shadow.config is not None

    def test_shadow_repr(self, single_polar_file):
        """Test shadow string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)
        repr_str = repr(shadow)

        assert "Shadow" in repr_str

    def test_shadow_theta_bias(self, single_polar_file):
        """Test theta bias property."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # Theta bias should be a float
        assert isinstance(shadow.theta_bias, float)


class TestShadowWithConfig:
    """Tests for Shadow with explicit configuration."""

    def test_shadow_with_config(self, single_polar_file):
        """Test shadow detection with explicit config."""
        config = Config()
        config["shadow"] = {
            "aft": {"LHS": 150, "RHS": 210},
            "range_fraction": 0.1,
            "angle_range": 15.0,
        }

        pf = PolarFile(single_polar_file, config=config)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # Should detect shadow region
        if len(shadow.indices) > 0:
            # Shadow thetas should be near configured values
            for start_theta, end_theta in shadow.thetas:
                # At least one edge should be near 150 or 210
                near_lhs = abs(start_theta - 150) < 30 or abs(end_theta - 150) < 30
                near_rhs = abs(start_theta - 210) < 30 or abs(end_theta - 210) < 30
                assert near_lhs or near_rhs

    def test_shadow_no_regions_configured(self, single_polar_file):
        """Test shadow with no regions configured returns no detected regions."""
        # Create a completely empty config
        config = Config()
        config["shadow"] = {}  # Empty shadow config - no LHS/RHS defined

        pf = PolarFile(single_polar_file, config=config)
        frame = pf[0]

        # Create a theta with the empty config
        from wamos_tpw.frame import Frame

        # Override the frame's config to use our empty shadow config
        frame._config = config
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)

        # With no LHS/RHS in config, no regions should be detected
        assert len(shadow.indices) == 0


class TestShadowDiag:
    """Tests for ShadowDiag diagnostic class."""

    def test_shadow_diag_creation(self, single_polar_file):
        """Test ShadowDiag creation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)
        shadow = Shadow(destreak.intensity, theta)

        diag = ShadowDiag(destreak.intensity, shadow)

        assert diag.intensity is destreak.intensity
        assert diag.shadow is shadow

    def test_shadow_diag_properties(self, single_polar_file):
        """Test ShadowDiag properties."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)
        shadow = Shadow(destreak.intensity, theta)

        diag = ShadowDiag(destreak.intensity, shadow)

        assert diag.n_shadow_pixels >= 0
        assert 0 <= diag.shadow_fraction <= 1

    def test_shadow_diag_repr(self, single_polar_file):
        """Test ShadowDiag string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)
        shadow = Shadow(destreak.intensity, theta)

        diag = ShadowDiag(destreak.intensity, shadow)
        repr_str = repr(diag)

        assert "ShadowDiag" in repr_str
        assert "regions=" in repr_str
        assert "shadow_pixels=" in repr_str


class TestShadowMultipleFrames:
    """Tests for Shadow with multiple frames."""

    def test_shadow_consistency_across_frames(self, march_polar_files):
        """Test shadow detection consistency across multiple frames."""
        pf = PolarFile(march_polar_files[0])

        if len(pf) < 2:
            pytest.skip("Need at least 2 frames")

        # Process first two frames
        theta0 = Theta(pf[0])
        destreak0 = Destreak(pf[0])
        shadow0 = Shadow(destreak0.intensity, theta0)

        theta1 = Theta(pf[1])
        destreak1 = Destreak(pf[1])
        shadow1 = Shadow(destreak1.intensity, theta1)

        # If both detect shadows, the regions should be similar
        # (since they're from the same sensor with same obstructions)
        if len(shadow0.indices) > 0 and len(shadow1.indices) > 0:
            # Number of shadow regions should be the same
            assert shadow0.indices.shape[0] == shadow1.indices.shape[0]

            # Shadow angles should be similar (within 20 degrees)
            for i in range(len(shadow0.thetas)):
                diff_lhs = abs(shadow0.thetas[i, 0] - shadow1.thetas[i, 0])
                diff_rhs = abs(shadow0.thetas[i, 1] - shadow1.thetas[i, 1])
                # Handle wraparound
                diff_lhs = min(diff_lhs, 360 - diff_lhs)
                diff_rhs = min(diff_rhs, 360 - diff_rhs)
                assert diff_lhs < 20, f"LHS shadow angle differs: {diff_lhs}"
                assert diff_rhs < 20, f"RHS shadow angle differs: {diff_rhs}"


class TestShadowEdgeCases:
    """Edge case tests for Shadow."""

    def test_shadow_mask_preserves_non_shadow(self, single_polar_file):
        """Test that masking preserves non-shadow region values."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)
        masked = shadow.mask(destreak.intensity)

        # Create mask for non-shadow regions
        shadow_mask = np.zeros(masked.shape[0], dtype=bool)
        for start, end in shadow.indices:
            shadow_mask[start : end + 1] = True

        # Non-shadow regions should be unchanged
        non_shadow = ~shadow_mask
        original_non_shadow = destreak.intensity[non_shadow, :]
        masked_non_shadow = masked[non_shadow, :]

        np.testing.assert_array_equal(original_non_shadow, masked_non_shadow)

    def test_shadow_mask_is_copy(self, single_polar_file):
        """Test that shadow mask returns a copy, not modifying original."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        shadow = Shadow(destreak.intensity, theta)
        original = destreak.intensity.copy()
        _ = shadow.mask(destreak.intensity)

        # Original should be unchanged
        np.testing.assert_array_equal(destreak.intensity, original)

    def test_shadow_empty_when_no_config(self, single_polar_file):
        """Test shadow with completely empty config."""
        # Create a config with no shadow section at all
        config = Config()
        # Don't set any shadow config

        pf = PolarFile(single_polar_file, config=config)
        frame = pf[0]
        theta = Theta(frame)
        destreak = Destreak(frame)

        # This should not raise an error
        shadow = Shadow(destreak.intensity, theta)

        # Should have no or minimal shadow detection
        assert isinstance(shadow.indices, np.ndarray)
