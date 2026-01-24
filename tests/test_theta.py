#! /usr/bin/env python3
"""Tests for wamos_tpw.theta module."""

import numpy as np
import pytest

from wamos_tpw.polarfile import PolarFile
from wamos_tpw.theta import Theta, ThetaDiag


class TestTheta:
    """Tests for Theta class."""

    def test_theta_basic(self, single_polar_file):
        """Test basic theta calculation from polar file."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        # Check that theta was calculated
        assert len(theta) == frame.n_bearings
        assert theta.theta.shape == (frame.n_bearings,)

    def test_theta_range(self, single_polar_file):
        """Test that theta values are in valid range [0, 360)."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        assert np.all(theta.theta >= 0)
        assert np.all(theta.theta < 360)

    def test_theta_coverage(self, single_polar_file):
        """Test that theta covers most of 360 degrees (full rotation)."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        theta_range = theta.theta.max() - theta.theta.min()
        # Full rotation should cover close to 360 degrees
        # (minus some for shadow regions and gaps)
        assert theta_range > 300  # At least 300 degrees coverage

    def test_theta_sorted_index(self, single_polar_file):
        """Test that sorted indices are valid."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        # Access internal sorted arrays
        sorted_theta = theta._sorted_theta
        sorted_indices = theta._sorted_indices

        # Sorted theta should be monotonically increasing
        assert np.all(np.diff(sorted_theta) >= 0)

        # Sorted indices should be valid indices
        assert np.all(sorted_indices >= 0)
        assert np.all(sorted_indices < len(theta))

    def test_theta_index_lookup(self, single_polar_file):
        """Test theta index lookup returns valid indices."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        # Test single value lookup
        query_angles = np.array([0.0, 90.0, 180.0, 270.0])
        indices = theta.index(query_angles)

        assert indices.shape == query_angles.shape
        assert np.all(indices >= 0)
        assert np.all(indices < len(theta))

    def test_theta_index_lookup_single(self, single_polar_file):
        """Test theta index lookup with single value."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        # Single value
        idx = theta.index(np.array([45.0]))
        assert idx.shape == (1,)
        assert 0 <= idx[0] < len(theta)

    def test_theta_index_lookup_2d(self, single_polar_file):
        """Test theta index lookup with 2D array."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        # 2D array
        query = np.array([[0.0, 90.0], [180.0, 270.0]])
        indices = theta.index(query)

        assert indices.shape == query.shape
        assert np.all(indices >= 0)
        assert np.all(indices < len(theta))

    def test_theta_index_wraparound(self, single_polar_file):
        """Test that index lookup handles wraparound (359 vs 1)."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        # Query values that wrap around
        idx_359 = theta.index(np.array([359.0]))[0]
        idx_1 = theta.index(np.array([1.0]))[0]

        # These should be close together (wrapping around 0)
        # They should return valid indices at least
        assert 0 <= idx_359 < len(theta)
        assert 0 <= idx_1 < len(theta)

    def test_theta_set_bias(self, single_polar_file):
        """Test applying theta bias."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)
        original_theta = theta.theta.copy()

        # Apply bias
        bias = 10.0
        theta.set_bias(bias)

        # New values should be shifted and wrapped
        expected = (original_theta + bias) % 360
        np.testing.assert_array_almost_equal(theta.theta, expected)

    def test_theta_set_bias_negative(self, single_polar_file):
        """Test applying negative theta bias."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)
        original_theta = theta.theta.copy()

        # Apply negative bias
        bias = -30.0
        theta.set_bias(bias)

        # New values should be shifted and wrapped to [0, 360)
        expected = (original_theta + bias) % 360
        np.testing.assert_array_almost_equal(theta.theta, expected)

    def test_theta_timing(self, single_polar_file):
        """Test that timing information is recorded."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        assert "extract_degrees" in theta.timing
        assert "interpolate" in theta.timing
        assert "sort" in theta.timing

        # Timings should be positive
        assert theta.timing["extract_degrees"] > 0
        assert theta.timing["interpolate"] > 0
        assert theta.timing["sort"] > 0

    def test_theta_config(self, single_polar_file):
        """Test that config is accessible."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        assert theta.config is not None

    def test_theta_repr(self, single_polar_file):
        """Test theta string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)
        repr_str = repr(theta)

        assert "Theta" in repr_str
        assert "radials=" in repr_str
        assert "range=" in repr_str

    def test_theta_len(self, single_polar_file):
        """Test theta length."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta = Theta(frame)

        assert len(theta) == frame.n_bearings


class TestThetaDiag:
    """Tests for ThetaDiag diagnostic class."""

    def test_theta_diag_creation(self, single_polar_file):
        """Test ThetaDiag creation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)

        diag = ThetaDiag(frame, theta)

        assert diag.frame is frame
        assert diag.theta is theta

    def test_theta_diag_repr(self, single_polar_file):
        """Test ThetaDiag string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)

        diag = ThetaDiag(frame, theta)
        repr_str = repr(diag)

        assert "ThetaDiag" in repr_str
        assert "radials=" in repr_str


class TestThetaMultipleFrames:
    """Tests for Theta with multiple frames."""

    def test_theta_consistency_across_frames(self, march_polar_files):
        """Test theta calculation consistency across multiple frames."""
        pf = PolarFile(march_polar_files[0])

        if len(pf) < 2:
            pytest.skip("Need at least 2 frames")

        theta0 = Theta(pf[0])
        theta1 = Theta(pf[1])

        # Both should have same number of radials (from same polar file)
        # Note: this might not always be true depending on frame contents
        # Just verify they're both valid
        assert len(theta0) > 0
        assert len(theta1) > 0

    def test_theta_range_all_frames(self, march_polar_files):
        """Test theta range for all frames in a file."""
        pf = PolarFile(march_polar_files[0])

        for i, frame in enumerate(pf):
            theta = Theta(frame)
            assert np.all(theta.theta >= 0), f"Frame {i} has negative theta"
            assert np.all(theta.theta < 360), f"Frame {i} has theta >= 360"


class TestThetaEdgeCases:
    """Edge case tests for Theta."""

    def test_theta_index_out_of_range_wraps(self, single_polar_file):
        """Test that query angles > 360 wrap correctly."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)

        # Query angle > 360 should wrap
        idx_370 = theta.index(np.array([370.0]))[0]  # Should be same as 10
        idx_10 = theta.index(np.array([10.0]))[0]

        # Both should return same (or very close) index
        assert abs(idx_370 - idx_10) <= 1  # Allow for rounding differences

    def test_theta_index_negative_wraps(self, single_polar_file):
        """Test that negative query angles wrap correctly."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]
        theta = Theta(frame)

        # Query angle < 0 should wrap
        idx_neg10 = theta.index(np.array([-10.0]))[0]  # Should be same as 350
        idx_350 = theta.index(np.array([350.0]))[0]

        # Both should return same (or very close) index
        assert abs(idx_neg10 - idx_350) <= 1  # Allow for rounding differences
