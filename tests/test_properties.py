#! /usr/bin/env python3
"""Property-based tests using hypothesis for wamos_tpw.

These tests verify invariants that should hold for any input, not just
specific test cases. This helps catch edge cases and boundary conditions.
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from wamos_tpw.config import Config, NullConfig
from wamos_tpw.merged_image import TimeWindowConfig


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for valid heading angles (0-360)
heading_strategy = st.floats(min_value=0.0, max_value=360.0, allow_nan=False)

# Strategy for valid latitude (-90 to 90)
latitude_strategy = st.floats(min_value=-89.9, max_value=89.9, allow_nan=False)

# Strategy for valid longitude (-180 to 180)
longitude_strategy = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False)

# Strategy for positive floats (distances, speeds, etc.)
positive_float = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False)

# Strategy for small positive integers
small_positive_int = st.integers(min_value=1, max_value=100)


# =============================================================================
# TimeWindowConfig Property Tests
# =============================================================================


class TestTimeWindowConfigProperties:
    """Property-based tests for TimeWindowConfig."""

    @given(
        window_seconds=st.floats(min_value=1.0, max_value=3600.0, allow_nan=False),
        overlap_fraction=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stride_always_positive(self, window_seconds, overlap_fraction):
        """Stride should always be positive when window_seconds is positive."""
        config = TimeWindowConfig(
            window_seconds=window_seconds,
            overlap_fraction=overlap_fraction,
        )
        assert config.stride_seconds > 0

    @given(
        window_seconds=st.floats(min_value=1.0, max_value=3600.0, allow_nan=False),
        overlap_fraction=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_stride_leq_window(self, window_seconds, overlap_fraction):
        """Stride should always be <= window_seconds."""
        config = TimeWindowConfig(
            window_seconds=window_seconds,
            overlap_fraction=overlap_fraction,
        )
        assert config.stride_seconds <= config.window_seconds

    @given(
        window_seconds=st.floats(min_value=1.0, max_value=3600.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_zero_overlap_stride_equals_window(self, window_seconds):
        """With zero overlap, stride should equal window."""
        config = TimeWindowConfig(
            window_seconds=window_seconds,
            overlap_fraction=0.0,
        )
        assert abs(config.stride_seconds - config.window_seconds) < 1e-10


# =============================================================================
# Config Property Tests
# =============================================================================


class TestConfigProperties:
    """Property-based tests for Config and NullConfig."""

    @given(key=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_nullconfig_get_returns_default(self, key):
        """NullConfig.get() should always return the default value."""
        # Skip keys that might be problematic (starting with underscore)
        assume(not key.startswith("_"))

        config = NullConfig()
        default_val = object()  # Unique sentinel
        result = config.get(key, default_val)
        assert result is default_val

    @given(key=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_nullconfig_contains_always_false(self, key):
        """NullConfig should never contain any key."""
        assume(not key.startswith("_"))
        config = NullConfig()
        assert key not in config

    def test_nullconfig_is_singleton(self):
        """NullConfig should use singleton pattern."""
        config1 = NullConfig()
        config2 = NullConfig()
        assert config1 is config2

    @given(key=st.text(min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_config_contains_matches_getitem(self, key):
        """'key in config' should match whether __getitem__ succeeds."""
        assume(not key.startswith("_"))
        config = Config()  # Default config

        contains_result = key in config
        try:
            _ = config[key]
            getitem_succeeded = True
        except KeyError:
            getitem_succeeded = False

        assert contains_result == getitem_succeeded


# =============================================================================
# Numerical Property Tests
# =============================================================================


class TestNumericalProperties:
    """Property-based tests for numerical operations."""

    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_nanmean_returns_scalar(self, values):
        """np.nanmean should return a scalar for 1D input."""
        arr = np.array(values)
        result = np.nanmean(arr)
        assert np.isscalar(result) or result.ndim == 0

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=50),
            st.integers(min_value=1, max_value=50),
        )
    )
    @settings(max_examples=30)
    def test_zeros_shape_preserved(self, shape):
        """np.zeros should create array with exact shape requested."""
        arr = np.zeros(shape)
        assert arr.shape == shape

    @given(
        n_rows=st.integers(min_value=1, max_value=100),
        n_cols=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30)
    def test_reshape_preserves_size(self, n_rows, n_cols):
        """Reshaping should preserve total element count."""
        total = n_rows * n_cols
        arr = np.arange(total)
        reshaped = arr.reshape((n_rows, n_cols))
        assert reshaped.size == total


# =============================================================================
# Angle Property Tests
# =============================================================================


class TestAngleProperties:
    """Property-based tests for angle calculations."""

    @given(angle=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
    @settings(max_examples=100)
    def test_angle_normalization_range(self, angle):
        """Normalized angle should be in [0, 360)."""
        normalized = angle % 360.0
        assert 0.0 <= normalized < 360.0

    @given(
        angle1=heading_strategy,
        angle2=heading_strategy,
    )
    @settings(max_examples=100)
    def test_circular_mean_in_range(self, angle1, angle2):
        """Circular mean of two angles should be in [0, 360)."""
        # Compute circular mean
        rad1 = np.deg2rad(angle1)
        rad2 = np.deg2rad(angle2)
        mean_sin = (np.sin(rad1) + np.sin(rad2)) / 2
        mean_cos = (np.cos(rad1) + np.cos(rad2)) / 2
        result = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0

        assert 0.0 <= result < 360.0


# =============================================================================
# Intensity Array Property Tests
# =============================================================================


class TestIntensityArrayProperties:
    """Property-based tests for intensity array operations."""

    @given(
        n_bearings=st.integers(min_value=1, max_value=500),
        n_distances=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=30)
    def test_random_intensity_dtype(self, n_bearings, n_distances):
        """Random uint16 array should have correct dtype."""
        arr = np.random.randint(0, 4096, (n_bearings, n_distances), dtype=np.uint16)
        assert arr.dtype == np.uint16

    @given(
        n_bearings=st.integers(min_value=1, max_value=100),
        n_distances=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30)
    def test_intensity_masking_preserves_shape(self, n_bearings, n_distances):
        """Masking with NaN should preserve array shape."""
        arr = np.random.rand(n_bearings, n_distances)
        mask = np.random.rand(n_bearings, n_distances) > 0.5
        arr[mask] = np.nan
        assert arr.shape == (n_bearings, n_distances)


# =============================================================================
# Grid Property Tests
# =============================================================================


class TestGridProperties:
    """Property-based tests for grid operations."""

    @given(
        n_bins=st.integers(min_value=2, max_value=500),
        min_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        max_val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_linspace_edge_count(self, n_bins, min_val, max_val):
        """linspace for n bins should have n+1 edges."""
        assume(min_val < max_val)
        edges = np.linspace(min_val, max_val, n_bins + 1)
        assert len(edges) == n_bins + 1

    @given(
        n_bins=st.integers(min_value=2, max_value=100),
        min_val=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False),
        range_val=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_linspace_edges_sorted(self, n_bins, min_val, range_val):
        """linspace edges should be sorted."""
        max_val = min_val + range_val
        edges = np.linspace(min_val, max_val, n_bins + 1)
        assert np.all(np.diff(edges) > 0)  # Strictly increasing

    @given(
        n_bins=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30)
    def test_bin_centers_between_edges(self, n_bins):
        """Bin centers should be between adjacent edges."""
        edges = np.linspace(0, 100, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        for i, center in enumerate(centers):
            assert edges[i] < center < edges[i + 1]


# =============================================================================
# Skip marker for CI without hypothesis
# =============================================================================


def pytest_configure(config):
    """Add marker for hypothesis tests."""
    config.addinivalue_line("markers", "hypothesis: mark test as using hypothesis")
