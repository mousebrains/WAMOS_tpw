#! /usr/bin/env python3
"""Tests for wamos_tpw.range module."""

import numpy as np
import pytest

from wamos_tpw.config import Config
from wamos_tpw.constants import C_AIR
from wamos_tpw.frame import Frame, FrameMetadata
from wamos_tpw.range import Range


def _make_frame(
    *,
    n_bearings: int = 10,
    n_distances: int = 100,
    sampling_frequency: float = 25.0,
    sample_delay_range: float = 50.0,
    radar_height: float | None = 20.0,
    range_bias: float = 0.0,
    validate: bool = True,
) -> Frame:
    """Create a synthetic Frame for range tests."""
    data = np.zeros((n_bearings, n_distances), dtype=np.uint16)
    metadata = FrameMetadata(
        timestamp=np.datetime64("2022-04-05T14:30:00"),
        filename="test_frame.pol",
        samples_in_range=n_distances,
        sampling_frequency=sampling_frequency,
        sample_delay_range=sample_delay_range,
        radar_height=radar_height,
    )
    config = Config()
    if radar_height is not None:
        config["tower.height"] = radar_height
    if range_bias != 0.0:
        config["bias.range"] = range_bias
    return Frame(data, metadata, config=config, validate=validate)


class TestRangeResolution:
    """Tests for range resolution calculation."""

    def test_standard_sampling_frequency(self):
        """Range resolution with typical 25 MHz sampling."""
        frame = _make_frame(sampling_frequency=25.0)
        rng = Range(frame)
        # c_air / (2 * 25e6) ≈ 5.994 m
        expected = C_AIR / (2.0 * 25e6)
        assert rng.range_resolution == pytest.approx(expected, rel=1e-6)

    def test_higher_sampling_frequency(self):
        """Higher sampling frequency gives finer resolution."""
        frame_25 = _make_frame(sampling_frequency=25.0)
        frame_50 = _make_frame(sampling_frequency=50.0)
        rng_25 = Range(frame_25)
        rng_50 = Range(frame_50)
        assert rng_50.range_resolution == pytest.approx(rng_25.range_resolution / 2, rel=1e-6)

    def test_zero_sampling_frequency_returns_zero(self):
        """Zero sampling frequency returns 0 range resolution with warning."""
        frame = _make_frame(sampling_frequency=0.0)
        rng = Range(frame)
        assert rng.range_resolution == 0.0

    def test_negative_sampling_frequency_returns_zero(self):
        """Negative sampling frequency returns 0 range resolution."""
        frame = _make_frame(sampling_frequency=-1.0, validate=False)
        rng = Range(frame)
        assert rng.range_resolution == 0.0


class TestSlantRange:
    """Tests for slant range calculation."""

    def test_shape(self):
        """Slant range array has correct shape."""
        n_distances = 50
        frame = _make_frame(n_distances=n_distances)
        rng = Range(frame)
        assert rng.slant_range.shape == (n_distances,)

    def test_first_bin_equals_sample_delay_range(self):
        """First bin slant range equals sample_delay_range."""
        sdr = 75.0
        frame = _make_frame(sample_delay_range=sdr)
        rng = Range(frame)
        assert rng.slant_range[0] == pytest.approx(sdr)

    def test_monotonically_increasing(self):
        """Slant range is monotonically increasing."""
        frame = _make_frame(sampling_frequency=25.0)
        rng = Range(frame)
        diffs = np.diff(rng.slant_range)
        assert np.all(diffs > 0)

    def test_linear_spacing(self):
        """Bins are linearly spaced by range_resolution."""
        frame = _make_frame(sampling_frequency=25.0)
        rng = Range(frame)
        diffs = np.diff(rng.slant_range)
        expected_step = C_AIR / (2.0 * 25e6)
        np.testing.assert_allclose(diffs, expected_step, rtol=1e-10)

    def test_zero_sample_delay(self):
        """Slant range starts at 0 when sample_delay_range is 0."""
        frame = _make_frame(sample_delay_range=0.0)
        rng = Range(frame)
        assert rng.slant_range[0] == pytest.approx(0.0)

    def test_at_bin_method(self):
        """slant_range_at_bin returns correct values."""
        frame = _make_frame(n_distances=20)
        rng = Range(frame)
        for i in range(20):
            assert rng.slant_range_at_bin(i) == rng.slant_range[i]


class TestGroundRange:
    """Tests for ground range calculation."""

    def test_shape(self):
        """Ground range array has correct shape."""
        n_distances = 50
        frame = _make_frame(n_distances=n_distances)
        rng = Range(frame)
        assert rng.ground_range.shape == (n_distances,)

    def test_dtype_is_float32(self):
        """Ground range uses float32 for memory efficiency."""
        frame = _make_frame()
        rng = Range(frame)
        assert rng.ground_range.dtype == np.float32

    def test_ground_less_than_slant(self):
        """Ground range is less than slant range when height > 0."""
        frame = _make_frame(radar_height=20.0, sample_delay_range=100.0)
        rng = Range(frame)
        # All slant ranges should be > height, so ground < slant
        mask = rng.slant_range > 20.0
        assert np.all(rng.ground_range[mask] < rng.slant_range[mask])

    def test_pythagorean_relationship(self):
        """ground^2 + height^2 = slant^2 (within bias)."""
        height = 20.0
        frame = _make_frame(radar_height=height, sample_delay_range=100.0)
        rng = Range(frame)
        mask = rng.slant_range > height
        ground = rng.ground_range[mask].astype(np.float64)
        slant = rng.slant_range[mask]
        computed_slant = np.sqrt(ground**2 + height**2)
        np.testing.assert_allclose(computed_slant, slant, rtol=1e-5)

    def test_zero_height_equals_slant(self):
        """Without radar height, ground range equals slant range."""
        frame = _make_frame(radar_height=None)
        rng = Range(frame)
        np.testing.assert_allclose(rng.ground_range, rng.slant_range.astype(np.float32), rtol=1e-6)

    def test_bins_below_height_are_zero(self):
        """Ground range is 0 for bins where slant < height."""
        # Large height, small sample_delay_range so first bins are < height
        height = 500.0
        frame = _make_frame(radar_height=height, sample_delay_range=10.0, sampling_frequency=25.0)
        rng = Range(frame)
        below_height = rng.slant_range < height
        if np.any(below_height):
            assert np.all(rng.ground_range[below_height] == 0.0)

    def test_range_bias_applied(self):
        """Range bias is added to ground range."""
        bias = 5.0
        frame_no_bias = _make_frame(range_bias=0.0, radar_height=None)
        frame_biased = _make_frame(range_bias=bias, radar_height=None)
        rng_no_bias = Range(frame_no_bias)
        rng_biased = Range(frame_biased)
        expected = rng_no_bias.ground_range + np.float32(bias)
        np.testing.assert_allclose(rng_biased.ground_range, expected, rtol=1e-5)

    def test_at_bin_method(self):
        """ground_range_at_bin returns correct values."""
        frame = _make_frame(n_distances=20)
        rng = Range(frame)
        for i in range(20):
            assert rng.ground_range_at_bin(i) == rng.ground_range[i]


class TestRangeProperties:
    """Tests for Range public properties."""

    def test_len(self):
        """len() returns number of distance bins."""
        n = 42
        frame = _make_frame(n_distances=n)
        rng = Range(frame)
        assert len(rng) == n

    def test_frame_property(self):
        """frame property returns the original Frame."""
        frame = _make_frame()
        rng = Range(frame)
        assert rng.frame is frame

    def test_config_property(self):
        """config property returns the Config object."""
        frame = _make_frame()
        rng = Range(frame)
        assert rng.config is not None

    def test_radar_height_property(self):
        """radar_height returns the configured height."""
        frame = _make_frame(radar_height=25.0)
        rng = Range(frame)
        assert rng.radar_height == 25.0

    def test_radar_height_none_when_not_set(self):
        """radar_height is None when not configured."""
        frame = _make_frame(radar_height=None)
        rng = Range(frame)
        assert rng.radar_height is None

    def test_range_bias_property(self):
        """range_bias returns the configured bias."""
        frame = _make_frame(range_bias=3.5)
        rng = Range(frame)
        assert rng.range_bias == 3.5

    def test_repr(self):
        """repr is informative."""
        frame = _make_frame(n_distances=50)
        rng = Range(frame)
        r = repr(rng)
        assert "Range(" in r
        assert "bins=50" in r
