"""
Benchmark tests for wamos_tpw.

Run with: pytest tests/test_benchmarks.py --benchmark-only
"""

import numpy as np
import pytest

from wamos_tpw.frame import Frame, FrameMetadata


# Skip if pytest-benchmark not installed
pytest.importorskip("pytest_benchmark")


@pytest.fixture
def sample_frame():
    """Create a sample frame for benchmarking."""
    n_bearings, n_distances = 360, 752
    raw_data = np.random.randint(0, 65535, (n_bearings, n_distances), dtype=np.uint16)
    metadata = FrameMetadata(
        timestamp=np.datetime64("2024-12-15T10:30:00"),
        filename="test_benchmark.pol",
        latitude=18.57,
        longitude=142.96,
        samples_in_range=n_distances,
        sampling_frequency=20.0,
        sample_delay_range=150.0,
        radar_height=25.0,
    )
    return Frame(raw_data, metadata, validate=False)


@pytest.fixture
def large_frame():
    """Create a large frame for benchmarking."""
    n_bearings, n_distances = 720, 1500
    raw_data = np.random.randint(0, 65535, (n_bearings, n_distances), dtype=np.uint16)
    metadata = FrameMetadata(
        timestamp=np.datetime64("2024-12-15T10:30:00"),
        filename="test_benchmark_large.pol",
        latitude=18.57,
        longitude=142.96,
        samples_in_range=n_distances,
        sampling_frequency=20.0,
        sample_delay_range=150.0,
        radar_height=25.0,
    )
    return Frame(raw_data, metadata, validate=False)


class TestFrameBenchmarks:
    """Benchmark tests for Frame class operations."""

    def test_intensity_extraction(self, benchmark, sample_frame):
        """Benchmark intensity extraction (bottom 12 bits)."""
        # Clear cache first
        sample_frame.clear_cache()

        def extract():
            sample_frame._intensity = None  # Force recomputation
            return sample_frame.intensity

        result = benchmark(extract)
        assert result.shape == (360, 752)

    def test_bit_extraction(self, benchmark, sample_frame):
        """Benchmark bit 13 extraction."""
        sample_frame.clear_cache()

        def extract():
            sample_frame._bit13 = None
            return sample_frame.bit13

        result = benchmark(extract)
        assert result.shape == (360, 752)

    def test_slant_range_calculation(self, benchmark, sample_frame):
        """Benchmark slant range calculation."""
        result = benchmark(sample_frame.slant_range)
        assert len(result) == 752

    def test_ground_range_calculation(self, benchmark, sample_frame):
        """Benchmark ground range calculation."""
        result = benchmark(sample_frame.ground_range)
        assert len(result) == 752

    def test_distance_row_extraction(self, benchmark, sample_frame):
        """Benchmark distance row extraction."""
        result = benchmark(sample_frame.get_distance_row, 100, "intensity")
        assert len(result) == 360


class TestLargeFrameBenchmarks:
    """Benchmark tests for large frame operations."""

    def test_large_intensity_extraction(self, benchmark, large_frame):
        """Benchmark intensity extraction on large frame."""
        large_frame.clear_cache()

        def extract():
            large_frame._intensity = None
            return large_frame.intensity

        result = benchmark(extract)
        assert result.shape == (720, 1500)

    def test_large_slant_range(self, benchmark, large_frame):
        """Benchmark slant range on large frame."""
        result = benchmark(large_frame.slant_range)
        assert len(result) == 1500


class TestArrayOperations:
    """Benchmark basic numpy operations used in processing."""

    @pytest.fixture
    def intensity_array(self):
        """Create sample intensity array."""
        return np.random.rand(360, 752).astype(np.float64)

    def test_mean_computation(self, benchmark, intensity_array):
        """Benchmark mean computation."""
        result = benchmark(np.mean, intensity_array, axis=1)
        assert len(result) == 360

    def test_std_computation(self, benchmark, intensity_array):
        """Benchmark std computation."""
        result = benchmark(np.std, intensity_array, axis=1)
        assert len(result) == 360

    def test_percentile_computation(self, benchmark, intensity_array):
        """Benchmark percentile computation."""

        def compute_percentiles():
            return np.percentile(intensity_array, [5, 95], axis=1)

        result = benchmark(compute_percentiles)
        assert result.shape == (2, 360)

    def test_gaussian_filter(self, benchmark, intensity_array):
        """Benchmark Gaussian filtering."""
        from scipy.ndimage import gaussian_filter1d

        result = benchmark(gaussian_filter1d, intensity_array[0], sigma=3)
        assert len(result) == 752


class TestThetaCalculations:
    """Benchmark tests for bearing calculations."""

    @pytest.fixture
    def bearing_array(self):
        """Create sample bearing array."""
        return np.linspace(0, 359.9, 360)

    def test_trig_calculations(self, benchmark, bearing_array):
        """Benchmark sin/cos calculations."""

        def compute():
            rad = np.deg2rad(bearing_array)
            return np.sin(rad), np.cos(rad)

        result = benchmark(compute)
        assert len(result[0]) == 360

    def test_coordinate_transform(self, benchmark, bearing_array):
        """Benchmark coordinate transformation."""
        ranges = np.linspace(150, 6000, 752)

        def transform():
            heading_rad = np.deg2rad(bearing_array)[:, np.newaxis]
            range_2d = ranges[np.newaxis, :]
            x = range_2d * np.sin(heading_rad)
            y = range_2d * np.cos(heading_rad)
            return x, y

        x, y = benchmark(transform)
        assert x.shape == (360, 752)
