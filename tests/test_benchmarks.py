"""
Benchmark tests for wamos_tpw.

Run with: pytest tests/test_benchmarks.py --benchmark-only

Performance Regression Testing
------------------------------

This module includes two types of tests:

1. **Benchmark tests** - Use pytest-benchmark to measure performance
   Run: pytest tests/test_benchmarks.py --benchmark-only

2. **Regression tests** - Assert that operations complete within time limits
   Run: pytest tests/test_benchmarks.py -k regression

The regression tests have conservative thresholds (10x typical) to avoid
false positives while still catching major regressions.
"""

import time

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


class TestProcessingBenchmarks:
    """Benchmark tests for deramp and destreak processing."""

    def test_deramp_processing(self, benchmark, single_polar_file):
        """Benchmark deramp processing on real data."""
        from wamos_tpw.deramp import Deramp
        from wamos_tpw.polarfile import PolarFile
        from wamos_tpw.range import Range

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        def process():
            deramp = Deramp(intensity.copy(), rng)
            return deramp.intensity

        result = benchmark(process)
        assert result.shape == frame.intensity.shape

    def test_destreak_processing(self, benchmark, single_polar_file):
        """Benchmark destreak processing on real data."""
        from wamos_tpw.destreak import Destreak
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        def process():
            ds = Destreak(frame)
            return ds.intensity

        result = benchmark(process)
        assert result.shape == frame.intensity.shape


class TestFileBenchmarks:
    """Benchmark tests for file I/O operations."""

    def test_polar_file_loading(self, benchmark, single_polar_file):
        """Benchmark polar file loading."""
        from wamos_tpw.polarfile import PolarFile

        result = benchmark(PolarFile, single_polar_file)
        assert len(result.frames) > 0

    def test_polar_file_metadata_only(self, benchmark, single_polar_file):
        """Benchmark metadata-only loading."""
        from wamos_tpw.polarfile import PolarFile

        result = benchmark(PolarFile, single_polar_file, metadata_only=True)
        assert result.header is not None


class TestGriddingBenchmarks:
    """Benchmark tests for gridding operations."""

    @pytest.fixture
    def grid_data(self):
        """Create sample data for gridding."""
        n_points = 100000
        x = np.random.uniform(-5000, 5000, n_points)
        y = np.random.uniform(-5000, 5000, n_points)
        values = np.random.rand(n_points)
        return x, y, values

    def test_histogram2d_gridding(self, benchmark, grid_data):
        """Benchmark histogram2d for gridding."""
        x, y, values = grid_data
        x_edges = np.linspace(-5000, 5000, 201)
        y_edges = np.linspace(-5000, 5000, 201)

        def grid():
            counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
            sums, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
            with np.errstate(invalid="ignore"):
                return sums / counts

        result = benchmark(grid)
        assert result.shape == (200, 200)

    def test_searchsorted_binning(self, benchmark, grid_data):
        """Benchmark searchsorted for bin assignment."""
        x, y, _ = grid_data
        x_edges = np.linspace(-5000, 5000, 201)

        def bin_assign():
            return np.searchsorted(x_edges, x) - 1

        result = benchmark(bin_assign)
        assert len(result) == len(x)


class TestEndToEndBenchmarks:
    """Benchmark tests for end-to-end processing pipelines."""

    def test_theta_calculation(self, benchmark, single_polar_file):
        """Benchmark Theta calculation."""
        from wamos_tpw.bearing import MultiTheta as Theta
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]  # Single frame for speed
        config = WamosConfig()

        def compute():
            return Theta(frames, config, refine=False)

        result = benchmark(compute)
        assert result is not None

    def test_bearing_calculation(self, benchmark, single_polar_file):
        """Benchmark Bearing coordinate calculation."""
        from wamos_tpw.bearing import MultiTheta as Theta, MultiBearing as Bearing
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        config = WamosConfig()
        theta = Theta(frames, config, refine=False)

        def compute():
            bearing = Bearing(theta, radar_height=25.0, cache_coordinates=False)
            return bearing.xy_earth(0)

        x, y = benchmark(compute)
        assert x.shape == frames[0].intensity.shape


class TestNormalizationBenchmarks:
    """Benchmark tests for normalization operations."""

    def test_percentile_calculation(self, benchmark):
        """Benchmark percentile calculation on large array."""
        data = np.random.rand(360 * 752 * 10)

        def compute():
            return np.percentile(data, [1, 99])

        result = benchmark(compute)
        assert len(result) == 2


# =============================================================================
# Performance Regression Tests
# =============================================================================
#
# These tests verify that operations complete within reasonable time limits.
# They use conservative thresholds (typically 10x expected time) to avoid
# false positives on slow CI machines while still catching major regressions.
# =============================================================================


class TestPerformanceRegression:
    """
    Performance regression tests with time assertions.

    These tests don't require pytest-benchmark - they use simple time checks
    to catch major performance regressions.
    """

    @pytest.fixture
    def frame_data(self):
        """Create standard frame data for testing."""
        n_bearings, n_distances = 360, 752
        raw_data = np.random.randint(0, 65535, (n_bearings, n_distances), dtype=np.uint16)
        return raw_data

    def test_regression_intensity_extraction(self, frame_data):
        """Intensity extraction should complete in < 50ms (typically ~1ms)."""
        from wamos_tpw.frame import Frame, FrameMetadata

        metadata = FrameMetadata(
            timestamp=np.datetime64("2024-12-15T10:30:00"),
            filename="test.pol",
            samples_in_range=752,
        )
        frame = Frame(frame_data, metadata, validate=False)

        start = time.perf_counter()
        _ = frame.intensity
        elapsed = time.perf_counter() - start

        assert elapsed < 0.050, f"Intensity extraction took {elapsed:.3f}s (limit: 0.050s)"

    def test_regression_bit_extraction(self, frame_data):
        """Bit extraction should complete in < 50ms (typically ~1ms)."""
        from wamos_tpw.frame import Frame, FrameMetadata

        metadata = FrameMetadata(
            timestamp=np.datetime64("2024-12-15T10:30:00"),
            filename="test.pol",
            samples_in_range=752,
        )
        frame = Frame(frame_data, metadata, validate=False)

        start = time.perf_counter()
        _ = frame.bit13
        elapsed = time.perf_counter() - start

        assert elapsed < 0.050, f"Bit extraction took {elapsed:.3f}s (limit: 0.050s)"

    def test_regression_slant_range(self, frame_data):
        """Slant range calculation should complete in < 10ms."""
        from wamos_tpw.frame import Frame, FrameMetadata

        metadata = FrameMetadata(
            timestamp=np.datetime64("2024-12-15T10:30:00"),
            filename="test.pol",
            samples_in_range=752,
            sampling_frequency=20.0,
            sample_delay_range=150.0,
        )
        frame = Frame(frame_data, metadata, validate=False)

        start = time.perf_counter()
        _ = frame.slant_range()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.010, f"Slant range took {elapsed:.3f}s (limit: 0.010s)"

    def test_regression_numpy_operations(self):
        """Standard numpy operations should meet expected performance."""
        arr = np.random.rand(360, 752).astype(np.float64)

        # Mean along axis (should be < 5ms, typically ~0.1ms)
        start = time.perf_counter()
        for _ in range(100):
            _ = np.mean(arr, axis=1)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.500, f"100 mean ops took {elapsed:.3f}s (limit: 0.500s)"

        # Percentile (should be < 50ms, typically ~5ms)
        start = time.perf_counter()
        _ = np.percentile(arr, [5, 95], axis=1)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.050, f"Percentile took {elapsed:.3f}s (limit: 0.050s)"

    def test_regression_coordinate_transform(self):
        """Coordinate transformation should complete in < 100ms."""
        bearings = np.linspace(0, 359.9, 360)
        ranges = np.linspace(150, 6000, 752)

        start = time.perf_counter()
        heading_rad = np.deg2rad(bearings)[:, np.newaxis]
        range_2d = ranges[np.newaxis, :]
        x = range_2d * np.sin(heading_rad)
        y = range_2d * np.cos(heading_rad)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.100, f"Coordinate transform took {elapsed:.3f}s (limit: 0.100s)"
        assert x.shape == (360, 752)
        assert y.shape == (360, 752)

    def test_regression_histogram2d(self):
        """Histogram2d gridding should complete in < 200ms for 100k points."""
        n_points = 100000
        x = np.random.uniform(-5000, 5000, n_points)
        y = np.random.uniform(-5000, 5000, n_points)
        values = np.random.rand(n_points)
        x_edges = np.linspace(-5000, 5000, 201)
        y_edges = np.linspace(-5000, 5000, 201)

        start = time.perf_counter()
        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        sums, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
        with np.errstate(invalid="ignore"):
            result = sums / counts
        elapsed = time.perf_counter() - start

        assert elapsed < 0.200, f"Histogram2d took {elapsed:.3f}s (limit: 0.200s)"
        assert result.shape == (200, 200)


class TestProcessingRegression:
    """Regression tests for processing modules using real data."""

    def test_regression_polar_file_loading(self, single_polar_file):
        """Polar file loading should complete in < 2s (includes decompression)."""
        from wamos_tpw.polarfile import PolarFile

        start = time.perf_counter()
        pf = PolarFile(single_polar_file)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"PolarFile loading took {elapsed:.3f}s (limit: 2.0s)"
        assert len(pf.frames) > 0

    def test_regression_destreak(self, single_polar_file):
        """Destreak should complete in < 500ms per frame."""
        from wamos_tpw.destreak import Destreak
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        start = time.perf_counter()
        ds = Destreak(frame)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.500, f"Destreak took {elapsed:.3f}s (limit: 0.500s)"
        assert ds.intensity.shape == frame.intensity.shape

    def test_regression_deramp(self, single_polar_file):
        """Deramp should complete in < 500ms per frame."""
        from wamos_tpw.deramp import Deramp
        from wamos_tpw.polarfile import PolarFile
        from wamos_tpw.range import Range

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        intensity = frame.intensity.astype(np.float32)
        rng = Range(frame)

        start = time.perf_counter()
        deramp = Deramp(intensity, rng)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.500, f"Deramp took {elapsed:.3f}s (limit: 0.500s)"
        assert deramp.intensity.shape == frame.intensity.shape

    def test_regression_theta_calculation(self, single_polar_file):
        """Theta calculation should complete in < 1s."""
        from wamos_tpw.bearing import MultiTheta as Theta
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        config = WamosConfig()

        start = time.perf_counter()
        theta = Theta(frames, config, refine=False)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Theta calculation took {elapsed:.3f}s (limit: 1.0s)"
        assert theta is not None
