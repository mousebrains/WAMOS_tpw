#! /usr/bin/env python3
"""Tests for wamos_tpw.window module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wamos_tpw.merged_image import MergedImage, TimeWindowConfig
from wamos_tpw.window import WindowAccumulator, create_time_windows


class TestTimeWindowConfig:
    """Tests for TimeWindowConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TimeWindowConfig()

        assert config.window_seconds == 60.0
        assert config.overlap_fraction == 0.5
        assert config.min_frames_per_window == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TimeWindowConfig(
            window_seconds=120.0,
            overlap_fraction=0.75,
            min_frames_per_window=10,
        )

        assert config.window_seconds == 120.0
        assert config.overlap_fraction == 0.75
        assert config.min_frames_per_window == 10

    def test_stride_calculation(self):
        """Test stride calculation from window and overlap."""
        config = TimeWindowConfig(window_seconds=60.0, overlap_fraction=0.5)
        # stride = window * (1 - overlap) = 60 * 0.5 = 30
        assert config.stride_seconds == 30.0

        config2 = TimeWindowConfig(window_seconds=100.0, overlap_fraction=0.25)
        # stride = 100 * 0.75 = 75
        assert config2.stride_seconds == 75.0


class TestCreateTimeWindows:
    """Tests for create_time_windows function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_test_files(self, temp_dir, timestamps):
        """Create test polar files with given timestamps."""
        files = []
        for ts in timestamps:
            # Format: YYYYMMDDHHmmss_TOWER.pol
            filename = f"{ts}_TOWER.pol"
            filepath = temp_dir / filename
            filepath.touch()
            files.append(str(filepath))
        return files

    def test_empty_files_list(self):
        """Test with empty file list."""
        config = TimeWindowConfig()
        windows = create_time_windows([], config)
        assert windows == []

    def test_single_window(self, temp_dir):
        """Test creating a single window."""
        # Create files spanning 30 seconds (well within 60s window)
        timestamps = [
            "20240115100000",
            "20240115100010",
            "20240115100020",
            "20240115100030",
            "20240115100040",
        ]
        files = self.create_test_files(temp_dir, timestamps)

        config = TimeWindowConfig(
            window_seconds=60.0, overlap_fraction=0.0, min_frames_per_window=3
        )
        windows = create_time_windows(files, config)

        assert len(windows) >= 1
        # First window should contain all files
        start_time, end_time, file_indices = windows[0]
        assert len(file_indices) == 5

    def test_overlapping_windows(self, temp_dir):
        """Test creating overlapping windows."""
        # Create files spanning 90 seconds
        timestamps = [
            "20240115100000",  # 0s
            "20240115100010",  # 10s
            "20240115100020",  # 20s
            "20240115100030",  # 30s
            "20240115100040",  # 40s
            "20240115100050",  # 50s
            "20240115100100",  # 60s
            "20240115100110",  # 70s
            "20240115100120",  # 80s
            "20240115100130",  # 90s
        ]
        files = self.create_test_files(temp_dir, timestamps)

        config = TimeWindowConfig(
            window_seconds=60.0,  # 60s windows
            overlap_fraction=0.5,  # 50% overlap -> 30s stride
            min_frames_per_window=3,
        )
        windows = create_time_windows(files, config)

        # With 90s of data, 60s window, 30s stride:
        # Window 0: [0, 60) - indices 0-5
        # Window 1: [30, 90) - indices 3-8
        # Window 2: [60, 120) - indices 6-9
        assert len(windows) >= 2

        # Check overlapping file membership
        _, _, indices0 = windows[0]
        _, _, indices1 = windows[1]

        # Windows should share some files
        set0 = set(indices0)
        set1 = set(indices1)
        assert len(set0 & set1) > 0  # Some overlap

    def test_min_frames_filtering(self, temp_dir):
        """Test that windows with too few frames are excluded."""
        timestamps = [
            "20240115100000",
            "20240115100010",
            "20240115100020",
        ]
        files = self.create_test_files(temp_dir, timestamps)

        # Require 5 frames minimum (more than we have)
        config = TimeWindowConfig(window_seconds=60.0, min_frames_per_window=5)
        windows = create_time_windows(files, config)

        # No windows should be created
        assert len(windows) == 0

        # Require 3 frames minimum
        config2 = TimeWindowConfig(window_seconds=60.0, min_frames_per_window=3)
        windows2 = create_time_windows(files, config2)

        # Now we should have at least one window
        assert len(windows2) >= 1

    def test_window_time_boundaries(self, temp_dir):
        """Test that window time boundaries are correct."""
        timestamps = [
            "20240115100000",
            "20240115100030",
            "20240115100100",
        ]
        files = self.create_test_files(temp_dir, timestamps)

        config = TimeWindowConfig(
            window_seconds=60.0, overlap_fraction=0.0, min_frames_per_window=1
        )
        windows = create_time_windows(files, config)

        # Check first window boundaries
        start_time, end_time, _ = windows[0]

        # End time should be start + window duration
        expected_duration = np.timedelta64(60, "s")
        actual_duration = end_time - start_time
        assert actual_duration == expected_duration


class TestWindowAccumulator:
    """Tests for WindowAccumulator class."""

    @pytest.fixture
    def simple_accumulator(self):
        """Create a simple accumulator for testing."""
        x_edges = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])  # 4 bins
        y_edges = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])  # 4 bins
        return WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=50.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

    def test_initial_state(self, simple_accumulator):
        """Test accumulator initial state."""
        acc = simple_accumulator

        assert acc.n_frames == 0
        assert acc.n_x == 4
        assert acc.n_y == 4
        assert acc.intensity_sum.shape == (4, 4)
        assert acc.intensity_count.shape == (4, 4)
        assert np.all(acc.intensity_sum == 0)
        assert np.all(acc.intensity_count == 0)

    def test_add_single_frame(self, simple_accumulator):
        """Test adding a single projected frame."""
        acc = simple_accumulator

        # Create projected data
        projected_intensity = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            dtype=np.float64,
        )
        projected_count = np.ones((4, 4), dtype=np.int32)

        acc.add_projected(
            projected_intensity=projected_intensity,
            projected_count=projected_count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=45.0,
            ship_speed=5.0,
            wind_speed=10.0,
            wind_direction=180.0,
        )

        assert acc.n_frames == 1
        assert np.array_equal(acc.intensity_sum, projected_intensity)
        assert np.array_equal(acc.intensity_count, projected_count)
        assert acc.headings == [45.0]
        assert acc.ship_speeds == [5.0]
        assert acc.wind_speeds == [10.0]
        assert acc.wind_directions == [180.0]

    def test_add_multiple_frames(self, simple_accumulator):
        """Test accumulating multiple frames."""
        acc = simple_accumulator

        # Add first frame
        intensity1 = np.ones((4, 4), dtype=np.float64) * 2.0
        count1 = np.ones((4, 4), dtype=np.int32)
        acc.add_projected(
            projected_intensity=intensity1,
            projected_count=count1,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=0.0,
        )

        # Add second frame
        intensity2 = np.ones((4, 4), dtype=np.float64) * 4.0
        count2 = np.ones((4, 4), dtype=np.int32)
        acc.add_projected(
            projected_intensity=intensity2,
            projected_count=count2,
            timestamp=np.datetime64("2024-01-15T10:00:10"),
            heading=90.0,
        )

        assert acc.n_frames == 2
        # Sum should be 6.0 (2.0 + 4.0)
        assert np.all(acc.intensity_sum == 6.0)
        # Count should be 2
        assert np.all(acc.intensity_count == 2)

    def test_add_frame_with_missing_metadata(self, simple_accumulator):
        """Test adding frame with optional metadata missing."""
        acc = simple_accumulator

        intensity = np.ones((4, 4), dtype=np.float64)
        count = np.ones((4, 4), dtype=np.int32)

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=45.0,
            # No ship_speed, wind_speed, wind_direction
        )

        assert acc.n_frames == 1
        assert len(acc.ship_speeds) == 0
        assert len(acc.wind_speeds) == 0
        assert len(acc.wind_directions) == 0

    def test_finalize_single_frame(self, simple_accumulator):
        """Test finalization with a single frame."""
        acc = simple_accumulator

        intensity = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            dtype=np.float64,
        )
        count = np.ones((4, 4), dtype=np.int32)

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=45.0,
            ship_speed=5.0,
        )

        merged = acc.finalize(window_index=0)

        assert isinstance(merged, MergedImage)
        assert merged.n_frames == 1
        assert merged.window_index == 0
        # Intensity should equal input (only 1 frame)
        np.testing.assert_array_almost_equal(merged.intensity, intensity)

    def test_finalize_multiple_frames_averaging(self, simple_accumulator):
        """Test that finalize correctly averages multiple frames."""
        acc = simple_accumulator

        # Frame 1: all 2s
        acc.add_projected(
            projected_intensity=np.ones((4, 4), dtype=np.float64) * 2.0,
            projected_count=np.ones((4, 4), dtype=np.int32),
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=0.0,
        )

        # Frame 2: all 4s
        acc.add_projected(
            projected_intensity=np.ones((4, 4), dtype=np.float64) * 4.0,
            projected_count=np.ones((4, 4), dtype=np.int32),
            timestamp=np.datetime64("2024-01-15T10:00:10"),
            heading=0.0,
        )

        merged = acc.finalize()

        # Average should be 3.0 ((2+4)/2)
        np.testing.assert_array_almost_equal(merged.intensity, np.full((4, 4), 3.0))

    def test_finalize_nan_for_empty_cells(self, simple_accumulator):
        """Test that cells with no data become NaN."""
        acc = simple_accumulator

        # Create data with some empty cells
        intensity = np.array(
            [
                [1.0, 2.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        count = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=0.0,
        )

        merged = acc.finalize()

        # Top-left 2x2 should have values
        assert not np.isnan(merged.intensity[0, 0])
        assert not np.isnan(merged.intensity[1, 1])

        # Rest should be NaN
        assert np.isnan(merged.intensity[2, 2])
        assert np.isnan(merged.intensity[3, 3])

    def test_finalize_circular_mean_heading(self, simple_accumulator):
        """Test that heading is computed using circular mean."""
        acc = simple_accumulator

        # Add frames with headings near 0/360 boundary
        intensity = np.ones((4, 4), dtype=np.float64)
        count = np.ones((4, 4), dtype=np.int32)

        # Heading 350 degrees
        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=350.0,
        )

        # Heading 10 degrees
        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:10"),
            heading=10.0,
        )

        merged = acc.finalize()

        # Circular mean of 350 and 10 should be close to 0 (or 360)
        # Not the arithmetic mean of 180
        assert merged.mean_heading < 30 or merged.mean_heading > 330

    def test_finalize_metadata(self, simple_accumulator):
        """Test that finalize preserves metadata correctly."""
        acc = simple_accumulator

        intensity = np.ones((4, 4), dtype=np.float64)
        count = np.ones((4, 4), dtype=np.int32)

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=45.0,
            ship_speed=5.0,
            wind_speed=10.0,
            wind_direction=180.0,
        )

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:30"),
            heading=90.0,
            ship_speed=6.0,
            wind_speed=12.0,
            wind_direction=200.0,
        )

        merged = acc.finalize(window_index=5)

        assert merged.utm_zone == 10
        assert merged.hemisphere == "north"
        assert merged.center_lat == 45.0
        assert merged.center_lon == -122.0
        assert merged.grid_spacing == 50.0
        assert merged.window_index == 5
        assert merged.n_frames == 2
        assert merged.start_time == np.datetime64("2024-01-15T10:00:00")
        assert merged.end_time == np.datetime64("2024-01-15T10:00:30")
        assert merged.mean_ship_speed == pytest.approx(5.5)
        assert merged.mean_wind_speed == pytest.approx(11.0)


class TestWindowIntegration:
    """Integration tests for window module."""

    def test_create_windows_and_accumulate(self, tmp_path):
        """Test the full workflow of creating windows and accumulating data."""
        # Create test files
        timestamps = [
            "20240115100000",
            "20240115100010",
            "20240115100020",
            "20240115100030",
            "20240115100040",
            "20240115100050",
        ]
        files = []
        for ts in timestamps:
            filepath = tmp_path / f"{ts}_TOWER.pol"
            filepath.touch()
            files.append(str(filepath))

        # Create windows
        config = TimeWindowConfig(
            window_seconds=40.0, overlap_fraction=0.5, min_frames_per_window=3
        )
        windows = create_time_windows(files, config)

        assert len(windows) >= 2

        # Create accumulator for first window
        x_edges = np.array([-50.0, 0.0, 50.0])
        y_edges = np.array([-50.0, 0.0, 50.0])

        start_time, end_time, file_indices = windows[0]

        acc = WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=50.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

        # Simulate adding projected frames for this window
        for i, _idx in enumerate(file_indices):
            intensity = np.random.rand(2, 2).astype(np.float64)
            count = np.ones((2, 2), dtype=np.int32)
            timestamp = np.datetime64(f"2024-01-15T10:00:{i * 10:02d}")

            acc.add_projected(
                projected_intensity=intensity,
                projected_count=count,
                timestamp=timestamp,
                heading=float(i * 10),
            )

        # Finalize
        merged = acc.finalize(window_index=0)

        assert merged.n_frames == len(file_indices)
        assert merged.intensity.shape == (2, 2)
        assert not np.all(np.isnan(merged.intensity))


class TestTimeWindowConfigNewFields:
    """Tests for new resolution_scale and interpolate_gaps fields."""

    def test_resolution_scale_default(self):
        """Test default resolution_scale is 1.0."""
        config = TimeWindowConfig()
        assert config.resolution_scale == 1.0

    def test_resolution_scale_custom(self):
        """Test custom resolution_scale value."""
        config = TimeWindowConfig(resolution_scale=2.0)
        assert config.resolution_scale == 2.0

    def test_resolution_scale_validation(self):
        """Test that invalid resolution_scale raises ValueError."""
        with pytest.raises(ValueError, match="resolution_scale must be positive"):
            TimeWindowConfig(resolution_scale=0)
        with pytest.raises(ValueError, match="resolution_scale must be positive"):
            TimeWindowConfig(resolution_scale=-1)

    def test_interpolate_gaps_default(self):
        """Test default interpolate_gaps is False."""
        config = TimeWindowConfig()
        assert config.interpolate_gaps is False

    def test_interpolate_gaps_custom(self):
        """Test custom interpolate_gaps value."""
        config = TimeWindowConfig(interpolate_gaps=True)
        assert config.interpolate_gaps is True


class TestInterpolateNanGaps:
    """Tests for _interpolate_nan_gaps function."""

    def test_no_gaps(self):
        """Test that array with no NaNs is unchanged."""
        from wamos_tpw.window import _interpolate_nan_gaps

        intensity = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _interpolate_nan_gaps(intensity)
        np.testing.assert_array_equal(result, intensity)

    def test_all_nan(self):
        """Test that all-NaN array returns NaNs (no valid data)."""
        from wamos_tpw.window import _interpolate_nan_gaps

        intensity = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        result = _interpolate_nan_gaps(intensity)
        assert np.all(np.isnan(result))

    def test_single_nan_gap(self):
        """Test filling a single NaN with nearest neighbor."""
        from wamos_tpw.window import _interpolate_nan_gaps

        intensity = np.array([[1.0, np.nan], [3.0, 4.0]])
        result = _interpolate_nan_gaps(intensity)

        assert not np.any(np.isnan(result))
        # The NaN at [0,1] should be filled from nearest neighbor
        # Could be 1.0 (left) or 4.0 (below-right) depending on distance_transform
        assert result[0, 1] in [1.0, 4.0]

    def test_center_nan_gap(self):
        """Test filling center NaN surrounded by values."""
        from wamos_tpw.window import _interpolate_nan_gaps

        intensity = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        result = _interpolate_nan_gaps(intensity)

        assert not np.any(np.isnan(result))
        # Center value should be filled from one of the 4-neighbors
        assert result[1, 1] in [2.0, 4.0, 6.0, 8.0]

    def test_corner_valid_rest_nan(self):
        """Test filling from a single valid corner value."""
        from wamos_tpw.window import _interpolate_nan_gaps

        intensity = np.array([[1.0, np.nan], [np.nan, np.nan]])
        result = _interpolate_nan_gaps(intensity)

        assert not np.any(np.isnan(result))
        # All NaNs should be filled with the corner value
        np.testing.assert_array_equal(result, np.array([[1.0, 1.0], [1.0, 1.0]]))

    def test_large_gap_preserved(self):
        """Test that large NaN regions (shadows, outside radar) are preserved."""
        from wamos_tpw.window import _interpolate_nan_gaps

        # Create array with valid data on left, large NaN gap on right
        intensity = np.full((10, 20), np.nan)
        intensity[:, :5] = 1.0  # Valid data on left 5 columns

        result = _interpolate_nan_gaps(intensity, max_distance=3)

        # Left side should be valid
        assert not np.any(np.isnan(result[:, :5]))
        # Columns 5-7 (within 3 pixels) should be filled
        assert not np.any(np.isnan(result[:, 5:8]))
        # Columns 8+ (more than 3 pixels away) should still be NaN
        assert np.all(np.isnan(result[:, 9:]))


class TestFinalizeWithInterpolation:
    """Tests for finalize with interpolate_gaps=True."""

    @pytest.fixture
    def accumulator_with_gaps(self):
        """Create an accumulator with gaps in the data."""
        x_edges = np.array([-150.0, -50.0, 50.0, 150.0])  # 3 bins
        y_edges = np.array([-150.0, -50.0, 50.0, 150.0])  # 3 bins
        return WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=100.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

    def test_finalize_without_interpolation(self, accumulator_with_gaps):
        """Test finalize without interpolation leaves NaNs."""
        acc = accumulator_with_gaps

        # Add data with gaps (some cells with count=0)
        intensity = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        count = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=0.0,
        )

        merged = acc.finalize(interpolate_gaps=False)

        # Should have NaNs where count was 0
        assert np.isnan(merged.intensity[0, 2])
        assert np.isnan(merged.intensity[1, 1])
        assert np.isnan(merged.intensity[2, 2])

    def test_finalize_with_interpolation(self, accumulator_with_gaps):
        """Test finalize with interpolation fills NaNs."""
        acc = accumulator_with_gaps

        # Add data with gaps
        intensity = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        count = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int32)

        acc.add_projected(
            projected_intensity=intensity,
            projected_count=count,
            timestamp=np.datetime64("2024-01-15T10:00:00"),
            heading=0.0,
        )

        merged = acc.finalize(interpolate_gaps=True)

        # Should have no NaNs
        assert not np.any(np.isnan(merged.intensity))
        # Original values should be preserved
        assert merged.intensity[0, 0] == 1.0
        assert merged.intensity[0, 1] == 2.0
        assert merged.intensity[1, 0] == 3.0
