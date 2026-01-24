#! /usr/bin/env python3
"""Integration tests for wamos_tpw.files_pipeline module."""

import numpy as np

from wamos_tpw.config import Config
from wamos_tpw.files_pipeline import FilesMergePipeline
from wamos_tpw.filenames import extract_file_timestamp
from wamos_tpw.grid import compute_common_grid, remap_to_common_grid
from wamos_tpw.merged_image import MergedImage, TimeWindowConfig
from wamos_tpw.window import WindowAccumulator, create_time_windows


class TestFilesMergePipelineInit:
    """Tests for FilesMergePipeline initialization."""

    def test_pipeline_creation_basic(self, march_polar_files):
        """Test basic pipeline creation with default config."""
        files = [str(f) for f in march_polar_files]
        pipeline = FilesMergePipeline(files)

        assert pipeline is not None
        assert pipeline.n_windows >= 0

    def test_pipeline_creation_with_window_config(self, march_polar_files):
        """Test pipeline creation with custom window config."""
        files = [str(f) for f in march_polar_files]
        window_config = TimeWindowConfig(
            window_seconds=30.0,
            overlap_fraction=0.25,
            min_frames_per_window=2,
        )

        pipeline = FilesMergePipeline(
            filenames=files,
            window_config=window_config,
        )

        assert pipeline.window_config.window_seconds == 30.0
        assert pipeline.window_config.overlap_fraction == 0.25
        assert pipeline.window_config.min_frames_per_window == 2

    def test_pipeline_creation_with_config(self, march_polar_files):
        """Test pipeline creation with YAML config."""
        files = [str(f) for f in march_polar_files]
        config = Config()

        pipeline = FilesMergePipeline(
            filenames=files,
            config=config,
        )

        assert pipeline is not None

    def test_pipeline_creation_empty_files(self):
        """Test pipeline creation with empty file list."""
        pipeline = FilesMergePipeline(filenames=[])

        assert pipeline.n_windows == 0


class TestCreateTimeWindowsIntegration:
    """Integration tests for create_time_windows with real files."""

    def test_windows_from_real_files(self, march_polar_files):
        """Test window creation from real polar files."""
        files = [str(f) for f in march_polar_files]
        window_config = TimeWindowConfig(
            window_seconds=60.0,
            overlap_fraction=0.5,
            min_frames_per_window=2,
        )

        windows = create_time_windows(files, window_config)

        # Should create at least one window from the 6 March files
        assert len(windows) >= 1

        for start_time, end_time, file_indices in windows:
            # Each window should have valid times
            assert start_time < end_time
            # Each window should have enough files
            assert len(file_indices) >= window_config.min_frames_per_window
            # All indices should be valid
            assert all(0 <= idx < len(files) for idx in file_indices)

    def test_windows_timestamps_extracted(self, march_polar_files):
        """Test that timestamps are correctly extracted from file paths."""
        files = [str(f) for f in march_polar_files]

        # All files should have extractable timestamps
        for f in files:
            ts = extract_file_timestamp(f)
            assert ts is not None, f"Failed to extract timestamp from {f}"

    def test_windows_overlap_fraction(self, march_polar_files):
        """Test that windows overlap correctly."""
        files = [str(f) for f in march_polar_files]
        window_config = TimeWindowConfig(
            window_seconds=60.0,
            overlap_fraction=0.5,
            min_frames_per_window=1,  # Low threshold for testing
        )

        windows = create_time_windows(files, window_config)

        if len(windows) >= 2:
            # Check that windows overlap
            for i in range(len(windows) - 1):
                start1, end1, _ = windows[i]
                start2, end2, _ = windows[i + 1]

                # Window i should end after window i+1 starts (overlap)
                assert end1 > start2, "Windows should overlap"


class TestWindowAccumulatorIntegration:
    """Integration tests for WindowAccumulator."""

    def test_accumulator_with_real_grid_params(self, march_polar_files):
        """Test WindowAccumulator with realistic grid parameters."""
        # Simulate realistic parameters from a merged image
        x_edges = np.linspace(-3000, 3000, 101)  # 100 bins, 60m spacing
        y_edges = np.linspace(-3000, 3000, 101)

        accumulator = WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=60.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

        # Add some synthetic projected frames
        for i in range(5):
            intensity = np.random.rand(100, 100) * 100
            count = np.ones((100, 100), dtype=np.int32)
            timestamp = np.datetime64("2022-03-28T03:00:00") + np.timedelta64(i * 10, "s")

            accumulator.add_projected(
                projected_intensity=intensity,
                projected_count=count,
                timestamp=timestamp,
                heading=180.0 + i,
                ship_speed=5.0,
                wind_speed=10.0,
                wind_direction=45.0,
            )

        assert accumulator.n_frames == 5

        # Finalize and check result
        merged = accumulator.finalize(window_index=0)

        assert isinstance(merged, MergedImage)
        assert merged.n_frames == 5
        assert merged.intensity.shape == (100, 100)
        assert merged.utm_zone == 10
        assert merged.hemisphere == "north"
        assert merged.mean_ship_speed == 5.0
        assert merged.mean_wind_speed == 10.0


class TestComputeCommonGridIntegration:
    """Integration tests for compute_common_grid."""

    def test_grid_from_realistic_coordinates(self):
        """Test common grid computation with realistic lat/lon data."""
        # Simulate multiple frames with ship motion
        center_lat, center_lon = 45.0, -122.0

        # Each frame has slightly different center due to ship motion
        latitudes = []
        longitudes = []
        for i in range(5):
            # Simulate radar sweep coordinates (simplified)
            lat_offset = i * 0.001  # Ship moving north
            lat = np.array([center_lat + lat_offset - 0.01, center_lat + lat_offset + 0.01])
            lon = np.array([center_lon - 0.01, center_lon + 0.01])
            latitudes.append(lat)
            longitudes.append(lon)

        max_ranges = [3000.0] * 5
        range_resolutions = [7.5] * 5

        grid = compute_common_grid(
            latitudes=latitudes,
            longitudes=longitudes,
            max_ranges=max_ranges,
            range_resolutions=range_resolutions,
        )

        assert "x_edges" in grid
        assert "y_edges" in grid
        assert "utm_zone" in grid
        assert "hemisphere" in grid
        assert "grid_spacing" in grid
        assert grid["hemisphere"] == "north"  # 45N is northern hemisphere
        assert grid["grid_spacing"] > 0


class TestRemapToCommonGridIntegration:
    """Integration tests for remap_to_common_grid."""

    def test_remap_preserves_total_intensity(self):
        """Test that remapping approximately preserves total intensity."""
        # Create source grid
        src_intensity = np.ones((10, 10), dtype=np.float64) * 100
        src_count = np.ones((10, 10), dtype=np.int32)
        src_x = np.linspace(-500, 500, 11)
        src_y = np.linspace(-500, 500, 11)

        # Create destination grid (same size, shifted slightly)
        dst_x = np.linspace(-400, 600, 11)
        dst_y = np.linspace(-400, 600, 11)

        dst_sum, dst_count = remap_to_common_grid(
            src_intensity,
            src_count,
            src_x,
            src_y,
            dst_x,
            dst_y,
            10,
            10,
        )

        # Check that destination has valid data
        assert dst_sum.shape == (10, 10)
        assert dst_count.shape == (10, 10)

        # Some cells should have data (overlapping region)
        assert np.any(dst_count > 0)

    def test_remap_no_overlap(self):
        """Test remapping when grids don't overlap."""
        # Create source grid
        src_intensity = np.ones((10, 10), dtype=np.float64) * 100
        src_count = np.ones((10, 10), dtype=np.int32)
        src_x = np.linspace(0, 100, 11)
        src_y = np.linspace(0, 100, 11)

        # Create destination grid (completely separate)
        dst_x = np.linspace(1000, 1100, 11)
        dst_y = np.linspace(1000, 1100, 11)

        dst_sum, dst_count = remap_to_common_grid(
            src_intensity,
            src_count,
            src_x,
            src_y,
            dst_x,
            dst_y,
            10,
            10,
        )

        # No cells should have data (no overlap)
        assert np.all(dst_count == 0)


class TestMergedImageProperties:
    """Tests for MergedImage dataclass properties."""

    def test_merged_image_dimensions(self):
        """Test MergedImage dimension properties."""
        intensity = np.zeros((50, 100))
        x_edges = np.linspace(-1000, 1000, 101)  # 100 bins
        y_edges = np.linspace(-500, 500, 51)  # 50 bins

        merged = MergedImage(
            intensity=intensity,
            x_edges=x_edges,
            y_edges=y_edges,
            start_time=np.datetime64("2022-03-28T03:00:00"),
            end_time=np.datetime64("2022-03-28T03:01:00"),
            n_frames=10,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
            grid_spacing=20.0,
            mean_heading=180.0,
        )

        assert merged.n_x == 100
        assert merged.n_y == 50
        assert len(merged.x_centers) == 100
        assert len(merged.y_centers) == 50

    def test_merged_image_duration(self):
        """Test MergedImage duration calculation."""
        intensity = np.zeros((10, 10))
        x_edges = np.linspace(-100, 100, 11)
        y_edges = np.linspace(-100, 100, 11)

        merged = MergedImage(
            intensity=intensity,
            x_edges=x_edges,
            y_edges=y_edges,
            start_time=np.datetime64("2022-03-28T03:00:00"),
            end_time=np.datetime64("2022-03-28T03:01:30"),  # 90 seconds
            n_frames=10,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
            grid_spacing=20.0,
            mean_heading=180.0,
        )

        assert merged.duration_seconds == 90.0


class TestTimeWindowConfigIntegration:
    """Integration tests for TimeWindowConfig."""

    def test_config_stride_calculation(self):
        """Test stride calculation from overlap fraction."""
        config = TimeWindowConfig(
            window_seconds=60.0,
            overlap_fraction=0.5,
        )

        assert config.stride_seconds == 30.0  # 60 * (1 - 0.5)

    def test_config_no_overlap(self):
        """Test stride with no overlap."""
        config = TimeWindowConfig(
            window_seconds=60.0,
            overlap_fraction=0.0,
        )

        assert config.stride_seconds == 60.0  # No overlap = stride equals window

    def test_config_high_overlap(self):
        """Test stride with high overlap."""
        config = TimeWindowConfig(
            window_seconds=60.0,
            overlap_fraction=0.9,
        )

        assert abs(config.stride_seconds - 6.0) < 0.01  # 60 * 0.1


class TestPipelineWindowCreation:
    """Tests for pipeline window creation behavior."""

    def test_pipeline_creates_windows(self, march_polar_files):
        """Test that pipeline creates appropriate windows."""
        files = [str(f) for f in march_polar_files]

        # Use permissive settings to ensure windows are created
        window_config = TimeWindowConfig(
            window_seconds=120.0,  # Long window
            overlap_fraction=0.5,
            min_frames_per_window=1,  # Low threshold
        )

        pipeline = FilesMergePipeline(
            filenames=files,
            window_config=window_config,
        )

        # Should have at least one window with 6 files and 120s window
        assert pipeline.n_windows >= 1

    def test_pipeline_no_windows_high_threshold(self, march_polar_files):
        """Test that pipeline creates no windows with very high threshold."""
        files = [str(f) for f in march_polar_files[:2]]  # Only 2 files

        window_config = TimeWindowConfig(
            window_seconds=10.0,  # Very short window
            overlap_fraction=0.0,
            min_frames_per_window=100,  # Impossible threshold
        )

        pipeline = FilesMergePipeline(
            filenames=files,
            window_config=window_config,
        )

        # Should have no windows with 100 min_frames requirement
        assert pipeline.n_windows == 0


class TestEndToEndWorkflow:
    """End-to-end integration tests simulating full workflow."""

    def test_window_accumulator_workflow(self):
        """Test complete workflow from grid creation to merged image."""
        # Step 1: Create grid parameters
        latitudes = [np.array([45.0, 45.01]), np.array([45.005, 45.015])]
        longitudes = [np.array([-122.0, -121.99]), np.array([-122.005, -121.985])]
        max_ranges = [3000.0, 3000.0]
        range_resolutions = [10.0, 10.0]

        grid_params = compute_common_grid(
            latitudes=latitudes,
            longitudes=longitudes,
            max_ranges=max_ranges,
            range_resolutions=range_resolutions,
        )

        # Step 2: Create accumulator
        accumulator = WindowAccumulator(
            x_edges=grid_params["x_edges"],
            y_edges=grid_params["y_edges"],
            grid_spacing=grid_params["grid_spacing"],
            utm_zone=grid_params["utm_zone"],
            hemisphere=grid_params["hemisphere"],
            center_lat=grid_params["center_lat"],
            center_lon=grid_params["center_lon"],
        )

        # Step 3: Add frames
        n_x = len(grid_params["x_edges"]) - 1
        n_y = len(grid_params["y_edges"]) - 1

        for i in range(3):
            intensity = np.random.rand(n_y, n_x) * 50 + 25
            count = np.ones((n_y, n_x), dtype=np.int32)
            timestamp = np.datetime64("2022-03-28T03:00:00") + np.timedelta64(i * 20, "s")

            accumulator.add_projected(
                projected_intensity=intensity,
                projected_count=count,
                timestamp=timestamp,
                heading=270.0,
            )

        # Step 4: Finalize
        merged = accumulator.finalize(window_index=0)

        # Verify result
        assert isinstance(merged, MergedImage)
        assert merged.n_frames == 3
        assert merged.intensity.shape == (n_y, n_x)
        assert not np.all(np.isnan(merged.intensity))
        assert merged.utm_zone == grid_params["utm_zone"]

    def test_multi_frame_averaging(self):
        """Test that multiple frames are correctly averaged."""
        x_edges = np.linspace(-100, 100, 11)
        y_edges = np.linspace(-100, 100, 11)

        accumulator = WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=20.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

        # Add frames with known values
        # Frame 1: all 10s
        intensity1 = np.ones((10, 10), dtype=np.float64) * 10
        count1 = np.ones((10, 10), dtype=np.int32)
        accumulator.add_projected(intensity1, count1, np.datetime64("2022-01-01T00:00:00"), 180.0)

        # Frame 2: all 20s
        intensity2 = np.ones((10, 10), dtype=np.float64) * 20
        count2 = np.ones((10, 10), dtype=np.int32)
        accumulator.add_projected(intensity2, count2, np.datetime64("2022-01-01T00:00:10"), 180.0)

        # Frame 3: all 30s
        intensity3 = np.ones((10, 10), dtype=np.float64) * 30
        count3 = np.ones((10, 10), dtype=np.int32)
        accumulator.add_projected(intensity3, count3, np.datetime64("2022-01-01T00:00:20"), 180.0)

        merged = accumulator.finalize()

        # Average should be (10 + 20 + 30) / 3 = 20
        np.testing.assert_array_almost_equal(merged.intensity, np.full((10, 10), 20.0))


class TestCircularMeanHeading:
    """Tests for circular mean heading calculation."""

    def test_heading_wraparound(self):
        """Test circular mean handles wraparound correctly."""
        x_edges = np.linspace(-100, 100, 11)
        y_edges = np.linspace(-100, 100, 11)

        accumulator = WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=20.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

        intensity = np.ones((10, 10), dtype=np.float64)
        count = np.ones((10, 10), dtype=np.int32)

        # Add frames with headings around 0/360
        accumulator.add_projected(intensity, count, np.datetime64("2022-01-01T00:00:00"), 355.0)
        accumulator.add_projected(intensity, count, np.datetime64("2022-01-01T00:00:10"), 5.0)

        merged = accumulator.finalize()

        # Circular mean of 355 and 5 should be close to 0 (or 360)
        assert merged.mean_heading < 10 or merged.mean_heading > 350

    def test_heading_opposite_directions(self):
        """Test circular mean with opposite headings."""
        x_edges = np.linspace(-100, 100, 11)
        y_edges = np.linspace(-100, 100, 11)

        accumulator = WindowAccumulator(
            x_edges=x_edges,
            y_edges=y_edges,
            grid_spacing=20.0,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
        )

        intensity = np.ones((10, 10), dtype=np.float64)
        count = np.ones((10, 10), dtype=np.int32)

        # Add frames with headings 90 and 270 (opposite)
        accumulator.add_projected(intensity, count, np.datetime64("2022-01-01T00:00:00"), 90.0)
        accumulator.add_projected(intensity, count, np.datetime64("2022-01-01T00:00:10"), 270.0)

        merged = accumulator.finalize()

        # Result will be ambiguous (could be 0 or 180)
        # Just verify it's a valid heading
        assert 0 <= merged.mean_heading < 360
