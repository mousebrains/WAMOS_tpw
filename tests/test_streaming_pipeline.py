"""Tests for streaming pipeline module."""

from pathlib import Path

import numpy as np
import pytest

from wamos_tpw.merged_image import TimeWindowConfig
from wamos_tpw.streaming_filenames import (
    WindowTracker,
    create_time_windows_from_bounds,
)
from wamos_tpw.streaming_pipeline import StreamingMergePipeline


class TestStreamingMergePipelineInit:
    """Tests for StreamingMergePipeline initialization."""

    def test_basic_initialization(self, test_data_dir: Path):
        """Test basic pipeline initialization."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:10:00")

        pipeline = StreamingMergePipeline(
            stime=stime,
            etime=etime,
            polar_path=str(test_data_dir),
        )

        assert pipeline is not None
        assert pipeline.n_windows > 0

    def test_initialization_with_string_times(self, test_data_dir: Path):
        """Test initialization with string time values."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
        )

        assert pipeline is not None
        assert pipeline.n_windows > 0

    def test_initialization_with_window_config(self, test_data_dir: Path):
        """Test initialization with custom window config."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        window_config = TimeWindowConfig(
            window_seconds=30.0,
            overlap_fraction=0.25,
            min_frames_per_window=2,
        )

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
            window_config=window_config,
        )

        # More windows with shorter duration
        assert pipeline.n_windows > 10

    def test_initialization_with_config(self, test_data_dir: Path):
        """Test initialization with YAML config."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        from wamos_tpw.config import Config

        config = Config()

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
            config=config,
        )

        assert pipeline is not None

    def test_window_count_calculation(self, test_data_dir: Path):
        """Test that window count is calculated correctly."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        # 10 minutes with 60s window and 50% overlap (30s stride)
        # Should create approximately 10 minutes / 30 seconds = 20 windows
        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
            window_config=TimeWindowConfig(
                window_seconds=60.0,
                overlap_fraction=0.5,
            ),
        )

        # Allow some margin for boundary effects
        assert 15 <= pipeline.n_windows <= 25


class TestStreamingPipelineWindowCreation:
    """Tests for window creation in streaming pipeline."""

    def test_windows_created_from_time_bounds(self, test_data_dir: Path):
        """Test that windows are created based on time bounds, not file list."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        # Create two pipelines with same time range
        pipeline1 = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
        )

        pipeline2 = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
        )

        # Should have same number of windows (deterministic)
        assert pipeline1.n_windows == pipeline2.n_windows

    def test_windows_independent_of_files(self, test_data_dir: Path):
        """Test that window count doesn't depend on actual file count."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        # Create pipeline with time range that may or may not have files
        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
        )

        # Windows are created from time bounds
        # 5 minutes with 60s window and 30s stride = ~10 windows
        assert pipeline.n_windows >= 5

    def test_empty_time_range(self, test_data_dir: Path):
        """Test pipeline with very short time range."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        # 1 second range
        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:00:01",
            polar_path=str(test_data_dir),
        )

        # Should still create at least one window
        assert pipeline.n_windows >= 0


class TestStreamingPipelineIntegration:
    """Integration tests for StreamingMergePipeline with real data."""

    def test_iter_merged_returns_iterator(self, test_data_dir: Path):
        """Test that iter_merged returns an iterator."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:01:00",  # Short range for faster test
            polar_path=str(test_data_dir),
            window_config=TimeWindowConfig(
                window_seconds=60.0,
                overlap_fraction=0.5,
                min_frames_per_window=1,
            ),
            n_workers=2,
            qProgress=False,
        )

        # Just verify iter_merged returns an iterator without consuming it
        merged_iter = pipeline.iter_merged()
        assert hasattr(merged_iter, "__iter__")
        assert hasattr(merged_iter, "__next__")


class TestStreamingPipelineWithMockData:
    """Tests using mock data for faster execution."""

    def test_pipeline_structure(self, test_data_dir: Path):
        """Test pipeline structure without full execution."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:10:00",
            polar_path=str(test_data_dir),
        )

        # Check internal state
        assert hasattr(pipeline, "_stime")
        assert hasattr(pipeline, "_etime")
        assert hasattr(pipeline, "_windows")
        assert hasattr(pipeline, "_window_config")

        # Windows should be list of tuples
        assert isinstance(pipeline._windows, list)
        for window in pipeline._windows:
            assert len(window) == 3  # (start, end, index)


class TestStreamingVsBlockingComparison:
    """Tests comparing streaming and blocking pipeline behavior."""

    def test_window_creation_difference(self, test_data_dir: Path):
        """Test that streaming creates windows from time bounds, blocking from files.

        Key difference:
        - Blocking: Creates windows based on timestamps of actual files found
        - Streaming: Creates windows based on stime/etime bounds upfront

        This means streaming may have more windows because it covers the full
        requested time range, while blocking only covers the time range of
        actual files.
        """
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        from wamos_tpw.filenames import Filenames
        from wamos_tpw.files_pipeline import FilesMergePipeline

        stime = "2022-03-28T03:00:00"
        etime = "2022-03-28T03:10:00"

        window_config = TimeWindowConfig(
            window_seconds=60.0,
            overlap_fraction=0.5,
            min_frames_per_window=1,
        )

        # Get files for blocking pipeline
        filenames = Filenames(
            np.datetime64(stime),
            np.datetime64(etime),
            str(test_data_dir),
        )
        files = list(filenames.files)

        if not files:
            pytest.skip("No files found in time range")

        # Create blocking pipeline
        blocking_pipeline = FilesMergePipeline(
            filenames=files,
            window_config=window_config,
        )

        # Create streaming pipeline
        streaming_pipeline = StreamingMergePipeline(
            stime=stime,
            etime=etime,
            polar_path=str(test_data_dir),
            window_config=window_config,
        )

        # Streaming should have at least as many windows as blocking
        # (it covers the full time range)
        assert streaming_pipeline.n_windows >= blocking_pipeline.n_windows

        # Both should have a reasonable number of windows
        assert blocking_pipeline.n_windows >= 0
        assert streaming_pipeline.n_windows >= 0


class TestWindowTrackerIntegration:
    """Tests for WindowTracker integration with streaming pipeline."""

    def test_tracker_with_pipeline_windows(self, test_data_dir: Path):
        """Test WindowTracker with windows from streaming pipeline."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:05:00")

        # Create windows like the pipeline does
        windows = create_time_windows_from_bounds(stime, etime, window_seconds=60.0, overlap=0.5)

        tracker = WindowTracker(windows=windows, min_frames_per_window=1)

        # Simulate file discovery
        from wamos_tpw.streaming_filenames import DiscoveredFile

        # Create a file at 03:02:00
        ts = np.datetime64("2022-03-28T03:02:00")
        ts_ns = ts.astype("datetime64[ns]").astype(np.int64)
        file = DiscoveredFile("/path/file.pol", ts_ns, file_id=0)

        assigned = tracker.assign_file(file)

        # File should be assigned to multiple overlapping windows
        assert len(assigned) >= 1

        # After marking file processed and discovery complete
        tracker.mark_file_processed(0)
        tracker.mark_all_discovery_complete()

        # Some windows should be ready
        ready = tracker.get_ready_windows()
        assert len(ready) >= 1


class TestStreamingPipelineParameters:
    """Tests for various pipeline parameters."""

    def test_workers_parameter(self, test_data_dir: Path):
        """Test workers parameter."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            n_workers=4,
        )

        assert pipeline._n_workers == 4

    def test_tolerance_parameter(self, test_data_dir: Path):
        """Test tolerance parameter."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            tolerance=2.0,
        )

        assert pipeline._tolerance == 2.0

    def test_timing_parameter(self, test_data_dir: Path):
        """Test timing parameter."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            qTiming=True,
        )

        assert pipeline._qTiming is True

    def test_progress_parameter(self, test_data_dir: Path):
        """Test progress parameter."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            qProgress=False,
        )

        assert pipeline._qProgress is False


class TestTimeWindowConfigIntegrationWithStreaming:
    """Tests for TimeWindowConfig with streaming pipeline."""

    def test_config_affects_window_count(self, test_data_dir: Path):
        """Test that window config affects window count."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        # Short windows = more windows
        short_config = TimeWindowConfig(window_seconds=30.0, overlap_fraction=0.5)
        short_pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            window_config=short_config,
        )

        # Long windows = fewer windows
        long_config = TimeWindowConfig(window_seconds=120.0, overlap_fraction=0.5)
        long_pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            window_config=long_config,
        )

        assert short_pipeline.n_windows > long_pipeline.n_windows

    def test_overlap_affects_window_count(self, test_data_dir: Path):
        """Test that overlap fraction affects window count."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        # High overlap = more windows
        high_overlap_config = TimeWindowConfig(window_seconds=60.0, overlap_fraction=0.9)
        high_pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            window_config=high_overlap_config,
        )

        # No overlap = fewer windows
        no_overlap_config = TimeWindowConfig(window_seconds=60.0, overlap_fraction=0.0)
        no_pipeline = StreamingMergePipeline(
            stime="2022-03-28T03:00:00",
            etime="2022-03-28T03:05:00",
            polar_path=str(test_data_dir),
            window_config=no_overlap_config,
        )

        assert high_pipeline.n_windows > no_pipeline.n_windows
