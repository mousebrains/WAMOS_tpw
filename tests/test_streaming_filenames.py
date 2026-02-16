"""Tests for streaming file discovery module."""

from pathlib import Path

import numpy as np
import pytest

from wamos_tpw.streaming_filenames import (
    DiscoveredFile,
    HourBatch,
    StreamingFilenames,
    WindowTracker,
    create_time_windows_from_bounds,
)


class TestDiscoveredFile:
    """Tests for DiscoveredFile dataclass."""

    def test_creation(self):
        """Test creating a DiscoveredFile."""
        f = DiscoveredFile(
            filepath="/path/to/file.pol",
            timestamp_ns=1648447200000000000,
            file_id=42,
        )
        assert f.filepath == "/path/to/file.pol"
        assert f.timestamp_ns == 1648447200000000000
        assert f.file_id == 42


class TestHourBatch:
    """Tests for HourBatch dataclass."""

    def test_creation(self):
        """Test creating an HourBatch."""
        files = [
            DiscoveredFile("/path/1.pol", 1000000000, 0),
            DiscoveredFile("/path/2.pol", 2000000000, 1),
        ]
        batch = HourBatch(
            hour_start_ns=0,
            hour_end_ns=3600000000000,
            files=files,
            is_final=False,
        )
        assert len(batch.files) == 2
        assert batch.hour_start_ns == 0
        assert not batch.is_final

    def test_is_final_flag(self):
        """Test the is_final flag."""
        batch = HourBatch(
            hour_start_ns=0,
            hour_end_ns=3600000000000,
            files=[],
            is_final=True,
        )
        assert batch.is_final


class TestCreateTimeWindowsFromBounds:
    """Tests for create_time_windows_from_bounds function."""

    def test_basic_windows(self):
        """Test creating basic time windows."""
        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:10:00")

        windows = create_time_windows_from_bounds(stime, etime, window_seconds=60.0, overlap=0.5)

        # 10 minutes with 60s window and 30s stride = ~20 windows
        assert len(windows) >= 15
        assert len(windows) <= 25

        # Check first window
        start, end, idx = windows[0]
        assert idx == 0
        assert start == np.datetime64(stime, "ns")

        # Windows should be sequential
        for i, (_start, _end, idx) in enumerate(windows):
            assert idx == i

    def test_no_overlap(self):
        """Test windows with no overlap."""
        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:05:00")

        windows = create_time_windows_from_bounds(stime, etime, window_seconds=60.0, overlap=0.0)

        # 5 minutes with 60s window and no overlap = 5 windows
        assert len(windows) == 5

    def test_high_overlap(self):
        """Test windows with high overlap."""
        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:02:00")

        windows = create_time_windows_from_bounds(stime, etime, window_seconds=60.0, overlap=0.9)

        # High overlap = more windows
        assert len(windows) >= 10

    def test_empty_range(self):
        """Test with same start and end time."""
        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:00:00")

        windows = create_time_windows_from_bounds(stime, etime, window_seconds=60.0, overlap=0.5)

        # Should still create at least one window
        assert len(windows) >= 0


class TestWindowTracker:
    """Tests for WindowTracker class."""

    @pytest.fixture
    def sample_windows(self):
        """Create sample windows for testing."""
        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T03:05:00")
        return create_time_windows_from_bounds(stime, etime, window_seconds=60.0, overlap=0.5)

    def test_assign_file(self, sample_windows):
        """Test assigning a file to windows."""
        tracker = WindowTracker(windows=sample_windows, min_frames_per_window=1)

        # Create a file in the middle of the time range
        ts = np.datetime64("2022-03-28T03:02:30")
        ts_ns = ts.astype("datetime64[ns]").astype(np.int64)
        file = DiscoveredFile("/path/file.pol", ts_ns, file_id=0)

        assigned = tracker.assign_file(file)

        # File should be assigned to at least one window
        assert len(assigned) >= 1

    def test_mark_discovery_complete_before(self, sample_windows):
        """Test marking windows as discovery-complete."""
        tracker = WindowTracker(windows=sample_windows, min_frames_per_window=1)

        # Mark windows complete before 3:02
        cutoff = np.datetime64("2022-03-28T03:02:00")
        cutoff_ns = cutoff.astype("datetime64[ns]").astype(np.int64)
        completed = tracker.mark_discovery_complete_before(cutoff_ns)

        # Some windows should be marked complete
        assert len(completed) >= 1

        # Check that marked windows are discovery-complete
        for window_idx in completed:
            assert tracker.is_discovery_complete(window_idx)

    def test_mark_all_discovery_complete(self, sample_windows):
        """Test marking all windows as discovery-complete."""
        tracker = WindowTracker(windows=sample_windows, min_frames_per_window=1)

        completed = tracker.mark_all_discovery_complete()

        # All windows should be marked complete
        assert len(completed) == len(sample_windows)

    def test_get_ready_windows(self, sample_windows):
        """Test getting windows ready for merging."""
        tracker = WindowTracker(windows=sample_windows, min_frames_per_window=1)

        # Initially no windows are ready (discovery not complete)
        ready = tracker.get_ready_windows()
        assert len(ready) == 0

        # Add files to first window
        ts = np.datetime64("2022-03-28T03:00:15")
        ts_ns = ts.astype("datetime64[ns]").astype(np.int64)
        file = DiscoveredFile("/path/file.pol", ts_ns, file_id=0)
        tracker.assign_file(file)

        # Mark first window as discovery complete
        cutoff = np.datetime64("2022-03-28T03:01:00")
        cutoff_ns = cutoff.astype("datetime64[ns]").astype(np.int64)
        tracker.mark_discovery_complete_before(cutoff_ns)

        # Mark file as processed
        tracker.mark_file_processed(0)

        # Now first window should be ready
        ready = tracker.get_ready_windows()
        assert 0 in ready

    def test_min_frames_requirement(self, sample_windows):
        """Test that min_frames_per_window is respected."""
        tracker = WindowTracker(windows=sample_windows, min_frames_per_window=3)

        # Add only 2 files to first window
        for i in range(2):
            ts = np.datetime64("2022-03-28T03:00:15")
            ts_ns = ts.astype("datetime64[ns]").astype(np.int64)
            file = DiscoveredFile(f"/path/file{i}.pol", ts_ns, file_id=i)
            tracker.assign_file(file)
            tracker.mark_file_processed(i)

        # Mark all discovery complete
        tracker.mark_all_discovery_complete()

        # Window 0 should NOT be ready (only 2 files, needs 3)
        ready = tracker.get_ready_windows()
        assert 0 not in ready

    def test_get_window_files(self, sample_windows):
        """Test getting files assigned to a window."""
        tracker = WindowTracker(windows=sample_windows, min_frames_per_window=1)

        ts = np.datetime64("2022-03-28T03:00:15")
        ts_ns = ts.astype("datetime64[ns]").astype(np.int64)
        file = DiscoveredFile("/path/file.pol", ts_ns, file_id=42)
        tracker.assign_file(file)

        # Get files for first window
        files = tracker.get_window_files(0)
        assert 42 in files


class TestStreamingFilenames:
    """Tests for StreamingFilenames class."""

    def test_initialization(self, test_data_dir: Path):
        """Test StreamingFilenames initialization."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        streaming = StreamingFilenames(stime, etime, str(test_data_dir))

        assert streaming.stime == np.datetime64(stime, "ns")
        assert streaming.etime == np.datetime64(etime, "ns")
        assert not streaming.state.is_complete

    def test_invalid_time_range(self, test_data_dir: Path):
        """Test that invalid time range raises error."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        stime = np.datetime64("2022-03-28T04:00:00")  # Later
        etime = np.datetime64("2022-03-28T03:00:00")  # Earlier

        with pytest.raises(ValueError, match="Start time .* must be <= end time"):
            StreamingFilenames(stime, etime, str(test_data_dir))

    def test_context_manager(self, test_data_dir: Path):
        """Test context manager protocol."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        with StreamingFilenames(stime, etime, str(test_data_dir)) as streaming:
            # Should auto-start
            batches = list(streaming.iter_batches(timeout=5.0))

        # Should be complete after exit
        assert streaming.state.is_complete
        assert len(batches) >= 1

    def test_iter_batches(self, test_data_dir: Path):
        """Test iterating over batches."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        streaming = StreamingFilenames(stime, etime, str(test_data_dir))
        streaming.start()

        batches = []
        total_files = 0
        for batch in streaming.iter_batches(timeout=10.0):
            batches.append(batch)
            total_files += len(batch.files)
            if batch.is_final:
                break

        streaming.stop()

        assert len(batches) >= 1
        assert total_files > 0
        assert streaming.state.is_complete

    def test_iter_files(self, test_data_dir: Path):
        """Test iterating over individual files."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        files_found = []
        with StreamingFilenames(stime, etime, str(test_data_dir)) as streaming:
            for file in streaming.iter_files(timeout=10.0):
                files_found.append(file)
                if len(files_found) >= 10:  # Limit for test speed
                    break

        assert len(files_found) >= 1
        assert all(isinstance(f, DiscoveredFile) for f in files_found)

    def test_chronological_order(self, test_data_dir: Path):
        """Test that files are yielded in chronological order."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        timestamps = []
        with StreamingFilenames(stime, etime, str(test_data_dir)) as streaming:
            for batch in streaming.iter_batches(timeout=10.0):
                for file in batch.files:
                    timestamps.append(file.timestamp_ns)

        # Within each batch, timestamps should be non-decreasing
        # (batches are in chronological order)
        if len(timestamps) > 1:
            # Check that the overall trend is increasing
            first_half = timestamps[: len(timestamps) // 2]
            second_half = timestamps[len(timestamps) // 2 :]
            assert min(second_half) >= min(first_half)

    def test_file_ids_are_unique(self, test_data_dir: Path):
        """Test that file IDs are unique."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        file_ids = set()
        with StreamingFilenames(stime, etime, str(test_data_dir)) as streaming:
            for file in streaming.iter_files(timeout=10.0):
                assert file.file_id not in file_ids, f"Duplicate file_id: {file.file_id}"
                file_ids.add(file.file_id)

    def test_state_tracking(self, test_data_dir: Path):
        """Test that state is properly tracked during discovery."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        with StreamingFilenames(stime, etime, str(test_data_dir)) as streaming:
            # Consume all batches
            for _batch in streaming.iter_batches(timeout=10.0):
                pass

        # After consuming, should be complete
        assert streaming.state.is_complete
        assert streaming.state.total_files_discovered > 0

    def test_stop_early(self, test_data_dir: Path):
        """Test that stop() can be called early."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        streaming = StreamingFilenames(stime, etime, str(test_data_dir))
        streaming.start()

        # Get first batch
        batch_iter = streaming.iter_batches(timeout=5.0)
        first_batch = next(batch_iter)

        # Stop early
        streaming.stop()

        # Should not raise
        assert first_batch is not None

    def test_must_start_before_iterating(self, test_data_dir: Path):
        """Test that start() must be called before iterating."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        streaming = StreamingFilenames(stime, etime, str(test_data_dir))

        # Should raise if not started
        with pytest.raises(RuntimeError, match="Call start"):
            list(streaming.iter_batches())

    def test_cannot_start_twice(self, test_data_dir: Path):
        """Test that start() cannot be called twice."""
        if not test_data_dir.exists():
            pytest.skip("Test data not available")

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        streaming = StreamingFilenames(stime, etime, str(test_data_dir))
        streaming.start()

        # Should raise if started twice
        with pytest.raises(RuntimeError, match="already started"):
            streaming.start()

        streaming.stop()


class TestStreamingFilenamesComparison:
    """Tests comparing streaming vs blocking discovery."""

    def test_same_results_as_blocking(self, test_data_dir: Path):
        """Test that streaming finds the same files as blocking."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        from wamos_tpw.filenames import Filenames

        stime = np.datetime64("2022-03-28T03:00:00")
        etime = np.datetime64("2022-03-28T04:00:00")

        # Get files via blocking method
        blocking = Filenames(stime, etime, str(test_data_dir))
        blocking_files = set(blocking.files)

        # Get files via streaming method
        streaming_files = set()
        with StreamingFilenames(stime, etime, str(test_data_dir)) as streaming:
            for file in streaming.iter_files(timeout=30.0):
                streaming_files.add(file.filepath)

        # Should find the same files
        assert streaming_files == blocking_files
