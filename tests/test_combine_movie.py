"""Tests for combine_movie module."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np


class TestDetectShadowEdges:
    """Tests for shadow edge detection."""

    def test_detect_shadow_edges_basic(self, single_polar_file: Path):
        """Test shadow edge detection with real data."""
        from wamos_tpw.combine_movie import _detect_shadow_edges
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.bearing import Theta
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        config = WamosConfig()
        theta = Theta([frame], config, refine=False)
        bearing = theta.bearing_for_frame(0)

        left_edge, right_edge = _detect_shadow_edges(frame, bearing, config)

        # Edges should be None or valid angles
        if left_edge is not None:
            assert 0 <= left_edge < 360
        if right_edge is not None:
            assert 0 <= right_edge < 360

    def test_detect_shadow_edges_empty_intensity(self):
        """Test shadow edge detection with zero intensity."""
        from wamos_tpw.combine_movie import _detect_shadow_edges
        from wamos_tpw.config import WamosConfig

        # Create mock frame with zero intensity
        mock_frame = Mock()
        mock_frame.intensity = np.zeros((360, 100))
        mock_config = WamosConfig()
        bearing = np.arange(360, dtype=float)

        left_edge, right_edge = _detect_shadow_edges(mock_frame, bearing, mock_config)

        # Should return None for both edges
        assert left_edge is None
        assert right_edge is None


class TestComputeChunkShadowOffset:
    """Tests for chunk shadow offset computation."""

    def test_compute_chunk_shadow_offset_basic(self, single_polar_file: Path):
        """Test shadow offset computation with real data."""
        from wamos_tpw.combine_movie import _compute_chunk_shadow_offset
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.bearing import Theta
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        config = WamosConfig()
        theta = Theta(frames, config, refine=False)

        offset, shadow_start, shadow_end = _compute_chunk_shadow_offset(frames, theta, config)

        # Offset should be a reasonable value
        assert isinstance(offset, float)
        assert -180 <= offset <= 180

        # Shadow bounds should be valid
        if shadow_start is not None:
            assert 0 <= shadow_start < 360
        if shadow_end is not None:
            assert 0 <= shadow_end < 360


class TestLoadFileMetadata:
    """Tests for metadata loading."""

    def test_load_file_metadata_success(self, single_polar_file: Path):
        """Test successful metadata loading."""
        from wamos_tpw.combine_movie import _load_file_metadata

        header, metadata_list = _load_file_metadata(str(single_polar_file))

        assert header is not None
        assert isinstance(header, dict)
        assert len(metadata_list) > 0
        assert "FIFO" in header

    def test_load_file_metadata_nonexistent_file(self, tmp_path: Path):
        """Test metadata loading with nonexistent file."""
        from wamos_tpw.combine_movie import _load_file_metadata

        nonexistent_file = tmp_path / "nonexistent.pol"

        header, metadata_list = _load_file_metadata(str(nonexistent_file))

        # Should return None on failure
        assert header is None
        assert metadata_list == []


class TestComputeGridBoundsFromMetadata:
    """Tests for grid bounds computation."""

    def test_compute_grid_bounds_basic(self, april_polar_files: list[Path]):
        """Test grid bounds computation from metadata."""
        from wamos_tpw.combine_movie import _compute_grid_bounds_from_metadata
        from wamos_tpw.config import WamosConfig

        file_list = [str(f) for f in april_polar_files[:2]]
        config = WamosConfig()

        result = _compute_grid_bounds_from_metadata(
            file_list, config, radar_height=25.0, max_frames=10, workers=2
        )

        assert result is not None
        (
            all_metadata,
            x_min,
            x_max,
            y_min,
            y_max,
            max_range,
            ship_x,
            ship_y,
            ship_speeds,
            ship_headings,
            wind_speeds,
            wind_dirs,
        ) = result

        # Check bounds are valid
        assert x_min < x_max
        assert y_min < y_max
        assert max_range > 0

        # Check ship track arrays
        assert len(ship_x) > 0
        assert len(ship_y) == len(ship_x)

    def test_compute_grid_bounds_empty_list(self):
        """Test grid bounds with empty file list."""
        from wamos_tpw.combine_movie import _compute_grid_bounds_from_metadata
        from wamos_tpw.config import WamosConfig

        # Empty list should return None or raise due to no files
        try:
            result = _compute_grid_bounds_from_metadata([], WamosConfig(), None, None)
            # If it doesn't raise, should return None
            assert result is None
        except ValueError:
            # Empty list may raise ValueError about max_workers
            pass


class TestProcessSingleFrame:
    """Tests for single frame processing."""

    def test_process_single_frame(self, single_polar_file: Path):
        """Test processing a single frame."""
        from wamos_tpw.combine_movie import _process_single_frame
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.bearing import Theta
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        config = WamosConfig()
        theta = Theta([frame], config, refine=False)

        # Process the frame
        _process_single_frame(
            frame_idx=0,
            frame=frame,
            theta=theta,
            config=config,
            offset=0.0,
            shadow_start=160.0,
            shadow_end=200.0,
        )

        # Frame should now have corrected_intensity set
        assert frame.corrected_intensity is not None
        assert frame.corrected_intensity.shape == frame.intensity.shape


class TestGridFrameStreaming:
    """Tests for streaming frame gridding."""

    def test_grid_frame_streaming(self, single_polar_file: Path):
        """Test streaming frame gridding."""
        from wamos_tpw.combine_movie import _grid_frame_streaming
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.bearing import Theta, Bearing
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        config = WamosConfig()
        theta = Theta([frame], config, refine=False)
        bearing = Bearing(theta, radar_height=25.0, cache_coordinates=False)

        # Set up grid accumulators
        x_edges = np.linspace(-5000, 5000, 101, dtype=np.float32)
        y_edges = np.linspace(-5000, 5000, 101, dtype=np.float32)
        sum_total = np.zeros((100, 100), dtype=np.float64)
        count_total = np.zeros((100, 100), dtype=np.int32)

        ref_lat = frame.metadata.latitude or 0.0
        ref_lon = frame.metadata.longitude or 0.0

        # Grid the frame
        _grid_frame_streaming(
            frame,
            0,
            theta,
            bearing,
            config,
            x_edges,
            y_edges,
            sum_total,
            count_total,
            ref_lat,
            ref_lon,
        )

        # Should have some gridded values
        assert count_total.sum() > 0
        assert sum_total.sum() > 0


class TestSaveGriddedFrame:
    """Tests for saving gridded frames."""

    def test_save_gridded_frame(self, tmp_path: Path):
        """Test saving a gridded frame as PNG."""
        from wamos_tpw.combine_movie import _save_gridded_frame
        import pandas as pd

        # Create synthetic gridded data
        gridded = np.random.rand(100, 100).astype(np.float32)
        gridded[40:60, 40:60] = np.nan  # Add some NaN values

        x_edges = np.linspace(-1000, 1000, 101, dtype=np.float32)
        y_edges = np.linspace(-1000, 1000, 101, dtype=np.float32)

        output_path = str(tmp_path / "test_frame.png")
        ref_lat, ref_lon = 18.0, 142.0

        first_ts = pd.Timestamp("2022-04-05 14:00:00")
        last_ts = pd.Timestamp("2022-04-05 14:10:00")

        # Ship track
        ship_x = np.linspace(-500, 500, 100)
        ship_y = np.linspace(-500, 500, 100)

        # Ship and wind data
        ship_speeds = [5.0, 5.1, 5.2]
        ship_headings = [45.0, 46.0, 47.0]
        wind_speeds = [10.0, 10.5, 11.0]
        wind_dirs = [270.0, 272.0, 274.0]

        _save_gridded_frame(
            gridded,
            x_edges,
            y_edges,
            output_path,
            ref_lat,
            ref_lon,
            first_ts,
            last_ts,
            10,
            ship_x,
            ship_y,
            ship_speeds,
            ship_headings,
            wind_speeds,
            wind_dirs,
        )

        # Check file was created
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_save_gridded_frame_no_wind(self, tmp_path: Path):
        """Test saving a gridded frame without wind data."""
        from wamos_tpw.combine_movie import _save_gridded_frame
        import pandas as pd

        gridded = np.random.rand(50, 50).astype(np.float32)
        x_edges = np.linspace(-1000, 1000, 51, dtype=np.float32)
        y_edges = np.linspace(-1000, 1000, 51, dtype=np.float32)

        output_path = str(tmp_path / "test_frame_no_wind.png")
        first_ts = pd.Timestamp("2022-04-05 14:00:00")
        last_ts = pd.Timestamp("2022-04-05 14:10:00")

        ship_x = np.linspace(-500, 500, 50)
        ship_y = np.linspace(-500, 500, 50)

        # No ship/wind data
        _save_gridded_frame(
            gridded,
            x_edges,
            y_edges,
            output_path,
            18.0,
            142.0,
            first_ts,
            last_ts,
            5,
            ship_x,
            ship_y,
            [],
            [],
            [],
            [],
        )

        assert Path(output_path).exists()


class TestNormalizeFrames:
    """Tests for frame normalization."""

    def test_normalize_frames_basic(self):
        """Test basic frame normalization."""
        from wamos_tpw.combine_movie import _normalize_frames

        # Create synthetic corrected frames
        frames = [
            np.random.rand(100, 100) * 4095,
            np.random.rand(100, 100) * 4095,
        ]

        normalized = _normalize_frames(frames)

        assert len(normalized) == 2
        for norm in normalized:
            # Normalized values should be mostly in [0, 1]
            valid = norm[np.isfinite(norm)]
            assert valid.min() >= -0.1  # Allow slight undershoot
            assert valid.max() <= 1.1  # Allow slight overshoot

    def test_normalize_frames_with_nan(self):
        """Test normalization with NaN values."""
        from wamos_tpw.combine_movie import _normalize_frames

        frames = [
            np.random.rand(50, 50) * 4095,
        ]
        frames[0][10:20, 10:20] = np.nan

        normalized = _normalize_frames(frames)

        assert len(normalized) == 1
        # NaN values should remain NaN
        assert np.isnan(normalized[0][10:20, 10:20]).all()

    def test_normalize_frames_empty(self):
        """Test normalization with empty list."""
        from wamos_tpw.combine_movie import _normalize_frames

        normalized = _normalize_frames([])
        assert normalized == []


class TestProcessGroup:
    """Tests for group processing."""

    def test_process_group_basic(self, april_polar_files: list[Path], tmp_path: Path):
        """Test processing a group of files."""
        from wamos_tpw.combine_movie import _process_group

        file_list = [str(f) for f in april_polar_files[:1]]
        output_path = str(tmp_path / "test_group.png")

        period_str, result_path, elapsed = _process_group(
            period_str="2022-04-05 14:00",
            file_list=file_list,
            output_path=output_path,
            config_path=None,
            radar_height=25.0,
            max_frames=5,
            do_process=True,
            chunk_size=10,
        )

        assert period_str == "2022-04-05 14:00"
        assert result_path == output_path
        assert elapsed > 0
        assert Path(output_path).exists()

    def test_process_group_empty_files(self, tmp_path: Path):
        """Test processing with empty file list."""
        from wamos_tpw.combine_movie import _process_group

        output_path = str(tmp_path / "test_empty.png")

        period_str, result_path, elapsed = _process_group(
            period_str="2022-04-05 14:00",
            file_list=[],
            output_path=output_path,
            config_path=None,
            radar_height=25.0,
            max_frames=None,
            do_process=True,
        )

        assert period_str == "2022-04-05 14:00"
        assert result_path is None
        assert elapsed >= 0


class TestGenerateMovie:
    """Tests for movie generation."""

    def test_generate_movie_frames_only(self, test_data_dir: Path, tmp_path: Path):
        """Test generating frames without movie."""
        from wamos_tpw.combine_movie import generate_movie
        from wamos_tpw.config import WamosConfig
        from unittest.mock import Mock

        # Create args mock
        args = Mock()
        args.stime = "2022-04-05 14:00"
        args.etime = "2022-04-05 15:00"
        args.polar_path = str(test_data_dir)
        args.groupby = "1h"
        args.radar_height = 25.0
        args.max_frames = 3
        args.process = True
        args.movie = None
        args.frames_dir = str(tmp_path / "frames")
        args.resume = False
        args.fps = 2
        args.workers = 2
        args.config = None

        config = WamosConfig()

        generate_movie(args, config)

        # Check frames were created
        frames_dir = Path(args.frames_dir)
        assert frames_dir.exists()
        png_files = list(frames_dir.glob("*.png"))
        assert len(png_files) >= 1

    def test_generate_movie_resume(self, test_data_dir: Path, tmp_path: Path):
        """Test resume functionality."""
        from wamos_tpw.combine_movie import generate_movie
        from wamos_tpw.config import WamosConfig
        from unittest.mock import Mock
        import pandas as pd

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Create a dummy existing frame
        ts_str = pd.Timestamp("2022-04-05 14:00").strftime("%Y%m%dT%H%M%S")
        existing_frame = frames_dir / f"frame_{ts_str}.png"
        existing_frame.write_bytes(b"PNG placeholder")

        args = Mock()
        args.stime = "2022-04-05 14:00"
        args.etime = "2022-04-05 15:00"
        args.polar_path = str(test_data_dir)
        args.groupby = "1h"
        args.radar_height = 25.0
        args.max_frames = 3
        args.process = True
        args.movie = None
        args.frames_dir = str(frames_dir)
        args.resume = True  # Enable resume
        args.fps = 2
        args.workers = 2
        args.config = None

        config = WamosConfig()

        generate_movie(args, config)

        # Original file should still exist
        assert existing_frame.exists()


class TestMemoryProfiling:
    """Tests for memory profiling feature."""

    def test_report_memory_profile_not_tracing(self, caplog):
        """Test _report_memory_profile does nothing when not tracing."""
        import tracemalloc
        from wamos_tpw.combine_movie import _report_memory_profile

        # Ensure tracemalloc is not running
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        # Should do nothing and not raise
        _report_memory_profile()

        # Should not log anything about memory
        assert "Memory Profile" not in caplog.text

    def test_report_memory_profile_when_tracing(self, caplog):
        """Test _report_memory_profile reports when tracing."""
        import logging
        import tracemalloc
        from wamos_tpw.combine_movie import _report_memory_profile

        # Start tracing
        tracemalloc.start()

        # Allocate some memory to track
        _ = [i for i in range(10000)]

        # Enable INFO logging for caplog
        with caplog.at_level(logging.INFO):
            _report_memory_profile()

        # Should have logged memory info
        assert "Memory Profile" in caplog.text
        assert "Peak" in caplog.text

        # Should have stopped tracing
        assert not tracemalloc.is_tracing()

    def test_generate_movie_with_profiling(self, test_data_dir: Path, tmp_path: Path, caplog):
        """Test generate_movie with memory profiling enabled."""
        import logging
        from unittest.mock import Mock
        from wamos_tpw.combine_movie import generate_movie
        from wamos_tpw.config import WamosConfig

        frames_dir = tmp_path / "profile_frames"

        args = Mock()
        args.stime = "2022-04-05 14:00"
        args.etime = "2022-04-05 15:00"
        args.polar_path = str(test_data_dir)
        args.groupby = "1h"
        args.radar_height = 25.0
        args.max_frames = 2
        args.process = False
        args.movie = None
        args.frames_dir = str(frames_dir)
        args.resume = False
        args.fps = 2
        args.workers = 2
        args.config = None
        args.profile_memory = True  # Enable profiling

        config = WamosConfig()

        with caplog.at_level(logging.INFO):
            generate_movie(args, config)

        # Should have logged memory profiling info
        assert "Memory profiling enabled" in caplog.text
        assert "Memory Profile" in caplog.text
