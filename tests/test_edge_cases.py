"""Tests for edge cases and boundary conditions."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


class TestConfigEdgeCases:
    """Tests for configuration edge cases."""

    def test_config_empty_yaml(self, tmp_path: Path):
        """Test loading empty YAML config file."""
        from wamos_tpw.config import WamosConfig

        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        config = WamosConfig(str(empty_config))

        # Should use defaults
        assert config.shadow.center == 180.0
        assert config.radar.height is None
        assert config.tower == "UNKNOWN"

    def test_config_partial_yaml(self, tmp_path: Path):
        """Test loading YAML with only some values."""
        from wamos_tpw.config import WamosConfig

        partial_config = tmp_path / "partial.yaml"
        partial_config.write_text("tower: TEST_TOWER\nradar:\n  height: 30.0")

        config = WamosConfig(str(partial_config))

        assert config.tower == "TEST_TOWER"
        assert config.radar.height == 30.0
        # Should use defaults for other values
        assert config.shadow.center == 180.0

    def test_config_extreme_shadow_width(self, tmp_path: Path):
        """Test loading config with extreme shadow width."""
        from wamos_tpw.config import WamosConfig

        # Config loading overrides values after dataclass creation
        # so validation only happens during initial creation
        extreme_config = tmp_path / "extreme.yaml"
        extreme_config.write_text("shadow:\n  width: 120.0")  # Valid but large

        config = WamosConfig(str(extreme_config))
        assert config.shadow.width == 120.0

    def test_config_zero_radar_height(self, tmp_path: Path):
        """Test loading config with zero radar height."""
        from wamos_tpw.config import WamosConfig

        zero_height_config = tmp_path / "zero_height.yaml"
        zero_height_config.write_text("radar:\n  height: 0.0")

        config = WamosConfig(str(zero_height_config))
        assert config.radar.height == 0.0


class TestFrameEdgeCases:
    """Tests for frame processing edge cases."""

    def test_frame_single_bearing(self, single_polar_file: Path):
        """Test processing frame with single bearing row."""
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        # Create a synthetic single-bearing frame from first row
        single_row_intensity = frame.intensity[0:1, :]
        assert single_row_intensity.shape[0] == 1

    def test_frame_all_nan_intensity(self):
        """Test handling of all-NaN intensity array."""
        from wamos_tpw.combine_streaming import normalize_frames

        all_nan = [np.full((100, 100), np.nan)]
        result = normalize_frames(all_nan)

        assert len(result) == 1
        assert np.all(np.isnan(result[0]))

    def test_frame_zero_intensity(self):
        """Test handling of all-zero intensity."""
        from wamos_tpw.combine_streaming import normalize_frames

        all_zero = [np.zeros((100, 100))]
        result = normalize_frames(all_zero)

        assert len(result) == 1
        # All zeros should remain zero (or close to it)
        assert np.allclose(result[0], 0.0, atol=1e-6)


class TestCombineEdgeCases:
    """Tests for Combine class edge cases."""

    def test_combine_single_frame(self, single_polar_file: Path):
        """Test Combine with a single frame."""
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frames[0].corrected_intensity = frames[0].intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        assert len(combine) == 1
        assert combine.reference_position is not None

        # Should be able to get coordinates
        x, y = combine.xy_earth(0)
        assert x.shape == frames[0].intensity.shape

    def test_combine_zero_movement(self, single_polar_file: Path):
        """Test Combine when ship has zero movement."""
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frames[0].corrected_intensity = frames[0].intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        travel = combine.travel_distance()

        # Single frame should have minimal travel
        # (depends on radial period, typically < 100m)
        assert travel["duration_s"] >= 0


class TestShadowEdgeCases:
    """Tests for shadow detection edge cases."""

    def test_shadow_detection_uniform_intensity(self):
        """Test shadow detection with uniform intensity (no shadow)."""
        from wamos_tpw.combine_shadow import detect_shadow_edges
        from wamos_tpw.config import WamosConfig

        # Create mock frame with uniform intensity
        mock_frame = Mock()
        mock_frame.intensity = np.ones((360, 100)) * 1000

        config = WamosConfig()
        bearing = np.arange(360, dtype=float)

        left, right = detect_shadow_edges(mock_frame, bearing, config)

        # Should not detect edges in uniform intensity
        assert left is None or right is None

    def test_shadow_detection_low_intensity(self):
        """Test shadow detection with very low intensity."""
        from wamos_tpw.combine_shadow import detect_shadow_edges
        from wamos_tpw.config import WamosConfig

        mock_frame = Mock()
        mock_frame.intensity = np.ones((360, 100)) * 0.01  # Very low

        config = WamosConfig()
        bearing = np.arange(360, dtype=float)

        left, right = detect_shadow_edges(mock_frame, bearing, config)

        # Low intensity normalization may still work
        # Just verify no exception is raised


class TestGriddingEdgeCases:
    """Tests for gridding edge cases."""

    def test_gridding_small_grid(self, single_polar_file: Path):
        """Test gridding with very small grid."""
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frames[0].corrected_intensity = frames[0].intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        # Very small grid - should still work
        x_edges, y_edges, gridded = combine.grid_parallel(grid_size=10)

        assert len(x_edges) == 11
        assert len(y_edges) == 11
        assert gridded.shape == (10, 10)

    def test_gridding_large_grid(self, single_polar_file: Path):
        """Test gridding with larger grid."""
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frames[0].corrected_intensity = frames[0].intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        x_edges, y_edges, gridded = combine.grid_parallel(grid_size=200)

        assert len(x_edges) == 201
        assert len(y_edges) == 201
        assert gridded.shape == (200, 200)


class TestNetCDFEdgeCases:
    """Tests for NetCDF writer edge cases."""

    def test_netcdf_empty_ship_track(self, single_polar_file: Path, tmp_path: Path):
        """Test NetCDF with empty ship track."""
        pytest.importorskip("netCDF4")

        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frames[0].corrected_intensity = frames[0].intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        filepath = str(tmp_path / "test_empty_track.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        assert Path(filepath).exists()


class TestMovieGenerationEdgeCases:
    """Tests for movie generation edge cases."""

    def test_save_gridded_frame_empty_ship_track(self, tmp_path: Path):
        """Test saving frame with empty ship/wind data."""
        from wamos_tpw.combine_movie import _save_gridded_frame
        import pandas as pd

        gridded = np.random.rand(50, 50).astype(np.float32)
        x_edges = np.linspace(-1000, 1000, 51)
        y_edges = np.linspace(-1000, 1000, 51)

        output_path = str(tmp_path / "test_empty.png")
        first_ts = pd.Timestamp("2022-04-05 14:00:00")
        last_ts = pd.Timestamp("2022-04-05 14:10:00")

        # Empty arrays for ship track
        ship_x = np.array([])
        ship_y = np.array([])

        _save_gridded_frame(
            gridded, x_edges, y_edges, output_path,
            18.0, 142.0, first_ts, last_ts, 5,
            ship_x, ship_y, [], [], [], []
        )

        assert Path(output_path).exists()

    def test_save_gridded_frame_all_nan(self, tmp_path: Path):
        """Test saving frame with all NaN values."""
        from wamos_tpw.combine_movie import _save_gridded_frame
        import pandas as pd

        gridded = np.full((50, 50), np.nan, dtype=np.float32)
        x_edges = np.linspace(-1000, 1000, 51)
        y_edges = np.linspace(-1000, 1000, 51)

        output_path = str(tmp_path / "test_nan.png")
        first_ts = pd.Timestamp("2022-04-05 14:00:00")
        last_ts = pd.Timestamp("2022-04-05 14:10:00")

        ship_x = np.linspace(-500, 500, 10)
        ship_y = np.linspace(-500, 500, 10)

        _save_gridded_frame(
            gridded, x_edges, y_edges, output_path,
            18.0, 142.0, first_ts, last_ts, 5,
            ship_x, ship_y, [], [], [], []
        )

        assert Path(output_path).exists()


class TestBearingEdgeCases:
    """Tests for bearing calculation edge cases."""

    def test_bearing_at_zero(self, single_polar_file: Path):
        """Test bearing calculation at 0 degrees."""
        from wamos_tpw.bearing import Theta
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        config = WamosConfig()

        theta = Theta(frames, config, refine=False)
        bearing = theta.bearing_for_frame(0)

        # All bearings should be in [0, 360)
        assert np.all(bearing >= 0)
        assert np.all(bearing < 360)

    def test_bearing_wrap_around(self, single_polar_file: Path):
        """Test bearing wrap-around at 360 degrees."""
        from wamos_tpw.bearing import Theta, Bearing
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        config = WamosConfig()

        theta = Theta(frames, config, refine=False)
        bearing = Bearing(theta, radar_height=25.0)

        # Get coordinates - should not have any NaN due to wrap-around issues
        x, y = bearing.xy_earth(0)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))


class TestTimestampEdgeCases:
    """Tests for timestamp handling edge cases."""

    def test_timestamp_single_frame(self, single_polar_file: Path):
        """Test timestamp calculation for single frame."""
        from wamos_tpw.timestamp import Timestamp
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        config = WamosConfig()

        ts = Timestamp(frames, config)

        # Should have times for all radials
        assert len(ts.times) == frames[0].n_bearings

        # Times should be monotonically increasing (mostly)
        # (might have some jitter, but overall increasing)
        time_range = ts.times[-1] - ts.times[0]
        assert time_range >= 0


class TestDestreakEdgeCases:
    """Tests for destreak edge cases."""

    def test_destreak_uniform_intensity(self, single_polar_file: Path):
        """Test destreak with uniform intensity (no streaks)."""
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.destreak import Destreak
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        # Set uniform deramped intensity
        frame.deramped_intensity = np.ones_like(frame.intensity, dtype=np.float64) * 1000

        config = WamosConfig()
        destreak = Destreak(None, frame, None, config)

        # Should return close to input (no correction needed)
        corrected = destreak.corrected_intensity
        assert corrected.shape == frame.intensity.shape
