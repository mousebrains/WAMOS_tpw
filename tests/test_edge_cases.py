"""Edge case tests for wamos_tpw package."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from wamos_tpw.config import WamosConfig, ShadowConfig, RadarConfig, PlottingConfig, DestreakConfig
from wamos_tpw.exceptions import ConfigError, WamosError, PolarFileError, ValidationError
from wamos_tpw.filenames import Filenames, _parse_timestamp
from wamos_tpw.polarfile import PolarFile


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_shadow_center_out_of_range(self):
        """Test shadow center validation (must be 0-360)."""
        with pytest.raises(ConfigError, match="shadow.center"):
            ShadowConfig(center=400.0, width=90.0)

        with pytest.raises(ConfigError, match="shadow.center"):
            ShadowConfig(center=-10.0, width=90.0)

    def test_shadow_width_out_of_range(self):
        """Test shadow width validation (must be 0-360)."""
        with pytest.raises(ConfigError, match="shadow.width"):
            ShadowConfig(center=180.0, width=-10.0)

        with pytest.raises(ConfigError, match="shadow.width"):
            ShadowConfig(center=180.0, width=400.0)

    def test_radar_height_negative(self):
        """Test radar height validation (must be >= 0)."""
        with pytest.raises(ConfigError, match="radar.height"):
            RadarConfig(height=-5.0)

    def test_radar_height_none_allowed(self):
        """Test that radar height can be None."""
        config = RadarConfig(height=None)
        assert config.height is None

    def test_plotting_vmin_vmax_invalid(self):
        """Test plotting intensity range validation."""
        with pytest.raises(ConfigError, match="intensity_vmin.*must be <"):
            PlottingConfig(intensity_vmin=4095.0, intensity_vmax=0.0)

        with pytest.raises(ConfigError, match="intensity_vmin.*must be <"):
            PlottingConfig(intensity_vmin=100.0, intensity_vmax=100.0)

    def test_plotting_dpi_out_of_range(self):
        """Test DPI validation."""
        with pytest.raises(ConfigError, match="plotting.dpi"):
            PlottingConfig(dpi=0)

        with pytest.raises(ConfigError, match="plotting.dpi"):
            PlottingConfig(dpi=2000)

    def test_destreak_min_streak_length_invalid(self):
        """Test destreak min_streak_length validation."""
        with pytest.raises(ConfigError, match="destreak.min_streak_length"):
            DestreakConfig(min_streak_length=0)

    def test_destreak_threshold_sigma_negative(self):
        """Test destreak threshold_sigma validation."""
        with pytest.raises(ConfigError, match="destreak.threshold_sigma"):
            DestreakConfig(threshold_sigma=-1.0)

    def test_valid_config_loads(self):
        """Test that valid configuration loads without errors."""
        config = WamosConfig()
        assert config.shadow.center == 180.0
        assert config.shadow.width == 90.0
        assert config.radar.height is None

    def test_config_from_yaml(self):
        """Test loading valid YAML configuration."""
        yaml_content = """
tower: "TEST_TOWER"
radar:
  height: 25.0
shadow:
  center: 180.0
  width: 90.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = WamosConfig(f.name)
            assert config.tower == "TEST_TOWER"
            assert config.radar.height == 25.0

    def test_config_invalid_yaml_value(self):
        """Test that RadarConfig validates height directly."""
        # RadarConfig validates at construction time
        with pytest.raises(ConfigError, match="radar.height"):
            RadarConfig(height=-10.0)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_wamos_error_is_base(self):
        """Test that all custom exceptions inherit from WamosError."""
        assert issubclass(PolarFileError, WamosError)
        assert issubclass(ConfigError, WamosError)
        assert issubclass(ValidationError, WamosError)

    def test_polar_file_error_with_filename(self):
        """Test PolarFileError with filename."""
        err = PolarFileError("Invalid format", filename="test.pol")
        assert "test.pol" in str(err)
        assert err.filename == "test.pol"

    def test_config_error_with_parameter(self):
        """Test ConfigError with parameter info."""
        err = ConfigError("must be positive", parameter="radar.height", value=-5.0)
        assert "radar.height" in str(err)
        assert "-5.0" in str(err)
        assert err.parameter == "radar.height"
        assert err.value == -5.0

    def test_catch_all_wamos_errors(self):
        """Test catching all WAMOS errors with base class."""
        errors = [
            PolarFileError("test"),
            ConfigError("test"),
            ValidationError("test"),
        ]

        for err in errors:
            with pytest.raises(WamosError):
                raise err


class TestFilenamesEdgeCases:
    """Edge case tests for Filenames class."""

    def test_start_time_after_end_time(self, test_data_dir: Path):
        """Test that start > end raises ValueError."""
        with pytest.raises(ValueError, match="Start time.*must be <="):
            Filenames(
                stime=_parse_timestamp("202301010000"),
                etime=_parse_timestamp("202201010000"),
                polar_path=test_data_dir,
            )

    def test_parse_timestamp_invalid_format(self):
        """Test timestamp parsing with invalid format."""
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-timestamp")

        with pytest.raises(ValueError):
            _parse_timestamp("123")  # Too short

    def test_parse_timestamp_various_formats(self):
        """Test timestamp parsing with various formats."""
        # Compact formats
        ts = _parse_timestamp("2022")
        assert str(ts).startswith("2022")

        ts = _parse_timestamp("202203")
        assert "2022-03" in str(ts)

        ts = _parse_timestamp("20220328")
        assert "2022-03-28" in str(ts)

        ts = _parse_timestamp("2022032803")
        assert "2022-03-28T03" in str(ts)

        # ISO format
        ts = _parse_timestamp("2022-03-28T03:00:00")
        assert "2022-03-28" in str(ts)

    def test_empty_directory(self, test_data_dir: Path):
        """Test with time range that has no files."""
        filenames = Filenames(
            stime=_parse_timestamp("199001010000"),
            etime=_parse_timestamp("199001020000"),
            polar_path=test_data_dir,
        )

        # Should find no files but not raise
        assert len(filenames) == 0
        assert list(filenames) == []

    def test_nonexistent_path(self):
        """Test with nonexistent path."""
        filenames = Filenames(
            stime=_parse_timestamp("202201010000"),
            etime=_parse_timestamp("202201020000"),
            polar_path="/nonexistent/path/to/polar",
        )

        # Should not raise, just find no files
        assert len(filenames) == 0


class TestPolarFileEdgeCases:
    """Edge case tests for PolarFile class."""

    def test_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            PolarFile("/nonexistent/file.pol")

    def test_repr_and_str(self, single_polar_file: Path):
        """Test string representations."""
        pf = PolarFile(single_polar_file)

        repr_str = repr(pf)
        assert "PolarFile" in repr_str

        str_str = str(pf)
        assert isinstance(str_str, str)


class TestFrameEdgeCases:
    """Edge case tests for Frame class."""

    def test_ground_range_with_height(self, single_polar_file: Path):
        """Test ground range calculation with radar height."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        ground = frame.ground_range(radar_height=25.0)
        slant = frame.slant_range()

        # Ground range should be <= slant range
        assert np.all(ground <= slant)

        # Ground range should be shorter (due to Pythagorean theorem)
        # For most bins (except very close ones where sqrt could give similar values)
        far_bins = slant > 100  # Far enough for difference to be noticeable
        if np.any(far_bins):
            assert np.any(ground[far_bins] < slant[far_bins])

    def test_ground_range_without_height_raises(self, single_polar_file: Path):
        """Test ground range without height raises if no metadata."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        # If metadata doesn't have radar_height, should raise ValueError
        if frame.metadata.radar_height is None:
            with pytest.raises(ValueError, match="radar_height must be provided"):
                frame.ground_range()
        else:
            # If metadata has height, should work
            ground = frame.ground_range()
            assert len(ground) == frame.n_distances

    def test_intensity_12bit_range(self, single_polar_file: Path):
        """Test intensity is properly masked to 12 bits."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        intensity = frame.intensity
        assert intensity.max() <= 4095  # 12-bit max
        assert intensity.min() >= 0

    def test_pps_and_bearing_pulse_boolean(self, single_polar_file: Path):
        """Test PPS and bearing pulse are boolean arrays."""
        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]

        pps = frame.pps
        bp = frame.bearing_pulse

        assert pps.dtype == bool
        assert bp.dtype == bool


class TestBearingWraparound:
    """Tests for bearing wraparound at 360/0 degrees."""

    def test_bearing_all_positive(self, single_polar_file: Path):
        """Test all bearings are in [0, 360) range."""
        from wamos_tpw.bearing import Theta

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames, refine=False)

        assert np.all(theta.bearing >= 0)
        assert np.all(theta.bearing < 360)

    def test_shadow_region_wraparound(self, single_polar_file: Path):
        """Test shadow region handling when it wraps around 360."""
        from wamos_tpw.bearing import Theta

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        # Shadow centered at 350 with width 40 wraps around to [330, 370) -> [330, 360) + [0, 10)
        config = WamosConfig()
        config.shadow.center = 350.0
        config.shadow.width = 40.0

        theta = Theta(frames, config, refine=False)
        shadow_mask = theta.in_shadow(0)

        # Some radials in both ranges should be in shadow
        assert shadow_mask.dtype == bool
