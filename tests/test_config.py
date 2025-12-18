"""Tests for WamosConfig class."""

import pytest
import tempfile

from wamos_tpw.config import (
    WamosConfig,
    ShadowConfig,
    RadarConfig,
    OffsetsConfig,
    ThetaRefinementConfig,
    PlottingConfig,
    DestreakConfig,
    _validate_range,
    SAMPLE_CONFIG,
)
from wamos_tpw.exceptions import ConfigError


class TestValidateRange:
    """Tests for _validate_range helper function."""

    def test_valid_value(self):
        """Test valid value passes validation."""
        _validate_range(50.0, "test", min_val=0.0, max_val=100.0)

    def test_none_allowed(self):
        """Test None passes when allow_none=True."""
        _validate_range(None, "test", min_val=0.0, max_val=100.0, allow_none=True)

    def test_none_not_allowed(self):
        """Test None raises when allow_none=False."""
        with pytest.raises(ConfigError, match="value is required"):
            _validate_range(None, "test", min_val=0.0, max_val=100.0, allow_none=False)

    def test_below_min(self):
        """Test value below min raises."""
        with pytest.raises(ConfigError, match="must be >= 0"):
            _validate_range(-5.0, "test", min_val=0.0)

    def test_above_max(self):
        """Test value above max raises."""
        with pytest.raises(ConfigError, match="must be <= 100"):
            _validate_range(150.0, "test", max_val=100.0)


class TestShadowConfig:
    """Tests for ShadowConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = ShadowConfig()
        assert config.center == 180.0
        assert config.width == 90.0

    def test_custom_values(self):
        """Test custom values."""
        config = ShadowConfig(center=90.0, width=45.0)
        assert config.center == 90.0
        assert config.width == 45.0

    def test_start_end(self):
        """Test start and end properties."""
        config = ShadowConfig(center=180.0, width=90.0)
        assert config.start == 135.0  # 180 - 45
        assert config.end == 225.0  # 180 + 45

    def test_start_end_wraparound(self):
        """Test start/end with wraparound."""
        config = ShadowConfig(center=10.0, width=40.0)
        assert config.start == 350.0  # 10 - 20 = -10 -> 350
        assert config.end == 30.0  # 10 + 20

    def test_validation_center(self):
        """Test center validation."""
        with pytest.raises(ConfigError, match="shadow.center"):
            ShadowConfig(center=400.0)

    def test_validation_width(self):
        """Test width validation."""
        with pytest.raises(ConfigError, match="shadow.width"):
            ShadowConfig(width=-10.0)


class TestRadarConfig:
    """Tests for RadarConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = RadarConfig()
        assert config.height is None
        assert config.tower == "UNKNOWN"

    def test_custom_values(self):
        """Test custom values."""
        config = RadarConfig(height=25.0, tower="TOWER1")
        assert config.height == 25.0
        assert config.tower == "TOWER1"

    def test_validation_height_negative(self):
        """Test negative height validation."""
        with pytest.raises(ConfigError, match="radar.height"):
            RadarConfig(height=-5.0)


class TestOffsetsConfig:
    """Tests for OffsetsConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = OffsetsConfig()
        assert config.compass == 0.0
        assert config.bow_to_radar == 0.0
        assert config.heading_delay == 0.0

    def test_custom_values(self):
        """Test custom values."""
        config = OffsetsConfig(compass=5.0, bow_to_radar=10.0, heading_delay=2.0)
        assert config.compass == 5.0
        assert config.bow_to_radar == 10.0
        assert config.heading_delay == 2.0


class TestThetaRefinementConfig:
    """Tests for ThetaRefinementConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = ThetaRefinementConfig()
        assert config.enabled is True
        assert config.search_range == 55.0
        assert config.min_frames == 3
        assert config.intensity_threshold == 0.2

    def test_validation(self):
        """Test validation."""
        with pytest.raises(ConfigError, match="theta_refinement.min_frames"):
            ThetaRefinementConfig(min_frames=0)

        with pytest.raises(ConfigError, match="theta_refinement.intensity_threshold"):
            ThetaRefinementConfig(intensity_threshold=1.5)


class TestPlottingConfig:
    """Tests for PlottingConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = PlottingConfig()
        assert config.cmap == "viridis"
        assert config.intensity_vmin == 0.0
        assert config.intensity_vmax == 4095.0
        assert config.dpi == 150

    def test_validation_vmin_vmax(self):
        """Test vmin/vmax validation."""
        with pytest.raises(ConfigError, match="intensity_vmin.*must be <"):
            PlottingConfig(intensity_vmin=100.0, intensity_vmax=50.0)

    def test_validation_dpi(self):
        """Test DPI validation."""
        with pytest.raises(ConfigError, match="plotting.dpi"):
            PlottingConfig(dpi=0)


class TestDestreakConfig:
    """Tests for DestreakConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = DestreakConfig()
        assert config.min_streak_length == 10
        assert config.threshold_sigma == 7.5

    def test_validation(self):
        """Test validation."""
        with pytest.raises(ConfigError, match="destreak.min_streak_length"):
            DestreakConfig(min_streak_length=0)

        with pytest.raises(ConfigError, match="destreak.threshold_sigma"):
            DestreakConfig(threshold_sigma=-1.0)


class TestWamosConfig:
    """Tests for WamosConfig class."""

    def test_defaults(self):
        """Test default configuration."""
        config = WamosConfig()
        assert config.tower == "UNKNOWN"
        assert isinstance(config.shadow, ShadowConfig)
        assert isinstance(config.radar, RadarConfig)
        assert isinstance(config.offsets, OffsetsConfig)
        assert isinstance(config.theta_refinement, ThetaRefinementConfig)
        assert isinstance(config.plotting, PlottingConfig)
        assert isinstance(config.destreak, DestreakConfig)

    def test_from_yaml_file(self):
        """Test loading from YAML file."""
        yaml_content = """
tower: "TEST_TOWER"
radar:
  height: 25.0
shadow:
  center: 170.0
  width: 80.0
offsets:
  bow_to_radar: 5.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = WamosConfig(f.name)

        assert config.tower == "TEST_TOWER"
        assert config.radar.height == 25.0
        assert config.shadow.center == 170.0
        assert config.shadow.width == 80.0
        assert config.offsets.bow_to_radar == 5.0

    def test_get_method(self):
        """Test get() method for nested access."""
        config = WamosConfig()

        # Dot notation
        assert config.get("shadow.center") == 180.0
        assert config.get("radar.height") is None
        assert config.get("tower") == "UNKNOWN"

        # Default for missing key
        assert config.get("nonexistent", "default") == "default"

    def test_getitem(self):
        """Test __getitem__ method."""
        config = WamosConfig()

        assert config["tower"] == "UNKNOWN"
        assert config["shadow.center"] == 180.0

    def test_repr(self):
        """Test string representation."""
        config = WamosConfig()
        repr_str = repr(config)

        assert "WamosConfig(" in repr_str
        assert "tower=" in repr_str

    def test_str(self):
        """Test __str__ method."""
        config = WamosConfig()
        str_str = str(config)

        # Check that string representation contains relevant info
        assert isinstance(str_str, str)
        assert len(str_str) > 0


class TestSampleConfig:
    """Tests for SAMPLE_CONFIG constant."""

    def test_sample_config_valid_yaml(self):
        """Test that SAMPLE_CONFIG is valid YAML."""
        import yaml

        data = yaml.safe_load(SAMPLE_CONFIG)
        assert isinstance(data, dict)
        assert "tower" in data
        assert "radar" in data
        assert "shadow" in data

    def test_sample_config_loads(self):
        """Test that SAMPLE_CONFIG can be loaded by WamosConfig."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(SAMPLE_CONFIG)
            f.flush()
            config = WamosConfig(f.name)

        # Should have loaded successfully
        assert config.tower != "UNKNOWN"  # Sample has a tower name
