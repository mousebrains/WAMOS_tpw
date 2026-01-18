"""Tests for Config class."""

import tempfile

import pytest

from wamos_tpw.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_config_loads(self):
        """Test that default config loads successfully."""
        config = Config()
        assert config is not None
        assert len(config) > 0

    def test_default_has_shadow(self):
        """Test default config has shadow settings."""
        config = Config()
        assert "shadow" in config
        shadow = config["shadow"]
        assert shadow["start"] == 130
        assert shadow["end"] == 230

    def test_default_has_theta_refinement(self):
        """Test default config has theta refinement settings."""
        config = Config()
        assert "theta_refinement" in config
        theta = config["theta_refinement"]
        assert theta["enabled"] is True
        assert theta["min_frames"] == 3

    def test_get_with_default(self):
        """Test get() returns default for missing keys."""
        config = Config()
        assert config.get("nonexistent", "default") == "default"
        assert config.get("nonexistent") is None

    def test_dot_notation_access(self):
        """Test dot notation for nested keys."""
        config = Config()
        assert config["shadow.start"] == 130
        assert config["theta_refinement.enabled"] is True

    def test_attribute_access(self):
        """Test attribute-style access."""
        config = Config()
        assert config.shadow.start == 130
        assert config.theta_refinement.enabled is True

    def test_setitem(self):
        """Test setting values with dot notation."""
        config = Config()
        config["test.nested.value"] = 42
        assert config["test.nested.value"] == 42

    def test_contains(self):
        """Test __contains__ method."""
        config = Config()
        assert "shadow" in config
        assert "shadow.start" in config
        assert "nonexistent" not in config

    def test_iter(self):
        """Test iteration over keys."""
        config = Config()
        keys = list(config)
        assert "shadow" in keys
        assert "theta_refinement" in keys

    def test_bool_nonempty(self):
        """Test bool for non-empty config."""
        config = Config()
        assert bool(config) is True

    def test_repr(self):
        """Test string representation."""
        config = Config()
        repr_str = repr(config)
        assert "Config(" in repr_str

    def test_from_yaml_file(self):
        """Test loading from YAML file."""
        yaml_content = """
tower: "TEST_TOWER"
radar:
  height: 25.0
shadow:
  start: 140
  end: 220
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = Config(f.name)

        assert config["tower"] == "TEST_TOWER"
        assert config["radar.height"] == 25.0
        assert config["shadow.start"] == 140

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")

    def test_child_config(self):
        """Test that nested dicts return child Config objects."""
        config = Config()
        shadow = config["shadow"]
        assert isinstance(shadow, Config)
        assert shadow["start"] == 130

    def test_keys_method(self):
        """Test keys() method."""
        config = Config()
        keys = config.keys()
        assert "shadow" in keys

    def test_items_method(self):
        """Test items() method."""
        config = Config()
        items = dict(config.items())
        assert "shadow" in items

    def test_values_method(self):
        """Test values() method."""
        config = Config()
        values = list(config.values())
        assert len(values) > 0

    def test_update(self):
        """Test update() method."""
        config = Config()
        config.update({"new_key": "new_value"})
        assert config["new_key"] == "new_value"

    def test_update_from_config(self):
        """Test update() from another Config."""
        config1 = Config()
        config2 = Config()
        config2["custom"] = "value"
        config1.update(config2)
        assert config1["custom"] == "value"


class TestWamosConfigAlias:
    """Test backward compatibility alias."""

    def test_wamos_config_alias(self):
        """Test WamosConfig is an alias for Config."""
        from wamos_tpw.config import WamosConfig

        assert WamosConfig is Config
