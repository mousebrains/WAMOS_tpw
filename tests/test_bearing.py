"""Tests for Theta and Bearing classes."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.config import WamosConfig
from wamos_tpw.polarfile import PolarFile


class TestTheta:
    """Tests for Theta class (bearing angle calculation)."""

    def test_theta_single_frame(self, single_polar_file: Path):
        """Test theta calculation with a single frame."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames)

        # Should produce bearing array
        assert theta.bearing is not None
        assert len(theta.bearing) == frames[0].n_bearings

        # All bearings should be in [0, 360)
        assert np.all(theta.bearing >= 0)
        assert np.all(theta.bearing < 360)

    def test_theta_multiple_frames(self, april_polar_files: list[Path]):
        """Test theta calculation with multiple frames."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 2:
            pytest.skip("Need at least 2 frames")

        theta = Theta(frames)

        # Total bearings should be sum of all frames
        expected_total = sum(f.n_bearings for f in frames)
        assert len(theta.bearing) == expected_total

        # Per-frame bearings should match
        assert len(theta.bearing_per_frame) == len(frames)
        for i, frame in enumerate(frames):
            assert len(theta.bearing_for_frame(i)) == frame.n_bearings

    def test_theta_with_config(self, single_polar_file: Path):
        """Test theta with custom configuration."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        config = WamosConfig()
        config.shadow.center = 180.0
        config.shadow.width = 90.0

        theta = Theta(frames, config, refine=False)

        assert theta.config is config
        assert theta.shadow_offset == 0.0  # No refinement

    def test_theta_empty_frames_raises(self):
        """Test that empty frames list raises ValueError."""
        with pytest.raises(ValueError, match="At least one frame"):
            Theta([])

    def test_theta_bearing_wraparound(self, single_polar_file: Path):
        """Test that bearing values wrap correctly around 360."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames)

        # Check wraparound: all values should be in [0, 360)
        bearing = theta.bearing
        assert bearing.min() >= 0.0
        assert bearing.max() < 360.0

    def test_theta_in_shadow(self, single_polar_file: Path):
        """Test shadow mask calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        config = WamosConfig()
        config.shadow.center = 180.0
        config.shadow.width = 90.0  # 135-225 degrees

        theta = Theta(frames, config, refine=False)
        shadow_mask = theta.in_shadow(0)

        # Some bearings should be in shadow, some not
        assert shadow_mask.dtype == bool
        assert len(shadow_mask) == frames[0].n_bearings

    def test_theta_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames, refine=False)
        repr_str = repr(theta)

        assert "Theta(" in repr_str
        assert "frames=" in repr_str
        assert "radials=" in repr_str


class TestBearing:
    """Tests for Bearing class (coordinate transformations)."""

    def test_bearing_basic(self, single_polar_file: Path):
        """Test basic Bearing creation and properties."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames, refine=False)
        bearing = Bearing(theta, radar_height=25.0)

        assert bearing.radar_height == 25.0
        assert bearing.theta is theta
        assert bearing.config is theta.config

    def test_heading_ship(self, single_polar_file: Path):
        """Test ship-relative heading calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames, refine=False)
        bearing = Bearing(theta)

        heading = bearing.heading_ship(0)

        assert len(heading) == frames[0].n_bearings
        assert np.all(heading >= 0)
        assert np.all(heading < 360)

    def test_heading_earth(self, single_polar_file: Path):
        """Test earth-relative heading calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames, refine=False)
        bearing = Bearing(theta)

        heading = bearing.heading_earth(0)

        assert len(heading) == frames[0].n_bearings
        assert np.all(heading >= 0)
        assert np.all(heading < 360)

    def test_xy_ship(self, single_polar_file: Path):
        """Test ship coordinate calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        theta = Theta(frames, refine=False)
        bearing = Bearing(theta, radar_height=25.0)

        x, y = bearing.xy_ship(0)

        # Should be 2D arrays
        assert x.shape == (frame.n_bearings, frame.n_distances)
        assert y.shape == (frame.n_bearings, frame.n_distances)

        # Origin should have small values (near ship)
        assert x[:, 0].mean() < 100  # Close to center

    def test_xy_earth(self, single_polar_file: Path):
        """Test earth coordinate calculation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]
        frame = frames[0]

        theta = Theta(frames, refine=False)
        bearing = Bearing(theta, radar_height=25.0)

        x, y = bearing.xy_earth(0)

        # Should be 2D arrays
        assert x.shape == (frame.n_bearings, frame.n_distances)
        assert y.shape == (frame.n_bearings, frame.n_distances)

    def test_bearing_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        theta = Theta(frames, refine=False)
        bearing = Bearing(theta, radar_height=25.0)

        repr_str = repr(bearing)
        assert "Bearing(" in repr_str
        assert "frames=" in repr_str
        assert "radar_height=" in repr_str

    def test_heading_with_offsets(self, single_polar_file: Path):
        """Test heading calculation with compass/mounting offsets."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        config = WamosConfig()
        config.offsets.bow_to_radar = 10.0  # 10 degree offset
        config.offsets.heading_delay = 5.0

        theta = Theta(frames, config, refine=False)
        bearing = Bearing(theta)

        heading_ship = bearing.heading_ship(0)
        heading_image = bearing.heading_image(0)

        # heading_image should differ from heading_ship by heading_delay
        diff = (heading_image - heading_ship) % 360
        # Should be approximately 5 degrees
        assert np.allclose(diff, 5.0, atol=0.1) or np.allclose(diff, 355.0, atol=0.1)


class TestCircularStatistics:
    """Tests for circular statistics helper methods."""

    def test_circular_mean_simple(self):
        """Test circular mean with simple values."""
        # Mean of 0, 90, 180, 270 should be undefined (near center)
        # but mean of 10, 20, 30 should be ~20
        angles = [10.0, 20.0, 30.0]
        mean, _ = Theta._circular_stats(angles)
        assert abs(mean - 20.0) < 1.0

    def test_circular_mean_wraparound(self):
        """Test circular mean handles wraparound correctly."""
        # Mean of 350, 10 should be ~0 (midnight)
        angles = [350.0, 10.0]
        mean, _ = Theta._circular_stats(angles)
        # Mean should be close to 0 (or 360)
        assert mean < 20.0 or mean > 340.0

    def test_circular_std(self):
        """Test circular standard deviation."""
        # Tightly clustered angles should have low std
        angles = [90.0, 91.0, 89.0, 90.5]
        _, std = Theta._circular_stats(angles)
        assert std < 5.0

        # Spread out angles should have higher std
        angles_spread = [0.0, 90.0, 180.0, 270.0]
        _, std_spread = Theta._circular_stats(angles_spread)
        assert std_spread > std
