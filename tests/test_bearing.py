"""Tests for Bearing class and coordinate conversion functions."""

from pathlib import Path

import numpy as np

from wamos_tpw.bearing import (
    Bearing,
    heading_to_xy,
    theta_to_heading_earth,
    theta_to_heading_ship,
)
from wamos_tpw.config import Config
from wamos_tpw.polarfile import PolarFile
from wamos_tpw.range import Range
from wamos_tpw.theta import Theta


class TestThetaToHeadingShip:
    """Tests for theta_to_heading_ship function."""

    def test_no_offset(self):
        """Test with zero offset - output equals input."""
        theta = np.array([0, 90, 180, 270], dtype=np.float32)
        heading = theta_to_heading_ship(theta, bow_to_radar=0.0)
        np.testing.assert_array_almost_equal(heading, theta)

    def test_with_offset(self):
        """Test with bow_to_radar offset."""
        theta = np.array([0, 90, 180, 270], dtype=np.float32)
        heading = theta_to_heading_ship(theta, bow_to_radar=10.0)
        expected = np.array([10, 100, 190, 280], dtype=np.float32)
        np.testing.assert_array_almost_equal(heading, expected)

    def test_wraparound(self):
        """Test that values wrap correctly at 360."""
        theta = np.array([350, 355, 359], dtype=np.float32)
        heading = theta_to_heading_ship(theta, bow_to_radar=20.0)
        expected = np.array([10, 15, 19], dtype=np.float32)
        np.testing.assert_array_almost_equal(heading, expected)

    def test_negative_offset(self):
        """Test with negative offset."""
        theta = np.array([0, 10, 350], dtype=np.float32)
        heading = theta_to_heading_ship(theta, bow_to_radar=-20.0)
        expected = np.array([340, 350, 330], dtype=np.float32)
        np.testing.assert_array_almost_equal(heading, expected)


class TestThetaToHeadingEarth:
    """Tests for theta_to_heading_earth function."""

    def test_ship_heading_only(self):
        """Test with only ship heading offset."""
        theta = np.array([0, 90, 180, 270], dtype=np.float32)
        heading = theta_to_heading_earth(theta, ship_heading=45.0)
        expected = np.array([45, 135, 225, 315], dtype=np.float32)
        np.testing.assert_array_almost_equal(heading, expected)

    def test_all_offsets(self):
        """Test with all offset parameters."""
        theta = np.array([0, 90], dtype=np.float32)
        heading = theta_to_heading_earth(
            theta,
            ship_heading=100.0,
            bow_to_radar=10.0,
            heading_delay=5.0,
            compass_offset=-3.0,
        )
        # 0 + 10 + 5 + 100 - 3 = 112
        # 90 + 10 + 5 + 100 - 3 = 202
        expected = np.array([112, 202], dtype=np.float32)
        np.testing.assert_array_almost_equal(heading, expected)

    def test_wraparound(self):
        """Test wraparound with ship heading."""
        theta = np.array([300, 350], dtype=np.float32)
        heading = theta_to_heading_earth(theta, ship_heading=100.0)
        expected = np.array([40, 90], dtype=np.float32)
        np.testing.assert_array_almost_equal(heading, expected)


class TestHeadingToXY:
    """Tests for heading_to_xy function."""

    def test_north_heading(self):
        """Test heading=0 (North) produces +Y."""
        heading = np.array([0.0])
        ground_range = np.array([100.0, 200.0])
        x, y = heading_to_xy(heading, ground_range)

        # At heading 0 (North), x=0 and y=range
        np.testing.assert_array_almost_equal(x[0], [0, 0], decimal=5)
        np.testing.assert_array_almost_equal(y[0], [100, 200], decimal=5)

    def test_east_heading(self):
        """Test heading=90 (East) produces +X."""
        heading = np.array([90.0])
        ground_range = np.array([100.0])
        x, y = heading_to_xy(heading, ground_range)

        # At heading 90 (East), x=range and y=0
        np.testing.assert_array_almost_equal(x[0], [100], decimal=5)
        np.testing.assert_array_almost_equal(y[0], [0], decimal=4)

    def test_south_heading(self):
        """Test heading=180 (South) produces -Y."""
        heading = np.array([180.0])
        ground_range = np.array([100.0])
        x, y = heading_to_xy(heading, ground_range)

        # At heading 180 (South), x=0 and y=-range
        np.testing.assert_array_almost_equal(x[0], [0], decimal=4)
        np.testing.assert_array_almost_equal(y[0], [-100], decimal=5)

    def test_west_heading(self):
        """Test heading=270 (West) produces -X."""
        heading = np.array([270.0])
        ground_range = np.array([100.0])
        x, y = heading_to_xy(heading, ground_range)

        # At heading 270 (West), x=-range and y=0
        np.testing.assert_array_almost_equal(x[0], [-100], decimal=5)
        np.testing.assert_array_almost_equal(y[0], [0], decimal=4)

    def test_output_shape(self):
        """Test output shape is (n_radials, n_distances)."""
        heading = np.array([0, 90, 180, 270], dtype=np.float32)
        ground_range = np.array([100, 200, 300], dtype=np.float32)
        x, y = heading_to_xy(heading, ground_range)

        assert x.shape == (4, 3)
        assert y.shape == (4, 3)


class TestBearing:
    """Tests for Bearing class."""

    def test_bearing_basic(self, single_polar_file: Path):
        """Test basic Bearing creation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        assert len(bearing.theta) == frame.n_bearings
        assert len(bearing.ground_range) == frame.n_distances

    def test_heading_ship(self, single_polar_file: Path):
        """Test ship-relative heading calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        heading = bearing.heading_ship()

        assert len(heading) == frame.n_bearings
        assert np.all(heading >= 0)
        assert np.all(heading < 360)

    def test_heading_earth(self, single_polar_file: Path):
        """Test earth-relative heading calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        heading = bearing.heading_earth()

        assert len(heading) == frame.n_bearings
        assert np.all(heading >= 0)
        assert np.all(heading < 360)

    def test_xy_ship(self, single_polar_file: Path):
        """Test ship coordinate calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        x, y = bearing.xy_ship()

        # Should be 2D arrays
        assert x.shape == (frame.n_bearings, frame.n_distances)
        assert y.shape == (frame.n_bearings, frame.n_distances)

    def test_xy_earth(self, single_polar_file: Path):
        """Test earth coordinate calculation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        x, y = bearing.xy_earth()

        # Should be 2D arrays
        assert x.shape == (frame.n_bearings, frame.n_distances)
        assert y.shape == (frame.n_bearings, frame.n_distances)

    def test_bearing_repr(self, single_polar_file: Path):
        """Test string representation."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        repr_str = repr(bearing)
        assert "Bearing(" in repr_str
        assert "n_radials=" in repr_str
        assert "n_distances=" in repr_str

    def test_heading_with_offsets(self, single_polar_file: Path):
        """Test heading calculation with compass/mounting offsets."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        config = Config()
        config["offsets.bow_to_radar"] = 10.0
        config["offsets.heading_delay"] = 5.0
        config["offsets.compass"] = 3.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
            config=config,
        )

        heading_ship = bearing.heading_ship()
        heading_earth = bearing.heading_earth()

        # All headings should be valid
        assert np.all(heading_ship >= 0)
        assert np.all(heading_ship < 360)
        assert np.all(heading_earth >= 0)
        assert np.all(heading_earth < 360)

    def test_caching(self, single_polar_file: Path):
        """Test that results are cached."""
        pf = PolarFile(single_polar_file)
        frame = pf[0]

        theta_obj = Theta(frame)
        range_obj = Range(frame)
        ship_heading = frame.metadata.heading or 0.0

        bearing = Bearing(
            theta=theta_obj.theta,
            ship_heading=ship_heading,
            ground_range=range_obj.ground_range,
        )

        # First call computes
        heading1 = bearing.heading_ship()
        # Second call should return cached
        heading2 = bearing.heading_ship()

        assert heading1 is heading2  # Same object (cached)


class TestBackwardCompatibility:
    """Test backward compatibility with Theta import from bearing."""

    def test_theta_import_from_bearing(self):
        """Test that Theta can be imported from bearing module."""
        from wamos_tpw.bearing import Theta as BearingTheta
        from wamos_tpw.theta import Theta as ThetaTheta

        # Should be the same class
        assert BearingTheta is ThetaTheta
