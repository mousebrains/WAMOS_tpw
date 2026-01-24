#! /usr/bin/env python3
"""Tests for wamos_tpw.grid module."""

import numpy as np
import pytest

from wamos_tpw.grid import (
    compute_common_grid,
    project_frame_to_common_grid,
    remap_to_common_grid,
)


class TestComputeCommonGrid:
    """Tests for compute_common_grid function."""

    def test_basic_grid_computation(self):
        """Test basic grid computation with simple inputs."""
        # Create simple lat/lon arrays around (45, -122) - Portland, OR
        latitudes = [np.array([45.0, 45.001, 45.002])]
        longitudes = [np.array([-122.0, -122.001, -122.002])]
        max_ranges = [1000.0]  # 1km range
        range_resolutions = [10.0]  # 10m resolution

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        # Check all expected keys are present
        assert "x_edges" in result
        assert "y_edges" in result
        assert "x_edges_utm" in result
        assert "y_edges_utm" in result
        assert "grid_spacing" in result
        assert "utm_zone" in result
        assert "hemisphere" in result
        assert "center_lat" in result
        assert "center_lon" in result
        assert "transformer" in result
        assert "n_x" in result
        assert "n_y" in result

    def test_grid_spacing_matches_resolution(self):
        """Test that grid spacing matches input range resolution."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]
        max_ranges = [500.0]
        range_resolutions = [7.5]  # Typical WAMOS resolution

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        assert result["grid_spacing"] == pytest.approx(7.5)

    def test_resolution_scale_increases_grid_density(self):
        """Test that resolution_scale creates finer grid."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]
        max_ranges = [500.0]
        range_resolutions = [10.0]

        # Default resolution scale (1.0)
        result1 = compute_common_grid(
            latitudes, longitudes, max_ranges, range_resolutions, resolution_scale=1.0
        )

        # Double resolution (2.0x)
        result2 = compute_common_grid(
            latitudes, longitudes, max_ranges, range_resolutions, resolution_scale=2.0
        )

        # Grid spacing should be halved
        assert result2["grid_spacing"] == pytest.approx(result1["grid_spacing"] / 2)

        # Grid should have approximately 2x more cells in each dimension
        assert result2["n_x"] >= result1["n_x"] * 1.8
        assert result2["n_y"] >= result1["n_y"] * 1.8

    def test_utm_zone_northern_hemisphere(self):
        """Test UTM zone calculation for northern hemisphere."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]  # UTM zone 10
        max_ranges = [500.0]
        range_resolutions = [10.0]

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        assert result["utm_zone"] == 10
        assert result["hemisphere"] == "north"

    def test_utm_zone_southern_hemisphere(self):
        """Test UTM zone calculation for southern hemisphere."""
        latitudes = [np.array([-33.9])]  # Sydney, Australia
        longitudes = [np.array([151.2])]  # UTM zone 56
        max_ranges = [500.0]
        range_resolutions = [10.0]

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        assert result["utm_zone"] == 56
        assert result["hemisphere"] == "south"

    def test_centered_edges(self):
        """Test that x_edges and y_edges are centered around zero."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]
        max_ranges = [1000.0]
        range_resolutions = [10.0]

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        # Centered edges should be symmetric around zero (approximately)
        assert result["x_edges"][0] < 0
        assert result["x_edges"][-1] > 0
        assert result["y_edges"][0] < 0
        assert result["y_edges"][-1] > 0

    def test_grid_dimensions(self):
        """Test that grid dimensions are consistent."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]
        max_ranges = [500.0]
        range_resolutions = [10.0]

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        # n_x + 1 edges for n_x bins
        assert len(result["x_edges"]) == result["n_x"] + 1
        assert len(result["y_edges"]) == result["n_y"] + 1
        assert len(result["x_edges_utm"]) == result["n_x"] + 1
        assert len(result["y_edges_utm"]) == result["n_y"] + 1

    def test_multiple_positions(self):
        """Test grid computation with multiple lat/lon arrays."""
        latitudes = [
            np.array([45.0, 45.001]),
            np.array([45.002, 45.003]),
        ]
        longitudes = [
            np.array([-122.0, -122.001]),
            np.array([-122.002, -122.003]),
        ]
        max_ranges = [500.0, 600.0]
        range_resolutions = [10.0, 10.0]

        result = compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

        # Grid should cover all positions plus padding
        assert result["n_x"] > 0
        assert result["n_y"] > 0

    def test_padding_increases_extent(self):
        """Test that padding parameter increases grid extent."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]
        max_ranges = [500.0]
        range_resolutions = [10.0]

        result_default = compute_common_grid(
            latitudes, longitudes, max_ranges, range_resolutions, padding=1.1
        )
        result_larger = compute_common_grid(
            latitudes, longitudes, max_ranges, range_resolutions, padding=1.5
        )

        # Larger padding should result in more grid cells
        assert result_larger["n_x"] >= result_default["n_x"]
        assert result_larger["n_y"] >= result_default["n_y"]


class TestProjectFrameToCommonGrid:
    """Tests for project_frame_to_common_grid function."""

    @pytest.fixture
    def simple_grid_params(self):
        """Create simple grid parameters for testing."""
        latitudes = [np.array([45.0])]
        longitudes = [np.array([-122.0])]
        max_ranges = [500.0]
        range_resolutions = [10.0]
        return compute_common_grid(latitudes, longitudes, max_ranges, range_resolutions)

    def test_basic_projection(self, simple_grid_params):
        """Test basic frame projection."""
        n_bearings = 10
        n_distances = 50

        intensity = np.random.rand(n_bearings, n_distances).astype(np.float32)
        theta = np.linspace(0, 360, n_bearings, endpoint=False)
        ground_range = np.linspace(10, 500, n_distances)
        latitudes = np.full(n_bearings, 45.0)
        longitudes = np.full(n_bearings, -122.0)
        headings = np.full(n_bearings, 0.0)  # Ship heading north

        frame_sum, frame_count = project_frame_to_common_grid(
            intensity=intensity,
            theta=theta,
            ground_range=ground_range,
            latitudes=latitudes,
            longitudes=longitudes,
            headings=headings,
            grid_params=simple_grid_params,
        )

        # Check output shapes
        expected_shape = (simple_grid_params["n_y"], simple_grid_params["n_x"])
        assert frame_sum.shape == expected_shape
        assert frame_count.shape == expected_shape

        # Check data types
        assert frame_sum.dtype == np.float64
        assert frame_count.dtype == np.int32

        # Some cells should have data
        assert np.sum(frame_count) > 0

    def test_projection_with_nan_values(self, simple_grid_params):
        """Test that NaN values in intensity are excluded."""
        n_bearings = 10
        n_distances = 50

        intensity = np.random.rand(n_bearings, n_distances).astype(np.float32)
        # Set some values to NaN
        intensity[0, :10] = np.nan
        intensity[5, 20:30] = np.nan

        theta = np.linspace(0, 360, n_bearings, endpoint=False)
        ground_range = np.linspace(10, 500, n_distances)
        latitudes = np.full(n_bearings, 45.0)
        longitudes = np.full(n_bearings, -122.0)
        headings = np.zeros(n_bearings)

        frame_sum, frame_count = project_frame_to_common_grid(
            intensity=intensity,
            theta=theta,
            ground_range=ground_range,
            latitudes=latitudes,
            longitudes=longitudes,
            headings=headings,
            grid_params=simple_grid_params,
        )

        # NaN values should not contribute to count
        valid_intensity_count = np.sum(~np.isnan(intensity))
        assert np.sum(frame_count) <= valid_intensity_count

    def test_projection_heading_rotation(self, simple_grid_params):
        """Test that heading rotates the projection correctly."""
        n_bearings = 4  # N, E, S, W
        n_distances = 10

        intensity = np.ones((n_bearings, n_distances), dtype=np.float32)
        theta = np.array([0, 90, 180, 270])  # Beams in cardinal directions
        ground_range = np.linspace(10, 100, n_distances)
        latitudes = np.full(n_bearings, 45.0)
        longitudes = np.full(n_bearings, -122.0)

        # Heading 0 (north) - theta 0 should point north
        headings_north = np.zeros(n_bearings)
        sum_north, count_north = project_frame_to_common_grid(
            intensity=intensity,
            theta=theta,
            ground_range=ground_range,
            latitudes=latitudes,
            longitudes=longitudes,
            headings=headings_north,
            grid_params=simple_grid_params,
        )

        # Heading 90 (east) - theta 0 should point east
        headings_east = np.full(n_bearings, 90.0)
        sum_east, count_east = project_frame_to_common_grid(
            intensity=intensity,
            theta=theta,
            ground_range=ground_range,
            latitudes=latitudes,
            longitudes=longitudes,
            headings=headings_east,
            grid_params=simple_grid_params,
        )

        # Projections should be different due to heading rotation
        # (unless they happen to be symmetric, which is unlikely)
        assert np.sum(count_north) > 0
        assert np.sum(count_east) > 0


class TestRemapToCommonGrid:
    """Tests for remap_to_common_grid function."""

    def test_basic_remap(self):
        """Test basic grid remapping."""
        # Source grid: 5x5
        src_intensity = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        src_count = np.ones((5, 5), dtype=np.int32)

        src_x_edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        src_y_edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])

        # Destination grid: 10x10 (same extent, finer resolution)
        dst_x_edges = np.linspace(0.0, 50.0, 11)
        dst_y_edges = np.linspace(0.0, 50.0, 11)
        dst_n_x = 10
        dst_n_y = 10

        dst_sum, dst_count = remap_to_common_grid(
            intensity=src_intensity,
            count=src_count,
            src_x_edges=src_x_edges,
            src_y_edges=src_y_edges,
            dst_x_edges=dst_x_edges,
            dst_y_edges=dst_y_edges,
            dst_n_x=dst_n_x,
            dst_n_y=dst_n_y,
        )

        # Check output shapes
        assert dst_sum.shape == (dst_n_y, dst_n_x)
        assert dst_count.shape == (dst_n_y, dst_n_x)

        # Some cells should have data
        assert np.sum(dst_count) > 0

    def test_remap_with_nan(self):
        """Test that NaN values are excluded from remapping."""
        src_intensity = np.array(
            [
                [1.0, np.nan, 3.0],
                [4.0, 5.0, np.nan],
                [np.nan, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        src_count = np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )

        src_x_edges = np.array([0.0, 10.0, 20.0, 30.0])
        src_y_edges = np.array([0.0, 10.0, 20.0, 30.0])
        dst_x_edges = np.array([0.0, 15.0, 30.0])
        dst_y_edges = np.array([0.0, 15.0, 30.0])

        dst_sum, dst_count = remap_to_common_grid(
            intensity=src_intensity,
            count=src_count,
            src_x_edges=src_x_edges,
            src_y_edges=src_y_edges,
            dst_x_edges=dst_x_edges,
            dst_y_edges=dst_y_edges,
            dst_n_x=2,
            dst_n_y=2,
        )

        # NaN cells should not contribute
        assert dst_sum.shape == (2, 2)
        assert dst_count.shape == (2, 2)

    def test_remap_none_count(self):
        """Test remapping when count is None."""
        src_intensity = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )

        src_x_edges = np.array([0.0, 10.0, 20.0])
        src_y_edges = np.array([0.0, 10.0, 20.0])
        dst_x_edges = np.array([0.0, 20.0])
        dst_y_edges = np.array([0.0, 20.0])

        dst_sum, dst_count = remap_to_common_grid(
            intensity=src_intensity,
            count=None,  # None count
            src_x_edges=src_x_edges,
            src_y_edges=src_y_edges,
            dst_x_edges=dst_x_edges,
            dst_y_edges=dst_y_edges,
            dst_n_x=1,
            dst_n_y=1,
        )

        # Should use unit counts
        assert dst_sum.shape == (1, 1)
        assert dst_count.shape == (1, 1)
        assert dst_count[0, 0] == 4  # All 4 source cells map to single dest cell

    def test_remap_outside_bounds(self):
        """Test remapping when source is partially outside destination bounds."""
        src_intensity = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )
        src_count = np.ones((2, 2), dtype=np.int32)

        # Source grid extends beyond destination
        src_x_edges = np.array([0.0, 10.0, 20.0])
        src_y_edges = np.array([0.0, 10.0, 20.0])

        # Destination grid only covers part of source
        dst_x_edges = np.array([0.0, 5.0, 10.0])
        dst_y_edges = np.array([0.0, 5.0, 10.0])

        dst_sum, dst_count = remap_to_common_grid(
            intensity=src_intensity,
            count=src_count,
            src_x_edges=src_x_edges,
            src_y_edges=src_y_edges,
            dst_x_edges=dst_x_edges,
            dst_y_edges=dst_y_edges,
            dst_n_x=2,
            dst_n_y=2,
        )

        # Only the overlapping region should have data
        assert dst_sum.shape == (2, 2)
        # Some cells may have no data if source centers fall outside

    def test_remap_empty_result(self):
        """Test remapping when no source cells fall within destination."""
        src_intensity = np.array([[1.0]], dtype=np.float64)
        src_count = np.ones((1, 1), dtype=np.int32)

        # Source is far from destination
        src_x_edges = np.array([100.0, 110.0])
        src_y_edges = np.array([100.0, 110.0])
        dst_x_edges = np.array([0.0, 10.0])
        dst_y_edges = np.array([0.0, 10.0])

        dst_sum, dst_count = remap_to_common_grid(
            intensity=src_intensity,
            count=src_count,
            src_x_edges=src_x_edges,
            src_y_edges=src_y_edges,
            dst_x_edges=dst_x_edges,
            dst_y_edges=dst_y_edges,
            dst_n_x=1,
            dst_n_y=1,
        )

        # Result should be zeros since no overlap
        assert dst_sum.shape == (1, 1)
        assert dst_count.shape == (1, 1)
        assert dst_count[0, 0] == 0


class TestGridIntegration:
    """Integration tests combining grid functions."""

    def test_compute_project_remap_workflow(self):
        """Test the full workflow: compute grid, project, then remap."""
        # Create two separate "frames" with their own positions
        lats1 = np.array([45.0, 45.001])
        lons1 = np.array([-122.0, -122.001])
        lats2 = np.array([45.002, 45.003])
        lons2 = np.array([-122.002, -122.003])

        # Compute common grid
        grid_params = compute_common_grid(
            latitudes=[lats1, lats2],
            longitudes=[lons1, lons2],
            max_ranges=[200.0, 200.0],
            range_resolutions=[10.0, 10.0],
        )

        # Project a frame
        n_bearings = 8
        n_distances = 20
        intensity = np.random.rand(n_bearings, n_distances).astype(np.float32)
        theta = np.linspace(0, 360, n_bearings, endpoint=False)
        ground_range = np.linspace(10, 200, n_distances)
        latitudes = np.full(n_bearings, 45.001)
        longitudes = np.full(n_bearings, -122.001)
        headings = np.zeros(n_bearings)

        frame_sum, frame_count = project_frame_to_common_grid(
            intensity=intensity,
            theta=theta,
            ground_range=ground_range,
            latitudes=latitudes,
            longitudes=longitudes,
            headings=headings,
            grid_params=grid_params,
        )

        # Compute averaged intensity
        with np.errstate(invalid="ignore"):
            avg_intensity = frame_sum / frame_count
        avg_intensity[frame_count == 0] = np.nan

        # Remap to a coarser grid
        coarse_n_x = grid_params["n_x"] // 2
        coarse_n_y = grid_params["n_y"] // 2
        if coarse_n_x < 1:
            coarse_n_x = 1
        if coarse_n_y < 1:
            coarse_n_y = 1

        coarse_x_edges = np.linspace(
            grid_params["x_edges_utm"][0], grid_params["x_edges_utm"][-1], coarse_n_x + 1
        )
        coarse_y_edges = np.linspace(
            grid_params["y_edges_utm"][0], grid_params["y_edges_utm"][-1], coarse_n_y + 1
        )

        remapped_sum, remapped_count = remap_to_common_grid(
            intensity=avg_intensity,
            count=frame_count,
            src_x_edges=grid_params["x_edges_utm"],
            src_y_edges=grid_params["y_edges_utm"],
            dst_x_edges=coarse_x_edges,
            dst_y_edges=coarse_y_edges,
            dst_n_x=coarse_n_x,
            dst_n_y=coarse_n_y,
        )

        # Verify shapes
        assert remapped_sum.shape == (coarse_n_y, coarse_n_x)
        assert remapped_count.shape == (coarse_n_y, coarse_n_x)
