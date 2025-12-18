"""Tests for combine_netcdf module."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


# Skip all tests if netCDF4 is not installed
pytest.importorskip("netCDF4")


class TestNetCDFWriter:
    """Tests for NetCDFWriter class."""

    def test_netcdf_writer_init(self, tmp_path: Path):
        """Test NetCDFWriter initialization."""
        from wamos_tpw.combine_netcdf import NetCDFWriter

        filepath = str(tmp_path / "test.nc")
        writer = NetCDFWriter(filepath, n_along=100, n_cross=100)

        assert writer._filepath == filepath
        assert writer._n_along == 100
        assert writer._n_cross == 100
        assert writer._nc is None
        assert writer._group_idx == 0

    def test_netcdf_writer_context_manager(self, tmp_path: Path):
        """Test NetCDFWriter as context manager."""
        from wamos_tpw.combine_netcdf import NetCDFWriter

        filepath = str(tmp_path / "test.nc")

        with NetCDFWriter(filepath) as writer:
            assert writer._nc is None  # Not created yet

        # After exiting, should be closed
        assert writer._nc is None

    def test_netcdf_writer_append_group(self, single_polar_file: Path, tmp_path: Path):
        """Test appending a group to NetCDF file."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        # Load and prepare data
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        # Write to NetCDF
        filepath = str(tmp_path / "test.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        # Verify file was created
        assert Path(filepath).exists()
        assert Path(filepath).stat().st_size > 0

        # Read back and verify
        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            assert "intensity" in ds.variables
            assert "lat" in ds.variables
            assert "lon" in ds.variables
            assert "ship_lat" in ds.variables
            assert "ship_lon" in ds.variables
            assert ds.dimensions["group"].size == 1

    def test_netcdf_writer_multiple_groups(self, single_polar_file: Path, tmp_path: Path):
        """Test appending multiple groups."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        filepath = str(tmp_path / "test_multi.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            # Append two groups with different periods
            group1 = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)
            group2 = grid_group("2022-04-05 14:10", combine, n_along=50, n_cross=50)

            writer.append_group(group1)
            writer.append_group(group2)

        # Verify both groups were written
        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            assert ds.dimensions["group"].size == 2
            assert ds.variables["intensity"].shape[0] == 2

    def test_netcdf_writer_metadata(self, single_polar_file: Path, tmp_path: Path):
        """Test that metadata is correctly written."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        filepath = str(tmp_path / "test_meta.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            # Check scalar metadata
            assert ds.variables["ref_lat"][0] == pytest.approx(group_data["ref_lat"], rel=1e-5)
            assert ds.variables["ref_lon"][0] == pytest.approx(group_data["ref_lon"], rel=1e-5)
            assert ds.variables["n_frames"][0] == group_data["n_frames"]

            # Check travel data
            assert ds.variables["travel_distance_m"][0] == pytest.approx(
                group_data["travel"]["total_m"], rel=1e-3
            )
            assert ds.variables["duration_s"][0] == pytest.approx(
                group_data["travel"]["duration_s"], rel=1e-3
            )

    def test_netcdf_writer_global_attributes(self, single_polar_file: Path, tmp_path: Path):
        """Test global attributes."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        filepath = str(tmp_path / "test_attrs.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            assert ds.Conventions == "CF-1.8"
            assert ds.title == "WAMOS Radar Combined Gridded Data"
            assert ds.grid_n_along == 50
            assert ds.grid_n_cross == 50

    def test_netcdf_writer_close_twice(self, tmp_path: Path):
        """Test that closing twice doesn't raise."""
        from wamos_tpw.combine_netcdf import NetCDFWriter

        filepath = str(tmp_path / "test_close.nc")
        writer = NetCDFWriter(filepath)

        # Close without writing anything
        writer.close()
        writer.close()  # Should not raise

    def test_netcdf_writer_ship_track_ragged(self, single_polar_file: Path, tmp_path: Path):
        """Test ragged array ship track storage."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        filepath = str(tmp_path / "test_ragged.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            group1 = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)
            group2 = grid_group("2022-04-05 14:10", combine, n_along=50, n_cross=50)

            writer.append_group(group1)
            writer.append_group(group2)

        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            # Check ragged array indices
            track_start = ds.variables["track_start"][:]
            track_count = ds.variables["track_count"][:]

            assert len(track_start) == 2
            assert len(track_count) == 2
            assert track_start[0] == 0
            assert track_start[1] == track_count[0]  # Second starts after first

            # Verify we can read back ship tracks
            for i in range(2):
                start = track_start[i]
                count = track_count[i]
                ship_lat = ds.variables["ship_lat"][start:start+count]
                ship_lon = ds.variables["ship_lon"][start:start+count]
                assert len(ship_lat) == count
                assert len(ship_lon) == count


class TestNetCDFWriterVariableAttributes:
    """Tests for variable attributes and units."""

    def test_variable_units(self, single_polar_file: Path, tmp_path: Path):
        """Test that variables have correct units."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        filepath = str(tmp_path / "test_units.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            # Check units
            assert ds.variables["ref_lat"].units == "degrees_north"
            assert ds.variables["ref_lon"].units == "degrees_east"
            assert ds.variables["lat"].units == "degrees_north"
            assert ds.variables["lon"].units == "degrees_east"
            assert ds.variables["travel_distance_m"].units == "meters"
            assert ds.variables["duration_s"].units == "seconds"
            assert ds.variables["avg_speed_m_s"].units == "m/s"
            assert ds.variables["start_time"].units == "seconds since 1970-01-01"

    def test_variable_long_names(self, single_polar_file: Path, tmp_path: Path):
        """Test that variables have long_name attributes."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        filepath = str(tmp_path / "test_long_names.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            # All variables should have long_name
            for var_name in ["intensity", "lat", "lon", "ref_lat", "ref_lon",
                            "n_frames", "n_pixels", "travel_distance_m"]:
                assert hasattr(ds.variables[var_name], "long_name")
                assert len(ds.variables[var_name].long_name) > 0

    def test_compression(self, single_polar_file: Path, tmp_path: Path):
        """Test that large variables are compressed."""
        from wamos_tpw.combine_netcdf import NetCDFWriter
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)

        filepath = str(tmp_path / "test_compression.nc")
        with NetCDFWriter(filepath, n_along=50, n_cross=50) as writer:
            writer.append_group(group_data)

        import netCDF4 as nc
        with nc.Dataset(filepath, "r") as ds:
            # Check that intensity is compressed
            intensity_var = ds.variables["intensity"]
            filters = intensity_var.filters()
            assert filters["zlib"] is True
            assert filters["complevel"] > 0
