#! /usr/bin/env python3
#
# NetCDF writer for WAMOS combined data
# Incrementally appends groups to NetCDF file
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import numpy as np


class NetCDFWriter:
    """
    Write gridded combine data to NetCDF file incrementally.

    Appends each group's data to the file to minimize memory usage.
    Uses netCDF4 directly for efficient incremental writes.
    """

    def __init__(self, filepath: str, n_along: int = 1200, n_cross: int = 1600):
        """
        Initialize the NetCDF writer.

        Args:
            filepath: Path to output NetCDF file
            n_along: Grid bins along ship track
            n_cross: Grid bins cross track
        """
        self._filepath = filepath
        self._n_along = n_along
        self._n_cross = n_cross
        self._nc = None
        self._group_idx = 0
        self._max_track_len = 0

    def _create_file(self, group_data: dict) -> None:
        """Create the NetCDF file with dimensions and variables."""
        import netCDF4 as nc

        # Determine grid shape from first group
        n_y, n_x = group_data["gridded"].shape
        # lat/lon grids have +1 for cell edges
        n_y_edge = group_data["lat_grid"].shape[0]
        n_x_edge = group_data["lat_grid"].shape[1]

        self._nc = nc.Dataset(self._filepath, "w", format="NETCDF4")

        # Create dimensions
        self._nc.createDimension("group", None)  # Unlimited
        self._nc.createDimension("y", n_y)
        self._nc.createDimension("x", n_x)
        self._nc.createDimension("y_edge", n_y_edge)
        self._nc.createDimension("x_edge", n_x_edge)
        self._nc.createDimension("track_point", None)  # Unlimited for ragged arrays
        self._nc.createDimension("str_len", 64)  # For period strings

        # Create variables
        # Group metadata
        self._nc.createVariable("period", "S1", ("group", "str_len"))
        self._nc.variables["period"].long_name = "Time period identifier"

        self._nc.createVariable("start_time", "f8", ("group",))
        self._nc.variables["start_time"].long_name = "Group start time"
        self._nc.variables["start_time"].units = "seconds since 1970-01-01"

        self._nc.createVariable("ref_lat", "f8", ("group",))
        self._nc.variables["ref_lat"].long_name = "Reference latitude"
        self._nc.variables["ref_lat"].units = "degrees_north"

        self._nc.createVariable("ref_lon", "f8", ("group",))
        self._nc.variables["ref_lon"].long_name = "Reference longitude"
        self._nc.variables["ref_lon"].units = "degrees_east"

        self._nc.createVariable("n_frames", "i4", ("group",))
        self._nc.variables["n_frames"].long_name = "Number of frames in group"

        self._nc.createVariable("n_pixels", "i8", ("group",))
        self._nc.variables["n_pixels"].long_name = "Total pixels in group"

        self._nc.createVariable("travel_distance_m", "f8", ("group",))
        self._nc.variables["travel_distance_m"].long_name = "Ship travel distance"
        self._nc.variables["travel_distance_m"].units = "meters"

        self._nc.createVariable("duration_s", "f8", ("group",))
        self._nc.variables["duration_s"].long_name = "Group duration"
        self._nc.variables["duration_s"].units = "seconds"

        self._nc.createVariable("avg_speed_m_s", "f8", ("group",))
        self._nc.variables["avg_speed_m_s"].long_name = "Average ship speed"
        self._nc.variables["avg_speed_m_s"].units = "m/s"

        # Gridded intensity data
        self._nc.createVariable(
            "intensity", "f4", ("group", "y", "x"), fill_value=np.nan, zlib=True, complevel=4
        )
        self._nc.variables["intensity"].long_name = "Gridded radar intensity"
        self._nc.variables["intensity"].coordinates = "lat lon"

        # Lat/lon grids (cell edges, vary per group due to rotation)
        self._nc.createVariable("lat", "f8", ("group", "y_edge", "x_edge"), zlib=True, complevel=4)
        self._nc.variables["lat"].long_name = "Latitude of grid cell edges"
        self._nc.variables["lat"].units = "degrees_north"

        self._nc.createVariable("lon", "f8", ("group", "y_edge", "x_edge"), zlib=True, complevel=4)
        self._nc.variables["lon"].long_name = "Longitude of grid cell edges"
        self._nc.variables["lon"].units = "degrees_east"

        # Ship track (ragged array - use track_start and track_count)
        self._nc.createVariable("track_start", "i4", ("group",))
        self._nc.variables["track_start"].long_name = "Start index in ship track arrays"

        self._nc.createVariable("track_count", "i4", ("group",))
        self._nc.variables["track_count"].long_name = "Number of points in ship track"

        self._nc.createVariable("ship_lat", "f8", ("track_point",), zlib=True)
        self._nc.variables["ship_lat"].long_name = "Ship track latitude"
        self._nc.variables["ship_lat"].units = "degrees_north"

        self._nc.createVariable("ship_lon", "f8", ("track_point",), zlib=True)
        self._nc.variables["ship_lon"].long_name = "Ship track longitude"
        self._nc.variables["ship_lon"].units = "degrees_east"

        # Global attributes
        self._nc.Conventions = "CF-1.8"
        self._nc.title = "WAMOS Radar Combined Gridded Data"
        self._nc.institution = "wamos_tpw"
        self._nc.history = f"Created {np.datetime64('now')}"
        self._nc.grid_n_along = self._n_along
        self._nc.grid_n_cross = self._n_cross

    def append_group(self, group_data: dict) -> None:
        """
        Append a group's data to the NetCDF file.

        Args:
            group_data: Dictionary from grid_group()
        """
        import netCDF4 as nc

        # Create file on first group
        if self._nc is None:
            self._create_file(group_data)

        idx = self._group_idx

        # Write period string
        period_str = group_data["period"][:64].ljust(64)
        self._nc.variables["period"][idx] = nc.stringtochar(np.array([period_str], dtype="S64"))

        # Parse period to get start time (approximate)
        try:
            import pandas as pd

            # Try to parse period string as timestamp
            start_time = pd.Timestamp(group_data["period"]).timestamp()
        except Exception:
            start_time = 0.0
        self._nc.variables["start_time"][idx] = start_time

        # Write scalar metadata
        self._nc.variables["ref_lat"][idx] = group_data["ref_lat"]
        self._nc.variables["ref_lon"][idx] = group_data["ref_lon"]
        self._nc.variables["n_frames"][idx] = group_data["n_frames"]
        self._nc.variables["n_pixels"][idx] = group_data["n_pixels"]
        self._nc.variables["travel_distance_m"][idx] = group_data["travel"]["total_m"]
        self._nc.variables["duration_s"][idx] = group_data["travel"]["duration_s"]
        self._nc.variables["avg_speed_m_s"][idx] = group_data["travel"]["speed_m_s"]

        # Write gridded intensity
        self._nc.variables["intensity"][idx, :, :] = group_data["gridded"]

        # Write lat/lon grids
        self._nc.variables["lat"][idx, :, :] = group_data["lat_grid"]
        self._nc.variables["lon"][idx, :, :] = group_data["lon_grid"]

        # Write ship track (ragged array)
        track_len = len(group_data["ship_lat"])
        self._nc.variables["track_start"][idx] = self._max_track_len
        self._nc.variables["track_count"][idx] = track_len

        start = self._max_track_len
        end = start + track_len
        self._nc.variables["ship_lat"][start:end] = group_data["ship_lat"]
        self._nc.variables["ship_lon"][start:end] = group_data["ship_lon"]
        self._max_track_len = end

        # Flush to disk
        self._nc.sync()

        self._group_idx += 1

    def close(self) -> None:
        """Close the NetCDF file."""
        if self._nc is not None:
            self._nc.close()
            self._nc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
