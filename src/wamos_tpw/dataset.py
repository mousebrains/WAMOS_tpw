#! /usr/bin/env python3
#
# WamosDataset - High-level API for WAMOS data processing
#
# Dec-2025, Pat Welch, pat@mousebrains.com

"""
High-level API for WAMOS radar data processing.

Provides a unified interface for loading, processing, and exporting
WAMOS polar radar data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Any

import numpy as np
from tqdm import tqdm

from wamos_tpw import __version__
from wamos_tpw.config import Config
from wamos_tpw.processed import ProcessedFrames
from wamos_tpw.frame import Frame
from wamos_tpw.bearing import Theta, Bearing
from wamos_tpw.combine import Combine


logger = logging.getLogger(__name__)


class WamosDataset:
    """
    High-level API for WAMOS radar data processing.

    Provides a unified interface for:
    - Loading polar files from a time range
    - Processing (deramp, destreak, normalize)
    - Coordinate transformations (polar, ship, earth)
    - Export to NetCDF, Zarr, or other formats
    - Animation generation

    Example:
        >>> ds = WamosDataset('/path/to/POLAR', '2022-04-05', '2022-04-06')
        >>> ds.load()
        >>> ds.process()
        >>> ds.to_netcdf('output.nc')
        >>> ds.animate('output.mp4')
    """

    def __init__(
        self,
        polar_path: str | Path,
        stime: str,
        etime: str,
        config: Config | str | Path | None = None,
        radar_height: float | None = None,
        groupby: str = "h",
        workers: int | None = None,
    ) -> None:
        """
        Initialize WamosDataset.

        Args:
            polar_path: Path to POLAR directory with YYYY/MM/DD/HH structure
            stime: Start time (various formats supported)
            etime: End time (various formats supported)
            config: Config instance or path to YAML config file
            radar_height: Radar height above water in meters
            groupby: Time grouping frequency (e.g., 'h', '30m', 'D')
            workers: Number of parallel workers (None = auto)
        """
        self._polar_path = Path(polar_path)
        self._stime = stime
        self._etime = etime
        self._groupby = groupby
        self._workers = workers
        self._radar_height = radar_height

        # Load config
        if config is None:
            self._config = Config()
        elif isinstance(config, (str, Path)):
            self._config = Config(str(config))
        else:
            self._config = config

        # Data storage
        self._frames: list[Frame] = []
        self._processed: dict[np.datetime64, list[np.ndarray]] = {}
        self._theta: Theta | None = None
        self._bearing: Bearing | None = None
        self._combine: Combine | None = None

        # State flags
        self._loaded = False
        self._processed_flag = False

        # Initialize ProcessedFrames
        self._pframes = ProcessedFrames(
            stime=stime,
            etime=etime,
            polar_path=str(polar_path),
            groupby=groupby,
            workers=workers,
            config=self._config,
            radar_height=radar_height,
        )

        logger.info(f"WamosDataset initialized: {len(self._pframes)} files found")

    @property
    def n_files(self) -> int:
        """Return number of files found."""
        return len(self._pframes)

    @property
    def n_frames(self) -> int:
        """Return number of loaded frames."""
        return len(self._frames)

    @property
    def frames(self) -> list[Frame]:
        """Return loaded frames."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._frames

    @property
    def config(self) -> Config:
        """Return configuration."""
        return self._config

    @property
    def is_loaded(self) -> bool:
        """Return True if data is loaded."""
        return self._loaded

    @property
    def is_processed(self) -> bool:
        """Return True if data is processed."""
        return self._processed_flag

    def load(self, max_frames: int | None = None, show_progress: bool = True) -> "WamosDataset":
        """
        Load frames from polar files.

        Args:
            max_frames: Maximum number of frames to load (None = all)
            show_progress: Show progress bar

        Returns:
            self for method chaining
        """
        self._frames = []
        frame_count = 0

        groups = list(self._pframes.itergroups())
        group_iter = tqdm(groups, desc="Loading groups", disable=not show_progress)

        for period, frames in group_iter:
            for frame in frames:
                if max_frames is not None and frame_count >= max_frames:
                    break
                self._frames.append(frame)
                frame_count += 1

            if max_frames is not None and frame_count >= max_frames:
                break

        self._loaded = True
        logger.info(f"Loaded {len(self._frames)} frames")
        return self

    def process(
        self,
        show_progress: bool = True,
        shadow_diagnostics: bool = False,
        deramp_diagnostics: bool = False,
        destreak_diagnostics: bool = False,
    ) -> "WamosDataset":
        """
        Process loaded frames (deramp, destreak, normalize).

        The shadow edges detected during theta refinement are propagated to the
        deramp phase for accurate shadow region masking.

        Args:
            show_progress: Show progress bars
            shadow_diagnostics: Show shadow detection plots
            deramp_diagnostics: Show deramp diagnostic plots
            destreak_diagnostics: Show destreak diagnostic plots

        Returns:
            self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        # Refine theta - returns Theta with detected shadow edges
        logger.info("Refining theta from shadow region...")
        theta = self._pframes.refine_theta(self._frames, shadow_diagnostics=shadow_diagnostics)

        # Extract detected shadow edges (if available)
        shadow_start = theta.shadow_left_mean
        shadow_end = theta.shadow_right_mean

        if shadow_start is not None and shadow_end is not None:
            logger.info(f"Using detected shadow region: {shadow_start:.1f}° - {shadow_end:.1f}°")
        else:
            logger.info("Using config-based shadow region (detection failed)")

        # Deramp frames
        logger.info("Deramping frames...")
        self._pframes.deramp_frames(
            self._frames,
            diagnostics=deramp_diagnostics,
            show_progress=show_progress,
        )

        # Destreak
        logger.info("Destreaking frames...")
        corrected = self._pframes.destreak_frames(
            self._frames, diagnostics=destreak_diagnostics, show_progress=show_progress
        )

        # Normalize and store on frames
        logger.info("Normalizing frames...")
        normalized = self._pframes.normalize_frames(corrected)
        for frame, norm in zip(self._frames, normalized):
            frame.corrected_intensity = norm

        # Store theta and build coordinate calculators (reuse the already-computed theta)
        self._theta = theta
        self._bearing = Bearing(self._theta, radar_height=self._radar_height)
        self._combine = Combine(self._frames, config=self._config, radar_height=self._radar_height)

        self._processed_flag = True
        logger.info("Processing complete")
        return self

    def to_netcdf(
        self, path: str | Path, include_raw: bool = False, compression_level: int = 4
    ) -> Path:
        """
        Export dataset to CF-1.11 compliant NetCDF format with compression.

        Creates a NetCDF-4 file following Climate and Forecast (CF) conventions
        version 1.11. All variables use zlib compression for efficient storage.

        Args:
            path: Output file path (.nc)
            include_raw: Include raw intensity data (increases file size)
            compression_level: zlib compression level 0-9 (default: 4)

        Returns:
            Path to created file

        Raises:
            ImportError: If xarray/netCDF4 not installed
            RuntimeError: If data not processed
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray required for NetCDF export. Install with: pip install xarray netCDF4"
            )

        if not self._processed_flag:
            raise RuntimeError("Data not processed. Call process() first.")

        from datetime import datetime, timezone

        path = Path(path)

        # Build coordinate arrays
        n_bearings = self._frames[0].n_bearings
        n_distances = self._frames[0].n_distances

        # Time coordinate (CF requires specific encoding)
        times = np.array([f.timestamp for f in self._frames])

        # Range coordinate (slant range in meters)
        slant_range = self._frames[0].slant_range()

        # Bearing angles in degrees (0-360)
        if self._theta is not None:
            # Use actual bearing from theta calculation for first frame
            bearing_angles = self._theta.bearing_for_frame(0)
        else:
            bearing_angles = np.linspace(0, 360, n_bearings, endpoint=False)

        # Build data arrays
        corrected = np.stack(
            [
                (
                    f.corrected_intensity
                    if f.corrected_intensity is not None
                    else f.intensity.astype(np.float32)
                )
                for f in self._frames
            ]
        ).astype(np.float32)

        # Create time coordinate with CF attributes
        time_coord = xr.DataArray(
            times,
            dims=["time"],
            attrs={
                "standard_name": "time",
                "long_name": "time of radar sweep",
                "axis": "T",
            },
        )

        # Create range coordinate with CF attributes
        range_coord = xr.DataArray(
            slant_range.astype(np.float32),
            dims=["range"],
            attrs={
                "standard_name": "range",
                "long_name": "slant range from radar",
                "units": "m",
                "axis": "X",
                "positive": "up",
            },
        )

        # Create bearing coordinate with CF attributes
        bearing_coord = xr.DataArray(
            bearing_angles.astype(np.float32),
            dims=["bearing"],
            attrs={
                "standard_name": "sensor_azimuth_angle",
                "long_name": "radar beam azimuth angle",
                "units": "degree",
                "axis": "Y",
                "comment": "Azimuth angle measured clockwise from north (0=north, 90=east)",
            },
        )

        # Create dataset with CF-compliant structure
        ds = xr.Dataset(
            coords={
                "time": time_coord,
                "bearing": bearing_coord,
                "range": range_coord,
            }
        )

        # Add intensity variable with CF attributes
        ds["radar_intensity"] = xr.DataArray(
            corrected,
            dims=["time", "bearing", "range"],
            attrs={
                "long_name": "normalized radar backscatter intensity",
                "units": "1",
                "standard_name": "surface_backwards_scattering_coefficient_of_radar_wave",
                "comment": "Processed intensity with deramp and destreak corrections applied, normalized to [0,1]",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "coverage_content_type": "physicalMeasurement",
            },
        )

        # Add raw intensity if requested
        if include_raw:
            raw = np.stack([f.intensity for f in self._frames]).astype(np.uint16)
            ds["raw_intensity"] = xr.DataArray(
                raw,
                dims=["time", "bearing", "range"],
                attrs={
                    "long_name": "raw radar backscatter intensity",
                    "units": "1",
                    "comment": "Original 12-bit intensity values (0-4095)",
                    "valid_min": 0,
                    "valid_max": 4095,
                    "coverage_content_type": "physicalMeasurement",
                },
            )

        # Add platform position variables (CF trajectory/platform conventions)
        meta = self._frames[0].metadata
        if meta.latitude is not None:
            lats = np.array(
                [
                    f.metadata.latitude if f.metadata.latitude is not None else np.nan
                    for f in self._frames
                ],
                dtype=np.float64,
            )
            ds["latitude"] = xr.DataArray(
                lats,
                dims=["time"],
                attrs={
                    "standard_name": "latitude",
                    "long_name": "latitude of radar platform",
                    "units": "degrees_north",
                    "valid_min": -90.0,
                    "valid_max": 90.0,
                    "axis": "Y",
                },
            )

        if meta.longitude is not None:
            lons = np.array(
                [
                    f.metadata.longitude if f.metadata.longitude is not None else np.nan
                    for f in self._frames
                ],
                dtype=np.float64,
            )
            ds["longitude"] = xr.DataArray(
                lons,
                dims=["time"],
                attrs={
                    "standard_name": "longitude",
                    "long_name": "longitude of radar platform",
                    "units": "degrees_east",
                    "valid_min": -180.0,
                    "valid_max": 180.0,
                    "axis": "X",
                },
            )

        if meta.heading is not None:
            headings = np.array(
                [
                    f.metadata.heading if f.metadata.heading is not None else np.nan
                    for f in self._frames
                ],
                dtype=np.float32,
            )
            ds["platform_heading"] = xr.DataArray(
                headings,
                dims=["time"],
                attrs={
                    "standard_name": "platform_azimuth_angle",
                    "long_name": "ship heading",
                    "units": "degree",
                    "comment": "Heading measured clockwise from north (0=north, 90=east)",
                    "valid_min": 0.0,
                    "valid_max": 360.0,
                },
            )

        # Add ship speed if available
        if meta.ship_speed is not None:
            speeds = np.array(
                [
                    f.metadata.ship_speed if f.metadata.ship_speed is not None else np.nan
                    for f in self._frames
                ],
                dtype=np.float32,
            )
            ds["platform_speed"] = xr.DataArray(
                speeds,
                dims=["time"],
                attrs={
                    "standard_name": "platform_speed_wrt_ground",
                    "long_name": "ship speed over ground",
                    "units": "m s-1",
                    "valid_min": 0.0,
                },
            )

        # Add wind variables if available
        if meta.wind_speed is not None:
            wind_speeds = np.array(
                [
                    f.metadata.wind_speed if f.metadata.wind_speed is not None else np.nan
                    for f in self._frames
                ],
                dtype=np.float32,
            )
            ds["wind_speed"] = xr.DataArray(
                wind_speeds,
                dims=["time"],
                attrs={
                    "standard_name": "wind_speed",
                    "long_name": "wind speed",
                    "units": "m s-1",
                    "valid_min": 0.0,
                },
            )

        if meta.wind_direction is not None:
            wind_dirs = np.array(
                [
                    f.metadata.wind_direction if f.metadata.wind_direction is not None else np.nan
                    for f in self._frames
                ],
                dtype=np.float32,
            )
            ds["wind_from_direction"] = xr.DataArray(
                wind_dirs,
                dims=["time"],
                attrs={
                    "standard_name": "wind_from_direction",
                    "long_name": "wind direction",
                    "units": "degree",
                    "comment": "Direction from which wind is blowing, measured clockwise from north",
                    "valid_min": 0.0,
                    "valid_max": 360.0,
                },
            )

        # CF-1.11 compliant global attributes
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ds.attrs = {
            # Required CF attributes
            "Conventions": "CF-1.11",
            "title": "WAMOS Marine Radar Backscatter Data",
            "institution": "Oregon State University",
            "source": "WAMOS marine radar system",
            "history": f"{now}: Created by wamos_tpw v{__version__}",
            "references": "https://github.com/wamos/wamos_tpw",
            # Recommended CF attributes
            "comment": "Processed radar backscatter data with range correction (deramp) and streak removal (destreak)",
            "date_created": now,
            "date_modified": now,
            "processing_level": "L2",
            # CF feature type for moving platform
            "featureType": "trajectory",
            "platform": "ship",
            "platform_vocabulary": "https://vocab.nerc.ac.uk/collection/L06/current/",
            "instrument": "marine_radar",
            "instrument_vocabulary": "https://vocab.nerc.ac.uk/collection/L22/current/",
            # Data provenance
            "source_files": str(self._polar_path),
            "time_coverage_start": str(times[0]),
            "time_coverage_end": str(times[-1]),
            "geospatial_lat_min": float(np.nanmin(ds["latitude"].values))
            if "latitude" in ds
            else "unknown",
            "geospatial_lat_max": float(np.nanmax(ds["latitude"].values))
            if "latitude" in ds
            else "unknown",
            "geospatial_lon_min": float(np.nanmin(ds["longitude"].values))
            if "longitude" in ds
            else "unknown",
            "geospatial_lon_max": float(np.nanmax(ds["longitude"].values))
            if "longitude" in ds
            else "unknown",
            # Radar-specific attributes
            "radar_height_above_sea_surface": float(self._radar_height)
            if self._radar_height
            else "unknown",
            "range_resolution_m": float(slant_range[1] - slant_range[0])
            if len(slant_range) > 1
            else "unknown",
            "number_of_range_bins": n_distances,
            "number_of_bearing_bins": n_bearings,
        }

        # Define compression encoding for all variables
        encoding: dict[str, Any] = {}
        comp: dict[str, Any] = {"zlib": True, "complevel": compression_level}

        for var in ds.data_vars:
            encoding[var] = comp.copy()
            # Use appropriate chunking for large arrays
            if ds[var].ndim == 3:
                encoding[var]["chunksizes"] = (1, n_bearings, n_distances)

        # Time encoding for CF compliance
        encoding["time"] = {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "calendar": "proleptic_gregorian",
            "dtype": "float64",
        }

        # Write to NetCDF-4 with compression
        ds.to_netcdf(path, format="NETCDF4", encoding=encoding)
        logger.info(f"Wrote CF-1.11 compliant NetCDF file: {path}")

        return path

    def animate(
        self,
        path: str | Path,
        view: str = "polar",
        fps: int = 10,
        dpi: int = 150,
        cmap: str = "viridis",
        show_progress: bool = True,
    ) -> Path:
        """
        Create animation from processed frames.

        Args:
            path: Output file path (.mp4, .gif)
            view: View type ('polar', 'ship', 'earth')
            fps: Frames per second
            dpi: Resolution
            cmap: Colormap name
            show_progress: Show progress bar

        Returns:
            Path to created file

        Raises:
            RuntimeError: If data not processed
        """
        if not self._processed_flag:
            raise RuntimeError("Data not processed. Call process() first.")

        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError("matplotlib required for animation")

        path = Path(path)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get data range for consistent colormap
        all_data = np.concatenate(
            [
                (
                    f.corrected_intensity if f.corrected_intensity is not None else f.intensity
                ).ravel()
                for f in self._frames
            ]
        )
        vmin, vmax = np.percentile(all_data, [1, 99])

        # Animation update function
        def update(frame_idx):
            ax.clear()
            frame = self._frames[frame_idx]
            data = (
                frame.corrected_intensity
                if frame.corrected_intensity is not None
                else frame.intensity
            )

            if view == "polar":
                im = ax.pcolormesh(data, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xlabel("Range bin")
                ax.set_ylabel("Bearing bin")
            else:
                # For ship/earth views, use bearing coordinates
                if self._bearing is None:
                    raise RuntimeError("Bearing not computed")
                x, y = (
                    self._bearing.xy_ship(frame_idx)
                    if view == "ship"
                    else self._bearing.xy_earth(frame_idx)
                )
                im = ax.pcolormesh(x, y, data, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_aspect("equal")

            ax.set_title(f"Frame {frame_idx + 1}/{len(self._frames)}: {frame.timestamp}")
            return [im]

        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(self._frames), interval=1000 / fps, blit=False
        )

        # Save
        writer = "ffmpeg" if str(path).endswith(".mp4") else "pillow"
        pbar = tqdm(total=len(self._frames), desc="Rendering", disable=not show_progress)

        def progress_callback(current_frame, total_frames):
            pbar.update(1)

        anim.save(
            path,
            writer=writer,
            fps=fps,
            dpi=dpi,
            progress_callback=progress_callback if show_progress else None,
        )

        pbar.close()
        plt.close(fig)

        logger.info(f"Wrote animation: {path}")
        return path

    def summary(self) -> dict[str, Any]:
        """
        Return summary statistics of the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        summary = {
            "polar_path": str(self._polar_path),
            "stime": self._stime,
            "etime": self._etime,
            "n_files": self.n_files,
            "n_frames": self.n_frames if self._loaded else 0,
            "is_loaded": self._loaded,
            "is_processed": self._processed_flag,
            "config": str(self._config),
        }

        if self._loaded and self._frames:
            frame = self._frames[0]
            summary["frame_shape"] = (frame.n_bearings, frame.n_distances)
            summary["first_timestamp"] = str(self._frames[0].timestamp)
            summary["last_timestamp"] = str(self._frames[-1].timestamp)

        return summary

    def __repr__(self) -> str:
        return (
            f"WamosDataset(polar_path={self._polar_path}, "
            f"n_files={self.n_files}, n_frames={self.n_frames if self._loaded else 0}, "
            f"loaded={self._loaded}, processed={self._processed_flag})"
        )

    def __len__(self) -> int:
        """Return number of frames (0 if not loaded)."""
        return len(self._frames) if self._loaded else 0

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return iter(self._frames)

    def __getitem__(self, idx: int) -> Frame:
        """Get frame by index."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._frames[idx]
