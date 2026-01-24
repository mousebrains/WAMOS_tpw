#! /usr/bin/env python3
#
# Data structures for merged radar images
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TimeWindowConfig:
    """Configuration for time-based windowing of frames."""

    window_seconds: float = 60.0  # Window duration in seconds
    overlap_fraction: float = 0.5  # Overlap between consecutive windows (0.0-1.0)
    min_frames_per_window: int = 5  # Minimum frames required to produce output
    resolution_scale: float = 1.0  # Grid resolution multiplier (2.0 = 2x finer grid)
    interpolate_gaps: bool = False  # Fill NaN gaps with interpolated values

    @property
    def stride_seconds(self) -> float:
        """Compute stride (time between window starts) from overlap."""
        return self.window_seconds * (1.0 - self.overlap_fraction)

    def __post_init__(self):
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("overlap_fraction must be in [0, 1)")
        if self.min_frames_per_window < 1:
            raise ValueError("min_frames_per_window must be at least 1")
        if self.resolution_scale <= 0:
            raise ValueError("resolution_scale must be positive")


@dataclass
class MergedImage:
    """A motion-corrected composite image from multiple frames."""

    intensity: np.ndarray  # 2D averaged intensity (n_y, n_x)
    x_edges: np.ndarray  # Grid x edges in meters (from center)
    y_edges: np.ndarray  # Grid y edges in meters (from center)
    start_time: np.datetime64  # Window start time
    end_time: np.datetime64  # Window end time
    n_frames: int  # Number of frames merged
    utm_zone: int  # UTM zone number
    hemisphere: str  # 'north' or 'south'
    center_lat: float  # Grid center latitude
    center_lon: float  # Grid center longitude
    grid_spacing: float  # Grid cell size in meters
    mean_heading: float  # Mean ship heading during window
    mean_ship_speed: float | None = None
    mean_wind_speed: float | None = None
    mean_wind_direction: float | None = None
    window_index: int = 0  # Index of this window in the sequence

    @property
    def n_x(self) -> int:
        """Number of x bins."""
        return len(self.x_edges) - 1

    @property
    def n_y(self) -> int:
        """Number of y bins."""
        return len(self.y_edges) - 1

    @property
    def x_centers(self) -> np.ndarray:
        """Grid x centers."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        """Grid y centers."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def duration_seconds(self) -> float:
        """Duration of the time window in seconds."""
        return (self.end_time - self.start_time) / np.timedelta64(1, "s")
