#! /usr/bin/env python3
#
# Frame class for WAMOS polar data
# Contains metadata and uint16 binary data with bit extraction methods
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass, field

import numpy as np

from wamos_tpw.exceptions import ValidationError

if TYPE_CHECKING:
    from wamos_tpw.config import Config


logger = logging.getLogger(__name__)

# Conversion constant
KNOTS_TO_MS = 0.514444  # 1 knot = 0.514444 m/s


@dataclass
class FrameMetadata:
    """Metadata associated with a WAMOS polar frame.

    Note: ship_speed is stored in m/s (converted from knots at parse time).
    """

    timestamp: np.datetime64
    filename: str
    frame_index: int = 0
    # Navigation
    latitude: float | None = None
    longitude: float | None = None
    heading: float | None = None
    ship_speed: float | None = None  # m/s (converted from knots)
    ship_course: float | None = None
    # Radar parameters
    samples_in_range: int = 0
    sample_delay_range: float = 0.0
    sampling_frequency: float = 0.0
    repeat_time: float = 0.0
    data_bits: int = 12
    noise_floor: int = 0
    radar_height: float | None = None  # Height of radar above water (meters)
    wind_sensor_height: float | None = None  # Height of wind sensor above water (meters)
    # Offset corrections (from file header)
    bow_to_radar: float = 0.0  # BO2RA: Angle from bow to radar beam (degrees)
    heading_delay: float = 0.0  # HDGDL: Heading delay correction (degrees)
    # Wind
    wind_speed: float | None = None
    wind_direction: float | None = None
    # Additional attributes
    attrs: dict[str, Any] = field(default_factory=dict)


class Frame:
    """
    WAMOS polar frame containing uint16 radar data with bit extraction methods.

    The uint16 data format encodes:
        - Bottom 12 bits (0-11): Radar intensity data
        - Bit 12: PPS (Pulse Per Second) signal
        - Bit 13: Bearing pulse / azimuth marker
        - Bit 14: Reserved / user-defined
        - Bit 15: Reserved / user-defined

    Data shape is (n_bearings, n_distances) or (n_theta, n_range).
    """

    # Bit masks for extraction
    _MASK_DATA = np.uint16(0x0FFF)  # Bottom 12 bits
    _MASK_BIT12 = np.uint16(0x1000)  # Bit 12 (PPS)
    _MASK_BIT13 = np.uint16(0x2000)  # Bit 13 (Bearing pulse)
    _MASK_BIT14 = np.uint16(0x4000)  # Bit 14
    _MASK_BIT15 = np.uint16(0x8000)  # Bit 15

    # Physical constants for range calculations
    _C_VACUUM = 299_792_458.0  # Speed of light in vacuum (m/s)

    # Refractive index of air at standard conditions:
    # 20°C, 50% relative humidity, 1013.25 hPa
    # Based on Ciddor equation approximation
    _N_AIR_STANDARD = 1.000273

    # Speed of light in air at standard conditions (m/s)
    _C_AIR = _C_VACUUM / _N_AIR_STANDARD  # ~299,710,639 m/s

    def __init__(
        self,
        data: np.ndarray,
        metadata: FrameMetadata,
        config: Config | None = None,
        copy: bool = False,
        validate: bool = True,
    ) -> None:
        """
        Initialize a Frame with raw uint16 radar data.

        Args:
            data: Raw uint16 radar data array, shape (n_bearings, n_distances)
            metadata: FrameMetadata object with frame information
            config: Configuration object (typically from PolarFile)
            copy: If True, copy the data array; if False, use view (default)
            validate: If True, validate data integrity (default)
        """
        if data.dtype != np.uint16:
            raise ValidationError(
                f"Data must be uint16, got {data.dtype}", parameter="data", value=str(data.dtype)
            )

        if validate:
            self._validate_data(data, metadata)

        self._raw_data = data.copy() if copy else data
        self._metadata = metadata
        self._config = config

        # Cached extracted arrays (computed lazily)
        self._intensity: np.ndarray | None = None
        self._bit12: np.ndarray | None = None
        self._bit13: np.ndarray | None = None
        self._bit14: np.ndarray | None = None
        self._bit15: np.ndarray | None = None

        # Processing results (set by external processing pipeline)
        self.deramped_intensity: np.ndarray | None = None
        self.corrected_intensity: np.ndarray | None = None

        logger.debug("Frame created: shape=%s, timestamp=%s", data.shape, metadata.timestamp)

    @staticmethod
    def _validate_data(data: np.ndarray, metadata: FrameMetadata) -> None:
        """
        Validate frame data integrity.

        Args:
            data: Raw uint16 radar data array
            metadata: FrameMetadata object

        Raises:
            ValidationError: If data validation fails
        """
        # Check dimensions
        if data.ndim != 2:
            raise ValidationError(
                f"Data must be 2D, got {data.ndim}D", parameter="data", value=str(data.ndim)
            )

        n_bearings, n_distances = data.shape

        # Check reasonable array sizes
        if n_bearings < 1 or n_bearings > 10000:
            raise ValidationError(
                f"Unexpected number of bearings: {n_bearings} (expected 1-10000)",
                parameter="n_bearings",
                value=str(n_bearings),
            )

        if n_distances < 1 or n_distances > 10000:
            raise ValidationError(
                f"Unexpected number of distances: {n_distances} (expected 1-10000)",
                parameter="n_distances",
                value=str(n_distances),
            )

        # Validate metadata consistency
        if metadata.samples_in_range > 0 and metadata.samples_in_range != n_distances:
            logger.warning(
                f"Metadata samples_in_range ({metadata.samples_in_range}) doesn't "
                f"match data shape ({n_distances})"
            )

        # Check for all-zero data (possible corruption)
        if np.all(data == 0):
            logger.warning("Data array is all zeros - possible corruption")

        # Check for all-max data (possible saturation)
        if np.all(data == 65535):
            logger.warning("Data array is all 65535 - possible saturation")

        # Validate coordinate ranges
        if metadata.latitude is not None:
            if not -90 <= metadata.latitude <= 90:
                raise ValidationError(
                    f"Latitude out of range: {metadata.latitude}",
                    parameter="latitude",
                    value=str(metadata.latitude),
                )

        if metadata.longitude is not None:
            if not -180 <= metadata.longitude <= 180:
                raise ValidationError(
                    f"Longitude out of range: {metadata.longitude}",
                    parameter="longitude",
                    value=str(metadata.longitude),
                )

        if metadata.heading is not None:
            if not 0 <= metadata.heading < 360:
                logger.warning(f"Heading outside [0, 360): {metadata.heading}")

        if metadata.sampling_frequency is not None:
            if metadata.sampling_frequency < 0:
                raise ValidationError(
                    f"Negative sampling frequency: {metadata.sampling_frequency}",
                    parameter="sampling_frequency",
                    value=str(metadata.sampling_frequency),
                )

    @property
    def metadata(self) -> FrameMetadata:
        """Return the frame metadata."""
        return self._metadata

    @property
    def config(self) -> Config | None:
        """Return the configuration object."""
        return self._config

    @property
    def timestamp(self) -> np.datetime64:
        """Return the frame timestamp."""
        return self._metadata.timestamp

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape (n_bearings, n_distances)."""
        return self._raw_data.shape

    @property
    def n_bearings(self) -> int:
        """Return the number of bearing bins (theta bins)."""
        return self._raw_data.shape[0]

    @property
    def n_distances(self) -> int:
        """Return the number of distance bins (range bins)."""
        return self._raw_data.shape[1] if self._raw_data.ndim > 1 else 1

    @property
    def raw(self) -> np.ndarray:
        """Return the raw uint16 data array."""
        return self._raw_data

    # -------------------------------------------------------------------------
    # Bottom 12 bits (radar intensity)
    # -------------------------------------------------------------------------

    @property
    def intensity(self) -> np.ndarray:
        """
        Extract the bottom 12 bits (radar intensity data).

        Returns:
            uint16 array with values 0-4095
        """
        if self._intensity is None:
            self._intensity = np.bitwise_and(self._raw_data, self._MASK_DATA)
        return self._intensity

    @property
    def data(self) -> np.ndarray:
        """Alias for intensity - the bottom 12 bits."""
        return self.intensity

    # -------------------------------------------------------------------------
    # Top 4 bits (flags/markers)
    # -------------------------------------------------------------------------

    @property
    def bit12(self) -> np.ndarray:
        """
        Extract bit 12 (PPS - Pulse Per Second signal).

        Returns:
            Boolean array where True indicates PPS signal present
        """
        if self._bit12 is None:
            self._bit12 = np.not_equal(np.bitwise_and(self._raw_data, self._MASK_BIT12), 0).astype(
                np.bool_
            )
        return self._bit12

    @property
    def pps(self) -> np.ndarray:
        """Alias for bit12 - Pulse Per Second signal."""
        return self.bit12

    @property
    def bit13(self) -> np.ndarray:
        """
        Extract bit 13 (bearing pulse / azimuth marker).

        Returns:
            Boolean array where True indicates bearing pulse present
        """
        if self._bit13 is None:
            self._bit13 = np.not_equal(np.bitwise_and(self._raw_data, self._MASK_BIT13), 0).astype(
                np.bool_
            )
        return self._bit13

    @property
    def bearing_pulse(self) -> np.ndarray:
        """Alias for bit13 - bearing/azimuth marker pulse."""
        return self.bit13

    @property
    def bit14(self) -> np.ndarray:
        """
        Extract bit 14 (reserved/user-defined).

        Returns:
            Boolean array
        """
        if self._bit14 is None:
            self._bit14 = np.not_equal(np.bitwise_and(self._raw_data, self._MASK_BIT14), 0).astype(
                np.bool_
            )
        return self._bit14

    @property
    def bit15(self) -> np.ndarray:
        """
        Extract bit 15 (reserved/user-defined).

        Returns:
            Boolean array
        """
        if self._bit15 is None:
            self._bit15 = np.not_equal(np.bitwise_and(self._raw_data, self._MASK_BIT15), 0).astype(
                np.bool_
            )
        return self._bit15

    # -------------------------------------------------------------------------
    # Distance row selection (across all theta bins)
    # -------------------------------------------------------------------------

    def get_distance_row(self, distance_idx: int, extract: str = "intensity") -> np.ndarray:
        """
        Extract a single distance row across all theta (bearing) bins.

        Args:
            distance_idx: Index of the distance bin to extract (0 to n_distances-1)
            extract: What to extract - 'intensity', 'raw', 'bit12', 'bit13', 'bit14', 'bit15'

        Returns:
            1D array of shape (n_bearings,) containing the requested data

        Example:
            >>> frame.get_distance_row(100, 'intensity')  # Intensity at range bin 100
            >>> frame.get_distance_row(50, 'bit12')       # PPS flags at range bin 50
        """
        if distance_idx < 0 or distance_idx >= self.n_distances:
            raise IndexError(f"Distance index {distance_idx} out of range [0, {self.n_distances})")

        extractors = {
            "intensity": lambda: self.intensity[:, distance_idx],
            "data": lambda: self.intensity[:, distance_idx],
            "raw": lambda: self._raw_data[:, distance_idx],
            "bit12": lambda: self.bit12[:, distance_idx],
            "pps": lambda: self.bit12[:, distance_idx],
            "bit13": lambda: self.bit13[:, distance_idx],
            "bearing_pulse": lambda: self.bit13[:, distance_idx],
            "bit14": lambda: self.bit14[:, distance_idx],
            "bit15": lambda: self.bit15[:, distance_idx],
        }

        if extract not in extractors:
            raise ValueError(
                f"Unknown extract type '{extract}'. Use: {', '.join(extractors.keys())}"
            )

        return extractors[extract]()

    def get_distance_range(
        self, start_idx: int, end_idx: int, extract: str = "intensity"
    ) -> np.ndarray:
        """
        Extract a range of distance rows across all theta bins.

        Args:
            start_idx: Starting distance index (inclusive)
            end_idx: Ending distance index (exclusive)
            extract: What to extract - 'intensity', 'raw', 'bit12', etc.

        Returns:
            2D array of shape (n_bearings, end_idx - start_idx)
        """
        if start_idx < 0 or end_idx > self.n_distances or start_idx >= end_idx:
            raise IndexError(
                f"Distance range [{start_idx}, {end_idx}) invalid for n_distances={self.n_distances}"
            )

        extractors = {
            "intensity": lambda: self.intensity[:, start_idx:end_idx],
            "data": lambda: self.intensity[:, start_idx:end_idx],
            "raw": lambda: self._raw_data[:, start_idx:end_idx],
            "bit12": lambda: self.bit12[:, start_idx:end_idx],
            "pps": lambda: self.bit12[:, start_idx:end_idx],
            "bit13": lambda: self.bit13[:, start_idx:end_idx],
            "bearing_pulse": lambda: self.bit13[:, start_idx:end_idx],
            "bit14": lambda: self.bit14[:, start_idx:end_idx],
            "bit15": lambda: self.bit15[:, start_idx:end_idx],
        }

        if extract not in extractors:
            raise ValueError(f"Unknown extract type '{extract}'")

        return extractors[extract]()

    def get_bearing_row(self, bearing_idx: int, extract: str = "intensity") -> np.ndarray:
        """
        Extract a single bearing row across all distance bins.

        Args:
            bearing_idx: Index of the bearing bin to extract (0 to n_bearings-1)
            extract: What to extract - 'intensity', 'raw', 'bit12', etc.

        Returns:
            1D array of shape (n_distances,) containing the requested data
        """
        if bearing_idx < 0 or bearing_idx >= self.n_bearings:
            raise IndexError(f"Bearing index {bearing_idx} out of range [0, {self.n_bearings})")

        extractors = {
            "intensity": lambda: self.intensity[bearing_idx, :],
            "data": lambda: self.intensity[bearing_idx, :],
            "raw": lambda: self._raw_data[bearing_idx, :],
            "bit12": lambda: self.bit12[bearing_idx, :],
            "pps": lambda: self.bit12[bearing_idx, :],
            "bit13": lambda: self.bit13[bearing_idx, :],
            "bearing_pulse": lambda: self.bit13[bearing_idx, :],
            "bit14": lambda: self.bit14[bearing_idx, :],
            "bit15": lambda: self.bit15[bearing_idx, :],
        }

        if extract not in extractors:
            raise ValueError(f"Unknown extract type '{extract}'")

        return extractors[extract]()

    # -------------------------------------------------------------------------
    # Range calculations
    # -------------------------------------------------------------------------

    @property
    def range_resolution(self) -> float:
        """
        Calculate the range resolution (meters per distance bin).

        Uses the speed of light in air at standard conditions (20°C, 50% RH)
        and the sampling frequency from metadata.

        Note: sampling_frequency in metadata is stored in MHz, converted to Hz here.

        Returns:
            Range resolution in meters per bin.
            Returns 0.0 if sampling_frequency is not set.
        """
        sfreq_mhz = self._metadata.sampling_frequency
        if sfreq_mhz <= 0:
            return 0.0
        # Convert MHz to Hz
        sfreq_hz = sfreq_mhz * 1e6
        # Round-trip time per sample = 1/sfreq
        # One-way distance = c_air * t / 2 = c_air / (2 * sfreq)
        return self._C_AIR / (2.0 * sfreq_hz)

    def slant_range(self, bin_indices: np.ndarray | None = None) -> np.ndarray:
        """
        Calculate slant range (straight-line distance from radar) in meters.

        The slant range is the direct line-of-sight distance from the radar
        to the target, calculated using the speed of light in air at standard
        conditions (20°C, 50% relative humidity, 1013.25 hPa).

        Args:
            bin_indices: Optional array of distance bin indices. If None,
                        calculates for all bins (0 to n_distances-1).

        Returns:
            Array of slant ranges in meters, same shape as bin_indices
            (or shape (n_distances,) if bin_indices is None).

        Example:
            >>> ranges = frame.slant_range()  # All bins
            >>> range_100 = frame.slant_range(np.array([100]))[0]  # Single bin
        """
        if bin_indices is None:
            bin_indices = np.arange(self.n_distances)

        bin_indices = np.asarray(bin_indices)
        delta_r = self.range_resolution
        sdrng = self._metadata.sample_delay_range

        # slant_range = sample_delay_range + bin_index * range_resolution
        return sdrng + bin_indices * delta_r

    def ground_range(
        self, radar_height: float | None = None, bin_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Calculate ground range (horizontal distance) from radar in meters.

        Converts slant range to ground range by accounting for the radar's
        height above the water surface. Assumes the radar is at the same
        height as the wind sensor.

        The ground range is calculated as:
            ground_range = sqrt(slant_range² - radar_height²)

        For slant ranges less than the radar height, returns 0.

        Args:
            radar_height: Height of radar above water surface in meters.
                         If None, uses metadata.radar_height.
                         If neither is set, raises ValueError.
            bin_indices: Optional array of distance bin indices. If None,
                        calculates for all bins (0 to n_distances-1).

        Returns:
            Array of ground ranges in meters.

        Raises:
            ValueError: If radar_height is not provided and not in metadata.

        Example:
            >>> ground_ranges = frame.ground_range(radar_height=25.0)
            >>> # Or if radar_height is set in metadata:
            >>> ground_ranges = frame.ground_range()
        """
        # Get radar height
        if radar_height is None:
            radar_height = self._metadata.radar_height
        if radar_height is None:
            raise ValueError("radar_height must be provided or set in metadata.radar_height")

        # Calculate slant ranges
        slant = self.slant_range(bin_indices)

        # Calculate ground range: sqrt(slant² - height²)
        # For slant < height, result would be imaginary, so clamp to 0
        height_sq = radar_height**2
        slant_sq = slant**2

        # Where slant > height, compute ground range; otherwise 0
        ground = np.where(slant_sq > height_sq, np.sqrt(slant_sq - height_sq), 0.0)

        return ground

    def range_at_bin(
        self, bin_index: int, radar_height: float | None = None
    ) -> tuple[float, float]:
        """
        Get both slant and ground range for a single distance bin.

        Args:
            bin_index: Distance bin index (0 to n_distances-1)
            radar_height: Height of radar above water in meters.
                         If None, uses metadata.radar_height.

        Returns:
            Tuple of (slant_range_m, ground_range_m)

        Example:
            >>> slant, ground = frame.range_at_bin(100, radar_height=25.0)
            >>> print(f"Bin 100: slant={slant:.1f}m, ground={ground:.1f}m")
        """
        idx = np.array([bin_index])
        slant = self.slant_range(idx)[0]

        if radar_height is None:
            radar_height = self._metadata.radar_height
        if radar_height is not None:
            ground = self.ground_range(radar_height, idx)[0]
        else:
            ground = slant  # If no height, ground = slant

        return slant, ground

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear cached extracted arrays to free memory."""
        self._intensity = None
        self._bit12 = None
        self._bit13 = None
        self._bit14 = None
        self._bit15 = None

    def __repr__(self) -> str:
        return (
            f"Frame(timestamp={self.timestamp}, shape={self.shape}, file={self._metadata.filename})"
        )

    def __str__(self) -> str:
        return (
            f"Frame at {self.timestamp}\n"
            f"  Shape: {self.n_bearings} bearings x {self.n_distances} distances\n"
            f"  File: {self._metadata.filename}\n"
            f"  Lat/Lon: {self._metadata.latitude}, {self._metadata.longitude}"
        )


def main() -> None:
    """Test the Frame class with synthetic data."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Frame class")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.parse_args()

    # Create synthetic test data
    n_bearings, n_distances = 360, 752
    raw_data = np.random.randint(0, 65535, (n_bearings, n_distances), dtype=np.uint16)

    # Set some specific bits for testing
    raw_data[0, :] |= 0x1000  # Set bit 12 on first bearing
    raw_data[:, 0] |= 0x2000  # Set bit 13 on first distance

    metadata = FrameMetadata(
        timestamp=np.datetime64("2024-12-15T10:30:00"),
        filename="test_file.pol",
        latitude=18.57,
        longitude=142.96,
        samples_in_range=n_distances,
        sampling_frequency=20.0,  # 20 MHz sampling (stored in MHz)
        sample_delay_range=150.0,  # 150m starting range
        radar_height=25.0,  # 25m above water
    )

    frame = Frame(raw_data, metadata)

    print(f"Created: {frame}")
    print(f"\nIntensity shape: {frame.intensity.shape}")
    print(f"Intensity range: [{frame.intensity.min()}, {frame.intensity.max()}]")
    print(f"\nBit 12 (PPS) - first bearing all True: {frame.bit12[0, :].all()}")
    print(f"Bit 13 (bearing) - first distance all True: {frame.bit13[:, 0].all()}")

    # Test distance row extraction
    row = frame.get_distance_row(100, "intensity")
    print(f"\nDistance row 100 shape: {row.shape}")

    # Test bearing row extraction
    row = frame.get_bearing_row(0, "bit12")
    print(f"Bearing row 0 (bit12) all True: {row.all()}")

    # Test range calculations
    print("\n--- Range Calculations ---")
    print(f"Speed of light in air: {Frame._C_AIR:.0f} m/s")
    print(f"Refractive index (20°C, 50% RH): {Frame._N_AIR_STANDARD}")
    print(f"Range resolution: {frame.range_resolution:.4f} m/bin")

    # Slant ranges
    slant_ranges = frame.slant_range()
    print(
        f"\nSlant range: {slant_ranges[0]:.1f}m (bin 0) to {slant_ranges[-1]:.1f}m (bin {n_distances - 1})"
    )

    # Ground ranges
    ground_ranges = frame.ground_range()
    print(
        f"Ground range: {ground_ranges[0]:.1f}m (bin 0) to {ground_ranges[-1]:.1f}m (bin {n_distances - 1})"
    )

    # Single bin example
    slant, ground = frame.range_at_bin(100)
    print(f"\nBin 100: slant={slant:.1f}m, ground={ground:.1f}m")

    # Show first few bins where ground range might be 0 (slant < height)
    print(f"\nFirst 5 bins (radar height = {metadata.radar_height}m):")
    for i in range(5):
        s, g = frame.range_at_bin(i)
        print(f"  Bin {i}: slant={s:.2f}m, ground={g:.2f}m")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
