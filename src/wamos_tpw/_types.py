"""
Type aliases and stubs for wamos_tpw.

Provides common type aliases used throughout the package,
especially for numpy array typing.
"""

from __future__ import annotations

from typing import TypeVar, Union, Sequence, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

# Generic type variable for numpy arrays
T = TypeVar('T', bound=np.generic)

# Common array type aliases
IntArray = NDArray[np.int_]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
UInt16Array = NDArray[np.uint16]
UInt32Array = NDArray[np.uint32]

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

BoolArray = NDArray[np.bool_]

DateTimeArray = NDArray[np.datetime64]
TimeDeltaArray = NDArray[np.timedelta64]

# Intensity data (12-bit stored in uint16)
IntensityArray = UInt16Array

# Bearing/angle data (float degrees)
BearingArray = Float64Array

# Range data (float meters)
RangeArray = Float64Array

# Coordinate arrays (x, y in meters)
CoordinateArray = Float64Array

# Array-like inputs (can be converted to numpy array)
ArrayLike = Union[
    np.ndarray,
    Sequence[float],
    Sequence[int],
    Sequence[Sequence[float]],
    Sequence[Sequence[int]],
]

# Shape types
Shape1D = tuple[int]
Shape2D = tuple[int, int]
Shape3D = tuple[int, int, int]
ShapeND = tuple[int, ...]


@runtime_checkable
class FrameProtocol(Protocol):
    """Protocol defining the Frame interface for type checking."""

    @property
    def timestamp(self) -> np.datetime64:
        """Return the frame timestamp."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape (n_bearings, n_distances)."""
        ...

    @property
    def n_bearings(self) -> int:
        """Return the number of bearing bins."""
        ...

    @property
    def n_distances(self) -> int:
        """Return the number of distance bins."""
        ...

    @property
    def intensity(self) -> UInt16Array:
        """Extract the bottom 12 bits (radar intensity data)."""
        ...

    def slant_range(self, bin_indices: FloatArray | None = None) -> FloatArray:
        """Calculate slant range in meters."""
        ...


@runtime_checkable
class BearingCalculator(Protocol):
    """Protocol for bearing calculation providers."""

    def heading_ship(self, frame_idx: int) -> BearingArray:
        """Get beam heading relative to ship bow."""
        ...

    def heading_earth(self, frame_idx: int) -> BearingArray:
        """Get beam heading in earth coordinates."""
        ...

    def xy_ship(self, frame_idx: int) -> tuple[CoordinateArray, CoordinateArray]:
        """Get x/y coordinates in ship reference frame."""
        ...

    def xy_earth(self, frame_idx: int) -> tuple[CoordinateArray, CoordinateArray]:
        """Get x/y coordinates in earth reference frame."""
        ...


# Type for config-like objects
ConfigDict = dict[str, Union[str, int, float, bool, None, 'ConfigDict']]

# Result types
ProcessingResult = dict[np.datetime64, list[FloatArray]]
