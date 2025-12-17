#! /usr/bin/env python3
#
# Protocol definitions for type checking
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

from typing import Protocol, runtime_checkable, Tuple, Iterator, Any

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class FrameLike(Protocol):
    """Protocol for objects that behave like Frame."""

    @property
    def intensity(self) -> NDArray[np.uint16]:
        """Return intensity data array."""
        ...

    @property
    def n_bearings(self) -> int:
        """Return number of bearing bins."""
        ...

    @property
    def n_distances(self) -> int:
        """Return number of distance bins."""
        ...

    @property
    def timestamp(self) -> np.datetime64:
        """Return frame timestamp."""
        ...

    def slant_range(self) -> NDArray[np.float64]:
        """Return slant range values in meters."""
        ...

    def ground_range(self, radar_height: float) -> NDArray[np.float64]:
        """Return ground range values in meters."""
        ...


@runtime_checkable
class BearingProvider(Protocol):
    """Protocol for objects that provide bearing calculations."""

    def bearing_for_frame(self, frame_idx: int) -> NDArray[np.float64]:
        """Return bearing array for specified frame index."""
        ...

    def xy_ship(self, frame_idx: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return x/y coordinates in ship reference frame."""
        ...

    def xy_earth(self, frame_idx: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return x/y coordinates in earth reference frame."""
        ...


@runtime_checkable
class ViewerProtocol(Protocol):
    """Protocol for interactive frame viewers."""

    def _draw_plot(self) -> None:
        """Draw the current plot."""
        ...

    def _update_title(self) -> None:
        """Update the plot title."""
        ...

    def _get_frame(self, idx: int) -> Any:
        """Get frame at specified index."""
        ...


@runtime_checkable
class FilesLike(Protocol):
    """Protocol for objects that provide file iteration."""

    def __len__(self) -> int:
        """Return number of files."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over files/frames."""
        ...

    def itergroups(self) -> Iterator[Tuple[Any, Iterator[Any]]]:
        """Iterate over grouped files/frames."""
        ...


@runtime_checkable
class ProcessedFilesLike(FilesLike, Protocol):
    """Protocol for objects that provide processed file iteration."""

    def process_group(self, frames: list[Any]) -> list[NDArray[np.float64]]:
        """Process a group of frames."""
        ...
