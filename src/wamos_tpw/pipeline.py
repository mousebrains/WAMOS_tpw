#! /usr/bin/env python3
#
# Unified pipeline interface for WAMOS radar data processing
#
# Provides a common Protocol for pipeline implementations and a factory function
# for creating pipelines based on processing mode.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""
Unified pipeline interface for WAMOS radar data processing.

This module provides:
- MergePipeline: Protocol for pipeline implementations
- ProgressCallback: Protocol for progress reporting
- create_pipeline: Factory function for creating pipelines

Example with progress callback::

    def my_progress(event: str, current: int, total: int, **kwargs) -> None:
        if event == "window_complete":
            print(f"Window {current}/{total} complete")
        elif event == "frame_loaded":
            print(f"Loaded frame {kwargs.get('filename')}")

    pipeline = create_pipeline("batch", filenames=files)
    for merged in pipeline.iter_merged():
        my_progress("window_complete", merged.window_index + 1, pipeline.n_windows)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from wamos_tpw.config import Config
    from wamos_tpw.merged_image import MergedImage, TimeWindowConfig


# =============================================================================
# Progress Callback Types
# =============================================================================


@dataclass
class ProgressEvent:
    """
    Progress event data for pipeline callbacks.

    Attributes:
        event_type: Type of event (see ProgressEventType for values)
        current: Current progress value (e.g., frame number, window number)
        total: Total expected value (0 if unknown)
        message: Optional human-readable message
        metadata: Additional event-specific data
    """

    event_type: str
    current: int
    total: int
    message: str = ""
    metadata: dict | None = None


# Type alias for progress callback functions
ProgressCallback = Callable[[ProgressEvent], None]


class ProgressEventType:
    """
    Standard progress event types.

    Use these constants when registering callbacks or checking event types.
    """

    # Pipeline-level events
    PIPELINE_START = "pipeline_start"  # Pipeline beginning
    PIPELINE_COMPLETE = "pipeline_complete"  # Pipeline finished
    PIPELINE_ERROR = "pipeline_error"  # Pipeline error occurred

    # Window-level events
    WINDOW_START = "window_start"  # Starting a time window
    WINDOW_COMPLETE = "window_complete"  # Window processing complete
    WINDOW_SKIP = "window_skip"  # Window skipped (insufficient frames)

    # Frame-level events
    FRAME_LOAD = "frame_load"  # Frame loaded from file
    FRAME_PROCESS = "frame_process"  # Frame processed
    FRAME_PROJECT = "frame_project"  # Frame projected to grid
    FRAME_ERROR = "frame_error"  # Frame processing error

    # File-level events
    FILE_DISCOVER = "file_discover"  # File discovered
    FILE_SKIP = "file_skip"  # File skipped


class ProgressReporter:
    """
    Simple progress reporter that can be used with pipelines.

    Example::

        reporter = ProgressReporter(verbose=True)

        # Can be used as a callback
        reporter.on_event(ProgressEvent("window_complete", 5, 10))

        # Or track progress manually
        reporter.start("Processing", total=100)
        for i in range(100):
            reporter.update(i + 1)
        reporter.complete()
    """

    def __init__(self, verbose: bool = False, prefix: str = "") -> None:
        """
        Initialize progress reporter.

        Args:
            verbose: If True, print all events
            prefix: Optional prefix for messages
        """
        self._verbose = verbose
        self._prefix = prefix
        self._current = 0
        self._total = 0
        self._task = ""

    def on_event(self, event: ProgressEvent) -> None:
        """Handle a progress event (callback interface)."""
        if self._verbose:
            pct = f" ({100 * event.current / event.total:.1f}%)" if event.total > 0 else ""
            msg = f"{self._prefix}{event.event_type}: {event.current}/{event.total}{pct}"
            if event.message:
                msg += f" - {event.message}"
            print(msg)

    def start(self, task: str, total: int = 0) -> None:
        """Start tracking a new task."""
        self._task = task
        self._total = total
        self._current = 0
        if self._verbose:
            print(f"{self._prefix}{task}: starting (total={total})")

    def update(self, current: int, message: str = "") -> None:
        """Update progress."""
        self._current = current
        if self._verbose:
            pct = f" ({100 * current / self._total:.1f}%)" if self._total > 0 else ""
            msg = f"{self._prefix}{self._task}: {current}/{self._total}{pct}"
            if message:
                msg += f" - {message}"
            print(msg)

    def complete(self, message: str = "") -> None:
        """Mark task as complete."""
        if self._verbose:
            msg = f"{self._prefix}{self._task}: complete"
            if message:
                msg += f" - {message}"
            print(msg)


# =============================================================================
# Pipeline Protocol
# =============================================================================


@runtime_checkable
class MergePipeline(Protocol):
    """
    Protocol defining the common interface for merge pipelines.

    Both FilesMergePipeline and StreamingMergePipeline implement this protocol,
    allowing code to work with either pipeline type through a common interface.

    Example::

        def process_pipeline(pipeline: MergePipeline) -> None:
            print(f"Processing {pipeline.n_windows} windows")
            for merged in pipeline.iter_merged():
                save_image(merged)
    """

    @property
    def n_windows(self) -> int:
        """Number of time windows to process."""
        ...

    @property
    def window_config(self) -> "TimeWindowConfig":
        """Time window configuration."""
        ...

    def iter_merged(self) -> Iterator["MergedImage"]:
        """
        Yield merged images as windows complete.

        Returns:
            Iterator of MergedImage objects, one per completed time window.
        """
        ...


def create_pipeline(
    mode: Literal["batch", "streaming"],
    *,
    # Common parameters
    config: "Config | None" = None,
    window_config: "TimeWindowConfig | None" = None,
    n_workers: int | None = None,
    tolerance: float = 1.2,
    qTiming: bool = False,
    qProgress: bool = True,
    max_windows: int | None = None,
    pending_multiplier: float = 3.0,
    max_queued_merges: int = 4,
    # Batch mode parameters
    filenames: list[str] | None = None,
    # Streaming mode parameters
    stime: str | None = None,
    etime: str | None = None,
    polar_path: str | None = None,
) -> MergePipeline:
    """
    Factory function to create a merge pipeline.

    Creates either a batch (FilesMergePipeline) or streaming (StreamingMergePipeline)
    pipeline based on the mode parameter.

    Args:
        mode: Pipeline mode - "batch" or "streaming"

        Common parameters:
            config: YAML configuration
            window_config: Time window configuration
            n_workers: Number of parallel workers (default: auto)
            tolerance: Multiplier for repeat_time to accept frame pair
            qTiming: Enable timing statistics
            qProgress: Show progress bars
            max_windows: Maximum number of windows to process (None = all)
            pending_multiplier: Max in-flight file loads as multiplier of n_workers
            max_queued_merges: Maximum windows queued for merge thread

        Batch mode parameters (required when mode="batch"):
            filenames: List of polar file paths to process

        Streaming mode parameters (required when mode="streaming"):
            stime: Start time for processing
            etime: End time for processing
            polar_path: Root directory containing polar files

    Returns:
        A MergePipeline instance (either FilesMergePipeline or StreamingMergePipeline)

    Raises:
        ValueError: If required parameters for the selected mode are missing

    Example::

        # Batch mode - all files known upfront
        pipeline = create_pipeline(
            "batch",
            filenames=glob.glob("/data/POLAR/**/*.pol.gz"),
            config=config,
        )

        # Streaming mode - files discovered progressively
        pipeline = create_pipeline(
            "streaming",
            stime="2022-03-01",
            etime="2022-03-31",
            polar_path="/data/POLAR",
            config=config,
        )

        # Both can be used the same way
        for merged in pipeline.iter_merged():
            save_netcdf(merged)
    """
    if mode == "batch":
        if filenames is None:
            raise ValueError("'filenames' is required for batch mode")

        from wamos_tpw.files_pipeline import FilesMergePipeline

        return FilesMergePipeline(
            filenames=filenames,
            config=config,
            window_config=window_config,
            n_workers=n_workers,
            tolerance=tolerance,
            qTiming=qTiming,
            qProgress=qProgress,
            max_windows=max_windows,
            pending_multiplier=pending_multiplier,
            max_queued_merges=max_queued_merges,
        )

    elif mode == "streaming":
        if stime is None or etime is None or polar_path is None:
            raise ValueError("'stime', 'etime', and 'polar_path' are required for streaming mode")

        from wamos_tpw.streaming_pipeline import StreamingMergePipeline

        return StreamingMergePipeline(
            stime=stime,
            etime=etime,
            polar_path=polar_path,
            config=config,
            window_config=window_config,
            n_workers=n_workers,
            tolerance=tolerance,
            qTiming=qTiming,
            qProgress=qProgress,
            max_windows=max_windows,
            pending_multiplier=pending_multiplier,
            max_queued_merges=max_queued_merges,
        )

    else:
        raise ValueError(f"Unknown pipeline mode: {mode!r}. Use 'batch' or 'streaming'.")


__all__ = [
    "MergePipeline",
    "create_pipeline",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressEventType",
    "ProgressReporter",
]
