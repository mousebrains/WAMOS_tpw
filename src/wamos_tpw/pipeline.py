#! /usr/bin/env python3
#
# Unified pipeline interface for WAMOS radar data processing
#
# Provides a common Protocol for pipeline implementations and a factory function
# for creating pipelines based on processing mode.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""Unified pipeline interface for WAMOS radar data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from wamos_tpw.config import Config
    from wamos_tpw.merged_image import MergedImage, TimeWindowConfig


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


__all__ = ["MergePipeline", "create_pipeline"]
