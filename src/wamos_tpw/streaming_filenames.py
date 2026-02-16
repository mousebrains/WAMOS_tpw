#! /usr/bin/env python3
#
# Streaming file discovery for WAMOS polar data
#
# Yields files as they're discovered, enabling pipeline processing to start
# immediately rather than waiting for all files to be found.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Empty, Queue

import numpy as np

from wamos_tpw.filenames import _scan_hour_directory, extract_file_timestamp

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredFile:
    """A file discovered during scanning."""

    filepath: str
    timestamp_ns: int  # Nanoseconds since epoch for fast comparison
    file_id: int  # Unique ID assigned during discovery


@dataclass
class HourBatch:
    """A batch of files from a single hour directory."""

    hour_start_ns: int  # Start of this hour (ns since epoch)
    hour_end_ns: int  # End of this hour (ns since epoch)
    files: list[DiscoveredFile]
    is_final: bool = False  # True if this is the last batch


@dataclass
class StreamingDiscoveryState:
    """
    Tracks the state of streaming file discovery.

    This allows the pipeline to know:
    - Which time ranges have been fully scanned
    - Whether discovery is complete
    - How many files have been found so far
    """

    stime_ns: int
    etime_ns: int
    current_hour_ns: int = 0  # Hour currently being/last scanned
    total_files_discovered: int = 0
    is_complete: bool = False
    error: str | None = None


class StreamingFilenames:
    """
    Stream-based file discovery that yields files as they're found.

    Unlike the blocking `Filenames` class, this yields files incrementally,
    allowing the processing pipeline to start immediately.

    Files are discovered chronologically (hour by hour), which enables:
    - Early pipeline startup (first files available within seconds)
    - Time-based window completion detection
    - Memory-efficient processing of large datasets

    Example::

        streaming = StreamingFilenames(stime, etime, polar_path)

        # Start discovery in background
        streaming.start()

        # Process files as they're discovered
        for batch in streaming.iter_batches():
            for file in batch.files:
                submit_for_processing(file)

            # Windows ending before batch.hour_end_ns are discovery-complete
            mark_windows_discovery_complete(batch.hour_end_ns)

        # All files have been discovered
        assert streaming.state.is_complete
    """

    # Number of hour directories to scan in parallel
    _PARALLEL_BATCH_SIZE = 8

    def __init__(
        self,
        stime: np.datetime64,
        etime: np.datetime64,
        polar_path: str,
        workers: int | None = None,
        buffer_size: int = 100,
    ) -> None:
        """
        Initialize streaming file discovery.

        Args:
            stime: Start time for file search
            etime: End time for file search
            polar_path: Root directory containing polar files
            workers: Number of worker processes for parallel scanning
            buffer_size: Maximum batches to buffer before blocking
        """
        self.stime = np.datetime64(stime, "ns")
        self.etime = np.datetime64(etime, "ns")
        if self.stime > self.etime:
            raise ValueError(f"Start time ({self.stime}) must be <= end time ({self.etime})")

        self.polar_path = Path(polar_path)
        self.workers = workers or min(os.cpu_count() or 1, self._PARALLEL_BATCH_SIZE)
        self._buffer_size = buffer_size

        # Convert to nanoseconds for fast comparison
        self._stime_ns = self.stime.astype(np.int64)
        self._etime_ns = self.etime.astype(np.int64)

        # Discovery state
        self._state = StreamingDiscoveryState(
            stime_ns=self._stime_ns,
            etime_ns=self._etime_ns,
        )

        # Threading infrastructure
        self._batch_queue: Queue[HourBatch | None] = Queue(maxsize=buffer_size)
        self._discovery_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

        # File ID counter (thread-safe via queue ordering)
        self._next_file_id = 0

    @property
    def state(self) -> StreamingDiscoveryState:
        """Current discovery state."""
        return self._state

    def _generate_hour_directories(self) -> Iterator[tuple[str, int, int]]:
        """
        Generate hour directory paths with their time bounds.

        Yields:
            Tuples of (dir_path, hour_start_ns, hour_end_ns)
        """
        current = np.datetime64(self.stime, "h")
        end_hour = np.datetime64(self.etime, "h")
        one_hour = np.timedelta64(1, "h")

        while current <= end_hour:
            dt_str = str(current)
            parts = dt_str.replace("T", "-").split("-")
            year_s, month_s, day_s, hour_s = parts[0], parts[1], parts[2], parts[3]

            dir_path = self.polar_path / year_s / month_s / day_s / hour_s

            hour_start_ns = current.astype("datetime64[ns]").astype(np.int64)
            next_hour = current + one_hour
            hour_end_ns = next_hour.astype("datetime64[ns]").astype(np.int64)

            yield str(dir_path), hour_start_ns, hour_end_ns
            current = next_hour

    def _scan_hours_parallel(
        self, hour_dirs: list[tuple[str, int, int]]
    ) -> list[tuple[int, int, list[str]]]:
        """
        Scan multiple hour directories in parallel.

        Returns list of (hour_start_ns, hour_end_ns, files) tuples.
        """
        if not hour_dirs:
            return []

        scan_func = partial(
            _scan_hour_directory,
            stime_ns=self._stime_ns,
            etime_ns=self._etime_ns,
        )

        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all directories
            future_to_hour: dict[Future, tuple[int, int]] = {}
            for dir_path, hour_start_ns, hour_end_ns in hour_dirs:
                future = executor.submit(scan_func, dir_path)
                future_to_hour[future] = (hour_start_ns, hour_end_ns)

            # Collect results as they complete
            for future in as_completed(future_to_hour):
                hour_start_ns, hour_end_ns = future_to_hour[future]
                try:
                    files = future.result()
                    results.append((hour_start_ns, hour_end_ns, files))
                except Exception as e:
                    logger.warning("Error scanning directory: %s", e)
                    results.append((hour_start_ns, hour_end_ns, []))

        # Sort by hour start time to maintain chronological order
        results.sort(key=lambda x: x[0])
        return results

    def _discovery_worker(self) -> None:
        """
        Background worker that discovers files and queues batches.

        Submits all directories for parallel scanning, then yields results
        in chronological order as they complete. This provides the parallelism
        benefits of the blocking approach while enabling incremental processing.
        """
        try:
            hour_dirs = list(self._generate_hour_directories())
            total_hours = len(hour_dirs)

            if total_hours == 0:
                self._batch_queue.put(
                    HourBatch(
                        hour_start_ns=self._stime_ns,
                        hour_end_ns=self._etime_ns,
                        files=[],
                        is_final=True,
                    )
                )
                self._state.is_complete = True
                return

            scan_func = partial(
                _scan_hour_directory,
                stime_ns=self._stime_ns,
                etime_ns=self._etime_ns,
            )

            # Submit ALL directories at once for maximum parallelism
            # Collect results keyed by hour for chronological ordering
            results_by_hour: dict[int, tuple[int, int, list[str]]] = {}
            next_hour_to_emit = 0  # Index of next hour to emit

            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                # Submit all directories
                future_to_idx: dict[Future, int] = {}
                for idx, (dir_path, _hour_start_ns, _hour_end_ns) in enumerate(hour_dirs):
                    if self._stop_event.is_set():
                        break
                    future = executor.submit(scan_func, dir_path)
                    future_to_idx[future] = idx

                # Process results as they complete
                for future in as_completed(future_to_idx):
                    if self._stop_event.is_set():
                        break

                    idx = future_to_idx[future]
                    dir_path, hour_start_ns, hour_end_ns = hour_dirs[idx]

                    try:
                        filepaths = future.result()
                    except Exception as e:
                        logger.warning("Error scanning %s: %s", dir_path, e)
                        filepaths = []

                    results_by_hour[idx] = (hour_start_ns, hour_end_ns, filepaths)

                    # Emit all consecutive hours that are ready
                    while next_hour_to_emit in results_by_hour:
                        if self._stop_event.is_set():
                            break

                        hour_start_ns, hour_end_ns, filepaths = results_by_hour.pop(
                            next_hour_to_emit
                        )

                        # Sort files by timestamp within this hour
                        files_with_ts = []
                        for fp in filepaths:
                            ts = extract_file_timestamp(fp)
                            if ts is not None:
                                ts_ns = ts.astype("datetime64[ns]").astype(np.int64)
                                files_with_ts.append((fp, ts_ns))

                        files_with_ts.sort(key=lambda x: x[1])

                        # Create DiscoveredFile objects with sequential IDs
                        discovered_files = []
                        for fp, ts_ns in files_with_ts:
                            discovered_files.append(
                                DiscoveredFile(
                                    filepath=fp,
                                    timestamp_ns=ts_ns,
                                    file_id=self._next_file_id,
                                )
                            )
                            self._next_file_id += 1

                        self._state.total_files_discovered += len(discovered_files)
                        self._state.current_hour_ns = hour_end_ns

                        is_final = next_hour_to_emit == total_hours - 1

                        hour_batch = HourBatch(
                            hour_start_ns=hour_start_ns,
                            hour_end_ns=hour_end_ns,
                            files=discovered_files,
                            is_final=is_final,
                        )

                        # This blocks if queue is full (backpressure)
                        self._batch_queue.put(hour_batch)
                        next_hour_to_emit += 1

            self._state.is_complete = True

        except Exception as e:
            logger.exception("Discovery worker error")
            self._state.error = str(e)
            self._state.is_complete = True
            # Signal completion even on error
            self._batch_queue.put(None)

    def start(self) -> None:
        """
        Start background file discovery.

        Call this before iterating over batches.
        """
        if self._started:
            raise RuntimeError("Discovery already started")

        self._started = True
        self._discovery_thread = threading.Thread(
            target=self._discovery_worker,
            name="StreamingFilenames-discovery",
            daemon=True,
        )
        self._discovery_thread.start()

    def stop(self) -> None:
        """
        Stop background discovery gracefully.

        Any batches already queued will still be available.
        """
        self._stop_event.set()
        if self._discovery_thread:
            self._discovery_thread.join(timeout=5.0)

    def iter_batches(self, timeout: float = 1.0) -> Iterator[HourBatch]:
        """
        Iterate over discovered file batches.

        Yields batches as they're discovered. Each batch contains all files
        from one hour directory, sorted by timestamp.

        Args:
            timeout: Seconds to wait for next batch before checking stop condition

        Yields:
            HourBatch objects containing discovered files
        """
        if not self._started:
            raise RuntimeError("Call start() before iterating")

        while True:
            try:
                batch = self._batch_queue.get(timeout=timeout)
                if batch is None:
                    # Discovery complete or error
                    break
                yield batch
                if batch.is_final:
                    break
            except Empty:
                # Check if discovery is complete
                if self._state.is_complete:
                    break
                # Check for stop signal
                if self._stop_event.is_set():
                    break

    def iter_files(self, timeout: float = 1.0) -> Iterator[DiscoveredFile]:
        """
        Iterate over discovered files (flattened view).

        Convenience method that yields individual files rather than batches.

        Args:
            timeout: Seconds to wait for next batch

        Yields:
            DiscoveredFile objects
        """
        for batch in self.iter_batches(timeout=timeout):
            yield from batch.files

    def __enter__(self) -> StreamingFilenames:
        """Context manager entry - starts discovery."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops discovery."""
        self.stop()


def create_time_windows_from_bounds(
    stime: np.datetime64,
    etime: np.datetime64,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
) -> list[tuple[np.datetime64, np.datetime64, int]]:
    """
    Create time windows based on time bounds (not file list).

    Unlike create_time_windows() which requires all files upfront,
    this creates windows purely from time bounds.

    Args:
        stime: Start time
        etime: End time
        window_seconds: Window duration in seconds
        overlap: Overlap fraction (0.5 = 50% overlap)

    Returns:
        List of (window_start, window_end, window_index) tuples
    """
    stime_ns = np.datetime64(stime, "ns")
    etime_ns = np.datetime64(etime, "ns")

    window_ns = np.timedelta64(int(window_seconds * 1e9), "ns")
    stride_ns = np.timedelta64(int(window_seconds * (1 - overlap) * 1e9), "ns")

    windows = []
    window_start = stime_ns
    window_idx = 0

    while window_start < etime_ns:
        window_end = window_start + window_ns
        windows.append((window_start, window_end, window_idx))
        window_start = window_start + stride_ns
        window_idx += 1

    return windows


@dataclass
class WindowTracker:
    """
    Tracks window state during streaming file discovery.

    Manages:
    - Which files belong to each window
    - Which windows are discovery-complete (no more files will arrive)
    - Which windows are ready for merging (all files processed)
    """

    windows: list[tuple[np.datetime64, np.datetime64, int]]
    min_frames_per_window: int = 3

    # Internal state
    _window_files: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    _window_discovery_complete: set[int] = field(default_factory=set)
    _files_processed: set[int] = field(default_factory=set)

    def __post_init__(self):
        # Convert window times to nanoseconds for fast comparison
        self._window_bounds_ns: list[tuple[int, int, int]] = []
        for start, end, idx in self.windows:
            start_ns = start.astype("datetime64[ns]").astype(np.int64)
            end_ns = end.astype("datetime64[ns]").astype(np.int64)
            self._window_bounds_ns.append((start_ns, end_ns, idx))

    def assign_file(self, file: DiscoveredFile) -> list[int]:
        """
        Assign a discovered file to all windows it belongs to.

        Args:
            file: The discovered file

        Returns:
            List of window indices this file was assigned to
        """
        assigned = []
        for start_ns, end_ns, window_idx in self._window_bounds_ns:
            if start_ns <= file.timestamp_ns < end_ns:
                self._window_files[window_idx].add(file.file_id)
                assigned.append(window_idx)
        return assigned

    def mark_discovery_complete_before(self, time_ns: int) -> list[int]:
        """
        Mark windows as discovery-complete if they end before given time.

        Args:
            time_ns: Nanoseconds since epoch

        Returns:
            List of window indices newly marked as discovery-complete
        """
        newly_complete = []
        for _start_ns, end_ns, window_idx in self._window_bounds_ns:
            if window_idx not in self._window_discovery_complete:
                if end_ns <= time_ns:
                    self._window_discovery_complete.add(window_idx)
                    newly_complete.append(window_idx)
        return newly_complete

    def mark_all_discovery_complete(self) -> list[int]:
        """
        Mark all windows as discovery-complete.

        Call when file discovery finishes.

        Returns:
            List of window indices newly marked as discovery-complete
        """
        newly_complete = []
        for _, _, window_idx in self._window_bounds_ns:
            if window_idx not in self._window_discovery_complete:
                self._window_discovery_complete.add(window_idx)
                newly_complete.append(window_idx)
        return newly_complete

    def mark_file_processed(self, file_id: int) -> None:
        """Mark a file as having completed processing."""
        self._files_processed.add(file_id)

    def get_ready_windows(self) -> list[int]:
        """
        Get windows that are ready for merging.

        A window is ready when:
        - Discovery is complete for that window
        - All assigned files have been processed
        - Window has minimum required frames

        Returns:
            List of window indices ready for merging
        """
        ready = []
        for window_idx in self._window_discovery_complete:
            files = self._window_files.get(window_idx, set())
            if len(files) < self.min_frames_per_window:
                continue
            if files.issubset(self._files_processed):
                ready.append(window_idx)
        return ready

    def get_window_files(self, window_idx: int) -> set[int]:
        """Get file IDs assigned to a window."""
        return self._window_files.get(window_idx, set())

    def is_discovery_complete(self, window_idx: int) -> bool:
        """Check if discovery is complete for a window."""
        return window_idx in self._window_discovery_complete

    @property
    def n_files_discovered(self) -> int:
        """Total number of unique files discovered."""
        all_files = set()
        for files in self._window_files.values():
            all_files.update(files)
        return len(all_files)


# CLI support for testing
def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import _directory_type, _timestamp_type

    parser.add_argument("stime", type=_timestamp_type, help="Start time")
    parser.add_argument("etime", type=_timestamp_type, help="End time")
    parser.add_argument("polar_path", type=_directory_type, help="Polar file directory")
    parser.add_argument("--workers", "-w", type=int, help="Number of workers")


def run(args) -> None:
    """Test streaming file discovery."""
    import time

    t0 = time.time()
    total_files = 0
    total_batches = 0

    with StreamingFilenames(
        args.stime, args.etime, args.polar_path, workers=args.workers
    ) as streaming:
        for batch in streaming.iter_batches():
            total_batches += 1
            total_files += len(batch.files)
            elapsed = time.time() - t0
            logger.info(
                "Batch %d: %d files, total %d, elapsed %.1fs",
                total_batches,
                len(batch.files),
                total_files,
                elapsed,
            )

    t1 = time.time()
    logger.info("Discovery complete: %d files in %.2fs", total_files, t1 - t0)


def add_subparser(subparsers) -> None:
    """Register the 'stream-list' subcommand."""
    p = subparsers.add_parser(
        "stream-list",
        help="Test streaming file discovery",
        description="Discover files incrementally and show progress.",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


if __name__ == "__main__":
    from wamos_tpw.cli_utils import create_standalone_main

    main = create_standalone_main(
        _add_arguments,
        run,
        "Test streaming file discovery",
    )
    main()
