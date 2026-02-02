#! /usr/bin/env python3
#
# Memory monitoring utilities for WAMOS pipelines
#
# Provides functions to track memory usage during processing.
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


def get_memory_mb() -> float:
    """
    Get current process memory usage in MB.

    Uses platform-specific methods for accuracy.

    Returns:
        Memory usage in megabytes
    """
    try:
        import resource

        # ru_maxrss is in bytes on Linux, KB on macOS
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Darwin":
            # macOS returns bytes
            return usage.ru_maxrss / (1024 * 1024)
        else:
            # Linux returns KB
            return usage.ru_maxrss / 1024
    except (ImportError, AttributeError):
        pass

    # Fallback to psutil if available
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    return 0.0


def get_peak_memory_mb() -> float:
    """
    Get peak memory usage in MB.

    Returns:
        Peak memory usage in megabytes
    """
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        if os.uname().sysname == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)
        else:
            return usage.ru_maxrss / 1024
    except (ImportError, AttributeError):
        return get_memory_mb()


@dataclass
class MemoryStats:
    """Statistics about memory usage during processing."""

    start_mb: float = 0.0
    peak_mb: float = 0.0
    end_mb: float = 0.0
    samples: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, memory_mb)

    @property
    def delta_mb(self) -> float:
        """Memory change from start to end."""
        return self.end_mb - self.start_mb

    @property
    def max_delta_mb(self) -> float:
        """Maximum memory increase from start."""
        return self.peak_mb - self.start_mb

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Memory: start={self.start_mb:.1f}MB, peak={self.peak_mb:.1f}MB, "
            f"end={self.end_mb:.1f}MB, delta={self.delta_mb:+.1f}MB"
        )


class MemoryMonitor:
    """
    Context manager for monitoring memory usage.

    Usage::

        with MemoryMonitor() as monitor:
            # Do work
            pass

        print(monitor.stats.summary())

    Or with periodic sampling::

        with MemoryMonitor(sample_interval=1.0) as monitor:
            # Do work - memory sampled every 1 second
            pass

        print(f"Peak memory: {monitor.stats.peak_mb:.1f}MB")
    """

    def __init__(
        self,
        sample_interval: float | None = None,
        on_sample: Callable[[float, float], None] | None = None,
    ):
        """
        Initialize memory monitor.

        Args:
            sample_interval: If set, sample memory at this interval (seconds)
            on_sample: Callback function called with (elapsed_time, memory_mb) on each sample
        """
        self._sample_interval = sample_interval
        self._on_sample = on_sample
        self._stats = MemoryStats()
        self._stop_event = threading.Event()
        self._sample_thread: threading.Thread | None = None
        self._start_time: float = 0.0

    @property
    def stats(self) -> MemoryStats:
        """Get memory statistics."""
        return self._stats

    def _sample_worker(self) -> None:
        """Background worker for periodic memory sampling."""
        while not self._stop_event.wait(self._sample_interval):
            current_mb = get_memory_mb()
            elapsed = time.perf_counter() - self._start_time
            self._stats.samples.append((elapsed, current_mb))
            self._stats.peak_mb = max(self._stats.peak_mb, current_mb)

            if self._on_sample:
                self._on_sample(elapsed, current_mb)

    def __enter__(self) -> "MemoryMonitor":
        """Start monitoring."""
        self._start_time = time.perf_counter()
        self._stats.start_mb = get_memory_mb()
        self._stats.peak_mb = self._stats.start_mb

        if self._sample_interval is not None:
            self._sample_thread = threading.Thread(
                target=self._sample_worker,
                name="MemoryMonitor",
                daemon=True,
            )
            self._sample_thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring and finalize stats."""
        if self._sample_thread is not None:
            self._stop_event.set()
            self._sample_thread.join(timeout=1.0)

        self._stats.end_mb = get_memory_mb()
        self._stats.peak_mb = max(self._stats.peak_mb, get_peak_memory_mb())

    def checkpoint(self, label: str = "") -> float:
        """
        Record a memory checkpoint and optionally log it.

        Args:
            label: Optional label for the checkpoint

        Returns:
            Current memory usage in MB
        """
        current_mb = get_memory_mb()
        self._stats.peak_mb = max(self._stats.peak_mb, current_mb)
        elapsed = time.perf_counter() - self._start_time

        if label:
            delta = current_mb - self._stats.start_mb
            logger.debug(
                "[Memory] %s: %.1fMB (delta: %+.1fMB, elapsed: %.1fs)",
                label,
                current_mb,
                delta,
                elapsed,
            )

        return current_mb


def log_memory_stats(stats: MemoryStats, prefix: str = "") -> None:
    """
    Log memory statistics.

    Args:
        stats: MemoryStats object
        prefix: Optional prefix for log message
    """
    msg = stats.summary()
    if prefix:
        msg = f"{prefix}: {msg}"
    logger.info(msg)


def format_memory_mb(mb: float) -> str:
    """
    Format memory value with appropriate units.

    Args:
        mb: Memory in megabytes

    Returns:
        Formatted string with units
    """
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    else:
        return f"{mb:.1f} MB"


# Convenience function for one-off memory checks
def log_current_memory(label: str = "Current memory") -> float:
    """
    Log current memory usage.

    Args:
        label: Label for the log message

    Returns:
        Current memory in MB
    """
    current_mb = get_memory_mb()
    logger.info("%s: %s", label, format_memory_mb(current_mb))
    return current_mb
