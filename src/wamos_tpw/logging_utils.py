#! /usr/bin/env python3
#
# Structured logging utilities for WAMOS processing
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""
Structured logging utilities for WAMOS radar processing.

This module provides utilities for consistent, structured logging across the
WAMOS pipeline. It supports both human-readable and JSON log formats.

Usage
-----

Basic setup::

    from wamos_tpw.logging_utils import setup_logging, get_logger

    # Setup logging for the application
    setup_logging(level="INFO", json_output=False)

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing started", extra={"n_files": 100})

JSON structured output::

    setup_logging(level="DEBUG", json_output=True)
    logger.info("Frame processed", extra={
        "filename": "file.pol",
        "n_frames": 10,
        "duration_ms": 123.4
    })
    # Output: {"timestamp": "...", "level": "INFO", "message": "Frame processed", ...}

Performance logging::

    from wamos_tpw.logging_utils import log_performance

    @log_performance("process_frame")
    def process_frame(frame):
        # ... processing ...
        return result

    # Or as context manager:
    with log_performance("loading"):
        data = load_file(path)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

__all__ = [
    "setup_logging",
    "get_logger",
    "log_performance",
    "StructuredFormatter",
    "JSONFormatter",
]


# =============================================================================
# Custom Formatters
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """
    Human-readable formatter with structured data support.

    Formats log messages with timestamp, level, logger name, and message.
    Extra fields are appended as key=value pairs.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base message
        timestamp = self.formatTime(record, self.datefmt)
        base = f"{timestamp} {record.levelname:8s} [{record.name}] {record.getMessage()}"

        # Add extra fields (exclude standard LogRecord attributes)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

        extra = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}

        if extra:
            extra_str = " ".join(f"{k}={v!r}" for k, v in sorted(extra.items()))
            base += f" | {extra_str}"

        return base


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging output.

    Outputs each log record as a single JSON line, suitable for log aggregation
    systems like ELK, Splunk, or CloudWatch.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Standard fields
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info for warnings and errors
        if record.levelno >= logging.WARNING:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

        extra = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        if extra:
            log_data["extra"] = extra

        return json.dumps(log_data, default=str)


# =============================================================================
# Setup Functions
# =============================================================================


def setup_logging(
    level: str | int = "INFO",
    json_output: bool = False,
    stream: Any = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, use JSON format; otherwise human-readable
        stream: Output stream (default: sys.stderr)
        datefmt: Date format string

    Example::

        # Human-readable output to stderr
        setup_logging(level="INFO")

        # JSON output for log aggregation
        setup_logging(level="DEBUG", json_output=True)
    """
    if stream is None:
        stream = sys.stderr

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Choose formatter
    if json_output:
        formatter = JSONFormatter(datefmt=datefmt)
    else:
        formatter = StructuredFormatter(datefmt=datefmt)

    # Configure root logger
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured Logger instance

    Example::

        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)


# =============================================================================
# Performance Logging
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


class log_performance:
    """
    Decorator and context manager for timing operations.

    Can be used as a decorator::

        @log_performance("process_frame")
        def process_frame(frame):
            return result

    Or as a context manager::

        with log_performance("loading"):
            data = load_file(path)

    Logs timing information at DEBUG level with structured data.
    """

    def __init__(
        self,
        name: str,
        logger: logging.Logger | None = None,
        level: int = logging.DEBUG,
    ) -> None:
        """
        Initialize performance logger.

        Args:
            name: Name of the operation being timed
            logger: Logger to use (default: wamos_tpw.performance)
            level: Log level for timing messages
        """
        self.name = name
        self.logger = logger or logging.getLogger("wamos_tpw.performance")
        self.level = level
        self.start_time: float = 0

    def __call__(self, func: F) -> F:
        """Decorator usage."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    def __enter__(self) -> "log_performance":
        """Context manager entry."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Context manager exit."""
        elapsed = time.perf_counter() - self.start_time
        elapsed_ms = elapsed * 1000

        self.logger.log(
            self.level,
            f"{self.name} completed",
            extra={
                "operation": self.name,
                "duration_ms": round(elapsed_ms, 2),
                "duration_s": round(elapsed, 4),
            },
        )


@contextmanager
def timed_section(name: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """
    Context manager that yields a dict with timing info.

    Useful when you want to capture timing without automatic logging.

    Example::

        with timed_section("processing") as timing:
            result = process_data(data)

        print(f"Took {timing['duration_ms']:.1f}ms")
    """
    timing: dict[str, float] = {}
    start = time.perf_counter()

    try:
        yield timing
    finally:
        elapsed = time.perf_counter() - start
        timing["duration_s"] = elapsed
        timing["duration_ms"] = elapsed * 1000

        if logger:
            logger.debug(
                f"{name} completed",
                extra={"operation": name, **timing},
            )
