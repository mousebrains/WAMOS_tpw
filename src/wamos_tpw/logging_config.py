"""
Logging configuration for WAMOS CLI tools.

Provides centralized logging setup with support for:
- Console output (without timestamps for readability)
- File output with rotation (with timestamps for log analysis)
"""

import logging
from argparse import ArgumentParser, Namespace
from logging.handlers import RotatingFileHandler
from typing import Optional


def add_logging_arguments(parser: ArgumentParser) -> None:
    """
    Add logging-related arguments to an argument parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output (DEBUG level)"
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Log file path (enables rotating file logging)"
    )
    parser.add_argument(
        "--log-max-mb",
        type=int,
        default=100,
        help="Maximum log file size in MB before rotation (default: 100)",
    )


def setup_logging(
    args: Optional[Namespace] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
    log_max_mb: int = 100,
) -> logging.Logger:
    """
    Configure logging for the application.

    Console output uses a simple format without timestamps.
    File output includes timestamps for log analysis.

    Default level is INFO. With --verbose, level is DEBUG.

    Args:
        args: Namespace with verbose, log_file, log_max_mb attributes
        verbose: Enable DEBUG level (overridden by args if provided)
        log_file: Path to log file (overridden by args if provided)
        log_max_mb: Max file size in MB (overridden by args if provided)

    Returns:
        The root logger
    """
    # Extract values from args if provided
    if args is not None:
        verbose = getattr(args, "verbose", verbose)
        log_file = getattr(args, "log_file", log_file)
        log_max_mb = getattr(args, "log_max_mb", log_max_mb)

    # Determine log level: INFO by default, DEBUG with --verbose
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler - no timestamp for cleaner output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    if verbose:
        console_format = "%(name)s %(levelname)s %(message)s"
    else:
        console_format = "%(levelname)s %(message)s"
    console_handler.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console_handler)

    # File handler - includes timestamp for log analysis
    if log_file:
        max_bytes = log_max_mb * 1024 * 1024
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=5)
        file_handler.setLevel(level)
        file_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file} (max {log_max_mb}MB)")

    return logger
