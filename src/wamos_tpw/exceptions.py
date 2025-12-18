#! /usr/bin/env python3
#
# Custom exceptions for WAMOS processing
#
# Dec-2025, Pat Welch, pat@mousebrains.com

"""
Custom exception hierarchy for the wamos_tpw package.

Exception Hierarchy:
    WamosError (base)
    ├── PolarFileError - File parsing and loading errors
    ├── ConfigError - Configuration validation errors
    ├── ProcessingError - Data processing errors
    └── ValidationError - Input validation errors
"""

from __future__ import annotations


class WamosError(Exception):
    """
    Base exception for all WAMOS-related errors.

    All custom exceptions in the wamos_tpw package inherit from this class,
    allowing callers to catch all WAMOS errors with a single except clause.

    Example:
        >>> try:
        ...     pf = PolarFile('nonexistent.pol')
        ... except WamosError as e:
        ...     print(f"WAMOS error: {e}")
    """
    pass


class PolarFileError(WamosError):
    """
    Exception raised for errors in polar file parsing or loading.

    This includes:
    - File not found or unreadable
    - Invalid file format (missing header, malformed data)
    - Unsupported compression format
    - Corrupted data blocks

    Attributes:
        filename: Path to the problematic file (if available)
        message: Human-readable error description

    Example:
        >>> raise PolarFileError("Invalid header format", filename="data.pol")
    """

    def __init__(self, message: str, filename: str | None = None):
        self.filename = filename
        self.message = message
        if filename:
            super().__init__(f"{filename}: {message}")
        else:
            super().__init__(message)


class ConfigError(WamosError):
    """
    Exception raised for configuration validation errors.

    This includes:
    - Invalid parameter values (out of range, wrong type)
    - Missing required configuration
    - Invalid YAML syntax
    - Conflicting settings

    Attributes:
        parameter: Name of the invalid parameter (if applicable)
        value: The invalid value (if applicable)
        message: Human-readable error description

    Example:
        >>> raise ConfigError("must be positive", parameter="radar.height", value=-5.0)
    """

    def __init__(self, message: str, parameter: str | None = None, value: object = None):
        self.parameter = parameter
        self.value = value
        self.message = message
        if parameter is not None:
            if value is not None:
                super().__init__(f"{parameter}={value!r}: {message}")
            else:
                super().__init__(f"{parameter}: {message}")
        else:
            super().__init__(message)


class ProcessingError(WamosError):
    """
    Exception raised for data processing errors.

    This includes:
    - Insufficient data for processing
    - Algorithm convergence failures
    - Invalid intermediate results

    Attributes:
        stage: Processing stage where error occurred (e.g., "deramp", "destreak")
        message: Human-readable error description

    Example:
        >>> raise ProcessingError("No valid bearings detected", stage="theta")
    """

    def __init__(self, message: str, stage: str | None = None):
        self.stage = stage
        self.message = message
        if stage:
            super().__init__(f"[{stage}] {message}")
        else:
            super().__init__(message)


class ValidationError(WamosError):
    """
    Exception raised for input validation errors.

    This includes:
    - Invalid time ranges
    - Path does not exist
    - Invalid parameter combinations
    - Data integrity errors

    Attributes:
        parameter: Name of the invalid parameter
        value: The invalid value (if applicable)
        message: Human-readable error description

    Example:
        >>> raise ValidationError("Start time must be before end time", parameter="stime")
        >>> raise ValidationError("must be uint16", parameter="data", value="float64")
    """

    def __init__(self, message: str, parameter: str | None = None, value: object = None):
        self.parameter = parameter
        self.value = value
        self.message = message
        if parameter is not None:
            if value is not None:
                super().__init__(f"{parameter}={value!r}: {message}")
            else:
                super().__init__(f"{parameter}: {message}")
        else:
            super().__init__(message)
