#! /usr/bin/env python3
#
# Backend abstraction for WAMOS pipeline
#
# Auto-detects CuPy GPU and Numba JIT availability.
# Priority: CuPy > Numba > NumPy.
#
# Environment variables:
#   WAMOS_NO_CUPY=1  — disable CuPy GPU acceleration
#   WAMOS_NO_NUMBA=1 — disable Numba JIT acceleration
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "HAS_CUPY_GPU",
    "HAS_NUMBA",
    "to_cupy",
    "from_cupy",
    "cupy_synchronize",
]

# ── CuPy GPU detection ──

HAS_CUPY_GPU: bool = False
_cupy = None

if not os.environ.get("WAMOS_NO_CUPY", ""):
    try:
        import cupy as _cupy_mod

        # Only check importability — do NOT call any CUDA runtime functions
        # (ones(), getDeviceCount(), etc.) because they create a CUDA context
        # that cannot survive multiprocessing fork(). The CUDA context will be
        # created lazily on first GPU operation in each worker process.
        _cupy = _cupy_mod
        HAS_CUPY_GPU = True
        logger.info("CuPy backend: available (version %s)", _cupy.__version__)
    except Exception:
        logger.debug("CuPy backend: not available")
else:
    logger.info("CuPy backend: disabled via WAMOS_NO_CUPY")

# ── Numba JIT detection ──

HAS_NUMBA: bool = False

if not os.environ.get("WAMOS_NO_NUMBA", ""):
    try:
        import numba as _numba_mod  # noqa: F401

        HAS_NUMBA = True
        logger.info("Numba JIT: available (version %s)", _numba_mod.__version__)
    except ImportError:
        logger.debug("Numba JIT: not installed")
else:
    logger.info("Numba JIT: disabled via WAMOS_NO_NUMBA")


# ── CuPy helpers ──


def to_cupy(array: np.ndarray) -> np.ndarray:
    """Convert a numpy array to a CuPy array on GPU.

    If CuPy is unavailable, returns the array as float32 numpy.

    Args:
        array: Input numpy array.

    Returns:
        A cupy.ndarray on GPU, or a numpy float32 array if no CuPy.
    """
    if not HAS_CUPY_GPU:
        return array.astype(np.float32) if array.dtype != np.float32 else array
    return _cupy.asarray(array, dtype=np.float32)


def from_cupy(array) -> np.ndarray:
    """Convert a CuPy array back to a numpy array.

    If the input is already a numpy array, returns it unchanged.

    Args:
        array: A cupy.ndarray or numpy array.

    Returns:
        A numpy array.
    """
    if isinstance(array, np.ndarray):
        return array
    return _cupy.asnumpy(array)


def cupy_synchronize() -> None:
    """Synchronize the CuPy CUDA device.

    No-op if CuPy is not available.
    """
    if not HAS_CUPY_GPU:
        return
    _cupy.cuda.Device(0).synchronize()
