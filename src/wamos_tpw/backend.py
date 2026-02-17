#! /usr/bin/env python3
#
# Backend abstraction for WAMOS pipeline
#
# Auto-detects PyTorch GPU availability (CUDA → MPS → CPU fallback)
# and Numba JIT availability.
#
# Environment variables:
#   WAMOS_NO_GPU=1   — force CPU-only mode (disable PyTorch GPU)
#   WAMOS_NO_NUMBA=1 — disable Numba JIT acceleration
#
# Feb-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "HAS_TORCH_GPU",
    "HAS_NUMBA",
    "get_device",
    "to_tensor",
    "to_numpy",
    "synchronize",
]

# ── PyTorch GPU detection ──

HAS_TORCH_GPU: bool = False
_device = None
_torch = None

if not os.environ.get("WAMOS_NO_GPU", ""):
    try:
        import torch as _torch_mod

        _torch = _torch_mod
        if _torch.cuda.is_available():
            _device = _torch.device("cuda")
            HAS_TORCH_GPU = True
            logger.info("GPU backend: CUDA (%s)", _torch.cuda.get_device_name(0))
        elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            _device = _torch.device("mps")
            HAS_TORCH_GPU = True
            logger.info("GPU backend: MPS (Apple Silicon)")
        else:
            _device = _torch.device("cpu")
            logger.info("GPU backend: PyTorch available but no GPU detected, using CPU")
    except ImportError:
        logger.debug("GPU backend: PyTorch not installed, using NumPy")
else:
    logger.info("GPU backend: disabled via WAMOS_NO_GPU")

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


def get_device():
    """Return the torch.device for GPU computation, or None if unavailable."""
    return _device


def to_tensor(
    array: np.ndarray,
    dtype=None,
) -> np.ndarray:
    """Convert a numpy array to a GPU tensor.

    If PyTorch GPU is unavailable, returns the array as float32 numpy.

    Args:
        array: Input numpy array.
        dtype: Optional torch dtype (default: float32).

    Returns:
        A torch.Tensor on GPU, or a numpy array if no GPU.
    """
    if not HAS_TORCH_GPU:
        return array.astype(np.float32) if array.dtype != np.float32 else array
    if dtype is None:
        dtype = _torch.float32
    return _torch.from_numpy(np.ascontiguousarray(array)).to(dtype=dtype, device=_device)


def to_numpy(tensor) -> np.ndarray:
    """Convert a GPU tensor back to a numpy array.

    If the input is already a numpy array, returns it unchanged.

    Args:
        tensor: A torch.Tensor or numpy array.

    Returns:
        A numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def synchronize() -> None:
    """Synchronize the GPU device (wait for all kernels to complete).

    No-op if no GPU is available.
    """
    if not HAS_TORCH_GPU:
        return
    if _device is not None and _device.type == "cuda":
        _torch.cuda.synchronize()
    elif _device is not None and _device.type == "mps":
        _torch.mps.synchronize()
