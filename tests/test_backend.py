#! /usr/bin/env python3
"""Tests for the GPU backend abstraction layer."""

from __future__ import annotations

import os

import numpy as np
import pytest


class TestBackendImport:
    """Test backend module imports and attribute availability."""

    def test_has_cupy_gpu_is_bool(self):
        from wamos_tpw.backend import HAS_CUPY_GPU

        assert isinstance(HAS_CUPY_GPU, bool)

    def test_has_numba_is_bool(self):
        from wamos_tpw.backend import HAS_NUMBA

        assert isinstance(HAS_NUMBA, bool)

    def test_to_cupy_returns_array(self):
        from wamos_tpw.backend import HAS_CUPY_GPU, to_cupy

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_cupy(arr)
        if HAS_CUPY_GPU:
            import cupy

            assert isinstance(result, cupy.ndarray)
        else:
            assert isinstance(result, np.ndarray)

    def test_from_cupy_identity_for_ndarray(self):
        from wamos_tpw.backend import from_cupy

        arr = np.array([1.0, 2.0, 3.0])
        result = from_cupy(arr)
        assert result is arr

    def test_cupy_synchronize_no_error(self):
        from wamos_tpw.backend import cupy_synchronize

        cupy_synchronize()  # Should not raise


@pytest.mark.gpu
class TestBackendCuPy:
    """Tests that require a CuPy GPU."""

    @pytest.fixture(autouse=True)
    def _require_cupy(self):
        from wamos_tpw.backend import HAS_CUPY_GPU

        if not HAS_CUPY_GPU:
            pytest.skip("No CuPy GPU available")

    def test_cupy_roundtrip(self):
        from wamos_tpw.backend import from_cupy, to_cupy

        arr = np.random.rand(100, 200).astype(np.float32)
        gpu_arr = to_cupy(arr)
        result = from_cupy(gpu_arr)
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

    def test_cupy_synchronize(self):
        from wamos_tpw.backend import cupy_synchronize

        cupy_synchronize()  # Should not raise


class TestBackendNoCuPy:
    """Test behavior when CuPy is disabled via environment variable."""

    def test_wamos_no_cupy_env(self):
        """Verify WAMOS_NO_CUPY=1 sets HAS_CUPY_GPU=False."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from wamos_tpw.backend import HAS_CUPY_GPU; print(HAS_CUPY_GPU)",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "WAMOS_NO_CUPY": "1"},
        )
        assert result.stdout.strip() == "False"
