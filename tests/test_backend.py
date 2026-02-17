#! /usr/bin/env python3
"""Tests for the GPU backend abstraction layer."""

from __future__ import annotations

import os

import numpy as np
import pytest


class TestBackendImport:
    """Test backend module imports and attribute availability."""

    def test_has_torch_gpu_is_bool(self):
        from wamos_tpw.backend import HAS_TORCH_GPU

        assert isinstance(HAS_TORCH_GPU, bool)

    def test_get_device_callable(self):
        from wamos_tpw.backend import get_device

        dev = get_device()
        # Returns None when no GPU, or a torch.device otherwise
        if dev is not None:
            assert hasattr(dev, "type")

    def test_to_tensor_returns_array_or_tensor(self):
        from wamos_tpw.backend import HAS_TORCH_GPU, to_tensor

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_tensor(arr)
        if HAS_TORCH_GPU:
            import torch

            assert isinstance(result, torch.Tensor)
        else:
            assert isinstance(result, np.ndarray)

    def test_to_numpy_identity_for_ndarray(self):
        from wamos_tpw.backend import to_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr

    def test_synchronize_no_error(self):
        from wamos_tpw.backend import synchronize

        synchronize()  # Should not raise


@pytest.mark.gpu
class TestBackendGPU:
    """Tests that require a PyTorch GPU."""

    @pytest.fixture(autouse=True)
    def _require_gpu(self):
        from wamos_tpw.backend import HAS_TORCH_GPU

        if not HAS_TORCH_GPU:
            pytest.skip("No PyTorch GPU available")

    def test_device_type(self):
        from wamos_tpw.backend import get_device

        dev = get_device()
        assert dev is not None
        assert dev.type in ("cuda", "mps")

    def test_tensor_roundtrip(self):
        from wamos_tpw.backend import to_numpy, to_tensor

        arr = np.random.rand(100, 200).astype(np.float32)
        tensor = to_tensor(arr)
        result = to_numpy(tensor)
        np.testing.assert_array_almost_equal(result, arr, decimal=6)

    def test_tensor_dtype(self):
        import torch

        from wamos_tpw.backend import get_device, to_tensor

        arr = np.array([1, 2, 3], dtype=np.int32)
        tensor = to_tensor(arr, dtype=torch.float64)
        assert tensor.dtype == torch.float64
        assert tensor.device.type == get_device().type

    def test_tensor_2d(self):
        from wamos_tpw.backend import to_numpy, to_tensor

        arr = np.random.rand(50, 100).astype(np.float32)
        tensor = to_tensor(arr)
        result = to_numpy(tensor)
        assert result.shape == (50, 100)
        np.testing.assert_array_almost_equal(result, arr, decimal=6)


class TestBackendNoGPU:
    """Test behavior when GPU is disabled via environment variable."""

    def test_wamos_no_gpu_env(self, monkeypatch):
        """Verify WAMOS_NO_GPU=1 forces CPU mode."""
        monkeypatch.setenv("WAMOS_NO_GPU", "1")
        # Import in subprocess to test fresh module state
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from wamos_tpw.backend import HAS_TORCH_GPU; print(HAS_TORCH_GPU)",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "WAMOS_NO_GPU": "1"},
        )
        assert result.stdout.strip() == "False"
