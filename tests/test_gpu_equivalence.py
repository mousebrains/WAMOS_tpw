#! /usr/bin/env python3
"""CPU/GPU numerical equivalence tests for GPU-accelerated modules.

These tests verify that GPU paths produce the same results as CPU paths.
All tests are skipped when no PyTorch GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from wamos_tpw.backend import HAS_TORCH_GPU

pytestmark = pytest.mark.gpu

requires_gpu = pytest.mark.skipif(not HAS_TORCH_GPU, reason="No PyTorch GPU available")


@requires_gpu
class TestGridProjectionEquivalence:
    """Verify GPU grid projection matches CPU."""

    def _make_grid_params(self):
        from wamos_tpw.grid import GridParams

        n_x, n_y = 200, 200
        spacing = 10.0
        x_edges = np.linspace(0, n_x * spacing, n_x + 1)
        y_edges = np.linspace(0, n_y * spacing, n_y + 1)
        return GridParams(
            x_edges=x_edges - x_edges.mean(),
            y_edges=y_edges - y_edges.mean(),
            x_edges_abs=x_edges,
            y_edges_abs=y_edges,
            grid_spacing=spacing,
            utm_zone=10,
            hemisphere="north",
            center_lat=45.0,
            center_lon=-122.0,
            ref_lat=45.0,
            ref_lon=-122.0,
            m_per_deg_lon=78846.0,
            n_x=n_x,
            n_y=n_y,
        )

    def test_project_frame_equivalence(self):
        # Import the torch version directly
        from wamos_tpw.grid import _project_frame_numpy, _project_frame_torch

        np.random.seed(42)
        n_bearings, n_distances = 360, 512
        intensity = np.random.rand(n_bearings, n_distances).astype(np.float32) * 4095
        # Sprinkle some NaNs
        intensity[0, :10] = np.nan
        intensity[100, 200:250] = np.nan

        theta = np.linspace(0, 359, n_bearings)
        ground_range = np.linspace(50, 5000, n_distances)
        latitudes = np.full(n_bearings, 45.0) + np.random.randn(n_bearings) * 1e-5
        longitudes = np.full(n_bearings, -122.0) + np.random.randn(n_bearings) * 1e-5
        headings = np.full(n_bearings, 90.0) + np.random.randn(n_bearings) * 0.5

        gp = self._make_grid_params()

        sum_cpu, count_cpu = _project_frame_numpy(
            intensity, theta, ground_range, latitudes, longitudes, headings, gp
        )
        sum_gpu, count_gpu = _project_frame_torch(
            intensity, theta, ground_range, latitudes, longitudes, headings, gp
        )

        # Counts should be identical
        np.testing.assert_array_equal(count_cpu, count_gpu)

        # Sums should be very close (float32 vs float64 accumulation differences)
        mask = count_cpu > 0
        if mask.any():
            np.testing.assert_allclose(sum_gpu[mask], sum_cpu[mask], rtol=1e-3, atol=1e-1)


@requires_gpu
class TestDestreakEquivalence:
    """Verify GPU destreak convolution+threshold matches CPU."""

    def test_convolve_and_threshold(self):
        from wamos_tpw.destreak import (
            _convolve_and_threshold_cpu,
            _convolve_and_threshold_gpu,
        )

        np.random.seed(42)
        n_bearings, n_distances = 360, 512
        intensity = np.random.rand(n_bearings, n_distances).astype(np.float32) * 4095

        # Add a fake streak (one row much brighter than neighbors)
        intensity[180, :] = 4000.0
        intensity[179, :] = 500.0
        intensity[181, :] = 500.0

        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
        kAdj = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        kAdj = kAdj / kAdj.sum()
        sigma = 3.0

        timing_cpu: dict[str, float] = {}
        timing_gpu: dict[str, float] = {}
        q_cpu, b_cpu = _convolve_and_threshold_cpu(intensity, kernel, kAdj, sigma, timing_cpu)
        q_gpu, b_gpu = _convolve_and_threshold_gpu(intensity, kernel, kAdj, sigma, timing_gpu)

        # Boolean mask should match closely (minor float differences at boundaries)
        # Allow up to 1% difference in mask
        diff_pct = np.abs(q_cpu.astype(int) - q_gpu.astype(int)).sum() / q_cpu.size * 100
        assert diff_pct < 1.0, f"Mask differs by {diff_pct:.2f}%"

        # Replacement values should be close where both masks are True.
        # Small differences at padding boundaries (cv2 wrap vs torch circular) are expected.
        both_true = q_cpu & q_gpu
        if both_true.any():
            mismatch = np.abs(b_gpu[both_true] - b_cpu[both_true])
            pct_close = (mismatch < 2.0).sum() / len(mismatch) * 100
            assert pct_close > 99.0, f"Only {pct_close:.1f}% of replacement values within tolerance"


@requires_gpu
class TestDerampEquivalence:
    """Verify GPU deramp matches CPU."""

    def test_deramp_equivalence(self):
        from wamos_tpw.deramp import _deramp_cpu, _deramp_gpu

        np.random.seed(42)
        n_bearings, n_distances = 360, 512
        slant = np.linspace(100, 5000, n_distances)
        x = 1.0 / slant

        # Create intensity with range-dependent falloff + noise
        falloff = 1000.0 / slant[np.newaxis, :]
        noise = np.random.rand(n_bearings, n_distances).astype(np.float32) * 50
        intensity = (falloff + noise).astype(np.float32)

        # Add some NaNs (shadow mask)
        intensity[10:20, :] = np.nan

        result_cpu, poly_cpu = _deramp_cpu(intensity.copy(), x, order=4, copy=False)
        result_gpu, poly_gpu = _deramp_gpu(intensity.copy(), x, order=4, copy=False)

        # Compare non-NaN values
        valid = ~np.isnan(result_cpu)
        assert valid.any()
        np.testing.assert_allclose(result_gpu[valid], result_cpu[valid], rtol=1e-3, atol=0.5)


@requires_gpu
class TestBearingEquivalence:
    """Verify GPU heading_to_xy matches CPU."""

    def test_heading_to_xy(self):
        from wamos_tpw.bearing import _heading_to_xy_gpu

        # Temporarily use CPU path for reference
        np.random.seed(42)
        heading = np.linspace(0, 359, 360).astype(np.float64)
        ground_range = np.linspace(50, 5000, 512).astype(np.float64)

        # CPU path
        heading_rad = np.deg2rad(heading)
        heading_2d = heading_rad[:, np.newaxis]
        range_2d = ground_range[np.newaxis, :]
        x_cpu = range_2d * np.sin(heading_2d)
        y_cpu = range_2d * np.cos(heading_2d)

        # GPU path
        x_gpu, y_gpu = _heading_to_xy_gpu(heading, ground_range)

        np.testing.assert_allclose(x_gpu, x_cpu, rtol=1e-5, atol=1e-2)
        np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-5, atol=1e-2)
