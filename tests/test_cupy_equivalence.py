#! /usr/bin/env python3
"""CPU/CuPy numerical equivalence tests for CuPy-accelerated modules.

These tests verify that CuPy paths produce the same results as CPU paths.
All tests are skipped when CuPy GPU is not available.

Note: CuPy is only used for grid projection and hard return sweeps (hybrid
approach). Pipeline steps (destreak, deramp, bearing) run on CPU only.
"""

from __future__ import annotations

import numpy as np
import pytest

from wamos_tpw.backend import HAS_CUPY_GPU

pytestmark = pytest.mark.gpu

requires_cupy = pytest.mark.skipif(not HAS_CUPY_GPU, reason="No CuPy GPU available")


@requires_cupy
class TestGridProjectionCuPyEquivalence:
    """Verify CuPy grid projection matches CPU."""

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
        from wamos_tpw.grid import _project_frame_cupy, _project_frame_numpy

        np.random.seed(42)
        n_bearings, n_distances = 360, 512
        intensity = np.random.rand(n_bearings, n_distances).astype(np.float32) * 4095
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
        sum_cupy, count_cupy = _project_frame_cupy(
            intensity, theta, ground_range, latitudes, longitudes, headings, gp
        )

        # Counts should be identical
        np.testing.assert_array_equal(count_cpu, count_cupy)

        # Sums should be very close (float32 vs float64 accumulation differences)
        mask = count_cpu > 0
        if mask.any():
            np.testing.assert_allclose(sum_cupy[mask], sum_cpu[mask], rtol=1e-3, atol=1e-1)
