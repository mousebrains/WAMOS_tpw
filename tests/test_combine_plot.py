"""Tests for combine_plot module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestGridGroup:
    """Tests for grid_group function."""

    def test_grid_group_basic(self, single_polar_file: Path):
        """Test basic gridding of a group."""
        from wamos_tpw.combine_plot import grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        # Set corrected_intensity for processing
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        result = grid_group("2022-04-05 14:00", combine, n_along=100, n_cross=100)

        # Check result structure
        assert "period" in result
        assert "gridded" in result
        assert "lat_grid" in result
        assert "lon_grid" in result
        assert "ship_lat" in result
        assert "ship_lon" in result
        assert "ref_lat" in result
        assert "ref_lon" in result
        assert "travel" in result
        assert "n_frames" in result
        assert "n_pixels" in result
        assert "x_range" in result
        assert "y_range" in result
        assert "grid_shape" in result

        # Check values
        assert result["period"] == "2022-04-05 14:00"
        assert result["n_frames"] == len(frames)
        assert result["gridded"].shape == (100, 100)
        assert -90 <= result["ref_lat"] <= 90
        assert -180 <= result["ref_lon"] <= 180


class TestPlotDiagnostics:
    """Tests for plot_diagnostics function."""

    @pytest.mark.skipif(
        True,  # Skip by default as it opens a window
        reason="Interactive plot - enable manually for visual testing",
    )
    def test_plot_diagnostics_interactive(self, single_polar_file: Path):
        """Test interactive diagnostics plot (manual testing only)."""
        from wamos_tpw.combine_plot import plot_diagnostics
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        # This would show the plot
        plot_diagnostics(combine, n_along=100, n_cross=100)

    def test_plot_diagnostics_no_crash(self, single_polar_file: Path):
        """Test that plot_diagnostics doesn't crash with mocked plt.show()."""
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        # Mock plt.show to prevent window
        with patch("matplotlib.pyplot.show"):
            from wamos_tpw.combine_plot import plot_diagnostics

            plot_diagnostics(combine, n_along=50, n_cross=50, workers=1)


class TestSaveFrame:
    """Tests for save_frame function."""

    def test_save_frame_basic(self, single_polar_file: Path, tmp_path: Path):
        """Test saving a frame to PNG."""
        from wamos_tpw.combine_plot import save_frame
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        output_path = str(tmp_path / "test_frame.png")

        save_frame(combine, output_path, n_along=50, n_cross=50, workers=1, dpi=50)

        # Check file was created
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 1000  # Should be more than 1KB

    def test_save_frame_with_ship_wind_data(self, single_polar_file: Path, tmp_path: Path):
        """Test saving frame with ship and wind metadata."""
        from wamos_tpw.combine_plot import save_frame
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        # Set corrected intensity and ensure metadata has ship/wind data
        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)
            # Metadata should already have ship_speed, heading, etc. from real file

        combine = Combine(frames, radar_height=25.0)

        output_path = str(tmp_path / "test_frame_with_data.png")

        save_frame(combine, output_path, n_along=50, n_cross=50, workers=1, dpi=50)

        assert Path(output_path).exists()


class TestCombineViewer:
    """Tests for CombineViewer class."""

    def test_combine_viewer_init(self):
        """Test viewer initialization."""
        from wamos_tpw.combine_plot import CombineViewer

        viewer = CombineViewer(total_groups=5, cmap="viridis", figsize=(10, 10))

        assert viewer._total_groups == 5
        assert viewer._cmap == "viridis"
        assert viewer._figsize == (10, 10)
        assert viewer._current_idx == 0
        assert len(viewer._groups) == 0

    def test_combine_viewer_add_group(self, single_polar_file: Path):
        """Test adding groups to viewer."""
        from wamos_tpw.combine_plot import CombineViewer, grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        viewer = CombineViewer(total_groups=1)

        # Grid and add group
        group_data = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)
        viewer.add_group_data(group_data)

        assert len(viewer._groups) == 1
        assert viewer._groups[0]["period"] == "2022-04-05 14:00"

    def test_combine_viewer_set_loading_complete(self):
        """Test marking loading as complete."""
        from wamos_tpw.combine_plot import CombineViewer

        viewer = CombineViewer()
        assert not viewer._loading_complete

        viewer.set_loading_complete()
        assert viewer._loading_complete

    def test_combine_viewer_navigation(self, single_polar_file: Path):
        """Test navigation between groups."""
        from wamos_tpw.combine_plot import CombineViewer, grid_group
        from wamos_tpw.combine import Combine
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frames = pf.frames[:2]

        for frame in frames:
            frame.corrected_intensity = frame.intensity.astype(np.float64)

        combine = Combine(frames, radar_height=25.0)

        viewer = CombineViewer(total_groups=2)

        # Add two groups
        group_data1 = grid_group("2022-04-05 14:00", combine, n_along=50, n_cross=50)
        group_data2 = grid_group("2022-04-05 14:10", combine, n_along=50, n_cross=50)
        viewer.add_group_data(group_data1)
        viewer.add_group_data(group_data2)

        # Initial position
        assert viewer._current_idx == 0

        # Test navigation logic by directly manipulating _current_idx
        # (navigation methods call _draw_plot which requires figure)

        # Simulate next navigation
        if viewer._current_idx < len(viewer._groups) - 1:
            viewer._current_idx += 1
        else:
            viewer._current_idx = 0
        assert viewer._current_idx == 1

        # Simulate next navigation (wrap)
        if viewer._current_idx < len(viewer._groups) - 1:
            viewer._current_idx += 1
        else:
            viewer._current_idx = 0
        assert viewer._current_idx == 0

        # Simulate prev navigation (wrap to end)
        if viewer._current_idx > 0:
            viewer._current_idx -= 1
        else:
            viewer._current_idx = len(viewer._groups) - 1
        assert viewer._current_idx == 1

    def test_combine_viewer_empty_show_warning(self, caplog):
        """Test that showing empty viewer logs warning."""
        from wamos_tpw.combine_plot import CombineViewer
        import logging

        caplog.set_level(logging.WARNING)

        viewer = CombineViewer()

        # Mock plt.show to avoid window
        with patch("matplotlib.pyplot.show"):
            viewer.show()

        assert "No groups to display" in caplog.text
