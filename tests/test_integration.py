"""Integration tests for the full WAMOS processing pipeline.

These tests exercise the complete data flow:
Filenames -> PolarFile -> Frame -> Theta/Bearing -> Deramp -> Destreak -> Combine
"""

import tempfile
from pathlib import Path

import numpy as np


class TestFullPipeline:
    """Integration tests for the complete processing pipeline."""

    def test_full_pipeline_april_data(self, april_polar_files: list[Path], test_data_dir: Path):
        """Test complete pipeline with April test data."""
        from wamos_tpw.bearing import Bearing, Theta
        from wamos_tpw.combine import Combine
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.deramp import Deramp
        from wamos_tpw.destreak import Destreak
        from wamos_tpw.filenames import Filenames
        from wamos_tpw.polarfile import PolarFile

        # Step 1: File discovery
        filenames = Filenames(
            stime="2022-04-05 14:00",
            etime="2022-04-05 15:00",
            polar_path=test_data_dir,
        )
        assert len(filenames) >= 1, "Should find at least one file"

        # Step 2: Load polar files and extract frames
        frames = []
        for filepath in filenames.files[:3]:  # Limit to 3 frames for speed
            pf = PolarFile(filepath)
            frames.extend(pf.frames)

        assert len(frames) >= 1, "Should have at least one frame"

        # Step 3: Calculate theta and bearing
        config = WamosConfig()
        theta = Theta(frames, config, refine=True)

        assert theta.bearing is not None
        assert len(theta.bearing) > 0
        assert np.all(theta.bearing >= 0) and np.all(theta.bearing < 360)

        bearing = Bearing(theta, radar_height=25.0)

        # Step 4: Test coordinate transformations
        x_ship, y_ship = bearing.xy_ship(0)
        x_earth, y_earth = bearing.xy_earth(0)

        assert x_ship.shape == frames[0].intensity.shape
        assert y_ship.shape == frames[0].intensity.shape
        assert x_earth.shape == frames[0].intensity.shape
        assert y_earth.shape == frames[0].intensity.shape

        # Step 5: Apply deramp and destreak
        for i, frame in enumerate(frames):
            bearing_arr = theta.bearing_for_frame(i)

            # Deramp
            deramp = Deramp(frame, config, bearing=bearing_arr)
            frame.deramped_intensity = deramp.corrected_intensity

            assert frame.deramped_intensity is not None
            assert frame.deramped_intensity.shape == frame.intensity.shape

            # Destreak (circular, no neighbors needed)
            destreak = Destreak(None, frame, None, config)
            frame.corrected_intensity = destreak.corrected_intensity

            assert frame.corrected_intensity is not None
            assert frame.corrected_intensity.shape == frame.intensity.shape

        # Step 6: Combine into earth coordinates
        combine = Combine(
            frames,
            config=config,
            theta=theta,
            radar_height=25.0,
            cache_coordinates=False,
        )

        # Verify combine object is properly constructed
        assert len(combine) == len(frames)
        assert combine.reference_position is not None
        ref_lat, ref_lon = combine.reference_position
        assert -90 <= ref_lat <= 90
        assert -180 <= ref_lon <= 180

        # Step 7: Save to file (this tests the gridding)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            combine.save_frame(output_path)
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_full_pipeline_march_data(self, march_polar_files: list[Path], test_data_dir: Path):
        """Test complete pipeline with March test data (different compression)."""
        from wamos_tpw.bearing import Bearing, Theta
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.deramp import Deramp
        from wamos_tpw.destreak import Destreak
        from wamos_tpw.polarfile import PolarFile

        # Load frames from March data (lzma compressed)
        frames = []
        for filepath in march_polar_files[:2]:  # Limit for speed
            pf = PolarFile(filepath)
            frames.extend(pf.frames)

        assert len(frames) >= 1

        # Process pipeline
        config = WamosConfig()
        theta = Theta(frames, config, refine=False)  # Skip refinement for speed
        bearing = Bearing(theta)

        # Verify all frames process correctly
        for i, frame in enumerate(frames):
            bearing_arr = theta.bearing_for_frame(i)

            deramp = Deramp(frame, config, bearing=bearing_arr)
            frame.deramped_intensity = deramp.corrected_intensity

            destreak = Destreak(None, frame, None, config)
            frame.corrected_intensity = destreak.corrected_intensity

            # Verify coordinate calculations work
            x_ship, y_ship = bearing.xy_ship(i)
            x_earth, y_earth = bearing.xy_earth(i)

            assert x_ship.shape == frame.intensity.shape
            assert x_earth.shape == frame.intensity.shape

    def test_metadata_only_loading(self, april_polar_files: list[Path]):
        """Test metadata-only loading for memory-efficient grid bounds calculation."""
        from wamos_tpw.polarfile import PolarFile

        filepath = april_polar_files[0]

        # Load with metadata only
        pf_meta = PolarFile(filepath, metadata_only=True)

        # Verify header is available
        assert pf_meta.header is not None
        assert "FIFO" in pf_meta.header

        # Verify frame metadata is available
        assert len(pf_meta.frame_metadata) > 0
        meta = pf_meta.frame_metadata[0]
        assert meta.latitude is not None or meta.longitude is not None

        # Verify frames are NOT loaded (internal state)
        assert len(pf_meta._frames) == 0

        # Load normally for comparison
        pf_full = PolarFile(filepath)
        assert len(pf_full.frames) > 0

        # Metadata should match
        assert pf_meta.header == pf_full.header
        assert len(pf_meta.frame_metadata) == len(pf_full.frame_metadata)

    def test_circular_destreak(self, single_polar_file: Path):
        """Test that circular destreak handles theta wraparound correctly."""
        from wamos_tpw.config import WamosConfig
        from wamos_tpw.destreak import Destreak
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        frame = pf.frames[0]
        config = WamosConfig()

        # Apply deramp first
        from wamos_tpw.bearing import Theta
        from wamos_tpw.deramp import Deramp

        theta = Theta([frame], config, refine=False)
        bearing_arr = theta.bearing_for_frame(0)
        deramp = Deramp(frame, config, bearing=bearing_arr)
        frame.deramped_intensity = deramp.corrected_intensity

        # Apply circular destreak (no neighbors)
        destreak = Destreak(None, frame, None, config)
        corrected = destreak.corrected_intensity

        # Verify output is valid
        assert corrected is not None
        assert corrected.shape == frame.intensity.shape
        assert not np.all(np.isnan(corrected))

        # Verify continuity at wraparound (first and last rows should be similar)
        # The correction should be smooth across the 360/0 boundary
        first_row_mean = np.nanmean(corrected[0, :])
        last_row_mean = np.nanmean(corrected[-1, :])
        # Allow reasonable difference (not testing exact equality)
        assert abs(first_row_mean - last_row_mean) < np.nanstd(corrected) * 3


class TestProcessedFramesPipeline:
    """Integration tests using the ProcessedFrames high-level interface."""

    def test_processed_frames_iteration(self, test_data_dir: Path):
        """Test ProcessedFrames iteration and processing."""
        from wamos_tpw.processed import ProcessedFrames

        with ProcessedFrames(
            stime="2022-04-05 14:00",
            etime="2022-04-05 15:00",
            polar_path=test_data_dir,
            groupby="1h",
        ) as pf:
            assert len(pf) >= 1, "Should find files"

            # Test groups() method
            groups = pf.groups()
            assert len(groups) >= 1, "Should have at least one group"

            # Test iteration with a single group
            for period, file_list in list(groups.items())[:1]:
                assert len(file_list) >= 1
                # Load frames for this group using period key
                frames = pf.load_group(period, max_frames=5)
                assert len(frames) >= 1

                # Process the frames using process_group
                # process_group returns normalized arrays and may clear frame attributes
                normalized = pf.process_group(frames)

                # Verify processing returned correct number of outputs
                assert len(normalized) == len(frames)
                for i, norm_arr in enumerate(normalized):
                    # Verify normalized output shape matches frame
                    assert norm_arr.shape == frames[i].intensity.shape
                    # Verify values are normalized (mostly in 0-1 range)
                    assert np.nanmin(norm_arr) >= -1  # Some negative allowed after correction
                    assert np.nanmax(norm_arr) <= 2  # Some overshoot allowed


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_list_command(self, test_data_dir: Path):
        """Test 'wamos list' command execution."""
        from wamos_tpw.filenames import Filenames

        # Simulate what the CLI does
        filenames = Filenames(
            stime="2022-04-05 14:00",
            etime="2022-04-05 15:00",
            polar_path=test_data_dir,
        )

        assert len(filenames) >= 1

    def test_cli_parse_command(self, single_polar_file: Path):
        """Test 'wamos parse' command execution."""
        from wamos_tpw.polarfile import PolarFile

        pf = PolarFile(single_polar_file)
        assert pf.header is not None
        assert len(pf.frames) > 0

        # Verify all expected header fields
        expected_fields = ["FIFO", "SFREQ"]
        for field in expected_fields:
            assert field in pf.header, f"Missing header field: {field}"
