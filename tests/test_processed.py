"""Tests for ProcessedFrames class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.processed import ProcessedFrames
from wamos_tpw.config import WamosConfig
from wamos_tpw.polarfile import PolarFile


class TestProcessedFrames:
    """Tests for ProcessedFrames class."""

    def test_processed_frames_basic(self, test_data_dir: Path):
        """Test basic ProcessedFrames creation."""
        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            groupby="h",
        )

        assert len(pf) > 0
        assert pf.config is not None
        assert pf.radar_height is None

    def test_processed_frames_with_config(self, test_data_dir: Path):
        """Test ProcessedFrames with custom config."""
        config = WamosConfig()
        config.shadow.center = 180.0
        config.shadow.width = 90.0

        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            config=config,
            radar_height=25.0,
        )

        assert pf.config is config
        assert pf.radar_height == 25.0

    def test_deramp_frames(self, april_polar_files: list[Path]):
        """Test deramp_frames method."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 1:
            pytest.skip("Need at least 1 frame")

        config = WamosConfig()
        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(april_polar_files[0].parent.parent.parent.parent.parent),
            config=config,
        )

        # Deramp should add deramped_intensity attribute
        pf.deramp_frames(frames)

        for frame in frames:
            assert hasattr(frame, "deramped_intensity")
            assert frame.deramped_intensity is not None
            assert frame.deramped_intensity.shape == frame.intensity.shape

    def test_destreak_frames(self, single_polar_file: Path):
        """Test destreak_frames method with a single frame."""
        pf = PolarFile(single_polar_file)
        frames = pf.frames[:1]

        if not frames or frames[0].intensity is None:
            pytest.skip("Need at least 1 frame with valid intensity")

        config = WamosConfig()
        pframes = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(single_polar_file.parent.parent.parent.parent.parent),
            config=config,
        )

        # Test with a single frame (no neighbors)
        corrected = pframes.destreak_frames(frames)

        assert len(corrected) == len(frames)
        for i, corr in enumerate(corrected):
            assert corr.shape == frames[i].intensity.shape

    def test_normalize_frames(self, april_polar_files: list[Path]):
        """Test normalize_frames method."""
        frames = []
        for fp in april_polar_files[:2]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 1:
            pytest.skip("Need at least 1 frame")

        config = WamosConfig()
        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(april_polar_files[0].parent.parent.parent.parent.parent),
            config=config,
        )

        # Create some test intensity arrays
        intensities = [frame.intensity.astype(float) for frame in frames]
        normalized = pf.normalize_frames(intensities)

        assert len(normalized) == len(intensities)
        for norm in normalized:
            assert norm.min() >= 0.0
            assert norm.max() <= 1.0

    def test_normalize_constant_array(self, april_polar_files: list[Path]):
        """Test normalize_frames with constant array."""
        config = WamosConfig()
        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(april_polar_files[0].parent.parent.parent.parent.parent),
            config=config,
        )

        # Constant array should normalize to 0.5
        constant = [np.full((100, 100), 42.0)]
        normalized = pf.normalize_frames(constant)

        assert len(normalized) == 1
        assert np.allclose(normalized[0], 0.5)

    def test_refine_theta(self, april_polar_files: list[Path]):
        """Test refine_theta method."""
        frames = []
        for fp in april_polar_files[:3]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 3:
            pytest.skip("Need at least 3 frames for theta refinement")

        config = WamosConfig()
        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(april_polar_files[0].parent.parent.parent.parent.parent),
            config=config,
        )

        # Should not raise
        pf.refine_theta(frames)

    def test_process_group(self, april_polar_files: list[Path]):
        """Test process_group method."""
        frames = []
        for fp in april_polar_files[:3]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 3:
            pytest.skip("Need at least 3 frames")

        config = WamosConfig()
        pf = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(april_polar_files[0].parent.parent.parent.parent.parent),
            config=config,
        )

        normalized = pf.process_group(frames)

        assert len(normalized) == len(frames)
        for norm in normalized:
            assert norm.min() >= 0.0
            assert norm.max() <= 1.0

    def test_context_manager(self, test_data_dir: Path):
        """Test context manager protocol."""
        with ProcessedFrames(
            stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir)
        ) as pf:
            assert len(pf) > 0


class TestProcessedFramesIntegration:
    """Integration tests for ProcessedFrames."""

    def test_full_processing_pipeline(self, april_polar_files: list[Path]):
        """Test full processing pipeline on real data."""
        if len(april_polar_files) < 3:
            pytest.skip("Need at least 3 files")

        frames = []
        for fp in april_polar_files[:3]:
            pf = PolarFile(fp)
            frames.extend(pf.frames[:1])

        if len(frames) < 3:
            pytest.skip("Need at least 3 frames")

        config = WamosConfig()
        pframes = ProcessedFrames(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(april_polar_files[0].parent.parent.parent.parent.parent),
            config=config,
        )

        # 1. Refine theta
        pframes.refine_theta(frames)

        # 2. Deramp
        pframes.deramp_frames(frames)
        for frame in frames:
            assert hasattr(frame, "deramped_intensity")

        # 3. Destreak
        corrected = pframes.destreak_frames(frames)
        assert len(corrected) == len(frames)

        # 4. Normalize
        normalized = pframes.normalize_frames(corrected)
        for norm in normalized:
            assert norm.min() >= 0.0
            assert norm.max() <= 1.0
