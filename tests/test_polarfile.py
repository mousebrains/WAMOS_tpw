"""Tests for PolarFile parsing."""

from pathlib import Path

from wamos_tpw.polarfile import PolarFile


class TestPolarFile:
    """Tests for PolarFile class."""

    def test_load_april_file(self, single_polar_file: Path):
        """Test loading an April POLAR file."""
        pf = PolarFile(single_polar_file)
        assert pf is not None
        assert len(pf) >= 1

    def test_load_march_file(self, march_single_polar_file: Path):
        """Test loading a March POLAR file (lzma compressed)."""
        pf = PolarFile(march_single_polar_file)
        assert pf is not None
        assert len(pf) >= 1

    def test_frame_extraction(self, single_polar_file: Path):
        """Test extracting a frame from a POLAR file."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()

        assert frame is not None
        assert frame.n_bearings > 0
        assert frame.n_distances > 0
        assert frame.intensity.shape == (frame.n_bearings, frame.n_distances)

    def test_metadata(self, single_polar_file: Path):
        """Test frame metadata extraction."""
        pf = PolarFile(single_polar_file)
        frame = pf.frame()
        meta = frame.metadata

        # Check that essential metadata fields exist
        assert meta.filename is not None
        assert meta.samples_in_range > 0
        assert meta.sampling_frequency > 0

    def test_multiple_files(self, april_polar_files: list[Path]):
        """Test loading multiple contiguous files."""
        for polar_path in april_polar_files:
            pf = PolarFile(polar_path)
            assert pf is not None
            assert len(pf) >= 1
