"""Tests for Files class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.files import Files


class TestFiles:
    """Tests for Files class."""

    def test_files_basic(self, test_data_dir: Path):
        """Test basic Files creation."""
        files = Files(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            groupby="h",
        )

        assert len(files) > 0
        assert files.files is not None

    def test_files_with_datetime64(self, test_data_dir: Path):
        """Test Files with np.datetime64 input."""
        stime = np.datetime64("2022-04-05T14:00:00")
        etime = np.datetime64("2022-04-05T15:00:00")

        files = Files(stime=stime, etime=etime, polar_path=str(test_data_dir), groupby="h")

        assert len(files) > 0

    def test_files_filenames_property(self, test_data_dir: Path):
        """Test filenames property returns Filenames instance."""
        files = Files(stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir))

        from wamos_tpw.filenames import Filenames

        assert isinstance(files.filenames, Filenames)

    def test_files_bool(self, test_data_dir: Path):
        """Test __bool__ method."""
        files = Files(stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir))
        assert bool(files) is True

        # Empty range
        empty_files = Files(
            stime="19900101000000", etime="19900101010000", polar_path=str(test_data_dir)
        )
        assert bool(empty_files) is False

    def test_files_groups(self, test_data_dir: Path):
        """Test groups() method."""
        files = Files(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            groupby="h",
        )

        groups = files.groups()
        assert isinstance(groups, dict)
        assert len(groups) > 0

        for period, file_list in groups.items():
            assert isinstance(period, np.datetime64)
            assert isinstance(file_list, list)

    def test_files_itergroups(self, test_data_dir: Path):
        """Test itergroups() method."""
        files = Files(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            groupby="h",
            workers=1,  # Sequential for predictable testing
        )

        count = 0
        for period, frames in files.itergroups():
            assert isinstance(period, np.datetime64)
            assert isinstance(frames, list)
            count += 1

        assert count > 0

    def test_files_load_group(self, test_data_dir: Path):
        """Test load_group() method."""
        files = Files(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            groupby="h",
            workers=1,
        )

        groups = files.groups()
        if groups:
            period = list(groups.keys())[0]
            frames = files.load_group(period, max_frames=2)
            assert isinstance(frames, list)
            assert len(frames) <= 2

    def test_files_load_group_invalid_period(self, test_data_dir: Path):
        """Test load_group() with invalid period."""
        files = Files(stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir))

        invalid_period = np.datetime64("1990-01-01T00:00:00")
        with pytest.raises(KeyError):
            files.load_group(invalid_period)

    def test_files_load_all(self, test_data_dir: Path):
        """Test load_all() method."""
        files = Files(
            stime="20220405140000",
            etime="20220405143000",  # Short range
            polar_path=str(test_data_dir),
            workers=1,
        )

        all_frames = files.load_all()
        assert isinstance(all_frames, list)

    def test_files_summary(self, test_data_dir: Path):
        """Test summary() method."""
        files = Files(
            stime="20220405140000",
            etime="20220405150000",
            polar_path=str(test_data_dir),
            groupby="h",
        )

        summary = files.summary()
        assert "stime" in summary
        assert "etime" in summary
        assert "polar_path" in summary
        assert "groupby" in summary
        assert "total_files" in summary
        assert "n_groups" in summary
        assert "groups" in summary

    def test_files_repr(self, test_data_dir: Path):
        """Test __repr__ method."""
        files = Files(stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir))

        repr_str = repr(files)
        assert "Files(" in repr_str
        assert "stime=" in repr_str
        assert "etime=" in repr_str

    def test_files_str(self, test_data_dir: Path):
        """Test __str__ method."""
        files = Files(stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir))

        str_str = str(files)
        assert "Files:" in str_str
        assert "Path:" in str_str

    def test_files_context_manager(self, test_data_dir: Path):
        """Test context manager protocol."""
        with Files(
            stime="20220405140000", etime="20220405150000", polar_path=str(test_data_dir)
        ) as files:
            assert len(files) > 0

    def test_files_different_groupby(self, test_data_dir: Path):
        """Test different groupby frequencies."""
        for freq in ["h", "30m", "D"]:
            files = Files(
                stime="20220405140000",
                etime="20220405150000",
                polar_path=str(test_data_dir),
                groupby=freq,
            )
            groups = files.groups()
            assert isinstance(groups, dict)

    def test_files_sequential_loading(self, test_data_dir: Path):
        """Test sequential file loading with workers=1."""
        files = Files(
            stime="20220405140000", etime="20220405143000", polar_path=str(test_data_dir), workers=1
        )

        for period, frames in files.itergroups():
            assert isinstance(frames, list)

    def test_files_check_gil_status(self):
        """Test GIL status check method."""
        # This should not raise, regardless of Python version
        result = Files._check_gil_status()
        assert isinstance(result, bool)
