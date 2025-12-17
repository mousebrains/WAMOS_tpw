"""Tests for Filenames class."""

import pytest
from pathlib import Path

from wamos_tpw.filenames import Filenames, _parse_timestamp


class TestFilenames:
    """Tests for Filenames class."""

    def test_discover_april_files(self, test_data_dir: Path):
        """Test file discovery for April data."""
        april_dir = test_data_dir / "2022" / "04" / "05" / "14"
        if not april_dir.exists():
            pytest.skip("April test data not available")

        fn = Filenames(
            stime=_parse_timestamp('202204051400'),
            etime=_parse_timestamp('202204051500'),
            polar_path=test_data_dir
        )

        files = list(fn.files)
        assert len(files) >= 1

    def test_discover_march_files(self, test_data_dir: Path):
        """Test file discovery for March data."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        fn = Filenames(
            stime=_parse_timestamp('202203280300'),
            etime=_parse_timestamp('202203280400'),
            polar_path=test_data_dir
        )

        files = list(fn.files)
        assert len(files) >= 1

    def test_groupby(self, test_data_dir: Path):
        """Test groupby functionality."""
        march_dir = test_data_dir / "2022" / "03" / "28" / "03"
        if not march_dir.exists():
            pytest.skip("March test data not available")

        fn = Filenames(
            stime=_parse_timestamp('202203280300'),
            etime=_parse_timestamp('202203280400'),
            polar_path=test_data_dir
        )

        groups = fn.groupby('h')
        assert len(groups) >= 1

        for period, files in groups.items():
            assert period is not None
            assert len(files) >= 1
