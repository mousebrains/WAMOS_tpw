"""Extended tests for Filenames class."""

import numpy as np
import pytest
from pathlib import Path

from wamos_tpw.filenames import Filenames, _parse_timestamp, extract_file_timestamp


class TestParseTimestamp:
    """Tests for _parse_timestamp function."""

    def test_year_only(self):
        """Test YYYY format."""
        ts = _parse_timestamp("2022")
        assert str(ts).startswith("2022-01-01")

    def test_year_month(self):
        """Test YYYYMM format."""
        ts = _parse_timestamp("202203")
        assert "2022-03-01" in str(ts)

    def test_year_month_day(self):
        """Test YYYYMMDD format."""
        ts = _parse_timestamp("20220328")
        assert "2022-03-28" in str(ts)

    def test_year_month_day_hour(self):
        """Test YYYYMMDDHH format."""
        ts = _parse_timestamp("2022032803")
        assert "2022-03-28T03" in str(ts)

    def test_year_month_day_hour_minute(self):
        """Test YYYYMMDDHHmm format."""
        ts = _parse_timestamp("202203280315")
        assert "2022-03-28T03:15" in str(ts)

    def test_full_compact(self):
        """Test YYYYMMDDHHmmss format."""
        ts = _parse_timestamp("20220328031530")
        assert "2022-03-28T03:15:30" in str(ts)

    def test_with_milliseconds(self):
        """Test YYYYMMDDHHmmssSSS format."""
        ts = _parse_timestamp("20220328031530123")
        assert "2022-03-28T03:15:30" in str(ts)

    def test_with_microseconds(self):
        """Test YYYYMMDDHHmmssSSSSSS format."""
        ts = _parse_timestamp("20220328031530123456")
        assert "2022-03-28T03:15:30" in str(ts)

    def test_iso_format(self):
        """Test ISO format."""
        ts = _parse_timestamp("2022-03-28T03:15:30")
        assert "2022-03-28" in str(ts)

    def test_iso_with_space(self):
        """Test ISO format with space separator."""
        ts = _parse_timestamp("2022-03-28 03:15:30")
        assert "2022-03-28" in str(ts)

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-timestamp")

    def test_invalid_length(self):
        """Test invalid compact length raises ValueError."""
        with pytest.raises(ValueError):
            _parse_timestamp("12345")  # 5 digits - invalid


class TestExtractFileTimestamp:
    """Tests for extract_file_timestamp function."""

    def test_valid_filename(self):
        """Test extraction from valid filename."""
        ts = extract_file_timestamp("20220328031530abc.pol")
        assert ts is not None
        assert "2022-03-28T03:15:30" in str(ts)

    def test_short_filename(self):
        """Test short filename returns None."""
        ts = extract_file_timestamp("short.pol")
        assert ts is None

    def test_non_digit_prefix(self):
        """Test non-digit prefix returns None."""
        ts = extract_file_timestamp("abcd0328031530.pol")
        assert ts is None


class TestFilenamesClass:
    """Tests for Filenames class."""

    def test_basic_creation(self, test_data_dir: Path):
        """Test basic Filenames creation."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )
        assert len(fn) > 0

    def test_iteration(self, test_data_dir: Path):
        """Test iteration over filenames."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        files = list(fn)
        assert len(files) == len(fn)

    def test_repr(self, test_data_dir: Path):
        """Test __repr__ method."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        repr_str = repr(fn)
        assert "Filenames(" in repr_str
        assert "stime=" in repr_str
        assert "etime=" in repr_str

    def test_str(self, test_data_dir: Path):
        """Test __str__ method."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        str_str = str(fn)
        # Should be newline-separated file paths
        assert isinstance(str_str, str)

    def test_context_manager(self, test_data_dir: Path):
        """Test context manager protocol."""
        with Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        ) as fn:
            assert len(fn) > 0

    def test_groupby_hour(self, test_data_dir: Path):
        """Test groupby with hour frequency."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        groups = fn.groupby("h")
        assert isinstance(groups, dict)
        for period, files in groups.items():
            assert isinstance(period, np.datetime64)
            assert isinstance(files, list)

    def test_groupby_minute(self, test_data_dir: Path):
        """Test groupby with minute frequency."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405141000"),
            polar_path=str(test_data_dir),
        )

        groups = fn.groupby("m")
        assert isinstance(groups, dict)

    def test_groupby_custom(self, test_data_dir: Path):
        """Test groupby with custom frequency."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        groups = fn.groupby("30m")
        assert isinstance(groups, dict)

    def test_itergroups(self, test_data_dir: Path):
        """Test itergroups method."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        for period, files in fn.itergroups("h"):
            assert isinstance(period, np.datetime64)
            assert isinstance(files, list)

    def test_chunks(self, test_data_dir: Path):
        """Test chunks method."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        chunks = fn.chunks(4)
        assert len(chunks) == 4
        # Total files should equal sum of chunks
        total = sum(len(c) for c in chunks)
        assert total == len(fn)

    def test_chunks_invalid(self, test_data_dir: Path):
        """Test chunks with invalid n."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        with pytest.raises(ValueError, match="must be positive"):
            fn.chunks(0)

    def test_iterchunks(self, test_data_dir: Path):
        """Test iterchunks method."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        chunks = list(fn.iterchunks(3))
        assert len(chunks) == 3

    def test_time_chunks(self, test_data_dir: Path):
        """Test time_chunks method."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        chunks = fn.time_chunks(np.timedelta64(30, "m"))
        assert isinstance(chunks, dict)

    def test_time_chunks_invalid(self, test_data_dir: Path):
        """Test time_chunks with invalid delta."""
        fn = Filenames(
            stime=_parse_timestamp("20220405140000"),
            etime=_parse_timestamp("20220405150000"),
            polar_path=str(test_data_dir),
        )

        with pytest.raises(ValueError, match="must be positive"):
            fn.time_chunks(np.timedelta64(0, "m"))


class TestParseFreq:
    """Tests for Filenames._parse_freq method."""

    def test_simple_units(self):
        """Test simple unit parsing."""
        assert Filenames._parse_freq("h") == (1, "h")
        assert Filenames._parse_freq("D") == (1, "D")
        assert Filenames._parse_freq("m") == (1, "m")
        assert Filenames._parse_freq("s") == (1, "s")

    def test_with_multiplier(self):
        """Test unit with multiplier."""
        assert Filenames._parse_freq("12m") == (12, "m")
        assert Filenames._parse_freq("30s") == (30, "s")
        assert Filenames._parse_freq("6h") == (6, "h")
        assert Filenames._parse_freq("2D") == (2, "D")

    def test_named_units(self):
        """Test named units."""
        assert Filenames._parse_freq("hour") == (1, "h")
        assert Filenames._parse_freq("minute") == (1, "m")
        assert Filenames._parse_freq("second") == (1, "s")
        assert Filenames._parse_freq("day") == (1, "D")

    def test_invalid_unit(self):
        """Test invalid unit raises ValueError."""
        with pytest.raises(ValueError, match="Unknown frequency unit"):
            Filenames._parse_freq("invalid")
