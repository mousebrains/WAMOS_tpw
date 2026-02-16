#! /usr/bin/env python3
"""Tests for wamos_tpw.instruments.nmea module."""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from wamos_tpw.instruments.nmea import (
    KNOTS_TO_MS,
    parse_latlon,
    parse_log_timestamp,
    parse_time_hhmmss,
    to_datetime64,
    validate_checksum,
)


class TestParseLogTimestamp:
    """Tests for parse_log_timestamp."""

    def test_basic_timestamp_with_sentence(self):
        """Parse a standard log line into timestamp and sentence."""
        line = "2022-04-02T00:00:00.003358Z $GPGGA,000000.00,3249.12345,N,11713.98765,W,1,12,0.8,15.3,M,-30.2,M,,*4F"
        ts, sentence = parse_log_timestamp(line)
        assert ts.year == 2022
        assert ts.month == 4
        assert ts.day == 2
        assert ts.hour == 0
        assert ts.minute == 0
        assert ts.second == 0
        assert ts.microsecond == 3358
        assert ts.tzinfo == UTC
        assert sentence.startswith("$GPGGA")

    def test_timestamp_without_fractional_seconds(self):
        """Parse timestamp without fractional seconds."""
        line = "2022-04-02T12:30:45Z $HEHDT,123.4,T*XX"
        ts, sentence = parse_log_timestamp(line)
        assert ts.hour == 12
        assert ts.minute == 30
        assert ts.second == 45
        assert ts.microsecond == 0

    def test_raises_on_empty_line(self):
        """Raise ValueError on line without timestamp + sentence."""
        with pytest.raises(ValueError, match="Cannot split"):
            parse_log_timestamp("single_token")

    def test_sentence_is_stripped(self):
        """Sentence has whitespace stripped."""
        line = "2022-04-02T00:00:00Z   $GPGGA,data   "
        _, sentence = parse_log_timestamp(line)
        assert sentence == "$GPGGA,data"


class TestValidateChecksum:
    """Tests for validate_checksum."""

    def test_valid_checksum(self):
        """Valid NMEA checksum passes."""
        # XOR of chars between $ and *: "GPGGA,1" = 0x4B
        assert validate_checksum("$GPGGA,1*4B") is True

    def test_invalid_checksum(self):
        """Invalid checksum fails."""
        assert validate_checksum("$GPGGA,1*FF") is False

    def test_no_checksum_passes(self):
        """Sentence without checksum returns True."""
        assert validate_checksum("$GPGGA,data,more") is True

    def test_hehdt_real_checksum(self):
        """Real HEHDT sentence with valid checksum."""
        # $HEHDT,123.4,T — compute checksum
        payload = "HEHDT,123.4,T"
        chk = 0
        for ch in payload:
            chk ^= ord(ch)
        sentence = f"$HEHDT,123.4,T*{chk:02X}"
        assert validate_checksum(sentence) is True

    def test_bad_hex_in_checksum(self):
        """Non-hex checksum returns False."""
        assert validate_checksum("$GPGGA*ZZ") is False

    def test_exclamation_prefix(self):
        """Sentence starting with ! is also handled."""
        # !AIVDM style sentences
        payload = "AIVDM,1,1,,A,1"
        chk = 0
        for ch in payload:
            chk ^= ord(ch)
        sentence = f"!AIVDM,1,1,,A,1*{chk:02X}"
        assert validate_checksum(sentence) is True


class TestParseLatlon:
    """Tests for parse_latlon."""

    def test_latitude_north(self):
        """Parse northern latitude."""
        # 32 degrees, 49.12345 minutes
        result = parse_latlon("3249.12345", "N")
        expected = 32 + 49.12345 / 60.0
        assert result == pytest.approx(expected, abs=1e-8)

    def test_latitude_south(self):
        """Parse southern latitude (negative)."""
        result = parse_latlon("3249.12345", "S")
        expected = -(32 + 49.12345 / 60.0)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_longitude_west(self):
        """Parse western longitude (negative)."""
        # 117 degrees, 13.98765 minutes
        result = parse_latlon("11713.98765", "W")
        expected = -(117 + 13.98765 / 60.0)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_longitude_east(self):
        """Parse eastern longitude."""
        result = parse_latlon("11713.98765", "E")
        expected = 117 + 13.98765 / 60.0
        assert result == pytest.approx(expected, abs=1e-8)

    def test_zero_latitude(self):
        """Parse equator (0 degrees, 0 minutes)."""
        result = parse_latlon("0000.00000", "N")
        assert result == pytest.approx(0.0)

    def test_high_latitude(self):
        """Parse near-pole latitude."""
        result = parse_latlon("8959.99999", "N")
        expected = 89 + 59.99999 / 60.0
        assert result == pytest.approx(expected, abs=1e-8)


class TestParseTimeHhmmss:
    """Tests for parse_time_hhmmss."""

    def test_basic_time(self):
        """Parse a basic HHMMSS time."""
        day = date(2022, 4, 2)
        result = parse_time_hhmmss("123456", day)
        assert result.hour == 12
        assert result.minute == 34
        assert result.second == 56
        assert result.microsecond == 0
        assert result.tzinfo == UTC

    def test_time_with_fractional_seconds(self):
        """Parse HHMMSS.SS with fractional seconds."""
        day = date(2022, 4, 2)
        result = parse_time_hhmmss("123456.78", day)
        assert result.hour == 12
        assert result.minute == 34
        assert result.second == 56
        assert result.microsecond == 780000

    def test_midnight(self):
        """Parse midnight time."""
        day = date(2022, 4, 2)
        result = parse_time_hhmmss("000000", day)
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_end_of_day(self):
        """Parse 23:59:59."""
        day = date(2022, 4, 2)
        result = parse_time_hhmmss("235959", day)
        assert result.hour == 23
        assert result.minute == 59
        assert result.second == 59

    def test_fractional_padded(self):
        """Short fractional seconds are zero-padded."""
        day = date(2022, 4, 2)
        result = parse_time_hhmmss("120000.1", day)
        assert result.microsecond == 100000

    def test_date_is_preserved(self):
        """The date from the day argument is used."""
        day = date(2023, 12, 25)
        result = parse_time_hhmmss("060000", day)
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25


class TestToDatetime64:
    """Tests for to_datetime64."""

    def test_basic_conversion(self):
        """Convert datetime to datetime64[ns]."""
        dt = datetime(2022, 4, 2, 12, 30, 45, tzinfo=UTC)
        result = to_datetime64(dt)
        assert result.dtype == np.dtype("datetime64[ns]")
        # Round-trip: convert back and compare
        expected = np.datetime64("2022-04-02T12:30:45", "ns")
        assert result == expected

    def test_microsecond_precision(self):
        """Microsecond precision is preserved."""
        dt = datetime(2022, 4, 2, 12, 30, 45, 123456, tzinfo=UTC)
        result = to_datetime64(dt)
        expected = np.datetime64("2022-04-02T12:30:45.123456", "ns")
        assert result == expected


class TestKnotsConversion:
    """Tests for KNOTS_TO_MS constant."""

    def test_knots_to_ms_value(self):
        """KNOTS_TO_MS is correct."""
        assert KNOTS_TO_MS == pytest.approx(0.514444)

    def test_one_knot_in_ms(self):
        """1 knot ≈ 0.514 m/s."""
        assert 1.0 * KNOTS_TO_MS == pytest.approx(0.514444)

    def test_ten_knots(self):
        """10 knots ≈ 5.14 m/s."""
        assert 10.0 * KNOTS_TO_MS == pytest.approx(5.14444)
