#! /usr/bin/env python3
"""Smoke tests for instrument parsers using synthetic log data.

Each parser is tested with minimal synthetic data to verify:
- Correct parsing of valid records
- Graceful handling of malformed lines
- Correct unit conversions
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from wamos_tpw.instruments.nmea import KNOTS_TO_MS


def _checksum(payload: str) -> str:
    """Compute NMEA checksum for payload (without $ and *)."""
    chk = 0
    for ch in payload:
        chk ^= ord(ch)
    return f"{chk:02X}"


def _nmea(payload: str) -> str:
    """Build a complete NMEA sentence with checksum."""
    return f"${payload}*{_checksum(payload)}"


# =============================================================================
# GPS Parser Tests
# =============================================================================


class TestGpsParser:
    """Tests for instruments/gps.py."""

    @pytest.fixture
    def gps_log(self, tmp_path: Path) -> Path:
        """Create a synthetic GPS log file with grouped sentences."""
        gpgga = _nmea("GPGGA,120000.00,3249.12345,N,11713.98765,W,1,12,0.8,15.3,M,-30.2,M,,")
        gprmc = _nmea("GPRMC,120000.00,A,3249.12345,N,11713.98765,W,5.2,123.4,020422,,,A")
        gphdt = _nmea("GPHDT,275.3,T")
        gngst = _nmea("GNGST,120000.00,1.2,0.5,0.4,30.0,0.3,0.2,0.6")

        # Group 1: 4 sentences within 250ms
        lines = [
            f"2022-04-02T12:00:00.000000Z {gpgga}",
            f"2022-04-02T12:00:00.050000Z {gprmc}",
            f"2022-04-02T12:00:00.100000Z {gphdt}",
            f"2022-04-02T12:00:00.150000Z {gngst}",
            # Group 2: 1 second later
            f"2022-04-02T12:00:01.000000Z {gpgga}",
            f"2022-04-02T12:00:01.050000Z {gprmc}",
        ]
        logfile = tmp_path / "gps_test.log"
        logfile.write_text("\n".join(lines) + "\n")
        return logfile

    def test_parse_gps_file(self, gps_log: Path):
        """Parse a synthetic GPS log file."""
        from wamos_tpw.instruments.gps import parse_gps_file

        records = parse_gps_file(gps_log)
        assert len(records["time"]) == 2  # Two groups
        assert records["latitude"][0] == pytest.approx(32 + 49.12345 / 60, abs=1e-6)
        assert records["longitude"][0] == pytest.approx(-(117 + 13.98765 / 60), abs=1e-6)
        assert records["heading"][0] == pytest.approx(275.3)
        assert records["sog"][0] == pytest.approx(5.2 * KNOTS_TO_MS, abs=1e-4)
        assert records["cog"][0] == pytest.approx(123.4)
        assert records["lat_error"][0] == pytest.approx(0.3)

    def test_skips_invalid_lines(self, tmp_path: Path):
        """Malformed lines are skipped without crashing."""
        from wamos_tpw.instruments.gps import parse_gps_file

        gpgga = _nmea("GPGGA,120000.00,3249.12345,N,11713.98765,W,1,12,0.8,15.3,M,-30.2,M,,")
        lines = [
            "this is garbage",
            "2022-04-02T12:00:00Z ",  # No sentence
            f"2022-04-02T12:00:01.000000Z {gpgga}",
        ]
        logfile = tmp_path / "gps_bad.log"
        logfile.write_text("\n".join(lines) + "\n")
        records = parse_gps_file(logfile)
        assert len(records["time"]) == 1  # Only the valid group


# =============================================================================
# Gyro Parser Tests
# =============================================================================


class TestGyroParser:
    """Tests for instruments/gyro.py."""

    @pytest.fixture
    def gyro_log(self, tmp_path: Path) -> Path:
        """Create a synthetic gyro log file."""
        lines = []
        for i in range(5):
            heading = 123.4 + i * 0.1
            sentence = _nmea(f"HEHDT,{heading:.1f},T")
            lines.append(f"2022-04-02T12:00:0{i}.000000Z {sentence}")
        logfile = tmp_path / "gyro_test.log"
        logfile.write_text("\n".join(lines) + "\n")
        return logfile

    def test_parse_gyro_file(self, gyro_log: Path):
        """Parse a synthetic gyro log file."""
        from wamos_tpw.instruments.gyro import parse_gyro_file

        records = parse_gyro_file(gyro_log)
        assert len(records["time"]) == 5
        assert records["heading"][0] == pytest.approx(123.4)
        assert records["heading"][4] == pytest.approx(123.8)

    def test_skips_pplan_sentences(self, tmp_path: Path):
        """$PPLAN sentences are skipped."""
        from wamos_tpw.instruments.gyro import parse_gyro_file

        hehdt = _nmea("HEHDT,180.0,T")
        pplan = "$PPLAN,some,data*00"
        lines = [
            f"2022-04-02T12:00:00Z {pplan}",
            f"2022-04-02T12:00:01Z {hehdt}",
        ]
        logfile = tmp_path / "gyro_pplan.log"
        logfile.write_text("\n".join(lines) + "\n")
        records = parse_gyro_file(logfile)
        assert len(records["time"]) == 1
        assert records["heading"][0] == pytest.approx(180.0)

    def test_bad_checksum_skipped(self, tmp_path: Path):
        """Lines with bad checksums are skipped."""
        from wamos_tpw.instruments.gyro import parse_gyro_file

        good = _nmea("HEHDT,90.0,T")
        bad = "$HEHDT,45.0,T*FF"  # Wrong checksum
        lines = [
            f"2022-04-02T12:00:00Z {bad}",
            f"2022-04-02T12:00:01Z {good}",
        ]
        logfile = tmp_path / "gyro_chk.log"
        logfile.write_text("\n".join(lines) + "\n")
        records = parse_gyro_file(logfile)
        assert len(records["time"]) == 1
        assert records["heading"][0] == pytest.approx(90.0)


# =============================================================================
# MRU Parser Tests
# =============================================================================


class TestMruParser:
    """Tests for instruments/mru.py."""

    @pytest.fixture
    def mru_log(self, tmp_path: Path) -> Path:
        """Create a synthetic MRU log file with multi-line groups."""
        hehdt = _nmea("HEHDT,275.3,T")
        pashr = _nmea("PASHR,120000.00,275.3,T,1.2,-0.5,0.3,,,")
        phgga = _nmea("PHGGA,120000.00,3249.12345,N,11713.98765,W,1,12,0.8,15.3,M,-30.2,M,,")
        phvtg = _nmea("PHVTG,123.4,T,,M,5.2,N,,K,A")

        # Group 1: first line has timestamp, subsequent lines don't
        lines = [
            f"2022-04-02T12:00:00.005222Z {hehdt}",
            pashr,  # No timestamp — continuation
            phgga,
            phvtg,
            # Group 2
            f"2022-04-02T12:00:00.512000Z {hehdt}",
            pashr,
        ]
        logfile = tmp_path / "mru_test.log"
        logfile.write_text("\n".join(lines) + "\n")
        return logfile

    def test_parse_mru_file(self, mru_log: Path):
        """Parse a synthetic MRU log file."""
        from wamos_tpw.instruments.mru import parse_mru_file

        records = parse_mru_file(mru_log)
        assert len(records["time"]) == 2
        assert records["heading"][0] == pytest.approx(275.3)
        assert records["roll"][0] == pytest.approx(1.2)
        assert records["pitch"][0] == pytest.approx(-0.5)
        assert records["heave"][0] == pytest.approx(0.3)

    def test_empty_file(self, tmp_path: Path):
        """Empty file returns empty records."""
        from wamos_tpw.instruments.mru import parse_mru_file

        logfile = tmp_path / "mru_empty.log"
        logfile.write_text("")
        records = parse_mru_file(logfile)
        assert len(records["time"]) == 0


# =============================================================================
# Wind Parser Tests
# =============================================================================


class TestWindParser:
    """Tests for instruments/wind.py."""

    @pytest.fixture
    def wind_log(self, tmp_path: Path) -> Path:
        """Create a synthetic wind bridge log file."""
        lines = []
        for i in range(3):
            angle = 45.0 + i * 10
            speed_knots = 10.0 + i
            sentence = _nmea(f"WIMWV,{angle:.1f},R,{speed_knots:.1f},N,A")
            lines.append(f"2022-04-02T12:00:0{i}.000000Z {sentence}")
        logfile = tmp_path / "wind_test.log"
        logfile.write_text("\n".join(lines) + "\n")
        return logfile

    def test_parse_wind_file(self, wind_log: Path):
        """Parse a synthetic wind log file."""
        from wamos_tpw.instruments.wind import parse_wind_file

        records = parse_wind_file(wind_log)
        assert len(records["time"]) == 3
        assert records["relative_wind_direction"][0] == pytest.approx(45.0)
        assert records["relative_wind_speed"][0] == pytest.approx(10.0 * KNOTS_TO_MS, abs=1e-4)

    def test_invalid_status_skipped(self, tmp_path: Path):
        """Lines with status != 'A' are skipped."""
        from wamos_tpw.instruments.wind import parse_wind_file

        good = _nmea("WIMWV,90.0,R,15.0,N,A")
        bad = _nmea("WIMWV,180.0,R,20.0,N,V")  # Status 'V' = void
        lines = [
            f"2022-04-02T12:00:00Z {bad}",
            f"2022-04-02T12:00:01Z {good}",
        ]
        logfile = tmp_path / "wind_status.log"
        logfile.write_text("\n".join(lines) + "\n")
        records = parse_wind_file(logfile)
        assert len(records["time"]) == 1
        assert records["relative_wind_direction"][0] == pytest.approx(90.0)


# =============================================================================
# MET Parser Tests
# =============================================================================


class TestMetParser:
    """Tests for instruments/met.py."""

    @pytest.fixture
    def met_file(self, tmp_path: Path) -> Path:
        """Create a synthetic MET data file."""
        content = textwrap.dedent("""\
            # MET data header
            # Sat 02-Apr-22  00:00:00
            # Some description
            # Time WS WD WS-2 WD-2 TW TI LA LO GY CR SP
              10  12.5  180.0  11.0  175.0  8.5  1.0  32.82  -117.23  275.3  123.4  5.2
             110  13.0  185.0  12.0  180.0  9.0  1.0  32.82  -117.23  275.5  123.6  5.4
            1000  -99.0  190.0  -99.0  185.0  -99.0  0.0  32.82  -117.23  275.7  123.8  5.6
        """)
        filepath = tmp_path / "020422.MET"
        filepath.write_text(content)
        return filepath

    def test_parse_met_file(self, met_file: Path):
        """Parse a synthetic MET data file."""
        from wamos_tpw.instruments.met import parse_met_file

        records = parse_met_file(met_file)
        assert len(records["time"]) == 3

        # Wind speed converted from knots to m/s
        assert records["wind_speed"][0] == pytest.approx(12.5 * KNOTS_TO_MS, abs=1e-4)
        assert records["wind_direction"][0] == pytest.approx(180.0)

        # Lat/lon not converted (not in knots)
        assert records["latitude"][0] == pytest.approx(32.82)
        assert records["longitude"][0] == pytest.approx(-117.23)

        # Heading
        assert records["heading"][0] == pytest.approx(275.3)

    def test_missing_values_become_nan(self, met_file: Path):
        """-99.0 values are converted to NaN."""
        from wamos_tpw.instruments.met import parse_met_file

        records = parse_met_file(met_file)
        # Third row has -99.0 for WS, WS-2, TW
        assert np.isnan(records["wind_speed"][2])
        assert np.isnan(records["true_wind_speed"][2])

    def test_short_met_file(self, tmp_path: Path):
        """File with too few lines returns empty records."""
        from wamos_tpw.instruments.met import parse_met_file

        filepath = tmp_path / "short.MET"
        filepath.write_text("# line1\n# line2\n")
        records = parse_met_file(filepath)
        assert len(records["time"]) == 0

    def test_time_parsing(self, met_file: Path):
        """HHMMSS times are correctly converted to datetime64."""
        from wamos_tpw.instruments.met import parse_met_file

        records = parse_met_file(met_file)
        # Time 10 = 00:00:10
        t0 = records["time"][0]
        assert t0 == np.datetime64("2022-04-02T00:00:10", "ns")
        # Time 110 = 00:01:10
        t1 = records["time"][1]
        assert t1 == np.datetime64("2022-04-02T00:01:10", "ns")
        # Time 1000 = 00:10:00
        t2 = records["time"][2]
        assert t2 == np.datetime64("2022-04-02T00:10:00", "ns")
