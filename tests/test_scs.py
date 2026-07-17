"""Tests for the NOAA SCS raw NMEA log parser (instruments/scs.py)."""

import numpy as np
import pytest

from wamos_tpw.instruments.scs import (
    _derive_sog_cog,
    _parse_gga,
    _parse_hdt,
    _parse_mwv,
    parse_scs_line,
)

xr = pytest.importorskip("xarray")


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


class TestParseScsLine:
    def test_valid_line(self):
        ts, sentence = parse_scs_line("04/29/2023,00:35:51.014,$HEHDT,306.68,T*14")
        assert ts == np.datetime64("2023-04-29T00:35:51.014000000")
        assert sentence == "$HEHDT,306.68,T*14"

    def test_sentence_with_commas_preserved(self):
        _, sentence = parse_scs_line(
            "05/20/2023,00:00:02.130,$GPGGA,000002.00,0649.936971,N,13421.321745,E,2,17,0.7,12.932,M,65.0,M,4.0,0643*74"
        )
        assert sentence.startswith("$GPGGA,000002.00")

    def test_rejects_non_scs_line(self):
        with pytest.raises(ValueError):
            parse_scs_line("2023-04-29T00:35:51Z $HEHDT,306.68,T*14")


class TestStreamParsers:
    def test_hdt(self, tmp_path):
        f = _write(
            tmp_path,
            "GYRO-01-HDT-RAW_20230429-000000.Raw",
            [
                "04/29/2023,00:35:51.014,$HEHDT,306.68,T*14",
                "04/29/2023,00:35:52.022,$HEHDT,306.69,T*15",
                "04/29/2023,00:35:53.000,$HEHDT,999.99,T*00",  # bad checksum
            ],
        )
        rec = _parse_hdt([f])
        assert rec["heading"] == [306.68, 306.69]

    def test_gga_position_and_fix_gate(self, tmp_path):
        good = "$GPGGA,000002.00,0649.936971,N,13421.321745,E,2,17,0.7,12.932,M,65.0,M,4.0,0643*74"
        f = _write(
            tmp_path,
            "CNAV3050-GGA-RAW_20230520-000000.Raw",
            [
                f"05/20/2023,00:00:02.130,{good}",
            ],
        )
        rec = _parse_gga([f])
        assert rec["latitude"][0] == pytest.approx(6 + 49.936971 / 60)
        assert rec["longitude"][0] == pytest.approx(134 + 21.321745 / 60)

    def test_mwv_relative_knots(self, tmp_path):
        f = _write(
            tmp_path,
            "BRIDGE-WIND-STBD_20230606-000000.Raw",
            [
                "06/06/2023,00:00:01.723,$WIMWV,037,R,013,N,A*25",
            ],
        )
        rec = _parse_mwv([f])
        assert rec["relative_wind_direction"] == [37.0]
        assert rec["relative_wind_speed"][0] == pytest.approx(13 * 0.514444)


class TestDerivedSogCog:
    def test_steady_northward(self):
        t = np.array([0, 10, 20], dtype="datetime64[s]").astype("datetime64[ns]")
        lat = np.array([7.0, 7.0 + 50 / 111_319.5, 7.0 + 100 / 111_319.5])
        lon = np.full(3, 134.2)
        sog, cog = _derive_sog_cog(t, lat, lon)
        assert sog == pytest.approx([5.0, 5.0, 5.0], rel=1e-3)
        assert cog == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


class TestDirectory:
    def test_end_to_end(self, tmp_path):
        from wamos_tpw.instruments.scs import parse_scs_directory

        d = tmp_path / "bags" / "data"
        d.mkdir(parents=True)
        _write(
            d,
            "GYRO-01-HDT-RAW_20230429-000000.Raw",
            [
                "04/29/2023,00:35:51.014,$HEHDT,306.68,T*14",
            ],
        )
        _write(
            d,
            "BRIDGE-WIND-STBD_20230606-000000.Raw",
            [
                "06/06/2023,00:00:01.723,$WIMWV,037,R,013,N,A*25",
            ],
        )
        out = tmp_path / "out"
        written = parse_scs_directory(tmp_path, out)
        names = {p.name for p in written}
        assert names == {"gyro_sperry.nc", "wind_bridge.nc"}
        with xr.open_dataset(out / "gyro_sperry.nc") as ds:
            assert ds.heading.values.tolist() == [306.68]

    def test_no_streams_raises(self, tmp_path):
        from wamos_tpw.instruments.scs import parse_scs_directory

        with pytest.raises(FileNotFoundError):
            parse_scs_directory(tmp_path, tmp_path / "out")
