"""Tests for the CORDC NAVNET-SAMPLE fixed-station parser."""

from datetime import UTC, datetime

import numpy as np
import pytest

from wamos_tpw.cordc import CordcFile, FixedStation

STATION = FixedStation(
    name="test",
    latitude=6.91677,
    longitude=134.14840,
    antenna_height_m=37.2,
    theta0_deg=18.836,
    cell_m=15.0003,
    range_offset_m=-36.8,
    clock_offset_s=23.0,
)


def make_file(tmp_path, words, sweep_len=8, count=None, name="20230520123456.pol"):
    """Assemble a synthetic CORDC file mirroring the real byte layout."""
    header = (
        "CC *********** PARAMETER WRITTEN BY NAVNET SAMPLE **************\r\n"
        "OWNER  CORDC   CC   OWNER OF PROGRAM\r\n"
        "DATE  05-20-2023   CC [MM-DD-YYYY]  \r\n"
        "TIME  12:34:56   CC [HH:MM:SS]  \r\n"
        "RPT  2.509   CC [sec]  ANTENNA REPETITION TIME\r\n"
        f"FIFO  {sweep_len}   CC   Number of samples in range\r\n"
        "RANGE  7.408   CC [km]  RADAR RANGE VALUE\r\n"
        "SCALE  496   CC\r\n"
        "F0001  05-20-2023 12:34:56 2.509\r\n"
        "EOH    CC ***********\r\n"
    ).encode("latin1")
    radials = []
    for i, w in enumerate(words):
        prefix = np.array([w & 0xFF, w >> 8], dtype=np.uint8).tobytes()
        radials.append(prefix + bytes([(i + j) % 256 for j in range(sweep_len)]))
    payload = b"".join(radials)
    n = len(payload) if count is None else count
    fn = tmp_path / name
    fn.write_bytes(header + f"{n:10d}".encode() + payload)
    return fn


def test_header_and_payload(tmp_path):
    words = [0, 4, 8, 8188]
    f = CordcFile(make_file(tmp_path, words), STATION)
    assert f.header["RANGE"] == pytest.approx(7.408)
    assert f.header["SCALE"] == pytest.approx(496)
    assert f.header["RPT"] == pytest.approx(2.509)
    assert f.intensity.shape == (4, 8)
    np.testing.assert_array_equal(f.words, words)
    # intensity content round-trips
    assert f.intensity[2, 3] == 5


def test_bearings_apply_encoder_zero(tmp_path):
    f = CordcFile(make_file(tmp_path, [0, 2048, 4096]), STATION)
    np.testing.assert_allclose(f.bearings, [18.836, 108.836, 198.836], atol=1e-9)


def test_ranges_calibrated_and_header_fallback(tmp_path):
    f = CordcFile(make_file(tmp_path, [0]), STATION)
    # calibrated: 15.0003 * (i + 0.5) - 36.8
    np.testing.assert_allclose(f.ranges[0], 15.0003 * 0.5 - 36.8)
    bare = FixedStation("bare", 0.0, 0.0, 0.0, theta0_deg=0.0)
    f2 = CordcFile(make_file(tmp_path, [0]), bare)
    np.testing.assert_allclose(f2.cell_m, 7408.0 / 496.0)
    np.testing.assert_allclose(f2.ranges[0], 7408.0 / 496.0 * 0.5)


def test_times_clock_and_intra_sweep(tmp_path):
    f = CordcFile(make_file(tmp_path, [0, 4096]), STATION)
    assert f.file_time == datetime(2023, 5, 20, 12, 34, 56, tzinfo=UTC)
    dt = (f.times - np.datetime64("2023-05-20T12:34:56", "ms")).astype("timedelta64[ms]").astype(
        float
    ) / 1000.0
    # word 0: clock only; word 4096: clock + half a rotation
    np.testing.assert_allclose(dt, [23.0, 23.0 + 2.509 / 2], atol=2e-3)


def test_count_truncation_and_errors(tmp_path):
    # count larger than actual payload -> truncated read, still parses
    f = CordcFile(make_file(tmp_path, [0, 4], count=10_000), STATION)
    assert f.intensity.shape[0] == 2
    # garbage count -> error
    fn = make_file(tmp_path, [0], name="bad.pol")
    raw = fn.read_bytes()
    i = raw.find(b"EOH")
    i_nl = raw.find(b"\n", i)
    fn.write_bytes(raw[: i_nl + 1] + b"nonsense??" + raw[i_nl + 11 :])
    with pytest.raises(ValueError, match="byte count"):
        CordcFile(fn, STATION)
    # not a CORDC file at all
    fn2 = tmp_path / "x.pol"
    fn2.write_bytes(b"WAMOS whatever")
    with pytest.raises(ValueError, match="EOH"):
        CordcFile(fn2, STATION)


def test_station_required_for_calibrated_views(tmp_path):
    f = CordcFile(make_file(tmp_path, [0, 4]))
    assert f.intensity.shape == (2, 8)  # raw decode works without station
    for prop in ("bearings", "ranges", "times"):
        with pytest.raises(ValueError, match="FixedStation"):
            getattr(f, prop)


def test_packaged_css_angaur_station():
    s = FixedStation.from_yaml("css_angaur_2023")
    assert s.latitude == pytest.approx(6.91677)
    assert s.longitude == pytest.approx(134.14840)
    assert s.theta0_deg == pytest.approx(18.836)
    assert s.cell_m == pytest.approx(15.0003)
    assert s.range_offset_m == pytest.approx(-36.8)
    assert s.clock_offset_s == pytest.approx(23.0)
