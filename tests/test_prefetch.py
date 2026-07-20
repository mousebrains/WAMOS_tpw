"""Tests for the bounded read-ahead iterator (prefetch.py)."""

from pathlib import Path

from wamos_tpw.prefetch import read_ahead


def _make_files(tmp_path: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i:03d}.bin"
        p.write_bytes(bytes([i % 256]) * (100 + i))
        paths.append(p)
    return paths


def test_yields_all_in_order(tmp_path):
    paths = _make_files(tmp_path, 20)
    out = list(read_ahead(paths, depth=4, workers=3))
    assert out == paths


def test_contents_readable_after_yield(tmp_path):
    paths = _make_files(tmp_path, 6)
    for i, p in enumerate(read_ahead(paths, depth=3)):
        assert p.read_bytes() == bytes([i % 256]) * (100 + i)


def test_empty_and_single(tmp_path):
    assert list(read_ahead([], depth=8)) == []
    paths = _make_files(tmp_path, 1)
    assert list(read_ahead(paths, depth=8)) == paths


def test_depth_one_passthrough(tmp_path):
    paths = _make_files(tmp_path, 5)
    assert list(read_ahead(paths, depth=1)) == paths


def test_missing_file_still_yielded(tmp_path):
    """Warming errors are swallowed; the consumer sees the path anyway."""
    paths = _make_files(tmp_path, 3)
    ghost = tmp_path / "missing.bin"
    seq = [paths[0], ghost, paths[1], paths[2]]
    assert list(read_ahead(seq, depth=2)) == seq


def test_accepts_strings(tmp_path):
    paths = _make_files(tmp_path, 4)
    out = list(read_ahead([str(p) for p in paths], depth=2))
    assert out == paths
    assert all(isinstance(p, Path) for p in out)


def test_depth_larger_than_input(tmp_path):
    paths = _make_files(tmp_path, 3)
    assert list(read_ahead(paths, depth=64, workers=8)) == paths


def test_early_close_no_hang(tmp_path):
    """Abandoning the iterator mid-stream must not deadlock."""
    paths = _make_files(tmp_path, 30)
    it = read_ahead(paths, depth=4)
    assert next(it) == paths[0]
    assert next(it) == paths[1]
    it.close()  # GeneratorExit inside the with-block; pool must shut down
