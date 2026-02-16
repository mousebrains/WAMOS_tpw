"""Shared pytest fixtures for wamos_tpw tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory (POLAR at repo root)."""
    return Path(__file__).parent.parent / "POLAR"


@pytest.fixture(scope="session")
def april_polar_files(test_data_dir: Path) -> list[Path]:
    """
    Return list of April POLAR test files.

    These are larger uncompressed-source files (~4.5MB each with xz compression).
    Two contiguous files from 2022/04/05/14.
    """
    april_dir = test_data_dir / "2022" / "04" / "05" / "14"
    if not april_dir.exists():
        pytest.skip("April test data not available")
    files = sorted(april_dir.glob("*.pol*"))
    if not files:
        pytest.skip("No April test files found")
    return files


@pytest.fixture(scope="session")
def march_polar_files(test_data_dir: Path) -> list[Path]:
    """
    Return list of March POLAR test files.

    These are smaller lzma-compressed files (~0.8MB each).
    Six contiguous files from 2022/03/28/03.
    """
    march_dir = test_data_dir / "2022" / "03" / "28" / "03"
    if not march_dir.exists():
        pytest.skip("March test data not available")
    files = sorted(march_dir.glob("*.pol*"))
    if not files:
        pytest.skip("No March test files found")
    return files


@pytest.fixture(scope="session")
def single_polar_file(april_polar_files: list[Path]) -> Path:
    """Return a single POLAR file for basic tests."""
    return april_polar_files[0]


@pytest.fixture(scope="session")
def march_single_polar_file(march_polar_files: list[Path]) -> Path:
    """Return a single March POLAR file for basic tests."""
    return march_polar_files[0]
