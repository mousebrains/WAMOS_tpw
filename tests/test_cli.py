"""Tests for the CLI interface."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLI:
    """Test the wamos CLI command."""

    def test_cli_help(self):
        """Test that the CLI help command works."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "wamos" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_no_args(self):
        """Test CLI without arguments shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli"], capture_output=True, text=True
        )
        # Should show help or error
        assert result.returncode in (0, 1, 2)

    def test_cli_list_help(self):
        """Test the list subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "list", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "list" in result.stdout.lower() or "polar" in result.stdout.lower()

    def test_cli_parse_help(self):
        """Test the parse subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "parse", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_list_with_test_data(self, test_data_dir: Path):
        """Test list command with actual test data."""
        april_dir = test_data_dir / "2022" / "04" / "05" / "14"
        if not april_dir.exists():
            pytest.skip("Test data not available")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "-v",
                "list",
                "202204051400",
                "202204051401",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
        )
        # Should find files and exit successfully
        assert result.returncode == 0
        # Output now goes to stderr via logging
        output = result.stdout + result.stderr
        assert ".pol" in output or "file" in output.lower()

    def test_cli_parse_with_test_file(self, single_polar_file: Path):
        """Test parse command with actual test file."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "-v", "parse", str(single_polar_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Output now goes to stderr via logging
        output = result.stdout + result.stderr
        # Should output some info about the file
        assert "frame" in output.lower() or "polar" in output.lower()


class TestCLIModuleImports:
    """Test that CLI modules can be imported correctly."""

    def test_import_cli(self):
        """Test importing the main CLI module."""
        from wamos_tpw import cli

        assert hasattr(cli, "main")

    def test_import_subcommands(self):
        """Test that all subcommand modules have add_subparser."""
        from wamos_tpw import (
            filenames,
            polarfile,
            deramp,
            destreak,
        )

        # Each module should have add_subparser function
        assert hasattr(filenames, "add_subparser")
        assert hasattr(polarfile, "add_subparser")
        assert hasattr(deramp, "add_subparser")
        assert hasattr(destreak, "add_subparser")
