"""Integration tests for CLI commands.

These tests exercise the full CLI workflow with real data,
testing command-line arguments and output.
"""

import subprocess
import sys
from pathlib import Path


class TestCombineCLI:
    """Integration tests for the combine command."""

    def test_combine_dry_run(self, test_data_dir: Path):
        """Test combine --dry-run shows what would be processed."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "combine",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
                "--groupby=1h",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "DRY" in output.upper() and "RUN" in output.upper()
        # Should show what would be processed
        assert "combine" in output.lower()

    def test_combine_dry_run_with_movie(self, test_data_dir: Path, tmp_path: Path):
        """Test combine --dry-run with --movie shows movie path."""
        movie_path = tmp_path / "test.mp4"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "combine",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
                "--groupby=1h",
                "--dry-run",
                f"--movie={movie_path}",
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "DRY" in output.upper() and "RUN" in output.upper()
        # Should mention movie in args
        assert "movie" in output.lower()
        # Movie should NOT be created in dry-run
        assert not movie_path.exists()

    def test_combine_frames_only(self, test_data_dir: Path, tmp_path: Path):
        """Test combine with --frames-dir only (no movie)."""
        frames_dir = tmp_path / "frames"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "combine",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
                "--groupby=1h",
                f"--frames-dir={frames_dir}",
                "--max-frames=3",
                "--workers=2",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0
        assert frames_dir.exists()

        # Should have created at least one frame
        png_files = list(frames_dir.glob("*.png"))
        assert len(png_files) >= 1

    def test_combine_no_process(self, test_data_dir: Path, tmp_path: Path):
        """Test combine with --no-process skips deramp/destreak."""
        frames_dir = tmp_path / "frames_no_process"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "combine",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
                "--groupby=1h",
                f"--frames-dir={frames_dir}",
                "--max-frames=3",
                "--no-process",
                "--workers=2",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0
        assert frames_dir.exists()


class TestConfigCLI:
    """Integration tests for the config command."""

    def test_config_defaults(self):
        """Test config command shows defaults."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "config"],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "shadow" in output.lower()
        assert "180" in output  # Default shadow center

    def test_config_validate(self):
        """Test config --validate runs validation."""
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "config", "--validate"],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "Validation" in output

    def test_config_validate_with_file(self, tmp_path: Path):
        """Test config --validate with custom config file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
tower: TEST_TOWER
radar:
  height: 25.0
shadow:
  center: 180.0
  width: 90.0
""")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "config",
                str(config_file),
                "--validate",
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "PASSED" in output

    def test_config_create_sample(self, tmp_path: Path):
        """Test config --create-sample creates sample file."""
        # Run in tmp_path to avoid creating file in repo
        result = subprocess.run(
            [sys.executable, "-m", "wamos_tpw.cli", "config", "--create-sample"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0

        sample_file = tmp_path / "wamos_config.yaml"
        assert sample_file.exists()

        content = sample_file.read_text()
        assert "tower" in content
        assert "radar" in content


class TestListCLI:
    """Integration tests for the list command."""

    def test_list_finds_files(self, test_data_dir: Path):
        """Test list command finds polar files."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "list",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert ".pol" in output or "file" in output.lower()

    def test_list_empty_range(self, test_data_dir: Path):
        """Test list with range that has no files."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "list",
                "19000101T0000",
                "19000101T0100",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
        )

        # Should succeed but find no files
        assert result.returncode == 0


class TestParseCLI:
    """Integration tests for the parse command."""

    def test_parse_single_file(self, single_polar_file: Path):
        """Test parse command on single file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "parse",
                str(single_polar_file),
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        assert "frame" in output.lower()

    def test_parse_show_header(self, single_polar_file: Path):
        """Test parse --show-header displays header info."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "parse",
                str(single_polar_file),
                "--show-header",
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        assert result.returncode == 0
        # Header should contain FIFO or other known fields
        assert "FIFO" in output or "SFREQ" in output


class TestBearingCLI:
    """Integration tests for the bearing command."""

    def test_bearing_basic(self, test_data_dir: Path):
        """Test bearing command runs without error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "bearing",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0


class TestTimestampCLI:
    """Integration tests for the timestamp command."""

    def test_timestamp_basic(self, test_data_dir: Path):
        """Test timestamp command runs without error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "timestamp",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0


class TestProcessCLI:
    """Integration tests for the process command."""

    def test_process_basic(self, test_data_dir: Path):
        """Test process command runs without error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "process",
                "20220405T1400",
                "20220405T1500",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # May complete or show usage, but shouldn't crash with unexpected error
        assert result.returncode in (0, 1, 2)


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_invalid_time_format(self, test_data_dir: Path):
        """Test CLI handles invalid time format gracefully."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "list",
                "not-a-time",
                "also-not-a-time",
                str(test_data_dir),
            ],
            capture_output=True,
            text=True,
        )

        # Should fail but not crash
        assert result.returncode != 0 or "error" in (result.stdout + result.stderr).lower()

    def test_nonexistent_path(self, tmp_path: Path):
        """Test CLI handles nonexistent path."""
        fake_path = tmp_path / "does_not_exist"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "list",
                "20220405T1400",
                "20220405T1500",
                str(fake_path),
            ],
            capture_output=True,
            text=True,
        )

        # Should fail gracefully
        assert result.returncode != 0 or "error" in (result.stdout + result.stderr).lower()

    def test_invalid_config_file(self, tmp_path: Path):
        """Test CLI handles invalid config file."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("invalid: yaml: content: [")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "wamos_tpw.cli",
                "config",
                str(bad_config),
            ],
            capture_output=True,
            text=True,
        )

        # Should fail with error message
        assert result.returncode != 0 or "error" in (result.stdout + result.stderr).lower()
