#! /usr/bin/env python3
#
# List frame timestamps and repeat times for polar files
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from wamos_tpw.config import Config


logger = logging.getLogger(__name__)


def list_frames(
    stime: str,
    etime: str,
    polar_path: str,
    config: "Config | None" = None,
) -> Iterator[dict]:
    """
    Yield frame information for all frames in a time interval.

    Args:
        stime: Start time (ISO format or YYYYMMDDTHHmm)
        etime: End time (ISO format or YYYYMMDDTHHmm)
        polar_path: Path to polar files directory
        config: Optional configuration object

    Yields:
        Dict with keys: filepath, frame_index, timestamp, repeat_time
    """
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.polarfile import PolarFile

    filenames = Filenames(stime, etime, polar_path)

    for filepath in filenames:
        try:
            pf = PolarFile(filepath, config=config)
            for frame_idx, frame in enumerate(pf):
                meta = frame.metadata
                yield {
                    "filepath": filepath,
                    "frame_index": frame_idx,
                    "timestamp": meta.timestamp,
                    "repeat_time": meta.repeat_time,
                }
        except Exception as e:
            logger.warning("Error reading %s: %s", filepath, e)


def print_frames(
    stime: str,
    etime: str,
    polar_path: str,
    config: "Config | None" = None,
) -> None:
    """
    Print frame timestamps and repeat times to stdout.

    Args:
        stime: Start time (ISO format or YYYYMMDDTHHmm)
        etime: End time (ISO format or YYYYMMDDTHHmm)
        polar_path: Path to polar files directory
        config: Optional configuration object
    """
    prev_ts = None

    for info in list_frames(stime, etime, polar_path, config):
        ts = info["timestamp"]
        repeat = info["repeat_time"]

        # Calculate gap from previous frame
        if prev_ts is not None:
            gap = (ts - prev_ts) / np.timedelta64(1, "s")
            gap_str = f"  gap={gap:.3f}s"
        else:
            gap_str = ""

        print(f"{np.datetime_as_string(ts, unit='ms')}  repeat={repeat:.3f}s{gap_str}")
        prev_ts = ts


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")


def add_subparser(subparsers) -> None:
    """Register the 'list-frames' subcommand."""
    p = subparsers.add_parser(
        "list-frames",
        help="List frame timestamps and repeat times",
        description="Print timestamp and repeat_time for every frame in a time interval",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'list-frames' command."""
    from wamos_tpw.config import Config

    config = Config(args.config) if args.config else None
    print_frames(args.stime, args.etime, args.polar_path, config)


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="List frame timestamps and repeat times")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
