#! /usr/bin/env python3
#
# Extract indices of PPS pulses in a frame
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging

import numpy as np
from wamos_tpw.frame import Frame
from wamos_tpw.config import Config

class PPS:
    """
    Extract indices of PPS pulses in a frame.
    """

    _PPS_RANGE_INDEX = 0 # First range bin is PPS bit 12

    def __init__(self, frame: Frame) -> None:
        """ Find the index of each PPS pulse in the frame. """

        bit12 = frame.bit12[:, self._PPS_RANGE_INDEX]
        self._indices = np.where(bit12 != 0)[0]

    def __bool__(self) -> bool:
        """Return True if any PPS pulses were found."""
        return self._indices.size > 0

    @property
    def indices(self) -> np.ndarray:
        """Return the PPS indices"""
        return self._indices

    def __repr__(self) -> str:
        return f"PPS indices({self._indices})"

def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("filename", nargs="+", type=str, help="Polar file to process")
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename to read")

def add_subparser(subparsers) -> None:
    """Register the 'PPS' subcommand."""
    p = subparsers.add_parser(
        "PPS", help="Standalone PPS tool", description="Test PPS on a polar file"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'PPS' command."""
    from wamos_tpw.polarfile import PolarFile

    # Load polar file
    config = Config(args.config)
    for fn in args.filename:
        pf = PolarFile(fn, config=config)
        if not pf:
            logging.warning("No frames in %s", fn)
            continue
        for index, frame in enumerate(pf.frames):
            pps = PPS(frame)
            logging.info("%s Frame(%d): PPS indices: %s",
                         fn,
                         index,
                         pps.indices)

def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test PPS on a polar file")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
