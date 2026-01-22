#! /usr/bin/env python3
#
# Single file processing pipeline for WAMOS polar data
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from wamos_tpw.config import Config

from wamos_tpw.polarfile import PolarFile
from wamos_tpw.frame_pipeline import FramePipeline


logger = logging.getLogger(__name__)


class FilePipeline:
    """
    Single WAMOS polar file processing pipeline.

    Processes all frames in a polar file through the FramePipeline.
    """

    def __init__(
        self,
        filename: str,
        config: Config | None = None,
        qSave: bool = False,
        qTiming: bool = False,
    ) -> None:
        """
        Process all frames in a polar file.

        Args:
            filename: Name of polar file to be processed
            config: YAML configuration information
            qSave: Save intermediate results for debugging
            qTiming: Time each processing step
        """
        self._filename: str = filename
        self._config: Config | None = config
        self._timings: dict[str, float] = {}
        self._polarfile: PolarFile | None = None
        self._frames: list[FramePipeline] = []

        try:
            t0 = time.perf_counter() if qTiming else None
            pf = PolarFile(filename, config=config)
            self._polarfile = pf if qSave else None
            if t0 is not None:
                self._timings["PolarFile"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            for frame in pf:
                frm = FramePipeline(frame, config=config, qSave=qSave, qTiming=qTiming)
                self._frames.append(frm)

            if t0 is not None:
                self._timings["Frames"] = time.perf_counter() - t0
        except Exception:
            logger.exception("Error in FilePipeline processing %s", filename)
            raise

    def __repr__(self) -> str:
        return f"<FilePipeline filename={self._filename} frames={len(self._frames)}>"

    def __bool__(self) -> bool:
        """Return True if the file pipeline has processed frames."""
        return len(self._frames) > 0

    def __len__(self) -> int:
        """Return the number of processed frames."""
        return len(self._frames)

    def __iter__(self) -> Iterator[FramePipeline]:
        """Iterate over the processed frames."""
        return iter(self._frames)

    @property
    def timings(self) -> dict[str, float]:
        """Return the file processing timings."""
        return self._timings

    @property
    def filename(self) -> str:
        """Return the polar file name."""
        return self._filename

    @property
    def config(self) -> Config | None:
        """Return the configuration object."""
        return self._config

    @property
    def frames(self) -> list[FramePipeline]:
        """Return the list of processed frames."""
        return self._frames


def _process_file(filepath: str, config: "Config", qTiming: bool, qSave: bool) -> dict:
    """Process a single file and return results with memory usage."""
    import resource

    fp = FilePipeline(filepath, config=config, qTiming=qTiming, qSave=qSave)
    frame_timings = [frm.timings for frm in fp.frames] if qTiming else []
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "filepath": filepath,
        "n_frames": len(fp),
        "file_timings": fp.timings,
        "frame_timings": frame_timings,
        "peak_rss": peak_rss,
    }


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument("--timing", "-t", action="store_true", help="Show timing statistics")
    parser.add_argument("--save", "-s", action="store_true", help="Save intermediate results")
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Number of parallel workers (default: auto)"
    )


def add_subparser(subparsers) -> None:
    """Register the 'file-pipeline' subcommand."""
    p = subparsers.add_parser(
        "file-pipeline",
        help="Process all frames in polar files",
        description="Test file processing pipeline with parallel execution",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'file-pipeline' command with parallel processing."""
    from functools import partial

    from wamos_tpw.config import Config
    from wamos_tpw.filenames import Filenames
    from wamos_tpw.parallel_runner import (
        aggregate_timings,
        display_benchmark_header,
        display_memory_stats,
        display_timing_stats,
        run_benchmark,
    )

    config = Config(args.config) if args.config else Config()

    filenames = Filenames(args.stime, args.etime, args.polar_path)
    files = list(filenames)

    if not files:
        logging.warning(
            "No files found in %s for time range %s to %s", args.polar_path, args.stime, args.etime
        )
        return

    logging.info("Found %d files to process", len(files))

    process_func = partial(
        _process_file,
        config=config,
        qTiming=args.timing,
        qSave=args.save,
    )

    for bench in run_benchmark(
        items=files,
        process_func=process_func,
        n_workers=args.workers,
        item_desc="file",
        get_rss=lambda r: r["peak_rss"],
    ):
        # Count total frames
        total_frames = sum(r["n_frames"] for r in bench.results)
        for r in bench.results:
            if r["n_frames"] == 0:
                logging.warning("No frames in %s", r["filepath"])

        display_benchmark_header(
            executor_name=bench.executor_name,
            n_items=len(files),
            item_label="Files",
            total_count=total_frames,
            count_label="Frames",
            elapsed=bench.elapsed,
            n_workers=bench.n_workers,
        )

        if args.timing:
            all_timings = aggregate_timings(
                bench.results,
                get_timings=lambda r: r["frame_timings"],
            )
            display_timing_stats(all_timings)

        display_memory_stats(bench.max_worker_rss)


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test file processing pipeline")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
