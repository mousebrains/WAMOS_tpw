#! /usr/bin/env python3
#
# Single frame processing pipeline for WAMOS polar data
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from wamos_tpw.frame import Frame, FrameMetadata
from wamos_tpw.theta import Theta
from wamos_tpw.range import Range
from wamos_tpw.destreak import Destreak
from wamos_tpw.shadow import Shadow
from wamos_tpw.deramp import Deramp
from wamos_tpw.dewind import Dewind
from wamos_tpw.pps import PPS

if TYPE_CHECKING:
    from wamos_tpw.config import Config


logger = logging.getLogger(__name__)


class FramePipeline:
    """
    Single WAMOS polar frame processing pipeline.

    Processes a single radar frame through the complete pipeline:
    PPS -> Theta -> Range -> Destreak -> Shadow -> Deramp -> Dewind -> Normalize
    """

    def __init__(
        self,
        frame: Frame,
        config: Config | None = None,
        qSave: bool = False,
        qTiming: bool = False,
    ) -> None:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Frame from a polar file
            config: YAML configuration information
            qSave: Save intermediate results for debugging
            qTiming: Time each processing step
        """
        self._config: Config | None = config
        self._metadata: FrameMetadata = frame.metadata
        self._shape: tuple[int, int] = frame.shape
        self._timings: dict[str, float] = {}
        self._polarfile = None
        self._frame: Frame | None = None
        self._theta: Theta | None = None
        self._range: Range | None = None
        self._destreak: Destreak | None = None
        self._shadow: Shadow | None = None
        self._intensity_shadowed: np.ndarray | None = None
        self._deramp: Deramp | None = None
        self._dewind: Dewind | None = None
        self._pps: PPS | None = None
        # Arrays always stored for earth projection (minimal memory footprint)
        self._theta_array: np.ndarray | None = None  # Beam angles per radial
        self._ground_range: np.ndarray | None = None  # Ground range per distance bin
        self._range_resolution: float = 0.0  # Range resolution in meters
        self._final_intensity: np.ndarray | None = None  # Dewinded intensity

        try:
            t0 = time.perf_counter() if qTiming else None

            self._pps = PPS(frame)
            if t0 is not None:
                self._timings["PPS"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            theta = Theta(frame)
            self._theta = theta if qSave else None
            # Note: theta_array is stored AFTER shadow bias adjustment below

            if t0 is not None:
                self._timings["Theta"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            rng = Range(frame)
            self._range = rng if qSave else None
            self._ground_range = rng.ground_range.copy()  # Always store for projection
            self._range_resolution = rng.range_resolution  # Store range resolution

            if t0 is not None:
                self._timings["Range"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            destreak = Destreak(frame)
            self._destreak = destreak if qSave else None
            if t0 is not None:
                self._timings["Destreak"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            shadow = Shadow(destreak.intensity, theta)
            self._shadow = shadow if qSave else None
            if t0 is not None:
                self._timings["Shadow"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            if shadow.theta_bias is not None and shadow.theta_bias != 0:
                theta.set_bias(shadow.theta_bias)
            # Store theta array AFTER bias adjustment for earth projection
            self._theta_array = theta.theta.copy()
            if t0 is not None:
                self._timings["theta_bias"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            intensity = shadow.mask(destreak.intensity)
            self._intensity_shadowed = intensity if qSave else None
            if t0 is not None:
                self._timings["MaskShadow"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            if not qSave:
                del destreak, shadow

            deramp = Deramp(intensity, rng)
            self._deramp = deramp if qSave else None
            if t0 is not None:
                self._timings["Deramp"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            if not qSave:
                del rng

            dewind = Dewind(deramp.intensity, theta)
            self._dewind = dewind if qSave else None
            if t0 is not None:
                self._timings["Dewind"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            # Normalize intensity to [0, 1] range
            intensity = dewind.intensity
            valid_mask = ~np.isnan(intensity)
            if np.any(valid_mask):
                imin = np.nanmin(intensity)
                imax = np.nanmax(intensity)
                if imax > imin:
                    normalized = (intensity - imin) / (imax - imin)
                else:
                    # All values are the same, set to 0.5
                    normalized = np.where(valid_mask, 0.5, np.nan)
            else:
                # All NaN, keep as is
                normalized = intensity
            self._final_intensity = normalized
            if t0 is not None:
                self._timings["Normalize"] = time.perf_counter() - t0
                t0 = time.perf_counter()

            if not qSave:
                del theta, deramp
        except Exception:
            logger.exception("Error in FramePipeline processing")
            raise

    @property
    def timings(self) -> dict[str, float]:
        """Return the processing step timings."""
        return self._timings

    @property
    def metadata(self) -> FrameMetadata:
        """Return the frame metadata."""
        return self._metadata

    @property
    def config(self) -> Config | None:
        """Return the configuration object."""
        return self._config

    @property
    def frame(self) -> Frame | None:
        """Return the original frame."""
        return self._frame

    @property
    def deramp(self) -> Deramp | None:
        """Return the deramped data."""
        return self._deramp

    @property
    def dewind(self) -> Dewind | None:
        """Return the dewinded data."""
        return self._dewind

    @property
    def pps(self) -> PPS | None:
        """Return the PPS pulse indices."""
        return self._pps

    @property
    def intensity_shadowed(self) -> np.ndarray | None:
        """Return the shadow-masked intensity data."""
        return self._intensity_shadowed

    @property
    def shadow(self) -> Shadow | None:
        """Return the shadow detection results."""
        return self._shadow

    @property
    def destreak(self) -> Destreak | None:
        """Return the destreaked data."""
        return self._destreak

    @property
    def theta(self) -> Theta | None:
        """Return the beam angle array."""
        return self._theta

    @property
    def range(self) -> Range | None:
        """Return the range object."""
        return self._range

    @property
    def shape(self) -> tuple[int, int]:
        """Return the frame shape (n_bearings, n_distances)."""
        return self._shape

    @property
    def n_bearings(self) -> int:
        """Return the number of bearings (radials)."""
        return self._shape[0]

    @property
    def n_distances(self) -> int:
        """Return the number of distance bins."""
        return self._shape[1]

    # Properties for earth projection
    @property
    def theta_array(self) -> np.ndarray:
        """Return the beam angle array (degrees) for each radial."""
        return self._theta_array

    @property
    def ground_range(self) -> np.ndarray:
        """Return the ground range array (meters) for each distance bin."""
        return self._ground_range

    @property
    def range_resolution(self) -> float:
        """Return the range resolution in meters per bin."""
        return self._range_resolution

    @property
    def final_intensity(self) -> np.ndarray:
        """Return the normalized intensity array [0, 1] (destreaked, shadow-masked, deramped, dewinded, normalized)."""
        return self._final_intensity

    def __repr__(self) -> str:
        return f"<FramePipeline timestamp={self._metadata.timestamp}>"


def process_polar_file(
    filename: str,
    config: "Config | None" = None,
    qSave: bool = False,
    qTiming: bool = False,
) -> list[FramePipeline]:
    """
    Process all frames in a polar file through the pipeline.

    This is a convenience function that processes each frame in a polar file
    through the FramePipeline.

    Args:
        filename: Path to the polar file to process
        config: Optional YAML configuration object
        qSave: Save intermediate results for debugging
        qTiming: Time each processing step

    Returns:
        List of FramePipeline objects, one per frame in the file

    Example:
        >>> from wamos_tpw.frame_pipeline import process_polar_file
        >>> frames = process_polar_file("20241215103045.pol.gz")
        >>> for fp in frames:
        ...     print(fp.timestamp, fp.final_intensity.shape)
    """
    from wamos_tpw.polarfile import PolarFile

    pf = PolarFile(filename, config=config)
    return [FramePipeline(frame, config=config, qSave=qSave, qTiming=qTiming) for frame in pf]


def _process_frame(
    filepath: str, frame_index: int, config: "Config", qTiming: bool, qSave: bool
) -> dict:
    """Process a single frame from a file and return results with memory usage."""
    import resource
    from wamos_tpw.polarfile import PolarFile

    pf = PolarFile(filepath, config=config)
    frames = list(pf)
    if frame_index >= len(frames):
        return {
            "filepath": filepath,
            "frame_index": frame_index,
            "timings": {},
            "peak_rss": 0,
            "success": False,
        }

    frame = frames[frame_index]
    fp = FramePipeline(frame, config=config, qSave=qSave, qTiming=qTiming)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "filepath": filepath,
        "frame_index": frame_index,
        "timings": fp.timings,
        "peak_rss": peak_rss,
        "success": True,
    }


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    from wamos_tpw.filenames import add_common_arguments

    add_common_arguments(parser)
    parser.add_argument("--config", "-c", type=str, help="Config YAML filename")
    parser.add_argument("--frame", "-f", type=int, default=0, help="Frame index to process")
    parser.add_argument("--timing", "-t", action="store_true", help="Show timing statistics")
    parser.add_argument("--save", "-s", action="store_true", help="Save intermediate results")
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Number of parallel workers (default: auto)"
    )


def add_subparser(subparsers) -> None:
    """Register the 'frame-pipeline' subcommand."""
    p = subparsers.add_parser(
        "frame-pipeline",
        help="Process frames through the pipeline",
        description="Test single frame processing pipeline with parallel execution",
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'frame-pipeline' command with parallel processing."""
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

    frame_index = args.frame
    process_func = partial(
        _process_frame,
        frame_index=frame_index,
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
        # Count successful frames
        successful = [r for r in bench.results if r["success"]]
        for r in bench.results:
            if not r["success"]:
                logging.warning("No frame %d in %s", frame_index, r["filepath"])

        display_benchmark_header(
            executor_name=bench.executor_name,
            n_items=len(files),
            item_label="Files",
            total_count=len(successful),
            count_label="Frames",
            elapsed=bench.elapsed,
            n_workers=bench.n_workers,
            extra_lines=[f"Frame index: {frame_index}"],
        )

        if args.timing:
            all_timings = aggregate_timings(
                successful,
                get_timings=lambda r: [r["timings"]] if r["timings"] else [],
            )
            display_timing_stats(all_timings)

        display_memory_stats(bench.max_worker_rss)


from wamos_tpw.cli_utils import create_standalone_main  # noqa: E402

main = create_standalone_main(_add_arguments, run, "Test single frame processing pipeline")

if __name__ == "__main__":
    main()
