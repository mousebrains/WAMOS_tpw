#!/usr/bin/env python3
"""
Check and visualize destreaking of WAMOS polar frames.

This script loads polar files for a given time range and displays them
frame by frame as pcolor plots with range on x-axis and bearing on y-axis.
Includes a Destreak class for removing radial streaks from the data.

Usage:
    python destreak_check.py 20220405T010000 20220405T010100 /path/to/POLAR
    python destreak_check.py 20220405T010000 20220405T010100 /path/to/POLAR --destreak

Dec-2025, Pat Welch, pat@mousebrains.com
        in collaboration with Anthropic's Claude Code
"""

import argparse
import logging
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from wamos_tpw import Filenames, PolarFile  # noqa: E402
from wamos_tpw.config import Config  # noqa: E402
from wamos_tpw.destreak import Destreak  # noqa: E402
from wamos_tpw.filenames import add_common_arguments  # noqa: E402
from wamos_tpw.logging_config import add_logging_arguments, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_frame(fn: str, config: Config, save_mask: bool = True, stats_only: bool = False) -> dict | None:
    """
    Worker function to load a single frame.

    Args:
        fn: Filename to load
        config: Configuration object
        save_mask: Whether to save the streak mask (needed for plotting)
        stats_only: If True, return only timing stats (reduces memory)

    Returns:
        Dictionary with frame info, or None on error.
    """
    try:
        t0 = time.perf_counter()
        pf = PolarFile(fn, config=config)
        t1 = time.perf_counter()
        frame = pf.frame()
        t2 = time.perf_counter()
        destreaked = Destreak(frame, save_mask=save_mask)
        t3 = time.perf_counter()

        result = {
            "filename": fn,
            "timestamp": frame.timestamp,
            "timing": {
                "polarfile": t1 - t0,
                "frame": t2 - t1,
                "destreak": t3 - t2,
                "total": t3 - t0,
            },
            "substeps": {
                "polarfile": pf.timing,
                "destreak": destreaked.timing,
            },
        }

        # Only store large objects if needed for plotting
        if not stats_only:
            result["frame"] = frame
            result["destreak"] = destreaked

        return result
    except Exception:
        logger.exception("Failed to load %s", fn)
        return None


def print_progress(current: int, total: int, width: int = 40, prefix: str = "Progress") -> None:
    """
    Print a progress bar that updates in place.

    Args:
        current: Current progress count
        total: Total count
        width: Width of the progress bar in characters
        prefix: Prefix text
    """
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    msg = f"\r{prefix}: [{bar}] {current:>{len(str(total))}}/{total} ({pct * 100:.1f}%)"
    print(msg, end="", flush=True)
    if current == total:
        print()  # Newline when complete


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and visualize destreaking of polar frames")

    add_common_arguments(parser)
    add_logging_arguments(parser)

    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument(
        "--progress",
        action="store_true",
        dest="progress_flag",
        help="Enable progress bar",
    )
    grp.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress_flag",
        help="Disable progress bar",
    )
    parser.set_defaults(progress_flag=True)

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of workers (default: CPU count = {os.cpu_count()})",
    )
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor (avoids GIL)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum intensity value for colormap",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum intensity value for colormap",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap to use (default: viridis)",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size as 'width,height' in inches (default: 12,8)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save frames to files instead of displaying (use %%d for frame number)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots (default: 150)",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=0,
        help="Starting range bin (default: 0)",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="Ending range bin (default: all)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show interactive plot viewer",
    )

    args = parser.parse_args()
    setup_logging(args)

    try:
        filenames = Filenames(args.stime, args.etime, str(args.polar_path))
        files = filenames.files
        n_files = len(files)

        if n_files == 0:
            logger.error("No files found in specified time range")
            return 1

        executor_type = "processes" if args.processes else "threads"
        logger.info(
            "Loading %d files with %d %s",
            n_files,
            args.workers or os.cpu_count(),
            executor_type,
        )

        # Load frames in parallel
        results: list[dict] = []
        config = Config(args.config)
        needs_plot_data = args.plot or args.output
        save_mask = needs_plot_data  # Only save mask if we're plotting
        stats_only = not needs_plot_data  # Don't store frames if just collecting stats

        t0 = time.perf_counter()
        ExecutorClass = ProcessPoolExecutor if args.processes else ThreadPoolExecutor
        with ExecutorClass(max_workers=args.workers) as executor:
            futures = {
                executor.submit(load_frame, fn, config, save_mask, stats_only): fn for fn in files
            }

            for i, future in enumerate(as_completed(futures)):
                if args.progress_flag:
                    print_progress(i + 1, n_files, prefix="Loading")
                result = future.result()
                if result is not None:
                    results.append(result)

        elapsed = time.perf_counter() - t0
        n_loaded = len(results)
        fps = n_loaded / elapsed if elapsed > 0 else 0
        print(
            f"Successfully loaded {n_loaded} of {n_files} frames "
            f"in {elapsed:.2f}s ({fps:.1f} frames/sec)"
        )

        # Print timing breakdown
        if n_loaded > 0:
            timing_keys = ["polarfile", "frame", "destreak"]
            totals = {k: sum(r["timing"][k] for r in results) for k in timing_keys}
            total_time = sum(totals.values())
            print("\n=== Per-Step Processing Statistics ===")
            print(f"{'Step':<12} {'Time (ms)':<12} {'Time %':<10}")
            print("-" * 34)
            for key in timing_keys:
                avg_ms = (totals[key] / n_loaded) * 1000
                pct = (totals[key] / total_time) * 100 if total_time > 0 else 0
                print(f"{key:<12} {avg_ms:<12.3f} {pct:<10.1f}")

            # Print sub-step timing breakdown
            print("\n=== Sub-Step Timing Breakdown ===")
            for step in ["polarfile", "destreak"]:
                # Aggregate substeps across all results
                substep_totals: dict[str, list[float]] = {}
                for r in results:
                    substeps = r.get("substeps", {}).get(step, {})
                    for sub_name, sub_time in substeps.items():
                        if sub_name not in substep_totals:
                            substep_totals[sub_name] = []
                        substep_totals[sub_name].append(sub_time)

                if substep_totals:
                    step_total = totals[step]
                    print(f"\n{step}:")
                    for sub_name, sub_times in sorted(substep_totals.items()):
                        if sub_times:
                            avg_sub_ms = np.mean(sub_times) * 1000
                            sub_total = sum(sub_times)
                            sub_pct = (sub_total / step_total * 100) if step_total > 0 else 0
                            print(f"  {sub_name:<20} {avg_sub_ms:>8.3f} ms  ({sub_pct:>5.1f}%)")

        if n_loaded == 0:
            logger.error("No valid frames loaded")
            return 1

        if not args.plot and not args.output:
            # Report peak memory usage and exit (no plotting requested)
            peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS, ru_maxrss is in bytes; on Linux it's in KB
            if sys.platform == "darwin":
                peak_mem_mb = peak_mem / (1024 * 1024)
            else:
                peak_mem_mb = peak_mem / 1024
            print(f"Peak memory: {peak_mem_mb:.1f} MB")
            return 0

        # Sort by timestamp
        def sort_key(x: dict) -> np.datetime64:
            ts = x.get("timestamp")
            return ts if ts is not None else np.datetime64(0, "ns")

        results.sort(key=sort_key)

        # Parse figure size
        try:
            figsize = tuple(float(x) for x in args.figsize.split(","))
        except ValueError:
            logger.error("Invalid figsize: %s", args.figsize)
            return 1

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        if args.output:
            # Batch mode: save all frames to files
            for i, r in enumerate(results):
                frame = r["frame"]
                destreaker = r["destreak"]
                timestamp = r["timestamp"]

                range_start = args.range_start
                original = frame.intensity
                range_end = args.range_end if args.range_end else original.shape[1]
                range_end = min(range_end, original.shape[1])

                if range_end <= range_start:
                    logger.warning("Invalid range for frame %d", i)
                    continue

                original_slice = original[:, range_start:range_end]
                n_bearings, n_ranges = original_slice.shape
                range_bins = np.arange(range_start, range_start + n_ranges + 1)
                bearing_bins = np.arange(n_bearings + 1)

                destreaked_slice = destreaker.intensity[:, range_start:range_end]
                streak_mask_slice = destreaker.streak_mask[:, range_start:range_end]
                n_streaks = destreaker.n_streak_pixels
                streak_pct = destreaker.streak_fraction * 100

                fig, axes = plt.subplots(
                    1, 3, figsize=(figsize[0] * 1.3, figsize[1] * 0.6), sharex=True, sharey=True
                )

                pcm0 = axes[0].pcolormesh(
                    range_bins,
                    bearing_bins,
                    original_slice,
                    shading="flat",
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )
                axes[0].set_xlabel("Range Bin")
                axes[0].set_ylabel("Bearing Index")
                axes[0].set_title("Original")
                fig.colorbar(pcm0, ax=axes[0], label="Intensity")

                pcm1 = axes[1].pcolormesh(
                    range_bins,
                    bearing_bins,
                    streak_mask_slice.astype(float),
                    shading="flat",
                    cmap="Reds",
                    vmin=0,
                    vmax=1,
                )
                axes[1].set_xlabel("Range Bin")
                axes[1].set_ylabel("Bearing Index")
                axes[1].set_title(f"Streak Mask ({n_streaks} pixels, {streak_pct:.2f}%)")
                fig.colorbar(pcm1, ax=axes[1], label="Streak")

                pcm2 = axes[2].pcolormesh(
                    range_bins,
                    bearing_bins,
                    destreaked_slice,
                    shading="flat",
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )
                axes[2].set_xlabel("Range Bin")
                axes[2].set_ylabel("Bearing Index")
                axes[2].set_title("Destreaked")
                fig.colorbar(pcm2, ax=axes[2], label="Intensity")

                suptitle = f"Frame {i + 1}/{n_loaded}: {timestamp}"
                suptitle += (
                    f"\nShape: {n_bearings} bearings x {n_ranges} ranges, {streak_pct:.2f}% streaks"
                )
                fig.suptitle(suptitle)

                fig.tight_layout()

                if "%d" in args.output:
                    out_path = args.output % (i + 1)
                else:
                    out_path = Path(args.output)
                    out_path = out_path.parent / f"{out_path.stem}_{i + 1:04d}{out_path.suffix}"
                fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved: {out_path}")
                plt.close(fig)

            print(f"\nSaved {n_loaded} frames")

        elif args.plot:
            # Interactive mode with navigation buttons
            class FrameViewer:
                """Interactive viewer for stepping through destreaked frames."""

                def __init__(
                    self,
                    results: list[dict],
                    range_start: int,
                    range_end: int | None,
                    cmap: str,
                    vmin: float | None,
                    vmax: float | None,
                    figsize: tuple,
                ) -> None:
                    self.results = results
                    self.n_frames = len(results)
                    self.current_idx = 0
                    self.range_start = range_start
                    self.range_end = range_end
                    self.cmap = cmap
                    self.vmin = vmin
                    self.vmax = vmax
                    self.figsize = figsize
                    self.playing = False
                    self.timer: object | None = None

                    # Create figure with 3 panels: original, mask, destreaked
                    self.fig, self.axes = plt.subplots(
                        1,
                        3,
                        figsize=(figsize[0] * 1.3, figsize[1] * 0.7),
                        sharex=True,
                        sharey=True,
                    )

                    # Add navigation buttons
                    plt.subplots_adjust(bottom=0.15)
                    ax_prev = plt.axes((0.3, 0.02, 0.1, 0.05))
                    ax_play = plt.axes((0.45, 0.02, 0.1, 0.05))
                    ax_next = plt.axes((0.6, 0.02, 0.1, 0.05))

                    self.btn_prev = Button(ax_prev, "Previous")
                    self.btn_play = Button(ax_play, "Play")
                    self.btn_next = Button(ax_next, "Next")

                    self.btn_prev.on_clicked(self.prev_frame)
                    self.btn_play.on_clicked(self.toggle_play)
                    self.btn_next.on_clicked(self.next_frame)

                    # Store colorbars for removal on update
                    self.colorbars: list = []

                    # Initial plot
                    self.update_plot()

                def update_plot(self) -> None:
                    """Update the plot with current frame."""
                    # Clear existing colorbars
                    for cbar in self.colorbars:
                        cbar.remove()
                    self.colorbars.clear()

                    # Clear axes
                    for ax in self.axes:
                        ax.clear()

                    r = self.results[self.current_idx]
                    frame = r["frame"]
                    timestamp = r["timestamp"]

                    original = frame.intensity
                    rng_end = self.range_end if self.range_end else original.shape[1]
                    rng_end = min(rng_end, original.shape[1])

                    original_slice = original[:, self.range_start : rng_end]
                    n_bearings, n_ranges = original_slice.shape
                    range_bins = np.arange(self.range_start, self.range_start + n_ranges + 1)
                    bearing_bins = np.arange(n_bearings + 1)

                    destreaker = r["destreak"]
                    destreaked_slice = destreaker.intensity[:, self.range_start : rng_end]
                    streak_mask_slice = destreaker.streak_mask[:, self.range_start : rng_end]
                    n_streaks = destreaker.n_streak_pixels
                    streak_pct = destreaker.streak_fraction * 100

                    pcm0 = self.axes[0].pcolormesh(
                        range_bins,
                        bearing_bins,
                        original_slice,
                        shading="flat",
                        cmap=self.cmap,
                        vmin=self.vmin,
                        vmax=self.vmax,
                    )
                    self.axes[0].set_xlabel("Range Bin")
                    self.axes[0].set_ylabel("Bearing Index")
                    self.axes[0].set_title("Original")
                    self.colorbars.append(self.fig.colorbar(pcm0, ax=self.axes[0]))

                    pcm1 = self.axes[1].pcolormesh(
                        range_bins,
                        bearing_bins,
                        streak_mask_slice.astype(float),
                        shading="flat",
                        cmap="Reds",
                        vmin=0,
                        vmax=1,
                    )
                    self.axes[1].set_xlabel("Range Bin")
                    self.axes[1].set_ylabel("Bearing Index")
                    self.axes[1].set_title(
                        f"Streak Mask ({n_streaks} pixels, {streak_pct:.2f}%)"
                    )
                    self.colorbars.append(self.fig.colorbar(pcm1, ax=self.axes[1]))

                    pcm2 = self.axes[2].pcolormesh(
                        range_bins,
                        bearing_bins,
                        destreaked_slice,
                        shading="flat",
                        cmap=self.cmap,
                        vmin=self.vmin,
                        vmax=self.vmax,
                    )
                    self.axes[2].set_xlabel("Range Bin")
                    self.axes[2].set_ylabel("Bearing Index")
                    self.axes[2].set_title("Destreaked")
                    self.colorbars.append(self.fig.colorbar(pcm2, ax=self.axes[2]))

                    suptitle = f"Frame {self.current_idx + 1}/{self.n_frames}: {timestamp}"
                    suptitle += f"\nShape: {n_bearings} x {n_ranges}"
                    self.fig.suptitle(suptitle)

                    self.fig.canvas.draw_idle()

                def next_frame(self, event: object = None) -> None:
                    """Go to next frame."""
                    if self.current_idx < self.n_frames - 1:
                        self.current_idx += 1
                        self.update_plot()

                def prev_frame(self, event: object = None) -> None:
                    """Go to previous frame."""
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        self.update_plot()

                def toggle_play(self, event: object = None) -> None:
                    """Toggle play/pause animation."""
                    if self.playing:
                        self.playing = False
                        self.btn_play.label.set_text("Play")
                        if self.timer is not None and hasattr(self.timer, "stop"):
                            self.timer.stop()  # type: ignore[union-attr]
                    else:
                        self.playing = True
                        self.btn_play.label.set_text("Pause")
                        self.timer = self.fig.canvas.new_timer(interval=500)
                        self.timer.add_callback(self.play_step)
                        self.timer.start()

                def play_step(self) -> None:
                    """Advance one frame during playback."""
                    if self.current_idx < self.n_frames - 1:
                        self.current_idx += 1
                        self.update_plot()
                    else:
                        # Stop at end
                        self.toggle_play()

                def show(self) -> None:
                    """Display the interactive viewer."""
                    plt.show()

            # Create and show the viewer
            viewer = FrameViewer(
                results=results,
                range_start=args.range_start,
                range_end=args.range_end,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                figsize=figsize,
            )
            viewer.show()

            print(f"\nViewed {n_loaded} frames")

        # Report peak memory usage
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == "darwin":
            peak_mem_mb = peak_mem / (1024 * 1024)
        else:
            peak_mem_mb = peak_mem / 1024
        print(f"Peak memory: {peak_mem_mb:.1f} MB")

        return 0

    except (FileNotFoundError, ValueError, OSError):
        logger.exception("Error loading files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
