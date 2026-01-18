#! /usr/bin/env python3
#
# Destreak class for removing radial streak artifacts from WAMOS polar frames
#
# Dec-2025, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import time as _time

import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage

from wamos_tpw.config import Config
from wamos_tpw.frame import Frame


class Destreak:
    """
    Remove radial streak artifacts from radar frames using angular gradient analysis.

    Streaks appear as radial lines of anomalously high intensity,
    typically caused by interference or hardware artifacts. The algorithm
    detects streaks by looking for sharp intensity transitions along the
    bearing direction (spikes that go up then down quickly).

    Algorithm:
        1. Apply 2D convolution with edge-detection kernel to find intensity spikes
        2. Apply 2D convolution with adjacent-average kernel for replacement values
        3. Threshold based on sigma * std(convolution) to identify candidate streaks
        4. Require streak pixels to have negative response in adjacent bearings
        5. Merge small gaps (<=max_gap_length) in detected streak segments
        6. Filter out short segments (<min_streak_length contiguous bins)
        7. Replace streak pixels with adjacent-bearing average

    Config Parameters:
        destreak.min_streak_length: Minimum contiguous bins to confirm streak (default: 10)
        destreak.max_gap_length: Maximum gap length to merge segments (default: 4)
        destreak.threshold_sigma: Threshold in std deviations (default: 3.0)

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> destreak = Destreak(frame)
        >>> corrected = destreak.intensity
        >>> print(f"Detected {destreak.n_streak_pixels} streak pixels ({destreak.streak_fraction:.2%})")
    """

    # Constants
    _MIN_STREAK_LENGTH = 10  # Minimum contiguous bins to confirm streak
    _MAX_GAP_LENGTH = 4  # Maximum gap length to merge streak segments
    _THRESHOLD_SIGMA = 3.0  # Threshold in terms of one-sided std deviations

    def __init__(self, frame: Frame, save_mask: bool = False) -> None:
        """
        Initialize destreaking for a frame.

        Args:
            frame: Frame to be destreaked (config is obtained from frame.config)
            save_mask: If True, store the streak mask for later access via streak_mask
                       property. Set to True when plotting diagnostics. (default: False)
        """
        if frame is None:
            raise ValueError("frame is required")

        self._config = frame.config
        self._timing: dict[str, float] = {}
        config = self._config

        # Get parameters from config (with class defaults as fallback)
        min_streak_length = config.get("destreak.min_streak_length", self._MIN_STREAK_LENGTH)
        max_gap_length = config.get("destreak.max_gap_length", self._MAX_GAP_LENGTH)
        threshold_sigma = config.get("destreak.threshold_sigma", self._THRESHOLD_SIGMA)

        # 2D convolution kernel for streak detection
        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
        kAdjacent = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        kAdjacent = kAdjacent / kAdjacent.sum()  # Normalize

        t0 = _time.perf_counter()
        intensity = frame.intensity.astype(np.float32)  # Signed for calculation
        a = convolve2d(intensity, kernel, mode="same", boundary="wrap")
        b = convolve2d(intensity, kAdjacent, mode="same", boundary="wrap")
        self._timing["convolve"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        sigma = np.std(a)
        thres_center = threshold_sigma * sigma
        thres_adjacent = thres_center / 2

        # We expect a large positive for a single radial streak
        # We expect large negative on either side for adjacent streaks

        qAdjacent = a <= -thres_adjacent
        qCenter = a >= thres_center
        q = qCenter & np.roll(qAdjacent, +1, axis=0) & np.roll(qAdjacent, -1, axis=0)
        self._timing["threshold"] = _time.perf_counter() - t0

        qAny = np.any(q, axis=1)

        # Early exit if no candidate streaks detected
        if not qAny.any():
            self._n_streak_pixels = 0
            self._streak_mask = q if save_mask else None
            self._corrected_intensity = intensity
            self._timing["label"] = 0.0
            self._timing["replace"] = 0.0
            return

        t0 = _time.perf_counter()
        qStreaks = q[qAny, :]

        kernelHorizontal = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

        # Find short False gaps to flip to True
        [labels, _] = ndimage.label(~qStreaks, structure=kernelHorizontal)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qShort = (RL <= max_gap_length) & ~qStreaks
        qStreaks |= qShort  # Flip short false gaps to true

        # Find streaks
        [labels, _] = ndimage.label(qStreaks, structure=kernelHorizontal)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qStreaks &= RL >= min_streak_length  # Keep only long enough streaks

        q[qAny] = qStreaks  # Update original mask
        self._timing["label"] = _time.perf_counter() - t0

        # Store statistics
        self._n_streak_pixels = int(q.sum())
        self._streak_mask = q if save_mask else None

        t0 = _time.perf_counter()
        destreaked = intensity.copy()
        destreaked[q] = b[q]  # Replace streak pixels with adjacent average
        self._timing["replace"] = _time.perf_counter() - t0

        self._corrected_intensity = destreaked

    @property
    def intensity(self) -> np.ndarray:
        """Return the destreaked intensity array."""
        return self._corrected_intensity

    @property
    def streak_mask(self) -> np.ndarray | None:
        """
        Return boolean mask indicating detected streaks.

        Returns:
            2D boolean array (True = streak detected), or None if save_mask=False
        """
        return self._streak_mask

    @property
    def n_streak_pixels(self) -> int:
        """Return the number of pixels detected as streaks."""
        return self._n_streak_pixels

    @property
    def streak_fraction(self) -> float:
        """Return the fraction of pixels detected as streaks."""
        total = self._corrected_intensity.size
        return self._n_streak_pixels / total if total > 0 else 0.0

    @property
    def config(self) -> Config:
        """Return the configuration object."""
        return self._config

    @property
    def timing(self) -> dict[str, float]:
        """Return timing information for sub-steps (convolve, threshold, label, replace)."""
        return self._timing

    def __repr__(self) -> str:
        return f"Destreak(streaks={self._n_streak_pixels} ({self.streak_fraction:.2%}))"


class DestreakDiag:
    """
    Diagnostic visualization for destreaking results.

    Provides plotting and statistics for comparing original and destreaked frames.

    Example:
        >>> from wamos_tpw.polarfile import PolarFile
        >>> pf = PolarFile('20241215103045.pol.gz')
        >>> frame = pf.frame()
        >>> destreak = Destreak(frame, save_mask=True)
        >>> diag = DestreakDiag(frame, destreak)
        >>> diag.plot()
    """

    def __init__(self, frame: Frame, destreak: Destreak) -> None:
        """
        Initialize diagnostic viewer.

        Args:
            frame: Original frame before destreaking
            destreak: Destreak object with corrected intensity
        """
        self._frame = frame
        self._destreak = destreak

    @property
    def frame(self) -> Frame:
        """Return the original frame."""
        return self._frame

    @property
    def destreak(self) -> Destreak:
        """Return the Destreak object."""
        return self._destreak

    def plot(
        self,
        figsize: tuple[float, float] = (14, 5),
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """
        Plot diagnostic comparison of before and after destreaking.

        Creates a 1x3 figure showing:
        - Left: Original intensity
        - Center: Streak mask (if saved)
        - Right: Destreaked intensity

        Args:
            figsize: Figure size as (width, height) in inches
            cmap: Colormap for intensity plots
            vmin: Minimum intensity value for colormap (auto if None)
            vmax: Maximum intensity value for colormap (auto if None)
        """
        import matplotlib.pyplot as plt

        original = self._frame.intensity.astype(np.float32)
        corrected = self._destreak.intensity

        # Auto-scale if not specified
        if vmin is None:
            vmin = min(original.min(), corrected.min())
        if vmax is None:
            vmax = max(original.max(), corrected.max())

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

        # Original
        im0 = axes[0].imshow(original, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title("Original")
        axes[0].set_xlabel("Distance bin")
        axes[0].set_ylabel("Bearing bin")
        plt.colorbar(im0, ax=axes[0], label="Intensity")

        # Streak mask
        streak_mask = self._destreak.streak_mask
        if streak_mask is not None:
            axes[1].imshow(streak_mask, aspect="auto", cmap="Reds")
            axes[1].set_title(
                f"Streak Mask ({self._destreak.n_streak_pixels} pixels, "
                f"{self._destreak.streak_fraction:.2%})"
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "Mask not saved\n(use save_mask=True)",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title("Streak Mask (not saved)")
        axes[1].set_xlabel("Distance bin")
        axes[1].set_ylabel("Bearing bin")

        # Destreaked
        im2 = axes[2].imshow(corrected, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title("Destreaked")
        axes[2].set_xlabel("Distance bin")
        axes[2].set_ylabel("Bearing bin")
        plt.colorbar(im2, ax=axes[2], label="Intensity")

        fig.suptitle(
            f"Destreak: {self._frame.timestamp} | "
            f"Streaks: {self._destreak.n_streak_pixels} ({self._destreak.streak_fraction:.2%})"
        )
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"DestreakDiag(frame={self._frame.timestamp}, "
            f"streaks={self._destreak.n_streak_pixels} ({self._destreak.streak_fraction:.2%}))"
        )


def destreak_frame(frame: Frame) -> np.ndarray:
    """
    Convenience function to destreak a single frame.

    Args:
        frame: Frame to be destreaked (config is obtained from frame.config)

    Returns:
        2D array of destreaked intensity values
    """
    ds = Destreak(frame)
    return ds.intensity


def _add_arguments(parser) -> None:
    """Add command arguments to parser."""
    parser.add_argument("polar_files", nargs="+", help="Polar files to process")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML configuration file")
    parser.add_argument("--plot", action="store_true", help="Plot destreak results")


def add_subparser(subparsers) -> None:
    """Register the 'destreak' subcommand."""
    p = subparsers.add_parser(
        "destreak", help="Standalone destreak tool", description="Test destreak algorithm"
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args) -> None:
    """Execute the 'destreak' command."""
    from wamos_tpw.polarfile import PolarFile
    from wamos_tpw.config import Config

    config = Config(args.config) if args.config else None

    for filepath in args.polar_files:
        pf = PolarFile(filepath, config=config)
        frame = pf.frame()
        ds = Destreak(frame, save_mask=args.plot)

        logging.info("File: %s", filepath)
        logging.info("Frame: %s", frame.timestamp)
        logging.info("Shape: %s", frame.shape)
        logging.info("Streaks: %d pixels (%.2f%%)", ds.n_streak_pixels, ds.streak_fraction * 100)
        logging.info(
            "Original intensity: [%.1f, %.1f]", frame.intensity.min(), frame.intensity.max()
        )
        logging.info("Destreaked intensity: [%.1f, %.1f]", ds.intensity.min(), ds.intensity.max())

        if args.plot:
            diag = DestreakDiag(frame, ds)
            diag.plot()


def main() -> None:
    """Standalone CLI entry point."""
    from argparse import ArgumentParser
    from wamos_tpw.logging_config import add_logging_arguments, setup_logging

    parser = ArgumentParser(description="Test destreak algorithm")
    add_logging_arguments(parser)
    _add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
