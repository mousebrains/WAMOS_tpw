#! /usr/bin/env python3
#
# Bounded read-ahead for file sequences on high-latency filesystems
#
# Serial per-file reads over SMB pay a full network round-trip per
# file (measured ~26 MB/s single-stream vs ~92 MB/s at 8-way on the
# same share). Warming the OS page cache from background threads a
# few files ahead of a serial consumer overlaps those round-trips
# with the consumer's compute; the consumer's own read is then served
# locally (measured ~0.06 s for 16 x 4 MB cached vs 2.4 s cold).
#
# Jul-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["read_ahead"]


def _warm(path: Path) -> int:
    """Read *path* fully to populate the page cache; never raises."""
    try:
        return len(path.read_bytes())
    except OSError as e:
        logger.debug("read-ahead skipped %s: %s", path, e)
        return -1


def read_ahead(
    files: Iterable[str | Path],
    depth: int = 8,
    workers: int = 6,
) -> Iterator[Path]:
    """Yield *files* in order, keeping up to *depth* warmed ahead.

    Each path is yielded only after a background thread has read it
    once, so the consumer's subsequent open hits the local page cache
    instead of a cold network fetch. Warming errors are swallowed --
    the consumer's own read is the one that reports them.

    Args:
        files: File paths in consumption order.
        depth: Files to keep in flight ahead of the consumer;
            ``depth <= 1`` degrades to a plain pass-through.
        workers: Concurrent warming threads.

    Yields:
        The input paths as :class:`~pathlib.Path`, in input order.
    """
    paths = (Path(f) for f in files)
    if depth <= 1:
        yield from paths
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: deque[tuple[Path, Future[int]]] = deque()
        for p in paths:
            pending.append((p, pool.submit(_warm, p)))
            if len(pending) >= depth:
                break
        while pending:
            path, fut = pending.popleft()
            fut.result()
            nxt = next(paths, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(_warm, nxt)))
            yield path
