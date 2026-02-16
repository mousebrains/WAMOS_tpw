#! /usr/bin/env python3
#
# Common parallel execution and benchmarking utilities for pipeline modules
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import os
import platform
import resource
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from concurrent.futures import Executor

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Input item type
R = TypeVar("R")  # Result type


@dataclass
class BenchmarkResult:
    """Results from a single executor run."""

    executor_name: str
    elapsed: float
    n_workers: int
    results: list[Any] = field(default_factory=list)
    errors: list[tuple[Any, Exception]] = field(default_factory=list)
    max_worker_rss: int = 0


def run_with_executor(
    executor_name: str,
    Executor: type[Executor],
    items: list[T],
    process_func: Callable[[T], R],
    n_workers: int,
    item_desc: str = "item",
    get_rss: Callable[[R], int] | None = None,
    qProgress: bool = True,
) -> BenchmarkResult:
    """
    Run process_func over items using the specified executor.

    Args:
        executor_name: Name for display (e.g., "ThreadPool")
        Executor: Executor class to use
        items: List of items to process
        process_func: Function to call for each item
        n_workers: Number of parallel workers
        item_desc: Description for progress bar (e.g., "file", "group")
        get_rss: Optional function to extract RSS from result
        qProgress: Show progress bar

    Returns:
        BenchmarkResult with all results and statistics
    """
    results = []
    errors = []
    max_worker_rss = 0
    t_start = time.perf_counter()

    if n_workers <= 1 or len(items) == 1:
        # Sequential processing
        for item in tqdm(items, desc=executor_name, unit=item_desc, disable=not qProgress):
            try:
                result = process_func(item)
                results.append(result)
                if get_rss is not None:
                    max_worker_rss = max(max_worker_rss, get_rss(result))
            except Exception as e:
                errors.append((item, e))
                logger.error("Error processing %s: %s", item, e)
    else:
        # Parallel processing
        with Executor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_func, item): item for item in items}
            for future in tqdm(
                as_completed(futures),
                total=len(items),
                desc=executor_name,
                unit=item_desc,
                disable=not qProgress,
            ):
                item = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if get_rss is not None:
                        max_worker_rss = max(max_worker_rss, get_rss(result))
                except Exception as e:
                    errors.append((item, e))
                    logger.error("Error processing %s: %s", item, e)

    elapsed = time.perf_counter() - t_start

    return BenchmarkResult(
        executor_name=executor_name,
        elapsed=elapsed,
        n_workers=n_workers,
        results=results,
        errors=errors,
        max_worker_rss=max_worker_rss,
    )


def run_benchmark(
    items: list[T],
    process_func: Callable[[T], R],
    n_workers: int | None = None,
    item_desc: str = "item",
    get_rss: Callable[[R], int] | None = None,
    executors: list[str] | None = None,
    qProgress: bool = True,
) -> Iterator[BenchmarkResult]:
    """
    Run process_func over items using selected executor types.

    Yields BenchmarkResult for each executor type.

    Args:
        items: List of items to process
        process_func: Function to call for each item
        n_workers: Number of workers (default: min(len(items), cpu_count))
        item_desc: Description for progress bar
        get_rss: Optional function to extract RSS from result
        executors: List of executor names to use. Valid values:
                   "threadpool", "processpool", "prioritypool".
                   If None, uses ["threadpool", "processpool"].
        qProgress: Show progress bar
    """
    if n_workers is None:
        n_workers = min(len(items), os.cpu_count() or 1)

    # Default to both standard executors
    if executors is None:
        executors = ["threadpool", "processpool"]

    # Map of executor names to (display_name, executor_class)
    executor_map = {
        "threadpool": ("ThreadPool", ThreadPoolExecutor),
        "processpool": ("ProcessPool", ProcessPoolExecutor),
    }

    for executor_key in executors:
        executor_key_lower = executor_key.lower()

        if executor_key_lower == "prioritypool":
            # PriorityPool uses different API - run separately
            result = run_with_priority_executor(
                items=items,
                process_func=process_func,
                n_workers=n_workers,
                item_desc=item_desc,
                get_rss=get_rss,
                qProgress=qProgress,
            )
            yield result
        elif executor_key_lower in executor_map:
            executor_name, Executor = executor_map[executor_key_lower]

            logger.info("")
            logger.info("=" * 60)
            logger.info("Running with %s (%d workers)", executor_name, n_workers)
            logger.info("=" * 60)

            result = run_with_executor(
                executor_name=executor_name,
                Executor=Executor,
                items=items,
                process_func=process_func,
                n_workers=n_workers,
                item_desc=item_desc,
                get_rss=get_rss,
                qProgress=qProgress,
            )

            yield result
        else:
            logger.warning("Unknown executor type: %s", executor_key)


def _priority_task_handler(task):
    """
    Module-level task handler for PriorityProcessExecutor.

    Must be at module level to be picklable for multiprocessing.
    Expects task.data to be a tuple of (process_func, item).
    """
    from wamos_tpw.priority_executor import Result

    process_func, item = task.data
    result = process_func(item)
    return Result(
        task_type="process",
        task_id=task.task_id,
        data=result,
    )


def run_with_priority_executor(
    items: list[T],
    process_func: Callable[[T], R],
    n_workers: int,
    item_desc: str = "item",
    get_rss: Callable[[R], int] | None = None,
    qProgress: bool = True,
) -> BenchmarkResult:
    """
    Run process_func over items using the PriorityProcessExecutor.

    This adapter wraps PriorityProcessExecutor to provide the same interface
    as the standard concurrent.futures executors.

    Args:
        items: List of items to process
        process_func: Function to call for each item
        n_workers: Number of parallel workers
        item_desc: Description for progress bar
        get_rss: Optional function to extract RSS from result
        qProgress: Show progress bar

    Returns:
        BenchmarkResult with all results and statistics
    """
    from wamos_tpw.priority_executor import Priority, PriorityProcessExecutor

    executor_name = "PriorityPool"
    results = []
    errors = []
    max_worker_rss = 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Running with %s (%d workers)", executor_name, n_workers)
    logger.info("=" * 60)

    t_start = time.perf_counter()

    if n_workers <= 1 or len(items) == 1:
        # Sequential processing - same as other executors
        for item in tqdm(items, desc=executor_name, unit=item_desc, disable=not qProgress):
            try:
                result = process_func(item)
                results.append(result)
                if get_rss is not None:
                    max_worker_rss = max(max_worker_rss, get_rss(result))
            except Exception as e:
                errors.append((item, e))
                logger.error("Error processing %s: %s", item, e)
    else:
        # Use module-level handler for pickling compatibility
        executor = PriorityProcessExecutor(
            max_workers=n_workers,
            task_handlers={"process": _priority_task_handler},
        )
        executor.start()

        # Submit all tasks with (process_func, item) tuple as data
        pending = len(items)
        for i, item in enumerate(items):
            executor.submit(Priority.MEDIUM, "process", data=(process_func, item), task_id=i)

        # Collect results with progress bar
        with tqdm(total=pending, desc=executor_name, unit=item_desc, disable=not qProgress) as pbar:
            while pending > 0:
                result = executor.get_result(timeout=0.1)
                if result is not None:
                    pending -= 1
                    pbar.update(1)

                    if result.error:
                        errors.append((items[result.task_id], Exception(result.error)))
                        logger.error("Error processing item %d: %s", result.task_id, result.error)
                    else:
                        results.append(result.data)
                        if get_rss is not None:
                            max_worker_rss = max(max_worker_rss, get_rss(result.data))

        executor.shutdown()

    elapsed = time.perf_counter() - t_start

    return BenchmarkResult(
        executor_name=executor_name,
        elapsed=elapsed,
        n_workers=n_workers,
        results=results,
        errors=errors,
        max_worker_rss=max_worker_rss,
    )


def aggregate_timings(
    results: list[R],
    get_timings: Callable[[R], list[dict[str, float]]],
) -> dict[str, list[float]]:
    """
    Aggregate timing dictionaries from results.

    Args:
        results: List of results from process_func
        get_timings: Function to extract list of timing dicts from each result

    Returns:
        Dict mapping step names to lists of timing values
    """
    all_timings: dict[str, list[float]] = {}
    for result in results:
        for timings in get_timings(result):
            for key, val in timings.items():
                if key not in all_timings:
                    all_timings[key] = []
                all_timings[key].append(val)
    return all_timings


def display_timing_stats(all_timings: dict[str, list[float]]) -> None:
    """Display timing statistics table."""
    if not all_timings:
        return

    total_time = sum(sum(vals) for vals in all_timings.values())

    logger.info("")
    logger.info("  Frame timing statistics:")
    header = f"    {'Step':<12} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'%':>6}"
    logger.info(header)
    logger.info("    " + "-" * (len(header) - 4))
    for key, vals in all_timings.items():
        arr = np.array(vals) * 1000  # Convert to ms
        pct = (sum(vals) / total_time * 100) if total_time > 0 else 0
        logger.info(
            "    %-12s %10.2f %10.2f %10.2f %10.2f %6.1f",
            key,
            arr.mean(),
            arr.std(),
            arr.min(),
            arr.max(),
            pct,
        )


def display_memory_stats(max_worker_rss: int) -> None:
    """Display memory statistics."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    main_rss = rusage.ru_maxrss
    peak_rss = max(main_rss, max_worker_rss)

    # ru_maxrss is in bytes on macOS, kilobytes on Linux
    is_darwin = platform.system() == "Darwin"
    divisor = (1024 * 1024) if is_darwin else 1024

    logger.info("")
    logger.info("  Memory statistics:")
    logger.info("    Peak RSS (main): %.1f MB", main_rss / divisor)
    logger.info("    Peak RSS (workers): %.1f MB", max_worker_rss / divisor)
    logger.info("    Peak RSS (max): %.1f MB", peak_rss / divisor)


def display_benchmark_header(
    executor_name: str,
    n_items: int,
    item_label: str,
    total_count: int,
    count_label: str,
    elapsed: float,
    n_workers: int,
    extra_lines: list[str] | None = None,
) -> None:
    """
    Display standard benchmark results header.

    Args:
        executor_name: Name of executor (e.g., "ThreadPool")
        n_items: Number of items processed (files, groups)
        item_label: Label for items (e.g., "Files", "Groups")
        total_count: Total count of sub-items (e.g., frames)
        count_label: Label for count (e.g., "Frames")
        elapsed: Total elapsed time in seconds
        n_workers: Number of workers used
        extra_lines: Additional lines to display
    """
    throughput = total_count / elapsed if elapsed > 0 else 0

    logger.info("")
    logger.info("%s Results:", executor_name)
    logger.info("  %s processed: %d", item_label, n_items)
    logger.info("  %s processed: %d", count_label, total_count)
    if extra_lines:
        for line in extra_lines:
            logger.info("  %s", line)
    logger.info("  Total time: %.2f s", elapsed)
    logger.info("  Throughput: %.1f %s/s", throughput, count_label.lower())
    logger.info("  Workers: %d", n_workers)
