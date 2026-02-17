#! /usr/bin/env python3
#
# Priority-based process pool executor with shared memory support
#
# Provides a process pool where tasks are executed in priority order,
# with shared memory management for efficient large data transfer.
#
# Jan-2026, Pat Welch, pat@mousebrains.com

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing import resource_tracker, shared_memory
from queue import Empty
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disable the resource tracker for shared_memory objects.
#
# SharedMemoryManager handles the full lifecycle (close + unlink) with
# reference counting.  Python's resource_tracker independently tracks
# shared_memory blocks and tries to clean them up at process shutdown,
# which causes spurious KeyError / FileNotFoundError tracebacks because
# the blocks have already been unlinked by SharedMemoryManager.
#
# We filter out "shared_memory" from both register and unregister while
# leaving other resource types (e.g. semaphores) untouched.
# ---------------------------------------------------------------------------
_orig_register = resource_tracker.register
_orig_unregister = resource_tracker.unregister


def _register_filter(name, rtype):
    if rtype == "shared_memory":
        return
    _orig_register(name, rtype)


def _unregister_filter(name, rtype):
    if rtype == "shared_memory":
        return
    _orig_unregister(name, rtype)


resource_tracker.register = _register_filter
resource_tracker.unregister = _unregister_filter


class Priority(IntEnum):
    """
    Task priority levels (lower = higher priority).

    Tasks with lower priority values are executed before those with higher values.
    Within the same priority level, tasks are executed in FIFO order.
    """

    HIGHEST = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    LOWEST = 4


@dataclass
class Task:
    """Task to be executed by a worker process."""

    task_type: str
    task_id: int
    data: Any


@dataclass
class Result:
    """Result from a completed task."""

    task_type: str
    task_id: int
    data: Any
    shm_to_release: list[str] = field(default_factory=list)
    error: str | None = None


class SharedMemoryManager:
    """
    Manages shared memory lifecycle with reference counting.

    Reference counting ensures shared memory is released exactly when
    all consumers are done with it, preventing both use-after-free
    and memory leaks.

    Usage:
        manager = SharedMemoryManager()

        # Producer registers with expected consumer count
        shm_name = create_shared_array(data)
        manager.register(shm_name, refcount=2)  # Two consumers will use it

        # Each consumer releases when done
        manager.release(shm_name)  # First consumer done, refcount -> 1
        manager.release(shm_name)  # Second consumer done, refcount -> 0, freed

        # At shutdown, clean up any remaining blocks
        manager.cleanup()
    """

    def __init__(self):
        self._refcounts: dict[str, int] = {}
        self._total_registered = 0
        self._total_released = 0

    def register(self, shm_name: str, refcount: int):
        """
        Register a shared memory block with initial reference count.

        Args:
            shm_name: Name of the shared memory block
            refcount: Number of consumers that will use this block
        """
        if shm_name in self._refcounts:
            self._refcounts[shm_name] += refcount
        else:
            self._refcounts[shm_name] = refcount
            self._total_registered += 1

    def release(self, shm_name: str) -> bool:
        """
        Decrement reference count and free if zero.

        Args:
            shm_name: Name of the shared memory block

        Returns:
            True if memory was freed, False if still has references
        """
        if shm_name not in self._refcounts:
            self._do_release(shm_name)
            return True

        self._refcounts[shm_name] -= 1

        if self._refcounts[shm_name] <= 0:
            self._do_release(shm_name)
            del self._refcounts[shm_name]
            self._total_released += 1
            return True

        return False

    def release_many(self, shm_names: list[str] | None) -> int:
        """
        Release multiple shared memory blocks.

        Args:
            shm_names: List of shared memory names to release

        Returns:
            Number of blocks that were actually freed
        """
        if not shm_names:
            return 0
        freed = 0
        for name in shm_names:
            if self.release(name):
                freed += 1
        return freed

    def _do_release(self, shm_name: str):
        """Actually release the shared memory."""
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass  # Already released

    def cleanup(self) -> int:
        """
        Release all remaining shared memory blocks.

        Call this at shutdown to clean up any orphaned blocks.

        Returns:
            Number of blocks that were cleaned up
        """
        remaining = len(self._refcounts)
        for shm_name in list(self._refcounts.keys()):
            self._do_release(shm_name)
        self._refcounts.clear()
        return remaining

    @property
    def active_count(self) -> int:
        """Number of shared memory blocks still being tracked."""
        return len(self._refcounts)

    @property
    def stats(self) -> dict:
        """Return statistics about shared memory usage."""
        return {
            "registered": self._total_registered,
            "released": self._total_released,
            "active": self.active_count,
        }


def create_shared_array(data: np.ndarray) -> tuple[str, tuple, np.dtype]:
    """
    Store numpy array in shared memory.

    Args:
        data: Numpy array to store

    Returns:
        Tuple of (shm_name, shape, dtype) for reconstruction
    """
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shm_array[:] = data[:]
    shm.close()
    return shm.name, data.shape, data.dtype


def read_shared_array(shm_name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """
    Read numpy array from shared memory.

    Args:
        shm_name: Name of shared memory block
        shape: Array shape
        dtype: Array data type

    Returns:
        Copy of the array (safe to use after shared memory is released)
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    result = shm_array.copy()
    shm.close()
    return result


def release_shared_memory(shm_name: str):
    """
    Release shared memory by name.

    Safe to call even if already released.
    """
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def _worker_loop(
    worker_id: int,
    priority_queues: list[mp.Queue],
    result_queue: mp.Queue,
    shutdown_event: Any,  # mp.synchronize.Event
    task_handlers: dict[str, Callable],
):
    """
    Worker process main loop.

    Pulls tasks from priority queues (checking higher priority first)
    and executes them using the registered handlers.
    """
    while not shutdown_event.is_set():
        task = None

        # Check queues in priority order (index 0 = highest priority)
        for pq in priority_queues:
            try:
                task = pq.get_nowait()
                break
            except Empty:
                continue

        if task is None:
            time.sleep(0.01)
            continue

        handler = task_handlers.get(task.task_type)
        if handler:
            try:
                result = handler(task)
                result_queue.put(result)
            except Exception as e:
                logger.exception("Worker %d error processing task %s", worker_id, task.task_type)
                result_queue.put(
                    Result(
                        task_type=task.task_type,
                        task_id=task.task_id,
                        data=None,
                        error=str(e),
                    )
                )


class PriorityProcessExecutor:
    """
    Process pool executor with priority-based task scheduling.

    Tasks are submitted with a priority level. Workers check higher-priority
    queues first, ensuring important tasks are processed before less urgent ones.

    Example:
        def my_handler(task: Task) -> Result:
            # Process task.data
            return Result(task.task_type, task.task_id, {"output": "done"})

        executor = PriorityProcessExecutor(
            max_workers=4,
            task_handlers={"my_task": my_handler}
        )

        executor.submit(Priority.HIGH, "my_task", task_id=1, data={"input": "value"})

        while pending:
            result = executor.get_result()
            if result:
                # Handle result

        executor.shutdown()
    """

    def __init__(
        self,
        max_workers: int = 4,
        n_priorities: int = 5,
        task_handlers: dict[str, Callable] | None = None,
    ):
        """
        Initialize the executor.

        Args:
            max_workers: Number of worker processes
            n_priorities: Number of priority levels (default 5 matches Priority enum)
            task_handlers: Dict mapping task_type strings to handler functions.
                          Must be module-level functions (picklable).
        """
        self.max_workers = max_workers
        self.n_priorities = n_priorities
        self._task_handlers = task_handlers or {}

        # One queue per priority level (index 0 = highest priority)
        self._priority_queues = [mp.Queue() for _ in range(n_priorities)]
        self._result_queue = mp.Queue()
        self._shutdown_event = mp.Event()

        self._workers: list[mp.Process] = []
        self._started = False
        self._task_counter = 0

    def start(self):
        """Start worker processes."""
        if self._started:
            return

        for i in range(self.max_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    i,
                    self._priority_queues,
                    self._result_queue,
                    self._shutdown_event,
                    self._task_handlers,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        self._started = True
        logger.debug("Started %d worker processes", self.max_workers)

    def submit(
        self,
        priority: int | Priority,
        task_type: str,
        data: Any,
        task_id: int | None = None,
    ) -> int:
        """
        Submit a task for execution.

        Args:
            priority: Task priority (lower = higher priority)
            task_type: Type identifier for the task (must have registered handler)
            data: Task data to pass to handler
            task_id: Optional task ID (auto-generated if not provided)

        Returns:
            Task ID
        """
        if not self._started:
            self.start()

        if task_id is None:
            task_id = self._task_counter
            self._task_counter += 1

        task = Task(task_type=task_type, task_id=task_id, data=data)
        priority_idx = max(0, min(int(priority), self.n_priorities - 1))
        self._priority_queues[priority_idx].put(task)

        return task_id

    def get_result(self, timeout: float = 0.1) -> Result | None:
        """
        Get a result from completed tasks.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Result object, or None if no results available
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None

    def shutdown(self, wait: bool = True, timeout: float = 2.0):
        """
        Shutdown all workers.

        Args:
            wait: Whether to wait for workers to finish
            timeout: Maximum time to wait for each worker
        """
        self._shutdown_event.set()

        if wait:
            for p in self._workers:
                p.join(timeout=timeout)
                if p.is_alive():
                    p.terminate()

        self._workers.clear()
        self._started = False
        logger.debug("Shutdown complete")

    @property
    def is_running(self) -> bool:
        """Return True if executor is running."""
        return self._started and not self._shutdown_event.is_set()


class PipelineRunner:
    """
    Base class for running multi-stage pipelines with priority scheduling.

    Subclass this to implement specific pipelines. Override:
    - get_task_handlers(): Return dict of task_type -> handler function
    - handle_result(): Process results and submit follow-up tasks

    Example:
        class MyPipeline(PipelineRunner):
            def get_task_handlers(self):
                return {"load": do_load, "process": do_process}

            def handle_result(self, result):
                if result.task_type == "load":
                    self.submit(Priority.MEDIUM, "process", result.data)
                elif result.task_type == "process":
                    self.outputs.append(result.data)
    """

    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.executor: PriorityProcessExecutor | None = None
        self.shm_manager = SharedMemoryManager()
        self._pending = 0

    def get_task_handlers(self) -> dict[str, Callable]:
        """Return dict mapping task_type to handler functions. Override in subclass."""
        raise NotImplementedError

    def handle_result(self, result: Result):
        """Process a result and submit follow-up tasks. Override in subclass."""
        raise NotImplementedError

    def submit(self, priority: int | Priority, task_type: str, data: Any) -> int:
        """Submit a task and track pending count."""
        self._pending += 1
        return self.executor.submit(priority, task_type, data)

    def run(self, initial_tasks: list[tuple[int | Priority, str, Any]]) -> Any:
        """
        Run the pipeline with initial tasks.

        Args:
            initial_tasks: List of (priority, task_type, data) tuples

        Returns:
            Pipeline-specific results (defined by subclass)
        """
        self.executor = PriorityProcessExecutor(
            max_workers=self.n_workers,
            task_handlers=self.get_task_handlers(),
        )
        self.executor.start()

        t0 = time.perf_counter()

        # Submit initial tasks
        for priority, task_type, data in initial_tasks:
            self.submit(priority, task_type, data)

        # Process results until all work is done
        while self._pending > 0:
            result = self.executor.get_result(timeout=0.1)
            if result:
                self._pending -= 1

                # Release any shared memory the task is done with
                self.shm_manager.release_many(result.shm_to_release)

                # Let subclass handle the result
                if result.error:
                    logger.error("Task %s failed: %s", result.task_type, result.error)
                else:
                    self.handle_result(result)

        elapsed = time.perf_counter() - t0

        # Cleanup
        orphaned = self.shm_manager.cleanup()
        self.executor.shutdown()

        logger.debug(
            "Pipeline complete: %.2fs, shm stats: %s, orphaned: %d",
            elapsed,
            self.shm_manager.stats,
            orphaned,
        )

        return self.get_results()

    def get_results(self) -> Any:
        """Return pipeline results. Override in subclass."""
        return None


# ============================================================
# Triplet accumulator for sliding window processing
# ============================================================


class TripletCollector:
    """
    Collects items and emits triplets (prev, current, next) when ready.

    Designed for parallel processing where items arrive out of order.
    Uses (file_index, frame_index) keys to determine true consecutive neighbors,
    enabling streaming emission of triplets as soon as neighbors are confirmed.

    Tracks file completions to know frame counts per file, which is necessary
    to determine when a frame at the end of one file is followed by a frame
    at the start of the next file.

    Example:
        collector = TripletCollector(total_files=100)

        # Files arrive out of order from parallel workers
        for file_result in incoming_results:
            # Add all frames from this file
            for frame in file_result.frames:
                collector.add(frame)
            # Mark file as complete with its frame count
            collector.file_complete(file_result.file_index, len(file_result.frames))

            # Check for complete triplets (can emit immediately when neighbors known)
            for triplet in collector.ready_triplets():
                prev, current, next_item = triplet
                # Process the triplet
    """

    def __init__(self, total_files: int | None = None) -> None:
        """
        Initialize the triplet collector.

        Args:
            total_files: Total number of files expected. Required to know
                        when the last file's last frame has no next neighbor.
        """
        self._total_files = total_files
        self._items: dict[tuple[int, int], Any] = {}  # (file_idx, frame_idx) -> item
        self._emitted: set[tuple[int, int]] = set()  # Keys already emitted as 'current'
        self._file_frame_counts: dict[int, int] = {}  # file_idx -> frame count
        self._n_pruned: int = 0  # Count of items removed from _items by prune_emitted

    def add(self, item: Any) -> None:
        """
        Add an item to the collector.

        Item must have file_index and frame_index attributes.
        """
        key = (item.file_index, item.frame_index)
        self._items[key] = item

    def file_complete(self, file_idx: int, frame_count: int) -> None:
        """
        Mark a file as complete with known frame count.

        This enables determining true neighbors across file boundaries.
        """
        self._file_frame_counts[file_idx] = frame_count

    def set_total_files(self, total: int) -> None:
        """Set the total number of files (enables last-frame detection)."""
        self._total_files = total

    def _get_prev_key(self, file_idx: int, frame_idx: int) -> tuple[int, int] | None:
        """Get the key of the previous item in sequence, or None if first."""
        if frame_idx > 0:
            # Previous frame in same file
            return (file_idx, frame_idx - 1)
        elif file_idx > 0:
            # Last frame of previous file
            prev_file = file_idx - 1
            if prev_file in self._file_frame_counts:
                return (prev_file, self._file_frame_counts[prev_file] - 1)
            # Previous file not complete yet - can't determine prev
            return None
        else:
            # First frame of first file - no previous
            return "FIRST"  # Sentinel to distinguish from "unknown"

    def _get_next_key(self, file_idx: int, frame_idx: int) -> tuple[int, int] | None:
        """Get the key of the next item in sequence, or None if last."""
        if file_idx not in self._file_frame_counts:
            # Current file not complete - can't determine next
            return None

        frame_count = self._file_frame_counts[file_idx]
        if frame_idx + 1 < frame_count:
            # Next frame in same file
            return (file_idx, frame_idx + 1)
        elif self._total_files is not None:
            if file_idx + 1 < self._total_files:
                # First frame of next file
                return (file_idx + 1, 0)
            else:
                # Last frame of last file - no next
                return "LAST"  # Sentinel
        # Don't know total files yet
        return None

    def ready_triplets(self) -> list[tuple[Any | None, Any, Any | None]]:
        """
        Return list of triplets that are ready to process.

        A triplet is ready when we can confirm the true prev/next neighbors:
        - Previous is confirmed when we have it OR know current is first
        - Next is confirmed when we have it OR know current is last

        Returns:
            List of (prev, current, next) tuples where prev and next may be None
        """
        ready = []

        for key in list(self._items.keys()):
            if key in self._emitted:
                continue

            file_idx, frame_idx = key
            current = self._items[key]

            # Determine true previous neighbor
            prev_key = self._get_prev_key(file_idx, frame_idx)
            if prev_key is None:
                # Can't determine prev yet (previous file not complete)
                continue
            elif prev_key == "FIRST":
                prev = None
            else:
                prev = self._items.get(prev_key)
                if prev is None:
                    # Previous frame hasn't arrived yet
                    continue

            # Determine true next neighbor
            next_key = self._get_next_key(file_idx, frame_idx)
            if next_key is None:
                # Can't determine next yet (current file not complete or total unknown)
                continue
            elif next_key == "LAST":
                next_item = None
            else:
                next_item = self._items.get(next_key)
                if next_item is None:
                    # Next frame hasn't arrived yet
                    continue

            # Both neighbors confirmed - emit triplet
            ready.append((prev, current, next_item))
            self._emitted.add(key)

        return ready

    @property
    def pending_count(self) -> int:
        """Number of items in memory not yet emitted as 'current'."""
        return len(self._items) - (len(self._emitted) - self._n_pruned)

    @property
    def item_count(self) -> int:
        """Number of items still in memory."""
        return len(self._items)

    @property
    def emitted_count(self) -> int:
        """Number of emitted items still in memory."""
        return len(self._emitted) - self._n_pruned

    @property
    def files_complete(self) -> int:
        """Number of files marked as complete."""
        return len(self._file_frame_counts)

    def prune_emitted(self) -> int:
        """
        Remove items that have been emitted and are no longer needed as neighbors.

        An item is safe to remove when:
        - It has been emitted as 'current'
        - Its previous neighbor has been emitted (so it's not needed as 'next')
        - Its next neighbor has been emitted (so it's not needed as 'prev')

        Returns:
            Number of items removed
        """
        to_remove = []

        for key in list(self._emitted):
            file_idx, frame_idx = key

            # Check if prev neighbor has been emitted
            prev_key = self._get_prev_key(file_idx, frame_idx)
            if prev_key is None:
                continue  # Can't determine yet
            if prev_key != "FIRST" and prev_key not in self._emitted:
                continue  # Prev not emitted yet, this item may be needed as 'next'

            # Check if next neighbor has been emitted
            next_key = self._get_next_key(file_idx, frame_idx)
            if next_key is None:
                continue  # Can't determine yet
            if next_key != "LAST" and next_key not in self._emitted:
                continue  # Next not emitted yet, this item may be needed as 'prev'

            # Safe to remove
            to_remove.append(key)

        for key in to_remove:
            if key in self._items:
                del self._items[key]
            # Keep key in _emitted so neighbors can still verify it was emitted

        self._n_pruned += len(to_remove)
        return len(to_remove)
