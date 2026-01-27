#!/usr/bin/env python3
"""
Prioritized pipeline toy - demonstrates multi-stage processing with priorities.

Uses multiprocessing with:
- Multiple priority queues (one per level) for task coordination
- Shared memory for passing large data (simulated frame arrays) between processes
- Reference counting for shared memory lifecycle management

Stages (low to high priority):
1. FileLoader (priority=3) - slow, produces many items
2. Interpolator (priority=2) - needs triplets, medium speed
3. NetCDF writer (priority=2) - one file per frame, no ordering needed
4. Projector (priority=1) - needs groups, fast
5. MovieMaker (priority=0) - highest priority, sequential output

Frame files are written independently as they complete - no ordering required.
"""

import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing import shared_memory
from queue import Empty
from typing import Any

import numpy as np


class Priority(IntEnum):
    """Priority levels (lower = higher priority)."""

    MOVIE = 0  # Highest - flush completed work
    PROJECT = 1  # High - grouping complete
    NETCDF = 2  # Medium - write individual frame files
    INTERP = 2  # Medium - interpolate triplets
    LOAD = 3  # Lowest - load files


@dataclass
class Task:
    """Task to be executed by a worker."""

    task_type: str
    task_id: int
    data: Any


@dataclass
class Result:
    """Result from a completed task."""

    task_type: str
    task_id: int
    data: Any
    shm_to_release: list[str] = field(default_factory=list)  # Shared memory done being used


# Simulated frame dimensions (smaller for toy)
FRAME_SHAPE = (360, 512)  # (bearings, ranges)
FRAME_DTYPE = np.float32
FRAME_SIZE = np.prod(FRAME_SHAPE) * np.dtype(FRAME_DTYPE).itemsize


class SharedMemoryManager:
    """
    Manages shared memory lifecycle with reference counting.

    Reference counting ensures shared memory is released exactly when
    all consumers are done with it, not before (use-after-free) or
    never (memory leak).

    Usage:
        manager = SharedMemoryManager()

        # Producer registers with expected consumer count
        manager.register("shm_123", refcount=2)  # NetCDF + Project will use it

        # Each consumer releases when done
        manager.release("shm_123")  # NetCDF done, refcount -> 1
        manager.release("shm_123")  # Project done, refcount -> 0, memory freed
    """

    def __init__(self):
        self._refcounts: dict[str, int] = {}
        self._total_registered = 0
        self._total_released = 0

    def register(self, shm_name: str, refcount: int):
        """Register a shared memory block with initial reference count."""
        if shm_name in self._refcounts:
            # Already registered, just add to refcount
            self._refcounts[shm_name] += refcount
        else:
            self._refcounts[shm_name] = refcount
            self._total_registered += 1

    def release(self, shm_name: str) -> bool:
        """
        Decrement reference count and free if zero.

        Returns:
            True if memory was freed, False if still has references
        """
        if shm_name not in self._refcounts:
            # Not tracked, release immediately (legacy/untracked block)
            self._do_release(shm_name)
            return True

        self._refcounts[shm_name] -= 1

        if self._refcounts[shm_name] <= 0:
            self._do_release(shm_name)
            del self._refcounts[shm_name]
            self._total_released += 1
            return True

        return False

    def release_many(self, shm_names: list[str]) -> int:
        """Release multiple shared memory blocks. Returns count freed."""
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

    def cleanup(self):
        """Release all remaining shared memory (for shutdown)."""
        for shm_name in list(self._refcounts.keys()):
            self._do_release(shm_name)
        remaining = len(self._refcounts)
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


def create_shared_frame(data: np.ndarray) -> tuple[str, tuple, np.dtype]:
    """
    Store frame data in shared memory.

    Returns:
        Tuple of (shm_name, shape, dtype) for reconstruction
    """
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shm_array[:] = data[:]
    shm.close()
    return shm.name, data.shape, data.dtype


def read_shared_frame(shm_name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Read frame data from shared memory (returns a copy)."""
    shm = shared_memory.SharedMemory(name=shm_name)
    shm_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    result = shm_array.copy()
    shm.close()
    return result


# ============================================================
# Worker functions (must be at module level for pickling)
# ============================================================


def do_load(task: Task) -> Result:
    """Load a file and create frame data in shared memory."""
    time.sleep(0.1)
    index, filename = task.data

    frame_data = np.random.rand(*FRAME_SHAPE).astype(FRAME_DTYPE)
    shm_name, shape, dtype = create_shared_frame(frame_data)

    print(f"  [LOAD] {filename} -> shm:{shm_name[:12]}...", flush=True)
    return Result(
        task_type="load",
        task_id=task.task_id,
        data={
            "index": index,
            "filename": filename,
            "shm_name": shm_name,
            "shape": shape,
            "dtype": dtype,
        },
    )


def do_interpolate(task: Task) -> Result:
    """Interpolate a triplet of frames using shared memory."""
    time.sleep(0.05)
    prev_info, curr_info, next_info, index = task.data

    # Read frames from shared memory
    prev_frame = read_shared_frame(prev_info["shm_name"], prev_info["shape"], prev_info["dtype"])
    curr_frame = read_shared_frame(curr_info["shm_name"], curr_info["shape"], curr_info["dtype"])
    next_frame = read_shared_frame(next_info["shm_name"], next_info["shape"], next_info["dtype"])

    # Simulate interpolation
    interpolated = (prev_frame + curr_frame + next_frame) / 3.0

    # Store result in new shared memory
    shm_name, shape, dtype = create_shared_frame(interpolated)

    print(f"  [INTERP] frame_{index} -> shm:{shm_name[:12]}...", flush=True)

    # Report which raw frame shm blocks we're done reading
    # (curr is done after this interpolation, prev/next may still be needed)
    return Result(
        task_type="interp",
        task_id=task.task_id,
        data={
            "index": index,
            "shm_name": shm_name,
            "shape": shape,
            "dtype": dtype,
        },
        shm_to_release=[
            prev_info["shm_name"],
            curr_info["shm_name"],
            next_info["shm_name"],
        ],
    )


def do_netcdf(task: Task) -> Result:
    """Write single frame to NetCDF file using shared memory data."""
    time.sleep(0.02)
    frame_info = task.data

    frame_data = read_shared_frame(frame_info["shm_name"], frame_info["shape"], frame_info["dtype"])

    filename = f"frame_{frame_info['index']:04d}.nc"
    checksum = np.sum(frame_data)

    print(f"  [NETCDF] wrote {filename} (sum={checksum:.2f})", flush=True)
    return Result(
        task_type="netcdf",
        task_id=task.task_id,
        data={"filename": filename, "index": frame_info["index"]},
        shm_to_release=[frame_info["shm_name"]],  # Done reading this frame
    )


def do_project(task: Task) -> Result:
    """Project a group of frames."""
    time.sleep(1.03)
    group_infos, start_index = task.data

    frames = []
    shm_names_read = []
    for info in group_infos:
        frame = read_shared_frame(info["shm_name"], info["shape"], info["dtype"])
        frames.append(frame)
        shm_names_read.append(info["shm_name"])

    projected = np.mean(frames, axis=0)
    shm_name, shape, dtype = create_shared_frame(projected)

    print(
        f"  [PROJECT] group {start_index} ({len(frames)} frames) -> shm:{shm_name[:12]}...",
        flush=True,
    )
    return Result(
        task_type="project",
        task_id=task.task_id,
        data={
            "start_index": start_index,
            "shm_name": shm_name,
            "shape": shape,
            "dtype": dtype,
        },
        shm_to_release=shm_names_read,  # Done reading these interpolated frames
    )


def do_movie(task: Task) -> Result:
    """Render movie segment from projected frames."""
    time.sleep(2.1)
    batch_infos = task.data

    frames = []
    shm_names_read = []
    for info in batch_infos:
        frame = read_shared_frame(info["shm_name"], info["shape"], info["dtype"])
        frames.append(frame)
        shm_names_read.append(info["shm_name"])

    total_mean = np.mean([np.mean(f) for f in frames])

    print(f"  [MOVIE] rendered {len(frames)} projections (mean={total_mean:.4f})", flush=True)
    return Result(
        task_type="movie",
        task_id=task.task_id,
        data=f"movie_segment_{batch_infos[0]['start_index']}",
        shm_to_release=shm_names_read,  # Done reading projected frames
    )


TASK_HANDLERS = {
    "load": do_load,
    "interp": do_interpolate,
    "netcdf": do_netcdf,
    "project": do_project,
    "movie": do_movie,
}


def worker_process(
    worker_id: int,
    priority_queues: list[mp.Queue],
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
):
    """Worker process that pulls tasks from priority queues."""
    while not shutdown_event.is_set():
        task = None

        for pq in priority_queues:
            try:
                task = pq.get_nowait()
                break
            except Empty:
                continue

        if task is None:
            time.sleep(0.01)
            continue

        handler = TASK_HANDLERS.get(task.task_type)
        if handler:
            try:
                result = handler(task)
                result_queue.put(result)
            except Exception as e:
                print(f"Worker {worker_id} error: {e}", flush=True)
                import traceback

                traceback.print_exc()
                result_queue.put(Result(task.task_type, task.task_id, {"error": str(e)}))


class PriorityProcessExecutor:
    """Process pool with priority queues."""

    def __init__(self, max_workers: int = 4, n_priorities: int = 4):
        self.max_workers = max_workers
        self.n_priorities = n_priorities

        self._priority_queues = [mp.Queue() for _ in range(n_priorities)]
        self._result_queue = mp.Queue()
        self._shutdown_event = mp.Event()

        self._workers = []
        for i in range(max_workers):
            p = mp.Process(
                target=worker_process,
                args=(i, self._priority_queues, self._result_queue, self._shutdown_event),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    def submit(self, priority: int, task_type: str, task_id: int, data: Any):
        """Submit a task at the given priority level."""
        task = Task(task_type=task_type, task_id=task_id, data=data)
        priority = max(0, min(priority, self.n_priorities - 1))
        self._priority_queues[priority].put(task)

    def get_result(self, timeout: float = 0.1) -> Result | None:
        """Get a result, or None if timeout."""
        try:
            return self._result_queue.get(timeout=timeout)
        except Empty:
            return None

    def shutdown(self, wait: bool = True):
        """Shutdown all workers."""
        self._shutdown_event.set()
        if wait:
            for p in self._workers:
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()


class TripletCollector:
    """Collects items and emits sliding window triplets."""

    def __init__(self):
        self._items = []

    def add(self, index: int, item: Any) -> list[tuple[Any, Any, Any, int]]:
        """Add item and return any complete triplets."""
        self._items.append((index, item))
        self._items.sort(key=lambda x: x[0])

        triplets = []
        while len(self._items) >= 3:
            prev_idx, prev = self._items[0]
            curr_idx, curr = self._items[1]
            next_idx, next_ = self._items[2]

            if prev_idx + 1 == curr_idx == next_idx - 1:
                triplets.append((prev, curr, next_, curr_idx))
                self._items.pop(0)
            else:
                break

        return triplets

    @property
    def pending_count(self) -> int:
        """Number of items waiting in collector."""
        return len(self._items)


class GroupCollector:
    """Collects items into groups of N."""

    def __init__(self, group_size: int):
        self._group_size = group_size
        self._items = []

    def add(self, index: int, item: Any) -> list[tuple[list, int]]:
        """Add item and return any complete groups."""
        self._items.append((index, item))
        self._items.sort(key=lambda x: x[0])

        groups = []
        while len(self._items) >= self._group_size:
            group = [item for _, item in self._items[: self._group_size]]
            start_idx = self._items[0][0]
            self._items = self._items[self._group_size :]
            groups.append((group, start_idx))

        return groups

    @property
    def pending_count(self) -> int:
        """Number of items waiting in collector."""
        return len(self._items)


class Pipeline:
    """Multi-stage prioritized pipeline using processes and shared memory."""

    def __init__(self, n_workers: int = 4, group_size: int = 3):
        self.executor = PriorityProcessExecutor(n_workers, n_priorities=4)
        self.group_size = group_size

        # Collectors (run in main process)
        self.triplet_collector = TripletCollector()
        self.group_collector = GroupCollector(group_size)
        self.movie_collector = []

        # Shared memory manager with reference counting
        self.shm_manager = SharedMemoryManager()

        # Results
        self.results = []
        self.netcdf_files = []

        # Tracking
        self._pending = 0
        self._task_id = 0

    def _next_task_id(self) -> int:
        tid = self._task_id
        self._task_id += 1
        return tid

    def _handle_load_result(self, result: Result):
        """Handle completed file load."""
        data = result.data
        index = data["index"]

        # Register raw frame shm with refcount based on expected usage:
        # - First frame (index 0): used as prev only → 1 use
        # - Last frame: used as next only → 1 use
        # - All others: used as prev, curr, next → up to 3 uses
        # For simplicity, use 3 and let extras be cleaned up
        # (In production, you'd compute exact refcount based on position)
        self.shm_manager.register(data["shm_name"], refcount=3)

        triplets = self.triplet_collector.add(index, data)

        for prev, curr, next_, idx in triplets:
            self._pending += 1
            self.executor.submit(
                Priority.INTERP,
                "interp",
                self._next_task_id(),
                (prev, curr, next_, idx),
            )

    def _handle_interp_result(self, result: Result):
        """Handle completed interpolation."""
        data = result.data
        index = data["index"]

        # Release references to raw frame shm blocks that interpolation read
        self.shm_manager.release_many(result.shm_to_release)

        # Register interpolated frame shm with refcount=2 (NetCDF + Project)
        self.shm_manager.register(data["shm_name"], refcount=2)

        # Submit NetCDF write
        self._pending += 1
        self.executor.submit(
            Priority.NETCDF,
            "netcdf",
            self._next_task_id(),
            data,
        )

        # Collect for grouping
        groups = self.group_collector.add(index, data)

        for group, start_idx in groups:
            self._pending += 1
            self.executor.submit(
                Priority.PROJECT,
                "project",
                self._next_task_id(),
                (group, start_idx),
            )

    def _handle_netcdf_result(self, result: Result):
        """Handle completed NetCDF write."""
        self.netcdf_files.append(result.data["filename"])
        # Release our reference to the interpolated frame
        self.shm_manager.release_many(result.shm_to_release)

    def _handle_project_result(self, result: Result):
        """Handle completed projection."""
        # Release references to interpolated frames that were grouped
        self.shm_manager.release_many(result.shm_to_release)

        # Register projected frame shm with refcount=1 (Movie only)
        self.shm_manager.register(result.data["shm_name"], refcount=1)

        self.movie_collector.append((result.data["start_index"], result.data))

        if len(self.movie_collector) >= 2:
            self.movie_collector.sort(key=lambda x: x[0])
            batch = [r for _, r in self.movie_collector[:2]]
            self.movie_collector = self.movie_collector[2:]

            self._pending += 1
            self.executor.submit(
                Priority.MOVIE,
                "movie",
                self._next_task_id(),
                batch,
            )

    def _handle_movie_result(self, result: Result):
        """Handle completed movie segment."""
        self.results.append(result.data)
        # Release references to projected frames
        self.shm_manager.release_many(result.shm_to_release)

    def run(self, filenames: list[str]):
        """Process all files through the pipeline."""
        print(f"\nProcessing {len(filenames)} files with {self.executor.max_workers} workers")
        print(f"Group size: {self.group_size}")
        print(f"Frame size: {FRAME_SHAPE} = {FRAME_SIZE:,} bytes")
        print("-" * 60)

        t0 = time.perf_counter()

        for i, fn in enumerate(filenames):
            self._pending += 1
            self.executor.submit(Priority.LOAD, "load", self._next_task_id(), (i, fn))

        handlers = {
            "load": self._handle_load_result,
            "interp": self._handle_interp_result,
            "netcdf": self._handle_netcdf_result,
            "project": self._handle_project_result,
            "movie": self._handle_movie_result,
        }

        while self._pending > 0:
            result = self.executor.get_result(timeout=0.1)
            if result:
                self._pending -= 1
                handler = handlers.get(result.task_type)
                if handler:
                    handler(result)

        elapsed = time.perf_counter() - t0

        # Clean up any remaining shared memory (edge frames, incomplete groups)
        remaining = self.shm_manager.cleanup()

        self.executor.shutdown()

        print("-" * 60)
        print(f"Completed in {elapsed:.2f}s:")
        print(f"  Movie segments: {len(self.results)}")
        print(f"  NetCDF files: {len(self.netcdf_files)}")
        print(f"  Shared memory stats: {self.shm_manager.stats}")
        if remaining > 0:
            print(f"  (Cleaned up {remaining} orphaned shm blocks from edge frames)")
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Prioritized multi-stage pipeline demo with shared memory and refcounting"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=12,
        help="Number of files to process (default: 12)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)",
    )
    parser.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=3,
        help="Number of frames per projection group (default: 3)",
    )
    args = parser.parse_args()

    filenames = [f"file_{i:03d}.pol" for i in range(args.num_files)]

    pipeline = Pipeline(n_workers=args.workers, group_size=args.group_size)
    results = pipeline.run(filenames)

    print(f"\nResults: {results}")
    print(f"NetCDF files: {sorted(pipeline.netcdf_files)}")


if __name__ == "__main__":
    main()
