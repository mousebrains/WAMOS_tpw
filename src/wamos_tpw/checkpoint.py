#! /usr/bin/env python3
#
# Checkpointing utilities for long-running WAMOS processing
#
# Feb-2026, Pat Welch, pat@mousebrains.com

"""
Checkpointing utilities for resumable WAMOS processing.

This module provides checkpoint management for long-running pipeline operations,
allowing processing to be resumed after interruption.

Usage
-----

Basic checkpointing::

    from wamos_tpw.checkpoint import Checkpoint

    checkpoint = Checkpoint("/tmp/processing_checkpoint.json")

    # Check if resuming
    if checkpoint.exists():
        state = checkpoint.load()
        start_window = state.get("last_completed_window", 0) + 1
    else:
        start_window = 0

    # Process windows
    for window_idx in range(start_window, n_windows):
        result = process_window(window_idx)
        save_result(result)

        # Save checkpoint after each window
        checkpoint.save({
            "last_completed_window": window_idx,
            "total_windows": n_windows,
            "timestamp": datetime.now().isoformat(),
        })

    # Clear checkpoint when complete
    checkpoint.clear()

With context manager::

    with Checkpoint.resumable("/tmp/checkpoint.json") as state:
        # state is loaded from existing checkpoint or empty dict
        for i in range(state.get("progress", 0), 100):
            process_item(i)
            state["progress"] = i + 1
            # Checkpoint saved automatically on exit

Atomic checkpointing (for safety)::

    checkpoint = Checkpoint("/tmp/checkpoint.json", atomic=True)
    # Uses write-to-temp-then-rename for crash safety
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["Checkpoint", "CheckpointState", "PipelineCheckpoint"]

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """
    Standard checkpoint state for pipeline processing.

    Attributes:
        last_completed: Index of last completed item (file, window, etc.)
        total: Total number of items to process
        started_at: ISO timestamp when processing started
        updated_at: ISO timestamp of last checkpoint update
        metadata: Additional state data
    """

    last_completed: int = -1
    total: int = 0
    started_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Create from dictionary."""
        return cls(
            last_completed=data.get("last_completed", -1),
            total=data.get("total", 0),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )


class Checkpoint:
    """
    Simple checkpoint manager for saving and loading processing state.

    Example::

        checkpoint = Checkpoint("/tmp/pipeline_checkpoint.json")

        # Save state
        checkpoint.save({"window": 5, "total": 100})

        # Load state
        state = checkpoint.load()  # Returns {} if not exists

        # Check existence
        if checkpoint.exists():
            ...

        # Clear when done
        checkpoint.clear()
    """

    def __init__(self, path: str | Path, atomic: bool = True) -> None:
        """
        Initialize checkpoint manager.

        Args:
            path: Path to checkpoint file (JSON format)
            atomic: If True, use atomic writes (temp file + rename)
        """
        self._path = Path(path)
        self._atomic = atomic

    @property
    def path(self) -> Path:
        """Return checkpoint file path."""
        return self._path

    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self._path.exists()

    def load(self) -> dict[str, Any]:
        """
        Load checkpoint state from file.

        Returns:
            Checkpoint state dict, or empty dict if file doesn't exist.
        """
        if not self._path.exists():
            return {}

        try:
            with open(self._path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load checkpoint %s: %s", self._path, e)
            return {}

    def save(self, state: dict[str, Any]) -> None:
        """
        Save checkpoint state to file.

        Args:
            state: State dictionary to save (must be JSON-serializable)
        """
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._atomic:
            # Write to temp file, then rename (atomic on POSIX)
            fd, temp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".checkpoint_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2, default=str)
                os.replace(temp_path, self._path)
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        else:
            with open(self._path, "w") as f:
                json.dump(state, f, indent=2, default=str)

    def clear(self) -> None:
        """Remove checkpoint file."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to remove checkpoint %s: %s", self._path, e)

    @classmethod
    @contextmanager
    def resumable(
        cls,
        path: str | Path,
        atomic: bool = True,
        auto_save: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """
        Context manager for resumable processing.

        Loads existing checkpoint on entry, saves on exit.

        Args:
            path: Checkpoint file path
            atomic: Use atomic writes
            auto_save: If True, save state on exit (even on exception)

        Yields:
            Mutable state dict (loaded from checkpoint or empty)

        Example::

            with Checkpoint.resumable("/tmp/cp.json") as state:
                start = state.get("progress", 0)
                for i in range(start, 100):
                    process(i)
                    state["progress"] = i + 1

            # Checkpoint file is removed if loop completes
        """
        checkpoint = cls(path, atomic=atomic)
        state = checkpoint.load()

        try:
            yield state
            # Completed successfully - clear checkpoint
            checkpoint.clear()
        except Exception:
            # Save state on exception for resume
            if auto_save:
                checkpoint.save(state)
            raise


class PipelineCheckpoint:
    """
    Checkpoint manager specifically for pipeline processing.

    Provides structured state management with helper methods for
    tracking window/file progress.

    Example::

        cp = PipelineCheckpoint("/tmp/pipeline_state.json")

        # Initialize or resume
        if cp.can_resume():
            state = cp.state
            start_idx = state.last_completed + 1
            logger.info("Resuming from window %d", start_idx)
        else:
            cp.initialize(total_windows=100)
            start_idx = 0

        # Process windows
        for idx in range(start_idx, cp.state.total):
            process_window(idx)
            cp.mark_completed(idx)

        # Complete - clears checkpoint
        cp.finish()
    """

    def __init__(self, path: str | Path) -> None:
        """
        Initialize pipeline checkpoint.

        Args:
            path: Path to checkpoint file
        """
        self._checkpoint = Checkpoint(path, atomic=True)
        self._state: CheckpointState | None = None

    @property
    def path(self) -> Path:
        """Return checkpoint file path."""
        return self._checkpoint.path

    @property
    def state(self) -> CheckpointState:
        """Return current checkpoint state."""
        if self._state is None:
            self._load()
        return self._state  # type: ignore

    def can_resume(self) -> bool:
        """Check if there's a valid checkpoint to resume from."""
        if not self._checkpoint.exists():
            return False

        self._load()
        return self._state is not None and self._state.last_completed >= 0

    def _load(self) -> None:
        """Load state from checkpoint file."""
        data = self._checkpoint.load()
        if data:
            self._state = CheckpointState.from_dict(data)
        else:
            self._state = CheckpointState()

    def initialize(
        self,
        total: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a new processing run.

        Args:
            total: Total number of items to process
            metadata: Additional metadata to store
        """
        self._state = CheckpointState(
            last_completed=-1,
            total=total,
            started_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self._save()

    def mark_completed(
        self,
        index: int,
        metadata_update: dict[str, Any] | None = None,
    ) -> None:
        """
        Mark an item as completed and save checkpoint.

        Args:
            index: Index of completed item
            metadata_update: Optional metadata to merge
        """
        if self._state is None:
            raise RuntimeError("Checkpoint not initialized - call initialize() first")

        self._state.last_completed = index
        self._state.updated_at = datetime.now().isoformat()

        if metadata_update:
            self._state.metadata.update(metadata_update)

        self._save()

    def _save(self) -> None:
        """Save current state to checkpoint file."""
        if self._state is not None:
            self._checkpoint.save(self._state.to_dict())

    def finish(self) -> None:
        """Mark processing as complete and remove checkpoint."""
        self._checkpoint.clear()
        self._state = None

    def get_progress(self) -> tuple[int, int]:
        """
        Return progress as (completed, total).

        Returns:
            Tuple of (items completed, total items)
        """
        if self._state is None:
            return (0, 0)
        # last_completed is -1 based, so add 1 for count
        return (self._state.last_completed + 1, self._state.total)

    def get_progress_pct(self) -> float:
        """
        Return progress as percentage (0-100).

        Returns:
            Progress percentage
        """
        completed, total = self.get_progress()
        if total == 0:
            return 0.0
        return 100.0 * completed / total
