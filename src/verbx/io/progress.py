"""Progress display utilities for long-running render stages."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


class RenderProgress:
    """Progress manager for read/process/write/analyze workflow.

    Designed as a context manager to guarantee progress teardown on errors.
    """

    __slots__ = (
        "_analyze_task",
        "_process_task",
        "_progress",
        "_read_task",
        "_started",
        "_total_passes",
        "_write_task",
        "enabled",
    )

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=26),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=not self.enabled,
        )
        self._started = False
        self._read_task: TaskID | None = None
        self._process_task: TaskID | None = None
        self._write_task: TaskID | None = None
        self._analyze_task: TaskID | None = None
        self._total_passes = 1

    def __enter__(self) -> RenderProgress:
        self._progress.start()
        self._started = True
        self._read_task = self._progress.add_task("Read", total=1)
        self._process_task = self._progress.add_task("Process pass 0/1", total=1)
        self._write_task = self._progress.add_task("Write", total=1)
        self._analyze_task = self._progress.add_task("Analyze", total=1)
        return self

    def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> None:
        if self._started:
            self._progress.stop()

    def set_passes(self, total_passes: int) -> None:
        """Set process-stage repeat count."""
        self._total_passes = max(1, total_passes)
        if self._process_task is not None:
            self._progress.update(
                self._process_task,
                total=self._total_passes,
                completed=0,
                description=f"Process pass 0/{self._total_passes}",
            )

    def mark_read(self) -> None:
        """Mark read stage complete."""
        if self._read_task is not None:
            self._progress.update(self._read_task, completed=1)

    def mark_process_pass(self, current_pass: int) -> None:
        """Advance processing progress with pass counters."""
        if self._process_task is not None:
            clamped = max(0, min(current_pass, self._total_passes))
            self._progress.update(
                self._process_task,
                completed=clamped,
                description=f"Process pass {clamped}/{self._total_passes}",
            )

    def mark_write(self) -> None:
        """Mark write stage complete."""
        if self._write_task is not None:
            self._progress.update(self._write_task, completed=1)

    def mark_analyze(self) -> None:
        """Mark analysis stage complete."""
        if self._analyze_task is not None:
            self._progress.update(self._analyze_task, completed=1)
