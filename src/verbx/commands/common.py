"""Shared helpers for CLI command modules."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

progress_console = Console(force_terminal=True, color_system="truecolor")


@contextmanager
def processing_status(description: str, *, enabled: bool = True) -> Any:
    """Render a compact single-task status indicator."""
    progress = Progress(
        SpinnerColumn(style="bold cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(
            bar_width=24,
            complete_style="bright_green",
            finished_style="green",
            pulse_style="bright_blue",
        ),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=progress_console,
        transient=True,
        disable=not enabled,
    )
    progress.start()
    task = progress.add_task(str(description), total=1)
    try:
        yield progress
        progress.update(task, completed=1)
    finally:
        progress.stop()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
