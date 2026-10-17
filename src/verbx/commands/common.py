"""Shared helpers for CLI command modules."""

from __future__ import annotations

import csv
import hashlib
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

from verbx.core.schema_versions import BATCH_CHECKPOINT_VERSION

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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{json.dumps(row, sort_keys=True)}\n")
    tmp.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write tabular rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    if not columns:
        columns = ["note"]
    if "metrics_error" in columns:
        columns.remove("metrics_error")
        columns.append("metrics_error")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically write UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_batch_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint payload with graceful fallback when missing/invalid."""
    if not path.exists():
        return {"version": BATCH_CHECKPOINT_VERSION, "results": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": BATCH_CHECKPOINT_VERSION, "results": []}
    if not isinstance(payload, dict):
        return {"version": BATCH_CHECKPOINT_VERSION, "results": []}
    results = payload.get("results")
    if not isinstance(results, list):
        payload["results"] = []
    return payload


def checkpoint_success_outfiles(payload: dict[str, Any]) -> set[str]:
    """Return canonical outfile paths marked successful in checkpoint payload."""
    completed: set[str] = set()
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return completed
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("success", False)):
            continue
        outfile = row.get("outfile")
        if not isinstance(outfile, str):
            continue
        out_path = Path(outfile)
        if out_path.exists():
            completed.add(str(out_path.resolve()))
    return completed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if chunk == b"":
                break
            digest.update(chunk)
    return digest.hexdigest()
