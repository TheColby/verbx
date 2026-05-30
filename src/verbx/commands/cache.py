"""Cache inspection and cleanup CLI commands."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def cache_info(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Show cache statistics."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    wavs = sorted(root.glob("*.wav"))
    metas = sorted(root.glob("*.meta.json"))
    total_bytes = (
        sum(path.stat().st_size for path in root.glob("*") if path.is_file())
        if root.exists()
        else 0
    )

    table = Table(title="Cache Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("cache_dir", str(root))
    table.add_row("wav_files", str(len(wavs)))
    table.add_row("meta_files", str(len(metas)))
    table.add_row("size_mb", f"{total_bytes / (1024 * 1024):.3f}")
    console.print(table)


def cache_clear(
    cache_dir: str = typer.Option(".verbx_cache/irs", "--cache-dir"),
) -> None:
    """Clear IR cache directory."""
    root = Path(cache_dir)
    if root.exists() and not root.is_dir():
        msg = f"Cache path is not a directory: {root}"
        raise typer.BadParameter(msg)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    console.print(f"Cleared cache: {root}")
