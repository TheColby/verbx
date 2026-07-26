"""Preset inspection and validation CLI commands."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from verbx.config import RenderConfig
from verbx.presets.default_presets import preset_names, resolve_preset
from verbx.presets.room_presets import is_room_preset_name, resolve_room_preset

console = Console()


def list_presets(
    show: str | None = typer.Option(
        None,
        "--show",
        help="Show resolved values for one preset.",
    ),
    validate: str | None = typer.Option(
        None,
        "--validate",
        help="Validate a preset's fields against RenderConfig and report any errors.",
    ),
) -> None:
    """Print available presets or one preset payload."""
    if validate is not None:
        resolved_name, payload = _resolve_preset_payload(validate)
        errors: list[str] = []
        warnings: list[str] = []
        try:
            known_fields = set(RenderConfig.__dataclass_fields__.keys())
            filtered_payload = {k: v for k, v in payload.items() if k in known_fields}
            RenderConfig(**filtered_payload)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        try:
            known = set(RenderConfig.__dataclass_fields__.keys())
        except AttributeError:
            known = set()
        unknown = [k for k in payload if k not in known]
        for key in unknown:
            warnings.append(f"unknown field '{key}' (not in RenderConfig)")

        table = Table(title=f"Preset Validation: {resolved_name}")
        table.add_column("Status", style="cyan")
        table.add_column("Detail", style="white")
        if not errors and not warnings:
            table.add_row("[green]PASS[/green]", f"All {len(payload)} fields are valid")
        for err in errors:
            table.add_row("[red]ERROR[/red]", err)
        for warning in warnings:
            table.add_row("[yellow]WARN[/yellow]", warning)
        console.print(table)
        if errors:
            raise typer.Exit(code=1)
        return

    if show is not None:
        resolved_name, payload = _resolve_preset_payload(show)
        table = Table(title=f"Preset: {resolved_name}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        for key in sorted(payload.keys()):
            table.add_row(key, str(payload[key]))
        console.print(table)
        return

    names = preset_names()
    table = Table(title="Available Presets")
    table.add_column("Preset", style="green")
    for name in names:
        table.add_row(name)
    table.add_row("room:<width>x<depth>x<height>/<material>")
    console.print(table)


def _resolve_preset_payload(value: str) -> tuple[str, dict[str, Any]]:
    try:
        if is_room_preset_name(value):
            return resolve_room_preset(value)
        return resolve_preset(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
