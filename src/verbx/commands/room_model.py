"""Room geometry inspection and RT60-to-geometry helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from verbx.core.early_reflections import material_absorption
from verbx.core.room_geometry import RoomGeometry, infer_room_geometry_from_rt60

console = Console()


def room_model(
    dims_m: Annotated[str | None, typer.Option(
        "--dims-m",
        help="Explicit room dimensions as width,depth,height in meters.",
    )] = None,
    rt60: Annotated[float | None, typer.Option(
        "--rt60",
        min=0.05,
        help="Infer a rectangular room from RT60 plus absorption/material assumptions.",
    )] = None,
    absorption: Annotated[float | None, typer.Option(
        "--absorption",
        min=0.01,
        max=0.99,
        help="Mean absorption coefficient used with --rt60 inference.",
    )] = None,
    material: Annotated[str, typer.Option(
        "--material",
        help="Material preset for wall absorption when --absorption is not given.",
    )] = "studio",
    source_pos_m: Annotated[str, typer.Option(
        "--source-pos-m",
        help="Source position as x,y,z in meters.",
    )] = "2.0,2.0,1.5",
    listener_pos_m: Annotated[str, typer.Option(
        "--listener-pos-m",
        help="Listener position as x,y,z in meters.",
    )] = "5.0,3.5,1.5",
    json_out: Annotated[Path | None, typer.Option(
        "--json-out",
        resolve_path=True,
        help="Optional path to write the full room-model payload as JSON.",
    )] = None,
) -> None:
    """Inspect a room geometry or infer one from RT60 and absorption."""
    if dims_m is None and rt60 is None:
        raise typer.BadParameter("Provide either --dims-m or --rt60.")
    if dims_m is not None and rt60 is not None:
        raise typer.BadParameter("Use either --dims-m or --rt60, not both.")

    try:
        source = _parse_vec3(source_pos_m, option_name="--source-pos-m")
        listener = _parse_vec3(listener_pos_m, option_name="--listener-pos-m")
        if dims_m is not None:
            geometry = RoomGeometry(
                room_dims_m=_parse_vec3(dims_m, option_name="--dims-m"),
                source_pos_m=source,
                listener_pos_m=listener,
                wall_materials=_wall_material_map(str(material)),
            )
            inferred = False
            assert geometry.mean_absorption is not None
            resolved_absorption = float(geometry.mean_absorption)
        else:
            assert rt60 is not None
            resolved_absorption = (
                float(absorption)
                if absorption is not None
                else material_absorption(material, 0.35)
            )
            geometry = infer_room_geometry_from_rt60(
                rt60_s=float(rt60),
                mean_absorption=resolved_absorption,
                source_pos_m=source,
                listener_pos_m=listener,
                wall_material=str(material),
            )
            inferred = True
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = geometry.summary()
    payload: dict[str, Any] = {
        "mode": "infer-rt60" if inferred else "inspect-geometry",
        "material": str(material),
        "assumed_absorption": float(resolved_absorption),
        "geometry": summary,
    }
    if rt60 is not None:
        payload["rt60_s"] = float(rt60)

    _print_room_model_table(payload)

    if json_out is not None:
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_vec3(raw: str, *, option_name: str) -> tuple[float, float, float]:
    cleaned = raw.strip()
    parts = [token.strip() for token in cleaned.split(",") if token.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"{option_name} must contain exactly 3 comma-separated values.")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{option_name} must contain numeric values.") from exc
    return (values[0], values[1], values[2])


def _print_room_model_table(payload: dict[str, object]) -> None:
    geometry = payload["geometry"]
    assert isinstance(geometry, dict)
    ratios = geometry.get("aspect_ratios", {})
    warnings = geometry.get("warnings", [])

    table = Table(title="Room Model")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("mode", str(payload.get("mode", "")))
    table.add_row("material", str(payload.get("material", "")))
    rt60_value = payload.get("rt60_s")
    if isinstance(rt60_value, (int, float)):
        table.add_row("rt60_s", f"{float(rt60_value):.3f}")
    assumed_absorption = payload.get("assumed_absorption", 0.0)
    table.add_row(
        "assumed_absorption",
        f"{float(assumed_absorption) if isinstance(assumed_absorption, (int, float)) else 0.0:.3f}",
    )
    dims_values = geometry.get("room_dims_m", [])
    table.add_row(
        "dims_m",
        ",".join(
            f"{float(v):.2f}" for v in dims_values if isinstance(v, (int, float))
        ),
    )
    table.add_row("volume_m3", f"{float(geometry.get('volume_m3', 0.0)):.2f}")
    table.add_row("surface_area_m2", f"{float(geometry.get('surface_area_m2', 0.0)):.2f}")
    table.add_row(
        "direct_distance_m",
        f"{float(geometry.get('direct_distance_m', 0.0)):.3f}",
    )
    table.add_row(
        "direct_pre_delay_ms",
        f"{float(geometry.get('direct_path_pre_delay_ms', 0.0)):.2f}",
    )
    table.add_row("bolt_score", f"{float(geometry.get('bolt_score', 0.0)):.2f}")
    if isinstance(ratios, dict):
        table.add_row(
            "aspect_ratios",
            ", ".join(
                f"{key}={float(value):.2f}"
                for key, value in ratios.items()
                if isinstance(value, (int, float))
            ),
        )
    if isinstance(warnings, list) and len(warnings) > 0:
        table.add_row("warnings", " | ".join(str(item) for item in warnings))
    console.print(table)


def _wall_material_map(material: str) -> dict[str, str]:
    return {
        "left": material,
        "right": material,
        "front": material,
        "rear": material,
        "ceiling": material,
        "floor": material,
    }
