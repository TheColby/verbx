"""Shared validation helpers for command modules."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import cast

import soundfile as sf
import typer


def parse_delay_list_ms(raw: str | None, *, option_name: str) -> tuple[float, ...]:
    """Parse a comma-separated millisecond delay list for CLI options."""

    if raw is None:
        return ()
    cleaned = raw.strip()
    if cleaned == "":
        return ()
    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            delay = float(part)
        except ValueError as exc:
            msg = f"{option_name} expects a comma-separated float list in milliseconds."
            raise typer.BadParameter(msg) from exc
        if delay <= 0.0:
            msg = f"{option_name} values must be > 0 ms."
            raise typer.BadParameter(msg)
        values.append(delay)
    if len(values) == 0:
        msg = f"{option_name} must include at least one numeric value."
        raise typer.BadParameter(msg)
    return tuple(values)


def parse_gain_list(
    raw: str,
    *,
    option_name: str,
    min_value: float,
    max_value: float,
) -> tuple[float, ...]:
    """Parse one or more comma-separated gain values for CLI options."""

    cleaned = raw.strip()
    if cleaned == "":
        msg = f"{option_name} requires at least one numeric value."
        raise typer.BadParameter(msg)

    values: list[float] = []
    for token in cleaned.split(","):
        part = token.strip()
        if part == "":
            continue
        try:
            gain = float(part)
        except ValueError as exc:
            msg = f"{option_name} expects float values, optionally comma-separated."
            raise typer.BadParameter(msg) from exc
        if gain < min_value or gain > max_value:
            msg = f"{option_name} values must be in [{min_value}, {max_value}]."
            raise typer.BadParameter(msg)
        values.append(gain)

    if len(values) == 0:
        msg = f"{option_name} requires at least one numeric value."
        raise typer.BadParameter(msg)
    return tuple(values)


def parse_vec3(raw: str, *, option_name: str) -> tuple[float, float, float]:
    """Parse a 3D vector from comma-separated CLI text."""

    cleaned = str(raw).strip()
    parts = [part.strip() for part in cleaned.split(",") if part.strip() != ""]
    if len(parts) != 3:
        msg = f"{option_name} expects exactly 3 comma-separated values: x,y,z"
        raise typer.BadParameter(msg)
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        msg = f"{option_name} expects float values: x,y,z"
        raise typer.BadParameter(msg) from exc
    return cast(tuple[float, float, float], values)


def did_you_mean(value: str, choices: set[str]) -> str | None:
    """Return a likely choice suggestion for short command identifiers."""

    token = str(value).strip().lower()
    if token == "":
        return None
    matches = get_close_matches(token, sorted(choices), n=1, cutoff=0.5)
    return str(matches[0]) if matches else None


def choice_error(option_name: str, choices: set[str], actual: str) -> str:
    """Build a consistent invalid-choice message with an optional suggestion."""

    options = ", ".join(sorted(choices))
    suggestion = did_you_mean(actual, choices)
    if suggestion is not None:
        return f"{option_name} must be one of: {options}. Did you mean '{suggestion}'?"
    return f"{option_name} must be one of: {options}."


def ensure_distinct_paths(
    in_path: Path,
    out_path: Path,
    in_label: str,
    out_label: str,
) -> None:
    """Ensure input and output paths are not identical."""

    if in_path.resolve() == out_path.resolve():
        msg = f"{in_label} and {out_label} must be different paths."
        raise typer.BadParameter(msg)


def validate_output_audio_path(path: Path, out_subtype_mode: str) -> None:
    """Validate output extension and requested SoundFile subtype support."""

    suffix = path.suffix.lower().lstrip(".")
    if suffix == "":
        msg = f"Output path must include an audio file extension: {path} (try .wav or .flac)."
        raise typer.BadParameter(msg)

    format_map = {
        "wav": "WAV",
        "w64": "W64",
        "rf64": "RF64",
        "flac": "FLAC",
        "aif": "AIFF",
        "aiff": "AIFF",
        "ogg": "OGG",
        "caf": "CAF",
        "au": "AU",
    }
    fmt = format_map.get(suffix)
    if fmt is None:
        supported = ", ".join(f".{ext}" for ext in sorted(format_map))
        suggestion = did_you_mean(suffix, set(format_map.keys()))
        if suggestion is None:
            for ext in sorted(format_map.keys()):
                if suffix.startswith(ext) or ext.startswith(suffix):
                    suggestion = ext
                    break
        if suggestion is not None:
            msg = (
                f"Unsupported output audio extension: .{suffix}. "
                f"Did you mean '.{suggestion}'? Supported: {supported}."
            )
        else:
            msg = f"Unsupported output audio extension: .{suffix}. Supported: {supported}."
        raise typer.BadParameter(msg)

    subtype_map = {
        "auto": None,
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
    }
    subtype = subtype_map.get(out_subtype_mode)
    if out_subtype_mode not in subtype_map:
        msg = f"Unsupported --out-subtype value: {out_subtype_mode}"
        raise typer.BadParameter(msg)

    if subtype is None:
        if not sf.check_format(fmt):
            msg = f"SoundFile cannot write format '{fmt}' for output path {path}"
            raise typer.BadParameter(msg)
    elif not sf.check_format(fmt, subtype):
        supported_subtypes: list[str] = []
        for mode, candidate in subtype_map.items():
            if candidate is None:
                continue
            if sf.check_format(fmt, candidate):
                supported_subtypes.append(mode)
        supported_text = ", ".join(sorted(supported_subtypes))
        msg = (
            f"Subtype '{subtype}' is not supported for format '{fmt}'. "
            f"Use --out-subtype auto or one of: {supported_text}."
        )
        raise typer.BadParameter(msg)
