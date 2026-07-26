"""Shared validation helpers for command modules."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

import soundfile as sf
import typer


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
        suggestion = _did_you_mean(suffix, set(format_map.keys()))
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


def _did_you_mean(value: str, choices: set[str]) -> str | None:
    """Return a likely choice suggestion for short command identifiers."""

    matches = get_close_matches(value, sorted(choices), n=1, cutoff=0.72)
    return matches[0] if matches else None
