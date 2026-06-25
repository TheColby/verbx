from __future__ import annotations

from pathlib import Path

import pytest
import typer

from verbx.commands.validators import ensure_distinct_paths, validate_output_audio_path


def test_ensure_distinct_paths_rejects_same_resolved_path(tmp_path: Path) -> None:
    path = tmp_path / "same.wav"
    path.write_bytes(b"")

    with pytest.raises(typer.BadParameter, match="INFILE and OUTFILE"):
        ensure_distinct_paths(path, path, "INFILE", "OUTFILE")


def test_validate_output_audio_path_suggests_likely_extension(tmp_path: Path) -> None:
    out = tmp_path / "render.wavee"

    with pytest.raises(typer.BadParameter, match=r"Did you mean '\.wav'"):
        validate_output_audio_path(out, "auto")


def test_validate_output_audio_path_accepts_float64_wav(tmp_path: Path) -> None:
    validate_output_audio_path(tmp_path / "render.wav", "float64")


def test_validate_output_audio_path_rejects_bad_subtype(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="Unsupported --out-subtype"):
        validate_output_audio_path(tmp_path / "render.wav", "int7")
