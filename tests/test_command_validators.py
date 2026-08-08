from __future__ import annotations

from pathlib import Path

import pytest
import typer

from verbx.commands.validators import (
    choice_error,
    ensure_distinct_paths,
    parse_delay_list_ms,
    parse_gain_list,
    parse_vec3,
    validate_output_audio_path,
)


def test_shared_numeric_parsers_accept_command_option_text() -> None:
    assert parse_delay_list_ms("1.5, 8,13", option_name="--delays") == (1.5, 8.0, 13.0)
    assert parse_gain_list(
        "0.1, 0.75",
        option_name="--gains",
        min_value=0.0,
        max_value=1.0,
    ) == (0.1, 0.75)
    assert parse_vec3("1, 2.5, -3", option_name="--position") == (1.0, 2.5, -3.0)


def test_shared_numeric_parsers_reject_invalid_values() -> None:
    with pytest.raises(typer.BadParameter, match="values must be > 0 ms"):
        parse_delay_list_ms("1,0", option_name="--delays")
    with pytest.raises(typer.BadParameter, match=r"values must be in \[0.0, 1.0\]"):
        parse_gain_list("1.2", option_name="--gains", min_value=0.0, max_value=1.0)
    with pytest.raises(typer.BadParameter, match="exactly 3"):
        parse_vec3("1,2", option_name="--position")


def test_choice_error_suggests_nearest_supported_value() -> None:
    message = choice_error("--mode", {"spectral", "linear"}, "spectrl")
    assert "Did you mean 'spectral'?" in message


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
