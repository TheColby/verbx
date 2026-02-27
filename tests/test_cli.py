from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from verbx.cli import app

runner = CliRunner()


def test_cli_boots() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout
    assert "analyze" in result.stdout
    assert "presets" in result.stdout
    assert "suggest" in result.stdout
    assert "ir" in result.stdout
    assert "cache" in result.stdout
    assert "batch" in result.stdout


def test_render_creates_output_and_analysis(tmp_path: Path) -> None:
    audio = np.zeros((2048, 2), dtype=np.float32)
    audio[100:130, 0] = 0.6
    audio[100:130, 1] = -0.6

    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--repeat",
            "1",
            "--rt60",
            "45",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.stdout

    out_audio, out_sr = sf.read(str(outfile), always_2d=True, dtype="float32")
    assert out_sr == 48_000
    assert out_audio.shape == audio.shape

    analysis_path = Path(f"{outfile}.analysis.json")
    with analysis_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "input" in payload
    assert "output" in payload
    assert payload["engine"] == "algo"


def test_analyze_lufs_mode(tmp_path: Path) -> None:
    audio = np.zeros((4096, 2), dtype=np.float32)
    audio[64:512, :] = 0.2
    infile = tmp_path / "analyze.wav"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(app, ["analyze", str(infile), "--lufs"])
    assert result.exit_code == 0
    assert "integrated_lufs" in result.stdout
