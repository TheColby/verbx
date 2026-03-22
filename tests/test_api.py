from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.api import analyze_file, generate_ir, render_file
from verbx.config import RenderConfig
from verbx.ir.generator import IRGenConfig


def _write_tone(path: Path, *, sr: int = 24_000, seconds: float = 0.2) -> None:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / float(sr)
    audio = (0.2 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float64)[:, np.newaxis]
    sf.write(str(path), audio, sr)


def test_render_file_api_writes_audio_and_report(tmp_path: Path) -> None:
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    _write_tone(infile)

    report = render_file(
        infile,
        outfile,
        config=RenderConfig(engine="algo", rt60=2.0, silent=True, progress=False),
    )

    assert outfile.exists()
    assert report["sample_rate"] == 24_000
    assert report["engine"] in {"algo", "conv"}


def test_render_file_rejects_unknown_override() -> None:
    try:
        render_file("in.wav", "out.wav", does_not_exist=True)
    except ValueError as exc:
        assert "Unknown RenderConfig override" in str(exc)
    else:
        raise AssertionError("expected unknown override validation")


def test_generate_ir_api_writes_meta_sidecar(tmp_path: Path) -> None:
    out_ir = tmp_path / "generated_ir.wav"
    payload = generate_ir(
        out_ir,
        config=IRGenConfig(length=0.1, sr=12_000, channels=1, mode="stochastic"),
    )

    meta_path = Path(str(payload["metadata_path"]))
    assert out_ir.exists()
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["mode"] == "stochastic"


def test_analyze_file_api_writes_optional_outputs(tmp_path: Path) -> None:
    infile = tmp_path / "analyze.wav"
    json_out = tmp_path / "analysis.json"
    frames_out = tmp_path / "frames.csv"
    _write_tone(infile)

    payload = analyze_file(
        infile,
        include_loudness=True,
        json_out=json_out,
        frames_out=frames_out,
    )

    assert payload["sample_rate"] == 24_000
    assert "integrated_lufs" in payload["metrics"]
    assert json_out.exists()
    assert frames_out.exists()
