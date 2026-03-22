from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.api import analyze_file, generate_ir, render_file


def _write_test_tone(path: Path, *, sr: int = 16_000, seconds: float = 0.25) -> None:
    t = np.arange(int(sr * seconds), dtype=np.float64) / float(sr)
    tone = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    audio = np.column_stack((tone, tone)).astype(np.float64)
    sf.write(str(path), audio, sr)


def test_generate_ir_api_returns_audio_and_metadata() -> None:
    audio, sr, meta = generate_ir(mode="fdn", length=0.2, sr=8_000, channels=1, seed=7)

    assert sr == 8_000
    assert audio.ndim == 2
    assert audio.shape[1] == 1
    assert isinstance(meta, dict)
    assert meta.get("mode") == "fdn"


def test_analyze_file_api_reads_file_and_returns_metrics(tmp_path: Path) -> None:
    infile = tmp_path / "analyze_in.wav"
    _write_test_tone(infile)

    metrics = analyze_file(infile, include_loudness=True)

    assert metrics["channels"] == 2.0
    assert "integrated_lufs" in metrics


def test_render_file_api_renders_and_returns_report(tmp_path: Path) -> None:
    infile = tmp_path / "render_in.wav"
    outfile = tmp_path / "render_out.wav"
    _write_test_tone(infile, seconds=0.15)

    report = render_file(
        infile,
        outfile,
        engine="algo",
        rt60=0.4,
        wet=0.25,
        dry=0.75,
        repeat=1,
        silent=True,
    )

    assert outfile.exists()
    assert report["sample_rate"] == 16_000
    assert int(report["channels"]) == 2


def test_render_file_rejects_unknown_option(tmp_path: Path) -> None:
    infile = tmp_path / "render_in.wav"
    outfile = tmp_path / "render_out.wav"
    _write_test_tone(infile)

    try:
        render_file(infile, outfile, nope=1)
    except ValueError as exc:
        assert "Unsupported RenderConfig option" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported option")
