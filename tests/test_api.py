from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.api import analyze_file, generate_ir, render_file


def _write_test_tone(path: Path, sr: int = 16_000) -> None:
    t = np.arange(sr // 4, dtype=np.float64) / float(sr)
    tone = (0.1 * np.sin(2.0 * np.pi * 220.0 * t))[:, np.newaxis]
    sf.write(str(path), tone, sr)


def test_analyze_file_returns_metrics(tmp_path: Path) -> None:
    infile = tmp_path / "in.wav"
    _write_test_tone(infile)

    metrics = analyze_file(infile)

    assert "rms" in metrics
    assert "spectral_centroid" in metrics
    assert metrics["duration"] > 0.0


def test_generate_ir_writes_audio_and_metadata(tmp_path: Path) -> None:
    out_ir = tmp_path / "generated_ir.wav"

    report = generate_ir(
        out_ir,
        mode="hybrid",
        length=0.2,
        sr=8_000,
        channels=1,
        seed=17,
    )

    assert out_ir.exists()
    assert report["sample_rate"] == 8_000
    assert report["channels"] == 1
    meta_path = out_ir.with_suffix(".wav.ir.meta.json")
    assert meta_path.exists()


def test_render_file_writes_output(tmp_path: Path) -> None:
    infile = tmp_path / "dry.wav"
    outfile = tmp_path / "wet.wav"
    _write_test_tone(infile)

    report = render_file(
        infile,
        outfile,
        engine="algo",
        rt60=0.3,
        wet=0.5,
        dry=0.5,
        silent=True,
        progress=False,
    )

    assert outfile.exists()
    assert report["engine"] == "algo"
    assert report["output_samples"] > 0
