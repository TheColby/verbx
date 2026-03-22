from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.api import analyze_file, generate_ir, render_file


def test_analyze_file_returns_metrics(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / float(sr)
    audio = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float64)[:, np.newaxis]
    infile = tmp_path / "tone.wav"
    sf.write(str(infile), audio, sr)

    metrics = analyze_file(infile)

    assert "rms" in metrics
    assert metrics["duration"] > 0.0


def test_generate_ir_writes_audio_and_meta(tmp_path: Path) -> None:
    outfile = tmp_path / "generated_ir.wav"

    result = generate_ir(outfile, mode="hybrid", length=0.25, sr=8_000, channels=1, seed=123)

    assert outfile.exists()
    assert Path(str(result["meta_path"])).exists()
    assert result["sample_rate"] == 8_000


def test_render_file_writes_output(tmp_path: Path) -> None:
    sr = 16_000
    audio = np.zeros((sr, 1), dtype=np.float64)
    audio[0, 0] = 1.0
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, sr)

    report = render_file(
        infile,
        outfile,
        engine="algo",
        rt60=0.5,
        wet=0.25,
        dry=0.75,
        progress=False,
        silent=True,
    )

    assert outfile.exists()
    assert report["engine"] == "algo"
