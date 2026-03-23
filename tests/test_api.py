from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.api import analyze_file, generate_ir, render_file
from verbx.ir.generator import IRGenConfig


def test_api_render_file_and_analyze_file(tmp_path: Path) -> None:
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[20:100, 0] = 0.3
    sf.write(str(infile), audio, 24_000)

    summary = render_file(
        infile,
        outfile,
        engine="algo",
        rt60=0.4,
        wet=0.2,
        dry=0.9,
        silent=True,
        write_analysis=False,
    )
    assert outfile.exists()
    assert isinstance(summary, dict)

    metrics = analyze_file(outfile, include_loudness=True)
    assert "peak_dbfs" in metrics


def test_api_generate_ir_writes_output(tmp_path: Path) -> None:
    outfile = tmp_path / "ir.wav"
    result = generate_ir(
        outfile,
        config=IRGenConfig(mode="fdn", length=0.25, sr=24_000, channels=1, seed=7),
        cache_dir=tmp_path / "cache",
        write=True,
    )
    assert outfile.exists()
    assert int(result["sample_rate"]) == 24_000
    assert int(result["channels"]) == 1
