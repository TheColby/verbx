from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from verbx.cli import app
from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine
from verbx.core.tempo import parse_note_duration_seconds, parse_pre_delay_ms
from verbx.ir.generator import IRGenConfig, generate_or_load_cached_ir

runner = CliRunner()


def test_ir_gen_writes_wav_and_meta(tmp_path: Path) -> None:
    out_ir = tmp_path / "test_ir.wav"

    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "hybrid",
            "--length",
            "1.0",
            "--sr",
            "16000",
            "--channels",
            "2",
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out_ir.exists()

    audio, sr = sf.read(str(out_ir), always_2d=True, dtype="float32")
    assert sr == 16000
    assert audio.shape[1] == 2
    assert audio.shape[0] == 16000

    meta_path = out_ir.with_suffix(f"{out_ir.suffix}.ir.meta.json")
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "hybrid"


def test_ir_cache_hit(tmp_path: Path) -> None:
    cfg = IRGenConfig(mode="stochastic", length=0.5, sr=8000, channels=1, seed=7)

    _, _, _, _, hit1 = generate_or_load_cached_ir(cfg, cache_dir=tmp_path)
    _, _, _, _, hit2 = generate_or_load_cached_ir(cfg, cache_dir=tmp_path)

    assert hit1 is False
    assert hit2 is True


def test_convolution_with_generated_ir_is_nonzero(tmp_path: Path) -> None:
    cfg = IRGenConfig(mode="modal", length=0.5, sr=16000, channels=1, seed=9)
    ir_audio, ir_sr, _, ir_path, _ = generate_or_load_cached_ir(cfg, cache_dir=tmp_path)
    assert ir_audio.shape[0] > 0

    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=1.0,
            dry=0.0,
            ir_path=str(ir_path),
            ir_normalize="none",
            partition_size=512,
            tail_limit=None,
            threads=None,
        )
    )

    impulse = np.zeros((1024, 1), dtype=np.float32)
    impulse[0, 0] = 1.0
    out = engine.process(impulse, sr=ir_sr)

    assert np.any(np.abs(out) > 1e-7)


def test_tempo_note_parsing() -> None:
    eighth_dotted = parse_note_duration_seconds("1/8D", bpm=120.0)
    assert abs(eighth_dotted - 0.375) < 1e-6

    quarter_triplet = parse_note_duration_seconds("1/4T", bpm=120.0)
    assert abs(quarter_triplet - (1.0 / 3.0)) < 1e-6

    ms = parse_pre_delay_ms("1/16", bpm=120.0, fallback_ms=20.0)
    assert abs(ms - 125.0) < 1e-6
