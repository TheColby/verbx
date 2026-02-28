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


def test_ir_gen_format_switch_overrides_extension(tmp_path: Path) -> None:
    out_base = tmp_path / "custom_name.placeholder"
    expected = out_base.with_suffix(".aiff")

    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_base),
            "--format",
            "aiff",
            "--mode",
            "fdn",
            "--length",
            "0.5",
            "--sr",
            "8000",
            "--channels",
            "1",
            "--seed",
            "17",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert expected.exists()
    audio, sr = sf.read(str(expected), always_2d=True, dtype="float32")
    assert sr == 8000
    assert audio.shape[1] == 1


def test_ir_gen_with_explicit_f0(tmp_path: Path) -> None:
    out_ir = tmp_path / "with_f0.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "modal",
            "--length",
            "0.6",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--f0",
            "64 Hz",
        ],
    )

    assert result.exit_code == 0, result.stdout
    meta_path = out_ir.with_suffix(".wav.ir.meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    params = payload["params"]
    assert abs(float(params["f0_hz"]) - 64.0) < 1e-6


def test_ir_gen_analyze_input_tuning(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    src = (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)[:, np.newaxis]
    in_wav = tmp_path / "source.wav"
    sf.write(str(in_wav), src, sr)

    out_ir = tmp_path / "tuned_ir.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "hybrid",
            "--length",
            "0.8",
            "--sr",
            "16000",
            "--channels",
            "1",
            "--analyze-input",
            str(in_wav),
        ],
    )

    assert result.exit_code == 0, result.stdout
    meta_path = out_ir.with_suffix(".wav.ir.meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    params = payload["params"]
    assert float(params["f0_hz"]) > 150.0
    assert float(params["f0_hz"]) < 300.0
    assert len(params["harmonic_targets_hz"]) >= 3


def test_ir_gen_resonator_layer(tmp_path: Path) -> None:
    out_ir = tmp_path / "resonated.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "hybrid",
            "--length",
            "0.8",
            "--sr",
            "12000",
            "--channels",
            "2",
            "--seed",
            "31",
            "--resonator",
            "--resonator-mix",
            "0.45",
            "--resonator-modes",
            "12",
            "--resonator-low-hz",
            "90",
            "--resonator-high-hz",
            "2400",
        ],
    )

    assert result.exit_code == 0, result.stdout
    audio, sr = sf.read(str(out_ir), always_2d=True, dtype="float32")
    assert sr == 12000
    assert np.all(np.isfinite(audio))
    assert float(np.max(np.abs(audio))) > 1e-6

    payload = json.loads(out_ir.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["resonator"] is True
    assert int(params["resonator_modes"]) == 12
    assert abs(float(params["resonator_mix"]) - 0.45) < 1e-9


def test_ir_cache_hit(tmp_path: Path) -> None:
    cfg = IRGenConfig(mode="stochastic", length=0.5, sr=8000, channels=1, seed=7)

    _, _, _, _, hit1 = generate_or_load_cached_ir(cfg, cache_dir=tmp_path)
    _, _, _, _, hit2 = generate_or_load_cached_ir(cfg, cache_dir=tmp_path)

    assert hit1 is False
    assert hit2 is True


def test_ir_fit_heuristic_scoring_outputs_top_k(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = (0.25 * np.sin(2.0 * np.pi * 196.0 * t)).astype(np.float32)[:, np.newaxis]
    infile = tmp_path / "fit_input.wav"
    out_ir = tmp_path / "fit_ir.wav"
    sf.write(str(infile), audio, sr)

    result = runner.invoke(
        app,
        [
            "ir",
            "fit",
            str(infile),
            str(out_ir),
            "--top-k",
            "2",
            "--candidate-pool",
            "4",
            "--length",
            "0.5",
            "--fit-workers",
            "1",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    out_1 = tmp_path / "fit_ir_01.wav"
    out_2 = tmp_path / "fit_ir_02.wav"
    assert out_1.exists()
    assert out_2.exists()

    meta_path = out_1.with_suffix(".wav.ir.meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    fit = payload["fit"]
    assert fit["rank"] == 1
    assert float(fit["score"]) > 0.0
    assert "errors" in fit


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
