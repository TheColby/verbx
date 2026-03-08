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


def test_ir_morph_command_writes_output_and_meta(tmp_path: Path) -> None:
    sr = 16_000
    a = np.zeros((2400, 1), dtype=np.float32)
    b = np.zeros((2400, 1), dtype=np.float32)
    a[0, 0] = 1.0
    a[220, 0] = 0.42
    b[0, 0] = 1.0
    b[900, 0] = 0.35
    a_path = tmp_path / "morph_a.wav"
    b_path = tmp_path / "morph_b.wav"
    out_path = tmp_path / "morphed.wav"
    sf.write(str(a_path), a, sr)
    sf.write(str(b_path), b, sr)

    result = runner.invoke(
        app,
        [
            "ir",
            "morph",
            str(a_path),
            str(b_path),
            str(out_path),
            "--mode",
            "envelope-aware",
            "--alpha",
            "0.4",
            "--early-ms",
            "55",
            "--phase-coherence",
            "0.8",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out_path.exists()
    out, out_sr = sf.read(str(out_path), always_2d=True, dtype="float32")
    assert out_sr == sr
    assert out.shape[1] == 1
    assert np.all(np.isfinite(out))
    payload = json.loads(out_path.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "ir-morph"
    assert payload["params"]["mode"] == "envelope-aware"
    assert abs(float(payload["params"]["alpha"]) - 0.4) < 1e-6
    assert "quality" in payload


def test_ir_gen_supports_tvu_and_dfm_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_tvu.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.5",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-lines",
            "4",
            "--fdn-matrix",
            "tv_unitary",
            "--fdn-tv-rate-hz",
            "0.12",
            "--fdn-tv-depth",
            "0.45",
            "--fdn-dfm-delays-ms",
            "0.4,0.6,0.8,1.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    meta_path = out_ir.with_suffix(".wav.ir.meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["fdn_matrix"] == "tv_unitary"
    assert abs(float(params["fdn_tv_rate_hz"]) - 0.12) < 1e-6
    assert abs(float(params["fdn_tv_depth"]) - 0.45) < 1e-6
    assert len(params["fdn_dfm_delays_ms"]) == 4


def test_ir_gen_supports_sparse_high_order_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_sparse.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.4",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-lines",
            "20",
            "--fdn-sparse",
            "--fdn-sparse-degree",
            "3",
        ],
    )

    assert result.exit_code == 0, result.stdout
    meta_path = out_ir.with_suffix(".wav.ir.meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["fdn_sparse"] is True
    assert int(params["fdn_sparse_degree"]) == 3


def test_ir_gen_supports_graph_fdn_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_graph.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.4",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-matrix",
            "graph",
            "--fdn-graph-topology",
            "path",
            "--fdn-graph-degree",
            "3",
            "--fdn-graph-seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_ir.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["fdn_matrix"] == "graph"
    assert params["fdn_graph_topology"] == "path"
    assert int(params["fdn_graph_degree"]) == 3
    assert int(params["fdn_graph_seed"]) == 42


def test_ir_gen_rejects_graph_options_without_graph_matrix(tmp_path: Path) -> None:
    out_ir = tmp_path / "bad_graph.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.2",
            "--sr",
            "8000",
            "--channels",
            "1",
            "--fdn-graph-topology",
            "star",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-graph-topology/--fdn-graph-degree are only valid with" in result.output
    assert "--fdn-matrix graph" in result.output


def test_ir_gen_supports_cascaded_fdn_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_cascade.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.4",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-lines",
            "8",
            "--fdn-cascade",
            "--fdn-cascade-mix",
            "0.65",
            "--fdn-cascade-delay-scale",
            "0.35",
            "--fdn-cascade-rt60-ratio",
            "0.45",
        ],
    )

    assert result.exit_code == 0, result.stdout
    meta_path = out_ir.with_suffix(".wav.ir.meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["fdn_cascade"] is True
    assert abs(float(params["fdn_cascade_mix"]) - 0.65) < 1e-6
    assert abs(float(params["fdn_cascade_delay_scale"]) - 0.35) < 1e-6
    assert abs(float(params["fdn_cascade_rt60_ratio"]) - 0.45) < 1e-6


def test_ir_gen_rejects_cascade_with_single_line_fdn(tmp_path: Path) -> None:
    out_ir = tmp_path / "bad_cascade.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.2",
            "--sr",
            "8000",
            "--channels",
            "1",
            "--fdn-lines",
            "1",
            "--fdn-cascade",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-cascade requires at least 2 FDN lines." in result.output


def test_ir_gen_supports_multiband_fdn_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_multiband.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.5",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-rt60-low",
            "20",
            "--fdn-rt60-mid",
            "12",
            "--fdn-rt60-high",
            "6",
            "--fdn-tonal-correction-strength",
            "0.6",
            "--fdn-xover-low-hz",
            "220",
            "--fdn-xover-high-hz",
            "3200",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_ir.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    params = payload["params"]
    assert abs(float(params["fdn_rt60_low"]) - 20.0) < 1e-6
    assert abs(float(params["fdn_rt60_mid"]) - 12.0) < 1e-6
    assert abs(float(params["fdn_rt60_high"]) - 6.0) < 1e-6
    assert abs(float(params["fdn_tonal_correction_strength"]) - 0.6) < 1e-6
    assert abs(float(params["fdn_xover_low_hz"]) - 220.0) < 1e-6
    assert abs(float(params["fdn_xover_high_hz"]) - 3200.0) < 1e-6


def test_ir_gen_supports_track_c_perceptual_fdn_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_track_c.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.5",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-rt60-tilt",
            "0.35",
            "--room-size-macro",
            "0.40",
            "--clarity-macro",
            "-0.25",
            "--warmth-macro",
            "0.55",
            "--envelopment-macro",
            "0.30",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_ir.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    params = payload["params"]
    assert abs(float(params["fdn_rt60_tilt"]) - 0.35) < 1e-6
    assert abs(float(params["room_size_macro"]) - 0.40) < 1e-6
    assert abs(float(params["clarity_macro"]) + 0.25) < 1e-6
    assert abs(float(params["warmth_macro"]) - 0.55) < 1e-6
    assert abs(float(params["envelopment_macro"]) - 0.30) < 1e-6


def test_ir_gen_rejects_partial_multiband_rt60_set(tmp_path: Path) -> None:
    out_ir = tmp_path / "bad_multiband.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.2",
            "--sr",
            "8000",
            "--channels",
            "1",
            "--fdn-rt60-low",
            "20",
            "--fdn-rt60-high",
            "6",
        ],
    )
    assert result.exit_code != 0
    assert "provide all three values" in result.output


def test_ir_gen_supports_filter_feedback_controls(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_filter.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.5",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-link-filter",
            "lowpass",
            "--fdn-link-filter-hz",
            "1800",
            "--fdn-link-filter-mix",
            "0.8",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_ir.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    params = payload["params"]
    assert params["fdn_link_filter"] == "lowpass"
    assert abs(float(params["fdn_link_filter_hz"]) - 1800.0) < 1e-6
    assert abs(float(params["fdn_link_filter_mix"]) - 0.8) < 1e-6


def test_ir_gen_rejects_invalid_filter_feedback_mode(tmp_path: Path) -> None:
    out_ir = tmp_path / "bad_filter.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.2",
            "--sr",
            "8000",
            "--channels",
            "1",
            "--fdn-link-filter",
            "tilt",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-link-filter must be one of" in result.output


def test_ir_gen_accepts_hyphenated_filter_feedback_alias(tmp_path: Path) -> None:
    out_ir = tmp_path / "fdn_filter_alias.wav"
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(out_ir),
            "--mode",
            "fdn",
            "--length",
            "0.3",
            "--sr",
            "12000",
            "--channels",
            "1",
            "--fdn-link-filter",
            "low-pass",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(out_ir.with_suffix(".wav.ir.meta.json").read_text(encoding="utf-8"))
    assert payload["params"]["fdn_link_filter"] == "lowpass"


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
