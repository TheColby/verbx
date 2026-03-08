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
    assert out_audio.shape[0] > audio.shape[0]
    assert out_audio.shape[1] == audio.shape[1]

    analysis_path = Path(f"{outfile}.analysis.json")
    with analysis_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert "input" in payload
    assert "output" in payload
    assert payload["engine"] == "algo"
    assert payload["effective"]["engine_requested"] == "algo"
    assert payload["effective"]["engine_resolved"] == "algo"
    assert payload["effective"]["device_requested"] == "auto"
    assert payload["effective"]["device_resolved"] in {"cpu", "mps", "cuda"}
    assert isinstance(payload["effective"]["compute_backend"], str)
    assert payload["effective"]["ir_used"] is None
    assert payload["effective"]["tail_padding_seconds"] > 0.0
    assert payload["output_samples"] > payload["input_samples"]


def test_render_allpass_and_comb_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((2048, 2), dtype=np.float32)
    audio[200:280, :] = 0.3
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
            "--allpass-stages",
            "8",
            "--allpass-gain",
            "0.72,0.70,0.68,0.66,0.64,0.62,0.60,0.58",
            "--allpass-delays-ms",
            "4,6,9,13,18,24,31,39",
            "--comb-delays-ms",
            "29,33,37,41,43,47,53,59,67,73",
            "--fdn-lines",
            "10",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["allpass_stages"] == 8
    assert abs(float(config["allpass_gain"]) - 0.72) < 1e-6
    assert len(config["allpass_gains"]) == 8
    assert len(config["allpass_delays_ms"]) == 8
    assert len(config["comb_delays_ms"]) == 10
    assert config["fdn_lines"] == 10


def test_render_tvu_and_dfm_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1536, 1), dtype=np.float32)
    audio[120:220, 0] = 0.4
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
            "--fdn-lines",
            "4",
            "--fdn-matrix",
            "tv-unitary",
            "--fdn-tv-rate-hz",
            "0.2",
            "--fdn-tv-depth",
            "0.5",
            "--fdn-dfm-delays-ms",
            "0.5,0.75,1.0,1.25",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["fdn_matrix"] == "tv_unitary"
    assert abs(float(config["fdn_tv_rate_hz"]) - 0.2) < 1e-6
    assert abs(float(config["fdn_tv_depth"]) - 0.5) < 1e-6
    assert len(config["fdn_dfm_delays_ms"]) == 4


def test_render_rejects_invalid_tvu_combo(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float32)
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
            "--fdn-matrix",
            "tv_unitary",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-matrix tv_unitary requires both" in result.output
    assert "--fdn-tv-depth > 0" in result.output


def test_render_sparse_high_order_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float32)
    audio[20:120, 0] = 0.35
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
            "--rt60",
            "2.0",
            "--fdn-lines",
            "24",
            "--fdn-sparse",
            "--fdn-sparse-degree",
            "4",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["fdn_sparse"] is True
    assert int(config["fdn_sparse_degree"]) == 4


def test_render_rejects_sparse_with_tv_unitary(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float32)
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
            "--fdn-matrix",
            "tv_unitary",
            "--fdn-tv-rate-hz",
            "0.15",
            "--fdn-tv-depth",
            "0.3",
            "--fdn-sparse",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-sparse cannot be combined with --fdn-matrix tv_unitary" in result.output


def test_render_graph_fdn_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1200, 1), dtype=np.float32)
    audio[80:180, 0] = 0.3
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
            "--fdn-matrix",
            "graph",
            "--fdn-graph-topology",
            "star",
            "--fdn-graph-degree",
            "3",
            "--fdn-graph-seed",
            "777",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["fdn_matrix"] == "graph"
    assert config["fdn_graph_topology"] == "star"
    assert int(config["fdn_graph_degree"]) == 3
    assert int(config["fdn_graph_seed"]) == 777
    assert "graph" in str(payload["effective"]["compute_backend"])


def test_render_rejects_graph_options_without_graph_matrix(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float32)
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
            "--fdn-graph-topology",
            "star",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-graph-topology/--fdn-graph-degree are only valid with" in result.output
    assert "--fdn-matrix graph" in result.output


def test_render_rejects_sparse_with_graph_matrix(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float32)
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
            "--fdn-matrix",
            "graph",
            "--fdn-sparse",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-sparse cannot be combined with --fdn-matrix graph" in result.output


def test_render_cascaded_fdn_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1400, 1), dtype=np.float32)
    audio[40:170, 0] = 0.3
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
            "--fdn-lines",
            "8",
            "--fdn-cascade",
            "--fdn-cascade-mix",
            "0.6",
            "--fdn-cascade-delay-scale",
            "0.4",
            "--fdn-cascade-rt60-ratio",
            "0.5",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["fdn_cascade"] is True
    assert abs(float(config["fdn_cascade_mix"]) - 0.6) < 1e-6
    assert abs(float(config["fdn_cascade_delay_scale"]) - 0.4) < 1e-6
    assert abs(float(config["fdn_cascade_rt60_ratio"]) - 0.5) < 1e-6
    assert "cascade" in str(payload["effective"]["compute_backend"])


def test_render_rejects_cascade_with_single_line_fdn(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float32)
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
            "--fdn-lines",
            "1",
            "--fdn-cascade",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-cascade requires at least 2 FDN lines." in result.output


def test_render_multiband_fdn_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1400, 1), dtype=np.float32)
    audio[40:170, 0] = 0.3
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
            "--rt60",
            "2.0",
            "--fdn-rt60-low",
            "22",
            "--fdn-rt60-mid",
            "14",
            "--fdn-rt60-high",
            "7",
            "--fdn-xover-low-hz",
            "240",
            "--fdn-xover-high-hz",
            "3600",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert abs(float(config["fdn_rt60_low"]) - 22.0) < 1e-6
    assert abs(float(config["fdn_rt60_mid"]) - 14.0) < 1e-6
    assert abs(float(config["fdn_rt60_high"]) - 7.0) < 1e-6
    assert abs(float(config["fdn_xover_low_hz"]) - 240.0) < 1e-6
    assert abs(float(config["fdn_xover_high_hz"]) - 3600.0) < 1e-6


def test_render_rejects_partial_multiband_rt60_set(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float32)
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
            "--fdn-rt60-low",
            "20",
            "--fdn-rt60-mid",
            "12",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "provide all three values" in result.output


def test_render_filter_feedback_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1200, 1), dtype=np.float32)
    audio[60:180, 0] = 0.32
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
            "--rt60",
            "2.0",
            "--fdn-link-filter",
            "highpass",
            "--fdn-link-filter-hz",
            "2200",
            "--fdn-link-filter-mix",
            "0.65",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["fdn_link_filter"] == "highpass"
    assert abs(float(config["fdn_link_filter_hz"]) - 2200.0) < 1e-6
    assert abs(float(config["fdn_link_filter_mix"]) - 0.65) < 1e-6


def test_render_rejects_invalid_filter_feedback_mode(tmp_path: Path) -> None:
    audio = np.zeros((640, 1), dtype=np.float32)
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
            "--fdn-link-filter",
            "bandpass",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--fdn-link-filter must be one of" in result.output


def test_render_convolution_route_map_and_trajectory(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "mono_in.wav"
    outfile = tmp_path / "stereo_out.wav"
    irfile = tmp_path / "mono_ir.wav"

    # Keep energy present across the full render so start/end trajectory checks
    # are meaningful.
    x = np.ones((sr // 2, 1), dtype=np.float32)
    ir = np.zeros((256, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--input-layout",
            "mono",
            "--output-layout",
            "stereo",
            "--ir-route-map",
            "broadcast",
            "--conv-route-start",
            "left",
            "--conv-route-end",
            "right",
            "--conv-route-curve",
            "equal-power",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float32")
    assert out_sr == sr
    assert out.shape[1] == 2
    q = max(8, out.shape[0] // 4)
    early_left = float(np.mean(np.abs(out[:q, 0])))
    early_right = float(np.mean(np.abs(out[:q, 1])))
    late_left = float(np.mean(np.abs(out[-q:, 0])))
    late_right = float(np.mean(np.abs(out[-q:, 1])))
    assert early_left > early_right
    assert late_right > late_left


def test_render_rejects_ambiguous_matrix_ir_without_route_hint(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "st_in.wav"
    outfile = tmp_path / "st_out.wav"
    irfile = tmp_path / "matrix_ir.wav"

    x = np.zeros((1024, 2), dtype=np.float32)
    x[0, :] = 1.0
    ir = np.zeros((128, 4), dtype=np.float32)
    ir[0, 0] = 1.0
    ir[0, 3] = 1.0
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    bad = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--no-progress",
        ],
    )
    assert bad.exit_code != 0
    assert "Ambiguous matrix-packed IR layout detected" in bad.output

    good = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--ir-route-map",
            "full",
            "--no-progress",
        ],
    )
    assert good.exit_code == 0, good.stdout


def test_render_allpass_gain_count_mismatch_rejected(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float32)
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
            "--allpass-stages",
            "4",
            "--allpass-gain",
            "0.7,0.65,0.6",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "exactly 4 entries" in result.output


def test_analyze_lufs_mode(tmp_path: Path) -> None:
    audio = np.zeros((4096, 2), dtype=np.float32)
    audio[64:512, :] = 0.2
    infile = tmp_path / "analyze.wav"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(app, ["analyze", str(infile), "--lufs"])
    assert result.exit_code == 0
    assert "integrated_lufs" in result.stdout


def test_analyze_edr_mode(tmp_path: Path) -> None:
    sr = 48_000
    n = sr * 2
    t = np.arange(n, dtype=np.float32) / sr
    signal = (np.exp(-t / 0.9).astype(np.float32) * np.sin(2.0 * np.pi * 200.0 * t)).astype(
        np.float32
    )
    infile = tmp_path / "edr.wav"
    sf.write(str(infile), signal[:, np.newaxis], sr)

    result = runner.invoke(app, ["analyze", str(infile), "--edr"])
    assert result.exit_code == 0
    assert "edr_rt60_median_s" in result.stdout


def test_render_output_subtype_and_peak_normalization_modes(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((1024, 2), dtype=np.float32)
    audio[100:140, :] = 0.25

    infile = tmp_path / "in.wav"
    irfile = tmp_path / "ir.wav"
    sf.write(str(infile), audio, sr)
    sf.write(str(irfile), np.array([[1.0]], dtype=np.float32), sr)

    full_scale_out = tmp_path / "out_fullscale.wav"
    result_full = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(full_scale_out),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--out-subtype",
            "float32",
            "--output-peak-norm",
            "full-scale",
            "--no-progress",
        ],
    )
    assert result_full.exit_code == 0, result_full.stdout

    info = sf.info(str(full_scale_out))
    assert info.subtype == "FLOAT"
    full_audio, _ = sf.read(str(full_scale_out), always_2d=True, dtype="float32")
    full_peak = float(np.max(np.abs(full_audio)))
    assert 0.95 <= full_peak <= 1.001

    input_peak_out = tmp_path / "out_input_peak.wav"
    result_input = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(input_peak_out),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--output-peak-norm",
            "input",
            "--no-progress",
        ],
    )
    assert result_input.exit_code == 0, result_input.stdout

    input_peak_audio, _ = sf.read(str(input_peak_out), always_2d=True, dtype="float32")
    input_peak = float(np.max(np.abs(audio)))
    output_peak = float(np.max(np.abs(input_peak_audio)))
    assert abs(output_peak - input_peak) <= 0.01

    target_peak_out = tmp_path / "out_target_peak.wav"
    result_target = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(target_peak_out),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--output-peak-norm",
            "target",
            "--output-peak-target-dbfs",
            "-6.0",
            "--no-progress",
        ],
    )
    assert result_target.exit_code == 0, result_target.stdout

    target_audio, _ = sf.read(str(target_peak_out), always_2d=True, dtype="float32")
    target_peak = float(np.max(np.abs(target_audio)))
    expected_target = float(10.0 ** (-6.0 / 20.0))
    assert abs(target_peak - expected_target) <= 0.01


def test_render_conv_streaming_mode(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((8192, 2), dtype=np.float32)
    audio[0:64, :] = 0.5
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    irfile = tmp_path / "ir.wav"
    sf.write(str(infile), audio, sr)

    ir = np.zeros((1024, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    ir[100, 0] = 0.2
    sf.write(str(irfile), ir, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    analysis_path = Path(f"{outfile}.analysis.json")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["effective"]["streaming_mode"] is True
    assert payload["output_samples"] >= payload["input_samples"]


def test_render_self_convolve(tmp_path: Path) -> None:
    sr = 24_000
    n = 2048
    t = np.arange(n, dtype=np.float32) / sr
    audio = (0.35 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)[:, np.newaxis]

    infile = tmp_path / "self_in.wav"
    outfile = tmp_path / "self_out.wav"
    sf.write(str(infile), audio, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--self-convolve",
            "--engine",
            "auto",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    out_audio, out_sr = sf.read(str(outfile), always_2d=True, dtype="float32")
    assert out_sr == sr
    assert out_audio.shape[0] > audio.shape[0]
    assert out_audio.shape[1] == 1
    assert np.any(np.abs(out_audio) > 1e-7)

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert payload["effective"]["engine_resolved"] == "conv"
    assert payload["effective"]["self_convolve"] is True
    assert str(payload["effective"]["ir_used"]).endswith("self_in.wav")


def test_render_modulation_multi_source(tmp_path: Path) -> None:
    sr = 24_000
    n = 4096
    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    audio = (0.35 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)[:, np.newaxis]
    side = np.zeros((n, 1), dtype=np.float32)
    side[500:1200, 0] = 0.8

    infile = tmp_path / "mod_in.wav"
    sidechain = tmp_path / "side.wav"
    outfile = tmp_path / "mod_out.wav"
    sf.write(str(infile), audio, sr)
    sf.write(str(sidechain), side, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--rt60",
            "1.2",
            "--mod-target",
            "mix",
            "--mod-min",
            "0.1",
            "--mod-max",
            "0.95",
            "--mod-source",
            "lfo:sine:0.2:1.0*0.8",
            "--mod-source",
            "env:10:150*0.4",
            "--mod-source",
            f"audio-env:{sidechain}:5:120*0.6",
            "--mod-combine",
            "avg",
            "--mod-smooth-ms",
            "25",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    modulation = payload["effective"]["modulation"]
    assert modulation is not None
    assert modulation["target"] == "mix"
    assert len(modulation["sources"]) == 3


def test_render_modulation_multiple_routes(tmp_path: Path) -> None:
    sr = 24_000
    n = 4096
    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    audio = (0.25 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float32)[:, np.newaxis]

    infile = tmp_path / "routes_in.wav"
    outfile = tmp_path / "routes_out.wav"
    sf.write(str(infile), audio, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--rt60",
            "1.5",
            "--mod-route",
            "wet:0.1:0.95:avg:20:lfo:sine:0.12:1.0*1.0",
            "--mod-route",
            "gain-db:-9.0:3.0:sum:15:lfo:triangle:0.04:1.0*0.9",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    modulation = payload["effective"]["modulation"]
    assert modulation is not None
    assert modulation["count"] == 2
    routes = modulation["routes"]
    assert isinstance(routes, list)
    assert routes[0]["target"] == "wet"
    assert routes[1]["target"] == "gain-db"


def test_render_beast_mode_scales_algo_tail(tmp_path: Path) -> None:
    sr = 16_000
    audio = np.zeros((1024, 1), dtype=np.float32)
    audio[0, 0] = 0.7

    infile = tmp_path / "beast_in.wav"
    base_out = tmp_path / "base.wav"
    beast_out = tmp_path / "beast.wav"
    sf.write(str(infile), audio, sr)

    base_result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(base_out),
            "--engine",
            "algo",
            "--rt60",
            "2.0",
            "--repeat",
            "1",
            "--no-progress",
        ],
    )
    assert base_result.exit_code == 0, base_result.stdout

    beast_result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(beast_out),
            "--engine",
            "algo",
            "--rt60",
            "2.0",
            "--repeat",
            "1",
            "--beast-mode",
            "5",
            "--no-progress",
        ],
    )
    assert beast_result.exit_code == 0, beast_result.stdout

    base_audio, _ = sf.read(str(base_out), always_2d=True, dtype="float32")
    beast_audio, _ = sf.read(str(beast_out), always_2d=True, dtype="float32")
    assert beast_audio.shape[0] > base_audio.shape[0]

    payload = json.loads(Path(f"{beast_out}.analysis.json").read_text(encoding="utf-8"))
    assert int(payload["effective"]["beast_mode"]) == 5
    assert float(payload["config"]["rt60"]) > 2.0


def test_render_lucky_mode_creates_multiple_outputs(tmp_path: Path) -> None:
    sr = 24_000
    audio = np.zeros((2048, 1), dtype=np.float32)
    audio[100:220, 0] = 0.5

    infile = tmp_path / "lucky_in.wav"
    outfile = tmp_path / "lucky_out.wav"
    out_dir = tmp_path / "lucky_outputs"
    sf.write(str(infile), audio, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--lucky",
            "3",
            "--lucky-out-dir",
            str(out_dir),
            "--lucky-seed",
            "1234",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    outputs = sorted(out_dir.glob("lucky_out.lucky_*.wav"))
    assert len(outputs) == 3
    for path in outputs:
        analysis_path = Path(f"{path}.analysis.json")
        assert analysis_path.exists()


def test_ir_gen_lucky_mode_creates_multiple_outputs(tmp_path: Path) -> None:
    out_ir = tmp_path / "gen_base.wav"
    out_dir = tmp_path / "gen_lucky"

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
            "12000",
            "--channels",
            "1",
            "--lucky",
            "3",
            "--lucky-out-dir",
            str(out_dir),
            "--lucky-seed",
            "123",
        ],
    )
    assert result.exit_code == 0, result.stdout

    outputs = sorted(out_dir.glob("gen_base.lucky_*.wav"))
    assert len(outputs) == 3
    for path in outputs:
        assert path.exists()
        assert Path(f"{path}.ir.meta.json").exists()


def test_ir_process_lucky_mode_creates_multiple_outputs(tmp_path: Path) -> None:
    sr = 12_000
    in_ir = tmp_path / "in_ir.wav"
    out_ir = tmp_path / "proc_base.wav"
    out_dir = tmp_path / "proc_lucky"
    ir_audio = np.zeros((2048, 1), dtype=np.float32)
    ir_audio[0, 0] = 1.0
    ir_audio[100, 0] = 0.25
    sf.write(str(in_ir), ir_audio, sr)

    result = runner.invoke(
        app,
        [
            "ir",
            "process",
            str(in_ir),
            str(out_ir),
            "--lucky",
            "2",
            "--lucky-out-dir",
            str(out_dir),
            "--lucky-seed",
            "555",
        ],
    )
    assert result.exit_code == 0, result.stdout

    outputs = sorted(out_dir.glob("proc_base.lucky_*.wav"))
    assert len(outputs) == 2
    for path in outputs:
        assert path.exists()
        assert Path(f"{path}.ir.meta.json").exists()


def test_batch_render_parallel_jobs(tmp_path: Path) -> None:
    sr = 16_000
    irfile = tmp_path / "ir.wav"
    ir = np.zeros((256, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    sf.write(str(irfile), ir, sr)

    in1 = tmp_path / "in1.wav"
    in2 = tmp_path / "in2.wav"
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"
    sf.write(str(in1), np.zeros((2048, 1), dtype=np.float32), sr)
    sf.write(str(in2), np.zeros((2048, 1), dtype=np.float32), sr)

    manifest = tmp_path / "manifest.json"
    payload = {
        "version": "0.3",
        "jobs": [
            {
                "infile": str(in1),
                "outfile": str(out1),
                "options": {
                    "engine": "conv",
                    "ir": str(irfile),
                    "normalize_stage": "none",
                    "progress": False,
                },
            },
            {
                "infile": str(in2),
                "outfile": str(out2),
                "options": {
                    "engine": "conv",
                    "ir": str(irfile),
                    "normalize_stage": "none",
                    "progress": False,
                },
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["batch", "render", str(manifest), "--jobs", "2"])
    assert result.exit_code == 0, result.stdout
    assert out1.exists()
    assert out2.exists()


def test_batch_render_lucky_mode_creates_multiple_outputs(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    out_dir = tmp_path / "batch_lucky"
    source = np.zeros((2048, 1), dtype=np.float32)
    source[0, 0] = 0.5
    sf.write(str(infile), source, sr)

    manifest = tmp_path / "manifest_lucky.json"
    payload = {
        "version": "0.4",
        "jobs": [
            {
                "infile": str(infile),
                "outfile": str(outfile),
                "options": {
                    "engine": "algo",
                    "rt60": 1.5,
                    "repeat": 1,
                    "progress": False,
                },
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "batch",
            "render",
            str(manifest),
            "--jobs",
            "1",
            "--lucky",
            "2",
            "--lucky-out-dir",
            str(out_dir),
            "--lucky-seed",
            "101",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert len(sorted(out_dir.glob("out.lucky_*.wav"))) == 2


def test_batch_render_checkpoint_resume_skips_completed(tmp_path: Path) -> None:
    sr = 16_000
    irfile = tmp_path / "ir.wav"
    ir = np.zeros((128, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    sf.write(str(irfile), ir, sr)

    in1 = tmp_path / "in1.wav"
    in2 = tmp_path / "in2.wav"
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"
    sf.write(str(in1), np.zeros((1024, 1), dtype=np.float32), sr)
    sf.write(str(in2), np.zeros((1024, 1), dtype=np.float32), sr)

    manifest = tmp_path / "manifest_resume.json"
    checkpoint = tmp_path / "batch.checkpoint.json"
    payload = {
        "version": "0.5",
        "jobs": [
            {
                "infile": str(in1),
                "outfile": str(out1),
                "options": {"engine": "conv", "ir": str(irfile), "normalize_stage": "none"},
            },
            {
                "infile": str(in2),
                "outfile": str(out2),
                "options": {"engine": "conv", "ir": str(irfile), "normalize_stage": "none"},
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    first = runner.invoke(
        app,
        [
            "batch",
            "render",
            str(manifest),
            "--jobs",
            "1",
            "--checkpoint-file",
            str(checkpoint),
        ],
    )
    assert first.exit_code == 0, first.stdout
    assert checkpoint.exists()

    second = runner.invoke(
        app,
        [
            "batch",
            "render",
            str(manifest),
            "--jobs",
            "1",
            "--checkpoint-file",
            str(checkpoint),
            "--resume",
        ],
    )
    assert second.exit_code == 0, second.stdout
    assert "skipped 2 completed jobs" in second.stdout


def test_render_validation_errors(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((512, 1), dtype=np.float32)
    infile = tmp_path / "in.wav"
    sf.write(str(infile), audio, sr)

    same_path = runner.invoke(
        app,
        ["render", str(infile), str(infile), "--engine", "algo", "--no-progress"],
    )
    assert same_path.exit_code != 0

    missing_ir = runner.invoke(
        app,
        ["render", str(infile), str(tmp_path / "out.wav"), "--engine", "conv", "--no-progress"],
    )
    assert missing_ir.exit_code != 0

    missing_peak_target = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out2.wav"),
            "--output-peak-norm",
            "target",
            "--engine",
            "algo",
            "--rt60",
            "1.0",
            "--no-progress",
        ],
    )
    assert missing_peak_target.exit_code != 0

    ir_file = tmp_path / "ir.wav"
    sf.write(str(ir_file), np.array([[1.0]], dtype=np.float32), sr)

    conflict_ir = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out3.wav"),
            "--self-convolve",
            "--ir",
            str(ir_file),
            "--engine",
            "conv",
            "--no-progress",
        ],
    )
    assert conflict_ir.exit_code != 0

    conflict_ir_gen = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out4.wav"),
            "--self-convolve",
            "--ir-gen",
            "--engine",
            "conv",
            "--no-progress",
        ],
    )
    assert conflict_ir_gen.exit_code != 0

    conflict_algo = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out5.wav"),
            "--self-convolve",
            "--engine",
            "algo",
            "--no-progress",
        ],
    )
    assert conflict_algo.exit_code != 0

    invalid_beast = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out6.wav"),
            "--engine",
            "algo",
            "--beast-mode",
            "101",
            "--no-progress",
        ],
    )
    assert invalid_beast.exit_code != 0

    mod_missing_target = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out7.wav"),
            "--engine",
            "algo",
            "--rt60",
            "1.0",
            "--mod-source",
            "lfo:sine:0.1",
            "--no-progress",
        ],
    )
    assert mod_missing_target.exit_code != 0

    bad_mod_route = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out8.wav"),
            "--engine",
            "algo",
            "--rt60",
            "1.0",
            "--mod-route",
            "wet:0.8:0.1:avg:20:lfo:sine:0.1",
            "--no-progress",
        ],
    )
    assert bad_mod_route.exit_code != 0


def test_ir_gen_validation_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "ir",
            "gen",
            str(tmp_path / "bad.wav"),
            "--rt60",
            "10",
            "--rt60-low",
            "5",
            "--rt60-high",
            "20",
        ],
    )
    assert result.exit_code != 0


def test_render_ambisonics_encode_rotate_decode(tmp_path: Path) -> None:
    sr = 24_000
    n = 4096
    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    left = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    right = (0.2 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    stereo = np.column_stack((left, right)).astype(np.float32)

    infile = tmp_path / "ambi_in.wav"
    outfile = tmp_path / "ambi_out.wav"
    sf.write(str(infile), stereo, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--rt60",
            "1.2",
            "--ambi-order",
            "1",
            "--ambi-encode-from",
            "stereo",
            "--ambi-rotate-yaw-deg",
            "35",
            "--ambi-decode-to",
            "stereo",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float32")
    assert out_sr == sr
    assert out.shape[1] == 2
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert int(payload["config"]["ambi_order"]) == 1
    assert payload["config"]["ambi_decode_to"] == "stereo"


def test_render_rejects_ambi_channel_mismatch_without_encode(tmp_path: Path) -> None:
    sr = 24_000
    stereo = np.zeros((2048, 2), dtype=np.float32)
    infile = tmp_path / "bad_ambi_in.wav"
    outfile = tmp_path / "bad_ambi_out.wav"
    sf.write(str(infile), stereo, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--ambi-order",
            "2",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "Input channels" in result.output
    assert "--ambi-order 2" in result.output


def test_analyze_ambisonic_metrics_mode(tmp_path: Path) -> None:
    sr = 48_000
    n = 4096
    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    foa = np.column_stack(
        (
            0.2 * np.sin(2.0 * np.pi * 120.0 * t),
            0.1 * np.sin(2.0 * np.pi * 150.0 * t),
            0.1 * np.sin(2.0 * np.pi * 80.0 * t),
            0.15 * np.sin(2.0 * np.pi * 200.0 * t),
        )
    ).astype(np.float32)
    infile = tmp_path / "foa.wav"
    sf.write(str(infile), foa, sr)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(infile),
            "--ambi-order",
            "1",
            "--ambi-normalization",
            "sn3d",
            "--channel-order",
            "acn",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "ambi_directionality_stability" in result.stdout


def test_render_automation_file_wet_ramp_and_trace(tmp_path: Path) -> None:
    sr = 16_000
    n = sr // 2
    x = np.ones((n, 1), dtype=np.float32)
    ir = np.zeros((64, 1), dtype=np.float32)
    ir[0, 0] = 1.0

    infile = tmp_path / "auto_in.wav"
    irfile = tmp_path / "auto_ir.wav"
    outfile = tmp_path / "auto_out.wav"
    auto_file = tmp_path / "automation.json"
    trace_file = tmp_path / "automation_trace.csv"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    auto_payload = {
        "mode": "block",
        "block_ms": 10.0,
        "lanes": [
            {
                "target": "wet",
                "type": "breakpoints",
                "interp": "linear",
                "points": [[0.0, 0.0], [0.5, 1.0]],
            }
        ],
    }
    auto_file.write_text(json.dumps(auto_payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--wet",
            "1.0",
            "--dry",
            "0.0",
            "--normalize-stage",
            "none",
            "--automation-file",
            str(auto_file),
            "--automation-trace-out",
            str(trace_file),
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    out, _ = sf.read(str(outfile), always_2d=True, dtype="float32")
    q = max(8, out.shape[0] // 4)
    early = float(np.mean(np.abs(out[:q, 0])))
    late = float(np.mean(np.abs(out[-q:, 0])))
    assert late > early
    assert trace_file.exists()

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "wet" in automation["targets"]


def test_render_rejects_automation_options_without_file(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((512, 1), dtype=np.float32)
    infile = tmp_path / "no_auto_in.wav"
    outfile = tmp_path / "no_auto_out.wav"
    sf.write(str(infile), x, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--automation-mode",
            "sample",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--automation-point" in result.output


def test_render_automation_points_wet_ramp_without_file(tmp_path: Path) -> None:
    sr = 16_000
    n = sr // 2
    x = np.ones((n, 1), dtype=np.float32)
    ir = np.zeros((64, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    infile = tmp_path / "auto_points_in.wav"
    irfile = tmp_path / "auto_points_ir.wav"
    outfile = tmp_path / "auto_points_out.wav"
    trace_file = tmp_path / "auto_points_trace.csv"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--wet",
            "1.0",
            "--dry",
            "0.0",
            "--normalize-stage",
            "none",
            "--automation-point",
            "wet:0.0:0.0:linear",
            "--automation-point",
            "wet:0.5:1.0:linear",
            "--automation-trace-out",
            str(trace_file),
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert trace_file.exists()
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "wet" in automation["targets"]
    assert "wet" in automation.get("post_targets", [])


def test_render_automation_points_drive_algo_engine_targets(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float32)
    x[0, 0] = 1.0
    infile = tmp_path / "algo_auto_in.wav"
    outfile = tmp_path / "algo_auto_out.wav"
    sf.write(str(infile), x, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--normalize-stage",
            "none",
            "--automation-point",
            "rt60:0.0:0.4:linear",
            "--automation-point",
            "rt60:0.25:6.0:linear",
            "--automation-point",
            "damping:0.0:0.65:linear",
            "--automation-point",
            "room-size:0.0:0.8:linear",
            "--automation-point",
            "room-size:0.25:1.6:linear",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    out, _ = sf.read(str(outfile), always_2d=True, dtype="float32")
    assert float(np.max(np.abs(out))) > 1e-6

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "rt60" in automation["targets"]
    assert "damping" in automation["targets"]
    assert "room-size" in automation["targets"]
    assert "rt60" in automation.get("engine_targets", [])
    assert "room-size" in automation.get("engine_targets", [])


def test_render_rejects_engine_automation_targets_for_convolution(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float32)
    x[0, 0] = 1.0
    ir = np.zeros((64, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    infile = tmp_path / "conv_auto_in.wav"
    irfile = tmp_path / "conv_auto_ir.wav"
    outfile = tmp_path / "conv_auto_out.wav"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--automation-point",
            "rt60:0.0:2.0",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "require algorithmic render path" in result.output
