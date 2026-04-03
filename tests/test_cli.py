from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import Result as ClickResult
from pytest import MonkeyPatch
from typer.testing import CliRunner

import verbx.cli as cli_module
from verbx import __version__
from verbx.cli import app
from verbx.core import accel

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_DEFAULT_RENDER_SR = 192_000


def _combined_cli_output(result: ClickResult) -> str:
    parts = [result.output]
    stdout_text = getattr(result, "stdout", "")
    stderr_text = getattr(result, "stderr", "")
    if isinstance(stdout_text, str):
        parts.append(stdout_text)
    if isinstance(stderr_text, str):
        parts.append(stderr_text)
    return _ANSI_ESCAPE_RE.sub("", "\n".join(parts))


def test_cli_boots() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout
    assert "realtime" in result.stdout
    assert "analyze" in result.stdout
    assert "dereverb" in result.stdout
    assert "quickstart" in result.stdout
    assert "doctor" in result.stdout
    assert "presets" in result.stdout
    assert "version" in result.stdout
    assert "suggest" in result.stdout
    assert "ir" in result.stdout
    assert "cache" in result.stdout
    assert "batch" in result.stdout
    assert "immersive" in result.stdout


def test_version_command_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"verbx {__version__}" in result.stdout


def test_quickstart_command_prints_copyable_workflows() -> None:
    result = runner.invoke(app, ["quickstart"])
    assert result.exit_code == 0
    text = _combined_cli_output(result)
    assert "verbx Quickstart" in text
    assert "verbx render ../in.wav out.wav" in text
    assert "verbx analyze in.wav" in text


def test_doctor_command_prints_runtime_diagnostics(tmp_path: Path) -> None:
    json_out = tmp_path / "doctor.json"
    result = runner.invoke(app, ["doctor", "--json-out", str(json_out)])
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "verbx Doctor" in text
    assert "python_version" in text
    assert "device_auto" in text

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["verbx_version"] == __version__
    assert "engine_auto_resolution" in payload
    assert "dependencies" in payload


def test_quickstart_verify_emits_readiness_report(tmp_path: Path) -> None:
    json_out = tmp_path / "quickstart_verify.json"
    result = runner.invoke(app, ["quickstart", "--verify", "--json-out", str(json_out)])
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "verbx Quickstart Verify" in text
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert isinstance(payload.get("checks"), list)
    assert "ready" in payload


def test_quickstart_smoke_test_writes_artifacts(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    json_out = tmp_path / "quickstart_smoke.json"
    result = runner.invoke(
        app,
        [
            "quickstart",
            "--smoke-test",
            "--strict",
            "--smoke-out-dir",
            str(smoke_dir),
            "--json-out",
            str(json_out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "verbx Quickstart Smoke Test" in text
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    smoke = payload.get("smoke_test", {})
    assert bool(smoke.get("ok", False))
    assert Path(str(smoke.get("outfile", ""))).exists()


def test_doctor_strict_fails_when_checks_fail(monkeypatch: MonkeyPatch) -> None:
    report: dict[str, object] = {
        "verbx_version": __version__,
        "python_version": "3.10.0",
        "platform": "test-platform",
        "machine": "test-machine",
        "cpu_count": 8,
        "apple_silicon": False,
        "cuda_available": False,
        "device_auto": "cpu",
        "engine_auto_resolution": {
            "algo": {"engine_device": "cpu", "platform_device": "cpu"},
            "conv": {"engine_device": "cpu", "platform_device": "cpu"},
        },
        "dependencies": {"cupy": None},
    }
    report["checks"] = [
        {
            "id": "python_min",
            "name": "Python >= 3.11",
            "ok": False,
            "value": "3.10.0",
            "hint": "Use Python 3.11 or newer.",
        }
    ]
    report["issues"] = list(report["checks"])
    report["checks_total"] = 1
    report["failed_checks"] = 1
    report["ready"] = False
    report["status"] = "warn"
    report["recommendations"] = ["Use Python 3.11 or newer."]
    monkeypatch.setattr(cli_module, "_collect_runtime_diagnostics", lambda: report)
    result = runner.invoke(app, ["doctor", "--strict"])
    assert result.exit_code == 2


def test_doctor_strict_fails_when_smoke_test_fails(monkeypatch: MonkeyPatch) -> None:
    def _fake_smoke_report(out_dir: Path) -> dict[str, object]:
        _ = out_dir
        return {
            "ok": False,
            "engine": "algo",
            "sample_rate": 24_000,
            "input_frames": 1000,
            "output_frames": 0,
            "error": "simulated failure",
        }

    monkeypatch.setattr(
        cli_module,
        "_run_render_smoke_test",
        _fake_smoke_report,
    )
    result = runner.invoke(app, ["doctor", "--render-smoke-test", "--strict"])
    assert result.exit_code == 2


def test_presets_show_displays_resolved_values() -> None:
    result = runner.invoke(app, ["presets", "--show", "cathedral-extreme"])
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "Preset:" in text
    assert "cathedral_extreme" in text
    assert "rt60" in text
    assert "90.0" in text


def test_presets_include_perceptual_regression_corpus_variants() -> None:
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "perceptual_small_room_regression" in text
    assert "perceptual_mid_room_regression" in text
    assert "perceptual_long_hall_regression" in text
    assert "perceptual_extreme_tail_regression" in text

    detail = runner.invoke(app, ["presets", "--show", "perceptual-long-hall-regression"])
    assert detail.exit_code == 0, detail.stdout
    detail_text = _combined_cli_output(detail)
    assert "fdn_tonal_correction_strength" in detail_text
    assert "room_size_macro" in detail_text


def test_render_dry_run_validates_without_writing_audio(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[10:40, 0] = 0.4
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
            "--dry-run",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "Render Dry-Run Plan" in text
    assert "audio_write" in text
    assert "skipped" in text
    assert "estimated_output_size_mb" in text
    assert not outfile.exists()
    assert not Path(f"{outfile}.analysis.json").exists()


def test_render_dry_run_accepts_extended_rt60_upper_bound(tmp_path: Path) -> None:
    audio = np.zeros((256, 1), dtype=np.float64)
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
            "3600",
            "--dry-run",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "Render Dry-Run Plan" in text
    assert not outfile.exists()
    assert not Path(f"{outfile}.analysis.json").exists()


def test_render_dry_run_accepts_w64_extension(tmp_path: Path) -> None:
    audio = np.zeros((256, 1), dtype=np.float64)
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.w64"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--dry-run",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert not outfile.exists()


def test_render_auto_fit_profile_applies_when_not_overridden(tmp_path: Path) -> None:
    audio = np.zeros((4096, 1), dtype=np.float64)
    audio[64:192, 0] = 0.4
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, 48_000, subtype="DOUBLE")

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--auto-fit",
            "speech",
            "--quiet",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert abs(float(config["rt60"]) - 1.2) < 1e-6
    assert abs(float(config["fdn_lines"]) - 8.0) < 1e-6
    assert abs(float(config["pre_delay_ms"]) - 14.0) < 1e-6


def test_render_algo_stream_proxy_path_reports_proxy_backend(tmp_path: Path) -> None:
    sr = 12_000
    audio = np.zeros((2048, 1), dtype=np.float64)
    audio[0, 0] = 1.0
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--algo-stream",
            "--mod-depth-ms",
            "0",
            "--mod-rate-hz",
            "0",
            "--normalize-stage",
            "none",
            "--target-sr",
            str(sr),
            "--quiet",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    effective = payload["effective"]
    assert effective["engine_resolved"] == "algo_proxy_stream"
    assert bool(effective["streaming_mode"])


def test_render_matrix_morph_and_er_geometry_complete(tmp_path: Path) -> None:
    sr = 16_000
    audio = np.zeros((4096, 1), dtype=np.float64)
    audio[80:220, 0] = 0.6
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--fdn-matrix",
            "hadamard",
            "--fdn-matrix-morph-to",
            "householder",
            "--fdn-matrix-morph-seconds",
            "0.2",
            "--er-geometry",
            "--er-room-dims-m",
            "9,7,3",
            "--er-source-pos-m",
            "1.5,2.0,1.4",
            "--er-listener-pos-m",
            "4.0,3.0,1.4",
            "--quiet",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert outfile.exists()
    out_audio, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
    assert out_audio.shape[1] == 1


def test_render_long_tail_regression_rt60_over_120_seconds(tmp_path: Path) -> None:
    sr = 2_000
    audio = np.zeros((1_000, 1), dtype=np.float64)
    audio[0, 0] = 0.8
    infile = tmp_path / "long_tail_in.wav"
    outfile = tmp_path / "long_tail_out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--rt60",
            "130",
            "--fdn-lines",
            "4",
            "--allpass-stages",
            "2",
            "--target-sr",
            str(sr),
            "--tail-stop-threshold-db",
            "-240",
            "--quiet",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert outfile.exists()
    y, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == sr
    # Tail completion now trims padding aggressively; assert long-tail behavior
    # without requiring raw RT60 seconds of explicit output duration.
    assert y.shape[0] >= int(60.0 * sr)
    assert np.max(np.abs(y[: min(y.shape[0], 512), :])) > 0.0
    assert np.max(np.abs(y[-max(1, sr // 100) :, :])) == 0.0


def test_render_preset_applies_defaults_and_respects_cli_override(tmp_path: Path) -> None:
    audio = np.zeros((1600, 1), dtype=np.float64)
    audio[100:200, 0] = 0.5
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
            "--preset",
            "cathedral-extreme",
            "--rt60",
            "12.0",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert abs(float(config["rt60"]) - 12.0) < 1e-6
    assert abs(float(config["wet"]) - 0.9) < 1e-6
    assert abs(float(config["dry"]) - 0.1) < 1e-6


def test_render_invalid_fdn_matrix_includes_suggestion(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float64)
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
            "hadmard",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    text = _combined_cli_output(result)
    assert "--fdn-matrix must be one of:" in text
    assert "hadamard" in text
    assert "hadamard" in text


def test_render_creates_output_and_analysis(tmp_path: Path) -> None:
    audio = np.zeros((2048, 2), dtype=np.float64)
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

    out_audio, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
    assert out_audio.shape[0] > audio.shape[0]
    assert out_audio.shape[1] == audio.shape[1]
    tail_zero_window = min(64, out_audio.shape[0])
    assert np.all(out_audio[-tail_zero_window:, :] == 0.0)

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
    assert payload["effective"]["perceptual_macros"] is None
    assert payload["output_samples"] > payload["input_samples"]


def test_render_prints_output_feature_table_by_default(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[10:30, 0] = 0.4
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
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Render Summary" in result.stdout
    assert "Output Audio Features and Statistics" in result.stdout


def test_render_writes_repro_bundle(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[10:30, 0] = 0.4
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
            "--repro-bundle",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    repro_path = Path(f"{outfile}.repro.json")
    assert repro_path.exists()
    payload = json.loads(repro_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "render-repro-bundle-v1"
    assert payload["input"]["path"] == str(infile.resolve())
    assert payload["output"]["path"] == str(outfile.resolve())
    assert isinstance(payload.get("run_signature"), str)
    assert len(str(payload.get("run_signature"))) == 64


def test_render_quiet_or_low_verbosity_suppresses_output_feature_table(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[20:40, 0] = 0.5
    infile = tmp_path / "in.wav"
    out_quiet = tmp_path / "out_quiet.wav"
    out_low = tmp_path / "out_low.wav"
    sf.write(str(infile), audio, 48_000)

    quiet_result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(out_quiet),
            "--engine",
            "algo",
            "--quiet",
            "--no-progress",
        ],
    )
    assert quiet_result.exit_code == 0, quiet_result.stdout
    assert "Output Audio Features and Statistics" not in quiet_result.stdout

    low_result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(out_low),
            "--engine",
            "algo",
            "--verbosity",
            "0",
            "--no-progress",
        ],
    )
    assert low_result.exit_code == 0, low_result.stdout
    assert "Render Summary" in low_result.stdout
    assert "Output Audio Features and Statistics" not in low_result.stdout


def test_render_algo_auto_reports_engine_specific_device(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(accel, "is_apple_silicon", lambda: False)

    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[80:120, 0] = 0.5
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
            "--device",
            "auto",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert payload["effective"]["engine_resolved"] == "algo"
    assert payload["effective"]["device_requested"] == "auto"
    assert payload["effective"]["device_platform_resolved"] == "cuda"
    assert payload["effective"]["device_resolved"] == "cpu"


def test_render_conv_auto_prefers_cuda_when_available(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(accel, "is_apple_silicon", lambda: True)

    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[80:120, 0] = 0.5
    infile = tmp_path / "in.wav"
    irfile = tmp_path / "ir.wav"
    outfile = tmp_path / "out.wav"
    ir = np.zeros((256, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    sf.write(str(infile), audio, 48_000)
    sf.write(str(irfile), ir, 48_000)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "auto",
            "--ir",
            str(irfile),
            "--device",
            "auto",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert payload["effective"]["engine_resolved"] == "conv"
    assert payload["effective"]["device_requested"] == "auto"
    assert payload["effective"]["device_platform_resolved"] == "cuda"
    assert payload["effective"]["device_resolved"] == "cuda"


def test_render_allpass_and_comb_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((2048, 2), dtype=np.float64)
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
    audio = np.zeros((1536, 1), dtype=np.float64)
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
    audio = np.zeros((512, 1), dtype=np.float64)
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
    text = _combined_cli_output(result)
    assert "--fdn-matrix tv_unitary requires both" in text
    assert "--fdn-tv-depth > 0" in text


def test_render_sparse_high_order_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
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
    audio = np.zeros((1024, 1), dtype=np.float64)
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
    assert "--fdn-sparse cannot be combined with --fdn-matrix tv_unitary" in _combined_cli_output(
        result
    )


def test_render_graph_fdn_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1200, 1), dtype=np.float64)
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
    audio = np.zeros((512, 1), dtype=np.float64)
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
    text = _combined_cli_output(result)
    assert "--fdn-graph-topology/--fdn-graph-degree are only valid with" in text
    assert "--fdn-matrix graph" in text


def test_render_rejects_sparse_with_graph_matrix(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float64)
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
    assert "--fdn-sparse cannot be combined with --fdn-matrix graph" in _combined_cli_output(
        result
    )


def test_render_sdn_spatial_and_nonlinear_switches_are_applied(tmp_path: Path) -> None:
    infile = tmp_path / "in_sdn.wav"
    outfile = tmp_path / "out_sdn.wav"
    sr = 24_000
    x = np.zeros((sr, 1), dtype=np.float64)
    x[200:800, 0] = 0.6
    sf.write(str(infile), x, sr)

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
            "--fdn-matrix",
            "sdn_hybrid",
            "--fdn-spatial-coupling-mode",
            "front_rear",
            "--fdn-spatial-coupling-strength",
            "0.2",
            "--fdn-nonlinearity",
            "tanh",
            "--fdn-nonlinearity-amount",
            "0.15",
            "--fdn-nonlinearity-drive",
            "2.2",
            "--output-layout",
            "7.1.2",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["fdn_matrix"] == "sdn_hybrid"
    assert config["fdn_spatial_coupling_mode"] == "front_rear"
    assert abs(float(config["fdn_spatial_coupling_strength"]) - 0.2) < 1e-6
    assert config["fdn_nonlinearity"] == "tanh"
    assert abs(float(config["fdn_nonlinearity_amount"]) - 0.15) < 1e-6
    assert abs(float(config["fdn_nonlinearity_drive"]) - 2.2) < 1e-6
    assert "nonlinear" in str(payload["effective"]["compute_backend"])


def test_render_cascaded_fdn_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1400, 1), dtype=np.float64)
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
    audio = np.zeros((1024, 1), dtype=np.float64)
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
    assert "--fdn-cascade requires at least 2 FDN lines." in _combined_cli_output(result)


def test_render_rejects_unsafe_loop_gain_without_unsafe_mode(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
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
            "--unsafe-loop-gain",
            "1.05",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--unsafe-loop-gain requires --unsafe-self-oscillate." in _combined_cli_output(result)


def test_render_accepts_unsafe_self_oscillation_settings_in_dry_run(tmp_path: Path) -> None:
    audio = np.zeros((1400, 1), dtype=np.float64)
    audio[30:140, 0] = 0.4
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
            "0.35",
            "--shimmer",
            "--shimmer-semitones",
            "0",
            "--shimmer-feedback",
            "1.05",
            "--unsafe-self-oscillate",
            "--unsafe-loop-gain",
            "1.04",
            "--dry-run",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "Render Dry-Run Plan" in text


def test_render_multiband_fdn_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1400, 1), dtype=np.float64)
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
            "--fdn-tonal-correction-strength",
            "0.7",
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
    assert abs(float(config["fdn_tonal_correction_strength"]) - 0.7) < 1e-6
    assert abs(float(config["fdn_xover_low_hz"]) - 240.0) < 1e-6
    assert abs(float(config["fdn_xover_high_hz"]) - 3600.0) < 1e-6
    assert "tonalcorr" in str(payload["effective"]["compute_backend"])


def test_render_rejects_partial_multiband_rt60_set(tmp_path: Path) -> None:
    audio = np.zeros((1024, 1), dtype=np.float64)
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
    assert "provide all three values" in _combined_cli_output(result)


def test_render_filter_feedback_switches_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1200, 1), dtype=np.float64)
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
    audio = np.zeros((640, 1), dtype=np.float64)
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
    assert "--fdn-link-filter must be one of" in _combined_cli_output(result)


def test_render_accepts_hyphenated_filter_feedback_alias(tmp_path: Path) -> None:
    audio = np.zeros((1200, 1), dtype=np.float64)
    audio[60:180, 0] = 0.32
    infile = tmp_path / "alias_filter_in.wav"
    outfile = tmp_path / "alias_filter_out.wav"
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
            "low-pass",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert payload["config"]["fdn_link_filter"] == "lowpass"


def test_render_track_c_perceptual_fdn_controls_are_applied(tmp_path: Path) -> None:
    audio = np.zeros((1400, 1), dtype=np.float64)
    audio[45:180, 0] = 0.28
    infile = tmp_path / "track_c_in.wav"
    outfile = tmp_path / "track_c_out.wav"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--fdn-rt60-tilt",
            "0.45",
            "--room-size-macro",
            "0.35",
            "--clarity-macro",
            "-0.20",
            "--warmth-macro",
            "0.55",
            "--envelopment-macro",
            "0.60",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    config = payload["config"]
    assert abs(float(config["fdn_rt60_tilt"]) - 0.45) < 1e-6
    assert abs(float(config["room_size_macro"]) - 0.35) < 1e-6
    assert abs(float(config["clarity_macro"]) + 0.20) < 1e-6
    assert abs(float(config["warmth_macro"]) - 0.55) < 1e-6
    assert abs(float(config["envelopment_macro"]) - 0.60) < 1e-6
    macro_report = payload["effective"]["perceptual_macros"]
    assert isinstance(macro_report, dict)
    assert abs(float(macro_report["input"]["room_size_macro"]) - 0.35) < 1e-6
    assert "resolved" in macro_report
    assert "delta_from_requested" in macro_report


def test_render_convolution_route_map_and_trajectory(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "mono_in.wav"
    outfile = tmp_path / "stereo_out.wav"
    irfile = tmp_path / "mono_ir.wav"

    # Keep energy present across the full render so start/end trajectory checks
    # are meaningful.
    x = np.ones((sr // 2, 1), dtype=np.float64)
    ir = np.zeros((256, 1), dtype=np.float64)
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

    out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
    assert out.shape[1] == 2
    q = max(8, out.shape[0] // 4)
    early_left = float(np.mean(np.abs(out[:q, 0])))
    early_right = float(np.mean(np.abs(out[:q, 1])))
    late_left = float(np.mean(np.abs(out[-q:, 0])))
    late_right = float(np.mean(np.abs(out[-q:, 1])))
    assert early_left > early_right
    assert late_right > late_left


def test_render_convolution_accepts_extended_output_layout_token(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "mono_in.wav"
    irfile = tmp_path / "mono_ir.wav"
    outfile = tmp_path / "layout_out.wav"

    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir = np.zeros((128, 1), dtype=np.float64)
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
            "7.2.4",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
    assert out.shape[1] == 13


def test_render_convolution_large_layout_requires_explicit_route_map(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "mono_in_large.wav"
    irfile = tmp_path / "mono_ir_large.wav"
    outfile = tmp_path / "layout_out_large.wav"

    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir = np.zeros((128, 1), dtype=np.float64)
    ir[0, 0] = 1.0
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
            "--input-layout",
            "mono",
            "--output-layout",
            "16.0",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert bad.exit_code != 0
    assert "Auto route-map is ambiguous for large output layouts" in _combined_cli_output(bad)
    assert "--ir-route-map explicitly" in _combined_cli_output(bad)

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
            "--input-layout",
            "mono",
            "--output-layout",
            "16.0",
            "--ir-route-map",
            "broadcast",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert good.exit_code == 0, good.stdout
    out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
    assert out.shape[1] == 16


def test_render_convolution_ir_blend_generates_composite_ir_runtime(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    base_ir = tmp_path / "base_ir.wav"
    blend_ir = tmp_path / "blend_ir.wav"

    x = np.zeros((sr // 3, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir_a = np.zeros((1024, 1), dtype=np.float64)
    ir_b = np.zeros((1024, 1), dtype=np.float64)
    ir_a[0, 0] = 1.0
    ir_a[120, 0] = 0.4
    ir_b[0, 0] = 1.0
    ir_b[600, 0] = 0.28
    sf.write(str(infile), x, sr)
    sf.write(str(base_ir), ir_a, sr)
    sf.write(str(blend_ir), ir_b, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(base_ir),
            "--ir-blend",
            str(blend_ir),
            "--ir-blend-mix",
            "0.65",
            "--ir-blend-mode",
            "spectral",
            "--ir-blend-phase-coherence",
            "0.85",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    ir_runtime = payload.get("ir_runtime")
    assert isinstance(ir_runtime, dict)
    assert ir_runtime["mode"] == "ir-blend"
    assert str(ir_runtime["ir_path"]).endswith(".wav")
    assert str(payload["effective"]["ir_used"]).endswith(".wav")
    blend_meta = ir_runtime.get("meta", {})
    assert blend_meta.get("mode") == "ir-blend"
    assert len(blend_meta.get("sources", [])) == 2


def test_render_rejects_ir_blend_without_base_ir_source(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    blend_ir = tmp_path / "blend_ir.wav"
    x = np.zeros((512, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    sf.write(str(infile), x, sr)
    sf.write(str(blend_ir), ir, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "auto",
            "--ir-blend",
            str(blend_ir),
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "--ir-blend requires base IR source" in _combined_cli_output(result)


def test_render_ir_blend_strict_policy_rejects_mismatch(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "blend_strict_in.wav"
    base_ir = tmp_path / "blend_strict_base.wav"
    blend_ir = tmp_path / "blend_strict_extra.wav"
    outfile = tmp_path / "blend_strict_out.wav"

    x = np.zeros((sr, 1), dtype=np.float64)
    x[120:240, 0] = 0.8
    base = np.zeros((512, 1), dtype=np.float64)
    base[0, 0] = 1.0
    extra = np.zeros((640, 2), dtype=np.float64)
    extra[0, :] = 1.0
    sf.write(str(infile), x, sr)
    sf.write(str(base_ir), base, sr)
    sf.write(str(blend_ir), extra, 22_050)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(base_ir),
            "--ir-blend",
            str(blend_ir),
            "--ir-blend-mismatch-policy",
            "strict",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    text = _combined_cli_output(result).lower()
    assert "strict policy" in text
    assert "sample-rate mismatch" in text


def test_render_rejects_ambiguous_matrix_ir_without_route_hint(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "st_in.wav"
    outfile = tmp_path / "st_out.wav"
    irfile = tmp_path / "matrix_ir.wav"

    x = np.zeros((1024, 2), dtype=np.float64)
    x[0, :] = 1.0
    ir = np.zeros((128, 4), dtype=np.float64)
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
    text = _combined_cli_output(bad)
    assert "Ambiguous matrix-packed IR layout detected" in text
    assert "Input channels=2," in text
    assert "IR channels=4, resolved output channels=2." in text
    assert "--ir-route-map full" in text

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


def test_render_layout_mismatch_error_reports_channel_math(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "layout_in.wav"
    outfile = tmp_path / "layout_out.wav"
    irfile = tmp_path / "layout_ir.wav"

    x = np.zeros((512, 2), dtype=np.float64)
    x[80:140, 0] = 0.5
    x[80:140, 1] = -0.4
    ir = np.zeros((96, 4), dtype=np.float64)
    ir[0, 0] = 1.0
    ir[0, 3] = 1.0
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
            "--output-layout",
            "7.1",
            "--ir-route-map",
            "full",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    text = _combined_cli_output(result)
    assert "Output layout '7.1' expects 8 channels" in text
    assert "input_channels=2, ir_channels=4" in text
    assert "matrix-packed IR channels = input_channels *" in text
    assert "output_channels." in text


def test_render_allpass_gain_count_mismatch_rejected(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float64)
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
    assert "exactly 4 entries" in _combined_cli_output(result)


def test_analyze_lufs_mode(tmp_path: Path) -> None:
    audio = np.zeros((4096, 2), dtype=np.float64)
    audio[64:512, :] = 0.2
    infile = tmp_path / "analyze.wav"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(app, ["analyze", str(infile), "--lufs"])
    assert result.exit_code == 0
    assert "integrated_lufs" in result.stdout


def test_analyze_edr_mode(tmp_path: Path) -> None:
    sr = 48_000
    n = sr * 2
    t = np.arange(n, dtype=np.float64) / sr
    signal = (np.exp(-t / 0.9).astype(np.float64) * np.sin(2.0 * np.pi * 200.0 * t)).astype(
        np.float64
    )
    infile = tmp_path / "edr.wav"
    sf.write(str(infile), signal[:, np.newaxis], sr)

    result = runner.invoke(app, ["analyze", str(infile), "--edr"])
    assert result.exit_code == 0
    assert "edr_rt60_median_s" in result.stdout


def test_dereverb_command_writes_output_and_json(tmp_path: Path) -> None:
    sr = 16_000
    n = sr
    x = np.zeros((n, 1), dtype=np.float64)
    x[64, 0] = 1.0
    ir = np.exp(-np.arange(int(0.5 * sr), dtype=np.float64) / (0.16 * sr))
    y = np.convolve(x[:, 0], ir, mode="full")[:n]
    infile = tmp_path / "wet.wav"
    outfile = tmp_path / "dryish.wav"
    json_out = tmp_path / "dereverb.json"
    sf.write(str(infile), y[:, None], sr, subtype="DOUBLE")

    result = runner.invoke(
        app,
        [
            "dereverb",
            str(infile),
            str(outfile),
            "--mode",
            "wiener",
            "--strength",
            "0.9",
            "--window-ms",
            "32",
            "--hop-ms",
            "8",
            "--tail-ms",
            "180",
            "--json-out",
            str(json_out),
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert outfile.exists()
    out_audio, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == sr
    assert out_audio.shape == (n, 1)
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "dereverb-report-v1"
    assert payload["sample_rate"] == sr
    assert payload["channels"] == 1
    assert "rms_delta_db" in payload["metrics"]


def test_dereverb_rejects_hop_ms_not_smaller_than_window(tmp_path: Path) -> None:
    audio = np.zeros((512, 1), dtype=np.float64)
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, 48_000, subtype="DOUBLE")

    result = runner.invoke(
        app,
        [
            "dereverb",
            str(infile),
            str(outfile),
            "--window-ms",
            "10",
            "--hop-ms",
            "10",
        ],
    )
    assert result.exit_code != 0
    text = _combined_cli_output(result)
    assert "--hop-ms must be smaller than --window-ms." in text


def test_render_output_subtype_and_peak_normalization_modes(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((1024, 2), dtype=np.float64)
    audio[100:140, :] = 0.25

    infile = tmp_path / "in.wav"
    irfile = tmp_path / "ir.wav"
    sf.write(str(infile), audio, sr)
    sf.write(str(irfile), np.array([[1.0]], dtype=np.float64), sr)

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
    full_audio, _ = sf.read(str(full_scale_out), always_2d=True, dtype="float64")
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

    input_peak_audio, _ = sf.read(str(input_peak_out), always_2d=True, dtype="float64")
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

    target_audio, _ = sf.read(str(target_peak_out), always_2d=True, dtype="float64")
    target_peak = float(np.max(np.abs(target_audio)))
    expected_target = float(10.0 ** (-6.0 / 20.0))
    assert abs(target_peak - expected_target) <= 0.01


def test_render_defaults_to_hd_output_definition(tmp_path: Path) -> None:
    sr_in = 48_000
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[16, 0] = 0.6

    infile = tmp_path / "in_hd.wav"
    irfile = tmp_path / "ir_hd.wav"
    outfile = tmp_path / "out_hd.wav"
    sf.write(str(infile), audio, sr_in)
    sf.write(str(irfile), np.array([[1.0]], dtype=np.float64), sr_in)

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

    info = sf.info(str(outfile))
    assert info.samplerate == _DEFAULT_RENDER_SR
    assert info.subtype == "FLOAT"


def test_render_quality_preset_sd_and_explicit_override_precedence(tmp_path: Path) -> None:
    sr_in = 48_000
    audio = np.zeros((512, 1), dtype=np.float64)
    audio[0, 0] = 0.7

    infile = tmp_path / "in_quality.wav"
    irfile = tmp_path / "ir_quality.wav"
    sd_out = tmp_path / "out_sd.wav"
    override_out = tmp_path / "out_override.wav"
    sf.write(str(infile), audio, sr_in)
    sf.write(str(irfile), np.array([[1.0]], dtype=np.float64), sr_in)

    sd_result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(sd_out),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--quality-preset",
            "sd",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert sd_result.exit_code == 0, sd_result.stdout
    sd_info = sf.info(str(sd_out))
    assert sd_info.samplerate == 44_100
    assert sd_info.subtype == "PCM_16"

    override_result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(override_out),
            "--engine",
            "conv",
            "--ir",
            str(irfile),
            "--quality-preset",
            "sd",
            "--target-sr",
            "96000",
            "--out-subtype",
            "float64",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert override_result.exit_code == 0, override_result.stdout
    override_info = sf.info(str(override_out))
    assert override_info.samplerate == 96_000
    assert override_info.subtype == "DOUBLE"


def test_render_target_sample_rate_conversion_and_float32_output(tmp_path: Path) -> None:
    sr_in = 48_000
    sr_out = 192_000
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[0, 0] = 0.5

    infile = tmp_path / "in_sr.wav"
    irfile = tmp_path / "ir_sr.wav"
    outfile = tmp_path / "out_sr.wav"
    sf.write(str(infile), audio, sr_in)
    sf.write(str(irfile), np.array([[1.0]], dtype=np.float64), sr_in)

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
            "--target-sr",
            str(sr_out),
            "--out-subtype",
            "float32",
            "--normalize-stage",
            "none",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    info = sf.info(str(outfile))
    assert info.samplerate == sr_out
    assert info.subtype == "FLOAT"

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert payload["sample_rate"] == sr_out
    assert payload["effective"]["streaming_mode"] is False
    assert payload["effective"]["sample_rate_action"] == f"resample:{sr_in}->{sr_out}"


def test_render_conv_streaming_mode(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((8192, 2), dtype=np.float64)
    audio[0:64, :] = 0.5
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    irfile = tmp_path / "ir.wav"
    sf.write(str(infile), audio, sr)

    ir = np.zeros((1024, 1), dtype=np.float64)
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
            "--target-sr",
            str(sr),
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    stream_tmp = outfile.with_name(f"{outfile.stem}.stream_tmp{outfile.suffix}")
    assert not stream_tmp.exists()

    analysis_path = Path(f"{outfile}.analysis.json")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["effective"]["streaming_mode"] is True
    assert payload["output_samples"] >= payload["input_samples"]
    out_audio, _ = sf.read(str(outfile), always_2d=True, dtype="float64")
    tail_zero_window = min(64, out_audio.shape[0])
    assert np.all(out_audio[-tail_zero_window:, :] == 0.0)


def test_render_self_convolve(tmp_path: Path) -> None:
    sr = 24_000
    n = 2048
    t = np.arange(n, dtype=np.float64) / sr
    audio = (0.35 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float64)[:, np.newaxis]

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

    out_audio, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
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
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    audio = (0.35 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float64)[:, np.newaxis]
    side = np.zeros((n, 1), dtype=np.float64)
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
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    audio = (0.25 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float64)[:, np.newaxis]

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
    audio = np.zeros((1024, 1), dtype=np.float64)
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

    base_audio, _ = sf.read(str(base_out), always_2d=True, dtype="float64")
    beast_audio, _ = sf.read(str(beast_out), always_2d=True, dtype="float64")
    assert beast_audio.shape[0] > base_audio.shape[0]

    payload = json.loads(Path(f"{beast_out}.analysis.json").read_text(encoding="utf-8"))
    assert int(payload["effective"]["beast_mode"]) == 5
    assert float(payload["config"]["rt60"]) > 2.0


def test_render_lucky_mode_creates_multiple_outputs(tmp_path: Path) -> None:
    sr = 24_000
    audio = np.zeros((2048, 1), dtype=np.float64)
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
    ir_audio = np.zeros((2048, 1), dtype=np.float64)
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


def test_ir_morph_sweep_generates_qa_bundle(tmp_path: Path) -> None:
    sr = 16_000
    ir_a = tmp_path / "a.wav"
    ir_b = tmp_path / "b.wav"
    out_dir = tmp_path / "morph_sweep"
    a = np.zeros((1024, 1), dtype=np.float64)
    b = np.zeros((1024, 1), dtype=np.float64)
    a[0, 0] = 1.0
    a[80, 0] = 0.4
    b[0, 0] = 1.0
    b[220, 0] = 0.25
    sf.write(str(ir_a), a, sr)
    sf.write(str(ir_b), b, sr)

    result = runner.invoke(
        app,
        [
            "ir",
            "morph-sweep",
            str(ir_a),
            str(ir_b),
            str(out_dir),
            "--alpha-start",
            "0.1",
            "--alpha-end",
            "0.9",
            "--alpha-steps",
            "3",
            "--workers",
            "1",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.stdout

    outputs = sorted(out_dir.glob("morph_*.wav"))
    assert len(outputs) == 3
    qa_csv = out_dir / "morph_sweep_metrics.csv"
    qa_json = out_dir / "morph_sweep_summary.json"
    assert qa_csv.exists()
    assert qa_json.exists()
    rows = [line for line in qa_csv.read_text(encoding="utf-8").splitlines() if line.strip() != ""]
    assert len(rows) == 4  # header + 3 rows
    summary = json.loads(qa_json.read_text(encoding="utf-8"))
    assert int(summary["planned"]) == 3
    assert int(summary["success"]) == 3
    assert int(summary["failed"]) == 0


def test_ir_morph_sweep_resume_skips_completed_outputs(tmp_path: Path) -> None:
    sr = 16_000
    ir_a = tmp_path / "a.wav"
    ir_b = tmp_path / "b.wav"
    out_dir = tmp_path / "resume_sweep"
    checkpoint = tmp_path / "resume_checkpoint.json"
    a = np.zeros((1024, 1), dtype=np.float64)
    b = np.zeros((1024, 1), dtype=np.float64)
    a[0, 0] = 1.0
    a[96, 0] = 0.35
    b[0, 0] = 1.0
    b[280, 0] = 0.2
    sf.write(str(ir_a), a, sr)
    sf.write(str(ir_b), b, sr)

    first = runner.invoke(
        app,
        [
            "ir",
            "morph-sweep",
            str(ir_a),
            str(ir_b),
            str(out_dir),
            "--alpha",
            "0.25",
            "--alpha",
            "0.75",
            "--checkpoint-file",
            str(checkpoint),
            "--workers",
            "1",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert first.exit_code == 0, first.stdout

    second = runner.invoke(
        app,
        [
            "ir",
            "morph-sweep",
            str(ir_a),
            str(ir_b),
            str(out_dir),
            "--alpha",
            "0.25",
            "--alpha",
            "0.75",
            "--checkpoint-file",
            str(checkpoint),
            "--resume",
            "--workers",
            "1",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert second.exit_code == 0, second.stdout
    summary = json.loads((out_dir / "morph_sweep_summary.json").read_text(encoding="utf-8"))
    assert int(summary["planned"]) == 2
    assert int(summary["executed"]) == 0
    assert int(summary["resumed_skipped"]) == 2


def test_ir_morph_sweep_retries_and_persists_failures(tmp_path: Path) -> None:
    ir_a = tmp_path / "a.wav"
    ir_b = tmp_path / "not_audio.wav"
    out_dir = tmp_path / "failed_sweep"
    checkpoint = tmp_path / "failed_checkpoint.json"
    a = np.zeros((1024, 1), dtype=np.float64)
    a[0, 0] = 1.0
    sf.write(str(ir_a), a, 16_000)
    ir_b.write_text("this is not a valid wav file", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ir",
            "morph-sweep",
            str(ir_a),
            str(ir_b),
            str(out_dir),
            "--alpha",
            "0.2",
            "--alpha",
            "0.6",
            "--retries",
            "2",
            "--continue-on-error",
            "--allow-failed",
            "--checkpoint-file",
            str(checkpoint),
            "--workers",
            "1",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert all(int(row["attempts"]) == 3 for row in rows if isinstance(row, dict))
    assert all(not bool(row["success"]) for row in rows if isinstance(row, dict))


def test_batch_render_parallel_jobs(tmp_path: Path) -> None:
    sr = 16_000
    irfile = tmp_path / "ir.wav"
    ir = np.zeros((256, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    sf.write(str(irfile), ir, sr)

    in1 = tmp_path / "in1.wav"
    in2 = tmp_path / "in2.wav"
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"
    sf.write(str(in1), np.zeros((2048, 1), dtype=np.float64), sr)
    sf.write(str(in2), np.zeros((2048, 1), dtype=np.float64), sr)

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


def test_batch_augment_template_emits_manifest_shape() -> None:
    result = runner.invoke(app, ["batch", "augment-template"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["version"] == "0.7"
    assert "jobs" in payload
    assert "profiles" in payload
    assert "asr-reverb-v1" in payload["profiles"]


def test_batch_augment_profiles_lists_builtin_profiles() -> None:
    result = runner.invoke(app, ["batch", "augment-profiles", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "asr-reverb-v1" in payload
    assert "music-reverb-v1" in payload
    assert "drums-room-v1" in payload


def test_batch_augment_dry_run_plans_without_writing_audio(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[40:90, 0] = 0.4
    sf.write(str(infile), audio, sr)

    out_root = tmp_path / "aug_out"
    manifest = tmp_path / "augment_manifest.json"
    payload = {
        "version": "0.7",
        "dataset_name": "dry_run_case",
        "profile": "asr-reverb-v1",
        "seed": 17,
        "variants_per_input": 2,
        "output_root": str(out_root),
        "jobs": [
            {
                "id": "utt_0001",
                "infile": str(infile),
                "split": "train",
                "label": "speaker_a",
                "tags": ["speech"],
                "options": {"rt60": 0.25, "wet": 0.28, "dry": 0.92},
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["batch", "augment", str(manifest), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    text = _combined_cli_output(result)
    assert "Batch Augment Dry-Run" in text
    assert "plans" in text
    assert not out_root.exists()


def test_batch_augment_split_isolation_rejects_overlapping_sources(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "source.wav"
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[10:40, 0] = 0.2
    sf.write(str(infile), audio, sr)

    manifest = tmp_path / "augment_overlap_manifest.json"
    payload = {
        "version": "0.7",
        "dataset_name": "leak_guard",
        "profile": "asr-reverb-v1",
        "seed": 7,
        "variants_per_input": 1,
        "jobs": [
            {
                "id": "shared_utt",
                "infile": str(infile),
                "split": "train",
                "label": "speaker_a",
            },
            {
                "id": "shared_utt",
                "infile": str(infile),
                "split": "val",
                "label": "speaker_a",
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["batch", "augment", str(manifest), "--dry-run"])
    assert result.exit_code != 0
    text = _combined_cli_output(result)
    assert "Split isolation violation" in text

    allowed = runner.invoke(
        app,
        ["batch", "augment", str(manifest), "--dry-run", "--allow-split-overlap"],
    )
    assert allowed.exit_code == 0, allowed.stdout


def test_batch_augment_generates_dataset_and_metadata(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "voice.wav"
    audio = np.zeros((1400, 1), dtype=np.float64)
    audio[120:240, 0] = 0.55
    sf.write(str(infile), audio, sr)

    out_root = tmp_path / "augmented_data"
    dataset_card = out_root / "DATASET_CARD.md"
    metrics_csv = out_root / "augmentation_metrics.csv"
    qa_bundle = out_root / "augmentation_qa_bundle.json"
    manifest = tmp_path / "augment_manifest_run.json"
    payload = {
        "version": "0.7",
        "dataset_name": "research_set_alpha",
        "profile": "asr-reverb-v1",
        "seed": 31,
        "variants_per_input": 2,
        "output_root": str(out_root),
        "default_options": {
            "engine": "algo",
            "rt60": 0.22,
            "wet": 0.30,
            "dry": 0.90,
            "repeat": 1,
            "fdn_matrix": "hadamard",
            "fdn_lines": 6,
            "output_subtype": "pcm16",
            "normalize_stage": "none",
            "output_peak_norm": "input",
        },
        "jobs": [
            {
                "id": "src_voice_01",
                "infile": str(infile),
                "split": "train",
                "label": "speaker_a",
                "tags": ["speech", "en"],
                "metadata": {"speaker_id": "A01"},
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "batch",
            "augment",
            str(manifest),
            "--jobs",
            "1",
            "--copy-dry",
            "--allow-failed",
            "--dataset-card-out",
            str(dataset_card),
            "--metrics-csv-out",
            str(metrics_csv),
            "--provenance-hash",
        ],
    )
    assert result.exit_code == 0, result.stdout

    augmented = sorted((out_root / "train").glob("*__a*.wav"))
    assert len(augmented) == 2
    dry = sorted((out_root / "train").glob("*__dry.wav"))
    assert len(dry) == 1

    jsonl_path = out_root / "augmentation_manifest.jsonl"
    summary_path = out_root / "augmentation_summary.json"
    assert jsonl_path.exists()
    assert summary_path.exists()
    assert dataset_card.exists()
    assert metrics_csv.exists()
    assert qa_bundle.exists()

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    ]
    assert len(rows) == 2
    assert all(bool(row["success"]) for row in rows)
    assert all(str(row["label"]) == "speaker_a" for row in rows)
    assert all(str(row["profile"]) == "asr-reverb-v1" for row in rows)
    assert all("render_config" in row for row in rows)
    assert all("source_metadata" in row for row in rows)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset_name"] == "research_set_alpha"
    assert int(summary["planned"]) == 2
    assert int(summary["success"]) == 2
    assert int(summary["failed"]) == 0
    assert "dataset_card" in summary
    assert "metrics_csv" in summary
    assert "qa_bundle" in summary
    assert "provenance_hash" in summary

    qa = json.loads(qa_bundle.read_text(encoding="utf-8"))
    assert qa["version"] == "augmentation-qa-v1"
    assert qa["baseline_present"] is False
    assert isinstance(qa.get("split_quality"), dict)


def test_batch_augment_baseline_summary_emits_class_balance_delta(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "voice_base.wav"
    audio = np.zeros((1200, 1), dtype=np.float64)
    audio[80:240, 0] = 0.45
    sf.write(str(infile), audio, sr)

    manifest = tmp_path / "augment_manifest_baseline.json"
    payload = {
        "version": "0.7",
        "dataset_name": "baseline_delta_case",
        "profile": "asr-reverb-v1",
        "seed": 99,
        "variants_per_input": 1,
        "default_options": {
            "engine": "algo",
            "rt60": 0.20,
            "wet": 0.25,
            "dry": 0.90,
            "normalize_stage": "none",
            "output_peak_norm": "input",
        },
        "jobs": [
            {
                "id": "src_a",
                "infile": str(infile),
                "split": "train",
                "label": "speaker_a",
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    out_a = tmp_path / "aug_a"
    out_b = tmp_path / "aug_b"
    first = runner.invoke(
        app,
        [
            "batch",
            "augment",
            str(manifest),
            "--output-root",
            str(out_a),
            "--jobs",
            "1",
            "--copy-dry",
        ],
    )
    assert first.exit_code == 0, first.stdout
    baseline_summary = out_a / "augmentation_summary.json"
    assert baseline_summary.exists()

    second = runner.invoke(
        app,
        [
            "batch",
            "augment",
            str(manifest),
            "--output-root",
            str(out_b),
            "--jobs",
            "1",
            "--copy-dry",
            "--baseline-summary",
            str(baseline_summary),
        ],
    )
    assert second.exit_code == 0, second.stdout
    qa_bundle = out_b / "augmentation_qa_bundle.json"
    qa = json.loads(qa_bundle.read_text(encoding="utf-8"))
    assert qa["baseline_present"] is True
    delta = qa.get("class_balance_delta")
    assert isinstance(delta, dict)
    assert isinstance(delta.get("global_label_delta"), dict)


def test_batch_augment_without_copy_dry_still_writes_outputs(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "voice.wav"
    audio = np.zeros((1024, 1), dtype=np.float64)
    audio[80:220, 0] = 0.42
    sf.write(str(infile), audio, sr)

    out_root = tmp_path / "augmented_data_no_dry"
    manifest = tmp_path / "augment_manifest_no_dry.json"
    payload = {
        "version": "0.7",
        "dataset_name": "research_set_no_dry",
        "profile": "asr-reverb-v1",
        "seed": 99,
        "variants_per_input": 1,
        "output_root": str(out_root),
        "default_options": {
            "engine": "algo",
            "rt60": 0.24,
            "wet": 0.28,
            "dry": 0.88,
            "repeat": 1,
            "fdn_matrix": "hadamard",
            "fdn_lines": 6,
            "output_subtype": "float32",
            "normalize_stage": "none",
            "output_peak_norm": "input",
        },
        "jobs": [
            {
                "id": "src_voice_01",
                "infile": str(infile),
                "split": "train",
                "label": "speaker_a",
            }
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "batch",
            "augment",
            str(manifest),
            "--jobs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    augmented = sorted((out_root / "train").glob("*__a*.wav"))
    assert len(augmented) == 1
    assert (out_root / "augmentation_manifest.jsonl").exists()
    assert (out_root / "augmentation_summary.json").exists()


def test_batch_render_lucky_mode_creates_multiple_outputs(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    out_dir = tmp_path / "batch_lucky"
    source = np.zeros((2048, 1), dtype=np.float64)
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
    ir = np.zeros((128, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    sf.write(str(irfile), ir, sr)

    in1 = tmp_path / "in1.wav"
    in2 = tmp_path / "in2.wav"
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"
    sf.write(str(in1), np.zeros((1024, 1), dtype=np.float64), sr)
    sf.write(str(in2), np.zeros((1024, 1), dtype=np.float64), sr)

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
    audio = np.zeros((512, 1), dtype=np.float64)
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
    sf.write(str(ir_file), np.array([[1.0]], dtype=np.float64), sr)

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


def test_render_validation_rejects_repro_bundle_in_lucky_mode(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((512, 1), dtype=np.float64)
    infile = tmp_path / "in.wav"
    sf.write(str(infile), audio, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out.wav"),
            "--engine",
            "algo",
            "--lucky",
            "2",
            "--repro-bundle",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    text = _combined_cli_output(result)
    assert "--repro-bundle" in text


def test_render_validation_suggests_supported_output_extension(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((512, 1), dtype=np.float64)
    infile = tmp_path / "in.wav"
    sf.write(str(infile), audio, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(tmp_path / "out.wavee"),
            "--engine",
            "algo",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    text = _combined_cli_output(result)
    assert "Unsupported output audio extension: .wavee." in text
    assert "Did you mean" in text
    assert ".wav" in text
    assert "Supported:" in text


def test_render_writes_failure_report_when_pipeline_raises(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    audio = np.zeros((512, 1), dtype=np.float64)
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    report_out = tmp_path / "failure_report.json"
    sf.write(str(infile), audio, 48_000)

    def _raise_pipeline(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(cli_module, "run_render_pipeline", _raise_pipeline)
    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--failure-report-out",
            str(report_out),
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert report_out.exists()
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "render-failure-report-v1"
    assert payload["error_type"] == "RuntimeError"
    assert "simulated pipeline failure" in str(payload["error"])


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
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    left = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float64)
    right = (0.2 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float64)
    stereo = np.column_stack((left, right)).astype(np.float64)

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

    out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == _DEFAULT_RENDER_SR
    assert out.shape[1] == 2
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    assert int(payload["config"]["ambi_order"]) == 1
    assert payload["config"]["ambi_decode_to"] == "stereo"


def test_render_rejects_ambi_channel_mismatch_without_encode(tmp_path: Path) -> None:
    sr = 24_000
    stereo = np.zeros((2048, 2), dtype=np.float64)
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
    text = _combined_cli_output(result)
    assert "Input channels" in text
    assert "--ambi-order 2" in text


def test_analyze_ambisonic_metrics_mode(tmp_path: Path) -> None:
    sr = 48_000
    n = 4096
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    foa = np.column_stack(
        (
            0.2 * np.sin(2.0 * np.pi * 120.0 * t),
            0.1 * np.sin(2.0 * np.pi * 150.0 * t),
            0.1 * np.sin(2.0 * np.pi * 80.0 * t),
            0.15 * np.sin(2.0 * np.pi * 200.0 * t),
        )
    ).astype(np.float64)
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
    x = np.ones((n, 1), dtype=np.float64)
    ir = np.zeros((64, 1), dtype=np.float64)
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
    out, _ = sf.read(str(outfile), always_2d=True, dtype="float64")
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
    x = np.zeros((512, 1), dtype=np.float64)
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
    assert "--automation-point" in _combined_cli_output(result)


def test_render_automation_points_wet_ramp_without_file(tmp_path: Path) -> None:
    sr = 16_000
    n = sr // 2
    x = np.ones((n, 1), dtype=np.float64)
    ir = np.zeros((64, 1), dtype=np.float64)
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


def test_render_rejects_invalid_automation_point_interp(tmp_path: Path) -> None:
    sr = 16_000
    n = sr // 4
    x = np.ones((n, 1), dtype=np.float64)
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "bad_interp_in.wav"
    irfile = tmp_path / "bad_interp_ir.wav"
    outfile = tmp_path / "bad_interp_out.wav"
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
            "wet:0.0:0.8:not-a-real-interp",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "automation interpolation" in _combined_cli_output(result)


def test_render_rejects_invalid_automation_file_interp(tmp_path: Path) -> None:
    sr = 16_000
    n = sr // 4
    x = np.ones((n, 1), dtype=np.float64)
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "bad_file_interp_in.wav"
    irfile = tmp_path / "bad_file_interp_ir.wav"
    outfile = tmp_path / "bad_file_interp_out.wav"
    auto_file = tmp_path / "bad_interp_automation.json"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)
    auto_file.write_text(
        json.dumps(
            {
                "mode": "block",
                "block_ms": 10.0,
                "lanes": [
                    {
                        "target": "wet",
                        "type": "breakpoints",
                        "interp": "invalid-curve",
                        "points": [[0.0, 0.1], [0.2, 0.9]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

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
            "--automation-file",
            str(auto_file),
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "automation interpolation" in _combined_cli_output(result)


def test_render_automation_points_drive_algo_engine_targets(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    infile = tmp_path / "algo_auto_in.wav"
    outfile = tmp_path / "algo_auto_out.wav"
    trace_file = tmp_path / "algo_auto_trace.csv"
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
            "rt60:0.25:1200.0:linear",
            "--automation-point",
            "damping:0.0:0.65:linear",
            "--automation-point",
            "room-size:0.0:0.8:linear",
            "--automation-point",
            "room-size:0.25:1.6:linear",
            "--automation-trace-out",
            str(trace_file),
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert trace_file.exists()
    out, _ = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert float(np.max(np.abs(out))) > 1e-6
    rows = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(rows) > 2
    header = rows[0].split(",")
    assert "rt60" in header
    rt60_col = header.index("rt60")
    rt60_values = [float(row.split(",")[rt60_col]) for row in rows[1:] if row.strip() != ""]
    assert max(rt60_values) > 1_000.0

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "rt60" in automation["targets"]
    assert "damping" in automation["targets"]
    assert "room-size" in automation["targets"]
    assert "rt60" in automation.get("engine_targets", [])
    assert "room-size" in automation.get("engine_targets", [])


def test_render_automation_points_drive_perceptual_macro_targets(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    infile = tmp_path / "macro_auto_in.wav"
    outfile = tmp_path / "macro_auto_out.wav"
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
            "warmth:0.0:0.7:linear",
            "--automation-point",
            "warmth:0.25:0.2:linear",
            "--automation-point",
            "clarity:0.0:-0.4:linear",
            "--automation-point",
            "size-macro:0.0:0.5:linear",
            "--automation-point",
            "enveloping:0.0:0.6:linear",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "warmth-macro" in automation["targets"]
    assert "clarity-macro" in automation["targets"]
    assert "room-size-macro" in automation["targets"]
    assert "envelopment-macro" in automation["targets"]
    assert "warmth-macro" in automation.get("engine_targets", [])


def test_render_automation_points_drive_track_c_targets(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    infile = tmp_path / "trackc_auto_in.wav"
    outfile = tmp_path / "trackc_auto_out.wav"
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
            "rt60-tilt:0.0:0.0:linear",
            "--automation-point",
            "rt60-tilt:0.25:0.55:linear",
            "--automation-point",
            "tonal-correction:0.0:0.15:linear",
            "--automation-point",
            "tonal-correction:0.25:0.75:linear",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "fdn-rt60-tilt" in automation["targets"]
    assert "fdn-tonal-correction-strength" in automation["targets"]
    assert "fdn-rt60-tilt" in automation.get("engine_targets", [])
    assert "fdn-tonal-correction-strength" in automation.get("engine_targets", [])
    assert "multiband" in str(payload["effective"]["compute_backend"])
    assert "tonalcorr" in str(payload["effective"]["compute_backend"])


def test_render_rejects_engine_automation_targets_for_convolution(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir = np.zeros((64, 1), dtype=np.float64)
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
    assert "require algorithmic render path" in _combined_cli_output(result)


def test_render_rejects_conv_automation_targets_for_algo(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float64)
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
            "--automation-point",
            "ir-blend-alpha:0.0:0.5",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "require convolution render path" in _combined_cli_output(result)


def test_render_rejects_ir_blend_alpha_without_ir_blend(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 4, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "conv_no_blend_in.wav"
    irfile = tmp_path / "conv_no_blend_ir.wav"
    outfile = tmp_path / "conv_no_blend_out.wav"
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
            "ir-blend-alpha:0.0:0.0",
            "--automation-point",
            "ir-blend-alpha:0.1:1.0",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "requires --ir-blend" in _combined_cli_output(result)


def test_render_conv_ir_blend_alpha_automation_applies(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((2 * sr, 1), dtype=np.float64)
    x[0, 0] = 1.0
    ir_base = np.zeros((128, 1), dtype=np.float64)
    ir_base[0, 0] = 1.0
    ir_blend = np.zeros((128, 1), dtype=np.float64)
    ir_blend[16, 0] = 0.75
    infile = tmp_path / "conv_blend_in.wav"
    base_ir = tmp_path / "conv_blend_base.wav"
    blend_ir = tmp_path / "conv_blend_b.wav"
    outfile = tmp_path / "conv_blend_out.wav"
    sf.write(str(infile), x, sr)
    sf.write(str(base_ir), ir_base, sr)
    sf.write(str(blend_ir), ir_blend, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "conv",
            "--ir",
            str(base_ir),
            "--ir-blend",
            str(blend_ir),
            "--normalize-stage",
            "none",
            "--automation-point",
            "ir-blend-alpha:0.0:0.0:linear",
            "--automation-point",
            "ir-blend-alpha:1.5:1.0:linear",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "ir-blend-alpha" in automation["targets"]
    assert "ir-blend-alpha" in automation.get("conv_targets", [])
    assert isinstance(automation.get("conv_summary"), dict)
    conv_summary = automation["conv_summary"]
    assert conv_summary.get("ir_blend_alpha_applied") is True
    assert float(conv_summary.get("alpha_max", 0.0)) >= float(conv_summary.get("alpha_min", 0.0))


def test_render_automation_mixed_lanes_deterministic_replay(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((3 * sr, 1), dtype=np.float64)
    x[100:300, 0] = 0.5
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "determin_in.wav"
    irfile = tmp_path / "determin_ir.wav"
    out_a = tmp_path / "determin_a.wav"
    out_b = tmp_path / "determin_b.wav"
    auto_file = tmp_path / "determin_automation.json"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    auto_file.write_text(
        json.dumps(
            {
                "mode": "sample",
                "lanes": [
                    {
                        "target": "wet",
                        "type": "breakpoints",
                        "interp": "smooth",
                        "points": [
                            {"time": 0.0, "value": 0.7},
                            {"time": 1.2, "value": 0.3},
                            {"time": 2.4, "value": 0.8},
                        ],
                    },
                    {
                        "target": "gain-db",
                        "type": "lfo",
                        "shape": "triangle",
                        "rate_hz": 0.4,
                        "depth": 4.0,
                        "center": -2.0,
                        "start_s": 0.0,
                        "end_s": 3.0,
                    },
                    {
                        "target": "dry",
                        "type": "segment",
                        "start_s": 1.5,
                        "end_s": 2.5,
                        "value": 0.25,
                        "ramp_ms": 120.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    args = [
        "render",
        str(infile),
        str(out_a),
        "--engine",
        "conv",
        "--ir",
        str(irfile),
        "--normalize-stage",
        "none",
        "--automation-file",
        str(auto_file),
        "--no-progress",
    ]
    result_a = runner.invoke(app, args)
    assert result_a.exit_code == 0, result_a.stdout

    args[2] = str(out_b)
    result_b = runner.invoke(app, args)
    assert result_b.exit_code == 0, result_b.stdout

    payload_a = json.loads(Path(f"{out_a}.analysis.json").read_text(encoding="utf-8"))
    payload_b = json.loads(Path(f"{out_b}.analysis.json").read_text(encoding="utf-8"))
    auto_a = payload_a["effective"]["automation"]
    auto_b = payload_b["effective"]["automation"]
    assert isinstance(auto_a, dict)
    assert isinstance(auto_b, dict)
    assert auto_a.get("signature") == auto_b.get("signature")
    assert "wet" in auto_a.get("post_targets", [])
    assert "dry" in auto_a.get("post_targets", [])
    assert "gain-db" in auto_a.get("post_targets", [])

    y_a, _ = sf.read(str(out_a), always_2d=True, dtype="float64")
    y_b, _ = sf.read(str(out_b), always_2d=True, dtype="float64")
    assert y_a.shape == y_b.shape
    assert np.allclose(y_a, y_b, atol=1e-7)


def test_render_automation_invalid_lane_context_is_reported(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 2, 1), dtype=np.float64)
    x[0, 0] = 1.0
    infile = tmp_path / "bad_lane_in.wav"
    outfile = tmp_path / "bad_lane_out.wav"
    auto_file = tmp_path / "bad_lane_automation.json"
    sf.write(str(infile), x, sr)

    auto_file.write_text(
        json.dumps(
            {
                "mode": "block",
                "lanes": [
                    {
                        "target": "wet",
                        "type": "segment",
                        "start_s": 0.4,
                        "end_s": 0.2,
                        "value": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--automation-file",
            str(auto_file),
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "lane #1" in _combined_cli_output(result)


def test_render_feature_vector_lanes_drive_wet_and_emit_trace(tmp_path: Path) -> None:
    sr = 16_000
    n = 2 * sr
    x = np.zeros((n, 1), dtype=np.float64)
    x[200:1200, 0] = 0.2
    x[4000:5200, 0] = 0.8
    x[12000:12500, 0] = 0.45
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0

    infile = tmp_path / "feature_lane_in.wav"
    irfile = tmp_path / "feature_lane_ir.wav"
    outfile = tmp_path / "feature_lane_out.wav"
    trace_file = tmp_path / "feature_lane_trace.csv"
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
            "--feature-vector-lane",
            "target=wet,source=loudness_norm,weight=0.75,bias=0.0,curve=smoothstep,combine=replace",
            "--feature-vector-lane",
            "target=wet,source=transient_strength,weight=0.35,bias=0.0,curve=power,curve_amount=1.5,hysteresis_up=0.02,hysteresis_down=0.01,combine=add",
            "--feature-vector-trace-out",
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
    feature_payload = automation.get("feature_vector")
    assert isinstance(feature_payload, dict)
    sources = feature_payload.get("sources", [])
    assert isinstance(sources, list)
    assert "loudness_norm" in sources
    assert "transient_strength" in sources
    assert isinstance(feature_payload.get("signature"), str)

    trace_header = trace_file.read_text(encoding="utf-8").splitlines()[0]
    assert "feature_loudness_norm" in trace_header
    assert "target_wet" in trace_header


def test_render_feature_vector_lanes_are_deterministic(tmp_path: Path) -> None:
    sr = 16_000
    n = 2 * sr
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    env = np.linspace(0.2, 1.0, n, dtype=np.float64)
    x = (env * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float64)[:, np.newaxis]
    ir = np.zeros((96, 1), dtype=np.float64)
    ir[0, 0] = 1.0

    infile = tmp_path / "feature_determin_in.wav"
    irfile = tmp_path / "feature_determin_ir.wav"
    out_a = tmp_path / "feature_determin_a.wav"
    out_b = tmp_path / "feature_determin_b.wav"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    args = [
        "render",
        str(infile),
        str(out_a),
        "--engine",
        "conv",
        "--ir",
        str(irfile),
        "--normalize-stage",
        "none",
        "--feature-vector-lane",
        "target=gain-db,source=loudness_norm,weight=9.0,bias=-9.0,curve=linear,combine=replace",
        "--feature-vector-lane",
        "target=gain-db,source=spectral_flux,weight=3.0,bias=0.0,curve=exp,curve_amount=1.2,hysteresis_up=0.01,hysteresis_down=0.01,combine=add",
        "--no-progress",
    ]
    result_a = runner.invoke(app, args)
    assert result_a.exit_code == 0, result_a.stdout

    args[2] = str(out_b)
    result_b = runner.invoke(app, args)
    assert result_b.exit_code == 0, result_b.stdout

    payload_a = json.loads(Path(f"{out_a}.analysis.json").read_text(encoding="utf-8"))
    payload_b = json.loads(Path(f"{out_b}.analysis.json").read_text(encoding="utf-8"))
    auto_a = payload_a["effective"]["automation"]
    auto_b = payload_b["effective"]["automation"]
    assert isinstance(auto_a, dict)
    assert isinstance(auto_b, dict)
    assert auto_a.get("signature") == auto_b.get("signature")
    feature_a = auto_a.get("feature_vector")
    feature_b = auto_b.get("feature_vector")
    assert isinstance(feature_a, dict)
    assert isinstance(feature_b, dict)
    assert feature_a.get("signature") == feature_b.get("signature")

    y_a, _ = sf.read(str(out_a), always_2d=True, dtype="float64")
    y_b, _ = sf.read(str(out_b), always_2d=True, dtype="float64")
    assert y_a.shape == y_b.shape
    assert np.allclose(y_a, y_b, atol=1e-7)


def test_render_feature_vector_target_source_mapping_graph_is_topological(
    tmp_path: Path,
) -> None:
    sr = 16_000
    n = sr
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    x = (0.7 * np.sin(2.0 * np.pi * 180.0 * t)).astype(np.float64)[:, np.newaxis]
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0

    infile = tmp_path / "feature_graph_in.wav"
    irfile = tmp_path / "feature_graph_ir.wav"
    outfile = tmp_path / "feature_graph_out.wav"
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
            "--normalize-stage",
            "none",
            "--feature-vector-lane",
            "target=dry,source=target:wet,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--feature-vector-lane",
            "target=wet,source=loudness_norm,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    feature_payload = automation.get("feature_vector")
    assert isinstance(feature_payload, dict)
    mapping = feature_payload.get("mapping")
    assert isinstance(mapping, dict)
    assert mapping.get("targets") == ["wet", "dry"]
    deps = mapping.get("dependencies")
    assert isinstance(deps, dict)
    assert deps.get("dry") == ["wet"]
    eval_order = mapping.get("evaluation_order")
    assert isinstance(eval_order, list)
    assert [str(item.get("target")) for item in eval_order] == ["wet", "dry"]
    assert isinstance(mapping.get("signature"), str)
    assert isinstance(feature_payload.get("signature"), str)


def test_render_rejects_feature_vector_target_source_cycle(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr, 1), dtype=np.float64)
    x[200:1200, 0] = 0.5
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "feature_cycle_in.wav"
    irfile = tmp_path / "feature_cycle_ir.wav"
    outfile = tmp_path / "feature_cycle_out.wav"
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
            "--normalize-stage",
            "none",
            "--feature-vector-lane",
            "target=wet,source=target:dry,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--feature-vector-lane",
            "target=dry,source=target:wet,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "contains a cycle" in _combined_cli_output(result).lower()


def test_render_rejects_unresolved_feature_vector_target_source(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr, 1), dtype=np.float64)
    x[200:800, 0] = 0.6
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "feature_unresolved_in.wav"
    irfile = tmp_path / "feature_unresolved_ir.wav"
    outfile = tmp_path / "feature_unresolved_out.wav"
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
            "--normalize-stage",
            "none",
            "--feature-vector-lane",
            "target=dry,source=target:wet,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "define automation for 'wet' first" in _combined_cli_output(result).lower()


def test_render_rejects_invalid_feature_vector_lane(tmp_path: Path) -> None:
    sr = 16_000
    x = np.zeros((sr // 2, 1), dtype=np.float64)
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "feature_bad_in.wav"
    irfile = tmp_path / "feature_bad_ir.wav"
    outfile = tmp_path / "feature_bad_out.wav"
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
            "--feature-vector-lane",
            "target=wet,source=does_not_exist",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "unsupported source" in _combined_cli_output(result).lower()


def test_render_automation_file_feature_vector_lane(tmp_path: Path) -> None:
    sr = 16_000
    n = sr
    x = np.zeros((n, 1), dtype=np.float64)
    x[200:1200, 0] = 0.4
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0

    infile = tmp_path / "feature_file_in.wav"
    irfile = tmp_path / "feature_file_ir.wav"
    outfile = tmp_path / "feature_file_out.wav"
    auto_file = tmp_path / "feature_file_auto.json"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)

    auto_file.write_text(
        json.dumps(
            {
                "mode": "block",
                "block_ms": 20.0,
                "lanes": [
                    {
                        "target": "wet",
                        "type": "feature-vector",
                        "source": "loudness_norm",
                        "weight": 1.0,
                        "bias": 0.0,
                        "curve": "linear",
                        "combine": "replace",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

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
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    assert "wet" in automation.get("targets", [])
    feature_payload = automation.get("feature_vector")
    assert isinstance(feature_payload, dict)
    assert "loudness_norm" in feature_payload.get("sources", [])


def test_render_feature_guide_align_reports_mismatch_actions(tmp_path: Path) -> None:
    sr = 16_000
    n = 2 * sr
    x = np.zeros((n, 1), dtype=np.float64)
    x[1000:2000, 0] = 0.8
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0

    guide_sr = 22_050
    guide_len = int(0.8 * guide_sr)
    t = np.arange(guide_len, dtype=np.float64) / float(guide_sr)
    guide = np.stack(
        (
            0.35 * np.sin(2.0 * np.pi * 220.0 * t),
            0.35 * np.sin(2.0 * np.pi * 330.0 * t),
        ),
        axis=1,
    ).astype(np.float64)

    infile = tmp_path / "feature_guide_in.wav"
    irfile = tmp_path / "feature_guide_ir.wav"
    guidefile = tmp_path / "feature_guide.wav"
    outfile = tmp_path / "feature_guide_out.wav"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)
    sf.write(str(guidefile), guide, guide_sr)

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
            "--feature-vector-lane",
            "target=wet,source=loudness_norm,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--feature-guide",
            str(guidefile),
            "--feature-guide-policy",
            "align",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    feature_payload = automation.get("feature_vector")
    assert isinstance(feature_payload, dict)
    guide_alignment = feature_payload.get("guide_alignment")
    assert isinstance(guide_alignment, dict)
    assert guide_alignment.get("policy") == "align"
    assert str(guide_alignment.get("sample_rate_action", "")).startswith("resample:")
    assert str(guide_alignment.get("channel_action", "")).startswith("mixdown:")
    assert str(guide_alignment.get("duration_action", "")).startswith("hold-last:")


def test_render_feature_guide_strict_rejects_mismatch(tmp_path: Path) -> None:
    sr = 16_000
    n = sr
    x = np.zeros((n, 1), dtype=np.float64)
    x[120:900, 0] = 0.6
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0

    guide_sr = 22_050
    guide = np.zeros((guide_sr // 2, 2), dtype=np.float64)
    guide[200:600, 0] = 0.4
    guide[300:700, 1] = 0.3

    infile = tmp_path / "feature_guide_strict_in.wav"
    irfile = tmp_path / "feature_guide_strict_ir.wav"
    guidefile = tmp_path / "feature_guide_strict.wav"
    outfile = tmp_path / "feature_guide_strict_out.wav"
    sf.write(str(infile), x, sr)
    sf.write(str(irfile), ir, sr)
    sf.write(str(guidefile), guide, guide_sr)

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
            "--feature-vector-lane",
            "target=wet,source=loudness_norm,weight=1.0,bias=0.0,curve=linear,combine=replace",
            "--feature-guide",
            str(guidefile),
            "--feature-guide-policy",
            "strict",
            "--no-progress",
        ],
    )
    assert result.exit_code != 0
    assert "feature-guide sample-rate mismatch" in _combined_cli_output(result).lower()


def test_render_feature_vector_payload_includes_schema_metadata(tmp_path: Path) -> None:
    sr = 16_000
    n = int(1.5 * sr)
    t = np.arange(n, dtype=np.float64) / float(sr)
    x = (
        0.22 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.14 * np.sin(2.0 * np.pi * 360.0 * t)
    ).astype(np.float64)[:, np.newaxis]
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "feature_schema_in.wav"
    irfile = tmp_path / "feature_schema_ir.wav"
    outfile = tmp_path / "feature_schema_out.wav"
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
            "--normalize-stage",
            "none",
            "--feature-vector-lane",
            "target=wet,source=mfcc_1_norm,weight=0.8,bias=0.0,curve=smoothstep,combine=replace",
            "--feature-vector-lane",
            "target=wet,source=formant_balance_norm,weight=0.3,bias=0.0,curve=linear,combine=add",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    feature_payload = payload["effective"]["automation"]["feature_vector"]
    assert isinstance(feature_payload, dict)
    assert feature_payload.get("schema_version") == "2.0.0"
    source_schema = feature_payload.get("source_schema")
    assert isinstance(source_schema, dict)
    assert "mfcc_1_norm" in source_schema
    assert "formant_balance_norm" in source_schema


def test_render_automation_safety_guards_reduce_control_step_delta(tmp_path: Path) -> None:
    sr = 16_000
    x = np.ones((sr, 1), dtype=np.float64)
    ir = np.zeros((64, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    infile = tmp_path / "auto_guard_in.wav"
    irfile = tmp_path / "auto_guard_ir.wav"
    outfile = tmp_path / "auto_guard_out.wav"
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
            "--normalize-stage",
            "none",
            "--automation-mode",
            "block",
            "--automation-block-ms",
            "10",
            "--automation-point",
            "wet:0.00:0.00:hold",
            "--automation-point",
            "wet:0.05:1.00:hold",
            "--automation-point",
            "wet:0.10:0.00:hold",
            "--automation-slew-limit-per-s",
            "0.20",
            "--automation-deadband",
            "0.03",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    automation = payload["effective"]["automation"]
    assert isinstance(automation, dict)
    guards = automation.get("safety_guards")
    assert isinstance(guards, dict)
    assert float(guards.get("slew_limit_per_s", 0.0)) > 0.0
    target_stats = guards.get("targets", {})
    assert isinstance(target_stats, dict)
    wet_stats = target_stats.get("wet")
    assert isinstance(wet_stats, dict)
    assert int(wet_stats.get("slew_hits", 0)) > 0
    assert float(wet_stats.get("max_delta_after", 0.0)) <= float(
        wet_stats.get("max_delta_before", 0.0)
    )


def test_render_track_c_calibration_diagnostics_are_emitted(tmp_path: Path) -> None:
    sr = 16_000
    n = int(1.2 * sr)
    t = np.arange(n, dtype=np.float64) / float(sr)
    x = (
        0.28 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 880.0 * t)
    ).astype(np.float64)[:, np.newaxis]
    infile = tmp_path / "trackc_diag_in.wav"
    outfile = tmp_path / "trackc_diag_out.wav"
    sf.write(str(infile), x, sr)

    result = runner.invoke(
        app,
        [
            "render",
            str(infile),
            str(outfile),
            "--engine",
            "algo",
            "--room-size-macro",
            "0.50",
            "--clarity-macro",
            "-0.25",
            "--warmth-macro",
            "0.45",
            "--envelopment-macro",
            "0.35",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(Path(f"{outfile}.analysis.json").read_text(encoding="utf-8"))
    calibration = payload["effective"].get("track_c_calibration")
    assert isinstance(calibration, dict)
    assert calibration.get("version") == "track-c-cal-v1"
    assert isinstance(calibration.get("targets"), dict)
    assert isinstance(calibration.get("measured"), dict)
    assert isinstance(calibration.get("errors"), dict)
    assert isinstance(calibration.get("within_envelope"), bool)
