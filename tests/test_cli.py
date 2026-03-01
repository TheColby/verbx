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


def test_analyze_lufs_mode(tmp_path: Path) -> None:
    audio = np.zeros((4096, 2), dtype=np.float32)
    audio[64:512, :] = 0.2
    infile = tmp_path / "analyze.wav"
    sf.write(str(infile), audio, 48_000)

    result = runner.invoke(app, ["analyze", str(infile), "--lufs"])
    assert result.exit_code == 0
    assert "integrated_lufs" in result.stdout


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
