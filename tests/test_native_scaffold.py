from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _native_sources(repo_root: Path) -> list[str]:
    return [
        str(repo_root / "native/verbx_c/src/audio.c"),
        str(repo_root / "native/verbx_c/src/algo_reverb.c"),
        str(repo_root / "native/verbx_c/src/render.c"),
        str(repo_root / "native/verbx_c/src/wav_io.c"),
        str(repo_root / "native/verbx_c/src/main.c"),
        str(repo_root / "native/verbx_c/src/cli.c"),
    ]


def _build_native_executable(tmp_path: Path) -> Path:
    cc = shutil.which("cc")
    assert cc is not None, "C compiler 'cc' is required for the native scaffold test."

    repo_root = Path(__file__).resolve().parents[1]
    exe = tmp_path / "verbx-c"
    command = [
        cc,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-I",
        str(repo_root / "native/verbx_c/include"),
        *_native_sources(repo_root),
        "-lm",
        "-o",
        str(exe),
    ]
    subprocess.run(command, check=True, cwd=repo_root)
    return exe


def test_native_scaffold_builds_and_reports_version(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)

    result = subprocess.run(
        [str(exe), "version"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "verbx-c 0.8.0-dev"


def test_native_render_mono_wav_round_trip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    sr = 16_000
    audio = np.zeros((256, 1), dtype=np.float64)
    audio[0, 0] = 0.75
    infile = tmp_path / "in.wav"
    outfile = tmp_path / "out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    result = subprocess.run(
        [
            str(exe),
            "render",
            str(infile),
            str(outfile),
            "--rt60",
            "0.6",
            "--wet",
            "1.0",
            "--dry",
            "0.0",
            "--tail-threshold-db",
            "-70",
            "--tail-hold-ms",
            "5",
            "--tail-metric",
            "rms",
            "--out-format",
            "float32",
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    rendered, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == sr
    assert rendered.shape[1] == 1
    assert rendered.shape[0] > audio.shape[0]
    assert np.max(np.abs(rendered)) > 1e-6
    assert np.max(np.abs(rendered[-8:, :])) == 0.0
    assert "render complete" in result.stdout
    assert "tail_metric: rms" in result.stdout
    assert "status: ok" in result.stdout


def test_native_doctor_reports_process_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)

    result = subprocess.run(
        [str(exe), "doctor"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert "process_contract:" in result.stdout
    assert "error_contract:" in result.stdout


def test_native_render_stereo_pcm16_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    sr = 22_050
    audio = np.zeros((320, 2), dtype=np.float64)
    audio[0, 0] = 0.6
    audio[10, 1] = -0.6
    infile = tmp_path / "stereo_in.wav"
    outfile = tmp_path / "stereo_out.wav"
    sf.write(str(infile), audio, sr, subtype="PCM_16")

    subprocess.run(
        [
            str(exe),
            "render",
            str(infile),
            str(outfile),
            "--rt60",
            "0.8",
            "--wet",
            "0.85",
            "--dry",
            "0.15",
            "--out-format",
            "pcm16",
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    info = sf.info(str(outfile))
    rendered, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == sr
    assert info.subtype == "PCM_16"
    assert rendered.shape[1] == 2
    assert rendered.shape[0] > audio.shape[0]


def test_native_render_silent_input_trims_to_short_zero_tail(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    sr = 48_000
    audio = np.zeros((1024, 1), dtype=np.float64)
    infile = tmp_path / "silent_in.wav"
    outfile = tmp_path / "silent_out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    subprocess.run(
        [
            str(exe),
            "render",
            str(infile),
            str(outfile),
            "--rt60",
            "12.0",
            "--tail-hold-ms",
            "5",
            "--out-format",
            "float64",
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    rendered, out_sr = sf.read(str(outfile), always_2d=True, dtype="float64")
    assert out_sr == sr
    assert rendered.shape == (240, 1)
    assert np.max(np.abs(rendered)) == 0.0
