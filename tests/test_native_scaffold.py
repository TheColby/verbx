from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _native_sources(repo_root: Path) -> list[str]:
    return [
        str(repo_root / "native/verbx_c/src/audio.c"),
        str(repo_root / "native/verbx_c/src/algo_reverb.c"),
        str(repo_root / "native/verbx_c/src/plugin_params.c"),
        str(repo_root / "native/verbx_c/src/plugin_realtime.c"),
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


def test_native_doctor_writes_machine_readable_json_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    json_out = tmp_path / "doctor.json"

    result = subprocess.run(
        [str(exe), "doctor", "--json-out", str(json_out)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert "process_contract:" in result.stdout
    assert payload["schema"] == "native-doctor-report-v1"
    assert payload["command"] == "verbx-c doctor"
    assert payload["status"] == "native-render-foundation"
    assert payload["process_contract"] == "read -> render -> tail_stop -> write (deterministic)"
    assert payload["compiler_family"] in {"clang", "gcc", "msvc", "unknown"}


def test_native_build_script_exposes_ergonomic_flags() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/build_verbx_c.sh"

    help_result = subprocess.run(
        [str(script), "--help"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    path_result = subprocess.run(
        [str(script), "--print-path"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert "--clean" in help_result.stdout
    assert "--doctor" in help_result.stdout
    assert "--print-path" in help_result.stdout
    assert path_result.stdout.strip().endswith("build/native/verbx_c/verbx-c")


def test_native_install_script_installs_binary_and_man_page(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/install_verbx_c.sh"
    prefix = tmp_path / "prefix"

    help_result = subprocess.run(
        [str(script), "--help"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert "--prefix" in help_result.stdout
    assert "--skip-build" in help_result.stdout
    assert "--no-man" in help_result.stdout

    subprocess.run(
        [str(script), "--prefix", str(prefix)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    installed = prefix / "bin/verbx-c"
    man_page = prefix / "share/man/man1/verbx-c.1"
    assert installed.exists()
    assert man_page.exists()
    version = subprocess.run(
        [str(installed), "version"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == "verbx-c 0.8.0-dev"


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


def test_native_render_peak_safe_scales_float_output_to_ceiling(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    sr = 16_000
    audio = np.zeros((256, 1), dtype=np.float64)
    audio[0, 0] = 2.0
    infile = tmp_path / "hot_in.wav"
    outfile = tmp_path / "hot_out.wav"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    result = subprocess.run(
        [
            str(exe),
            "render",
            str(infile),
            str(outfile),
            "--rt60",
            "0.2",
            "--wet",
            "0.0",
            "--dry",
            "1.0",
            "--tail-threshold-db",
            "-120",
            "--tail-hold-ms",
            "5",
            "--peak-safe",
            "--peak-ceiling-db",
            "-6",
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
    assert np.max(np.abs(rendered)) <= 10 ** (-6.0 / 20.0) + 1e-9
    assert "peak_safe: true" in result.stdout
    assert "peak_ceiling_db: -6.00" in result.stdout
    assert "peak_gain:" in result.stdout


def test_native_render_writes_machine_readable_json_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    exe = _build_native_executable(tmp_path)
    sr = 16_000
    audio = np.zeros((256, 1), dtype=np.float64)
    audio[0, 0] = 0.75
    infile = tmp_path / "report_in.wav"
    outfile = tmp_path / "report_out.wav"
    json_out = tmp_path / "native_report.json"
    sf.write(str(infile), audio, sr, subtype="DOUBLE")

    subprocess.run(
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
            "--tail-metric",
            "rms",
            "--peak-safe",
            "--peak-ceiling-db",
            "-3",
            "--out-format",
            "float32",
            "--json-out",
            str(json_out),
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema"] == "native-render-report-v1"
    assert payload["command"] == "verbx-c render"
    assert payload["status"] == "ok"
    assert payload["sample_rate"] == sr
    assert payload["channels"] == 1
    assert payload["out_format"] == "float32"
    assert payload["tail_metric"] == "rms"
    assert payload["peak_safe"] is True
    assert payload["peak_ceiling_db"] == -3
    assert payload["input_frames"] == 256
    assert payload["output_frames"] > payload["input_frames"]
    assert payload["output_peak_abs"] <= 10 ** (-3.0 / 20.0) + 1e-9


def test_native_render_parity_contract_is_narrow_and_deterministic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contract_path = repo_root / "tests/fixtures/native_render_parity_contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["python_reference"]["command"] == "verbx render"
    assert payload["native_candidate"]["command"] == "verbx-c render"
    assert payload["formats"]["channels"] == [1, 2]
    assert payload["controls"]["deferred"]
    assert {
        "rt60",
        "wet",
        "dry",
        "pre_delay_ms",
        "tail_threshold_db",
        "tail_hold_ms",
        "tail_metric",
        "out_format",
        "peak_safe",
        "peak_ceiling_db",
    }.issubset(set(payload["controls"]["required"]))
    assert len(payload["fixtures"]) >= 2
    assert payload["acceptance_metrics"]["finite_samples_only"] is True
    assert payload["acceptance_metrics"]["tail_ends_in_exact_zeros"] is True
