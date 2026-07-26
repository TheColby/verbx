#!/usr/bin/env python3
"""Compare the Python renderer and native verbx-c against the parity contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "tests/fixtures/native_render_parity_contract.json"
DEFAULT_NATIVE_EXE = ROOT / "build/native/verbx_c/verbx-c"


@dataclass(slots=True)
class FixturePaths:
    """Resolved paths for one parity fixture run."""

    input_wav: Path
    python_wav: Path
    native_wav: Path


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Load and lightly validate the native parity contract."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        msg = f"unsupported native parity contract schema: {payload.get('schema_version')!r}"
        raise ValueError(msg)
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("native parity contract must define at least one fixture")
    return payload


def build_fixture_audio(fixture: dict[str, Any]) -> np.ndarray:
    """Create a deterministic impulse fixture from the contract entry."""
    sr = int(fixture["sample_rate"])
    channels = int(fixture["channels"])
    frames = max(1, round(float(fixture["duration_seconds"]) * sr))
    audio = np.zeros((frames, channels), dtype=np.float64)
    audio[0, 0] = 0.75
    if channels > 1:
        offset = min(frames - 1, max(1, round(0.00045 * sr)))
        audio[offset, 1] = -0.65
    return audio


def write_fixture_input(fixture: dict[str, Any], path: Path) -> None:
    """Write the deterministic input WAV for one parity fixture."""
    subtype = _input_subtype_to_soundfile(str(fixture["input_subtype"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), build_fixture_audio(fixture), int(fixture["sample_rate"]), subtype=subtype)


def compare_audio_files(
    *,
    fixture: dict[str, Any],
    python_wav: Path,
    native_wav: Path,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Return metric deltas between Python and native rendered WAV files."""
    py_audio, py_sr = sf.read(str(python_wav), always_2d=True, dtype="float64")
    native_audio, native_sr = sf.read(str(native_wav), always_2d=True, dtype="float64")
    py_audio = np.asarray(py_audio, dtype=np.float64)
    native_audio = np.asarray(native_audio, dtype=np.float64)

    py_rms = _rms(py_audio)
    native_rms = _rms(native_audio)
    py_peak = _peak(py_audio)
    native_peak = _peak(native_audio)
    duration_delta_ms = (
        abs(len(py_audio) / max(1, py_sr) - len(native_audio) / max(1, native_sr)) * 1000.0
    )
    rms_delta_db = abs(_db(py_rms) - _db(native_rms))
    peak_abs_delta = abs(py_peak - native_peak)

    checks = {
        "sample_rate_exact": (py_sr == native_sr == int(fixture["sample_rate"])),
        "channel_count_exact": (
            py_audio.shape[1] == native_audio.shape[1] == int(fixture["channels"])
        ),
        "finite_samples_only": bool(
            np.isfinite(py_audio).all() and np.isfinite(native_audio).all()
        ),
        "tail_ends_in_exact_zeros": bool(
            _tail_is_exact_zero(py_audio) and _tail_is_exact_zero(native_audio)
        ),
        "peak_abs_tolerance": peak_abs_delta <= float(acceptance["peak_abs_tolerance"]),
        "rms_tolerance_db": rms_delta_db <= float(acceptance["rms_tolerance_db"]),
        "duration_tolerance_ms": duration_delta_ms <= float(acceptance["duration_tolerance_ms"]),
    }

    return {
        "name": str(fixture["name"]),
        "sample_rate": int(py_sr),
        "channels": int(py_audio.shape[1]),
        "python_frames": int(py_audio.shape[0]),
        "native_frames": int(native_audio.shape[0]),
        "python_peak_abs": py_peak,
        "native_peak_abs": native_peak,
        "peak_abs_delta": peak_abs_delta,
        "python_rms_dbfs": _db(py_rms),
        "native_rms_dbfs": _db(native_rms),
        "rms_delta_db": rms_delta_db,
        "duration_delta_ms": duration_delta_ms,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_contract(
    *,
    contract: dict[str, Any],
    native_exe: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Run all contract fixtures through Python and native renderers."""
    scenarios = []
    acceptance = dict(contract["acceptance_metrics"])
    for fixture in contract["fixtures"]:
        fixture_dir = work_dir / str(fixture["name"])
        paths = FixturePaths(
            input_wav=fixture_dir / "input.wav",
            python_wav=fixture_dir / "python.wav",
            native_wav=fixture_dir / "native.wav",
        )
        write_fixture_input(fixture, paths.input_wav)
        _run_python_reference(fixture, paths)
        native_stdout = _run_native_candidate(fixture, paths, native_exe)
        result = compare_audio_files(
            fixture=fixture,
            python_wav=paths.python_wav,
            native_wav=paths.native_wav,
            acceptance=acceptance,
        )
        result["native_stdout"] = native_stdout
        scenarios.append(result)

    return {
        "schema_version": 1,
        "contract_target": contract.get("target"),
        "native_exe": str(native_exe),
        "scenarios": scenarios,
        "passed": all(bool(item["passed"]) for item in scenarios),
    }


def ensure_native_exe(path: Path, *, build_native: bool, force_build: bool = False) -> Path:
    """Return a usable native executable, optionally building it first."""
    if path.exists() and not force_build:
        return path
    if not build_native:
        raise FileNotFoundError(f"native executable not found: {path}")
    subprocess.run([str(ROOT / "scripts/build_verbx_c.sh")], check=True, cwd=ROOT)
    if not path.exists():
        raise FileNotFoundError(f"native build did not produce expected executable: {path}")
    return path


def _run_python_reference(fixture: dict[str, Any], paths: FixturePaths) -> None:
    args = dict(fixture["args"])
    config = RenderConfig(
        engine="algo",
        rt60=float(args["rt60"]),
        wet=float(args["wet"]),
        dry=float(args["dry"]),
        pre_delay_ms=float(args["pre_delay_ms"]),
        damping=float(args["damping"]),
        tail_stop_threshold_db=float(args["tail_threshold_db"]),
        tail_stop_hold_ms=float(args["tail_hold_ms"]),
        tail_stop_metric=str(args["tail_metric"]),
        output_subtype=_output_subtype_to_render_config(str(fixture["output_subtype"])),
        output_peak_norm="target" if bool(args.get("peak_safe", False)) else "none",
        output_peak_target_dbfs=float(args.get("peak_ceiling_db", -1.0)),
        limiter=False,
        normalize_stage="none",
        progress=False,
        silent=True,
    )
    run_render_pipeline(paths.input_wav, paths.python_wav, config)


def _run_native_candidate(fixture: dict[str, Any], paths: FixturePaths, native_exe: Path) -> str:
    args = dict(fixture["args"])
    command = [
        str(native_exe),
        "render",
        str(paths.input_wav),
        str(paths.native_wav),
        "--rt60",
        str(args["rt60"]),
        "--wet",
        str(args["wet"]),
        "--dry",
        str(args["dry"]),
        "--pre-delay-ms",
        str(args["pre_delay_ms"]),
        "--damping",
        str(args["damping"]),
        "--tail-threshold-db",
        str(args["tail_threshold_db"]),
        "--tail-hold-ms",
        str(args["tail_hold_ms"]),
        "--tail-metric",
        str(args["tail_metric"]),
        "--out-format",
        str(fixture["output_subtype"]),
    ]
    if bool(args.get("peak_safe", False)):
        command.append("--peak-safe")
    command.extend(["--peak-ceiling-db", str(args.get("peak_ceiling_db", -1.0))])
    result = subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)
    return result.stdout


def _input_subtype_to_soundfile(value: str) -> str:
    mapping = {
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
        "float32": "FLOAT",
        "float64": "DOUBLE",
    }
    if value not in mapping:
        raise ValueError(f"unsupported fixture input subtype: {value}")
    return mapping[value]


def _output_subtype_to_render_config(value: str) -> str:
    mapping = {
        "pcm16": "pcm16",
        "float32": "float32",
        "float64": "float64",
    }
    if value not in mapping:
        raise ValueError(f"unsupported fixture output subtype: {value}")
    return mapping[value]


def _peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _tail_is_exact_zero(audio: np.ndarray, frames: int = 8) -> bool:
    if audio.size == 0:
        return True
    tail = audio[-min(int(frames), len(audio)) :, :]
    return bool(np.max(np.abs(tail)) == 0.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--native-exe", type=Path, default=DEFAULT_NATIVE_EXE)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--build-native",
        action="store_true",
        help="Rebuild verbx-c before running comparisons.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any metric fails.",
    )
    parser.add_argument(
        "--no-build-native",
        action="store_true",
        help="Do not build verbx-c when --native-exe is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    native_exe = ensure_native_exe(
        args.native_exe,
        build_native=not args.no_build_native,
        force_build=bool(args.build_native),
    )
    contract = load_contract(args.contract)

    if args.work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="verbx-native-parity-")
        work_dir = Path(temp_ctx.name)
    else:
        temp_ctx = None
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        report = compare_contract(contract=contract, native_exe=native_exe, work_dir=work_dir)
        report["work_dir"] = str(work_dir)
        payload = json.dumps(report, indent=2, sort_keys=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 1 if args.strict and not bool(report["passed"]) else 0
    finally:
        if temp_ctx is not None and not args.keep_work_dir:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
