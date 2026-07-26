from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts/compare_native_render_parity.py"
    spec = importlib.util.spec_from_file_location("compare_native_render_parity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_fixture_audio_is_deterministic_stereo_impulse() -> None:
    module = _load_module()
    fixture = {
        "sample_rate": 16_000,
        "channels": 2,
        "duration_seconds": 0.01,
    }

    first = module.build_fixture_audio(fixture)
    second = module.build_fixture_audio(fixture)

    assert first.shape == (160, 2)
    np.testing.assert_array_equal(first, second)
    assert first[0, 0] == 0.75
    assert np.min(first[:, 1]) == -0.65


def test_compare_audio_files_reports_passing_identical_outputs(tmp_path: Path) -> None:
    module = _load_module()
    fixture = {
        "name": "same",
        "sample_rate": 8_000,
        "channels": 1,
    }
    acceptance = {
        "peak_abs_tolerance": 0.001,
        "rms_tolerance_db": 0.001,
        "duration_tolerance_ms": 0.001,
    }
    audio = np.zeros((64, 1), dtype=np.float64)
    audio[0, 0] = 0.5
    python_wav = tmp_path / "python.wav"
    native_wav = tmp_path / "native.wav"
    sf.write(str(python_wav), audio, 8_000, subtype="DOUBLE")
    sf.write(str(native_wav), audio, 8_000, subtype="DOUBLE")

    result = module.compare_audio_files(
        fixture=fixture,
        python_wav=python_wav,
        native_wav=native_wav,
        acceptance=acceptance,
    )

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["peak_abs_delta"] == 0.0
    assert result["duration_delta_ms"] == 0.0


def test_compare_audio_files_flags_metric_deltas(tmp_path: Path) -> None:
    module = _load_module()
    fixture = {
        "name": "different",
        "sample_rate": 8_000,
        "channels": 1,
    }
    acceptance = {
        "peak_abs_tolerance": 0.001,
        "rms_tolerance_db": 0.001,
        "duration_tolerance_ms": 0.001,
    }
    python_audio = np.zeros((64, 1), dtype=np.float64)
    native_audio = np.zeros((96, 1), dtype=np.float64)
    python_audio[0, 0] = 0.5
    native_audio[0, 0] = 0.25
    python_wav = tmp_path / "python.wav"
    native_wav = tmp_path / "native.wav"
    sf.write(str(python_wav), python_audio, 8_000, subtype="DOUBLE")
    sf.write(str(native_wav), native_audio, 8_000, subtype="DOUBLE")

    result = module.compare_audio_files(
        fixture=fixture,
        python_wav=python_wav,
        native_wav=native_wav,
        acceptance=acceptance,
    )

    assert result["passed"] is False
    assert result["checks"]["peak_abs_tolerance"] is False
    assert result["checks"]["duration_tolerance_ms"] is False
