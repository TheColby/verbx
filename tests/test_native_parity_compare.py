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


def test_structural_gate_can_pass_while_full_dsp_parity_is_pending() -> None:
    module = _load_module()
    checks = {
        "sample_rate_exact": True,
        "channel_count_exact": True,
        "finite_samples_only": True,
        "tail_ends_in_exact_zeros": True,
        "peak_abs_tolerance": False,
        "rms_tolerance_db": False,
        "duration_tolerance_ms": False,
    }
    report = {"scenarios": [{"checks": checks, "passed": False}], "passed": False}

    assert module.evaluate_required_checks(report) is True


def test_structural_gate_rejects_missing_or_failed_checks() -> None:
    module = _load_module()
    checks = {
        "sample_rate_exact": True,
        "channel_count_exact": True,
        "finite_samples_only": False,
        "tail_ends_in_exact_zeros": True,
    }

    assert module.evaluate_required_checks({"scenarios": [{"checks": checks}]}) is False
    assert module.evaluate_required_checks({"scenarios": []}) is False


def test_python_peak_safety_preserves_below_ceiling_audio(tmp_path: Path, monkeypatch) -> None:
    _check_python_reference_peak_safety(tmp_path, monkeypatch, peak=0.25)


def test_python_peak_safety_attenuates_above_ceiling_before_pcm_encoding(
    tmp_path: Path, monkeypatch
) -> None:
    _check_python_reference_peak_safety(tmp_path, monkeypatch, peak=2.0)


def _check_python_reference_peak_safety(tmp_path: Path, monkeypatch, peak: float) -> None:
    module = _load_module()
    fixture = module.load_contract()["fixtures"][1]
    paths = module.FixturePaths(
        tmp_path / "input.wav", tmp_path / "python.wav", tmp_path / "native.wav"
    )
    audio = np.array([[peak, -peak / 2.0], [0.0, 0.0]])

    def fake_pipeline(input_path, output_path, config):
        assert config.output_peak_norm == "none"
        assert config.output_subtype == "float64"
        assert config.limiter is False
        sf.write(output_path, audio, 22050, subtype="DOUBLE")

    monkeypatch.setattr(module, "run_render_pipeline", fake_pipeline)
    module._run_python_reference(fixture, paths)
    rendered, sr = sf.read(paths.python_wav, always_2d=True)
    ceiling = 10.0 ** (-1.0 / 20.0)
    expected = audio * min(1.0, ceiling / peak)
    # PCM16 rounding is the only permitted discrepancy; both channels share gain.
    np.testing.assert_allclose(rendered, expected, atol=1.0 / 32768.0, rtol=0.0)
    assert sr == 22050
    assert sf.info(paths.python_wav).subtype == "PCM_16"
    assert not list(tmp_path.glob("peak-safe-*"))
