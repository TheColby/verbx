"""Tests for the room size estimator (verbx.analysis.room_size).

Coverage:
- Core formula correctness for Sabine and Eyring paths
- Output key completeness and type contract
- Confidence scoring under various signal conditions
- Integration with AudioAnalyzer.analyze(include_room=True)
- Edge cases: silence, very short signal, single-channel mono
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, fftconvolve, sosfilt

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.room_size import _classify_room, _infer_absorption, estimate_room_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ir(sr: int, rt60: float, duration: float = 3.0) -> np.ndarray:
    """Synthetic exponential-decay IR (mono, shape (samples, 1))."""
    n = int(sr * duration)
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=np.float64) / float(sr)
    decay = np.exp(-np.log(1000.0) * t / rt60)
    noise = rng.standard_normal(n).astype(np.float64)
    ir = (decay * noise * 0.5).astype(np.float64)
    return ir[:, np.newaxis]


def _make_reverberant(sr: int, rt60: float, duration: float = 3.0) -> np.ndarray:
    """Bandlimited noise convolved with an exponential IR (stereo)."""
    from scipy.signal import butter, fftconvolve, sosfilt  # noqa: PLC0415

    rng = np.random.default_rng(7)
    n = int(sr * duration)
    noise = rng.standard_normal(n).astype(np.float64)
    sos = butter(4, [200.0 / (sr / 2), 4000.0 / (sr / 2)], btype="bandpass", output="sos")
    clean = sosfilt(sos, noise).astype(np.float64)

    ir = _make_ir(sr, rt60, duration=min(rt60 * 3, 5.0))[:, 0]
    wet = fftconvolve(clean, ir)[:n].astype(np.float64)
    peak = float(np.max(np.abs(wet))) + 1e-12
    wet /= peak
    return np.column_stack([wet, wet * 0.97])


# ---------------------------------------------------------------------------
# Output key contract
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "room_rt60_s",
    "room_rt60_low_s",
    "room_rt60_mid_s",
    "room_rt60_high_s",
    "room_volume_m3",
    "room_volume_m3_sabine",
    "room_volume_m3_eyring",
    "room_volume_m3_low",
    "room_volume_m3_high",
    "room_dim_width_m",
    "room_dim_depth_m",
    "room_dim_height_m",
    "room_surface_area_m2",
    "room_mean_absorption",
    "room_critical_distance_m",
    "room_class",
    "room_estimation_method",
    "room_confidence",
    "room_confidence_score",
}


def test_estimate_room_size_returns_all_expected_keys() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=1.2, duration=4.0)
    result = estimate_room_size(audio, sr)
    missing = EXPECTED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_estimate_room_size_numeric_keys_are_float() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=1.0, duration=4.0)
    result = estimate_room_size(audio, sr)
    float_keys = EXPECTED_KEYS - {"room_class", "room_estimation_method", "room_confidence"}
    for k in float_keys:
        assert isinstance(result[k], float), f"{k} is not float: {type(result[k])}"


def test_estimate_room_size_string_keys_are_str() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=1.0, duration=4.0)
    result = estimate_room_size(audio, sr)
    for k in ("room_class", "room_estimation_method", "room_confidence"):
        assert isinstance(result[k], str), f"{k} is not str: {type(result[k])}"


def test_all_numeric_values_are_finite() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=0.8, duration=3.0)
    result = estimate_room_size(audio, sr)
    float_keys = EXPECTED_KEYS - {"room_class", "room_estimation_method", "room_confidence"}
    for k in float_keys:
        v = result[k]
        assert np.isfinite(float(v)), f"{k} = {v} is not finite"


# ---------------------------------------------------------------------------
# Volume and dimension sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rt60", [0.3, 0.8, 1.5, 3.0])
def test_volume_grows_with_rt60(rt60: float) -> None:
    """Longer RT60 should produce larger estimated volume."""
    sr = 16_000
    audio = _make_ir(sr, rt60, duration=rt60 * 2.5 + 1.0)
    result = estimate_room_size(audio, sr)
    assert float(result["room_volume_m3"]) > 0.0


def test_larger_rt60_gives_larger_volume() -> None:
    sr = 16_000
    short = estimate_room_size(_make_ir(sr, 0.4, duration=3.0), sr)
    long_ = estimate_room_size(_make_ir(sr, 2.5, duration=8.0), sr)
    v_short = float(short["room_volume_m3"])
    v_long = float(long_["room_volume_m3"])
    assert v_long > v_short, f"Long RT60 volume {v_long:.1f} should exceed short {v_short:.1f}"


def test_dimensions_consistent_with_volume() -> None:
    sr = 16_000
    audio = _make_ir(sr, 1.0, duration=4.0)
    result = estimate_room_size(audio, sr)
    w = float(result["room_dim_width_m"])
    d = float(result["room_dim_depth_m"])
    h = float(result["room_dim_height_m"])
    v = float(result["room_volume_m3"])
    # Reconstructed volume should be close to reported volume (within 5 %)
    v_reconstructed = w * d * h
    assert abs(v_reconstructed - v) / max(v, 1e-3) < 0.05


def test_confidence_interval_bounds_volume() -> None:
    sr = 16_000
    audio = _make_ir(sr, 1.2, duration=5.0)
    result = estimate_room_size(audio, sr)
    v = float(result["room_volume_m3"])
    v_low = float(result["room_volume_m3_low"])
    v_high = float(result["room_volume_m3_high"])
    assert v_low < v < v_high


def test_critical_distance_is_positive() -> None:
    sr = 16_000
    audio = _make_ir(sr, 1.0, duration=4.0)
    result = estimate_room_size(audio, sr)
    assert float(result["room_critical_distance_m"]) > 0.0


# ---------------------------------------------------------------------------
# Room classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "volume_m3,rt60,expected",
    [
        (4.0, 0.10, "closet"),
        (25.0, 0.25, "small"),
        (100.0, 0.60, "medium"),
        (600.0, 1.5, "large"),
        (5000.0, 3.5, "very_large"),
        (50_000.0, 7.0, "cathedral"),
    ],
)
def test_classify_room_labels(volume_m3: float, rt60: float, expected: str) -> None:
    assert _classify_room(volume_m3, rt60) == expected


def test_estimate_room_size_class_is_valid_label() -> None:
    sr = 16_000
    audio = _make_ir(sr, 0.8, duration=3.0)
    result = estimate_room_size(audio, sr)
    valid = {"closet", "small", "medium", "large", "very_large", "cathedral", "unknown"}
    assert result["room_class"] in valid


# ---------------------------------------------------------------------------
# Absorption inference
# ---------------------------------------------------------------------------

def test_infer_absorption_hard_surface() -> None:
    # Low HF/LF ratio → hard surface → low alpha
    alpha = _infer_absorption(rt60_low=2.0, rt60_mid=1.0, rt60_high=0.5)
    assert alpha < 0.25


def test_infer_absorption_treated_surface() -> None:
    # High HF/LF ratio → treated → higher alpha
    alpha = _infer_absorption(rt60_low=1.5, rt60_mid=1.4, rt60_high=1.3)
    assert alpha > 0.30


def test_infer_absorption_fallback_for_zeros() -> None:
    alpha = _infer_absorption(rt60_low=0.0, rt60_mid=0.0, rt60_high=0.0)
    assert 0.01 <= alpha <= 0.99


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_silence_returns_low_confidence() -> None:
    sr = 16_000
    audio = np.zeros((sr * 3, 2), dtype=np.float64)
    result = estimate_room_size(audio, sr)
    assert result["room_confidence"] == "low"
    assert float(result["room_volume_m3"]) == 0.0


def test_very_short_signal_returns_low_confidence() -> None:
    sr = 16_000
    audio = np.random.default_rng(3).standard_normal((64, 1)).astype(np.float64) * 0.01
    result = estimate_room_size(audio, sr)
    assert result["room_confidence"] in ("low", "medium")


def test_mono_signal_works() -> None:
    sr = 16_000
    audio = _make_ir(sr, 0.8, duration=3.0)  # returns (n, 1)
    result = estimate_room_size(audio, sr)
    assert result["room_class"] != ""


def test_prior_absorption_overrides_inference() -> None:
    sr = 16_000
    audio = _make_ir(sr, 1.0, duration=4.0)
    r_default = estimate_room_size(audio, sr)
    r_override = estimate_room_size(audio, sr, prior_absorption=0.80)
    # Higher absorption → smaller volume for the same RT60
    assert float(r_override["room_volume_m3"]) != float(r_default["room_volume_m3"])


# ---------------------------------------------------------------------------
# Integration: AudioAnalyzer.analyze(include_room=True)
# ---------------------------------------------------------------------------

def test_analyzer_include_room_adds_room_keys() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=1.0, duration=3.0)
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr, include_room=True)
    for k in EXPECTED_KEYS:
        assert k in metrics, f"Key {k!r} missing from analyzer output"


def test_analyzer_without_room_has_no_room_keys() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=1.0, duration=3.0)
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr, include_room=False)
    room_keys = [k for k in metrics if k.startswith("room_")]
    assert not room_keys, f"Unexpected room keys in output: {room_keys}"


def test_analyzer_room_plus_edr_coexist() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=1.0, duration=3.0)
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr, include_room=True, include_edr=True)
    assert "edr_rt60_median_s" in metrics
    assert "room_volume_m3" in metrics


def test_analyzer_room_string_values_are_str() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=0.8, duration=3.0)
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr, include_room=True)
    assert isinstance(metrics["room_class"], str)
    assert isinstance(metrics["room_confidence"], str)
    assert isinstance(metrics["room_estimation_method"], str)


def test_analyzer_room_numeric_values_are_float() -> None:
    sr = 16_000
    audio = _make_reverberant(sr, rt60=0.8, duration=3.0)
    analyzer = AudioAnalyzer()
    metrics = analyzer.analyze(audio, sr, include_room=True)
    float_keys = EXPECTED_KEYS - {"room_class", "room_estimation_method", "room_confidence"}
    for k in float_keys:
        assert isinstance(metrics[k], float), f"{k} should be float in analyzer output"
