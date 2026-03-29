from __future__ import annotations

import numpy as np
import pytest

from verbx.core.dereverb import (
    DereverbConfig,
    _bark_weighted_snr_db,
    _mcd_db,
    _stoi_approx,
    apply_dereverb,
    run_dereverb_benchmark,
)


def _synthetic_reverberant_impulse(*, sr: int, length_s: float) -> np.ndarray:
    n = int(sr * length_s)
    dry = np.zeros((n,), dtype=np.float64)
    dry[32] = 1.0
    ir = np.exp(-np.arange(int(0.6 * sr), dtype=np.float64) / (0.18 * sr))
    wet = np.convolve(dry, ir, mode="full")[:n]
    return wet[:, np.newaxis]


def test_apply_dereverb_reduces_late_tail_energy() -> None:
    sr = 16_000
    wet = _synthetic_reverberant_impulse(sr=sr, length_s=1.0)
    out = apply_dereverb(
        wet,
        sr,
        DereverbConfig(
            mode="wiener",
            strength=1.0,
            floor=0.05,
            window_ms=32.0,
            hop_ms=8.0,
            tail_ms=200.0,
            mix=1.0,
        ),
    )
    assert out.shape == wet.shape
    assert np.isfinite(out).all()
    late_start = int(0.25 * sr)
    in_late = float(np.mean(np.square(wet[late_start:, 0])))
    out_late = float(np.mean(np.square(out[late_start:, 0])))
    assert out_late < in_late


def test_apply_dereverb_spectral_sub_mode_is_stable() -> None:
    sr = 48_000
    x = np.zeros((2048, 2), dtype=np.float64)
    x[32:128, 0] = 0.5
    x[64:192, 1] = 0.3
    out = apply_dereverb(
        x,
        sr,
        DereverbConfig(
            mode="spectral_sub",
            strength=0.7,
            floor=0.1,
            window_ms=20.0,
            hop_ms=5.0,
            tail_ms=120.0,
            pre_emphasis=0.25,
            mix=0.9,
        ),
    )
    assert out.shape == x.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Objective quality metric tests
# ---------------------------------------------------------------------------

def _make_voiced_signal(sr: int, duration_s: float) -> np.ndarray:
    """Bandlimited voiced-noise burst (voice-like shape)."""
    from scipy.signal import butter, sosfilt  # noqa: PLC0415

    rng = np.random.default_rng(99)
    n = int(sr * duration_s)
    noise = rng.standard_normal(n).astype(np.float64)
    sos = butter(4, [200.0 / (sr / 2), 4000.0 / (sr / 2)], btype="bandpass", output="sos")
    sig = sosfilt(sos, noise)
    return (sig / (np.max(np.abs(sig)) + 1e-12) * 0.5).astype(np.float64)


def test_bark_weighted_snr_identical_signals_gives_high_value() -> None:
    sr = 16_000
    clean = _make_voiced_signal(sr, 2.0)
    value = _bark_weighted_snr_db(clean[:, np.newaxis], clean[:, np.newaxis], sr)
    assert np.isfinite(value)
    assert value > 20.0


def test_bark_weighted_snr_noisy_signal_is_lower() -> None:
    sr = 16_000
    rng = np.random.default_rng(7)
    clean = _make_voiced_signal(sr, 2.0)
    noisy = (clean + 0.2 * rng.standard_normal(len(clean))).astype(np.float64)
    snr_clean = _bark_weighted_snr_db(clean[:, np.newaxis], clean[:, np.newaxis], sr)
    snr_noisy = _bark_weighted_snr_db(clean[:, np.newaxis], noisy[:, np.newaxis], sr)
    assert snr_clean > snr_noisy


def test_stoi_approx_identical_signals_is_close_to_one() -> None:
    sr = 16_000
    clean = _make_voiced_signal(sr, 2.0)
    score = _stoi_approx(clean[:, np.newaxis], clean[:, np.newaxis], sr)
    assert 0.0 <= score <= 1.0
    assert score > 0.8


def test_stoi_approx_degraded_signal_scores_lower_than_clean() -> None:
    sr = 16_000
    clean = _make_voiced_signal(sr, 2.0)
    rng = np.random.default_rng(13)
    degraded = (clean + 0.5 * rng.standard_normal(len(clean))).astype(np.float64)
    score_clean = _stoi_approx(clean[:, np.newaxis], clean[:, np.newaxis], sr)
    score_degraded = _stoi_approx(clean[:, np.newaxis], degraded[:, np.newaxis], sr)
    assert score_clean > score_degraded


def test_mcd_db_identical_signals_is_near_zero() -> None:
    sr = 16_000
    clean = _make_voiced_signal(sr, 2.0)
    value = _mcd_db(clean[:, np.newaxis], clean[:, np.newaxis], sr)
    assert np.isfinite(value)
    assert value < 1.0


def test_mcd_db_degraded_signal_has_higher_mcd() -> None:
    sr = 16_000
    clean = _make_voiced_signal(sr, 2.0)
    rng = np.random.default_rng(21)
    degraded = np.fft.irfft(
        np.fft.rfft(clean) * rng.uniform(0.1, 2.0, len(np.fft.rfft(clean)))
    ).astype(np.float64)[: len(clean)]
    mcd_clean = _mcd_db(clean[:, np.newaxis], clean[:, np.newaxis], sr)
    mcd_degraded = _mcd_db(clean[:, np.newaxis], degraded[:, np.newaxis], sr)
    assert mcd_degraded > mcd_clean


def test_run_dereverb_benchmark_includes_all_v2_metrics() -> None:
    report = run_dereverb_benchmark(sr=16_000, duration_s=2.0, rt60=0.8)

    assert report["schema"] == "dereverb-benchmark-v2"
    assert "bark_snr_reverberant_db" in report
    assert "stoi_reverberant" in report
    assert "mcd_reverberant_db" in report

    for result in report["results"]:
        assert "bark_snr_db" in result
        assert "bark_snr_improvement_db" in result
        assert "stoi_approx" in result
        assert "stoi_improvement" in result
        assert "mcd_db" in result
        assert "mcd_improvement_db" in result
        assert np.isfinite(result["bark_snr_db"])
        assert np.isfinite(result["stoi_approx"])
        assert 0.0 <= float(result["stoi_approx"]) <= 1.0
        assert np.isfinite(result["mcd_db"])
        assert float(result["mcd_db"]) >= 0.0


def test_run_dereverb_benchmark_shows_improvement_over_reverberant() -> None:
    """Both modes should improve at least one perceptual metric vs the reverberant baseline."""
    report = run_dereverb_benchmark(sr=16_000, duration_s=3.0, rt60=1.5)
    for result in report["results"]:
        improved = (
            float(result["snr_improvement_db"]) > 0.0
            or float(result["bark_snr_improvement_db"]) > 0.0
            or float(result["stoi_improvement"]) > 0.0
            or float(result["mcd_improvement_db"]) > 0.0
        )
        assert improved, f"Mode {result['mode']} showed no improvement on any metric"


@pytest.mark.parametrize("rt60", [0.4, 1.2, 3.0])
def test_benchmark_metrics_are_finite_across_rt60(rt60: float) -> None:
    report = run_dereverb_benchmark(sr=16_000, duration_s=2.5, rt60=rt60)
    for result in report["results"]:
        for key in ("bark_snr_db", "stoi_approx", "mcd_db", "snr_db"):
            assert np.isfinite(result[key]), f"Non-finite {key} at RT60={rt60}"
