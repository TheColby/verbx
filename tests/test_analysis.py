from __future__ import annotations

import numpy as np

from verbx.analysis.analyzer import AudioAnalyzer
from verbx.analysis.framewise import framewise_metrics
from verbx.analysis.reverb_metrics import extract_reverb_metrics

EXPECTED_KEYS = {
    "duration",
    "rms",
    "peak",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "stereo_width",
    "dynamic_range",
    "crest_factor",
}


def test_analyzer_returns_expected_keys() -> None:
    analyzer = AudioAnalyzer()
    audio = np.zeros((1024, 2), dtype=np.float64)

    metrics = analyzer.analyze(audio, sr=48_000)

    assert EXPECTED_KEYS.issubset(metrics.keys())
    assert metrics["duration"] > 0.0


def test_analyzer_loudness_keys() -> None:
    analyzer = AudioAnalyzer()
    audio = np.random.default_rng(0).standard_normal((8192, 2)).astype(np.float64) * 0.01

    metrics = analyzer.analyze(audio, sr=48_000, include_loudness=True)

    assert "integrated_lufs" in metrics
    assert "true_peak_dbfs" in metrics
    assert "lra" in metrics


def test_analyzer_edr_keys() -> None:
    analyzer = AudioAnalyzer()
    sr = 48_000
    n = sr * 2
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t / 0.8).astype(np.float64)
    tone = np.sin(2.0 * np.pi * 330.0 * t).astype(np.float64)
    audio = (env * tone)[:, np.newaxis]

    metrics = analyzer.analyze(audio, sr=sr, include_edr=True)

    assert "edr_rt60_median_s" in metrics
    assert "edr_rt60_low_s" in metrics
    assert "edr_rt60_mid_s" in metrics
    assert "edr_rt60_high_s" in metrics
    assert "edr_valid_bins" in metrics
    assert metrics["edr_valid_bins"] >= 0.0


def test_reverb_metrics_recover_known_exponential_rt60() -> None:
    sr = 16_000
    expected_rt60 = 1.2
    t = np.arange(sr * 3, dtype=np.float64) / float(sr)
    decay = np.power(10.0, -3.0 * t / expected_rt60)
    audio = (decay * np.sin(2.0 * np.pi * 997.0 * t))[:, np.newaxis]
    audio[0, 0] = 1.0

    metrics = extract_reverb_metrics(audio, sr, input_kind="ir")

    assert metrics["reverb_input_kind"] == "impulse_response"
    assert metrics["reverb_rt60_fit"] == "t30"
    assert abs(float(metrics["reverb_rt60_seconds"]) - expected_rt60) < 0.02
    assert float(metrics["reverb_decay_fit_r2"]) > 0.999
    assert metrics["reverb_confidence"] == "high"


def test_reverb_metrics_align_decay_to_peak_after_leading_silence() -> None:
    sr = 8_000
    leading_frames = sr // 4
    t = np.arange(sr * 2, dtype=np.float64) / float(sr)
    tail = np.power(10.0, -3.0 * t / 0.8) * np.cos(2.0 * np.pi * 503.0 * t)
    audio = np.concatenate((np.zeros(leading_frames), tail))[:, np.newaxis]

    metrics = extract_reverb_metrics(audio, sr, input_kind="ir")

    assert abs(float(metrics["reverb_decay_start_seconds"]) - 0.25) < (1.0 / sr)
    assert abs(float(metrics["reverb_rt60_seconds"]) - 0.8) < 0.03


def test_reverb_metrics_reject_unknown_input_kind() -> None:
    audio = np.ones((512, 1), dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "--input-kind must be one of"):
        extract_reverb_metrics(audio, 48_000, input_kind="field-recording")


def test_framewise_modulation_metrics_present() -> None:
    sr = 8_000
    t = np.arange(sr, dtype=np.float64) / sr
    carrier = np.sin(2.0 * np.pi * 220.0 * t)
    mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * 2.0 * t))
    audio = (carrier * mod).astype(np.float64)[:, np.newaxis]

    rows = framewise_metrics(audio, sr=sr, frame_size=512, hop_size=128)
    assert len(rows) > 0

    row = rows[len(rows) // 2]
    assert "amp_mod_depth" in row
    assert "amp_mod_rate_hz" in row
    assert "amp_mod_confidence" in row
    assert "centroid_mod_depth" in row
    assert "centroid_mod_rate_hz" in row
    assert "centroid_mod_confidence" in row
    assert "channel_coherence" in row
    assert "coherence_drift" in row
    assert row["amp_mod_depth"] >= 0.0
    assert 0.0 <= row["amp_mod_confidence"] <= 1.0
    assert 0.0 <= row["centroid_mod_confidence"] <= 1.0


def test_analyzer_ambisonic_metrics_keys() -> None:
    analyzer = AudioAnalyzer()
    sr = 48_000
    n = 4096
    t = np.arange(n, dtype=np.float64) / np.float64(sr)
    w = (0.3 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float64)
    y = (0.2 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float64)
    z = (0.1 * np.sin(2.0 * np.pi * 90.0 * t)).astype(np.float64)
    x = (0.25 * np.sin(2.0 * np.pi * 160.0 * t)).astype(np.float64)
    foa = np.column_stack((w, y, z, x)).astype(np.float64)

    metrics = analyzer.analyze(
        foa,
        sr=sr,
        ambi_order=1,
        ambi_normalization="sn3d",
        ambi_channel_order="acn",
    )

    assert "ambi_order" in metrics
    assert "ambi_energy_omni_ratio" in metrics
    assert "ambi_directionality_stability" in metrics
    assert "ambi_front_ratio" in metrics
    assert 0.0 <= metrics["ambi_directionality_stability"] <= 1.0
