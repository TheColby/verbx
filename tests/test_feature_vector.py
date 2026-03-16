from __future__ import annotations

import numpy as np

from verbx.core.feature_vector import (
    FEATURE_SCHEMA_VERSION,
    FeatureVectorBus,
    build_feature_vector_bus,
    render_feature_vector_lane,
    render_feature_vector_lane_from_values,
)


def _speech_like(sr: int, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / float(sr)
    f0 = 120.0 + (35.0 * np.sin(2.0 * np.pi * 0.9 * t))
    harmonic = np.sin(2.0 * np.pi * f0 * t) + (0.45 * np.sin(2.0 * np.pi * 2.0 * f0 * t))
    formant_env = 0.5 + (0.5 * np.sin(2.0 * np.pi * 3.0 * t))
    return (0.25 * harmonic * formant_env).astype(np.float64)[:, np.newaxis]


def _music_like(sr: int, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / float(sr)
    chord = (
        0.45 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 277.18 * t)
        + 0.30 * np.sin(2.0 * np.pi * 329.63 * t)
    )
    trem = 0.7 + (0.3 * np.sin(2.0 * np.pi * 2.3 * t))
    return (0.22 * chord * trem).astype(np.float64)[:, np.newaxis]


def _percussive_like(sr: int, seconds: float) -> np.ndarray:
    n = int(sr * seconds)
    x = np.zeros((n,), dtype=np.float64)
    positions = np.arange(0, n, max(1, int(0.125 * sr)), dtype=np.int64)
    for idx, start in enumerate(positions):
        length = min(n - int(start), int(0.04 * sr))
        if length <= 0:
            continue
        env = np.exp(-np.linspace(0.0, 5.0, length, dtype=np.float64))
        tone_hz = 140.0 + (40.0 * float((idx % 3) - 1))
        t = np.arange(length, dtype=np.float64) / float(sr)
        x[start : start + length] += 0.4 * env * np.sin(2.0 * np.pi * tone_hz * t)
    return x.astype(np.float64)[:, np.newaxis]


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _assert_between(value: float, low: float, high: float, *, label: str) -> None:
    assert low <= value <= high, (
        f"{label}: expected {low:.6f} <= value <= {high:.6f}, got {value:.6f}"
    )


def test_feature_vector_bus_exposes_schema_metadata_for_new_families() -> None:
    sr = 16_000
    audio = _speech_like(sr, 1.2)
    ctrl_times = np.linspace(0.0, (audio.shape[0] - 1) / sr, 128, dtype=np.float64)
    requested = {
        "mfcc_1_norm",
        "mfcc_2_norm",
        "formant_spread_norm",
        "formant_balance_norm",
        "rhythm_pulse",
        "rhythm_periodicity",
    }
    bus = build_feature_vector_bus(
        audio=audio,
        sr=sr,
        ctrl_times=ctrl_times,
        frame_ms=40.0,
        hop_ms=20.0,
        requested_sources=requested,
    )
    assert bus.schema_version == FEATURE_SCHEMA_VERSION
    for source in requested:
        assert source in bus.control_features
        assert source in bus.source_metadata
        family = str(bus.source_metadata[source].get("family", ""))
        assert family in {"mfcc", "formant", "rhythm"}


def test_feature_vector_golden_signatures_for_content_classes() -> None:
    sr = 16_000
    seconds = 2.0
    requested = {
        "loudness_norm",
        "mfcc_1_norm",
        "mfcc_2_norm",
        "formant_balance_norm",
        "rhythm_pulse",
        "rhythm_periodicity",
    }
    ctrl_times = np.linspace(0.0, seconds, 200, dtype=np.float64)

    buses: dict[str, FeatureVectorBus] = {}
    for name, audio in {
        "speech": _speech_like(sr, seconds),
        "music": _music_like(sr, seconds),
        "percussive": _percussive_like(sr, seconds),
    }.items():
        bus_a = build_feature_vector_bus(
            audio=audio,
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        )
        bus_b = build_feature_vector_bus(
            audio=audio,
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        )
        assert bus_a.signature == bus_b.signature
        assert len(bus_a.signature) == 16
        assert set(bus_a.signature).issubset(set("0123456789abcdef"))
        buses[name] = bus_a

    assert len({bus.signature for bus in buses.values()}) == 3

    # CPUs/FFT backends can disagree on tiny float crumbs; behavior envelopes should not.
    profiles: dict[str, dict[str, dict[str, float]]] = {
        name: {source: _stats(bus.control_features[source]) for source in requested}
        for name, bus in buses.items()
    }

    _assert_between(
        profiles["speech"]["formant_balance_norm"]["mean"],
        0.95,
        1.00,
        label="speech formant_balance_norm mean",
    )
    _assert_between(
        profiles["music"]["formant_balance_norm"]["mean"],
        0.94,
        1.00,
        label="music formant_balance_norm mean",
    )
    _assert_between(
        profiles["percussive"]["formant_balance_norm"]["mean"],
        0.55,
        0.90,
        label="percussive formant_balance_norm mean",
    )
    _assert_between(
        profiles["speech"]["mfcc_1_norm"]["mean"],
        0.50,
        0.57,
        label="speech mfcc_1_norm mean",
    )
    _assert_between(
        profiles["music"]["mfcc_1_norm"]["mean"],
        0.50,
        0.57,
        label="music mfcc_1_norm mean",
    )
    _assert_between(
        profiles["percussive"]["mfcc_1_norm"]["mean"],
        0.45,
        0.54,
        label="percussive mfcc_1_norm mean",
    )
    _assert_between(
        profiles["speech"]["rhythm_pulse"]["mean"],
        0.08,
        0.30,
        label="speech rhythm_pulse mean",
    )
    _assert_between(
        profiles["music"]["rhythm_pulse"]["mean"],
        0.08,
        0.32,
        label="music rhythm_pulse mean",
    )
    _assert_between(
        profiles["percussive"]["rhythm_pulse"]["mean"],
        0.03,
        0.20,
        label="percussive rhythm_pulse mean",
    )
    _assert_between(
        profiles["speech"]["rhythm_periodicity"]["mean"],
        0.60,
        0.90,
        label="speech rhythm_periodicity mean",
    )
    _assert_between(
        profiles["music"]["rhythm_periodicity"]["mean"],
        0.65,
        0.92,
        label="music rhythm_periodicity mean",
    )
    _assert_between(
        profiles["percussive"]["rhythm_periodicity"]["mean"],
        0.75,
        0.97,
        label="percussive rhythm_periodicity mean",
    )
    assert (
        profiles["music"]["rhythm_pulse"]["mean"]
        > profiles["speech"]["rhythm_pulse"]["mean"]
        > profiles["percussive"]["rhythm_pulse"]["mean"]
    )
    assert (
        profiles["percussive"]["rhythm_periodicity"]["mean"]
        > profiles["music"]["rhythm_periodicity"]["mean"]
        > profiles["speech"]["rhythm_periodicity"]["mean"]
    )


def test_feature_lane_fusion_trajectory_golden_signatures() -> None:
    sr = 16_000
    seconds = 2.0
    requested = {"loudness_norm", "rhythm_pulse", "formant_balance_norm"}
    ctrl_times = np.linspace(0.0, seconds, 200, dtype=np.float64)
    lane = {
        "target": "wet",
        "source": "formant_balance_norm",
        "weight": 0.7,
        "bias": 0.05,
        "curve": "smoothstep",
        "hysteresis_up": 0.01,
        "hysteresis_down": 0.01,
        "combine": "replace",
    }

    trajectories: dict[str, np.ndarray] = {}
    for name, audio in {
        "speech": _speech_like(sr, seconds),
        "music": _music_like(sr, seconds),
        "percussive": _percussive_like(sr, seconds),
    }.items():
        bus_a = build_feature_vector_bus(
            audio=audio,
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        )
        bus_b = build_feature_vector_bus(
            audio=audio,
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        )
        trajectory_a = np.asarray(
            render_feature_vector_lane(lane, feature_bus=bus_a),
            dtype=np.float64,
        )
        trajectory_b = np.asarray(
            render_feature_vector_lane(lane, feature_bus=bus_b),
            dtype=np.float64,
        )
        np.testing.assert_array_equal(trajectory_a, trajectory_b)
        trajectories[name] = trajectory_a

    stats = {name: _stats(values) for name, values in trajectories.items()}

    _assert_between(stats["speech"]["min"], 0.749, 0.751, label="speech trajectory min")
    _assert_between(stats["speech"]["max"], 0.749, 0.751, label="speech trajectory max")
    _assert_between(stats["speech"]["mean"], 0.749, 0.751, label="speech trajectory mean")
    _assert_between(stats["speech"]["std"], 0.0, 1e-7, label="speech trajectory std")

    _assert_between(stats["music"]["min"], 0.65, 0.75, label="music trajectory min")
    _assert_between(stats["music"]["max"], 0.749, 0.751, label="music trajectory max")
    _assert_between(stats["music"]["mean"], 0.73, 0.751, label="music trajectory mean")
    _assert_between(stats["music"]["std"], 0.0, 0.02, label="music trajectory std")

    _assert_between(stats["percussive"]["min"], 0.35, 0.50, label="percussive trajectory min")
    _assert_between(stats["percussive"]["max"], 0.72, 0.76, label="percussive trajectory max")
    _assert_between(stats["percussive"]["mean"], 0.52, 0.66, label="percussive trajectory mean")
    _assert_between(stats["percussive"]["std"], 0.08, 0.20, label="percussive trajectory std")

    assert stats["speech"]["mean"] > stats["percussive"]["mean"]
    assert stats["percussive"]["std"] > stats["music"]["std"] > stats["speech"]["std"]


def test_feature_vector_target_source_golden_curve_values() -> None:
    lane = {
        "target": "wet",
        "source": "target:rt60",
        "weight": 0.7,
        "bias": -0.1,
        "curve": "smoothstep",
        "curve_amount": 1.0,
        "hysteresis_up": 0.0,
        "hysteresis_down": 0.0,
        "combine": "replace",
    }
    source_values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    rendered = render_feature_vector_lane_from_values(
        lane,
        source_values=source_values,
        source_kind="target",
        sample_rate=48_000,
        source_range=(0.0, 1.0),
    )
    expected = np.asarray(
        [-0.1, 0.009375, 0.25, 0.490625, 0.6],
        dtype=np.float64,
    )
    np.testing.assert_allclose(rendered, expected, atol=1e-12)


def test_feature_vector_target_source_hysteresis_golden_values() -> None:
    lane = {
        "target": "wet",
        "source": "target:room-size",
        "weight": 1.0,
        "bias": 0.0,
        "curve": "linear",
        "curve_amount": 1.0,
        "hysteresis_up": 0.15,
        "hysteresis_down": 0.15,
        "combine": "replace",
    }
    source_values = np.asarray([0.0, 0.9, 0.8, 0.85, 0.6], dtype=np.float64)
    rendered = render_feature_vector_lane_from_values(
        lane,
        source_values=source_values,
        source_kind="target",
        sample_rate=48_000,
        source_range=(0.0, 1.0),
    )
    expected = np.asarray([0.0, 0.9, 0.9, 0.9, 0.6], dtype=np.float64)
    np.testing.assert_allclose(rendered, expected, atol=1e-12)
