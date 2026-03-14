from __future__ import annotations

import hashlib

import numpy as np

from verbx.core.feature_vector import (
    FEATURE_SCHEMA_VERSION,
    build_feature_vector_bus,
    render_feature_vector_lane,
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

    buses = {
        "speech": build_feature_vector_bus(
            audio=_speech_like(sr, seconds),
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        ),
        "music": build_feature_vector_bus(
            audio=_music_like(sr, seconds),
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        ),
        "percussive": build_feature_vector_bus(
            audio=_percussive_like(sr, seconds),
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        ),
    }

    signatures = {key: bus.signature for key, bus in buses.items()}
    expected = {
        "speech": "2b82d77c78dcfd22",
        "music": "455c35865d6c93cb",
        "percussive": "f68de02308c3b278",
    }
    assert signatures == expected


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

    signatures: dict[str, str] = {}
    for name, audio in {
        "speech": _speech_like(sr, seconds),
        "music": _music_like(sr, seconds),
        "percussive": _percussive_like(sr, seconds),
    }.items():
        bus = build_feature_vector_bus(
            audio=audio,
            sr=sr,
            ctrl_times=ctrl_times,
            frame_ms=40.0,
            hop_ms=20.0,
            requested_sources=requested,
        )
        trajectory = render_feature_vector_lane(lane, feature_bus=bus)
        signatures[name] = hashlib.sha256(
            np.asarray(trajectory, dtype=np.float64).tobytes(order="C")
        ).hexdigest()[:16]

    expected = {
        "speech": "d3bce8631b2badb1",
        "music": "eee5d08085e08161",
        "percussive": "a52c9ddd1d15eca1",
    }
    assert signatures == expected
