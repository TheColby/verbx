from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.core.pipeline import complete_stream_file_tail_to_zero, complete_tail_to_zero


def test_complete_tail_to_zero_trims_padding_and_finishes_cleanly() -> None:
    sr = 8_000
    hold_ms = 5.0
    hold = max(1, round(float(sr) * (hold_ms / 1000.0)))
    threshold = 1e-3

    head = np.exp(-np.arange(320, dtype=np.float64) / 60.0)[:, np.newaxis]
    padded = np.concatenate((head, np.zeros((sr * 2, 1), dtype=np.float64)), axis=0)

    out = complete_tail_to_zero(padded, sr, threshold=threshold, hold_ms=hold_ms, metric="peak")
    active = np.flatnonzero(np.max(np.abs(padded), axis=1) > threshold)
    assert active.size > 0
    expected_len = int(active[-1]) + 1 + hold
    assert out.shape[0] == expected_len
    assert np.max(np.abs(out[-hold:, :])) == 0.0
    # Last non-hold sample should land at zero to avoid stop-clicks.
    assert np.max(np.abs(out[expected_len - hold - 1, :])) == 0.0


def test_complete_tail_to_zero_keeps_length_for_silence_input() -> None:
    x = np.zeros((1_024, 2), dtype=np.float64)
    out = complete_tail_to_zero(x, 48_000, threshold=1e-6, hold_ms=5.0, metric="peak")
    assert out.shape == x.shape
    assert np.max(np.abs(out)) == 0.0


def test_complete_stream_file_tail_to_zero_trims_file(tmp_path: Path) -> None:
    sr = 16_000
    hold_ms = 4.0
    hold = max(1, round(float(sr) * (hold_ms / 1000.0)))
    threshold = 1e-3

    head = np.exp(-np.arange(600, dtype=np.float64) / 90.0)[:, np.newaxis]
    x = np.concatenate((head, np.zeros((sr, 1), dtype=np.float64)), axis=0)
    path = tmp_path / "tail_trim.wav"
    sf.write(str(path), x, sr, subtype="DOUBLE")

    out_frames = complete_stream_file_tail_to_zero(
        path,
        threshold=threshold,
        hold_ms=hold_ms,
        metric="peak",
    )
    y, out_sr = sf.read(str(path), always_2d=True, dtype="float64")
    assert out_sr == sr
    active = np.flatnonzero(np.max(np.abs(x), axis=1) > threshold)
    assert active.size > 0
    expected_len = int(active[-1]) + 1 + hold
    assert out_frames == expected_len
    assert y.shape[0] == expected_len
    assert np.max(np.abs(y[-hold:, :])) == 0.0
    assert np.max(np.abs(y[expected_len - hold - 1, :])) == 0.0
