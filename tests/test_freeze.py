from __future__ import annotations

import numpy as np

from verbx.core.freeze import create_crossfaded_loop, freeze_segment


def test_create_crossfaded_loop():
    sr = 1000
    # Create a 2 second stereo signal
    audio = np.zeros((2000, 2), dtype=np.float32)

    # segment: start=0.5s (500), end=1.5s (1500), length=1000.
    # We want beginning of segment to be 2.0, end of segment to be 1.0.
    audio[500:1000, :] = 2.0
    audio[1000:1500, :] = 1.0

    # xfade_ms = 100.0 -> xfade = 100 samples
    out = create_crossfaded_loop(audio, sr, start=0.5, end=1.5, xfade_ms=100.0)

    xfade = 100
    # out should be length (1000 - 100) = 900
    assert out.shape == (900, 2)

    theta = np.linspace(0.0, np.pi / 2.0, xfade, dtype=np.float32)
    fade_out = np.cos(theta)[:, np.newaxis]
    fade_in = np.sin(theta)[:, np.newaxis]

    expected_blended = np.tile((1.0 * fade_out) + (2.0 * fade_in), (1, 2))

    # check first xfade samples of the output
    np.testing.assert_allclose(out[:xfade, :], expected_blended, rtol=1e-5, atol=1e-5)

    # Check that after xfade, the signal behaves normally (is 2.0)
    np.testing.assert_array_equal(
        out[xfade:500, :], np.full((500 - xfade, 2), 2.0, dtype=np.float32)
    )

def test_create_crossfaded_loop_zero_xfade():
    sr = 1000
    audio = np.ones((2000, 2), dtype=np.float32)

    # segment: start=0.5s (500), end=1.5s (1500), length=1000.
    out = create_crossfaded_loop(audio, sr, start=0.5, end=1.5, xfade_ms=0.0)

    assert out.shape == (1000, 2)
    np.testing.assert_array_equal(out, np.ones((1000, 2), dtype=np.float32))

def test_create_crossfaded_loop_short_segment():
    sr = 1000
    audio = np.ones((2000, 2), dtype=np.float32)

    # 0.05 seconds = 50 samples. xfade_ms=100 -> 100 samples.
    # This should clamp xfade to 50 // 4 = 12 samples.
    out = create_crossfaded_loop(audio, sr, start=0.5, end=0.55, xfade_ms=100.0)

    assert out.shape == (50 - 12, 2)

def test_create_crossfaded_loop_invalid_bounds():
    sr = 1000
    audio = np.ones((2000, 2), dtype=np.float32)

    # end <= start
    out = create_crossfaded_loop(audio, sr, start=1.5, end=0.5)
    assert out.shape == (0,)


def test_freeze_passthrough():
    sr = 48000
    audio = np.random.randn(sr * 2, 2).astype(np.float32)
    out = freeze_segment(audio, sr, start=0.5, end=1.5, mode="passthrough")
    assert out.shape == audio.shape
    np.testing.assert_array_equal(out, audio)
    assert out is not audio


def test_freeze_invalid_bounds():
    sr = 48000
    audio = np.random.randn(sr * 2, 2).astype(np.float32)

    # None bounds
    out = freeze_segment(audio, sr, start=None, end=1.0)
    np.testing.assert_array_equal(out, audio)
    out = freeze_segment(audio, sr, start=0.0, end=None)
    np.testing.assert_array_equal(out, audio)
    out = freeze_segment(audio, sr, start=None, end=None)
    np.testing.assert_array_equal(out, audio)

    # Swapped bounds
    out = freeze_segment(audio, sr, start=1.0, end=0.5)
    np.testing.assert_array_equal(out, audio)

    # End <= start
    out = freeze_segment(audio, sr, start=1.0, end=1.0)
    np.testing.assert_array_equal(out, audio)

    # Very short segment (< 2 samples)
    out = freeze_segment(audio, sr, start=0.0, end=0.5 / sr)
    np.testing.assert_array_equal(out, audio)


def test_freeze_loop_tiling_no_xfade():
    sr = 1000
    audio = np.arange(1000, dtype=np.float32).reshape(-1, 1)  # 1 sec, mono
    # Freeze segment from 0.1 to 0.4s (100 to 400 samples -> 300 length loop)
    # Total input len = 1000. Loop tiled 4 times, sliced to 1000.
    out = freeze_segment(audio, sr, start=0.1, end=0.4, xfade_ms=0.0)

    assert out.shape == audio.shape
    segment = audio[100:400, :]
    expected = np.tile(segment, (4, 1))[:1000, :]
    np.testing.assert_array_equal(out, expected)


def test_freeze_crossfade_blending():
    sr = 1000
    # Create a 2 second stereo signal
    audio = np.zeros((2000, 2), dtype=np.float32)

    # segment: start=0.5s (500), end=1.5s (1500), length=1000.
    # We want beginning of segment to be 2.0, end of segment to be 1.0.
    audio[500:1000, :] = 2.0
    audio[1000:1500, :] = 1.0

    # xfade_ms = 100.0 -> xfade = 100 samples
    out = freeze_segment(audio, sr, start=0.5, end=1.5, xfade_ms=100.0)

    xfade = 100
    theta = np.linspace(0.0, np.pi / 2.0, xfade, dtype=np.float32)
    fade_out = np.cos(theta)[:, np.newaxis]
    fade_in = np.sin(theta)[:, np.newaxis]

    expected_blended = np.tile((1.0 * fade_out) + (2.0 * fade_in), (1, 2))

    # check first xfade samples of the output (since tiling means out starts with segment)
    np.testing.assert_allclose(out[:xfade, :], expected_blended, rtol=1e-5, atol=1e-5)

    # Check that after xfade, the signal behaves normally
    np.testing.assert_array_equal(
        out[xfade:500, :], np.full((500 - xfade, 2), 2.0, dtype=np.float32)
    )


def test_freeze_xfade_clamping():
    sr = 1000
    audio = np.zeros((2000, 2), dtype=np.float32)
    # segment: start=0.0, end=0.1 -> length=100
    # xfade_ms = 100.0 -> requested xfade=100. Max xfade = loop_len // 4 = 25.
    out = freeze_segment(audio, sr, start=0.0, end=0.1, xfade_ms=100.0)

    assert out.shape == audio.shape
