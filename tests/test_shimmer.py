from __future__ import annotations

from unittest.mock import patch

import numpy as np

from verbx.core.shimmer import ShimmerConfig, ShimmerProcessor, _bandlimit, _pitch_shift_audio


def _sine(samples: int, freq: float = 440.0, sr: int = 44100, channels: int = 1) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / sr
    mono = np.sin(2.0 * np.pi * freq * t) * 0.5
    return np.tile(mono[:, np.newaxis], (1, channels))


# ── 1. ShimmerConfig defaults are sane ──────────────────────────────────


def test_shimmer_config_defaults() -> None:
    cfg = ShimmerConfig()
    assert cfg.enabled is False
    assert cfg.semitones == 12.0
    assert 0.0 <= cfg.mix <= 1.0
    assert 0.0 <= cfg.feedback < 1.0
    assert cfg.highcut is not None and cfg.highcut > 0.0
    assert cfg.lowcut is not None and cfg.lowcut > 0.0
    assert cfg.lowcut < cfg.highcut


# ── 2. process() returns input unchanged when enabled=False ─────────────


def test_process_disabled_returns_input() -> None:
    cfg = ShimmerConfig(enabled=False, mix=0.5, semitones=12.0)
    proc = ShimmerProcessor(cfg)
    audio = _sine(1024, channels=2)
    out = proc.process(audio, sr=44100)
    np.testing.assert_array_equal(out, audio)


# ── 3. process() returns input unchanged when mix=0 ─────────────────────


def test_process_mix_zero_returns_input() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.0, semitones=12.0)
    proc = ShimmerProcessor(cfg)
    audio = _sine(1024, channels=1)
    out = proc.process(audio, sr=44100)
    np.testing.assert_array_equal(out, audio)


# ── 4. process() with shimmer enabled produces different output ─────────


def test_process_enabled_modifies_signal() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.5, semitones=12.0, feedback=0.3)
    proc = ShimmerProcessor(cfg)
    audio = _sine(2048, channels=1)
    out = proc.process(audio, sr=44100)
    assert out.shape == audio.shape
    assert not np.allclose(out, audio)


# ── 5. Feedback state accumulates across multiple calls ─────────────────


def test_feedback_accumulates_across_calls() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.4, semitones=7.0, feedback=0.5)
    proc = ShimmerProcessor(cfg)
    block = _sine(512, channels=1)

    out1 = proc.process(block, sr=44100)
    out2 = proc.process(block, sr=44100)

    # Second call includes feedback from the first, so outputs must differ.
    assert not np.allclose(out1, out2)


# ── 6. Feedback state resets when shape changes ─────────────────────────


def test_feedback_resets_on_shape_change() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.4, semitones=7.0, feedback=0.5)
    proc = ShimmerProcessor(cfg)

    # Process mono block to build up feedback state.
    mono = _sine(512, channels=1)
    proc.process(mono, sr=44100)
    assert proc._feedback_state is not None
    assert proc._feedback_state.shape == (512, 1)

    # Switch to stereo — feedback state must be reset.
    stereo = _sine(512, channels=2)
    proc.process(stereo, sr=44100)
    assert proc._feedback_state.shape == (512, 2)


# ── 7. Output shape matches input shape for mono and stereo ─────────────


def test_output_shape_mono() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.3)
    proc = ShimmerProcessor(cfg)
    audio = _sine(1024, channels=1)
    out = proc.process(audio, sr=44100)
    assert out.shape == audio.shape


def test_output_shape_stereo() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.3)
    proc = ShimmerProcessor(cfg)
    audio = _sine(1024, channels=2)
    out = proc.process(audio, sr=44100)
    assert out.shape == audio.shape


# ── 8. Output stays within soft limiter bounds ──────────────────────────


def test_output_within_limiter_bounds() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.9, semitones=12.0, feedback=0.9)
    proc = ShimmerProcessor(cfg)
    # Drive it hard with multiple passes to build feedback.
    audio = _sine(2048, channels=2) * 2.0
    for _ in range(5):
        out = proc.process(audio, sr=44100)
    # soft_limiter uses a knee, so values can slightly exceed 1.0 but should be bounded
    assert np.all(np.abs(out) <= 2.0)
    # Verify limiter is actually attenuating — output should be quieter than raw sum
    assert np.max(np.abs(out)) < np.max(np.abs(audio)) * 2.0


# ── 9. _pitch_shift_audio with semitones=0 returns copy ────────────────


def test_pitch_shift_zero_semitones_returns_copy() -> None:
    audio = _sine(512, channels=1)
    out = _pitch_shift_audio(audio, sr=44100, semitones=0.0)
    np.testing.assert_array_equal(out, audio)
    # Must be a copy, not a view.
    assert out is not audio


# ── 10. _pitch_shift_audio without librosa uses fallback ───────────────


def test_pitch_shift_fallback_without_librosa() -> None:
    audio = _sine(1024, channels=1)
    with patch("verbx.core.shimmer.librosa", None):
        out = _pitch_shift_audio(audio, sr=44100, semitones=12.0)
    assert out.shape == audio.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert not np.allclose(out, audio)


# ── 11. _bandlimit with None highcut/lowcut passes through unchanged ───


def test_bandlimit_none_cuts_passthrough() -> None:
    audio = _sine(1024, channels=1)
    out = _bandlimit(audio, sr=44100, lowcut=None, highcut=None)
    np.testing.assert_array_equal(out, audio)


# ── 12. _bandlimit with very short signals falls back to causal filter ──


def test_bandlimit_short_signal_causal_fallback() -> None:
    # Very short signal that will trip the sosfiltfilt padlen requirement.
    audio = np.array([[0.5], [-0.3]], dtype=np.float64)
    out = _bandlimit(audio, sr=44100, lowcut=300.0, highcut=8000.0)
    assert out.shape == audio.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


# ── 13. Zero-length input doesn't crash ─────────────────────────────────


def test_zero_length_input_no_crash() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.5, semitones=12.0)
    proc = ShimmerProcessor(cfg)
    audio = np.empty((0, 1), dtype=np.float64)
    out = proc.process(audio, sr=44100)
    assert out.shape[0] == 0


# ── 14. Single-sample input doesn't crash ───────────────────────────────


def test_single_sample_input_no_crash() -> None:
    cfg = ShimmerConfig(enabled=True, mix=0.5, semitones=12.0)
    proc = ShimmerProcessor(cfg)
    audio = np.array([[0.42]], dtype=np.float64)
    out = proc.process(audio, sr=44100)
    assert out.shape == (1, 1)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
