from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.ir.morph import (
    IRMorphConfig,
    generate_or_load_cached_blended_ir,
    generate_or_load_cached_morphed_ir,
    morph_ir_arrays,
    resolve_blend_mix_values,
)


def _impulse_with_echo(
    *,
    length: int,
    channels: int,
    echo_index: int,
    echo_gain: float,
) -> np.ndarray:
    x = np.zeros((length, channels), dtype=np.float64)
    x[0, :] = 1.0
    if 0 <= echo_index < length:
        x[echo_index, :] = np.float64(echo_gain)
    return x


def test_morph_ir_arrays_spectral_is_finite() -> None:
    sr = 12_000
    a = _impulse_with_echo(length=4096, channels=2, echo_index=180, echo_gain=0.45)
    b = _impulse_with_echo(length=4096, channels=2, echo_index=720, echo_gain=0.33)
    cfg = IRMorphConfig(mode="spectral", alpha=0.4, phase_coherence=0.8, spectral_smooth_bins=4)
    out, quality = morph_ir_arrays(a, b, sr=sr, config=cfg)
    assert out.shape == a.shape
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert isinstance(quality, dict)
    assert float(quality["spectral_distance_db"]) >= 0.0


def test_generate_or_load_cached_morphed_ir_reuses_cache(tmp_path: Path) -> None:
    sr = 16_000
    ir_a = _impulse_with_echo(length=2048, channels=1, echo_index=120, echo_gain=0.4)
    ir_b = _impulse_with_echo(length=2048, channels=1, echo_index=640, echo_gain=0.28)
    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    sf.write(str(path_a), ir_a, sr)
    sf.write(str(path_b), ir_b, sr)

    cfg = IRMorphConfig(mode="equal-power", alpha=0.35)
    _, _, _, wav_path_1, cache_hit_1 = generate_or_load_cached_morphed_ir(
        ir_a_path=path_a,
        ir_b_path=path_b,
        config=cfg,
        cache_dir=tmp_path / "cache",
        target_sr=sr,
    )
    _, _, _, wav_path_2, cache_hit_2 = generate_or_load_cached_morphed_ir(
        ir_a_path=path_a,
        ir_b_path=path_b,
        config=cfg,
        cache_dir=tmp_path / "cache",
        target_sr=sr,
    )
    assert wav_path_1 == wav_path_2
    assert cache_hit_1 is False
    assert cache_hit_2 is True


def test_generate_or_load_cached_blended_ir_supports_broadcast_mix(tmp_path: Path) -> None:
    sr = 12_000
    base = _impulse_with_echo(length=3072, channels=1, echo_index=90, echo_gain=0.5)
    b1 = _impulse_with_echo(length=3072, channels=1, echo_index=450, echo_gain=0.25)
    b2 = _impulse_with_echo(length=3072, channels=1, echo_index=900, echo_gain=0.20)
    p0 = tmp_path / "base.wav"
    p1 = tmp_path / "b1.wav"
    p2 = tmp_path / "b2.wav"
    sf.write(str(p0), base, sr)
    sf.write(str(p1), b1, sr)
    sf.write(str(p2), b2, sr)

    mix_values = resolve_blend_mix_values((0.6,), 2)
    assert mix_values == (0.6, 0.6)

    cfg = IRMorphConfig(mode="envelope-aware", alpha=0.5, early_ms=60.0)
    out, out_sr, meta, _, cache_hit = generate_or_load_cached_blended_ir(
        base_ir_path=p0,
        blend_ir_paths=(p1, p2),
        blend_mix=mix_values,
        config=cfg,
        cache_dir=tmp_path / "cache",
        target_sr=sr,
    )
    assert cache_hit is False
    assert out_sr == sr
    assert out.shape[1] == 1
    assert np.all(np.isfinite(out))
    assert meta["mode"] == "ir-blend"
    assert len(meta["weights"]) == 3
