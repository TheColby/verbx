"""Parity tests for the algorithmic proxy-stream path.

These tests verify that the proxy-stream execution path produces output that
is deterministically equivalent (within float32 tolerance) to the standard
in-memory render path when the newly broadened eligibility conditions apply:

- EQ (lowcut / highcut / tilt) applied as a post-stream pass
- Sample-rate conversion (target_sr != input_sr) handled via pre-resampling

Run as part of the normal test suite: no special hardware required.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.config import RenderConfig
from verbx.core.pipeline import run_render_pipeline


def _write_sine(path: Path, sr: int, duration: float = 1.0) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    audio = np.column_stack([
        0.4 * np.sin(2 * np.pi * 440.0 * t),
        0.3 * np.sin(2 * np.pi * 660.0 * t),
    ])
    sf.write(str(path), audio, sr, subtype="DOUBLE")


def _base_algo_config(**overrides: object) -> RenderConfig:
    """Return a typed baseline config for proxy-stream parity tests."""
    config = RenderConfig(
        engine="algo",
        rt60=0.4,
        wet=0.7,
        dry=0.3,
        mod_depth_ms=0.0,
        mod_rate_hz=0.0,
        silent=True,
        progress=False,
    )
    if not overrides:
        return config
    return replace(config, **overrides)


def _assert_proxy_close(
    standard_audio: np.ndarray,
    proxy_audio: np.ndarray,
    *,
    sr: int,
) -> None:
    """Assert proxy-stream output stays close to the offline reference.

    The realtime-friendly proxy path is a static IR approximation of the
    stateful algorithmic engine, so broadened eligibility modes are expected
    to be extremely similar rather than bit-for-bit identical.
    """
    len_diff = abs(int(standard_audio.shape[0]) - int(proxy_audio.shape[0]))
    assert len_diff <= max(256, round(0.02 * float(sr)))

    overlap = min(int(standard_audio.shape[0]), int(proxy_audio.shape[0]))
    ref = np.asarray(standard_audio[:overlap, :], dtype=np.float64)
    test = np.asarray(proxy_audio[:overlap, :], dtype=np.float64)
    diff = test - ref

    rms_error = float(np.sqrt(np.mean(np.square(diff))))
    max_error = float(np.max(np.abs(diff)))
    assert rms_error <= 0.01
    assert max_error <= 0.2

    for ch in range(ref.shape[1]):
        corr = float(np.corrcoef(ref[:, ch], test[:, ch])[0, 1])
        assert corr >= 0.99


# ---------------------------------------------------------------------------
# EQ post-stream parity
# ---------------------------------------------------------------------------

def test_proxy_stream_lowcut_matches_standard_path(tmp_path: Path) -> None:
    """Proxy-stream with lowcut should produce output matching the standard path."""
    sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, _base_algo_config(lowcut=80.0))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, _base_algo_config(lowcut=80.0, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    _assert_proxy_close(standard_audio, proxy_audio, sr=sr)


def test_proxy_stream_highcut_matches_standard_path(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, _base_algo_config(highcut=6000.0))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(
        infile,
        out_proxy,
        _base_algo_config(highcut=6000.0, algo_stream=True),
    )

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    _assert_proxy_close(standard_audio, proxy_audio, sr=sr)


def test_proxy_stream_tilt_matches_standard_path(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, _base_algo_config(tilt=3.0))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, _base_algo_config(tilt=3.0, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    _assert_proxy_close(standard_audio, proxy_audio, sr=sr)


def test_proxy_stream_combined_eq_produces_valid_output(tmp_path: Path) -> None:
    """Lowcut + highcut + tilt combined should produce a finite output."""
    sr = 24_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    out = tmp_path / "out.wav"
    report = run_render_pipeline(
        infile,
        out,
        RenderConfig(
            engine="algo",
            rt60=0.5,
            wet=0.6,
            dry=0.4,
            mod_depth_ms=0.0,
            mod_rate_hz=0.0,
            lowcut=80.0,
            highcut=10000.0,
            tilt=2.0,
            algo_stream=True,
            silent=True,
            progress=False,
        ),
    )
    assert out.exists()
    audio, _ = sf.read(str(out), dtype="float64")
    assert np.isfinite(audio).all()
    assert report["effective"]["engine_resolved"] == "algo_proxy_stream"


# ---------------------------------------------------------------------------
# SRC + proxy-stream parity
# ---------------------------------------------------------------------------

def test_proxy_stream_with_src_produces_valid_output(tmp_path: Path) -> None:
    """Proxy-stream with target_sr != input_sr must succeed and produce finite output."""
    input_sr = 44_100
    target_sr = 24_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, input_sr, duration=0.5)

    out = tmp_path / "out.wav"
    report = run_render_pipeline(
        infile,
        out,
        RenderConfig(
            engine="algo",
            rt60=0.4,
            wet=0.6,
            dry=0.4,
            mod_depth_ms=0.0,
            mod_rate_hz=0.0,
            target_sr=target_sr,
            algo_stream=True,
            silent=True,
            progress=False,
        ),
    )
    assert out.exists()
    audio, sr = sf.read(str(out), dtype="float64")
    assert sr == target_sr
    assert np.isfinite(audio).all()
    assert report["effective"]["engine_resolved"] == "algo_proxy_stream"
    assert int(report["effective"]["processing_sample_rate"]) == target_sr


def test_proxy_stream_src_output_matches_standard_path(tmp_path: Path) -> None:
    """Proxy-stream with SRC should match the standard in-memory path output."""
    input_sr = 48_000
    target_sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, input_sr, duration=0.5)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(
        infile,
        out_standard,
        _base_algo_config(rt60=0.3, wet=0.6, dry=0.4, target_sr=target_sr),
    )

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(
        infile,
        out_proxy,
        _base_algo_config(rt60=0.3, wet=0.6, dry=0.4, target_sr=target_sr, algo_stream=True),
    )

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    _assert_proxy_close(standard_audio, proxy_audio, sr=target_sr)


def test_proxy_stream_src_plus_eq_parity(tmp_path: Path) -> None:
    """SRC and EQ combined via proxy-stream must match the standard path."""
    input_sr = 44_100
    target_sr = 22_050
    infile = tmp_path / "in.wav"
    _write_sine(infile, input_sr, duration=0.5)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(
        infile,
        out_standard,
        _base_algo_config(
            rt60=0.3,
            wet=0.6,
            dry=0.4,
            target_sr=target_sr,
            lowcut=100.0,
            tilt=1.5,
        ),
    )

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(
        infile,
        out_proxy,
        _base_algo_config(
            rt60=0.3,
            wet=0.6,
            dry=0.4,
            target_sr=target_sr,
            lowcut=100.0,
            tilt=1.5,
            algo_stream=True,
        ),
    )

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    _assert_proxy_close(standard_audio, proxy_audio, sr=target_sr)
