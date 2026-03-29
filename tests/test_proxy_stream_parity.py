"""Parity tests for the algorithmic proxy-stream path.

These tests verify that the proxy-stream execution path produces output that
is deterministically equivalent (within float32 tolerance) to the standard
in-memory render path when the newly broadened eligibility conditions apply:

- EQ (lowcut / highcut / tilt) applied as a post-stream pass
- Sample-rate conversion (target_sr != input_sr) handled via pre-resampling

Run as part of the normal test suite: no special hardware required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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


# ---------------------------------------------------------------------------
# EQ post-stream parity
# ---------------------------------------------------------------------------

def test_proxy_stream_lowcut_matches_standard_path(tmp_path: Path) -> None:
    """Proxy-stream with lowcut should produce output matching the standard path."""
    sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    base_cfg = dict(engine="algo", rt60=0.4, wet=0.7, dry=0.3, silent=True, progress=False)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, RenderConfig(**base_cfg, lowcut=80.0))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, RenderConfig(**base_cfg, lowcut=80.0, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    assert standard_audio.shape == proxy_audio.shape
    np.testing.assert_allclose(proxy_audio, standard_audio, atol=1e-3, rtol=0)


def test_proxy_stream_highcut_matches_standard_path(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    base_cfg = dict(engine="algo", rt60=0.4, wet=0.7, dry=0.3, silent=True, progress=False)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, RenderConfig(**base_cfg, highcut=6000.0))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, RenderConfig(**base_cfg, highcut=6000.0, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    assert standard_audio.shape == proxy_audio.shape
    np.testing.assert_allclose(proxy_audio, standard_audio, atol=1e-3, rtol=0)


def test_proxy_stream_tilt_matches_standard_path(tmp_path: Path) -> None:
    sr = 16_000
    infile = tmp_path / "in.wav"
    _write_sine(infile, sr)

    base_cfg = dict(engine="algo", rt60=0.4, wet=0.7, dry=0.3, silent=True, progress=False)

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, RenderConfig(**base_cfg, tilt=3.0))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, RenderConfig(**base_cfg, tilt=3.0, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    assert standard_audio.shape == proxy_audio.shape
    np.testing.assert_allclose(proxy_audio, standard_audio, atol=1e-3, rtol=0)


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

    base_cfg = dict(
        engine="algo", rt60=0.3, wet=0.6, dry=0.4,
        target_sr=target_sr, silent=True, progress=False,
    )

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, RenderConfig(**base_cfg))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, RenderConfig(**base_cfg, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    assert standard_audio.shape == proxy_audio.shape
    np.testing.assert_allclose(proxy_audio, standard_audio, atol=1e-3, rtol=0)


def test_proxy_stream_src_plus_eq_parity(tmp_path: Path) -> None:
    """SRC and EQ combined via proxy-stream must match the standard path."""
    input_sr = 44_100
    target_sr = 22_050
    infile = tmp_path / "in.wav"
    _write_sine(infile, input_sr, duration=0.5)

    base_cfg = dict(
        engine="algo", rt60=0.3, wet=0.6, dry=0.4,
        target_sr=target_sr, lowcut=100.0, tilt=1.5,
        silent=True, progress=False,
    )

    out_standard = tmp_path / "out_standard.wav"
    run_render_pipeline(infile, out_standard, RenderConfig(**base_cfg))

    out_proxy = tmp_path / "out_proxy.wav"
    run_render_pipeline(infile, out_proxy, RenderConfig(**base_cfg, algo_stream=True))

    standard_audio, _ = sf.read(str(out_standard), dtype="float64")
    proxy_audio, _ = sf.read(str(out_proxy), dtype="float64")

    assert standard_audio.shape == proxy_audio.shape
    np.testing.assert_allclose(proxy_audio, standard_audio, atol=1e-3, rtol=0)
