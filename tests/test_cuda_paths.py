"""Tests for CUDA and Apple Silicon (MPS) acceleration paths.

These tests are designed to:
- Skip gracefully when the required hardware/library is not available.
- Verify that GPU-accelerated paths produce numerically consistent output
  compared to the CPU reference path (within expected float32 tolerance).
- Confirm that device selection logic falls back correctly to CPU when
  requested hardware is unavailable.

Run manually or via the ``ci-gpu.yml`` workflow dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------
_cupy_available = False
try:
    import cupy as cp  # noqa: F401
    _cupy_available = True
except ImportError:
    pass

_mps_available = False
try:
    import torch
    _mps_available = torch.backends.mps.is_available()
except ImportError:
    pass

requires_cuda = pytest.mark.skipif(not _cupy_available, reason="CuPy not installed")
requires_mps = pytest.mark.skipif(not _mps_available, reason="MPS not available")
requires_any_gpu = pytest.mark.skipif(
    not (_cupy_available or _mps_available),
    reason="No GPU backend (CuPy/MPS) available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(sr: int = 24000, freq: float = 440.0, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    return np.column_stack([np.sin(2 * np.pi * freq * t)] * 2)


# ---------------------------------------------------------------------------
# Device selection / fallback tests (always run — no GPU required)
# ---------------------------------------------------------------------------

def test_device_auto_resolves_without_crash(tmp_path: pytest.TempPathFactory) -> None:
    """device='auto' must not raise even when no GPU is present."""
    from verbx.config import RenderConfig

    cfg = RenderConfig(engine="algo", rt60=0.5, device="auto")
    assert cfg.device == "auto"


def test_device_cuda_field_accepted() -> None:
    """RenderConfig must accept device='cuda' without validation error."""
    from verbx.config import RenderConfig

    cfg = RenderConfig(engine="algo", rt60=0.5, device="cuda")
    assert cfg.device == "cuda"


def test_device_mps_field_accepted() -> None:
    """RenderConfig must accept device='mps' without validation error."""
    from verbx.config import RenderConfig

    cfg = RenderConfig(engine="algo", rt60=0.5, device="mps")
    assert cfg.device == "mps"


def test_algo_gpu_proxy_flag_accepted() -> None:
    """RenderConfig must accept algo_gpu_proxy=True without error."""
    from verbx.config import RenderConfig

    cfg = RenderConfig(engine="algo", rt60=0.5, algo_gpu_proxy=True)
    assert cfg.algo_gpu_proxy is True


def test_render_with_device_cuda_falls_back_to_cpu(tmp_path: pytest.TempPathFactory) -> None:
    """Rendering with device='cuda' must succeed (falling back to CPU) when no GPU present."""
    from pathlib import Path
    from verbx.config import RenderConfig
    from verbx.core.pipeline import run_render_pipeline

    sr = 16000
    audio = _make_sine(sr=sr, duration=0.5)
    infile = Path(tmp_path) / "in.wav"
    outfile = Path(tmp_path) / "out.wav"
    sf.write(str(infile), audio, sr, subtype="FLOAT")

    # Should not raise — falls back to CPU when CUDA unavailable
    report = run_render_pipeline(
        infile, outfile,
        RenderConfig(engine="algo", rt60=0.3, wet=0.5, device="cuda"),
    )
    assert outfile.exists()
    assert report["engine"] in {"algo", "conv"}


def test_render_with_device_mps_falls_back_to_cpu(tmp_path: pytest.TempPathFactory) -> None:
    """Rendering with device='mps' must succeed (falling back to CPU) when MPS unavailable."""
    from pathlib import Path
    from verbx.config import RenderConfig
    from verbx.core.pipeline import run_render_pipeline

    sr = 16000
    audio = _make_sine(sr=sr, duration=0.5)
    infile = Path(tmp_path) / "in.wav"
    outfile = Path(tmp_path) / "out.wav"
    sf.write(str(infile), audio, sr, subtype="FLOAT")

    report = run_render_pipeline(
        infile, outfile,
        RenderConfig(engine="algo", rt60=0.3, wet=0.5, device="mps"),
    )
    assert outfile.exists()
    assert report["engine"] in {"algo", "conv"}


# ---------------------------------------------------------------------------
# CUDA-specific tests (skipped without CuPy)
# ---------------------------------------------------------------------------

@requires_cuda
def test_cupy_import_and_basic_array() -> None:
    """Smoke test: CuPy is importable and basic array ops work."""
    import cupy as cp  # noqa: F811

    a = cp.array([1.0, 2.0, 3.0], dtype=cp.float64)
    assert float(cp.sum(a)) == pytest.approx(6.0)


@requires_cuda
def test_convolution_cuda_matches_cpu(tmp_path: pytest.TempPathFactory) -> None:
    """CUDA convolution output must match CPU output within 1e-4 tolerance."""
    from pathlib import Path
    from verbx.config import RenderConfig
    from verbx.core.pipeline import run_render_pipeline

    sr = 16000
    audio = _make_sine(sr=sr, duration=1.0)
    infile = Path(tmp_path) / "in.wav"
    out_cpu = Path(tmp_path) / "out_cpu.wav"
    out_gpu = Path(tmp_path) / "out_gpu.wav"
    sf.write(str(infile), audio, sr, subtype="FLOAT")

    base_cfg = dict(engine="algo", rt60=0.5, wet=0.6, dry=0.4, progress=False)
    run_render_pipeline(infile, out_cpu, RenderConfig(**base_cfg, device="cpu"))
    run_render_pipeline(infile, out_gpu, RenderConfig(**base_cfg, device="cuda"))

    cpu_audio, _ = sf.read(str(out_cpu), dtype="float64")
    gpu_audio, _ = sf.read(str(out_gpu), dtype="float64")

    assert cpu_audio.shape == gpu_audio.shape
    np.testing.assert_allclose(gpu_audio, cpu_audio, atol=1e-4, rtol=0)


@requires_cuda
def test_algo_gpu_proxy_produces_output(tmp_path: pytest.TempPathFactory) -> None:
    """algo_gpu_proxy=True must produce a valid output file without error."""
    from pathlib import Path
    from verbx.config import RenderConfig
    from verbx.core.pipeline import run_render_pipeline

    sr = 24000
    audio = _make_sine(sr=sr, duration=0.5)
    infile = Path(tmp_path) / "in.wav"
    outfile = Path(tmp_path) / "out.wav"
    sf.write(str(infile), audio, sr, subtype="FLOAT")

    run_render_pipeline(
        infile, outfile,
        RenderConfig(engine="algo", rt60=1.0, algo_stream=True, algo_gpu_proxy=True, progress=False),
    )
    assert outfile.exists()
    assert outfile.stat().st_size > 0


# ---------------------------------------------------------------------------
# Apple Silicon / MPS tests (skipped without MPS)
# ---------------------------------------------------------------------------

@requires_mps
def test_mps_device_available() -> None:
    """Confirm MPS is flagged available by PyTorch."""
    import torch
    assert torch.backends.mps.is_available()


@requires_mps
def test_render_with_mps_device_produces_output(tmp_path: pytest.TempPathFactory) -> None:
    """Rendering with device='mps' must produce a valid output on Apple Silicon."""
    from pathlib import Path
    from verbx.config import RenderConfig
    from verbx.core.pipeline import run_render_pipeline

    sr = 24000
    audio = _make_sine(sr=sr, duration=0.5)
    infile = Path(tmp_path) / "in.wav"
    outfile = Path(tmp_path) / "out.wav"
    sf.write(str(infile), audio, sr, subtype="FLOAT")

    run_render_pipeline(
        infile, outfile,
        RenderConfig(engine="algo", rt60=0.5, device="mps", progress=False),
    )
    assert outfile.exists()
    assert outfile.stat().st_size > 0
