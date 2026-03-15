from __future__ import annotations

import numpy as np

from verbx.analysis.fdn_qa import compute_fdn_qa_metrics, echo_density_curve
from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine


def _render_impulse(
    cfg: AlgoReverbConfig,
    *,
    sr: int = 48_000,
    seconds: float = 1.25,
) -> np.ndarray:
    length = max(256, int(sr * seconds))
    impulse = np.zeros((length, 1), dtype=np.float64)
    impulse[0, 0] = 1.0
    engine = AlgoReverbEngine(cfg)
    return engine.process(impulse, sr=sr)


def test_fdn_qa_metrics_are_finite() -> None:
    rendered = _render_impulse(
        AlgoReverbConfig(
            rt60=14.0,
            wet=1.0,
            dry=0.0,
            fdn_lines=8,
            fdn_matrix="hadamard",
            block_size=512,
        )
    )
    metrics = compute_fdn_qa_metrics(rendered, 48_000)
    assert np.isfinite(metrics.echo_density_start)
    assert np.isfinite(metrics.echo_density_end)
    assert np.isfinite(metrics.echo_density_growth)
    assert np.isfinite(metrics.ringing_index)
    assert 0.0 <= metrics.ringing_index <= 1.0
    assert 0.0 <= metrics.echo_density_start <= 1.0
    assert 0.0 <= metrics.echo_density_end <= 1.0


def test_sparse_high_order_mode_has_valid_qa_profile() -> None:
    baseline = _render_impulse(
        AlgoReverbConfig(
            rt60=16.0,
            wet=1.0,
            dry=0.0,
            fdn_lines=16,
            fdn_matrix="hadamard",
            block_size=512,
        )
    )
    sparse = _render_impulse(
        AlgoReverbConfig(
            rt60=16.0,
            wet=1.0,
            dry=0.0,
            fdn_lines=32,
            fdn_matrix="hadamard",
            fdn_sparse=True,
            fdn_sparse_degree=4,
            block_size=512,
        )
    )

    base_metrics = compute_fdn_qa_metrics(baseline, 48_000)
    sparse_metrics = compute_fdn_qa_metrics(sparse, 48_000)

    # Sparse high-order mode should not regress into pathological ringing.
    assert sparse_metrics.ringing_index <= min(1.0, base_metrics.ringing_index + 0.25)
    assert sparse_metrics.echo_density_end >= 0.0


def test_echo_density_curve_has_time_alignment() -> None:
    signal = np.zeros((4096,), dtype=np.float64)
    signal[0] = 1.0
    signal[256:2048] = 0.02

    times, density = echo_density_curve(signal, 48_000, frame_ms=8.0, hop_ms=4.0)
    assert times.shape[0] == density.shape[0]
    assert times.shape[0] > 0
    assert np.all(np.diff(times) > 0.0)
    assert np.all((density >= 0.0) & (density <= 1.0))
