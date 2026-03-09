"""Heuristics for scoring and selecting generated IR candidates.

The fitter derives coarse acoustic targets from source metrics, proposes a
deterministic candidate pool, and ranks candidates against those targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from verbx.analysis.features_spectral import spectral_centroid, spectral_flatness
from verbx.ir.generator import IRGenConfig, IRMode
from verbx.ir.metrics import analyze_ir

AudioArray = npt.NDArray[np.float64]
BatchPolicy = Literal["fifo", "shortest-first", "longest-first"]


@dataclass(slots=True)
class IRFitTarget:
    """Target profile inferred from source-audio analysis.

    These are perceptual proxies, not strict physical room parameters.
    """

    rt60_seconds: float
    early_late_ratio_db: float
    stereo_coherence: float
    centroid_hz: float
    flatness: float
    damping: float
    diffusion: float
    density: float


@dataclass(slots=True)
class IRFitCandidate:
    """Candidate IR configuration plus generation strategy marker."""

    config: IRGenConfig
    strategy: str


@dataclass(slots=True)
class IRFitScore:
    """Scoring details for IR candidate selection."""

    score: float
    rt60_error: float
    early_late_error: float
    coherence_error: float
    centroid_error: float
    flatness_error: float


def derive_ir_fit_target(metrics: dict[str, float], sr: int) -> IRFitTarget:
    """Infer desired IR behavior from source analysis metrics.

    The mapping emphasizes robust defaults over exact invertibility.
    """
    duration = float(metrics.get("duration", 10.0))
    dynamic_range_db = float(metrics.get("dynamic_range", 20.0))
    transient_density = float(metrics.get("transient_density", 0.01))
    width = float(metrics.get("stereo_width", 0.5))
    centroid_hz = float(metrics.get("spectral_centroid", 1200.0))
    flatness = float(metrics.get("spectral_flatness", 0.2))

    rt60_seconds = float(np.clip(duration * (1.1 + (dynamic_range_db / 70.0)), 6.0, 180.0))
    early_late_ratio_db = float(np.clip(2.0 - (transient_density * 160.0), -20.0, 8.0))
    stereo_coherence = float(np.clip(1.0 - (width * 0.7), -0.25, 0.99))
    centroid_target_hz = float(np.clip(centroid_hz * 0.9, 80.0, float(sr) * 0.45))

    damping = float(np.clip(0.35 + (0.45 * (1.0 - flatness)), 0.1, 0.95))
    diffusion = float(np.clip(0.3 + (0.7 * (1.0 - transient_density * 10.0)), 0.1, 1.0))
    density = float(np.clip(0.6 + (0.8 * (1.0 - flatness)), 0.2, 1.8))

    return IRFitTarget(
        rt60_seconds=rt60_seconds,
        early_late_ratio_db=early_late_ratio_db,
        stereo_coherence=stereo_coherence,
        centroid_hz=centroid_target_hz,
        flatness=float(np.clip(flatness, 0.0, 1.0)),
        damping=damping,
        diffusion=diffusion,
        density=density,
    )


def build_ir_fit_candidates(
    *,
    base_mode: IRMode,
    length: float,
    sr: int,
    channels: int,
    seed: int,
    pool_size: int,
    target: IRFitTarget,
    f0_hz: float | None = None,
    harmonic_targets_hz: tuple[float, ...] = (),
) -> list[IRFitCandidate]:
    """Create deterministic IR candidate pool from target profile.

    Candidate variation spans RT60, damping, diffusion, density, mode family,
    and selected mode-specific topology settings.
    """
    pool = max(1, pool_size)
    modes = _prioritized_modes(base_mode, target.flatness, target.early_late_ratio_db)

    rt60_scale = (0.72, 0.86, 1.0, 1.18, 1.38)
    damping_offsets = (-0.08, 0.0, 0.08)
    diffusion_offsets = (-0.15, 0.0, 0.12)
    density_offsets = (-0.18, 0.0, 0.15)

    out: list[IRFitCandidate] = []
    for idx in range(pool):
        mode = modes[idx % len(modes)]
        rt60 = float(np.clip(target.rt60_seconds * rt60_scale[idx % len(rt60_scale)], 4.0, 240.0))
        damping = float(
            np.clip(target.damping + damping_offsets[idx % len(damping_offsets)], 0.05, 0.98)
        )
        diffusion = float(
            np.clip(
                target.diffusion + diffusion_offsets[idx % len(diffusion_offsets)],
                0.05,
                1.0,
            )
        )
        density = float(
            np.clip(target.density + density_offsets[idx % len(density_offsets)], 0.05, 2.2)
        )
        modal_count = int(np.clip(28 + (idx % 5) * 10, 12, 96))
        fdn_lines = int(np.clip(8 + (idx % 4) * 2, 4, 16))

        cfg = IRGenConfig(
            mode=mode,
            length=length,
            sr=sr,
            channels=channels,
            seed=seed + idx,
            rt60=rt60,
            damping=damping,
            diffusion=diffusion,
            density=density,
            normalize="peak",
            peak_dbfs=-1.0,
            modal_count=modal_count,
            fdn_lines=fdn_lines,
            f0_hz=f0_hz,
            harmonic_targets_hz=harmonic_targets_hz,
        )
        out.append(IRFitCandidate(config=cfg, strategy=f"{mode}-heuristic-{idx + 1:02d}"))

    return out


def score_ir_candidate(
    *,
    ir_audio: AudioArray,
    sr: int,
    target: IRFitTarget,
) -> tuple[IRFitScore, dict[str, float]]:
    """Score candidate IR against inferred target profile.

    Returns both normalized score components and key measured metrics used for
    ranking/inspection.
    """
    metrics = analyze_ir(ir_audio, sr)
    rt60 = _metric_as_float(metrics, "rt60_estimate_seconds", 0.0)
    early_late = _metric_as_float(metrics, "early_late_ratio_db", 0.0)
    coherence = _metric_as_float(metrics, "stereo_coherence", 1.0)
    centroid = spectral_centroid(ir_audio, sr)
    flatness = spectral_flatness(ir_audio, sr)

    rt60_error = abs(rt60 - target.rt60_seconds) / max(target.rt60_seconds, 1.0)
    early_late_error = abs(early_late - target.early_late_ratio_db) / 24.0
    coherence_error = abs(coherence - target.stereo_coherence) / 2.0
    centroid_error = abs(np.log2((centroid + 1.0) / (target.centroid_hz + 1.0)))
    flatness_error = abs(flatness - target.flatness)

    # Weights prioritize decay behavior while keeping tone/spatial terms active.
    weighted = (
        (2.8 * rt60_error)
        + (1.8 * early_late_error)
        + (1.4 * coherence_error)
        + (1.1 * centroid_error)
        + (0.9 * flatness_error)
    )
    score = float(1.0 / (1.0 + weighted))

    fit_score = IRFitScore(
        score=score,
        rt60_error=float(rt60_error),
        early_late_error=float(early_late_error),
        coherence_error=float(coherence_error),
        centroid_error=float(centroid_error),
        flatness_error=float(flatness_error),
    )
    detail_metrics = {
        "rt60_estimate_seconds": rt60,
        "early_late_ratio_db": early_late,
        "stereo_coherence": coherence,
        "spectral_centroid_hz": float(centroid),
        "spectral_flatness": float(flatness),
    }
    return fit_score, detail_metrics


def _prioritized_modes(
    base_mode: IRMode, flatness: float, early_late_ratio_db: float
) -> list[IRMode]:
    """Rank likely IR modes based on broad source timbre/transient cues."""
    tonal = flatness < 0.15
    diffuse = flatness > 0.33
    transient_rich = early_late_ratio_db > 1.5

    ranked: list[IRMode] = [base_mode]
    if tonal:
        ranked.extend(["modal", "hybrid", "fdn", "stochastic"])
    elif diffuse:
        ranked.extend(["stochastic", "hybrid", "fdn", "modal"])
    elif transient_rich:
        ranked.extend(["hybrid", "fdn", "modal", "stochastic"])
    else:
        ranked.extend(["hybrid", "fdn", "stochastic", "modal"])

    deduped: list[IRMode] = []
    for mode in ranked:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def _metric_as_float(metrics: dict[str, float | list[float]], key: str, default: float) -> float:
    """Safely extract float metric values from mixed metric dictionaries."""
    value = metrics.get(key, default)
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return float(default)
