"""Peak-aligned reverberation metrics for impulse responses and program audio."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float64]
ReverbMetric = float | str

_EPS = 1e-24
_VALID_INPUT_KINDS = {"auto", "ir", "program"}


def extract_reverb_metrics(
    audio: AudioArray,
    sr: int,
    *,
    input_kind: str = "auto",
    direct_window_ms: float = 2.5,
) -> dict[str, ReverbMetric]:
    """Return broadband decay and room-acoustic estimates for one audio signal.

    The signal is aligned to its strongest sample before Schroeder integration.
    This produces conventional room-acoustic metrics for impulse responses and
    explicitly qualified estimates for reverberant program audio.
    """
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    if samples.ndim != 2:
        msg = f"Audio must have shape (samples, channels), received {samples.shape!r}."
        raise ValueError(msg)
    if int(sr) <= 0:
        msg = f"Sample rate must be positive, received {sr}."
        raise ValueError(msg)

    requested_kind = _normalize_input_kind(input_kind)
    if samples.shape[0] == 0:
        return _empty_metrics("insufficient_audio")

    absolute_peak = np.max(np.abs(samples), axis=1)
    peak_index = int(np.argmax(absolute_peak))
    peak_amplitude = float(absolute_peak[peak_index])
    if peak_amplitude <= 1e-15:
        metrics = _empty_metrics("silence")
        metrics["reverb_decay_start_seconds"] = float(peak_index / int(sr))
        return metrics

    frame_energy: npt.NDArray[np.float64] = np.asarray(
        np.sum(np.square(samples), axis=1, dtype=np.float64) / float(samples.shape[1]),
        dtype=np.float64,
    )
    tail_energy = np.asarray(frame_energy[peak_index:], dtype=np.float64)
    tail_times = np.arange(tail_energy.size, dtype=np.float64) / float(sr)
    decay_db = _schroeder_decay_db(tail_energy)

    fits = {
        "t30": _fit_decay(decay_db, tail_times, upper_db=-5.0, lower_db=-35.0),
        "t20": _fit_decay(decay_db, tail_times, upper_db=-5.0, lower_db=-25.0),
        "edt": _fit_decay(decay_db, tail_times, upper_db=0.0, lower_db=-10.0),
    }
    selected_name, selected_rt60, selected_r2 = _select_decay_fit(fits)

    direct_frames = max(1, round(float(sr) * float(direct_window_ms) / 1000.0))
    c50_db, d50_percent = _clarity_and_definition(tail_energy, sr=int(sr), split_ms=50.0)
    c80_db, _ = _clarity_and_definition(tail_energy, sr=int(sr), split_ms=80.0)
    drr_db = _energy_ratio_db(tail_energy[:direct_frames], tail_energy[direct_frames:])
    center_time_ms = _center_time_ms(tail_energy, int(sr))
    iacc_early = _early_iacc(samples[peak_index:, :], int(sr), split_ms=80.0)

    rms_amplitude = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    crest = peak_amplitude / max(rms_amplitude, 1e-15)
    direct_fraction = float(
        np.sum(tail_energy[:direct_frames], dtype=np.float64)
        / max(float(np.sum(tail_energy, dtype=np.float64)), _EPS)
    )
    resolved_kind = _resolve_input_kind(
        requested_kind,
        peak_index=peak_index,
        total_frames=samples.shape[0],
        sr=int(sr),
        crest_factor=crest,
        direct_fraction=direct_fraction,
    )

    noise_frames = max(1, min(tail_energy.size, round(0.1 * tail_energy.size)))
    noise_rms = float(np.sqrt(np.mean(tail_energy[-noise_frames:], dtype=np.float64)))
    noise_floor_dbfs = _amplitude_dbfs(noise_rms)
    peak_dbfs = _amplitude_dbfs(peak_amplitude)
    decay_range_db = float(np.clip(peak_dbfs - noise_floor_dbfs, 0.0, 240.0))
    confidence_score = _confidence_score(
        input_kind=resolved_kind,
        fit_name=selected_name,
        fit_r2=selected_r2,
        decay_range_db=decay_range_db,
        tail_duration_s=float(tail_energy.size / int(sr)),
        rt60_s=selected_rt60,
    )
    confidence = (
        "high" if confidence_score >= 0.80 else "medium" if confidence_score >= 0.55 else "low"
    )

    return {
        "reverb_analysis_basis": "peak_aligned_schroeder_edc",
        "reverb_input_kind": resolved_kind,
        "reverb_interpretation": (
            "room_acoustics_candidate"
            if resolved_kind == "impulse_response"
            else "program_audio_estimate"
        ),
        "reverb_decay_start_seconds": float(peak_index / int(sr)),
        "reverb_tail_duration_seconds": float(tail_energy.size / int(sr)),
        "reverb_rt60_seconds": selected_rt60,
        "reverb_rt60_fit": selected_name,
        "reverb_edt_seconds": fits["edt"][0],
        "reverb_t20_seconds": fits["t20"][0],
        "reverb_t30_seconds": fits["t30"][0],
        "reverb_decay_fit_r2": selected_r2,
        "reverb_decay_range_db": decay_range_db,
        "reverb_noise_floor_dbfs": noise_floor_dbfs,
        "reverb_c50_db": c50_db,
        "reverb_c80_db": c80_db,
        "reverb_d50_percent": d50_percent,
        "reverb_center_time_ms": center_time_ms,
        "reverb_direct_to_reverberant_db": drr_db,
        "reverb_iacc_early": iacc_early,
        "reverb_confidence_score": confidence_score,
        "reverb_confidence": confidence,
    }


def _normalize_input_kind(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "impulse": "ir",
        "impulse_response": "ir",
        "music": "program",
        "program_audio": "program",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_INPUT_KINDS:
        choices = ", ".join(sorted(_VALID_INPUT_KINDS))
        msg = f"--input-kind must be one of: {choices}; received {value!r}."
        raise ValueError(msg)
    return normalized


def _resolve_input_kind(
    requested: str,
    *,
    peak_index: int,
    total_frames: int,
    sr: int,
    crest_factor: float,
    direct_fraction: float,
) -> str:
    if requested == "ir":
        return "impulse_response"
    if requested == "program":
        return "program_audio"
    early_limit = max(round(0.1 * total_frames), round(0.1 * sr))
    looks_like_ir = (
        peak_index <= early_limit and crest_factor >= 6.0 and direct_fraction >= 0.005
    )
    return "impulse_response" if looks_like_ir else "program_audio"


def _schroeder_decay_db(energy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    integrated = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    integrated = np.maximum(integrated, _EPS)
    return np.asarray(10.0 * np.log10(integrated / integrated[0]), dtype=np.float64)


def _fit_decay(
    decay_db: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    *,
    upper_db: float,
    lower_db: float,
) -> tuple[float, float]:
    mask = (decay_db <= upper_db) & (decay_db >= lower_db)
    if int(np.count_nonzero(mask)) < 8:
        return 0.0, 0.0
    x = times[mask]
    y = decay_db[mask]
    if x.size < 2 or float(x[-1] - x[0]) <= 0.0:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or slope >= -1e-9:
        return 0.0, 0.0
    predicted = (slope * x) + intercept
    residual = float(np.sum(np.square(y - predicted), dtype=np.float64))
    centered = y - float(np.mean(y))
    total = float(np.sum(np.square(centered), dtype=np.float64))
    r2 = 0.0 if total <= _EPS else float(np.clip(1.0 - (residual / total), 0.0, 1.0))
    rt60 = float(np.clip(-60.0 / float(slope), 0.01, 3600.0))
    return rt60, r2


def _select_decay_fit(fits: dict[str, tuple[float, float]]) -> tuple[str, float, float]:
    for name in ("t30", "t20", "edt"):
        rt60, r2 = fits[name]
        if rt60 > 0.0 and r2 >= 0.80:
            return name, rt60, r2
    candidates = [(r2, name, rt60) for name, (rt60, r2) in fits.items() if rt60 > 0.0]
    if not candidates:
        return "none", 0.0, 0.0
    r2, name, rt60 = max(candidates)
    return name, rt60, r2


def _clarity_and_definition(
    energy: npt.NDArray[np.float64], *, sr: int, split_ms: float
) -> tuple[float, float]:
    split = max(1, min(energy.size, round(float(sr) * split_ms / 1000.0)))
    early = energy[:split]
    late = energy[split:]
    early_sum = float(np.sum(early, dtype=np.float64))
    late_sum = float(np.sum(late, dtype=np.float64))
    clarity = float(10.0 * np.log10((early_sum + _EPS) / (late_sum + _EPS)))
    definition = float(100.0 * early_sum / max(early_sum + late_sum, _EPS))
    return clarity, definition


def _energy_ratio_db(
    numerator: npt.NDArray[np.float64], denominator: npt.NDArray[np.float64]
) -> float:
    numerator_sum = float(np.sum(numerator, dtype=np.float64))
    denominator_sum = float(np.sum(denominator, dtype=np.float64))
    return float(10.0 * np.log10((numerator_sum + _EPS) / (denominator_sum + _EPS)))


def _center_time_ms(energy: npt.NDArray[np.float64], sr: int) -> float:
    total = float(np.sum(energy, dtype=np.float64))
    if total <= _EPS:
        return 0.0
    times = np.arange(energy.size, dtype=np.float64) / float(sr)
    return float(1000.0 * np.sum(times * energy, dtype=np.float64) / total)


def _early_iacc(audio: AudioArray, sr: int, *, split_ms: float) -> float:
    if audio.shape[1] < 2:
        return 1.0
    frames = max(2, min(audio.shape[0], round(float(sr) * split_ms / 1000.0)))
    left = np.asarray(audio[:frames, 0], dtype=np.float64)
    right = np.asarray(audio[:frames, 1], dtype=np.float64)
    max_lag = max(1, round(0.001 * float(sr)))
    correlations: list[float] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            lhs, rhs = left[-lag:], right[: frames + lag]
        elif lag > 0:
            lhs, rhs = left[: frames - lag], right[lag:]
        else:
            lhs, rhs = left, right
        denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denominator > 1e-15:
            correlations.append(abs(float(np.dot(lhs, rhs) / denominator)))
    return float(np.clip(max(correlations, default=0.0), 0.0, 1.0))


def _amplitude_dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(abs(float(value)), 1e-12)))


def _confidence_score(
    *,
    input_kind: str,
    fit_name: str,
    fit_r2: float,
    decay_range_db: float,
    tail_duration_s: float,
    rt60_s: float,
) -> float:
    window_score = {"t30": 1.0, "t20": 0.8, "edt": 0.5}.get(fit_name, 0.0)
    kind_score = 1.0 if input_kind == "impulse_response" else 0.45
    range_score = float(np.clip(decay_range_db / 45.0, 0.0, 1.0))
    duration_score = (
        0.0
        if rt60_s <= 0.0
        else float(np.clip(tail_duration_s / max(1.5 * rt60_s, 1e-9), 0.0, 1.0))
    )
    return float(
        np.clip(
            (0.40 * fit_r2)
            + (0.20 * window_score)
            + (0.20 * kind_score)
            + (0.10 * range_score)
            + (0.10 * duration_score),
            0.0,
            1.0,
        )
    )


def _empty_metrics(input_kind: str) -> dict[str, ReverbMetric]:
    return {
        "reverb_analysis_basis": "peak_aligned_schroeder_edc",
        "reverb_input_kind": input_kind,
        "reverb_interpretation": "insufficient_audio",
        "reverb_decay_start_seconds": 0.0,
        "reverb_tail_duration_seconds": 0.0,
        "reverb_rt60_seconds": 0.0,
        "reverb_rt60_fit": "none",
        "reverb_edt_seconds": 0.0,
        "reverb_t20_seconds": 0.0,
        "reverb_t30_seconds": 0.0,
        "reverb_decay_fit_r2": 0.0,
        "reverb_decay_range_db": 0.0,
        "reverb_noise_floor_dbfs": -240.0,
        "reverb_c50_db": 0.0,
        "reverb_c80_db": 0.0,
        "reverb_d50_percent": 0.0,
        "reverb_center_time_ms": 0.0,
        "reverb_direct_to_reverberant_db": 0.0,
        "reverb_iacc_early": 0.0,
        "reverb_confidence_score": 0.0,
        "reverb_confidence": "low",
    }
