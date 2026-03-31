"""Room size estimation from audio analysis.

Estimates room dimensions, volume, absorption, and acoustic class from an
audio recording (reverberant program audio or impulse response) using
Sabine/Eyring reverberation formulas and EDR-derived RT60 measurements.

The estimator is designed for creative and diagnostic use — it provides
*practical* estimates with explicit confidence ratings, not laboratory-grade
measurements.  For best results, analyse a reverberant recording or a rendered
IR rather than a dry/anechoic source.

Key formulas used
-----------------
Sabine:
    RT60 = 0.161 * V / A
    → V  = RT60 * A / 0.161

Eyring (more accurate at mean absorption > 0.3):
    RT60 = -0.161 * V / (S * ln(1 - α))
    → α  = 1 - exp(-0.161 * V / (S * RT60))

Critical distance (omnidirectional source):
    Dc = sqrt(A / (16π))

Where V = volume (m³), A = total absorption (m²·Sabine = α·S),
S = total surface area (m²), α = mean absorption coefficient [0, 1].

Room dimension aspect ratios
----------------------------
Dimensions are extracted from the estimated volume by assuming a standard
rectangular room with the aspect ratio W : D : H = 1 : 1.25 : 0.62.
This approximates Bolt's optimum ratio for small-room acoustics.

Pipeline stages
---------------
The estimation process is split into independently callable stages so that
each can be tested, replaced, or called in isolation:

1. ``extract_edr_rt60(audio, sr)``     — derive RT60 values from audio via EDR
2. ``infer_absorption(...)``           — estimate mean α from spectral RT60 shape
3. ``estimate_volume(rt60, alpha)``    — Sabine + Eyring + blended volume
4. ``project_dimensions(volume_m3)``  — W / D / H / SA from aspect ratios
5. ``score_confidence(...)``           — composite quality score for the estimate
6. ``classify_room(volume_m3, rt60)`` — qualitative size label

``estimate_room_size`` is the high-level orchestrator that calls all stages.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from verbx.analysis.edr import edr_summary

AudioArray = npt.NDArray[np.float64]

# Speed of sound (m/s) at 20 °C
_C = 343.0

# Sabine proportionality constant  (s·m⁻¹ = 0.161)
_SABINE_K = 24.0 * math.log(10.0) / _C  # ≈ 0.1611

# Room dimension aspect ratios W : D : H
_ASPECT_W = 1.00
_ASPECT_D = 1.25
_ASPECT_H = 0.62

# Volume multiplier for the chosen aspect ratios
_ASPECT_VOLUME = _ASPECT_W * _ASPECT_D * _ASPECT_H  # ≈ 0.775

# Surface-area-to-(W²) ratio for the chosen aspect ratios
_ASPECT_SA_COEF = 2.0 * (
    _ASPECT_W * _ASPECT_D
    + _ASPECT_W * _ASPECT_H
    + _ASPECT_D * _ASPECT_H
)  # ≈ 4.85  → S = _ASPECT_SA_COEF * W²


# ---------------------------------------------------------------------------
# Public pipeline stage 1 — EDR extraction
# ---------------------------------------------------------------------------


def extract_edr_rt60(audio: AudioArray, sr: int) -> dict[str, float]:
    """Extract per-band and summary RT60 values from audio via EDR analysis.

    Parameters
    ----------
    audio:
        Input signal ``(samples, channels)``.
    sr:
        Sample rate in Hz.

    Returns
    -------
    dict
        Subset of ``edr_summary`` keys relevant to room sizing:
        ``rt60_s``, ``rt60_low_s``, ``rt60_mid_s``, ``rt60_high_s``,
        ``valid_bins``, ``duration_s``, ``silence_ratio``.
    """
    edr = edr_summary(audio, sr)
    rt60_median = float(edr.get("edr_rt60_median_s", 0.0))
    rt60_low = float(edr.get("edr_rt60_low_s", 0.0))
    rt60_mid = float(edr.get("edr_rt60_mid_s", 0.0))
    rt60_high = float(edr.get("edr_rt60_high_s", 0.0))
    valid_bins = float(edr.get("edr_valid_bins", 0.0))

    # Use mid-band RT60 as the primary sizing input; fall back to median
    rt60 = rt60_mid if rt60_mid > 0.0 else rt60_median

    # Derive basic signal properties needed for confidence scoring
    mono = np.mean(audio, axis=1)
    duration_s = float(audio.shape[0]) / max(1.0, float(sr))
    silence_ratio = (
        float(np.mean(np.abs(mono) < 10.0 ** (-50.0 / 20.0))) if mono.size > 0 else 1.0
    )

    return {
        "rt60_s": rt60,
        "rt60_low_s": rt60_low,
        "rt60_mid_s": rt60_mid,
        "rt60_high_s": rt60_high,
        "valid_bins": valid_bins,
        "duration_s": duration_s,
        "silence_ratio": silence_ratio,
    }


# ---------------------------------------------------------------------------
# Public pipeline stage 2 — absorption inference
# ---------------------------------------------------------------------------


def infer_absorption(
    rt60_low: float,
    rt60_mid: float,
    rt60_high: float,
    *,
    prior: float | None = None,
) -> float:
    """Estimate mean absorption coefficient from spectral RT60 shape.

    Hard surfaces (concrete, stone, tile) have strongly frequency-dependent
    absorption — RT60 falls steeply from low to high frequencies.
    Acoustic treatment flattens the decay curve.

    Parameters
    ----------
    rt60_low / rt60_mid / rt60_high:
        Per-band RT60 estimates (s).  Any value ≤ 0 triggers a moderate prior.
    prior:
        If given, return ``clip(prior, 0.01, 0.99)`` directly (bypass inference).

    Returns
    -------
    float
        Estimated mean absorption coefficient in [0.01, 0.99].
    """
    if prior is not None:
        return float(np.clip(prior, 0.01, 0.99))

    if rt60_low <= 0.0 or rt60_high <= 0.0 or rt60_mid <= 0.0:
        return 0.20  # moderate prior when band info is unavailable

    # High/low RT60 ratio: close to 1 means spectrally flat decay
    hf_lf_ratio = float(np.clip(rt60_high / max(rt60_low, 1e-6), 0.0, 2.0))

    if hf_lf_ratio < 0.35:
        alpha = 0.10  # very hard (stone/concrete)
    elif hf_lf_ratio < 0.55:
        alpha = 0.15  # hard (plaster/brick)
    elif hf_lf_ratio < 0.70:
        alpha = 0.22  # light treatment (typical office/home)
    elif hf_lf_ratio < 0.85:
        alpha = 0.30  # moderate treatment (studio/meeting room)
    else:
        alpha = 0.42  # heavy treatment (broadcast/mastering studio)

    return float(alpha)


# ---------------------------------------------------------------------------
# Public pipeline stage 3 — volume estimation
# ---------------------------------------------------------------------------


def estimate_volume(rt60: float, alpha: float) -> dict[str, float]:
    """Estimate room volume using Sabine and Eyring equations.

    Derives both a Sabine estimate and an Eyring estimate (one refinement
    pass), then blends toward Eyring as absorption increases.

    Parameters
    ----------
    rt60:
        Reverberation time (s).  Must be > 0.
    alpha:
        Mean absorption coefficient [0.01, 0.99].

    Returns
    -------
    dict with keys:
        ``sabine_m3``, ``eyring_m3``, ``primary_m3``, ``low_m3``, ``high_m3``.
    """
    if rt60 <= 0.0:
        return {"sabine_m3": 0.0, "eyring_m3": 0.0, "primary_m3": 0.0,
                "low_m3": 0.0, "high_m3": 0.0}

    alpha_c = float(np.clip(alpha, 0.01, 0.99))

    # Sabine: solve for V given aspect-ratio geometry
    # V = RT60 * α * S / K, with S = _ASPECT_SA_COEF * W² and V = _ASPECT_VOLUME * W³
    # → V^(1/3) = RT60 * α * (_ASPECT_SA_COEF / _ASPECT_VOLUME^(2/3)) / K
    sa_v23 = _ASPECT_SA_COEF / (_ASPECT_VOLUME ** (2.0 / 3.0))
    v_cube_root = (rt60 * alpha_c * sa_v23) / _SABINE_K
    v_sabine = max(0.0, v_cube_root ** 3.0)

    # Eyring: one refinement pass using Sabine V as seed
    w_est = max(1e-3, (v_sabine / _ASPECT_VOLUME) ** (1.0 / 3.0))
    s_est = _ASPECT_SA_COEF * w_est * w_est
    v_eyring = _v_eyring(rt60, s_est, alpha_c)
    w_ey = max(1e-3, (v_eyring / _ASPECT_VOLUME) ** (1.0 / 3.0))
    s_ey = _ASPECT_SA_COEF * w_ey * w_ey
    v_eyring = _v_eyring(rt60, s_ey, alpha_c)

    # Blend: prefer Eyring at high absorption, Sabine at low
    blend = float(np.clip((alpha_c - 0.1) / 0.3, 0.0, 1.0))
    v_primary = (1.0 - blend) * v_sabine + blend * v_eyring

    return {
        "sabine_m3": v_sabine,
        "eyring_m3": v_eyring,
        "primary_m3": v_primary,
        "low_m3": v_primary * 0.70,
        "high_m3": v_primary * 1.30,
    }


# ---------------------------------------------------------------------------
# Public pipeline stage 4 — dimension projection
# ---------------------------------------------------------------------------


def project_dimensions(volume_m3: float) -> dict[str, float]:
    """Derive rectangular room dimensions from volume using Bolt aspect ratios.

    Applies W : D : H = 1 : 1.25 : 0.62 to extract width, depth, height,
    total surface area, and critical distance.

    Parameters
    ----------
    volume_m3:
        Room volume estimate (m³).

    Returns
    -------
    dict with keys:
        ``width_m``, ``depth_m``, ``height_m``, ``surface_area_m2``,
        ``mean_absorption``, ``critical_distance_m``.
        (``mean_absorption`` and ``critical_distance_m`` require a separate
        absorption input; they are set to 0.0 here and filled by the
        orchestrator.)
    """
    w = max(0.1, (max(0.0, volume_m3) / _ASPECT_VOLUME) ** (1.0 / 3.0))
    d = _ASPECT_D * w
    h = _ASPECT_H * w
    sa = _ASPECT_SA_COEF * w * w
    return {
        "width_m": w,
        "depth_m": d,
        "height_m": h,
        "surface_area_m2": sa,
    }


# ---------------------------------------------------------------------------
# Public pipeline stage 5 — confidence scoring
# ---------------------------------------------------------------------------


def score_confidence(
    *,
    rt60: float,
    valid_bins: float,
    duration_s: float,
    silence_ratio: float,
    rt60_low: float,
    rt60_high: float,
) -> tuple[float, str]:
    """Return ``(score, label)`` confidence pair in [0, 1].

    Scores are additive across five signal-quality signals:
    RT60 magnitude, valid EDR frequency bins, signal duration,
    non-silence ratio, and spectral consistency.

    Labels: ``"high"`` (≥0.70), ``"medium"`` (≥0.40), ``"low"`` (<0.40).
    """
    s = 0.0

    if rt60 > 0.05:
        s += 0.20
    if rt60 > 0.20:
        s += 0.10

    if valid_bins >= 30:
        s += 0.25
    elif valid_bins >= 10:
        s += 0.15
    elif valid_bins >= 3:
        s += 0.05

    if duration_s >= 3.0:
        s += 0.20
    elif duration_s >= 1.5:
        s += 0.12
    elif duration_s >= 0.5:
        s += 0.05

    if silence_ratio < 0.3:
        s += 0.15
    elif silence_ratio < 0.6:
        s += 0.08

    if rt60_low > 0.0 and rt60_high > 0.0:
        s += 0.10

    score = float(np.clip(s, 0.0, 1.0))
    label = "high" if score >= 0.70 else ("medium" if score >= 0.40 else "low")
    return score, label


# ---------------------------------------------------------------------------
# Public pipeline stage 6 — classification
# ---------------------------------------------------------------------------


def classify_room(volume_m3: float, rt60_s: float) -> str:
    """Return a qualitative room-size label based on volume and RT60.

    Thresholds
    ----------
    ``closet``     V <     8 m³ or RT60 < 0.12 s
    ``small``      V <    50 m³ or RT60 < 0.35 s
    ``medium``     V <   250 m³ or RT60 < 0.80 s
    ``large``      V < 1 500 m³ or RT60 < 2.0  s
    ``very_large`` V <10 000 m³ or RT60 < 5.0  s
    ``cathedral``  everything larger
    """
    if volume_m3 < 8.0 or rt60_s < 0.12:
        return "closet"
    if volume_m3 < 50.0 or rt60_s < 0.35:
        return "small"
    if volume_m3 < 250.0 or rt60_s < 0.80:
        return "medium"
    if volume_m3 < 1500.0 or rt60_s < 2.0:
        return "large"
    if volume_m3 < 10_000.0 or rt60_s < 5.0:
        return "very_large"
    return "cathedral"


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def estimate_room_size(
    audio: AudioArray,
    sr: int,
    *,
    prior_absorption: float | None = None,
) -> dict[str, object]:
    """Estimate room acoustic properties from a reverberant audio recording.

    Calls the six pipeline stages in sequence:
    ``extract_edr_rt60`` → ``infer_absorption`` → ``estimate_volume``
    → ``project_dimensions`` → ``score_confidence`` → ``classify_room``.

    Parameters
    ----------
    audio:
        Input signal in ``(samples, channels)`` format.  Works best with
        reverberant recordings or rendered impulse responses.  Dry/anechoic
        material will produce low-confidence estimates.
    sr:
        Sample rate in Hz.
    prior_absorption:
        Optional prior mean absorption coefficient [0.01, 0.99].  When
        ``None`` (default) the mean absorption is inferred from the ratio of
        high-frequency to low-frequency RT60 values.

    Returns
    -------
    dict
        Flat dictionary with ``room_*`` prefixed keys (all JSON-serializable).
        Key classes:

        - ``room_rt60_s`` — best RT60 estimate used as sizing input (s)
        - ``room_volume_m3`` — blended (Sabine/Eyring) volume estimate (m³)
        - ``room_volume_m3_sabine`` / ``_eyring`` — individual estimates (m³)
        - ``room_volume_m3_low`` / ``_high`` — ±30 % confidence interval (m³)
        - ``room_dim_width_m``, ``_depth_m``, ``_height_m`` — box dimensions (m)
        - ``room_surface_area_m2`` — total surface area (m²)
        - ``room_mean_absorption`` — estimated mean absorption coefficient
        - ``room_critical_distance_m`` — Schroeder critical distance (m)
        - ``room_class`` — qualitative size label (string)
        - ``room_estimation_method`` — ``"sabine"`` or ``"eyring"`` (string)
        - ``room_confidence`` — ``"high"``, ``"medium"``, or ``"low"`` (string)
        - ``room_confidence_score`` — numeric confidence in [0, 1]
    """
    # --- Stage 1: EDR RT60 extraction --------------------------------------
    edr_result = extract_edr_rt60(audio, sr)
    rt60 = edr_result["rt60_s"]
    rt60_low = edr_result["rt60_low_s"]
    rt60_mid = edr_result["rt60_mid_s"]
    rt60_high = edr_result["rt60_high_s"]
    valid_bins = edr_result["valid_bins"]
    duration_s = edr_result["duration_s"]
    silence_ratio = edr_result["silence_ratio"]

    # Guard: degenerate / silent input
    mono = np.mean(audio, axis=1)
    rms_val = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0
    if rt60 <= 0.01 or rms_val < 1e-9:
        return _empty_result()

    # --- Stage 2: Absorption inference -------------------------------------
    alpha = infer_absorption(rt60_low, rt60_mid, rt60_high, prior=prior_absorption)

    # --- Stage 3: Volume estimation ----------------------------------------
    vol = estimate_volume(rt60, alpha)
    v_primary = vol["primary_m3"]
    v_sabine = vol["sabine_m3"]
    v_eyring = vol["eyring_m3"]

    # --- Stage 4: Dimension projection -------------------------------------
    dims = project_dimensions(v_primary)
    w = dims["width_m"]
    d = dims["depth_m"]
    h = dims["height_m"]
    sa = dims["surface_area_m2"]

    # Compute mean absorption output and critical distance from volume
    a_total = _SABINE_K * v_primary / rt60  # A = α·S
    alpha_out = float(np.clip(a_total / max(sa, 1e-3), 0.01, 0.99))
    dc = math.sqrt(a_total / (16.0 * math.pi)) if a_total > 0.0 else 0.0

    # --- Stage 5: Confidence scoring ---------------------------------------
    confidence_score, confidence = score_confidence(
        rt60=rt60,
        valid_bins=valid_bins,
        duration_s=duration_s,
        silence_ratio=silence_ratio,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
    )

    # --- Stage 6: Classification -------------------------------------------
    room_class = classify_room(v_primary, rt60)
    method = "eyring" if alpha > 0.3 and v_eyring > 0.0 else "sabine"

    return {
        "room_rt60_s": round(rt60, 4),
        "room_rt60_low_s": round(rt60_low, 4),
        "room_rt60_mid_s": round(rt60_mid, 4),
        "room_rt60_high_s": round(rt60_high, 4),
        "room_volume_m3": round(v_primary, 2),
        "room_volume_m3_sabine": round(v_sabine, 2),
        "room_volume_m3_eyring": round(v_eyring, 2),
        "room_volume_m3_low": round(vol["low_m3"], 2),
        "room_volume_m3_high": round(vol["high_m3"], 2),
        "room_dim_width_m": round(w, 2),
        "room_dim_depth_m": round(d, 2),
        "room_dim_height_m": round(h, 2),
        "room_surface_area_m2": round(sa, 2),
        "room_mean_absorption": round(alpha_out, 4),
        "room_critical_distance_m": round(dc, 3),
        "room_class": room_class,
        "room_estimation_method": method,
        "room_confidence": confidence,
        "room_confidence_score": round(confidence_score, 3),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _v_eyring(rt60: float, surface_area: float, alpha: float) -> float:
    """Eyring volume estimate for a given surface area and absorption."""
    alpha_safe = float(np.clip(alpha, 0.001, 0.999))
    denominator = -math.log(max(1.0 - alpha_safe, 1e-9))
    if denominator <= 1e-12:
        return 0.0
    return float(_SABINE_K * surface_area * rt60 / denominator)


def _empty_result() -> dict[str, object]:
    """Return a zeroed/placeholder result for degenerate inputs."""
    return {
        "room_rt60_s": 0.0,
        "room_rt60_low_s": 0.0,
        "room_rt60_mid_s": 0.0,
        "room_rt60_high_s": 0.0,
        "room_volume_m3": 0.0,
        "room_volume_m3_sabine": 0.0,
        "room_volume_m3_eyring": 0.0,
        "room_volume_m3_low": 0.0,
        "room_volume_m3_high": 0.0,
        "room_dim_width_m": 0.0,
        "room_dim_depth_m": 0.0,
        "room_dim_height_m": 0.0,
        "room_surface_area_m2": 0.0,
        "room_mean_absorption": 0.0,
        "room_critical_distance_m": 0.0,
        "room_class": "unknown",
        "room_estimation_method": "none",
        "room_confidence": "low",
        "room_confidence_score": 0.0,
    }
