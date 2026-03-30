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


def estimate_room_size(
    audio: AudioArray,
    sr: int,
    *,
    prior_absorption: float | None = None,
) -> dict[str, object]:
    """Estimate room acoustic properties from a reverberant audio recording.

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
        high-frequency to low-frequency RT60 values:
        heavily damped (ratio < 0.5)   → α ≈ 0.10–0.20 (hard/stone/concrete)
        moderately treated (0.5–0.8)   → α ≈ 0.20–0.35 (typical room)
        broadly treated (> 0.8)        → α ≈ 0.35–0.55 (acoustic treatment)

    Returns
    -------
    dict
        Flat dictionary with ``room_*`` prefixed keys (all JSON-serializable).
        Key classes:

        - ``room_rt60_s`` — best RT60 estimate used as sizing input (s)
        - ``room_volume_m3`` — Sabine volume estimate (m³)
        - ``room_volume_m3_eyring`` — Eyring volume estimate (m³)
        - ``room_volume_m3_low`` / ``_high`` — ±30 % confidence interval (m³)
        - ``room_dim_width_m``, ``_depth_m``, ``_height_m`` — rectangular box
          dimensions derived from the Sabine volume (m)
        - ``room_surface_area_m2`` — total surface area of the estimated box (m²)
        - ``room_mean_absorption`` — estimated mean absorption coefficient [0, 1]
        - ``room_critical_distance_m`` — Schroeder critical distance (m)
        - ``room_class`` — qualitative size label (string)
        - ``room_estimation_method`` — ``"sabine"`` or ``"eyring"`` (string)
        - ``room_confidence`` — ``"high"``, ``"medium"``, or ``"low"`` (string)
        - ``room_confidence_score`` — numeric confidence in [0, 1]
    """
    edr = edr_summary(audio, sr)
    rt60_median = float(edr.get("edr_rt60_median_s", 0.0))
    rt60_low = float(edr.get("edr_rt60_low_s", 0.0))
    rt60_mid = float(edr.get("edr_rt60_mid_s", 0.0))
    rt60_high = float(edr.get("edr_rt60_high_s", 0.0))
    valid_bins = float(edr.get("edr_valid_bins", 0.0))

    # Use mid-band RT60 as the primary sizing input; fall back to median
    rt60 = rt60_mid if rt60_mid > 0.0 else rt60_median

    # Basic signal properties for confidence scoring
    mono = np.mean(audio, axis=1)
    duration_s = float(audio.shape[0]) / max(1.0, float(sr))
    rms_val = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0
    silence_ratio = (
        float(np.mean(np.abs(mono) < 10.0 ** (-50.0 / 20.0))) if mono.size > 0 else 1.0
    )

    # --- Absorption coefficient estimation --------------------------------
    if prior_absorption is not None:
        alpha = float(np.clip(prior_absorption, 0.01, 0.99))
    else:
        alpha = _infer_absorption(rt60_low, rt60_mid, rt60_high)

    # --- Empty / degenerate signal guard -----------------------------------
    if rt60 <= 0.01 or rms_val < 1e-9:
        return _empty_result()

    # --- Sabine volume estimate -------------------------------------------
    # V = RT60 * α * S / _SABINE_K, but S depends on V via aspect ratios.
    # Solving the system:
    #   V = _ASPECT_VOLUME * W³
    #   S = _ASPECT_SA_COEF * W²  → W = sqrt(S / _ASPECT_SA_COEF)
    #   V = RT60 * α * S / _SABINE_K
    # Substitute S = V^(2/3) * (_ASPECT_SA_COEF / _ASPECT_VOLUME^(2/3)):
    #   V^(1/3) = RT60 * α * (_ASPECT_SA_COEF / _ASPECT_VOLUME^(2/3)) / _SABINE_K
    sa_v23 = _ASPECT_SA_COEF / (_ASPECT_VOLUME ** (2.0 / 3.0))
    v_cube_root_sabine = (rt60 * alpha * sa_v23) / _SABINE_K
    v_sabine = v_cube_root_sabine ** 3.0

    # --- Eyring volume estimate (iterative) --------------------------------
    # α_eyring = 1 - exp(-_SABINE_K * V / (S * RT60))
    # Use Sabine V as initial guess and iterate once
    w_est = (v_sabine / _ASPECT_VOLUME) ** (1.0 / 3.0)
    s_est = _ASPECT_SA_COEF * w_est * w_est
    v_eyring = _v_eyring(rt60, s_est, alpha)
    # One refinement pass with updated S from Eyring V
    w_ey = max(1e-3, (v_eyring / _ASPECT_VOLUME) ** (1.0 / 3.0))
    s_ey = _ASPECT_SA_COEF * w_ey * w_ey
    v_eyring = _v_eyring(rt60, s_ey, alpha)

    # --- Primary volume: prefer Eyring, blend toward Sabine at low alpha --
    blend = float(np.clip((alpha - 0.1) / 0.3, 0.0, 1.0))
    v_primary = (1.0 - blend) * v_sabine + blend * v_eyring

    # --- Room dimensions from volume (aspect ratios) -----------------------
    w = max(0.1, (v_primary / _ASPECT_VOLUME) ** (1.0 / 3.0))
    d = _ASPECT_D * w
    h = _ASPECT_H * w
    sa = _ASPECT_SA_COEF * w * w

    # --- Mean absorption from Sabine (round-trip for output) ---------------
    a_total = _SABINE_K * v_primary / rt60  # A = α·S
    alpha_out = float(np.clip(a_total / max(sa, 1e-3), 0.01, 0.99))

    # --- Critical distance -------------------------------------------------
    dc = math.sqrt(a_total / (16.0 * math.pi)) if a_total > 0.0 else 0.0

    # --- Room class --------------------------------------------------------
    room_class = _classify_room(v_primary, rt60)

    # --- Estimation method -------------------------------------------------
    method = "eyring" if alpha > 0.3 and v_eyring > 0.0 else "sabine"

    # --- Confidence scoring ------------------------------------------------
    confidence_score, confidence = _score_confidence(
        rt60=rt60,
        valid_bins=valid_bins,
        duration_s=duration_s,
        silence_ratio=silence_ratio,
        rt60_low=rt60_low,
        rt60_high=rt60_high,
    )

    # --- Uncertainty bounds (±30 % on volume) ------------------------------
    v_low = v_primary * 0.70
    v_high = v_primary * 1.30

    return {
        "room_rt60_s": round(rt60, 4),
        "room_rt60_low_s": round(rt60_low, 4),
        "room_rt60_mid_s": round(rt60_mid, 4),
        "room_rt60_high_s": round(rt60_high, 4),
        "room_volume_m3": round(v_primary, 2),
        "room_volume_m3_sabine": round(v_sabine, 2),
        "room_volume_m3_eyring": round(v_eyring, 2),
        "room_volume_m3_low": round(v_low, 2),
        "room_volume_m3_high": round(v_high, 2),
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
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_absorption(rt60_low: float, rt60_mid: float, rt60_high: float) -> float:
    """Estimate mean absorption from the spectral RT60 ratio.

    Hard surfaces (concrete, stone, tile) have strongly frequency-dependent
    absorption — RT60 falls steeply from low to high frequencies.
    Acoustic treatment flattens the decay curve.
    """
    if rt60_low <= 0.0 or rt60_high <= 0.0 or rt60_mid <= 0.0:
        # No band information — use a moderate prior
        return 0.20

    # High/low RT60 ratio: close to 1 means spectrally flat decay
    hf_lf_ratio = float(np.clip(rt60_high / max(rt60_low, 1e-6), 0.0, 2.0))

    if hf_lf_ratio < 0.35:
        alpha = 0.10  # Very hard (stone/concrete)
    elif hf_lf_ratio < 0.55:
        alpha = 0.15  # Hard (plaster/brick)
    elif hf_lf_ratio < 0.70:
        alpha = 0.22  # Light treatment (typical office/home)
    elif hf_lf_ratio < 0.85:
        alpha = 0.30  # Moderate treatment (studio/meeting room)
    else:
        alpha = 0.42  # Heavy treatment (broadcast/mastering studio)

    return float(alpha)


def _v_eyring(rt60: float, surface_area: float, alpha: float) -> float:
    """Eyring volume estimate for a given surface area and absorption."""
    alpha_safe = float(np.clip(alpha, 0.001, 0.999))
    denominator = -math.log(max(1.0 - alpha_safe, 1e-9))
    if denominator <= 1e-12:
        return 0.0
    return float(_SABINE_K * surface_area * rt60 / denominator)


def _classify_room(volume_m3: float, rt60_s: float) -> str:
    """Return a qualitative room-size label based on volume and RT60."""
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


def _score_confidence(
    *,
    rt60: float,
    valid_bins: float,
    duration_s: float,
    silence_ratio: float,
    rt60_low: float,
    rt60_high: float,
) -> tuple[float, str]:
    """Return ``(score, label)`` confidence pair in [0, 1]."""
    score = 0.0

    # RT60 estimate quality
    if rt60 > 0.05:
        score += 0.20
    if rt60 > 0.20:
        score += 0.10

    # Valid frequency bins in EDR
    if valid_bins >= 30:
        score += 0.25
    elif valid_bins >= 10:
        score += 0.15
    elif valid_bins >= 3:
        score += 0.05

    # Signal duration (longer = better decay observation)
    if duration_s >= 3.0:
        score += 0.20
    elif duration_s >= 1.5:
        score += 0.12
    elif duration_s >= 0.5:
        score += 0.05

    # Non-silence ratio (signal should have content throughout)
    if silence_ratio < 0.3:
        score += 0.15
    elif silence_ratio < 0.6:
        score += 0.08

    # Spectral consistency (both low and high RT60 estimates available)
    if rt60_low > 0.0 and rt60_high > 0.0:
        score += 0.10

    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.70:
        label = "high"
    elif score >= 0.40:
        label = "medium"
    else:
        label = "low"

    return score, label


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
