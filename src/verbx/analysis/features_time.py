import numpy as np


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS of audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))


def calculate_peak(audio: np.ndarray) -> float:
    """Calculate Peak amplitude of audio signal."""
    if len(audio) == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def calculate_zcr(audio: np.ndarray) -> float:
    """Calculate Zero Crossing Rate (mean)."""
    if len(audio) == 0:
        return 0.0
    # Simple ZCR: count sign changes
    # Use zero_crossings helper or diff
    # We want rate (crossings per sample or per second?)
    # Usually per frame, but here overall.
    # zcr = sum(|sgn(x[n]) - sgn(x[n-1])|) / (2N)
    # Just use numpy
    zero_crossings = np.diff(np.signbit(audio), axis=0)
    return float(np.mean(zero_crossings))


def calculate_crest_factor(peak: float, rms: float) -> float:
    """Calculate Crest Factor (Peak / RMS)."""
    if rms == 0:
        return 0.0
    return peak / rms


def calculate_dc_offset(audio: np.ndarray) -> float:
    """Calculate DC Offset (mean amplitude)."""
    if len(audio) == 0:
        return 0.0
    return float(np.mean(audio))
