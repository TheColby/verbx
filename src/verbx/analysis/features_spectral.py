from typing import Dict

import librosa
import numpy as np


def analyze_spectrum(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute spectral features from audio using librosa.
    Returns mean values.

    Args:
        audio: Audio array (n_samples). Assumes mono for now.
        sr: Sample rate.

    Returns:
        Dictionary of spectral features (mean).
    """
    if len(audio) == 0:
        return {}

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Calculate STFT once
    # Use smaller hop/win for long files if memory constrained,
    # or just rely on librosa defaults (2048 win, 512 hop).
    # If file is huge, this call uses RAM.
    # For v0.1 we assume it fits or is chunked before calling this.

    try:
        S, phase = librosa.magphase(librosa.stft(y=audio))

        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        centroid_mean = float(np.mean(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        bandwidth_mean = float(np.mean(bandwidth))

        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
        rolloff_mean = float(np.mean(rolloff))

        flatness = librosa.feature.spectral_flatness(S=S)
        flatness_mean = float(np.mean(flatness))

        # Spectral Flux (onset strength proxy)
        flux = librosa.onset.onset_strength(S=S, sr=sr)
        flux_mean = float(np.mean(flux))
        flux_std = float(np.std(flux))

        return {
            "spectral_centroid_mean": centroid_mean,
            "spectral_bandwidth_mean": bandwidth_mean,
            "spectral_rolloff_85_mean": rolloff_mean,
            "spectral_flatness_mean": flatness_mean,
            "spectral_flux_mean": flux_mean,
            "spectral_flux_std": flux_std,
        }
    except Exception:
        # Fallback or log error
        # print(f"Spectral analysis error: {e}")
        return {
            "spectral_centroid_mean": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "spectral_rolloff_85_mean": 0.0,
            "spectral_flatness_mean": 0.0,
            "spectral_flux_mean": 0.0,
            "spectral_flux_std": 0.0,
        }
