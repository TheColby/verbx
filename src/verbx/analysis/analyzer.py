import numpy as np

from verbx.analysis.features_spectral import analyze_spectrum
from verbx.analysis.features_time import (
    calculate_crest_factor,
    calculate_dc_offset,
    calculate_peak,
    calculate_rms,
    calculate_zcr,
)


class AudioAnalyzer:
    """Analyze audio features (SoX style)."""

    def analyze(self, audio: np.ndarray, sr: int) -> dict[str, float | str | int]:
        """
        Analyze audio and return a dictionary of features.

        Args:
            audio: Input audio array (n_samples, n_channels).
            sr: Sample rate.

        Returns:
            Dictionary of audio features.
        """
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        n_samples, n_channels = audio.shape
        duration = n_samples / sr if sr > 0 else 0.0

        # Time Domain Features (per channel, then average/max)
        rms_vals = [calculate_rms(audio[:, c]) for c in range(n_channels)]
        peak_vals = [calculate_peak(audio[:, c]) for c in range(n_channels)]
        zcr_vals = [calculate_zcr(audio[:, c]) for c in range(n_channels)]
        dc_vals = [calculate_dc_offset(audio[:, c]) for c in range(n_channels)]

        rms_mean = float(np.mean(rms_vals))
        peak_max = float(np.max(peak_vals))
        zcr_mean = float(np.mean(zcr_vals))
        dc_mean = float(np.mean(dc_vals))

        # Convert to dBFS
        rms_dbfs = 20 * np.log10(rms_mean) if rms_mean > 0 else -120.0
        peak_dbfs = 20 * np.log10(peak_max) if peak_max > 0 else -120.0
        crest_factor = calculate_crest_factor(peak_max, rms_mean)

        # Spectral Features (Mono Mix)
        if n_channels > 1:
            mono_audio = np.mean(audio, axis=1)
        else:
            mono_audio = audio[:, 0]

        # Downsample for spectral analysis speed if needed?
        # Let's keep full SR for accuracy unless huge.
        # But if file is > 1 min, maybe take a representative chunk?
        # For v0.1: analyze whole file (or chunk provided by pipeline).
        # Assuming `audio` is what we want to analyze.

        spectral_stats = analyze_spectrum(mono_audio, sr)

        results = {
            "sr": sr,
            "channels": n_channels,
            "samples": n_samples,
            "duration_s": float(duration),
            "rms_dbfs": rms_dbfs,
            "peak_dbfs": peak_dbfs,
            "crest_factor": crest_factor,
            "zcr_mean": zcr_mean,
            "dc_offset": dc_mean,
        }
        results.update(spectral_stats)

        return results
