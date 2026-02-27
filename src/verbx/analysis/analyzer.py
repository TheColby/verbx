"""Analyzer façade for reporting canonical audio metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.analysis.features_spectral import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
    spectral_slope,
)
from verbx.analysis.features_time import (
    crest_factor,
    dc_offset,
    duration_seconds,
    dynamic_range,
    energy,
    l1_energy,
    peak,
    peak_dbfs,
    rms,
    rms_dbfs,
    silence_ratio,
    stereo_correlation,
    stereo_width,
    transient_density,
    zero_crossing_rate,
)

AudioArray = npt.NDArray[np.float32]


class AudioAnalyzer:
    """Analyze audio and return a dictionary of typed metrics."""

    def analyze(self, audio: AudioArray, sr: int) -> dict[str, float]:
        """Return baseline and extended analysis metrics for CLI consumption."""
        channel_rms = np.sqrt(np.mean(np.square(audio), axis=0, dtype=np.float64))

        result: dict[str, float] = {
            "duration": duration_seconds(audio, sr),
            "samples": float(audio.shape[0]),
            "channels": float(audio.shape[1]),
            "rms": rms(audio),
            "rms_dbfs": rms_dbfs(audio),
            "peak": peak(audio),
            "peak_dbfs": peak_dbfs(audio),
            "crest_factor": crest_factor(audio),
            "dc_offset": dc_offset(audio),
            "dynamic_range": dynamic_range(audio),
            "energy": energy(audio),
            "l1_energy": l1_energy(audio),
            "sample_min": float(np.min(audio)),
            "sample_max": float(np.max(audio)),
            "silence_ratio": silence_ratio(audio),
            "transient_density": transient_density(audio),
            "zero_crossing_rate": zero_crossing_rate(audio),
            "spectral_centroid": spectral_centroid(audio, sr),
            "spectral_bandwidth": spectral_bandwidth(audio, sr),
            "spectral_rolloff": spectral_rolloff(audio, sr),
            "spectral_flatness": spectral_flatness(audio, sr),
            "spectral_flux": spectral_flux(audio, sr),
            "spectral_slope": spectral_slope(audio, sr),
            "stereo_correlation": stereo_correlation(audio),
            "stereo_width": stereo_width(audio),
        }

        for idx, value in enumerate(channel_rms.tolist(), start=1):
            result[f"channel_{idx}_rms"] = float(value)

        return result
