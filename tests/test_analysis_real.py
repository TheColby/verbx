import numpy as np

from verbx.analysis.analyzer import AudioAnalyzer


def test_analyzer_real():
    """Test analyzer with real audio generation."""
    analyzer = AudioAnalyzer()
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave 440Hz
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add noise
    audio += 0.1 * np.random.randn(len(t))
    # Reshape to (samples, 1)
    audio = audio[:, np.newaxis]

    result = analyzer.analyze(audio, sr)

    assert result["duration_s"] == 1.0
    assert result["sr"] == sr
    assert float(result["rms_dbfs"]) < 0
    assert float(result["peak_dbfs"]) < 0
    assert float(result["spectral_centroid_mean"]) > 0
    assert "spectral_flux_mean" in result
