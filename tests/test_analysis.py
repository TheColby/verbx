import numpy as np

from verbx.analysis.analyzer import AudioAnalyzer


def test_analyzer_stub():
    """Test analyzer stub."""
    analyzer = AudioAnalyzer()
    audio = np.random.rand(1024)
    result = analyzer.analyze(audio, 44100)
    assert "duration_s" in result
    assert "rms_dbfs" in result
    assert "peak_dbfs" in result
