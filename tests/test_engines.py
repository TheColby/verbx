import numpy as np

from verbx.core.algo_reverb import AlgoReverbEngine
from verbx.core.convolution_reverb import ConvolutionReverbEngine


def test_algo_engine_stub():
    """Test algorithmic reverb engine stub."""
    engine = AlgoReverbEngine()
    audio = np.zeros(1024)
    processed = engine.process(audio, 44100)
    assert len(processed) == 1024
    assert processed.dtype == np.float64 or processed.dtype == np.float32


def test_conv_engine_stub():
    """Test convolution reverb engine stub."""
    engine = ConvolutionReverbEngine()
    audio = np.zeros(1024)
    processed = engine.process(audio, 44100)
    assert len(processed) == 1024
