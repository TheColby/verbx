import numpy as np

from verbx.io.audio import (
    iter_audio_blocks,
    peak_normalize,
    read_audio,
    soft_limiter,
    write_audio,
)


def test_read_write_audio(tmp_path):
    """Test reading and writing audio."""
    path = tmp_path / "test.wav"
    sr = 44100
    audio = np.random.rand(44100, 2).astype(np.float32)
    write_audio(path, audio, sr)

    loaded, loaded_sr = read_audio(path)
    assert loaded_sr == sr
    assert loaded.shape == audio.shape
    assert np.allclose(loaded, audio, atol=1e-4)


def test_iter_audio_blocks(tmp_path):
    """Test iterating audio blocks."""
    path = tmp_path / "test_blocks.wav"
    sr = 44100
    audio = np.random.rand(88200, 2).astype(np.float32)  # 2 seconds
    write_audio(path, audio, sr)

    blocks = list(iter_audio_blocks(path, block_size=4096))
    assert len(blocks) > 0
    reconstructed = np.concatenate(blocks, axis=0)
    assert reconstructed.shape == audio.shape
    assert np.allclose(reconstructed, audio, atol=1e-4)


def test_peak_normalize():
    """Test peak normalization."""
    audio = np.array([0.5, -0.5], dtype=np.float32)
    normalized = peak_normalize(audio, target_dbfs=0.0)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)

    normalized_half = peak_normalize(audio, target_dbfs=-6.0)
    assert np.isclose(np.max(np.abs(normalized_half)), 0.5, atol=1e-2)


def test_soft_limiter():
    """Test soft limiter."""
    audio = np.array([10.0, -10.0], dtype=np.float32)
    limited = soft_limiter(audio, threshold_dbfs=0.0)
    assert np.max(np.abs(limited)) < 1.0 + 1e-4
    assert np.max(np.abs(limited)) > 0.9
