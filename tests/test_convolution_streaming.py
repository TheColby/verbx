from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine
from verbx.io.audio import iter_audio_blocks


# ---------------------------------------------------------------------------
# iter_audio_blocks
# ---------------------------------------------------------------------------


def test_iter_audio_blocks_exact_multiple(tmp_path: Path) -> None:
    """Blocks divide evenly into file length — every block has full size."""
    sr = 48_000
    block_size = 512
    n_blocks = 4
    total = block_size * n_blocks
    audio = np.random.default_rng(0).standard_normal((total, 1)).astype(np.float64)
    path = tmp_path / "exact.wav"
    sf.write(str(path), audio, sr)

    blocks = list(iter_audio_blocks(str(path), block_size))

    assert len(blocks) == n_blocks
    for blk in blocks:
        assert blk.shape == (block_size, 1)
        assert blk.dtype == np.float64


def test_iter_audio_blocks_short_final_block(tmp_path: Path) -> None:
    """Non-multiple length produces a shorter final block."""
    sr = 48_000
    block_size = 512
    total = block_size * 3 + 100
    audio = np.random.default_rng(1).standard_normal((total, 2)).astype(np.float64)
    path = tmp_path / "nonmult.wav"
    sf.write(str(path), audio, sr)

    blocks = list(iter_audio_blocks(str(path), block_size))

    assert len(blocks) == 4
    for blk in blocks[:-1]:
        assert blk.shape == (block_size, 2)
    assert blocks[-1].shape == (100, 2)
    assert blocks[-1].dtype == np.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_impulse_ir(length: int, channels: int = 1) -> np.ndarray:
    """Unit impulse IR — convolving with it returns the input unchanged."""
    ir = np.zeros((length, channels), dtype=np.float64)
    ir[0, :] = 1.0
    return ir


def _make_engine(
    ir_path: str,
    partition_size: int = 512,
    wet: float = 1.0,
    dry: float = 0.0,
) -> ConvolutionReverbEngine:
    return ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=wet,
            dry=dry,
            ir_path=ir_path,
            ir_normalize="none",
            partition_size=partition_size,
            tail_limit=None,
            threads=1,
            device="cpu",
        )
    )


# ---------------------------------------------------------------------------
# process_streaming_file — output shape / sample count
# ---------------------------------------------------------------------------


def test_streaming_output_sample_count(tmp_path: Path) -> None:
    """Output file length equals input + IR-tail samples."""
    sr = 48_000
    input_len = 2048
    ir_len = 512

    audio = np.random.default_rng(2).standard_normal((input_len, 1)).astype(np.float64) * 0.05
    ir = _make_impulse_ir(ir_len)

    audio_path = tmp_path / "input.wav"
    ir_path = tmp_path / "ir.wav"
    out_path = tmp_path / "output.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    engine = _make_engine(str(ir_path), partition_size=512)
    stats = engine.process_streaming_file(str(audio_path), str(out_path))

    expected_output = input_len + ir_len - 1
    assert stats["output_samples"] == expected_output
    assert stats["sample_rate"] == sr

    out_audio, out_sr = sf.read(str(out_path), always_2d=True, dtype="float64")
    assert out_sr == sr
    assert out_audio.shape[0] == expected_output


# ---------------------------------------------------------------------------
# process_streaming_file matches in-memory processing
# ---------------------------------------------------------------------------


def test_streaming_matches_in_memory(tmp_path: Path) -> None:
    """Streaming path produces the same result as the in-memory process() path."""
    sr = 48_000
    rng = np.random.default_rng(3)
    audio = rng.standard_normal((2048, 2)).astype(np.float64) * 0.05

    ir = np.zeros((1024, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    ir[200, 0] = 0.45
    ir[700, 0] = 0.2

    audio_path = tmp_path / "input.wav"
    ir_path = tmp_path / "ir.wav"
    out_path = tmp_path / "output.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    engine = _make_engine(str(ir_path), partition_size=512)

    # In-memory reference
    mem_output = engine.process(audio, sr=sr)

    # Streaming
    engine.process_streaming_file(str(audio_path), str(out_path))
    stream_output, _ = sf.read(str(out_path), always_2d=True, dtype="float64")

    assert mem_output.shape == stream_output.shape
    # Streaming path writes through soundfile (PCM quantization), so tolerance
    # is looser than pure in-memory float64 comparison.
    np.testing.assert_allclose(stream_output, mem_output, atol=1e-3)


# ---------------------------------------------------------------------------
# Mono IR + stereo input
# ---------------------------------------------------------------------------


def test_streaming_mono_ir_stereo_input(tmp_path: Path) -> None:
    """Mono IR applied to stereo input produces stereo output."""
    sr = 48_000
    rng = np.random.default_rng(4)
    audio = rng.standard_normal((1024, 2)).astype(np.float64) * 0.05

    ir = _make_impulse_ir(256, channels=1)

    audio_path = tmp_path / "stereo_input.wav"
    ir_path = tmp_path / "mono_ir.wav"
    out_path = tmp_path / "out.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    engine = _make_engine(str(ir_path), partition_size=512)
    stats = engine.process_streaming_file(str(audio_path), str(out_path))

    assert stats["channels"] == 2
    out_audio, _ = sf.read(str(out_path), always_2d=True, dtype="float64")
    assert out_audio.shape[1] == 2
    assert np.all(np.isfinite(out_audio))


# ---------------------------------------------------------------------------
# Very short IR (< 1 block)
# ---------------------------------------------------------------------------


def test_streaming_very_short_ir(tmp_path: Path) -> None:
    """IR shorter than one partition block still convolves correctly."""
    sr = 48_000
    rng = np.random.default_rng(5)
    audio = rng.standard_normal((2048, 1)).astype(np.float64) * 0.05

    ir = _make_impulse_ir(16, channels=1)

    audio_path = tmp_path / "input.wav"
    ir_path = tmp_path / "short_ir.wav"
    out_path = tmp_path / "out.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    engine = _make_engine(str(ir_path), partition_size=512)
    stats = engine.process_streaming_file(str(audio_path), str(out_path))

    expected_samples = 2048 + 16 - 1
    assert stats["output_samples"] == expected_samples

    out_audio, _ = sf.read(str(out_path), always_2d=True, dtype="float64")
    assert out_audio.shape[0] == expected_samples
    assert np.all(np.isfinite(out_audio))

    # With a unit impulse IR, the wet signal should equal the input (padded).
    np.testing.assert_allclose(out_audio[:2048, 0], audio[:, 0], atol=1e-3)


# ---------------------------------------------------------------------------
# Very short input (< 1 block)
# ---------------------------------------------------------------------------


def test_streaming_very_short_input(tmp_path: Path) -> None:
    """Input shorter than partition block processes correctly."""
    sr = 48_000
    rng = np.random.default_rng(6)
    audio = rng.standard_normal((64, 1)).astype(np.float64) * 0.05

    ir = _make_impulse_ir(512, channels=1)

    audio_path = tmp_path / "tiny.wav"
    ir_path = tmp_path / "ir.wav"
    out_path = tmp_path / "out.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    engine = _make_engine(str(ir_path), partition_size=512)
    stats = engine.process_streaming_file(str(audio_path), str(out_path))

    expected_samples = 64 + 512 - 1
    assert stats["output_samples"] == expected_samples

    out_audio, _ = sf.read(str(out_path), always_2d=True, dtype="float64")
    assert out_audio.shape[0] == expected_samples
    assert np.all(np.isfinite(out_audio))


# ---------------------------------------------------------------------------
# Large block_size > input length
# ---------------------------------------------------------------------------


def test_streaming_block_size_larger_than_input(tmp_path: Path) -> None:
    """Block size exceeding input length still produces correct output."""
    sr = 48_000
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((256, 1)).astype(np.float64) * 0.05

    ir = _make_impulse_ir(128, channels=1)

    audio_path = tmp_path / "short.wav"
    ir_path = tmp_path / "ir.wav"
    out_path = tmp_path / "out.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    # partition_size=4096 >> input length of 256
    engine = _make_engine(str(ir_path), partition_size=4096)
    stats = engine.process_streaming_file(str(audio_path), str(out_path))

    expected_samples = 256 + 128 - 1
    assert stats["output_samples"] == expected_samples

    out_audio, _ = sf.read(str(out_path), always_2d=True, dtype="float64")
    assert out_audio.shape[0] == expected_samples
    assert np.all(np.isfinite(out_audio))

    # Compare with in-memory
    mem_output = engine.process(audio, sr=sr)
    np.testing.assert_allclose(out_audio, mem_output, atol=1e-3)


# ---------------------------------------------------------------------------
# Tail decays to silence
# ---------------------------------------------------------------------------


def test_streaming_output_tail_decays_to_silence(tmp_path: Path) -> None:
    """The tail portion of the output (past the input) decays to zero for a simple IR."""
    sr = 48_000
    input_len = 1024

    # Short impulse followed by silence
    audio = np.zeros((input_len, 1), dtype=np.float64)
    audio[0, 0] = 1.0

    # IR with a couple of taps — finite energy that decays.
    ir = np.zeros((512, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    ir[100, 0] = 0.3

    audio_path = tmp_path / "impulse.wav"
    ir_path = tmp_path / "ir.wav"
    out_path = tmp_path / "out.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    engine = _make_engine(str(ir_path), partition_size=256)
    engine.process_streaming_file(str(audio_path), str(out_path))

    out_audio, _ = sf.read(str(out_path), always_2d=True, dtype="float64")

    # The last 100 samples should be silent — the IR's last tap is at index 100,
    # input energy is only at sample 0, so the last IR-tail region is all zeros.
    tail = out_audio[-100:, :]
    assert np.max(np.abs(tail)) < 1e-12


# ---------------------------------------------------------------------------
# Partition size does not affect audible result
# ---------------------------------------------------------------------------


def test_partition_size_does_not_affect_result(tmp_path: Path) -> None:
    """Different partition sizes yield the same output (within float tolerance)."""
    sr = 48_000
    rng = np.random.default_rng(8)
    audio = rng.standard_normal((2048, 1)).astype(np.float64) * 0.05

    ir = np.zeros((1024, 1), dtype=np.float64)
    ir[0, 0] = 1.0
    ir[300, 0] = 0.5
    ir[800, 0] = 0.15

    audio_path = tmp_path / "input.wav"
    ir_path = tmp_path / "ir.wav"
    sf.write(str(audio_path), audio, sr)
    sf.write(str(ir_path), ir, sr)

    outputs = []
    for part_size in (256, 512, 1024, 2048):
        out_path = tmp_path / f"out_{part_size}.wav"
        engine = _make_engine(str(ir_path), partition_size=part_size)
        engine.process_streaming_file(str(audio_path), str(out_path))
        out_audio, _ = sf.read(str(out_path), always_2d=True, dtype="float64")
        outputs.append(out_audio)

    # All outputs should have the same length and be numerically identical.
    for i in range(1, len(outputs)):
        assert outputs[i].shape == outputs[0].shape
        np.testing.assert_allclose(outputs[i], outputs[0], atol=1e-3)
