from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine
from verbx.core.convolution_reverb import ConvolutionReverbConfig, ConvolutionReverbEngine


def test_algo_engine_stable_and_typed() -> None:
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=80.0,
            pre_delay_ms=12.0,
            damping=0.5,
            width=1.0,
            mod_depth_ms=1.5,
            mod_rate_hz=0.08,
            wet=0.7,
            dry=0.3,
            block_size=512,
            shimmer=True,
            shimmer_semitones=7.0,
            shimmer_mix=0.3,
            shimmer_feedback=0.25,
            shimmer_lowcut=200.0,
            shimmer_highcut=8000.0,
        )
    )
    audio = np.random.default_rng(0).standard_normal((4096, 2)).astype(np.float32) * 0.1

    output = engine.process(audio, sr=48_000)

    assert isinstance(output, np.ndarray)
    assert output.shape == audio.shape
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_algo_engine_custom_allpass_and_comb_delay_controls() -> None:
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=22.0,
            pre_delay_ms=8.0,
            damping=0.35,
            width=1.0,
            mod_depth_ms=1.0,
            mod_rate_hz=0.07,
            allpass_stages=4,
            allpass_gain=0.68,
            allpass_delays_ms=(3.0, 5.0, 8.0, 13.0),
            comb_delays_ms=(19.0, 23.0, 29.0, 31.0, 37.0),
            fdn_lines=5,
            wet=0.85,
            dry=0.15,
            block_size=256,
        )
    )
    audio = np.random.default_rng(7).standard_normal((2048, 1)).astype(np.float32) * 0.05
    output = engine.process(audio, sr=48_000)
    assert output.shape == audio.shape
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_algo_engine_rejects_allpass_gain_count_mismatch() -> None:
    try:
        _ = AlgoReverbEngine(
            AlgoReverbConfig(
                allpass_stages=4,
                allpass_gains=(0.7, 0.6, 0.5),
            )
        )
    except ValueError as exc:
        assert "allpass_gains length must match" in str(exc)
        return
    raise AssertionError("Expected ValueError for mismatched allpass_gains length")


def test_convolution_engine_partitioned_fft(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    audio = (rng.standard_normal((2048, 2)).astype(np.float32)) * 0.05

    ir = np.zeros((1024, 1), dtype=np.float32)
    ir[0, 0] = 1.0
    ir[200, 0] = 0.45
    ir[700, 0] = 0.2

    ir_path = tmp_path / "ir.wav"
    sf.write(str(ir_path), ir, 48_000)

    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=0.8,
            dry=0.2,
            ir_path=str(ir_path),
            ir_normalize="none",
            partition_size=512,
            tail_limit=None,
            threads=None,
        )
    )

    output = engine.process(audio, sr=48_000)

    assert isinstance(output, np.ndarray)
    assert output.dtype == np.float32
    assert output.shape[1] == audio.shape[1]
    assert output.shape[0] >= audio.shape[0]
    assert np.all(np.isfinite(output))


def test_convolution_engine_cross_channel_ir_matrix(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((256, 2), dtype=np.float32)
    audio[0, 0] = 1.0
    audio[0, 1] = 1.0

    # output-major packed matrix for 2-in x 2-out:
    # ch0 = h(out0,in0), ch1 = h(out0,in1), ch2 = h(out1,in0), ch3 = h(out1,in1)
    ir = np.zeros((16, 4), dtype=np.float32)
    ir[0, 0] = 1.0
    ir[0, 1] = 0.5
    ir[0, 2] = 0.25
    ir[0, 3] = 1.0
    ir_path = tmp_path / "matrix_ir.wav"
    sf.write(str(ir_path), ir, sr)

    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            wet=1.0,
            dry=0.0,
            ir_path=str(ir_path),
            ir_normalize="none",
            ir_matrix_layout="output-major",
            partition_size=128,
            tail_limit=None,
            threads=1,
            device="cpu",
        )
    )
    out = engine.process(audio, sr=sr)

    assert out.shape[1] == 2
    assert out.shape[0] >= audio.shape[0]
    assert np.isclose(out[0, 0], 1.5, atol=5e-5)
    assert np.isclose(out[0, 1], 1.25, atol=5e-5)


def test_convolution_engine_invalid_ir_path(tmp_path: Path) -> None:
    invalid_path = tmp_path / "does_not_exist.wav"
    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            ir_path=str(invalid_path),
            wet=1.0,
            dry=0.0,
        )
    )
    audio = np.zeros((100, 1), dtype=np.float32)
    with pytest.raises(sf.LibsndfileError) as excinfo:
        engine.process(audio, sr=48_000)
    assert "Error opening" in str(excinfo.value)


def test_convolution_engine_non_audio_file(tmp_path: Path) -> None:
    text_file = tmp_path / "not_audio.txt"
    text_file.write_text("This is not an audio file")

    engine = ConvolutionReverbEngine(
        ConvolutionReverbConfig(
            ir_path=str(text_file),
            wet=1.0,
            dry=0.0,
        )
    )
    audio = np.zeros((100, 1), dtype=np.float32)
    with pytest.raises(sf.LibsndfileError) as excinfo:
        engine.process(audio, sr=48_000)
    assert "Error opening" in str(excinfo.value)
