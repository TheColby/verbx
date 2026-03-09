from __future__ import annotations

from pathlib import Path

import numpy as np
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
    audio = np.random.default_rng(0).standard_normal((4096, 2)).astype(np.float64) * 0.1

    output = engine.process(audio, sr=48_000)

    assert isinstance(output, np.ndarray)
    assert output.shape == audio.shape
    assert output.dtype == np.float64
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
    audio = np.random.default_rng(7).standard_normal((2048, 1)).astype(np.float64) * 0.05
    output = engine.process(audio, sr=48_000)
    assert output.shape == audio.shape
    assert output.dtype == np.float64
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
    audio = (rng.standard_normal((2048, 2)).astype(np.float64)) * 0.05

    ir = np.zeros((1024, 1), dtype=np.float64)
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
    assert output.dtype == np.float64
    assert output.shape[1] == audio.shape[1]
    assert output.shape[0] >= audio.shape[0]
    assert np.all(np.isfinite(output))


def test_convolution_engine_cross_channel_ir_matrix(tmp_path: Path) -> None:
    sr = 48_000
    audio = np.zeros((256, 2), dtype=np.float64)
    audio[0, 0] = 1.0
    audio[0, 1] = 1.0

    # output-major packed matrix for 2-in x 2-out:
    # ch0 = h(out0,in0), ch1 = h(out0,in1), ch2 = h(out1,in0), ch3 = h(out1,in1)
    ir = np.zeros((16, 4), dtype=np.float64)
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


def test_algo_engine_matrix_families() -> None:
    for matrix_type in [
        "hadamard",
        "householder",
        "random_orthogonal",
        "circulant",
        "elliptic",
    ]:
        engine = AlgoReverbEngine(
            AlgoReverbConfig(
                rt60=20.0,
                fdn_lines=4,
                fdn_matrix=matrix_type,
                block_size=256,
            )
        )
        audio = np.random.default_rng(42).standard_normal((1024, 2)).astype(np.float64) * 0.1
        output = engine.process(audio, sr=48_000)
        assert output.shape == audio.shape
        assert output.dtype == np.float64
        assert np.all(np.isfinite(output))


def test_algo_engine_graph_matrix_mode() -> None:
    for topology in ["ring", "path", "star", "random"]:
        engine = AlgoReverbEngine(
            AlgoReverbConfig(
                rt60=18.0,
                fdn_lines=10,
                fdn_matrix="graph",
                fdn_graph_topology=topology,
                fdn_graph_degree=3,
                fdn_graph_seed=314,
                block_size=256,
            )
        )
        audio = np.random.default_rng(11).standard_normal((1536, 1)).astype(np.float64) * 0.05
        output = engine.process(audio, sr=48_000)
        assert output.shape == audio.shape
        assert output.dtype == np.float64
        assert np.all(np.isfinite(output))
        assert "graph" in engine.backend_name()


def test_algo_engine_tv_unitary_and_dfm() -> None:
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=18.0,
            fdn_lines=4,
            fdn_matrix="tv_unitary",
            fdn_tv_rate_hz=0.15,
            fdn_tv_depth=0.4,
            fdn_dfm_delays_ms=(0.5, 0.7, 0.9, 1.1),
            block_size=256,
        )
    )
    audio = np.random.default_rng(123).standard_normal((1536, 1)).astype(np.float64) * 0.05
    output = engine.process(audio, sr=48_000)
    assert output.shape == audio.shape
    assert output.dtype == np.float64
    assert np.all(np.isfinite(output))


def test_algo_engine_sparse_high_order_mode() -> None:
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=22.0,
            fdn_lines=32,
            fdn_matrix="hadamard",
            fdn_sparse=True,
            fdn_sparse_degree=4,
            block_size=256,
        )
    )
    audio = np.random.default_rng(321).standard_normal((2048, 1)).astype(np.float64) * 0.05
    output = engine.process(audio, sr=48_000)
    assert output.shape == audio.shape
    assert output.dtype == np.float64
    assert np.all(np.isfinite(output))
    assert engine.backend_name() in {"cpu-python-fdn-sparse", "cpu-numba-fdn-sparse"}


def test_algo_engine_cascaded_fdn_mode() -> None:
    audio = np.random.default_rng(456).standard_normal((4096, 1)).astype(np.float64) * 0.05
    base_engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=20.0,
            pre_delay_ms=0.0,
            fdn_lines=8,
            comb_delays_ms=(3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0),
            fdn_matrix="hadamard",
            block_size=256,
        )
    )
    cascade_engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=20.0,
            pre_delay_ms=0.0,
            fdn_lines=8,
            comb_delays_ms=(3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0),
            fdn_matrix="hadamard",
            fdn_cascade=True,
            fdn_cascade_mix=0.7,
            fdn_cascade_delay_scale=0.4,
            fdn_cascade_rt60_ratio=0.5,
            block_size=256,
        )
    )

    out_base = base_engine.process(audio, sr=48_000)
    out_cascade = cascade_engine.process(audio, sr=48_000)

    assert out_cascade.shape == audio.shape
    assert out_cascade.dtype == np.float64
    assert np.all(np.isfinite(out_cascade))
    assert "cascade" in cascade_engine.backend_name()
    delta = float(np.mean(np.abs(out_base - out_cascade)))
    assert delta > 1e-5


def test_algo_engine_multiband_decay_mode() -> None:
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=18.0,
            fdn_lines=12,
            fdn_matrix="hadamard",
            fdn_rt60_low=24.0,
            fdn_rt60_mid=14.0,
            fdn_rt60_high=6.0,
            fdn_xover_low_hz=220.0,
            fdn_xover_high_hz=3600.0,
            block_size=256,
        )
    )
    audio = np.random.default_rng(999).standard_normal((2048, 1)).astype(np.float64) * 0.04
    output = engine.process(audio, sr=48_000)
    assert output.shape == audio.shape
    assert output.dtype == np.float64
    assert np.all(np.isfinite(output))
    assert engine.backend_name() == "cpu-python-fdn-multiband"


def test_algo_engine_filter_feedback_mode() -> None:
    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=16.0,
            fdn_lines=10,
            fdn_matrix="circulant",
            fdn_link_filter="highpass",
            fdn_link_filter_hz=1800.0,
            fdn_link_filter_mix=0.7,
            block_size=256,
        )
    )
    audio = np.random.default_rng(212).standard_normal((2048, 1)).astype(np.float64) * 0.05
    output = engine.process(audio, sr=48_000)
    assert output.shape == audio.shape
    assert output.dtype == np.float64
    assert np.all(np.isfinite(output))
    assert engine.backend_name() == "cpu-python-fdn-linkfilter"


def test_algo_engine_surround_decorrelation_controls() -> None:
    signal = np.random.default_rng(77).standard_normal((4096, 1)).astype(np.float64) * 0.05
    audio = np.repeat(signal, 6, axis=1)

    base_engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=4.0,
            fdn_lines=8,
            output_layout="5.1",
            block_size=256,
        )
    )
    deco_engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=4.0,
            fdn_lines=8,
            output_layout="5.1",
            algo_decorrelation_front=0.35,
            algo_decorrelation_rear=0.65,
            algo_decorrelation_top=0.0,
            block_size=256,
        )
    )

    out_base = base_engine.process(audio, sr=48_000)
    out_deco = deco_engine.process(audio, sr=48_000)

    assert out_base.shape == audio.shape
    assert out_deco.shape == audio.shape
    assert np.all(np.isfinite(out_deco))
    # Identical-channel source should decorrelate rear from front when enabled.
    base_delta = float(np.mean(np.abs(out_base[:, 0] - out_base[:, 4])))
    deco_delta = float(np.mean(np.abs(out_deco[:, 0] - out_deco[:, 4])))
    assert deco_delta > base_delta
