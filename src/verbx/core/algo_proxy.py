"""Helpers for turning algorithmic reverb settings into reusable proxy IRs."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from verbx.config import RenderConfig
from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine
from verbx.io.audio import write_audio

AudioArray = npt.NDArray[np.float64]


def render_algo_proxy_ir(
    *,
    config: RenderConfig,
    sr: int,
    input_channels: int,
) -> tuple[Path, int]:
    """Render an algorithmic response as a matrix IR temp file.

    Real-time monitoring cannot safely call the current algorithmic engine per
    callback because the offline ``process()`` API rebuilds state each time. We
    sidestep that by rendering a deterministic proxy IR once, then using the
    streaming convolution path for live playback.
    """
    tail_seconds = min(
        float(config.algo_proxy_ir_max_seconds),
        float(_algo_tail_padding_seconds(config, sr)),
    )
    ir_samples = max(64, int(np.ceil(max(0.1, tail_seconds) * float(sr))))
    if ir_samples <= 0:
        ir_samples = 64

    proxy_config = RenderConfig(**asdict(config))
    proxy_config.engine = "algo"
    proxy_config.wet = 1.0
    proxy_config.dry = 0.0
    proxy_config.device = "cpu"
    proxy_config.algo_stream = False
    proxy_config.algo_gpu_proxy = False

    responses: list[AudioArray] = []
    output_channels = input_channels
    for in_ch in range(max(1, int(input_channels))):
        impulse = np.zeros((ir_samples, input_channels), dtype=np.float64)
        impulse[0, in_ch] = 1.0
        response = _build_algo_engine(proxy_config).process(impulse, sr)
        output_channels = int(response.shape[1])
        responses.append(np.asarray(response, dtype=np.float64))

    ir_matrix = np.zeros((ir_samples, output_channels * input_channels), dtype=np.float64)
    for in_ch, response in enumerate(responses):
        for out_ch in range(output_channels):
            packed_ch = (out_ch * input_channels) + in_ch
            ir_matrix[:, packed_ch] = response[:, out_ch]

    fd, raw_path = tempfile.mkstemp(prefix="verbx_algo_proxy_ir_", suffix=".wav")
    os.close(fd)
    Path(raw_path).unlink(missing_ok=True)
    ir_path = Path(raw_path)
    write_audio(str(ir_path), ir_matrix, sr, subtype="DOUBLE")
    return ir_path, output_channels


def _algo_tail_padding_seconds(config: RenderConfig, sr: int) -> float:
    """Mirror pipeline tail-padding rules for proxy IR generation."""
    pre_delay = max(0.0, float(config.pre_delay_ms)) / 1000.0
    base_tail = max(0.25, float(config.rt60) + pre_delay)

    if not config.shimmer:
        return base_tail

    feedback = float(np.clip(config.shimmer_feedback, 0.0, 0.98))
    if feedback < 1e-9:
        return base_tail

    block_seconds = max(1, int(config.block_size)) / max(1.0, float(sr))
    blocks_to_silence = int(np.ceil(np.log(1e-6) / np.log(feedback)))
    shimmer_tail = blocks_to_silence * block_seconds
    return base_tail + shimmer_tail


def _build_algo_engine(config: RenderConfig) -> AlgoReverbEngine:
    """Build the algorithmic engine without depending on pipeline internals."""
    return AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=config.rt60,
            pre_delay_ms=config.pre_delay_ms,
            damping=config.damping,
            width=config.width,
            mod_depth_ms=config.mod_depth_ms,
            mod_rate_hz=config.mod_rate_hz,
            allpass_stages=config.allpass_stages,
            allpass_gain=config.allpass_gain,
            allpass_gains=config.allpass_gains,
            allpass_delays_ms=config.allpass_delays_ms,
            comb_delays_ms=config.comb_delays_ms,
            fdn_lines=config.fdn_lines,
            fdn_matrix=config.fdn_matrix,
            fdn_tv_rate_hz=config.fdn_tv_rate_hz,
            fdn_tv_depth=config.fdn_tv_depth,
            fdn_tv_seed=config.fdn_tv_seed,
            fdn_dfm_delays_ms=config.fdn_dfm_delays_ms,
            fdn_sparse=config.fdn_sparse,
            fdn_sparse_degree=config.fdn_sparse_degree,
            fdn_cascade=config.fdn_cascade,
            fdn_cascade_mix=config.fdn_cascade_mix,
            fdn_cascade_delay_scale=config.fdn_cascade_delay_scale,
            fdn_cascade_rt60_ratio=config.fdn_cascade_rt60_ratio,
            fdn_rt60_low=config.fdn_rt60_low,
            fdn_rt60_mid=config.fdn_rt60_mid,
            fdn_rt60_high=config.fdn_rt60_high,
            fdn_rt60_tilt=config.fdn_rt60_tilt,
            fdn_tonal_correction_strength=config.fdn_tonal_correction_strength,
            fdn_xover_low_hz=config.fdn_xover_low_hz,
            fdn_xover_high_hz=config.fdn_xover_high_hz,
            fdn_link_filter=config.fdn_link_filter,
            fdn_link_filter_hz=config.fdn_link_filter_hz,
            fdn_link_filter_mix=config.fdn_link_filter_mix,
            fdn_graph_topology=config.fdn_graph_topology,
            fdn_graph_degree=config.fdn_graph_degree,
            fdn_graph_seed=config.fdn_graph_seed,
            fdn_matrix_morph_to=config.fdn_matrix_morph_to,
            fdn_matrix_morph_seconds=config.fdn_matrix_morph_seconds,
            fdn_spatial_coupling_mode=config.fdn_spatial_coupling_mode,
            fdn_spatial_coupling_strength=config.fdn_spatial_coupling_strength,
            fdn_nonlinearity=config.fdn_nonlinearity,
            fdn_nonlinearity_amount=config.fdn_nonlinearity_amount,
            fdn_nonlinearity_drive=config.fdn_nonlinearity_drive,
            room_size_macro=config.room_size_macro,
            clarity_macro=config.clarity_macro,
            warmth_macro=config.warmth_macro,
            envelopment_macro=config.envelopment_macro,
            algo_decorrelation_front=config.algo_decorrelation_front,
            algo_decorrelation_rear=config.algo_decorrelation_rear,
            algo_decorrelation_top=config.algo_decorrelation_top,
            wet=config.wet,
            dry=config.dry,
            block_size=config.block_size,
            shimmer=config.shimmer,
            shimmer_semitones=config.shimmer_semitones,
            shimmer_mix=config.shimmer_mix,
            shimmer_feedback=config.shimmer_feedback,
            shimmer_highcut=config.shimmer_highcut,
            shimmer_lowcut=config.shimmer_lowcut,
            shimmer_spatial=config.shimmer_spatial,
            shimmer_spread_cents=config.shimmer_spread_cents,
            shimmer_decorrelation_ms=config.shimmer_decorrelation_ms,
            unsafe_self_oscillate=config.unsafe_self_oscillate,
            unsafe_loop_gain=config.unsafe_loop_gain,
            output_layout=config.output_layout,
            device="cpu",
        )
    )
