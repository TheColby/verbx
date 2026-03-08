"""FDN-mode IR synthesis via the algorithmic reverb engine.

This mode renders an impulse through :class:`AlgoReverbEngine` to derive a
late-tail IR with controllable RT60/damping/modulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from verbx.core.algo_reverb import AlgoReverbConfig, AlgoReverbEngine

AudioArray = npt.NDArray[np.float32]


def generate_fdn_ir(
    length_samples: int,
    sr: int,
    channels: int,
    rt60: float,
    damping: float,
    mod_depth_ms: float,
    mod_rate_hz: float,
    fdn_lines: int,
    fdn_matrix: str,
    fdn_tv_rate_hz: float,
    fdn_tv_depth: float,
    fdn_tv_seed: int,
    fdn_dfm_delays_ms: tuple[float, ...],
    fdn_sparse: bool,
    fdn_sparse_degree: int,
    fdn_cascade: bool,
    fdn_cascade_mix: float,
    fdn_cascade_delay_scale: float,
    fdn_cascade_rt60_ratio: float,
    fdn_rt60_low: float | None,
    fdn_rt60_mid: float | None,
    fdn_rt60_high: float | None,
    fdn_xover_low_hz: float,
    fdn_xover_high_hz: float,
    fdn_link_filter: str,
    fdn_link_filter_hz: float,
    fdn_link_filter_mix: float,
    fdn_graph_topology: str,
    fdn_graph_degree: int,
    fdn_graph_seed: int,
    fdn_stereo_inject: float,
    seed: int,
) -> AudioArray:
    """Generate IR by feeding an impulse through the algorithmic FDN engine.

    This path reuses the algorithmic engine so matrix/topology options align
    with ``verbx render --engine algo`` behavior.
    """
    n = max(1, length_samples)
    ch = max(1, channels)

    impulse = np.zeros((n, ch), dtype=np.float32)
    impulse[0, 0] = 1.0
    if ch > 1:
        inject = float(np.clip(fdn_stereo_inject, 0.0, 1.0))
        impulse[0, 1] = np.float32(inject)

    engine = AlgoReverbEngine(
        AlgoReverbConfig(
            rt60=max(0.1, rt60),
            pre_delay_ms=0.0,
            damping=np.clip(damping, 0.0, 1.0),
            width=1.0,
            mod_depth_ms=max(0.0, mod_depth_ms),
            mod_rate_hz=max(0.0, mod_rate_hz),
            fdn_lines=max(1, int(fdn_lines)),
            fdn_matrix=fdn_matrix.strip().lower().replace("-", "_"),
            fdn_tv_rate_hz=max(0.0, float(fdn_tv_rate_hz)),
            fdn_tv_depth=float(np.clip(fdn_tv_depth, 0.0, 1.0)),
            fdn_tv_seed=int(fdn_tv_seed if fdn_tv_seed != 0 else seed),
            fdn_dfm_delays_ms=tuple(float(value) for value in fdn_dfm_delays_ms),
            fdn_sparse=bool(fdn_sparse),
            fdn_sparse_degree=max(1, int(fdn_sparse_degree)),
            fdn_cascade=bool(fdn_cascade),
            fdn_cascade_mix=float(np.clip(fdn_cascade_mix, 0.0, 1.0)),
            fdn_cascade_delay_scale=float(np.clip(fdn_cascade_delay_scale, 0.2, 1.0)),
            fdn_cascade_rt60_ratio=float(np.clip(fdn_cascade_rt60_ratio, 0.1, 1.0)),
            fdn_rt60_low=fdn_rt60_low,
            fdn_rt60_mid=fdn_rt60_mid,
            fdn_rt60_high=fdn_rt60_high,
            fdn_xover_low_hz=max(20.0, float(fdn_xover_low_hz)),
            fdn_xover_high_hz=max(100.0, float(fdn_xover_high_hz)),
            fdn_link_filter=fdn_link_filter.strip().lower().replace("-", "_"),
            fdn_link_filter_hz=max(20.0, float(fdn_link_filter_hz)),
            fdn_link_filter_mix=float(np.clip(fdn_link_filter_mix, 0.0, 1.0)),
            fdn_graph_topology=fdn_graph_topology.strip().lower().replace("-", "_"),
            fdn_graph_degree=max(1, int(fdn_graph_degree)),
            fdn_graph_seed=int(fdn_graph_seed if fdn_graph_seed != 0 else seed),
            wet=1.0,
            dry=0.0,
            block_size=4096,
        )
    )

    out = engine.process(impulse, sr)

    return np.asarray(out, dtype=np.float32)
