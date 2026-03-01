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
    fdn_stereo_inject: float,
    seed: int,
) -> AudioArray:
    """Generate IR by feeding an impulse through the algorithmic FDN engine.

    ``fdn_lines`` is currently reserved for future topologies; v0.4 keeps the
    core engine line count fixed for runtime stability.
    """
    _ = fdn_lines  # Reserved for future engine topology variants.
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
            wet=1.0,
            dry=0.0,
            block_size=4096,
        )
    )

    out = engine.process(impulse, sr)

    matrix_name = fdn_matrix.strip().lower()
    if ch > 1 and matrix_name in {"householder", "random_orthogonal"}:
        if matrix_name == "householder":
            v = np.ones((ch, 1), dtype=np.float32)
            matrix = np.eye(ch, dtype=np.float32) - (2.0 / ch) * (v @ v.T)
        else:
            rng = np.random.default_rng(seed)
            base = rng.standard_normal((ch, ch)).astype(np.float32)
            q, _ = np.linalg.qr(base)
            matrix = q.astype(np.float32)

        # Optional output-space decorrelation matrix for multi-channel spread.
        out = out @ matrix.T

    return np.asarray(out, dtype=np.float32)
