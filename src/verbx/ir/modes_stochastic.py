"""Stochastic IR mode synthesis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]


def generate_stochastic_ir(
    length_samples: int,
    sr: int,
    channels: int,
    rt60_low: float,
    rt60_high: float,
    damping: float,
    diffusion: float,
    density: float,
    seed: int,
) -> AudioArray:
    """Generate deterministic stochastic tail IR."""
    rng = np.random.default_rng(seed)
    n = max(1, length_samples)
    ch = max(1, channels)

    noise = rng.standard_normal((n, ch), dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / float(sr)

    rt_low = max(0.1, rt60_low)
    rt_high = max(rt_low, rt60_high)
    rt_curve = np.linspace(rt_low, rt_high, n, dtype=np.float64)
    envelope = np.power(10.0, (-3.0 * t) / rt_curve).astype(np.float32)

    out = noise * envelope[:, np.newaxis] * np.float32(np.clip(density, 0.01, 3.0))

    diff = float(np.clip(diffusion, 0.0, 1.0))
    if diff > 0.0:
        delays = [31, 73, 151, 313]
        for delay in delays:
            if delay >= n:
                break
            shifted = np.zeros_like(out)
            shifted[delay:, :] = out[:-delay, :]
            out = ((1.0 - diff * 0.35) * out) + ((diff * 0.35) * shifted)

    damp = float(np.clip(damping, 0.0, 1.0))
    alpha = np.float32(0.02 + (0.95 * damp))
    state = np.zeros(ch, dtype=np.float32)
    for i in range(n):
        state = ((1.0 - alpha) * out[i, :]) + (alpha * state)
        out[i, :] = state

    out[0, :] += np.float32(1.0)
    return np.asarray(out, dtype=np.float32)
