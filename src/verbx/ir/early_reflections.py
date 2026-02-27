"""Early reflection synthesis helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]


def generate_early_reflections(
    sr: int,
    channels: int,
    er_count: int,
    er_max_delay_ms: float,
    er_decay_shape: str,
    er_stereo_width: float,
    er_room: float,
    rng: np.random.Generator,
) -> AudioArray:
    """Generate sparse deterministic early reflections."""
    max_delay_samples = max(1, int((er_max_delay_ms / 1000.0) * sr))
    out = np.zeros((max_delay_samples + 1, channels), dtype=np.float32)

    # Direct path.
    out[0, :] = 1.0

    shape = er_decay_shape.strip().lower()
    room_scale = float(np.clip(er_room, 0.1, 3.0))
    width = float(np.clip(er_stereo_width, 0.0, 2.0))

    for _ in range(max(0, er_count)):
        delay = int(rng.integers(1, max_delay_samples + 1))
        norm = delay / max_delay_samples

        if shape == "linear":
            amp = 1.0 - norm
        elif shape == "sqrt":
            amp = np.sqrt(max(1e-6, 1.0 - norm))
        else:
            amp = float(np.exp(-3.2 * norm))

        amp *= room_scale * float(rng.uniform(0.35, 1.0))
        amp = float(np.clip(amp, 0.0, 1.2))

        if channels == 1:
            out[delay, 0] += np.float32(amp)
            continue

        pan = float(rng.uniform(-1.0, 1.0) * width)
        left = amp * (0.5 * (2.0 - max(0.0, pan)))
        right = amp * (0.5 * (2.0 + min(0.0, pan)))

        out[delay, 0] += np.float32(left)
        out[delay, 1] += np.float32(right)
        for ch in range(2, channels):
            out[delay, ch] += np.float32(amp / channels)

    return out
