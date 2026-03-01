"""Modal IR synthesis based on decaying sinusoidal resonances.

Each mode is a damped sinusoid with randomized frequency/phase/Q, optionally
nudged toward harmonic targets derived from user tuning or source analysis.
"""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt

from verbx.ir.tuning import tune_frequency_to_targets

AudioArray = npt.NDArray[np.float32]

_TUNING_RE = re.compile(r"^\s*A4\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)


def _parse_tuning(tuning: str) -> float:
    """Parse strings like ``A4=440``; fallback to 440 Hz."""
    match = _TUNING_RE.match(tuning)
    if not match:
        return 440.0
    return float(match.group(1))


def generate_modal_ir(
    length_samples: int,
    sr: int,
    channels: int,
    rt60: float,
    seed: int,
    tuning: str,
    modal_count: int,
    modal_q_min: float,
    modal_q_max: float,
    modal_spread_cents: float,
    modal_low_hz: float,
    modal_high_hz: float,
    f0_hz: float | None = None,
    harmonic_targets_hz: tuple[float, ...] = (),
    align_strength: float = 0.7,
) -> AudioArray:
    """Generate deterministic modal-bank IR.

    The modal bank is intentionally synthetic and musical, not a room-physics
    solver. It works well for tonal or resonant IR design.
    """
    rng = np.random.default_rng(seed)
    n = max(1, length_samples)
    ch = max(1, channels)
    t = np.arange(n, dtype=np.float64) / float(sr)

    a4 = _parse_tuning(tuning)
    base_ref = a4 / 440.0

    low_hz = max(20.0, modal_low_hz)
    high_hz = max(low_hz + 1.0, modal_high_hz)
    spread = float(modal_spread_cents)
    q_min = max(0.5, modal_q_min)
    q_max = max(q_min, modal_q_max)

    out = np.zeros((n, ch), dtype=np.float64)

    modes = max(1, modal_count)
    for _ in range(modes):
        freq = float(np.exp(rng.uniform(np.log(low_hz), np.log(high_hz))))
        cents = float(rng.uniform(-spread, spread))
        freq *= 2.0 ** (cents / 1200.0)
        freq *= base_ref
        freq = tune_frequency_to_targets(
            freq_hz=freq,
            f0_hz=f0_hz,
            harmonic_targets_hz=harmonic_targets_hz,
            align_strength=align_strength,
            max_hz=high_hz,
        )

        q = float(rng.uniform(q_min, q_max))
        # Blend Q-based decay with global RT target so tails stay plausible.
        tau_q = q / (np.pi * max(freq, 1.0))
        tau_rt = max(0.01, rt60 / 6.91)
        tau = min(max(tau_q, 0.01), max(tau_rt * 2.0, 0.03))

        amp = float(rng.uniform(0.25, 1.0)) / np.sqrt(float(modes))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        env = np.exp(-t / tau)
        mode = amp * np.sin((2.0 * np.pi * freq * t) + phase) * env

        if ch == 1:
            out[:, 0] += mode
            continue

        pan = float(rng.uniform(-1.0, 1.0))
        left = mode * np.sqrt(0.5 * (1.0 - pan))
        right = mode * np.sqrt(0.5 * (1.0 + pan))
        out[:, 0] += left
        out[:, 1] += right

        for idx in range(2, ch):
            out[:, idx] += mode / ch

    # Low-level air bed.
    bed = rng.standard_normal((n, ch)) * np.exp(-t / max(0.2, rt60 / 5.0))[:, np.newaxis]
    out += 0.02 * bed

    out[0, :] += 1.0
    return np.asarray(out, dtype=np.float32)
