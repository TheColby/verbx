"""Pure delay-layout and fractional-read helpers for algorithmic FDNs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float64]


def resolve_fdn_delays_ms(
    configured_delays_ms: tuple[float, ...],
    *,
    line_count: int,
    defaults_ms: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Resolve positive FDN delays, extending the default sequence as needed."""
    if configured_delays_ms:
        return np.asarray([max(0.1, float(value)) for value in configured_delays_ms], dtype=np.float64)

    requested = max(1, int(line_count))
    delays = np.asarray(defaults_ms, dtype=np.float64).tolist()
    while len(delays) < requested:
        next_delay = (delays[-1] * 1.11) + 1.25
        delays.append(next_delay if next_delay > delays[-1] else delays[-1] + 0.25)
    return np.asarray(delays[:requested], dtype=np.float64)


def resolve_comb_cloud_delays_ms(
    configured_delays_ms: tuple[float, ...],
    *,
    enabled: bool,
    count: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Build a deterministic, ordered comb-cloud delay layout."""
    if not enabled:
        return np.zeros((0,), dtype=np.float64)
    if configured_delays_ms:
        return np.asarray([max(0.1, float(value)) for value in configured_delays_ms], dtype=np.float64)

    requested = max(1, int(count))
    rng = np.random.default_rng(int(seed))
    base = np.linspace(7.5, 89.0, requested, dtype=np.float64)
    delays = np.clip(
        (base * rng.uniform(0.94, 1.06, size=requested))
        + rng.uniform(-1.5, 1.5, size=requested),
        3.0,
        120.0,
    )
    delays.sort()
    for index in range(1, requested):
        delays[index] = max(delays[index], delays[index - 1] + 0.35)
    return np.asarray(np.clip(delays, 3.0, 120.0), dtype=np.float64)


def resolve_dfm_delays_ms(
    configured_delays_ms: tuple[float, ...], *, line_count: int
) -> npt.NDArray[np.float64]:
    """Resolve DFM delays, allowing a single value to broadcast to all lines."""
    if not configured_delays_ms:
        return np.zeros((0,), dtype=np.float64)
    delays = [max(0.05, float(value)) for value in configured_delays_ms]
    if len(delays) == 1 and line_count > 1:
        delays *= line_count
    if len(delays) != line_count:
        raise ValueError(
            "fdn_dfm_delays_ms length must be 1 or match FDN line count "
            f"({line_count}), got {len(delays)}"
        )
    return np.asarray(delays, dtype=np.float64)


def resolve_diffusion_delays_ms(
    configured_delays_ms: tuple[float, ...],
    *,
    stage_count: int,
    defaults_ms: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Resolve all-pass diffusion delays, extending defaults deterministically."""
    requested = max(0, int(stage_count))
    if requested == 0:
        return np.zeros((0,), dtype=np.float64)
    delays = (
        [max(0.1, float(value)) for value in configured_delays_ms]
        if configured_delays_ms
        else np.asarray(defaults_ms, dtype=np.float64).tolist()
    )
    while len(delays) < requested:
        next_delay = (delays[-1] * 1.28) + 0.75
        delays.append(next_delay if next_delay > delays[-1] else delays[-1] + 0.2)
    return np.asarray(delays[:requested], dtype=np.float64)


def read_fractional_delay(
    buffer: AudioArray, write_index: int, delay_samples: float
) -> np.float64:
    """Read a circular delay line using deterministic linear interpolation."""
    size = buffer.shape[0]
    read_pos = (float(write_index) - delay_samples) % size
    index0 = int(np.floor(read_pos))
    index1 = (index0 + 1) % size
    fraction = np.float64(read_pos - index0)
    return np.float64((np.float64(1.0) - fraction) * buffer[index0] + fraction * buffer[index1])
