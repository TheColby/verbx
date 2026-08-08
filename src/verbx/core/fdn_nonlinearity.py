"""Bounded feedback nonlinearity helpers for FDN processing."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NONLINEARITY_MODES = frozenset({"none", "tanh", "softclip"})


def normalize_nonlinearity_mode(mode: str) -> str:
    """Normalize and validate an in-loop nonlinearity identifier."""
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in NONLINEARITY_MODES:
        return normalized
    raise ValueError(f"Unsupported FDN nonlinearity mode: {mode}")


def apply_feedback_nonlinearity(
    values: npt.NDArray[np.float64],
    *,
    mode: str,
    amount: float,
    drive: float,
) -> npt.NDArray[np.float64]:
    """Apply bounded nonlinear feedback shaping without changing array shape."""
    source = np.asarray(values, dtype=np.float64)
    resolved_mode = normalize_nonlinearity_mode(mode)
    resolved_amount = float(np.clip(amount, 0.0, 1.0))
    if resolved_mode == "none" or resolved_amount <= 0.0:
        return source
    resolved_drive = float(max(1e-6, drive))
    driven = np.asarray(source * resolved_drive, dtype=np.float64)
    if resolved_mode == "tanh":
        shaped = np.tanh(driven)
    else:
        shaped = driven / (1.0 + np.abs(driven))
    normalized = np.asarray(shaped / resolved_drive, dtype=np.float64)
    blended = ((1.0 - resolved_amount) * source) + (resolved_amount * normalized)
    return np.asarray(np.clip(blended, -32.0, 32.0), dtype=np.float64)
