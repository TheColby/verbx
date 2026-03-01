"""Acceleration and platform tuning helpers.

These helpers provide conservative runtime detection and fallback behavior for
CPU, CUDA, and Apple Silicon environments.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Literal

DeviceName = Literal["auto", "cpu", "cuda", "mps"]

LOGGER = logging.getLogger(__name__)


def configure_cpu_threads(threads: int | None) -> None:
    """Configure common CPU thread env vars for FFT/BLAS-backed libraries."""
    if threads is None or threads < 1:
        return

    value = str(int(threads))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = value


def is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon (arm64 Darwin)."""
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def cuda_available() -> bool:
    """Detect CUDA support through CuPy availability/runtime probe."""
    try:
        import cupy as cp  # type: ignore[import-untyped]

        count = int(cp.cuda.runtime.getDeviceCount())
        return count > 0
    except Exception:
        return False


def resolve_device(requested: DeviceName) -> DeviceName:
    """Resolve runtime device with graceful fallbacks."""
    if requested == "auto":
        if cuda_available():
            return "cuda"
        if is_apple_silicon():
            return "mps"
        return "cpu"

    if requested == "cuda" and not cuda_available():
        LOGGER.warning("CUDA requested but CuPy/CUDA runtime unavailable. Falling back to CPU.")
        return "cpu"

    if requested == "mps" and not is_apple_silicon():
        LOGGER.warning("MPS requested but host is not Apple Silicon. Falling back to CPU.")
        return "cpu"

    return requested
