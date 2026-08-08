"""Bounded offline modal finite-element solvers for electro-mechanical reverbs.

The models intentionally solve a small, deterministic structural system and
render its modal impulse response.  They are suitable for offline sound design,
not a calibrated model of any particular hardware tank or plate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh
from scipy.signal import fftconvolve

AudioArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ModalFEResponse:
    """A synthesized causal impulse response plus reproducibility metadata."""

    impulse: AudioArray
    report: dict[str, float | int | str]


def render_modal_response(
    *,
    model: str,
    sr: int,
    samples: int,
    rt60: float,
    damping: float,
    spring_specs: tuple[dict[str, float], ...] = (),
    spring_nodes: int = 24,
    spring_modes: int = 24,
    spring_coupling: float = 0.08,
    spring_loss: float = 0.30,
    plate_width_m: float = 1.8,
    plate_height_m: float = 1.2,
    plate_thickness_mm: float = 0.6,
    plate_density_kg_m3: float = 7850.0,
    plate_youngs_gpa: float = 200.0,
    plate_poisson_ratio: float = 0.29,
    plate_tension_n: float = 0.0,
    plate_pickup_x: float = 0.72,
    plate_pickup_y: float = 0.38,
    plate_nx: int = 12,
    plate_ny: int = 8,
    plate_modes: int = 32,
    plate_loss: float = 0.24,
) -> ModalFEResponse:
    """Solve a bounded structural model and synthesize its damped modal IR."""
    if model == "spring":
        return _spring_response(
            sr=sr,
            samples=samples,
            rt60=rt60,
            damping=damping,
            specs=spring_specs,
            nodes=spring_nodes,
            modes=spring_modes,
            coupling=spring_coupling,
            loss=spring_loss,
        )
    if model == "plate":
        return _plate_response(
            sr=sr,
            samples=samples,
            rt60=rt60,
            damping=damping,
            width=plate_width_m,
            height=plate_height_m,
            thickness=plate_thickness_mm / 1000.0,
            density=plate_density_kg_m3,
            youngs=plate_youngs_gpa * 1e9,
            poisson=plate_poisson_ratio,
            tension=plate_tension_n,
            pickup_x=plate_pickup_x,
            pickup_y=plate_pickup_y,
            nx=plate_nx,
            ny=plate_ny,
            modes=plate_modes,
            loss=plate_loss,
        )
    raise ValueError(f"Modal FE is only available for spring or plate, got {model!r}")


def apply_modal_response(audio: AudioArray, response: ModalFEResponse) -> AudioArray:
    """Convolve each channel with a modal response, preserving input duration."""
    out = np.zeros_like(audio, dtype=np.float64)
    for channel in range(audio.shape[1]):
        out[:, channel] = fftconvolve(audio[:, channel], response.impulse, mode="full")[
            : audio.shape[0]
        ]
    return np.asarray(out, dtype=np.float64)


def _modal_impulse(
    frequencies: AudioArray,
    weights: AudioArray,
    *,
    sr: int,
    samples: int,
    rt60: float,
    loss: float,
    report: dict[str, float | int | str],
) -> ModalFEResponse:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(frequencies)
        & np.isfinite(weights)
        & (frequencies > 1.0)
        & (frequencies < sr * 0.46)
    )
    frequencies, weights = frequencies[valid], weights[valid]
    time = np.arange(samples, dtype=np.float64) / float(sr)
    if frequencies.size == 0:
        return ModalFEResponse(np.zeros(samples, dtype=np.float64), {**report, "active_modes": 0})
    omega = 2.0 * np.pi * frequencies[:, None]
    # RT60 controls the base energy decay; loss adds physically useful HF loss.
    sigma = (3.0 * log(10.0) / max(rt60, 0.02)) * (
        1.0 + loss * frequencies / max(float(np.max(frequencies)), 1.0)
    )
    damped = np.sqrt(np.maximum(omega[:, 0] ** 2 - sigma**2, 1.0))[:, None]
    impulse = np.sum(
        (weights / damped[:, 0])[:, None] * np.exp(-sigma[:, None] * time) * np.sin(damped * time),
        axis=0,
    )
    impulse[0] += 0.15
    peak = float(np.max(np.abs(impulse)))
    if peak > 0.0:
        impulse *= 0.85 / peak
    return ModalFEResponse(
        np.asarray(impulse, dtype=np.float64),
        {
            **report,
            "active_modes": int(frequencies.size),
            "lowest_hz": float(np.min(frequencies)),
            "highest_hz": float(np.max(frequencies)),
        },
    )


def _spring_response(
    *,
    sr: int,
    samples: int,
    rt60: float,
    damping: float,
    specs: tuple[dict[str, float], ...],
    nodes: int,
    modes: int,
    coupling: float,
    loss: float,
) -> ModalFEResponse:
    count, nodes = max(1, len(specs)), int(np.clip(nodes, 4, 128))
    size = count * nodes
    mass = np.zeros(size, dtype=np.float64)
    stiffness = np.zeros((size, size), dtype=np.float64)
    excitation = np.zeros(size, dtype=np.float64)
    pickup = np.zeros(size, dtype=np.float64)
    for spring_index, spec in enumerate(specs):
        start = spring_index * nodes
        total_mass = max(1e-5, spec["mass_g"] / 1000.0)
        compliance = max(1e-8, spec["compliance_mm_n"] / 1000.0)
        segment_k = (1.0 / compliance) * (nodes - 1)
        mass[start : start + nodes] = total_mass / nodes
        for i in range(nodes):
            stiffness[start + i, start + i] += segment_k
            if i > 0:
                stiffness[start + i, start + i] += segment_k
                stiffness[start + i, start + i - 1] -= segment_k
                stiffness[start + i - 1, start + i] -= segment_k
        # Clamp the driven end; tension adds a stable axial stiffness term.
        stiffness[start, start] += segment_k * 4.0
        stiffness[start : start + nodes, start : start + nodes] += (
            np.eye(nodes) * max(0.0, spec["tension_n"]) / max(spec["length_m"], 0.05)
        )
        excitation[start + 1] = 1.0 / count
        pickup[start + nodes - 1] = (1.0 - 0.5 * spec["damping"]) / count
    if count > 1 and coupling > 0.0:
        k_couple = float(coupling) * float(np.mean(np.diag(stiffness)))
        for spring_index in range(count - 1):
            left, right = (spring_index + 1) * nodes - 1, (spring_index + 1) * nodes
            stiffness[left, left] += k_couple
            stiffness[right, right] += k_couple
            stiffness[left, right] -= k_couple
            stiffness[right, left] -= k_couple
    values, vectors = eigh(
        stiffness,
        np.diag(mass),
        subset_by_index=[0, min(size - 1, int(np.clip(modes, 1, 128)) - 1)],
    )
    frequencies = np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)
    weights = (vectors.T @ excitation) * (vectors.T @ pickup)
    return _modal_impulse(
        frequencies,
        weights,
        sr=sr,
        samples=samples,
        rt60=rt60,
        loss=float(np.clip(loss + damping * 0.35, 0.0, 2.0)),
        report={
            "solver": "modal-fe",
            "model": "spring",
            "nodes": size,
            "requested_modes": int(modes),
            "spring_count": count,
        },
    )


def _plate_response(
    *,
    sr: int,
    samples: int,
    rt60: float,
    damping: float,
    width: float,
    height: float,
    thickness: float,
    density: float,
    youngs: float,
    poisson: float,
    tension: float,
    pickup_x: float,
    pickup_y: float,
    nx: int,
    ny: int,
    modes: int,
    loss: float,
) -> ModalFEResponse:
    nx, ny = int(np.clip(nx, 4, 32)), int(np.clip(ny, 4, 32))
    dx, dy = width / (nx + 1), height / (ny + 1)
    size = nx * ny
    laplacian = np.zeros((size, size), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            laplacian[i, i] = 2.0 / dx**2 + 2.0 / dy**2
            for xx, yy, coeff in (
                (x - 1, y, -1.0 / dx**2),
                (x + 1, y, -1.0 / dx**2),
                (x, y - 1, -1.0 / dy**2),
                (x, y + 1, -1.0 / dy**2),
            ):
                if 0 <= xx < nx and 0 <= yy < ny:
                    laplacian[i, yy * nx + xx] = coeff
    rigidity = youngs * thickness**3 / (12.0 * (1.0 - poisson**2))
    stiffness = rigidity * (laplacian.T @ laplacian) + max(0.0, tension) * laplacian
    node_mass = max(1e-12, density * thickness * dx * dy)
    values, vectors = eigh(
        stiffness,
        np.eye(size) * node_mass,
        subset_by_index=[0, min(size - 1, int(np.clip(modes, 1, 128)) - 1)],
    )
    excitation = _grid_weights(nx, ny, 0.28, 0.64)
    pickup = _grid_weights(nx, ny, pickup_x, pickup_y)
    frequencies = np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)
    weights = (vectors.T @ excitation) * (vectors.T @ pickup)
    return _modal_impulse(
        frequencies,
        weights,
        sr=sr,
        samples=samples,
        rt60=rt60,
        loss=float(np.clip(loss + damping * 0.2, 0.0, 2.0)),
        report={
            "solver": "modal-fe",
            "model": "plate",
            "nodes": size,
            "grid_x": nx,
            "grid_y": ny,
            "requested_modes": int(modes),
        },
    )


def _grid_weights(nx: int, ny: int, x: float, y: float) -> AudioArray:
    """Bilinear pickup/excitation weights on interior clamped-grid nodes."""
    gx, gy = np.clip(x, 0.0, 1.0) * (nx - 1), np.clip(y, 0.0, 1.0) * (ny - 1)
    x0, y0 = int(gx), int(gy)
    weights = np.zeros(nx * ny, dtype=np.float64)
    for xx in (x0, min(nx - 1, x0 + 1)):
        for yy in (y0, min(ny - 1, y0 + 1)):
            weights[yy * nx + xx] += (1.0 - abs(gx - xx)) * (1.0 - abs(gy - yy))
    return weights
