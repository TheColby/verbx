"""Frequency-dependent room material absorption profiles.

The coefficients are practical, rounded Sabine-style random-incidence values
for common architectural materials at octave-band center frequencies. They are
intended for deterministic IR/report generation rather than laboratory-grade
material certification; keep units explicit when adding new profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

OCTAVE_BANDS_HZ: tuple[int, ...] = (125, 250, 500, 1000, 2000, 4000)


@dataclass(frozen=True, slots=True)
class MaterialProfile:
    """Octave-band absorption and scattering profile for one surface material."""

    name: str
    absorption: tuple[float, float, float, float, float, float]
    scattering: float
    description: str

    def broadband_absorption(self) -> float:
        return float(np.mean(np.asarray(self.absorption, dtype=np.float64)))

    def to_report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "octave_bands_hz": list(OCTAVE_BANDS_HZ),
            "absorption": [float(value) for value in self.absorption],
            "mean_absorption": self.broadband_absorption(),
            "scattering": float(self.scattering),
        }


_PROFILES: dict[str, MaterialProfile] = {
    "anechoic": MaterialProfile(
        name="anechoic",
        absorption=(0.95, 0.97, 0.99, 0.99, 0.99, 0.99),
        scattering=0.35,
        description="Deep broadband absorber or anechoic wedge treatment.",
    ),
    "dead": MaterialProfile(
        name="dead",
        absorption=(0.48, 0.62, 0.76, 0.84, 0.88, 0.90),
        scattering=0.20,
        description="Very absorptive vocal booth or heavily treated small room.",
    ),
    "studio": MaterialProfile(
        name="studio",
        absorption=(0.22, 0.32, 0.46, 0.56, 0.60, 0.58),
        scattering=0.22,
        description="Balanced project-studio mix of drywall, panels, and furnishings.",
    ),
    "hall": MaterialProfile(
        name="hall",
        absorption=(0.12, 0.18, 0.26, 0.34, 0.42, 0.46),
        scattering=0.28,
        description="Moderately live performance hall surfaces and audience area.",
    ),
    "stone": MaterialProfile(
        name="stone",
        absorption=(0.01, 0.01, 0.02, 0.02, 0.03, 0.04),
        scattering=0.18,
        description="Hard stone, tile, or masonry with very low absorption.",
    ),
    "concrete": MaterialProfile(
        name="concrete",
        absorption=(0.01, 0.01, 0.02, 0.02, 0.02, 0.03),
        scattering=0.12,
        description="Painted or sealed concrete wall/floor.",
    ),
    "drywall": MaterialProfile(
        name="drywall",
        absorption=(0.29, 0.10, 0.05, 0.04, 0.07, 0.09),
        scattering=0.10,
        description="Gypsum board on studs, lightly damped at low frequencies.",
    ),
    "glass": MaterialProfile(
        name="glass",
        absorption=(0.35, 0.25, 0.18, 0.12, 0.07, 0.04),
        scattering=0.05,
        description="Large glass pane or window wall.",
    ),
    "wood": MaterialProfile(
        name="wood",
        absorption=(0.15, 0.11, 0.10, 0.07, 0.06, 0.06),
        scattering=0.16,
        description="Wood paneling or stage flooring.",
    ),
    "carpet": MaterialProfile(
        name="carpet",
        absorption=(0.08, 0.24, 0.57, 0.69, 0.71, 0.73),
        scattering=0.08,
        description="Wall-to-wall carpet on pad.",
    ),
    "curtain": MaterialProfile(
        name="curtain",
        absorption=(0.07, 0.31, 0.49, 0.75, 0.70, 0.60),
        scattering=0.12,
        description="Heavy pleated drape with air gap.",
    ),
    "plaster": MaterialProfile(
        name="plaster",
        absorption=(0.02, 0.03, 0.04, 0.05, 0.04, 0.04),
        scattering=0.10,
        description="Smooth plaster on masonry or lath.",
    ),
    "brick": MaterialProfile(
        name="brick",
        absorption=(0.03, 0.03, 0.03, 0.04, 0.05, 0.07),
        scattering=0.24,
        description="Exposed brick wall with moderate surface roughness.",
    ),
    "audience": MaterialProfile(
        name="audience",
        absorption=(0.32, 0.42, 0.58, 0.74, 0.82, 0.86),
        scattering=0.35,
        description="Seated audience or dense upholstered seating.",
    ),
    "acoustic-panel": MaterialProfile(
        name="acoustic-panel",
        absorption=(0.18, 0.55, 0.86, 0.95, 0.97, 0.97),
        scattering=0.18,
        description="Broadband porous absorber panel.",
    ),
    "diffuser": MaterialProfile(
        name="diffuser",
        absorption=(0.08, 0.10, 0.12, 0.14, 0.16, 0.18),
        scattering=0.72,
        description="Geometric diffuser or strongly irregular reflective surface.",
    ),
    "ceiling-tile": MaterialProfile(
        name="ceiling-tile",
        absorption=(0.30, 0.45, 0.65, 0.75, 0.82, 0.84),
        scattering=0.16,
        description="Suspended mineral-fiber acoustic ceiling tile.",
    ),
    "vinyl-floor": MaterialProfile(
        name="vinyl-floor",
        absorption=(0.02, 0.03, 0.03, 0.03, 0.03, 0.02),
        scattering=0.04,
        description="Hard vinyl or linoleum flooring.",
    ),
    "water": MaterialProfile(
        name="water",
        absorption=(0.01, 0.01, 0.01, 0.02, 0.03, 0.04),
        scattering=0.03,
        description="Smooth water-like reflective boundary.",
    ),
    "open-air": MaterialProfile(
        name="open-air",
        absorption=(0.99, 0.99, 0.99, 0.99, 0.99, 0.99),
        scattering=0.0,
        description="Non-reflecting exterior or missing boundary approximation.",
    ),
}


def material_names() -> tuple[str, ...]:
    """Return stable material profile names for CLI/docs/tests."""
    return tuple(sorted(_PROFILES))


def get_material_profile(name: str) -> MaterialProfile:
    """Resolve a material profile or raise a CLI-friendly ValueError."""
    key = str(name).strip().lower()
    if key in _PROFILES:
        return _PROFILES[key]
    choices = ", ".join(material_names())
    raise ValueError(f"Unknown material '{name}'. Available materials: {choices}.")


def material_absorption(name: str, default: float = 0.35) -> float:
    """Return broadband absorption while preserving legacy fallback behavior."""
    key = str(name).strip().lower()
    if key in _PROFILES:
        return _PROFILES[key].broadband_absorption()
    return float(np.clip(default, 0.0, 0.99))
