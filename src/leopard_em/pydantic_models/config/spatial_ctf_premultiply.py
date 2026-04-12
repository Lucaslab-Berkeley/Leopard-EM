"""Spatial CTF PSF grid + field models (used by ``SpatialCtfMatchTemplateManager``)."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import AfterValidator, Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


def _kernel_must_be_odd(v: int) -> int:
    if v % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    return v


class SpatialPsfConfig(BaseModel2DTM):
    """2D vertex grid for spatial PSF(``nx * ny`` CTF evaluations per defocus plane)."""

    kernel_size: Annotated[
        int,
        Field(ge=101, description="Side length of PSF kernel."),
        AfterValidator(_kernel_must_be_odd),
    ]
    grid_nx: Annotated[int, Field(ge=4, description="Vertices along normalized x.")]
    grid_ny: Annotated[int, Field(ge=4, description="Vertices along normalized y.")]


class LinearDefocusSpatialConfig(BaseModel2DTM):
    """Spatial defocus via linear gradient in normalized image coordinates."""

    kind: Literal["linear_defocus"] = "linear_defocus"
    grad_mag_angstrom: float
    grad_angle_deg: float = 0.0


class QuadraticPhaseSpatialConfig(BaseModel2DTM):
    """Spatial phase shift (degrees): c + g*s + k*s**2 along alpha, clamped[0, 180]."""

    kind: Literal["quadratic_phase"] = "quadratic_phase"
    phase_c: float
    phase_g: float = 0.0
    phase_k: float = 0.0
    phase_alpha_deg: float = 0.0


SpatialModelConfig = Annotated[
    Union[LinearDefocusSpatialConfig, QuadraticPhaseSpatialConfig],
    Field(discriminator="kind"),
]
