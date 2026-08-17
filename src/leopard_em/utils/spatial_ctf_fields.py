"""Spatial defocus and phase-shift fields as tensors (no I/O).

Coordinates are normalized to [0, 1] along rows (y) and columns (x), matching
the prototype ``visualize_fields`` geometry.
"""

from __future__ import annotations

import math

import torch


def make_positions_2d(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``(H, W, 3)`` with ``[t=0.5, x_norm, y_norm]`` in ``[0, 1]``."""
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([torch.full_like(gx, 0.5), gx, gy], dim=-1)


def defocus_linear_field(
    height: int,
    width: int,
    grad_mag_angstrom: float,
    grad_angle_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Spatial defocus increment (Å), zero at image center.

    Value is ``grad_mag_angstrom * projection`` of ``(x-0.5, y-0.5)`` onto the
    direction ``grad_angle_deg`` (degrees from x-axis).
    """
    positions = make_positions_2d(height, width, device=device, dtype=dtype)
    angle_rad = math.radians(grad_angle_deg)
    px = positions[..., 1]
    py = positions[..., 2]
    projected = (px - 0.5) * math.cos(angle_rad) + (py - 0.5) * math.sin(angle_rad)
    return torch.full_like(projected, grad_mag_angstrom) * projected


def phase_quadratic_field(
    height: int,
    width: int,
    phase_c: float,
    phase_g: float,
    phase_k: float,
    phase_alpha_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase shift in degrees: ``c + g*s + k*s**2`` with ``s`` along ``alpha``.

    Here ``x, y`` are mapped from normalized ``[0, 1]`` to ``[-1, 1]`` as in the
    prototype.
    """
    positions = make_positions_2d(height, width, device=device, dtype=dtype)
    alpha_rad = math.radians(phase_alpha_deg)
    x = 2.0 * positions[..., 1] - 1.0
    y = 2.0 * positions[..., 2] - 1.0
    s = x * math.cos(alpha_rad) + y * math.sin(alpha_rad)
    return torch.clamp(
        torch.full_like(s, phase_c, dtype=dtype) + phase_g * s + phase_k * s**2,
        0.0,
        180.0,
    )


def _norm_axis_coords(
    n: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Coordinates in ``[0, 1]`` matching ``make_positions_2d`` ``linspace``."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return torch.tensor([0.5], device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)


def defocus_linear_increment_vertex_grid(
    grid_nx: int,
    grid_ny: int,
    grad_mag_angstrom: float,
    grad_angle_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Spatial defocus increment (Å) on an ``(grid_nx, grid_ny)``, zero at center."""
    gx = _norm_axis_coords(grid_nx, device=device, dtype=dtype)
    gy = _norm_axis_coords(grid_ny, device=device, dtype=dtype)
    ggx, ggy = torch.meshgrid(gx, gy, indexing="ij")
    angle_rad = math.radians(grad_angle_deg)
    projected = (ggx - 0.5) * math.cos(angle_rad) + (ggy - 0.5) * math.sin(angle_rad)
    return (grad_mag_angstrom * projected).to(dtype=dtype)


def phase_quadratic_vertex_grid(  # pylint: disable=too-many-locals
    grid_nx: int,
    grid_ny: int,
    phase_c: float,
    phase_g: float,
    phase_k: float,
    phase_alpha_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Phase (deg) on vertex lattice; same formula as ``phase_quadratic_field``."""
    gx = _norm_axis_coords(grid_nx, device=device, dtype=dtype)
    gy = _norm_axis_coords(grid_ny, device=device, dtype=dtype)
    ggx, ggy = torch.meshgrid(gx, gy, indexing="ij")
    alpha_rad = math.radians(phase_alpha_deg)
    x = 2.0 * ggx - 1.0
    y = 2.0 * ggy - 1.0
    s = x * math.cos(alpha_rad) + y * math.sin(alpha_rad)
    return torch.clamp(
        torch.full_like(s, phase_c, dtype=dtype) + phase_g * s + phase_k * s**2,
        0.0,
        180.0,
    )


def pixel_norm_coords(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(H, W)`` normalized ``x`` and ``y`` matching ``make_positions_2d`` sampling."""
    y_1d = _norm_axis_coords(height, device=device, dtype=dtype)
    x_1d = _norm_axis_coords(width, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(y_1d, x_1d, indexing="ij")
    return gx, gy
