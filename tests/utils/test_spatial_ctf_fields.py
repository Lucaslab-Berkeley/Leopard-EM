"""Tests for spatial CTF field tensors and real-space PSF helpers."""

import torch

from leopard_em.utils.spatial_ctf_fields import (
    defocus_linear_field,
    phase_quadratic_field,
)
from leopard_em.utils.spatial_ctf_realspace import (
    apply_spatial_psf_grid,
    apply_spatially_varying_psf,
    ctf_to_psf_crop,
    template_ctf_unit_variance_gain,
)


def test_defocus_linear_zero_at_center():
    # Odd sizes so linspace includes 0.5 at the central index.
    h, w = 63, 63
    dev = torch.device("cpu")
    f = defocus_linear_field(h, w, 100.0, 33.0, device=dev)
    assert f.shape == (h, w)
    cy, cx = h // 2, w // 2
    assert f[cy, cx].abs() < 1e-5


def test_phase_quadratic_clamped():
    h, w = 32, 32
    dev = torch.device("cpu")
    # Strong curvature to push past 180
    phi = phase_quadratic_field(h, w, 100.0, 0.0, 500.0, 0.0, device=dev)
    assert phi.min() >= 0.0 - 1e-5
    assert phi.max() <= 180.0 + 1e-5


def test_apply_spatial_psf_grid_single_kernel():
    """1x1 grid: every pixel uses the same PSF (circular convolution)."""
    dev = torch.device("cpu")
    k = 5
    h, w = 10, 12
    torch.manual_seed(0)
    img = torch.randn(h, w, device=dev)
    kernel = torch.ones(k, k, device=dev) / (k * k)
    grid = kernel.view(1, 1, k, k)
    out = apply_spatial_psf_grid(
        img, grid, grid_nx=1, grid_ny=1, blend="bilinear", kernel_size=k
    )
    pad = (k - 1) // 2
    padded = (
        torch.nn.functional.pad(
            img.unsqueeze(0).unsqueeze(0),
            (pad, pad, pad, pad),
            mode="circular",
        )
        .squeeze(0)
        .squeeze(0)
    )
    expected_rows = []
    for r in range(h):
        row_patches = padded[r : r + k, :].unfold(1, k, 1).permute(1, 0, 2)
        expected_rows.append((row_patches * kernel).sum(dim=(-1, -2)))
    expected = torch.stack(expected_rows, dim=0)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_spatial_psf_uniform_weights_match_single_kernel():
    """Two-level stack of identical kernels: output matches Frobenius inner product."""
    dev = torch.device("cpu")
    k = 5
    h, w = 12, 14
    torch.manual_seed(0)
    img = torch.randn(h, w, device=dev)
    kernel = torch.ones(k, k, device=dev) / (k * k)
    stack = torch.stack([kernel, kernel], dim=0)
    scalar = torch.full((h, w), 3.14, device=dev)
    out = apply_spatially_varying_psf(img, scalar, stack, kernel_size=k)
    pad = (k - 1) // 2
    padded = (
        torch.nn.functional.pad(
            img.unsqueeze(0).unsqueeze(0),
            (pad, pad, pad, pad),
            mode="circular",
        )
        .squeeze(0)
        .squeeze(0)
    )
    expected_rows = []
    for r in range(h):
        row_patches = padded[r : r + k, :].unfold(1, k, 1).permute(1, 0, 2)
        expected_rows.append((row_patches * kernel).sum(dim=(-1, -2)))
    expected = torch.stack(expected_rows, dim=0)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_template_ctf_unit_variance_gain_constant_ctf():
    whitening = torch.ones(8, 5)
    ctf = torch.full((8, 5), 0.5)
    gain = template_ctf_unit_variance_gain(ctf, whitening)
    assert abs(float(gain) - 2.0) < 1e-5
    slices = torch.randn(4, 8, 5, dtype=torch.complex64)
    gain_slices = template_ctf_unit_variance_gain(ctf, whitening, slices)
    assert abs(float(gain_slices) - 2.0) < 1e-4


def test_ctf_to_psf_crop_not_sum_normalized():
    h, w = 32, 32
    ksz = 5
    # rFFT of a scaled delta is a constant; crop sum tracks the scale, not 1.
    ctf = torch.ones(h, w // 2 + 1) * 2.0
    crop = ctf_to_psf_crop(ctf, (h, w), ksz)
    assert crop.shape == (ksz, ksz)
    assert abs(float(crop.sum()) - 2.0) < 0.05
