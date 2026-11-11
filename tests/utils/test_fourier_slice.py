"""Tests for template volume padding and Fourier slice extraction.

The padded volume setup and post-extraction crop in ``leopard_em.utils.fourier_slice``
copy the steps of ``torch_fourier_slice.project_3d_to_2d``, so the projections are
checked against that function directly.
"""

import pytest
import roma
import torch
from torch_fourier_slice import project_3d_to_2d

from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate_cpu,
)
from leopard_em.utils.fourier_slice import (
    _crop_rfft_slices,
    _rfft_slices_to_real_projections,
    extract_template_fourier_slices,
    get_real_space_projections_from_volume,
    projection_shape_from_filters,
    volume_to_rfft_fourier_slice,
)

SIZE = 32


def _smooth_volume(size: int = SIZE, background: float = 0.3) -> torch.Tensor:
    """Low-pass filtered noise blob on a non-zero background (exercises edge value)."""
    gen = torch.Generator().manual_seed(0)
    noise = torch.randn((size, size, size), generator=gen)
    k = torch.fft.fftfreq(size)
    kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
    lowpass = torch.exp(-(kz**2 + ky**2 + kx**2) / (2 * 0.08**2))
    blob = torch.fft.ifftn(torch.fft.fftn(noise) * lowpass).real
    # Taper to a sphere so the "particle" sits inside the box
    r = torch.arange(size) - size // 2
    rz, ry, rx = torch.meshgrid(r, r, r, indexing="ij")
    mask = ((rz**2 + ry**2 + rx**2).float().sqrt() < size * 0.35).float()
    return blob * mask / blob.abs().max() + background


def _euler_angles(n: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    angles = torch.rand((3, n), generator=gen) * torch.tensor(
        [[360.0], [180.0], [360.0]]
    )
    return angles[0], angles[1], angles[2]


@pytest.mark.parametrize("pad_factor", [1.0, 2.0])
def test_projections_match_torch_fourier_slice(pad_factor):
    """Real-space projections equal project_3d_to_2d (up to the contrast flip)."""
    volume = _smooth_volume()
    phi, theta, psi = _euler_angles()
    rot = roma.euler_to_rotmat("ZYZ", (phi, theta, psi), degrees=True)

    ours = get_real_space_projections_from_volume(
        volume, phi, theta, psi, pad_factor=pad_factor
    )
    reference = project_3d_to_2d(volume, rot, pad_factor=pad_factor)

    assert ours.shape == reference.shape == (5, SIZE, SIZE)
    # Leopard-EM flips the projection contrast to match the micrographs
    torch.testing.assert_close(
        -ours, reference, atol=1e-4 * reference.abs().max(), rtol=0
    )


def test_padded_slices_with_mag_matrix_match_torch_fourier_slice():
    """Magnification transform is applied at the padded size, before the crop."""
    volume = _smooth_volume()
    phi, theta, psi = _euler_angles()
    rot = roma.euler_to_rotmat("ZYZ", (phi, theta, psi), degrees=True)
    mag = torch.tensor([[1.02, 0.01], [0.01, 0.98]])

    slices = extract_template_fourier_slices(
        volume_to_rfft_fourier_slice(volume, pad_factor=2.0),
        rot,
        projection_shape=(SIZE, SIZE),
        mag_matrix=mag,
    )
    # No contrast flip here: the backends flip after extraction
    ours = _rfft_slices_to_real_projections(slices)
    reference = project_3d_to_2d(volume, rot, pad_factor=2.0, transform_matrix=mag)

    # The edge value only shifts the mean (DC term), which the backends zero anyway
    ours = ours - ours.mean(dim=(-2, -1), keepdim=True)
    reference = reference - reference.mean(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(
        ours, reference, atol=1e-4 * reference.abs().max(), rtol=0
    )


@pytest.mark.parametrize("small_shape", [(6, 8), (7, 8)])
def test_crop_rfft_slices_recovers_embedded_image(small_shape):
    """Cropping the slice of a zero-embedded image gives back the image's slice."""
    h, w = small_shape
    big_h, big_w = 2 * h, 2 * w
    image = torch.randn((3, h, w), generator=torch.Generator().manual_seed(2))
    embedded = torch.zeros((3, big_h, big_w))
    y0, x0 = big_h // 2 - h // 2, big_w // 2 - w // 2
    embedded[:, y0 : y0 + h, x0 : x0 + w] = image

    def centered_rfft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return torch.fft.fftshift(torch.fft.rfftn(x, dim=(-2, -1)), dim=(-2,))

    cropped = _crop_rfft_slices(centered_rfft(embedded), (big_h, big_w), (h, w))
    torch.testing.assert_close(cropped, centered_rfft(image), atol=1e-5, rtol=1e-5)


def test_backend_uses_filter_shape_with_padded_template():
    """A padded template DFT yields template-size correlations close to unpadded."""
    volume = _smooth_volume()
    phi, theta, psi = _euler_angles(4)
    rot = roma.euler_to_rotmat("ZYZ", (phi, theta, psi), degrees=True)
    image_dft = torch.fft.rfftn(torch.randn((64, 64)), dim=(-2, -1))
    filters = torch.ones((1, 2, SIZE, SIZE // 2 + 1))  # (num_Cs, num_defocus, h, w)
    assert projection_shape_from_filters(filters) == (SIZE, SIZE)

    results = {}
    for pad_factor in (1.0, 2.0):
        template_dft = volume_to_rfft_fourier_slice(volume, pad_factor=pad_factor)
        results[pad_factor] = do_batched_orientation_cross_correlate_cpu(
            image_dft=image_dft,
            template_dft=template_dft,
            rotation_matrices=rot,
            projective_filters=filters,
        )

    assert tuple(volume_to_rfft_fourier_slice(volume).shape) == (64, 64, 33)
    assert results[2.0].shape == results[1.0].shape == (1, 2, 4, 64, 64)
    corr = torch.corrcoef(torch.stack([results[1.0].flatten(), results[2.0].flatten()]))
    assert corr[0, 1] > 0.99


def test_invalid_padding_arguments_raise():
    volume = _smooth_volume(16)
    with pytest.raises(ValueError, match="pad_factor"):
        volume_to_rfft_fourier_slice(volume, pad_factor=0.5)
    with pytest.raises(ValueError, match="larger than"):
        extract_template_fourier_slices(
            volume_to_rfft_fourier_slice(volume, pad_factor=1.0),
            torch.eye(3)[None],
            projection_shape=(32, 32),
        )
