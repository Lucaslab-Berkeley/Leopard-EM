"""Template projections and correlations for odd and even template sizes.

The template volume is centered at index ``d // 2``; its projections must keep the
size ``(d, d)`` and that center for odd ``d`` as well as even ``d``.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch_fourier_slice.volume_utils import separable_sinc2_correction

from leopard_em.backend.core_refine_template import cross_correlate_particle_stack
from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate_cpu,
)
from leopard_em.utils.fourier_slice import (
    get_real_space_projections_from_volume,
    volume_to_rfft_fourier_slice,
)


def _blob_volume(d: int) -> torch.Tensor:
    """Blob at the center (d // 2) plus a weaker one off-center."""
    c = d // 2
    volume = np.zeros((d, d, d))
    volume[c, c, c] = 1.0
    volume[c - 2, c + 1, c + 2] = 0.5
    return torch.tensor(gaussian_filter(volume, sigma=1.0), dtype=torch.float32)


def _identity_projection(volume: torch.Tensor) -> torch.Tensor:
    """Contrast-flipped z-sum of the sinc^2-corrected volume."""
    return -(volume / separable_sinc2_correction(volume.shape)).sum(0)


@pytest.mark.parametrize("d", [16, 17])
def test_identity_projection_is_the_z_sum(d):
    volume = _blob_volume(d)
    zero = torch.zeros(1)
    projection = get_real_space_projections_from_volume(volume, zero, zero, zero)[0]

    assert projection.shape == (d, d)
    assert divmod(int(projection.argmin()), d) == (d // 2, d // 2)
    torch.testing.assert_close(
        projection, _identity_projection(volume), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("d", [8, 9])
def test_cross_correlation_matches_direct_correlation(d):
    """Valid region of the FFT correlation equals the direct sliding sum."""
    volume = _blob_volume(d)
    image = torch.randn(32, 32, generator=torch.Generator().manual_seed(0))
    image = image - image.mean()

    correlation = do_batched_orientation_cross_correlate_cpu(
        image_dft=torch.fft.rfft2(image),
        template_dft=volume_to_rfft_fourier_slice(volume),
        rotation_matrices=torch.eye(3)[None],
        projective_filters=torch.ones(1, 1, d, d // 2 + 1),
        apply_normalization=False,
    )[0, 0, 0]

    projection = _identity_projection(volume)
    projection = projection - projection.mean()  # the zeroed DC term
    expected = F.conv2d(image[None, None], projection[None, None])[0, 0]
    valid = correlation[: 32 - d + 1, : 32 - d + 1]
    torch.testing.assert_close(valid, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("d", [8, 9])
def test_particle_stack_correlation_with_identity_magnification(d):
    """An identity magnification matrix leaves the particle correlations unchanged."""
    template_dft = volume_to_rfft_fourier_slice(_blob_volume(d))
    images = torch.randn(2, 24, 24, generator=torch.Generator().manual_seed(1))
    particles_dft = torch.fft.rfft2(images)
    rotations = torch.eye(3).expand(2, 3, 3)
    filters = torch.ones(2, d, d // 2 + 1)

    plain = cross_correlate_particle_stack(
        particles_dft, template_dft, rotations, filters
    )
    magnified = cross_correlate_particle_stack(
        particles_dft, template_dft, rotations, filters, mag_matrix=torch.eye(2)
    )

    assert plain.shape == (2, 24 - d + 1, 24 - d + 1)
    torch.testing.assert_close(magnified, plain, atol=1e-4, rtol=1e-4)
