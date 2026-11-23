"""Backend correlations for odd and even real-space image widths.

An RFFT of width ``w // 2 + 1`` comes from a real width of either ``2k`` or ``2k + 1``,
so the backends take the true image shape (``image_shape_real``) instead of assuming an
even width.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch_fourier_slice.volume_utils import separable_sinc2_correction

from leopard_em.backend.core_match_template import _prepare_image_dft
from leopard_em.backend.core_refine_template import (
    core_refine_template,
    cross_correlate_particle_stack,
)
from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate_cpu,
    do_batched_orientation_frc,
)
from leopard_em.utils.ctf_utils import calculate_ctf_filter_stack_full_args
from leopard_em.utils.fourier_slice import volume_to_rfft_fourier_slice

CTF_KWARGS = {
    "voltage": 300.0,
    "spherical_aberration": 2.7,
    "amplitude_contrast_ratio": 0.07,
    "ctf_B_factor": 0.0,
    "phase_shift": 0.0,
    "pixel_size": 1.0,
    "even_zernikes": None,
    "odd_zernikes": None,
    "mag_matrix": None,
}


def _blob_volume(d: int) -> torch.Tensor:
    c = d // 2
    volume = np.zeros((d, d, d))
    volume[c, c, c] = 1.0
    volume[c - 2, c + 1, c + 2] = 0.5
    return torch.tensor(gaussian_filter(volume, sigma=1.0), dtype=torch.float32)


def _zero_mean_images(*shape: int) -> torch.Tensor:
    images = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    return images - images.mean(dim=(-2, -1), keepdim=True)


@pytest.mark.parametrize("shape", [(32, 32), (32, 33)])
def test_prepare_image_dft_uses_the_given_shape(shape):
    image_dft = torch.fft.rfft2(torch.randn(shape))
    _, reported = _prepare_image_dft(image_dft, "batched", shape)
    assert reported == shape

    wrong = (shape[0], shape[1] + 2)
    with pytest.raises(ValueError):
        _prepare_image_dft(image_dft, "batched", wrong)


@pytest.mark.parametrize("d", [8, 9])
@pytest.mark.parametrize("image_shape", [(32, 32), (31, 33)])
def test_cross_correlation_matches_direct_correlation(image_shape, d):
    """Valid region of the FFT correlation equals the direct sliding sum."""
    volume = _blob_volume(d)
    image = _zero_mean_images(*image_shape)

    correlation = do_batched_orientation_cross_correlate_cpu(
        image_dft=torch.fft.rfft2(image),
        template_dft=volume_to_rfft_fourier_slice(volume),
        rotation_matrices=torch.eye(3)[None],
        projective_filters=torch.ones(1, 1, d, d // 2 + 1),
        apply_normalization=False,
        image_shape_real=image_shape,
    )[0, 0, 0]
    assert correlation.shape == image_shape

    projection = -(volume / separable_sinc2_correction(volume.shape)).sum(0)
    projection = projection - projection.mean()  # the zeroed DC term
    expected = F.conv2d(image[None, None], projection[None, None])[0, 0]
    valid_h, valid_w = image_shape[0] - d + 1, image_shape[1] - d + 1
    torch.testing.assert_close(
        correlation[:valid_h, :valid_w], expected, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("d", [8, 9])
def test_frc_of_an_image_the_template_size(d):
    template_dft = volume_to_rfft_fourier_slice(_blob_volume(d))
    image = torch.randn(d, d, generator=torch.Generator().manual_seed(2))
    frc, frequency_bins = do_batched_orientation_frc(
        image_dft=torch.fft.rfft2(image),
        template_dft=template_dft,
        rotation_matrices=torch.eye(3)[None],
        projective_filters=torch.ones(1, 1, d, d // 2 + 1),
        image_shape_real=(d, d),
    )
    torch.testing.assert_close(frequency_bins, torch.fft.rfftfreq(d))
    assert frc.shape == (1, 1, 1, d // 2 + 1)


@pytest.mark.parametrize("d", [8, 9])
@pytest.mark.parametrize("box", [(24, 24), (23, 25)])
def test_particle_stack_correlation_matches_batched_correlation(box, d):
    template_dft = volume_to_rfft_fourier_slice(_blob_volume(d))
    images = _zero_mean_images(2, *box)
    particles_dft = torch.fft.rfft2(images)

    stack = cross_correlate_particle_stack(
        particles_dft,
        template_dft,
        torch.eye(3).expand(2, 3, 3),
        torch.ones(2, d, d // 2 + 1),
        image_shape_real=box,
    )
    valid = (box[0] - d + 1, box[1] - d + 1)
    assert stack.shape == (2, *valid)

    for i in range(2):
        batched = do_batched_orientation_cross_correlate_cpu(
            image_dft=particles_dft[i],
            template_dft=template_dft,
            rotation_matrices=torch.eye(3)[None],
            projective_filters=torch.ones(1, 1, d, d // 2 + 1),
            image_shape_real=box,
        )[0, 0, 0]
        torch.testing.assert_close(
            stack[i], batched[: valid[0], : valid[1]], atol=1e-4, rtol=1e-4
        )


@pytest.mark.parametrize("box", [(24, 24), (23, 25)])
def test_core_refine_template_on_odd_and_even_boxes(box):
    """Refined correlation and position match the particle-stack correlation."""
    d, n = 9, 2
    template_dft = volume_to_rfft_fourier_slice(_blob_volume(d))
    particles_dft = torch.fft.rfft2(_zero_mean_images(n, *box))
    valid = (box[0] - d + 1, box[1] - d + 1)
    defocus = torch.full((n,), 10000.0)
    ctf_kwargs = {**CTF_KWARGS, "template_shape": (d, d)}

    result = core_refine_template(
        particle_stack_dft=particles_dft,
        template_dft=template_dft,
        euler_angles=torch.zeros(n, 3),
        euler_angle_offsets=torch.zeros(1, 3),
        defocus_offsets=torch.zeros(1),
        defocus_u=defocus,
        defocus_v=defocus,
        defocus_angle=torch.zeros(n),
        pixel_size_offsets=torch.zeros(1),
        corr_mean=torch.zeros(n, *valid),
        corr_std=torch.ones(n, *valid),
        ctf_kwargs=ctf_kwargs,
        projective_filters=torch.ones(n, d, d // 2 + 1),
        device=torch.device("cpu"),
        batch_size=1,
        image_shape_real=box,
    )

    ctf = calculate_ctf_filter_stack_full_args(
        defocus_u=10000.0,
        defocus_v=10000.0,
        astigmatism_angle=0.0,
        defocus_offsets=torch.zeros(1),
        pixel_size_offsets=torch.zeros(1),
        **ctf_kwargs,
    )[0, 0]
    expected = cross_correlate_particle_stack(
        particles_dft,
        template_dft,
        torch.eye(3).expand(n, 3, 3),
        ctf.expand(n, -1, -1),
        image_shape_real=box,
    )
    flat_max = expected.flatten(1).max(dim=1)
    peak_y, peak_x = flat_max.indices // valid[1], flat_max.indices % valid[1]

    torch.testing.assert_close(
        result["refined_cross_correlation"], flat_max.values, atol=1e-4, rtol=1e-4
    )
    assert torch.equal(result["refined_pos_y"], peak_y - (box[0] - d + 1) // 2)
    assert torch.equal(result["refined_pos_x"], peak_x - (box[1] - d + 1) // 2)
