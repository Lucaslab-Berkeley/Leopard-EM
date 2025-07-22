"""Testing script to check self-consistency of backend cross-correlation methods.

We currently have three main methods for cross-correlation, each which perform better/
worse depending on the specifics of the image shape, template shape, search space size,
etc:
  - do_streamed_orientation_cross_correlate: Computes a grid of 2D cross-correlations
    each over a different CUDA stream (all cross-correlations are 2D, but many
    different correlations are computed).
  - do_batched_orientation_cross_correlate: Computes a grid of 2D cross-correlations
    in sets of batches based on number of orientations all on the same CUDA stream.
  - do_batched_orientation_cross_correlate_cpu: CPU version of the batched
    cross-correlation (does not use compiled torch functions).

Although the execution model of these three methods are different, the should all return
the same results for the same set of inputs. This test script generates some example
data and compares the outputs of the three methods.
"""

import numpy as np
import pytest
import roma
import torch
from scipy.ndimage import gaussian_filter
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_filter.envelopes import b_envelope

from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate,
    do_streamed_orientation_cross_correlate,
)
from leopard_em.pydantic_models.utils import get_cs_range

IMAGE_SHAPE = (1024, 1024)
TEMPLATE_SHAPE = (128, 128, 128)
NUM_ORIENTATIONS = 10
NUM_DEFOCUS_VALUES = 5
NUM_PIXEL_SIZES = 4
NUM_STREAMS = 1


def smooth_random_noise(shape, sigma):
    """
    Generate a 3D smooth random noise array using Gaussian filtering.

    Args:
        shape (tuple): Shape of the output array, e.g., (D, H, W).
        sigma (float or tuple): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Smoothed noise array of shape `shape`.
    """
    noise = np.random.randn(*shape)  # Raw white noise
    smooth_noise = gaussian_filter(
        noise, sigma=sigma, mode="reflect"
    )  # Apply Gaussian smoothing
    return smooth_noise


@pytest.fixture
def sample_input_data() -> dict[str, torch.Tensor]:
    image = torch.randn(*IMAGE_SHAPE, dtype=torch.float32, device="cuda")
    image /= (IMAGE_SHAPE[0] * IMAGE_SHAPE[1]) ** 0.5
    template = smooth_random_noise(TEMPLATE_SHAPE, sigma=6.0)
    template = torch.tensor(template, dtype=torch.float32, device="cuda")
    rotation_matrices = roma.random_rotmat(size=NUM_ORIENTATIONS, device="cuda")

    # FFT the image and template to prepare for cross-correlation
    image_fft = torch.fft.rfft2(image)
    template = torch.fft.fftshift(template, dim=(0, 1, 2))
    template_fft = torch.fft.rfftn(template, dim=(0, 1, 2))
    template_fft = torch.fft.fftshift(template_fft, dim=(0, 1))

    # Generate a set of projective filters (CTFs) for the template
    defocus_values = torch.linspace(2000, 4000, NUM_DEFOCUS_VALUES)
    pixel_sizes = torch.linspace(0.8, 1.2, NUM_PIXEL_SIZES)
    cs_values = get_cs_range(1.0, pixel_sizes, 2.7)

    ctf_list = []
    for cs_val in cs_values:
        tmp = calculate_ctf_2d(
            defocus=defocus_values * 1e-4,  # Convert to um from Angstrom
            astigmatism=200 * 1e-4,  # Convert to um from Angstrom
            astigmatism_angle=0.0,
            voltage=300,  # 300 kV
            spherical_aberration=cs_val,
            amplitude_contrast=0.07,
            phase_shift=0.0,
            pixel_size=1.0,
            image_shape=template.shape[-2:],
            rfft=True,
            fftshift=False,
        )
        ctf_list.append(tmp)

    # Apply a b-factor envelope to the CTFs
    b_factor = 100.0  # arbitrary value
    b_envelope_values = b_envelope(
        B=b_factor,
        image_shape=template.shape[-2:],
        pixel_size=1.0,
        device="cuda",
    )

    projective_filters = torch.stack(ctf_list, dim=0).to(device="cuda")
    projective_filters *= b_envelope_values[None, None]

    return {
        "image_dft": image_fft,
        "template_dft": template_fft,
        "rotation_matrices": rotation_matrices,
        "projective_filters": projective_filters,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_stream_and_batch_cross_correlate_consistency(sample_input_data):
    """Test that the streamed and batched cross-correlation methods return the same."""
    cross_correlate_kwargs = sample_input_data

    batched_result = do_batched_orientation_cross_correlate(**cross_correlate_kwargs)
    streamed_result = do_streamed_orientation_cross_correlate(
        streams=[torch.cuda.Stream() for _ in range(NUM_STREAMS)],
        **cross_correlate_kwargs,
    )

    assert streamed_result.shape == batched_result.shape

    max_abs_diff = (streamed_result - batched_result).abs().max().item()

    # NOTE: using lighter tolerances here since FFT plans execute differently, and
    # cross-correlation results should be distributed roughly normally around zero.
    assert torch.allclose(streamed_result, batched_result, atol=5e-3, rtol=5e-3), (
        f"Streamed and batched cross-correlation results not within tolerance.\n"
        f"Max absolute difference: {max_abs_diff}\n"
    )
