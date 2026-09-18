"""Padded and unpadded searches must produce the same results.

Automatic fast-FFT padding is enabled by default, so it must not change what a search
finds. Padding is applied only to the bottom and right of the image, which means the
valid correlation region of the unpadded image is exactly the top-left block of the
padded one, and the normalization scalar is derived from the original image so that
absolute correlation values are preserved too.

These tests run entirely on the CPU via ``do_batched_orientation_cross_correlate_cpu``.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pytest
import roma
import torch
from scipy.ndimage import gaussian_filter

from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate_cpu,
)
from leopard_em.pydantic_models.config import (
    ComputationalConfigMatch,
    DefocusSearchConfig,
    FastFFTPaddingConfig,
    OrientationSearchConfig,
    PreprocessingFilters,
)
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.pydantic_models.managers.match_template_manager import (
    MatchTemplateManager,
)

# 118 and 122 both pad up to 128 under the default [2, 3] factors, and both the input
# and the padded size are even (an odd padded size would break the backend's
# 2 * (w_rfft - 1) real-width reconstruction).
IMAGE_SHAPE = (118, 122)
PADDED_SHAPE = (128, 128)
TEMPLATE_SIZE = 16
NUM_ORIENTATIONS = 24
SPECTRAL_DYNAMIC_RANGE = 1000.0

FILTERS_OFF = PreprocessingFilters.model_validate(
    {"whitening_filter": {"enabled": False}, "bandpass_filter": {"enabled": False}}
)


def _colored_noise(shape, dynamic_range, seed):
    """Noise with a low-frequency-dominated spectrum, like a real micrograph."""
    torch.manual_seed(seed)
    height, width = shape
    ky = torch.fft.fftfreq(height)[:, None]
    kx = torch.fft.rfftfreq(width)[None, :]
    radius = torch.sqrt(ky**2 + kx**2)
    power = dynamic_range * torch.exp(-radius / 0.04) + 1.0
    white = torch.fft.rfftn(torch.randn(height, width))
    image = torch.fft.irfftn(white * power.sqrt(), s=shape)
    return image.to(torch.float32)


@pytest.fixture(name="template_volume")
def fixture_template_volume():
    np.random.seed(0)
    volume = gaussian_filter(
        np.random.randn(TEMPLATE_SIZE, TEMPLATE_SIZE, TEMPLATE_SIZE), sigma=1.5
    )
    return torch.tensor(volume, dtype=torch.float32)


@pytest.fixture(name="optics_group")
def fixture_optics_group():
    return OpticsGroup(
        label="test",
        voltage=300.0,
        pixel_size=1.06,
        defocus_u=5000.0,
        defocus_v=5000.0,
        astigmatism_angle=0.0,
        spherical_aberration=2.7,
        amplitude_contrast_ratio=0.07,
        phase_shift=0.0,
        ctf_B_factor=60.0,
    )


def _make_manager(image, volume, optics_group, filters, enabled):
    manager = MatchTemplateManager.model_construct(
        optics_group=optics_group,
        preprocessing_filters=filters,
        computational_config=ComputationalConfigMatch(),
        defocus_search_config=DefocusSearchConfig(enabled=False),
        orientation_search_config=OrientationSearchConfig(),
    )
    manager.micrograph = image.clone()
    manager.template_volume = volume
    manager.fast_fft_padding = FastFFTPaddingConfig(enabled=enabled)
    return manager


def _correlate(manager, rotation_matrices):
    """Run the manager's real preprocessing, then cross-correlate on the CPU."""
    kwargs = manager.make_backend_core_function_kwargs()
    projective_filters = (
        kwargs["ctf_filters"] * kwargs["whitening_filter_template"][None, None, ...]
    )
    correlation = do_batched_orientation_cross_correlate_cpu(
        image_dft=kwargs["image_dft"],
        template_dft=kwargs["template_dft"],
        rotation_matrices=rotation_matrices,
        projective_filters=projective_filters,
    )
    valid_h, valid_w = manager._fft_padding_plan.unpadded_valid_shape
    return correlation[..., :valid_h, :valid_w].reshape(-1, valid_h, valid_w)


def _z_score(correlation):
    mip = correlation.max(dim=0).values
    return mip, (mip - correlation.mean(dim=0)) / correlation.std(dim=0)


@contextmanager
def _ignore_padding_warnings():
    """Silence the informational FFTPaddingWarning these runs deliberately trigger."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _run_pair(volume, optics_group, filters):
    """Correlate the same image with padding disabled and enabled."""
    image = _colored_noise(IMAGE_SHAPE, SPECTRAL_DYNAMIC_RANGE, seed=7)
    torch.manual_seed(0)
    rotation_matrices = roma.random_rotmat(size=NUM_ORIENTATIONS)

    out = {}
    for enabled in (False, True):
        manager = _make_manager(image, volume, optics_group, filters, enabled)
        with _ignore_padding_warnings():
            correlation = _correlate(manager, rotation_matrices)
        out[enabled] = (correlation, manager._fft_padding_plan)
    return out


def test_padding_plan_targets_expected_shape(template_volume, optics_group):
    manager = _make_manager(
        torch.zeros(IMAGE_SHAPE), template_volume, optics_group, FILTERS_OFF, True
    )
    with _ignore_padding_warnings():
        manager.make_backend_core_function_kwargs()
    plan = manager._fft_padding_plan
    assert plan.original_shape == IMAGE_SHAPE
    assert plan.padded_shape == PADDED_SHAPE
    assert plan.unpadded_valid_shape == (
        IMAGE_SHAPE[0] - TEMPLATE_SIZE + 1,
        IMAGE_SHAPE[1] - TEMPLATE_SIZE + 1,
    )


def test_padded_image_dft_has_padded_shape(template_volume, optics_group):
    manager = _make_manager(
        _colored_noise(IMAGE_SHAPE, 100.0, seed=1),
        template_volume,
        optics_group,
        FILTERS_OFF,
        True,
    )
    with _ignore_padding_warnings():
        kwargs = manager.make_backend_core_function_kwargs()
    assert tuple(kwargs["image_dft"].shape) == (
        PADDED_SHAPE[0],
        PADDED_SHAPE[1] // 2 + 1,
    )


def test_disabled_padding_leaves_image_untouched(template_volume, optics_group):
    manager = _make_manager(
        _colored_noise(IMAGE_SHAPE, 100.0, seed=1),
        template_volume,
        optics_group,
        FILTERS_OFF,
        False,
    )
    kwargs = manager.make_backend_core_function_kwargs()
    assert tuple(kwargs["image_dft"].shape) == (
        IMAGE_SHAPE[0],
        IMAGE_SHAPE[1] // 2 + 1,
    )
    assert not manager._fft_padding_plan.is_padded


def test_correlations_match_exactly_without_fourier_filters(
    template_volume, optics_group
):
    """Without filters the padded run must reproduce the unpadded one to float error.

    This isolates the shape/index algebra and the normalization scalar from any
    resampling of the whitening profile.
    """
    out = _run_pair(template_volume, optics_group, FILTERS_OFF)
    unpadded, _ = out[False]
    padded, _ = out[True]

    assert unpadded.shape == padded.shape

    scale = (padded * unpadded).sum() / (unpadded * unpadded).sum()
    residual = (padded - scale * unpadded).norm() / unpadded.norm()
    assert scale.item() == pytest.approx(1.0, abs=1e-3)
    assert residual.item() < 1e-4


def test_z_scores_match_without_fourier_filters(template_volume, optics_group):
    out = _run_pair(template_volume, optics_group, FILTERS_OFF)
    _, z_unpadded = _z_score(out[False][0])
    _, z_padded = _z_score(out[True][0])
    assert torch.allclose(z_padded, z_unpadded, atol=1e-3)
    assert torch.argmax(z_padded) == torch.argmax(z_unpadded)


def test_peak_position_and_z_score_preserved_with_filters(
    template_volume, optics_group
):
    """With whitening enabled, peak detection must still agree.

    Whitening is estimated from the unpadded image but evaluated on the padded Fourier
    grid, so individual pixels shift slightly. What must not change is which pixel wins
    and what z-score it reports.
    """
    out = _run_pair(template_volume, optics_group, PreprocessingFilters())
    _, z_unpadded = _z_score(out[False][0])
    _, z_padded = _z_score(out[True][0])

    peak = torch.unravel_index(torch.argmax(z_unpadded), z_unpadded.shape)
    assert torch.argmax(z_padded) == torch.argmax(z_unpadded)
    assert z_padded[peak].item() == pytest.approx(z_unpadded[peak].item(), rel=0.02)

    top_unpadded = set(z_unpadded.flatten().topk(10).indices.tolist())
    top_padded = set(z_padded.flatten().topk(10).indices.tolist())
    assert len(top_unpadded & top_padded) >= 8
