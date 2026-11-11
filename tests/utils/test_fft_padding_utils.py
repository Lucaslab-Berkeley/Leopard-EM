"""Tests for the Gaussian FFT-padding helpers."""

import pytest
import torch

from leopard_em.utils.fft_padding import (
    next_even_fft_shape,
    next_even_fft_size,
    pad_image_gaussian,
)

PAD_CASES = [
    ((118, 122), (128, 128)),
    ((100, 100), (100, 128)),  # width only
    ((100, 100), (128, 100)),  # height only
    ((5, 7), (6, 8)),
]


def _mask_based_reference(image, padded_shape, seed=0):
    """The mask-based implementation this helper replaced.

    Kept here so the vectorized version stays bitwise reproducible for a given seed;
    a change in fill order would silently alter every padded search result.
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    padded = torch.empty(padded_shape, dtype=image.dtype, device=image.device)
    padded[:height, :width] = image

    generator = torch.Generator(device=image.device).manual_seed(seed)
    pad_mask = torch.ones(padded_shape, dtype=torch.bool, device=image.device)
    pad_mask[:height, :width] = False
    num_pad_values = int(pad_mask.sum())

    noise = torch.randn(
        num_pad_values, generator=generator, dtype=image.dtype, device=image.device
    )
    noise = (noise - noise.mean()) / noise.std()
    padded[pad_mask] = noise * image.std() + image.mean()
    return padded


@pytest.mark.parametrize(("image_shape", "padded_shape"), PAD_CASES)
@pytest.mark.parametrize("seed", [0, 7])
def test_pad_matches_the_mask_based_fill_order(image_shape, padded_shape, seed):
    torch.manual_seed(1234)
    image = torch.randn(*image_shape) * 3.7 + 1.5

    expected = _mask_based_reference(image, padded_shape, seed=seed)
    assert torch.equal(pad_image_gaussian(image, padded_shape, seed=seed), expected)


@pytest.mark.parametrize(("image_shape", "padded_shape"), PAD_CASES)
def test_pad_preserves_the_image_in_the_top_left(image_shape, padded_shape):
    image = torch.randn(*image_shape)
    padded = pad_image_gaussian(image, padded_shape)

    assert padded.shape == padded_shape
    assert torch.equal(padded[: image_shape[0], : image_shape[1]], image)


def test_pad_is_a_no_op_without_padding():
    image = torch.randn(8, 8)
    assert pad_image_gaussian(image, (8, 8)) is image


def test_pad_rejects_a_smaller_target():
    with pytest.raises(ValueError, match="smaller than the image"):
        pad_image_gaussian(torch.randn(16, 16), (8, 16))


def test_pad_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D image"):
        pad_image_gaussian(torch.randn(2, 16, 16), (16, 16))


def test_next_even_fft_size_is_even_and_not_smaller():
    for n in range(1, 600):
        size = next_even_fft_size(n)
        assert size >= n
        assert size % 2 == 0


def test_next_even_fft_size_requires_factor_two():
    with pytest.raises(ValueError, match="must contain 2"):
        next_even_fft_size(100, factors=(3, 5))


def test_next_even_fft_shape_applies_per_dimension():
    assert next_even_fft_shape((4000, 2050)) == (4096, 2304)
