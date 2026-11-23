"""Tests for the image normalization in ``leopard_em.utils.image_processing``."""

import pytest
import torch

from leopard_em.utils.image_processing import (
    get_image_normalization_factor,
    preprocess_image,
)


@pytest.mark.parametrize("shape", [(16, 16), (16, 17), (15, 15)])
def test_normalized_image_has_unit_power(shape):
    """With no filtering, the normalized image has unit sum of squares (Parseval).

    Every RFFT column but the zero-frequency one, and for an even width the Nyquist
    one, stands in for its conjugate mirror too.
    """
    image = torch.randn(shape, generator=torch.Generator().manual_seed(0))
    image_rfft = torch.fft.rfft2(image)
    ones = torch.ones(image_rfft.shape)

    factor = get_image_normalization_factor(image_rfft, ones, ones, shape, shape)
    torch.testing.assert_close(factor.squeeze(), image.square().sum().rsqrt())

    normalized = preprocess_image(image_rfft, ones, ones, shape, shape)
    power = torch.fft.irfft2(normalized, s=shape).square().sum()
    torch.testing.assert_close(power, torch.tensor(1.0))


def test_normalization_rejects_a_mismatched_shape():
    image_rfft = torch.fft.rfft2(torch.randn(16, 16))
    ones = torch.ones(image_rfft.shape)
    with pytest.raises(ValueError, match="columns"):
        get_image_normalization_factor(image_rfft, ones, ones, (16, 18), (16, 18))
