"""Tests for the correlation filter models"""

import pytest
import torch
from torch_fourier_filter.whitening import whitening_filter

from leopard_em.pydantic_models.config import (
    ArbitraryCurveFilterConfig,
    BandpassFilterConfig,
    PhaseRandomizationFilterConfig,
    PreprocessingFilters,
    WhiteningFilterConfig,
)
from leopard_em.pydantic_models.config.correlation_filters import (
    _WHITENING_TAKES_IMAGE_SHAPE,
)

ODD_AND_EVEN_SHAPES = [(32, 32), (32, 33), (33, 33), (33, 32)]


def _random_image_rfft(image_shape: tuple[int, int]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(image_shape, generator=generator)
    return torch.fft.rfft2(image)  # pylint: disable=not-callable


def test_whitening_filter_config_default():
    """
    Test the default values of the WhiteningFilterConfig.

    Verifies that the default properties of WhiteningFilterConfig are set correctly.
    """
    config = WhiteningFilterConfig()
    assert config.enabled is True
    assert config.num_freq_bins is None
    assert config.max_freq == 0.5
    assert config.do_power_spectrum is True


def test_whitening_filter_config_calculate():
    """
    Test the calculate_whitening_filter method of WhiteningFilterConfig.

    Verifies that the filter tensor has the same shape as the input reference image.
    """
    config = WhiteningFilterConfig()
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    filter_tensor = config.calculate_whitening_filter(ref_img_rfft)
    assert filter_tensor.shape == ref_img_rfft.shape


@pytest.mark.parametrize("image_shape", ODD_AND_EVEN_SHAPES)
def test_whitening_filter_odd_and_even_image_shapes(image_shape):
    """Whitening an odd- or even-sized image gives a filter of the right shape."""
    config = WhiteningFilterConfig()
    image_rfft = _random_image_rfft(image_shape)
    rfft_shape = (image_shape[0], image_shape[1] // 2 + 1)

    filter_tensor = config.calculate_whitening_filter(
        image_rfft, image_shape=image_shape
    )
    assert filter_tensor.shape == rfft_shape
    assert torch.isfinite(filter_tensor).all()

    # Filters for another grid (e.g. the template's) have that grid's shape
    filter_tensor = config.calculate_whitening_filter(
        image_rfft, output_shape=(16, 9), image_shape=image_shape
    )
    assert filter_tensor.shape == (16, 9)

    filters = PreprocessingFilters()
    combined = filters.get_combined_filter(
        image_rfft, rfft_shape, image_shape=image_shape
    )
    assert combined.shape == rfft_shape
    stack = filters.get_combined_filter_stack(
        torch.stack([image_rfft, 2 * image_rfft]), rfft_shape, image_shape=image_shape
    )
    assert stack.shape == (2, *rfft_shape)


@pytest.mark.skipif(
    not _WHITENING_TAKES_IMAGE_SHAPE,
    reason="torch-fourier-filter before 0.7 has no 'image_shape' argument",
)
@pytest.mark.parametrize("image_shape", [(32, 33), (33, 33)])
def test_whitening_filter_odd_width_matches_torch_fourier_filter(image_shape):
    """An odd-width image is whitened on its own real-space frequency grid."""
    config = WhiteningFilterConfig()
    image_rfft = _random_image_rfft(image_shape)
    expected = whitening_filter(  # pylint: disable=unexpected-keyword-arg
        image_rfft,
        rfft=True,
        max_freq=config.max_freq,
        num_freq_bins=config.num_freq_bins,
        do_power_spectrum=config.do_power_spectrum,
        image_shape=image_shape,
    )
    torch.testing.assert_close(
        config.calculate_whitening_filter(image_rfft, image_shape=image_shape),
        expected,
    )


def test_whitening_filter_even_width_unchanged_by_image_shape():
    """Passing an even image shape gives the same filter as the default."""
    config = WhiteningFilterConfig()
    image_rfft = _random_image_rfft((32, 32))
    for output_shape in (None, (16, 9)):
        torch.testing.assert_close(
            config.calculate_whitening_filter(
                image_rfft, output_shape=output_shape, image_shape=(32, 32)
            ),
            config.calculate_whitening_filter(image_rfft, output_shape=output_shape),
            rtol=0,
            atol=0,
        )


def test_whitening_filter_rejects_mismatched_image_shape():
    """An image shape that does not give the reference's RFFT shape raises."""
    config = WhiteningFilterConfig()
    image_rfft = _random_image_rfft((32, 33))
    with pytest.raises(ValueError, match="does not match"):
        config.calculate_whitening_filter(image_rfft, image_shape=(32, 35))
    with pytest.raises(ValueError, match="does not match"):
        config.calculate_whitening_filter(image_rfft, image_shape=(31, 33))


def test_phase_randomization_filter_config_default():
    """
    Test the default values of the PhaseRandomizationFilterConfig.

    Verifies that the default properties of PhaseRandomizationFilterConfig are
    set correctly.
    """
    config = PhaseRandomizationFilterConfig()
    assert config.enabled is False
    assert config.cuton is None


def test_phase_randomization_filter_config_calculate():
    """
    Test the calculate_phase_randomization_filter method.

    Verifies that the filter tensor has the same shape as the input reference image.
    """
    config = PhaseRandomizationFilterConfig(enabled=True, cuton=0.5)
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    filter_tensor = config.calculate_phase_randomization_filter(ref_img_rfft)
    assert filter_tensor.shape == ref_img_rfft.shape


def test_bandpass_filter_config_default():
    """
    Test the default values of the BandpassFilterConfig.

    Verifies that the default properties of BandpassFilterConfig are set correctly.
    """
    config = BandpassFilterConfig()
    assert config.enabled is False
    assert config.low_freq_cutoff is None
    assert config.high_freq_cutoff is None
    assert config.falloff is None


def test_bandpass_filter_config_calculate():
    """
    Test the calculate_bandpass_filter method of BandpassFilterConfig.

    Verifies that the filter tensor has the expected shape when given specific
    parameters.
    """
    config = BandpassFilterConfig(
        enabled=True, low_freq_cutoff=0.1, high_freq_cutoff=0.5, falloff=0.1
    )
    output_shape = (10, 10)
    filter_tensor = config.calculate_bandpass_filter(output_shape)
    assert filter_tensor.shape == output_shape


def test_arbitrary_curve_filter_config_default():
    """
    Test the default values of the ArbitraryCurveFilterConfig.

    Verifies that the default properties of ArbitraryCurveFilterConfig are
    set correctly.
    """
    config = ArbitraryCurveFilterConfig()
    assert config.enabled is False
    assert config.frequencies is None
    assert config.amplitudes is None


def test_arbitrary_curve_filter_config_calculate():
    """
    Test the calculate_arbitrary_curve_filter method.

    Verifies that the filter tensor has the expected shape when given specific
    frequencies and amplitudes.
    """
    config = ArbitraryCurveFilterConfig(
        enabled=True, frequencies=[0.1, 0.5], amplitudes=[1.0, 0.5]
    )
    output_shape = (10, 10)
    filter_tensor = config.calculate_arbitrary_curve_filter(output_shape)
    assert filter_tensor.shape == output_shape


def test_preprocessing_filters_default():
    """
    Test the default values of the PreprocessingFilters.

    Verifies that the PreprocessingFilters correctly creates instances of each
    filter type.
    """
    config = PreprocessingFilters()
    assert isinstance(config.whitening_filter, WhiteningFilterConfig)
    assert isinstance(config.bandpass_filter, BandpassFilterConfig)
    assert isinstance(config.phase_randomization_filter, PhaseRandomizationFilterConfig)
    assert isinstance(config.arbitrary_curve_filter, ArbitraryCurveFilterConfig)


def test_preprocessing_filters_combined_filter():
    """
    Test the get_combined_filter method of PreprocessingFilters.

    Verifies that the combined filter has the expected shape when multiple
    filters are applied.
    """
    config = PreprocessingFilters()
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    output_shape = (10, 10)
    combined_filter = config.get_combined_filter(ref_img_rfft, output_shape)
    assert combined_filter.shape == output_shape


def test_preprocessing_filters_combined_filter_stack_uses_each_image():
    """Each image of the batch is the reference for its own combined filter."""

    class _MeanFilters(PreprocessingFilters):
        def get_combined_filter(self, ref_img_rfft, output_shape, **kwargs):
            return torch.full(output_shape, float(ref_img_rfft.real.mean()))

    images_dft = torch.stack([torch.full((4, 3), 1.0), torch.full((4, 3), 3.0)])
    filters = _MeanFilters().get_combined_filter_stack(images_dft, (2, 2))
    assert filters.shape == (2, 2, 2)
    assert filters[:, 0, 0].tolist() == [1.0, 3.0]

    # The real filters give the same result per image as the single-image method
    real = PreprocessingFilters()
    images_dft = torch.fft.rfft2(torch.randn(2, 16, 16))
    stack = real.get_combined_filter_stack(images_dft, (16, 9))
    for i in range(2):
        torch.testing.assert_close(
            stack[i], real.get_combined_filter(images_dft[i], (16, 9))
        )
