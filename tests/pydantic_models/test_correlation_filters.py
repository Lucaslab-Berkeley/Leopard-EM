"""Tests for the correlation filter models"""

import pytest
import torch
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.phase_randomize import phase_randomize
from torch_fourier_filter.utils import curve_1dim_to_ndim
from torch_fourier_filter.whitening import whitening_filter
from torch_grid_utils import fftfreq_grid

from leopard_em.pydantic_models.config import (
    ArbitraryCurveFilterConfig,
    BandpassFilterConfig,
    PhaseRandomizationFilterConfig,
    PreprocessingFilters,
    WhiteningFilterConfig,
)


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


@pytest.mark.parametrize("image_shape", [(10, 18), (10, 19)])
def test_whitening_filter_config_calculate(image_shape):
    """
    Test the calculate_whitening_filter method of WhiteningFilterConfig.

    Verifies that the filter tensor has the same shape as the input reference image.
    """
    config = WhiteningFilterConfig()
    ref_img_rfft = torch.fft.rfft2(torch.randn(image_shape))
    filter_tensor = config.calculate_whitening_filter(
        ref_img_rfft, image_shape=image_shape
    )
    assert filter_tensor.shape == ref_img_rfft.shape


@pytest.mark.parametrize("output_image_shape", [(8, 8), (9, 9)])
def test_whitening_filter_config_output_grid(output_image_shape):
    """The filter is evaluated on the real-space output grid, odd or even."""
    config = WhiteningFilterConfig()
    image_shape = (32, 31)
    ref_img_rfft = torch.fft.rfft2(torch.randn(image_shape))
    output_shape = (output_image_shape[0], output_image_shape[1] // 2 + 1)
    filter_tensor = config.calculate_whitening_filter(
        ref_img_rfft,
        output_shape,
        image_shape=image_shape,
        output_image_shape=output_image_shape,
    )
    assert filter_tensor.shape == output_shape

    # The output grid defaults to the reference image's, which does not match here
    with pytest.raises(ValueError):
        config.calculate_whitening_filter(
            ref_img_rfft, output_shape, image_shape=image_shape
        )


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
    filter_tensor = config.calculate_phase_randomization_filter(
        ref_img_rfft, image_shape=(10, 18)
    )
    assert filter_tensor.shape == ref_img_rfft.shape


@pytest.mark.parametrize("image_shape", [(16, 16), (16, 17)])
def test_phase_randomization_filter_matches_torch_fourier_filter(image_shape):
    """The phase randomization uses the real-space shape, odd or even."""
    config = PhaseRandomizationFilterConfig(enabled=True, cuton=0.25)
    ref_img_rfft = torch.fft.rfft2(torch.randn(image_shape))
    torch.manual_seed(1)
    randomized = config.calculate_phase_randomization_filter(
        ref_img_rfft, image_shape=image_shape
    )
    torch.manual_seed(1)
    expected = phase_randomize(
        ref_img_rfft, image_shape=image_shape, rfft=True, fftshift=False, cuton=0.25
    )
    torch.testing.assert_close(randomized, expected)


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
    filter_tensor = config.calculate_bandpass_filter(output_shape, image_shape=(10, 18))
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
    filter_tensor = config.calculate_arbitrary_curve_filter(
        output_shape, image_shape=(10, 18)
    )
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
    combined_filter = config.get_combined_filter(
        ref_img_rfft, output_shape, image_shape=(10, 18)
    )
    assert combined_filter.shape == output_shape


def test_preprocessing_filters_combined_filter_stack_uses_each_image():
    """Each image of the batch is the reference for its own combined filter."""

    class _MeanFilters(PreprocessingFilters):
        def get_combined_filter(self, ref_img_rfft, output_shape, **kwargs):
            return torch.full(output_shape, float(ref_img_rfft.real.mean()))

    images_dft = torch.stack([torch.full((4, 3), 1.0), torch.full((4, 3), 3.0)])
    filters = _MeanFilters().get_combined_filter_stack(
        images_dft, (2, 2), image_shape=(4, 4)
    )
    assert filters.shape == (2, 2, 2)
    assert filters[:, 0, 0].tolist() == [1.0, 3.0]

    # The real filters give the same result per image as the single-image method
    real = PreprocessingFilters()
    images_dft = torch.fft.rfft2(torch.randn(2, 16, 16))
    stack = real.get_combined_filter_stack(images_dft, (16, 9), image_shape=(16, 16))
    for i in range(2):
        torch.testing.assert_close(
            stack[i],
            real.get_combined_filter(images_dft[i], (16, 9), image_shape=(16, 16)),
        )


@pytest.mark.parametrize("output_image_shape", [(16, 16), (16, 17), (17, 17)])
def test_combined_filter_matches_direct_filters(output_image_shape):
    """Every filter is evaluated on the real-space output grid, odd or even."""
    frequencies, amplitudes = [0.0, 0.25, 0.5], [1.0, 0.5, 0.2]
    config = PreprocessingFilters(
        bandpass_filter=BandpassFilterConfig(
            enabled=True, low_freq_cutoff=0.05, high_freq_cutoff=0.4, falloff=0.05
        ),
        arbitrary_curve_filter=ArbitraryCurveFilterConfig(
            enabled=True, frequencies=frequencies, amplitudes=amplitudes
        ),
    )
    image_shape = (32, 31)
    ref_img_rfft = torch.fft.rfft2(torch.randn(image_shape))
    output_shape = (output_image_shape[0], output_image_shape[1] // 2 + 1)

    combined = config.get_combined_filter(
        ref_img_rfft,
        output_shape,
        image_shape=image_shape,
        output_image_shape=output_image_shape,
    )

    whitening = whitening_filter(
        ref_img_rfft,
        max_freq=0.5,
        image_shape=image_shape,
        output_shape=output_image_shape,
    )
    bandpass = bandpass_filter(
        0.05, 0.4, 0.05, output_image_shape, rfft=True, fftshift=False
    )
    curve = curve_1dim_to_ndim(
        frequency_1d=torch.tensor(frequencies),
        values_1d=torch.tensor(amplitudes),
        frequency_grid=fftfreq_grid(output_image_shape, rfft=True, norm=True),
        fill_lower=1.0,
        fill_upper=1.0,
    )
    torch.testing.assert_close(combined, whitening * bandpass * curve)
