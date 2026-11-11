"""Tests for the correlation filter models"""

import torch

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


def test_whitening_filter_config_calculate():
    """
    Test the calculate_whitening_filter method of WhiteningFilterConfig.

    Verifies that the filter tensor has the same shape as the input reference image.
    """
    config = WhiteningFilterConfig()
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    filter_tensor = config.calculate_whitening_filter(ref_img_rfft)
    assert filter_tensor.shape == ref_img_rfft.shape


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
