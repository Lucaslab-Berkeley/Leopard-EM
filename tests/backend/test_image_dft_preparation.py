"""Tests for how the real-space image shape reaches the cross-correlation backends."""

import pytest
import torch

from leopard_em.backend.core_match_template import _prepare_image_dft
from leopard_em.backend.cross_correlation import (
    _image_shape_from_rfft,
    do_batched_orientation_cross_correlate_zipfft,
)

IMAGE_SHAPE = (118, 122)


def _rfft_of(shape):
    return torch.fft.rfftn(torch.randn(*shape))


@pytest.mark.parametrize("backend", ["streamed", "batched"])
def test_prepare_leaves_torch_backends_untouched(backend):
    image_dft = _rfft_of(IMAGE_SHAPE)
    prepared, shape = _prepare_image_dft(image_dft, backend)

    assert prepared is image_dft
    assert shape == IMAGE_SHAPE


def test_prepare_transposes_for_zipfft_but_reports_the_true_shape():
    """The reported shape must not follow the buffer's layout."""
    image_dft = _rfft_of(IMAGE_SHAPE)
    prepared, shape = _prepare_image_dft(image_dft, "zipfft")

    # Buffer is transposed and contiguous for the kernel...
    assert tuple(prepared.shape) == (IMAGE_SHAPE[1] // 2 + 1, IMAGE_SHAPE[0])
    assert prepared.is_contiguous()
    # ...but the shape handed downstream is still true (rows, cols).
    assert shape == IMAGE_SHAPE
    assert torch.equal(prepared.transpose(-2, -1), image_dft)


def test_prepared_shape_matches_the_legacy_inference_for_torch_backends():
    image_dft = _rfft_of(IMAGE_SHAPE)
    _, shape = _prepare_image_dft(image_dft, "batched")

    assert shape == _image_shape_from_rfft(image_dft)


# Only the last axis is reconstructed as 'W = 2 * (w_rfft - 1)', so only it has to
# be even. 3125 = 5**5 is the size of a compiled zipFFT kernel.
@pytest.mark.parametrize(
    "shape", [(118, 122), (128, 128), (4096, 4096), (3125, 128), (121, 122)]
)
def test_legacy_inference_round_trips_odd_and_even_first_axes(shape):
    assert _image_shape_from_rfft(_rfft_of(shape)) == shape


@pytest.mark.parametrize("backend", ["streamed", "batched", "zipfft"])
def test_prepare_preserves_an_odd_first_axis(backend):
    """An odd number of rows must survive both the shape report and the transpose."""
    shape = (3125, 128)
    prepared, reported = _prepare_image_dft(_rfft_of(shape), backend)

    assert reported == shape
    expected = (
        (shape[1] // 2 + 1, shape[0])
        if backend == "zipfft"
        else (
            shape[0],
            shape[1] // 2 + 1,
        )
    )
    assert tuple(prepared.shape) == expected


def test_zipfft_rejects_a_dft_that_was_not_transposed():
    """Guards the convention that made the pre-a84ed12 axis swap invisible."""
    pytest.importorskip("zipfft", reason="guard runs after the ImportError check")

    image_dft = _rfft_of(IMAGE_SHAPE)  # deliberately NOT transposed
    with pytest.raises(ValueError, match="transposed for zipFFT"):
        do_batched_orientation_cross_correlate_zipfft(
            image_dft=image_dft,
            template_dft=torch.zeros(4, 16, 9, dtype=torch.complex64),
            rotation_matrices=torch.eye(3)[None],
            projective_filters=torch.zeros(1, 1, 16, 9, dtype=torch.complex64),
            image_shape_real=IMAGE_SHAPE,
        )


def test_zipfft_rejects_an_uncompiled_image_shape():
    """Reachable today via target_shape / enabled:false, so it must fail loudly."""
    pytest.importorskip("zipfft", reason="guard runs after the ImportError check")

    image_shape = (118, 122)  # no compiled zipFFT config for this
    image_dft, _ = _prepare_image_dft(_rfft_of(image_shape), "zipfft")
    with pytest.raises(ValueError, match="no compiled configuration"):
        do_batched_orientation_cross_correlate_zipfft(
            image_dft=image_dft,
            template_dft=torch.zeros(4, 16, 9, dtype=torch.complex64),
            rotation_matrices=torch.eye(3)[None],
            projective_filters=torch.zeros(1, 1, 16, 9, dtype=torch.complex64),
            image_shape_real=image_shape,
        )
