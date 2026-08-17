"""Image preprocessing and processing utility functions."""

from typing import TYPE_CHECKING

import torch
from torch_fourier_filter.dose_weight import dose_weight_movie

# Using the TYPE_CHECKING statement to avoid circular imports
if TYPE_CHECKING:
    from leopard_em.pydantic_models.config.correlation_filters import (
        PreprocessingFilters,
    )
    from leopard_em.pydantic_models.data_structures.particle_stack import (
        ParticleStackCSV,
        ParticleStackHDF5,
    )


def preprocess_image(
    image_rfft: torch.Tensor,
    cumulative_fourier_filters: torch.Tensor,
    bandpass_filter: torch.Tensor,
    full_image_shape: tuple[int, int],
    extracted_box_shape: tuple[int, int],
    normalization_ref_rfft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preprocess and normalize image FFTs with computed normalization factors.

    Parameters
    ----------
    image_rfft : torch.Tensor
        The real Fourier-transformed image (unshifted).
    cumulative_fourier_filters : torch.Tensor
        The cumulative Fourier filters. Multiplication of the whitening filter, phase
        randomization filter, bandpass filter, and arbitrary curve filter.
    bandpass_filter : torch.Tensor
        The bandpass filter used for the image. Used for dimensionality normalization.
    full_image_shape : tuple[int, int]
        The shape of the full image.
    extracted_box_shape : tuple[int, int]
        The shape of the extracted box.
    normalization_ref_rfft : torch.Tensor, optional
        If set, Parseval normalization is computed from this RFFT (e.g. the
        unconvolved micrograph) while ``image_rfft`` still receives the filters.
        Defaults to ``image_rfft``.

    Returns
    -------
    torch.Tensor
        Preprocessed and normalized image in Fourier space.
    """
    norm_rfft = image_rfft if normalization_ref_rfft is None else normalization_ref_rfft
    normalization_factor = get_image_normalization_factor(
        image_rfft=norm_rfft,
        cumulative_fourier_filters=cumulative_fourier_filters,
        bandpass_filter=bandpass_filter,
        full_image_shape=full_image_shape,
        extracted_box_shape=extracted_box_shape,
    )
    image_rfft = image_rfft * cumulative_fourier_filters
    return image_rfft * normalization_factor


def get_image_normalization_factor(
    image_rfft: torch.Tensor,
    cumulative_fourier_filters: torch.Tensor,
    bandpass_filter: torch.Tensor,
    full_image_shape: tuple[int, int],
    extracted_box_shape: tuple[int, int],
) -> torch.Tensor:
    """Compute per-image normalization factors for filtered Fourier images.

    Parameters
    ----------
    image_rfft : torch.Tensor
        Real Fourier-transformed images (unshifted), typically ``(..., H, W_rfft)``.
    cumulative_fourier_filters : torch.Tensor
        Combined Fourier-domain filtering stack applied to each image.
    bandpass_filter : torch.Tensor
        Bandpass filter used to estimate effective Fourier dimensionality.
    full_image_shape : tuple[int, int]
        Shape ``(H_full, W_full)`` of the source image used for extraction.
    extracted_box_shape : tuple[int, int]
        Shape ``(H_box, W_box)`` of extracted particle boxes.

    Returns
    -------
    torch.Tensor
        Multiplicative normalization factor broadcastable to ``image_rfft``.
    """
    # Normalize image variance after filtering via Parseval-conjugate accounting.
    filtered = image_rfft * cumulative_fourier_filters
    squared_image_rfft = torch.abs(filtered) ** 2
    squared_sum = torch.sum(squared_image_rfft, dim=(-2, -1), keepdim=True)
    squared_sum += torch.sum(
        squared_image_rfft[..., :, 1:-1], dim=(-2, -1), keepdim=True
    )

    # NOTE: For two Gaussian random variables in d-dimensional space --  A and B --
    # each with mean 0 and variance 1 their correlation will have on average a
    # variance of d.
    # NOTE: Since we have the variance of the image and template projections each at
    # 1, we need to multiply the image by the square root of the number of pixels
    # so the cross-correlograms have a variance of 1 and not d.
    # NOTE: When applying the Fourier filters to the image and template, any
    # elements that get set to zero effectively reduce the dimensionality of our
    # cross-correlation. Therefore, instead of multiplying by the number of pixels,
    # we need to multiply tby the effective number of pixels that are non-zero.
    # Below, we calculate the dimensionality of our cross-correlation and divide
    # by the square root of that number to normalize the image.
    dimensionality = bandpass_filter.sum() + bandpass_filter[:, 1:-1].sum()
    normalization_factor = (dimensionality**0.5) / torch.sqrt(squared_sum)

    # NOTE: We need to rescale based on the relative area of the extracted box
    # to the full image.
    img_h, img_w = full_image_shape
    box_h, box_w = extracted_box_shape
    normalization_factor = normalization_factor * (
        ((img_h * img_w) / ((box_h) * (box_w))) ** 0.5
    )
    return normalization_factor


def apply_image_filtering(
    particle_stack: "ParticleStackCSV | ParticleStackHDF5",
    preprocessing_filters: "PreprocessingFilters",
    images_dft: torch.Tensor,
    full_image_shape: tuple[int, int],
    extracted_box_shape: tuple[int, int],
    precomputed_filter_stack: torch.Tensor | None = None,
    precomputed_normalization_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply filtering and normalization to a set of Fourier images.

    Parameters
    ----------
    particle_stack : ParticleStackCSV | ParticleStackHDF5
        The particle stack
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the images.
    images_dft : torch.Tensor
        The images in Fourier space.
    full_image_shape: tuple[int, int]
        The shape of the full image.
    extracted_box_shape: tuple[int, int]
        The shape of the extracted box.
    precomputed_filter_stack : torch.Tensor | None
        Optional precomputed per-particle Fourier filter stack. If provided,
        reuse these filters instead of recomputing from ``images_dft``.
    precomputed_normalization_factor : torch.Tensor | None
        Optional fixed per-particle normalization factors (old-style frame
        normalization). If provided, apply these directly after filtering.

    Returns
    -------
    torch.Tensor
        Filtered images in Fourier space.
    """
    device = images_dft.device

    # Compute filters without gradient tracking (filters are just preprocessing)
    with torch.no_grad():
        bandpass_filter = (
            preprocessing_filters.bandpass_filter.calculate_bandpass_filter(
                images_dft.shape[-2:]
            ).to(device)
        )
    if precomputed_filter_stack is None:
        filter_stack = particle_stack.construct_image_filters(
            preprocessing_filters,
            output_shape=images_dft.shape[-2:],
            images_dft=images_dft.detach(),
        ).to(device)
    else:
        filter_stack = precomputed_filter_stack.to(device)

    if precomputed_normalization_factor is not None:
        # Apply the precomputed filters and fixed normalization factor directly.
        return images_dft * filter_stack * precomputed_normalization_factor.to(device)

    return preprocess_image(
        image_rfft=images_dft,
        cumulative_fourier_filters=filter_stack,
        bandpass_filter=bandpass_filter,
        full_image_shape=full_image_shape,
        extracted_box_shape=extracted_box_shape,
    )


def dose_weight_movie_to_micrograph(
    movie_fft: torch.Tensor,
    pixel_size: float,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
) -> torch.Tensor:
    """Dose weight a movie to create a micrograph.

    Parameters
    ----------
    movie_fft : torch.Tensor
        The movie in Fourier space.
    pixel_size : float
        The pixel size.
    pre_exposure : float
        The pre-exposure fluence in electrons per Angstrom squared.
    fluence_per_frame : float
        The dose per frame in electrons per Angstrom squared.
    voltage : float
        The voltage in kV.

    Returns
    -------
    torch.Tensor
        The dose weighted movie. Shape (h, w).
    """
    # get the height and width from the last two dimensions
    frame_shape = (movie_fft.shape[-2], movie_fft.shape[-1] * 2 - 2)
    # mean zero
    # movie_fft[..., 0, 0] = 0.0 + 0.0j
    # apply dose weight
    movie_dw_dft = dose_weight_movie(
        movie_dft=movie_fft,
        image_shape=frame_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=fluence_per_frame,
        voltage=voltage,
        crit_exposure_bfactor=-1,
        rfft=True,
        fftshift=False,
    )
    # inverse FFT
    movie_dw = torch.fft.irfft2(movie_dw_dft, s=frame_shape, dim=(-2, -1))  # pylint: disable=not-callable
    image_dw = torch.sum(movie_dw, dim=0)
    return image_dw


def volume_to_rfft_fourier_slice(volume: torch.Tensor) -> torch.Tensor:
    """Prepares a 3D volume for Fourier slice extraction.

    Parameters
    ----------
    volume : torch.Tensor
        The input volume.

    Returns
    -------
    torch.Tensor
        The prepared volume in Fourier space ready for slice extraction.
    """
    assert volume.dim() == 3, "Volume must be 3D"

    # NOTE: There is an extra FFTshift step before the RFFT since, for some reason,
    # omitting this step will cause a 180 degree phase shift on odd (i, j, k)
    # structure factors in the Fourier domain. This just requires an extra
    # IFFTshift after converting a slice back to real-space (handled already).
    volume = torch.fft.fftshift(volume, dim=(0, 1, 2))  # pylint: disable=E1102
    volume_rfft = torch.fft.rfftn(volume, dim=(0, 1, 2))  # pylint: disable=E1102
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(0, 1))  # pylint: disable=E1102

    return volume_rfft
