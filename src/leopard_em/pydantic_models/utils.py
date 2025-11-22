# pylint: disable=duplicate-code # pylint: disable=too-many-lines
"""Utility functions shared between pydantic models."""

from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_filter.dose_weight import dose_weight_movie
from torch_fourier_filter.envelopes import b_envelope

# Using the TYPE_CHECKING statement to avoid circular imports
if TYPE_CHECKING:
    from .config.correlation_filters import PreprocessingFilters
    from .data_structures.optics_group import OpticsGroup
    from .data_structures.particle_stack import ParticleStack


def preprocess_image(
    image_rfft: torch.Tensor,
    cumulative_fourier_filters: torch.Tensor,
    bandpass_filter: torch.Tensor,
    full_image_shape: tuple[int, int],
    extracted_box_shape: tuple[int, int],
) -> torch.Tensor:
    """Preprocesses and normalizes the image based on the given filters.

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

    Returns
    -------
    torch.Tensor
        Preprocessed and normalized image in Fourier space
    """
    image_rfft = image_rfft * cumulative_fourier_filters

    # Normalize the image after filtering
    squared_image_rfft = torch.abs(image_rfft) ** 2
    squared_sum = torch.sum(squared_image_rfft, dim=(-2, -1), keepdim=True)
    squared_sum += torch.sum(
        squared_image_rfft[..., :, 1:-1], dim=(-2, -1), keepdim=True
    )
    image_rfft = image_rfft / torch.sqrt(squared_sum)  # Non-in-place preserves gradient

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
    image_rfft = image_rfft * dimensionality**0.5

    # NOTE: We need to rescale based on the relative area of the extracted box
    # to the full image.
    img_h, img_w = full_image_shape
    box_h, box_w = extracted_box_shape
    image_rfft = image_rfft * ((img_h * img_w) / ((box_h) * (box_w))) ** 0.5

    return image_rfft


def calculate_ctf_filter_stack(
    template_shape: tuple[int, int],
    optics_group: "OpticsGroup",
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
) -> torch.Tensor:
    """Calculate searched CTF filter values for a given shape and optics group.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    optics_group : OpticsGroup
        OpticsGroup object containing the optics defining the CTF parameters.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and optics group. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)
    """
    return calculate_ctf_filter_stack_full_args(
        template_shape,
        optics_group.defocus_u,
        optics_group.defocus_v,
        defocus_offsets,
        pixel_size_offsets,
        astigmatism_angle=optics_group.astigmatism_angle,
        voltage=optics_group.voltage,
        spherical_aberration=optics_group.spherical_aberration,
        amplitude_contrast_ratio=optics_group.amplitude_contrast_ratio,
        ctf_B_factor=optics_group.ctf_B_factor,
        phase_shift=optics_group.phase_shift,
        pixel_size=optics_group.pixel_size,
    )


def calculate_ctf_filter_stack_full_args(
    template_shape: tuple[int, int],
    defocus_u: float,  # in Angstrom
    defocus_v: float,  # in Angstrom
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
    **kwargs: Any,
) -> torch.Tensor:
    """Calculate a CTF filter stack for a given set of parameters and search offsets.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    defocus_u : float
        Defocus along the major axis, in Angstroms.
    defocus_v : float
        Defocus along the minor axis, in Angstroms.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.
    **kwargs
        Additional keyword to pass to the calculate_ctf_2d function.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and parameters. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)

    # Raises
    # ------
    # ValueError
    #     If not all the required parameters are passed as additional keyword arguments.
    """
    # Calculate the defocus values + offsets in terms of Angstrom
    defocus = defocus_offsets + ((defocus_u + defocus_v) / 2)
    astigmatism = abs(defocus_u - defocus_v) / 2

    # The different Cs values to search over as another dimension
    cs_values = get_cs_range(
        pixel_size=kwargs["pixel_size"],
        pixel_size_offsets=pixel_size_offsets,
        cs=kwargs["spherical_aberration"],
    )

    # Ensure defocus and astigmatism have a batch dimension so Cs and defocus can be
    # interleaved correctly
    if defocus.dim() == 0:
        defocus = defocus.unsqueeze(0)

    # Loop over spherical aberrations one at a time and collect results
    ctf_list = []
    for cs_val in cs_values:
        tmp = calculate_ctf_2d(
            defocus=defocus * 1e-4,  # Convert to um from Angstrom
            astigmatism=astigmatism * 1e-4,  # Convert to um from Angstrom
            astigmatism_angle=kwargs["astigmatism_angle"],
            voltage=kwargs["voltage"],
            spherical_aberration=cs_val,
            amplitude_contrast=kwargs["amplitude_contrast_ratio"],
            phase_shift=kwargs["phase_shift"],
            pixel_size=kwargs["pixel_size"],
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
        )
        # calc B-envelope and apply
        b_envelope_tmp = b_envelope(
            B=kwargs["ctf_B_factor"],
            image_shape=template_shape,
            pixel_size=kwargs["pixel_size"],
            rfft=True,
            fftshift=False,
            device=tmp.device,
        )
        tmp *= b_envelope_tmp
        ctf_list.append(tmp)

    ctf = torch.stack(ctf_list, dim=0)

    return ctf


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


def get_cs_range(
    pixel_size: float,
    pixel_size_offsets: torch.Tensor,
    cs: float = 2.7,
) -> torch.Tensor:
    """Get the Cs values for a  range of pixel sizes.

    Parameters
    ----------
    pixel_size : float
        The nominal pixel size.
    pixel_size_offsets : torch.Tensor
        The pixel size offsets.
    cs : float, optional
        The Cs (spherical aberration) value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The Cs values for the range of pixel sizes.
    """
    pixel_sizes = pixel_size + pixel_size_offsets
    cs_values = cs / torch.pow(pixel_sizes / pixel_size, 4)
    return cs_values


def cs_to_pixel_size(
    cs_vals: torch.Tensor,
    nominal_pixel_size: float,
    nominal_cs: float = 2.7,
) -> torch.Tensor:
    """Convert Cs values to pixel sizes.

    Parameters
    ----------
    cs_vals : torch.Tensor
        The Cs (spherical aberration) values.
    nominal_pixel_size : float
        The nominal pixel size.
    nominal_cs : float, optional
        The nominal Cs value, by default 2.7.

    Returns
    -------
    torch.Tensor
        The pixel sizes.
    """
    pixel_size = torch.pow(nominal_cs / cs_vals, 0.25) * nominal_pixel_size
    return pixel_size


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


def _setup_ctf_kwargs_from_particle_stack(
    particle_stack: "ParticleStack", template_shape: tuple[int, int]
) -> dict[str, Any]:
    """Helper function for per-particle CTF kwargs.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack to extract the CTF parameters from.
    template_shape : tuple[int, int]
        The shape of the template to use for the CTF calculation.

    Returns
    -------
    dict[str, Any]
        A dictionary of CTF parameters to pass to the CTF calculation function.
    """
    # Keyword arguments for the CTF filter calculation call
    # NOTE: We currently enforce the parameters (other than the defocus values) are
    # all the same. This could be updated in the future...
    assert particle_stack["voltage"].nunique() == 1
    assert particle_stack["spherical_aberration"].nunique() == 1
    assert particle_stack["amplitude_contrast_ratio"].nunique() == 1
    assert particle_stack["phase_shift"].nunique() == 1
    assert particle_stack["ctf_B_factor"].nunique() == 1

    return {
        "voltage": particle_stack["voltage"][0].item(),
        "spherical_aberration": particle_stack["spherical_aberration"][0].item(),
        "amplitude_contrast_ratio": particle_stack["amplitude_contrast_ratio"][
            0
        ].item(),
        "ctf_B_factor": particle_stack["ctf_B_factor"][0].item(),
        "phase_shift": particle_stack["phase_shift"][0].item(),
        "pixel_size": particle_stack["refined_pixel_size"].mean().item(),
        "template_shape": template_shape,
    }


def get_search_tensors(
    min_val: float,
    max_val: float,
    step_size: float,
    skip_enforce_zero: bool = False,
) -> torch.tensor:
    """Get the search tensors (pixel or defocus) for a given range and step size.

    Parameters
    ----------
    min_val : float
        The minimum value.
    max_val : float
        The maximum value.
    step_size : float
        The step size.
    skip_enforce_zero : bool, optional
        Whether to skip enforcing a zero value, by default False.

    Returns
    -------
    torch.tensor
        The search tensors.
    """
    vals = torch.arange(
        min_val,
        max_val + step_size,
        step_size,
        dtype=torch.float32,
    )

    if abs(torch.min(torch.abs(vals))) > 1e-6 and not skip_enforce_zero:
        vals = torch.cat([vals, torch.tensor([0.0])])
        # Re-sort pixel sizes
        vals = torch.sort(vals)[0]

    return vals


def apply_image_filtering(
    particle_stack: "ParticleStack",
    preprocessing_filters: "PreprocessingFilters",
    images_dft: torch.Tensor,
    full_image_shape: tuple[int, int],
    extracted_box_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Apply filtering to a set of images.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the images.
    images_dft : torch.Tensor
        The images in Fourier space.
    full_image_shape: tuple[int, int]
        The shape of the full image.
    extracted_box_shape: tuple[int, int]
        The shape of the extracted box.

    Returns
    -------
    torch.Tensor
        The filtered images in Fourier space
    """
    device = images_dft.device

    # Compute filters without gradient tracking (filters are just preprocessing)
    with torch.no_grad():
        bandpass_filter = (
            preprocessing_filters.bandpass_filter.calculate_bandpass_filter(
                images_dft.shape[-2:]
            ).to(device)
        )
    filter_stack = particle_stack.construct_image_filters(
        preprocessing_filters,
        output_shape=images_dft.shape[-2:],
        images_dft=images_dft.detach(),
    ).to(device)

    return preprocess_image(
        image_rfft=images_dft,
        cumulative_fourier_filters=filter_stack,
        bandpass_filter=bandpass_filter,
        full_image_shape=full_image_shape,
        extracted_box_shape=extracted_box_shape,
    )


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def _process_particle_images_for_filters(
    particle_stack: "ParticleStack",
    preprocessing_filters: "PreprocessingFilters",
    template: torch.Tensor,
    particle_images: torch.Tensor,
    apply_global_filtering: bool,
    projective_filters: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process particle images and compute filters.

    Shared logic for both micrograph and particle image paths.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack containing images to process.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    template : torch.Tensor
        The 3D template volume.
    particle_images : torch.Tensor
        The particle images to process.
    apply_global_filtering : bool
        Whether global filtering was applied.
    projective_filters : torch.Tensor | None
        Pre-computed projective filters (if global filtering was used).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - particle_images_dft: The particle images in Fourier space
        - template_dft: The Fourier transformed template
        - projective_filters: Filters applied to the template
    """
    device = template.device
    box_h, box_w = particle_stack.extracted_box_size

    if not apply_global_filtering:
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))  # pylint: disable=not-callable
        particle_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        # Compute filters without gradient tracking (filters are just preprocessing)
        with torch.no_grad():
            projective_filters = particle_stack.construct_image_filters(
                preprocessing_filters,
                output_shape=(template.shape[-2], template.shape[-1] // 2 + 1),
                images_dft=particle_images_dft.detach(),
            ).to(device)
        particle_images_dft = apply_image_filtering(
            particle_stack,
            preprocessing_filters,
            particle_images_dft,
            full_image_shape=(box_h, box_w),
            extracted_box_shape=(box_h, box_w),
        )
    else:
        particle_images_dft = torch.fft.rfftn(particle_images, dim=(-2, -1))  # pylint: disable=not-callable

    template_dft = volume_to_rfft_fourier_slice(template)

    return (
        particle_images_dft,
        template_dft,
        projective_filters,
    )


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def _setup_images_filters_from_micrographs(
    particle_stack: "ParticleStack",
    preprocessing_filters: "PreprocessingFilters",
    template: torch.Tensor,
    apply_global_filtering: bool,
    movie: torch.Tensor | None,
    deformation_field: torch.Tensor | None,
    pre_exposure: float,
    fluence_per_frame: float,
    image_stack: torch.Tensor | None,
    particle_indices: list[pd.Index] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Setup images and filters when extracting from micrographs.

    Handles the case where images_are_particles=False.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack containing images to process.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    template : torch.Tensor
        The 3D template volume.
    apply_global_filtering : bool
        If True, apply filtering to the full micrograph before particle extraction.
    movie: torch.Tensor | None
        The movie tensor.
    deformation_field: torch.Tensor | None
        The deformation field tensor.
    pre_exposure: float
        The pre-exposure fluence in electrons per Angstrom squared.
    fluence_per_frame: float
        The fluence per frame in electrons per Angstrom squared.
    image_stack: torch.Tensor | None
        The image stack tensor.
    particle_indices: list[pd.Index] | None
        The particle indices to process.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - particle_images_dft: The particle images in Fourier space
        - template_dft: The Fourier transformed template
        - projective_filters: Filters applied to the template
    """
    device = template.device
    box_h, box_w = particle_stack.extracted_box_size
    projective_filters = None

    # Load micrograph images
    if image_stack is not None:
        micrograph_images = image_stack
        micrograph_indexes = particle_indices
        if micrograph_indexes is None:
            raise ValueError(
                "particle_indices must be provided when image_stack is provided."
            )
    else:
        micrograph_images, micrograph_indexes = (
            particle_stack.load_images_grouped_by_column(column_name="micrograph_path")
        )
        micrograph_images = micrograph_images.to(device)

    # Apply global filtering if needed
    if apply_global_filtering:
        img_h, img_w = micrograph_images.shape[-2:]
        micrograph_images_dft = torch.fft.rfftn(micrograph_images, dim=(-2, -1))  # pylint: disable=not-callable
        if micrograph_images.requires_grad:
            micrograph_images_dft = micrograph_images_dft.clone()
        micrograph_images_dft[..., 0, 0] = 0.0 + 0.0j  # Zero out DC component

        # Compute filters without gradient tracking (filters are just preprocessing)
        with torch.no_grad():
            projective_filters = particle_stack.construct_projective_filters(
                preprocessing_filters,
                output_shape=(template.shape[-2], template.shape[-1] // 2 + 1),
                images_dft=micrograph_images_dft.detach(),
                indices=micrograph_indexes,
            ).to(device)
        micrograph_images_dft = apply_image_filtering(
            particle_stack,
            preprocessing_filters,
            micrograph_images_dft,
            full_image_shape=(img_h, img_w),
            extracted_box_shape=(box_h + 1, box_w + 1),
        )
        micrograph_images = torch.fft.irfftn(  # pylint: disable=not-callable
            micrograph_images_dft, dim=(-2, -1)
        )

    # Extract particle images
    if movie is not None and deformation_field is not None:
        particle_images = particle_stack.construct_image_stack_from_movie(
            movie=movie,
            deformation_field=deformation_field,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="reflect",
            padding_value=0.0,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
        )
    else:
        particle_images = particle_stack.construct_image_stack(
            images=micrograph_images,
            indices=micrograph_indexes,
            extraction_size=particle_stack.extracted_box_size,
            pos_reference="top-left",
            padding_value=0.0,
            handle_bounds="pad",
            padding_mode="reflect",  # avoid issues of zeros
        )

    particle_images = particle_images.to(device)

    # Process particle images
    return _process_particle_images_for_filters(
        particle_stack=particle_stack,
        preprocessing_filters=preprocessing_filters,
        template=template,
        particle_images=particle_images,
        apply_global_filtering=apply_global_filtering,
        projective_filters=projective_filters,
    )


# pylint: disable=too-many-arguments
def _setup_images_filters_from_particles(
    particle_stack: "ParticleStack",
    preprocessing_filters: "PreprocessingFilters",
    template: torch.Tensor,
    apply_global_filtering: bool,
    image_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Setup images and filters when images are already particles.

    Handles the case where images_are_particles=True.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack containing images to process.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    template : torch.Tensor
        The 3D template volume.
    apply_global_filtering : bool
        If True, apply filtering to the full micrograph before particle extraction.
    image_stack: torch.Tensor
        The image stack tensor (must be provided).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - particle_images_dft: The particle images in Fourier space
        - template_dft: The Fourier transformed template
        - projective_filters: Filters applied to the template
    """
    particle_images = image_stack
    particle_stack.image_stack = particle_images

    return _process_particle_images_for_filters(
        particle_stack=particle_stack,
        preprocessing_filters=preprocessing_filters,
        template=template,
        particle_images=particle_images,
        apply_global_filtering=apply_global_filtering,
        projective_filters=None,
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def setup_images_filters_particle_stack(
    particle_stack: "ParticleStack",
    preprocessing_filters: "PreprocessingFilters",
    template: torch.Tensor,
    apply_global_filtering: bool = True,
    movie: torch.Tensor | None = None,
    deformation_field: torch.Tensor | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    image_stack: torch.Tensor | None = None,
    particle_indices: list[pd.Index] | None = None,
    images_are_particles: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and preprocess particle images and calculate filters.

    This function extracts particle images from a particle stack, performs FFT,
    applies filters, and prepares the template for further processing.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack containing images to process.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    template : torch.Tensor
        The 3D template volume.
    apply_global_filtering : bool, optional
        If True, apply filtering to the full micrograph before particle extraction.
        If False, filters are calculated and applied to the cropped particle images.
        Default is True.
    movie: torch.Tensor | None
        The movie tensor.
    deformation_field: torch.Tensor | None
        The deformation field tensor.
    pre_exposure: float
        The pre-exposure fluence in electrons per Angstrom squared.
    fluence_per_frame: float
        The fluence per frame in electrons per Angstrom squared.
    image_stack: torch.Tensor | None
        The image stack tensor.
    particle_indices: list[pd.Index] | None
        The particle indices to process.
    images_are_particles: bool
        Whether the images are particles or not. Defaults to False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - particle_images_dft: The particle images in Fourier space
        - template_dft: The Fourier transformed template
        - projective_filters: Filters applied to the template
    """
    if images_are_particles:
        if image_stack is None:
            raise ValueError(
                "image_stack must be provided when images_are_particles=True."
            )
        return _setup_images_filters_from_particles(
            particle_stack=particle_stack,
            preprocessing_filters=preprocessing_filters,
            template=template,
            apply_global_filtering=apply_global_filtering,
            image_stack=image_stack,
        )
    return _setup_images_filters_from_micrographs(
        particle_stack=particle_stack,
        preprocessing_filters=preprocessing_filters,
        template=template,
        apply_global_filtering=apply_global_filtering,
        movie=movie,
        deformation_field=deformation_field,
        pre_exposure=pre_exposure,
        fluence_per_frame=fluence_per_frame,
        image_stack=image_stack,
        particle_indices=particle_indices,
    )


# pylint: disable=too-many-arguments
def _setup_correlation_stacks_from_micrographs(
    particle_stack: "ParticleStack",
    mean_stack: torch.Tensor | None,
    std_stack: torch.Tensor | None,
    particle_indices: list[pd.Index] | None,
    extracted_box_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Setup correlation mean and std stacks from micrographs.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack containing images to process.
    mean_stack : torch.Tensor | None
        Pre-loaded mean stack tensor.
    std_stack : torch.Tensor | None
        Pre-loaded std stack tensor.
    particle_indices : list[pd.Index] | None
        The particle indices to process.
    extracted_box_size : tuple[int, int]
        The size of the extracted box.
    device : torch.device
        The device to use.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - corr_mean_stack: The mean correlation stack
        - corr_std_stack: The standard deviation correlation stack
    """
    # Setup mean stack
    if mean_stack is None:
        correlation_avg_images, correlation_avg_indexes = (
            particle_stack.load_images_grouped_by_column(
                column_name="correlation_average_path"
            )
        )
    else:
        correlation_avg_images = mean_stack
        if particle_indices is None:
            raise ValueError(
                "particle_indices must be provided when mean_stack is provided."
            )
        correlation_avg_indexes = particle_indices

    corr_mean_stack = particle_stack.construct_image_stack(
        images=correlation_avg_images,
        indices=correlation_avg_indexes,
        extraction_size=extracted_box_size,
        pos_reference="top-left",
        handle_bounds="pad",
        padding_mode="constant",
        padding_value=0.0,
    ).to(device)

    # Setup std stack
    if std_stack is None:
        correlation_var_images, correlation_var_indexes = (
            particle_stack.load_images_grouped_by_column(
                column_name="correlation_variance_path"
            )
        )
    else:
        correlation_var_images = std_stack
        if particle_indices is None:
            raise ValueError(
                "particle_indices must be provided when std_stack is provided."
            )
        correlation_var_indexes = particle_indices

    corr_std_stack = particle_stack.construct_image_stack(
        images=correlation_var_images,
        indices=correlation_var_indexes,
        extraction_size=extracted_box_size,
        pos_reference="top-left",
        handle_bounds="pad",
        padding_mode="constant",
        padding_value=1e10,  # large to avoid out of bound pixels having inf z-score
    ).to(device)

    corr_std_stack = corr_std_stack**0.5  # Convert variance to standard deviation

    return corr_mean_stack, corr_std_stack


def _setup_correlation_stacks_from_particles(
    mean_stack: torch.Tensor | None,
    std_stack: torch.Tensor | None,
    particle_indices: list[pd.Index] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Setup correlation mean and std stacks from pre-loaded particles.

    Parameters
    ----------
    mean_stack : torch.Tensor | None
        Pre-loaded mean stack tensor.
    std_stack : torch.Tensor | None
        Pre-loaded std stack tensor.
    particle_indices : list[pd.Index] | None
        The particle indices to process.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - corr_mean_stack: The mean correlation stack
        - corr_std_stack: The standard deviation correlation stack

    Raises
    ------
    ValueError
        If particle_indices, mean_stack, or std_stack are None.
    """
    if particle_indices is None:
        raise ValueError(
            "particle_indices must be provided when images_are_particles is True."
        )
    if mean_stack is None or std_stack is None:
        raise ValueError(
            "mean_stack and std_stack must be provided when "
            "images_are_particles is True."
        )

    corr_std_stack = std_stack**0.5  # Convert variance to standard deviation

    return mean_stack, corr_std_stack


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def setup_particle_backend_kwargs(
    particle_stack: "ParticleStack",
    template: torch.Tensor,
    preprocessing_filters: "PreprocessingFilters",
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    apply_global_filtering: bool,
    device_list: list,
    movie: torch.Tensor | None = None,
    deformation_field: CubicCatmullRomGrid3d | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    image_stack: torch.Tensor | None = None,
    mean_stack: torch.Tensor | None = None,
    std_stack: torch.Tensor | None = None,
    particle_indices: list[pd.Index] | None = None,
    images_are_particles: bool = False,
) -> dict[str, Any]:
    """Create common kwargs dictionary for template backend functions.

    This function extracts the common code between RefineTemplateManager and
    OptimizeTemplateManager's make_backend_core_function_kwargs methods.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack containing images to process.
    template : torch.Tensor
        The 3D template volume.
    preprocessing_filters : PreprocessingFilters
        Filters to apply to the particle images.
    euler_angles : torch.Tensor
        The set of Euler angles to use.
    euler_angle_offsets : torch.Tensor
        The relative Euler angle offsets to search over.
    defocus_offsets : torch.Tensor
        The relative defocus values to search over.
    pixel_size_offsets : torch.Tensor
        The relative pixel size values to search over.
    apply_global_filtering : bool
        If True, apply filtering to the full micrograph before particle extraction.
        If False, filters are calculated and applied to the cropped particle images.
    device_list : list
        List of computational devices to use.
    movie: torch.Tensor | None
        The movie tensor.
    deformation_field: torch.Tensor | None
        The deformation field tensor.
    pre_exposure: float
        The pre-exposure fluence in electrons per Angstrom squared.
    fluence_per_frame: float
        The fluence per frame in electrons per Angstrom squared.
    image_stack: torch.Tensor | None
        The image stack tensor.
    mean_stack: torch.Tensor | None
        The mean stack tensor.
    std_stack: torch.Tensor | None
        The std stack tensor.
    particle_indices: list[pd.Index] | None
        The particle indices to process.
    images_are_particles: bool
        Whether the images are particles or not. Defaults to False.

    Returns
    -------
    dict[str, Any]
        Dictionary of keyword arguments for backend functions.
    """
    device = template.device
    h, w = particle_stack.original_template_size
    box_h, box_w = particle_stack.extracted_box_size
    extracted_box_size = (box_h - h + 1, box_w - w + 1)

    # Setup correlation stacks
    if images_are_particles:
        corr_mean_stack, corr_std_stack = _setup_correlation_stacks_from_particles(
            mean_stack=mean_stack,
            std_stack=std_stack,
            particle_indices=particle_indices,
        )
    else:
        corr_mean_stack, corr_std_stack = _setup_correlation_stacks_from_micrographs(
            particle_stack=particle_stack,
            mean_stack=mean_stack,
            std_stack=std_stack,
            particle_indices=particle_indices,
            extracted_box_size=extracted_box_size,
            device=device,
        )

    (
        particle_images_dft,
        template_dft,
        projective_filters,
    ) = setup_images_filters_particle_stack(
        particle_stack=particle_stack,
        preprocessing_filters=preprocessing_filters,
        template=template,
        apply_global_filtering=apply_global_filtering,
        movie=movie,
        deformation_field=deformation_field,
        pre_exposure=pre_exposure,
        fluence_per_frame=fluence_per_frame,
        image_stack=image_stack,
        particle_indices=particle_indices,
        images_are_particles=images_are_particles,
    )

    # The best defocus values for each particle (+ astigmatism)
    defocus_u, defocus_v = particle_stack.get_absolute_defocus()
    defocus_u = defocus_u.to(device)
    defocus_v = defocus_v.to(device)
    defocus_angle = torch.tensor(particle_stack["astigmatism_angle"], device=device)

    ctf_kwargs = _setup_ctf_kwargs_from_particle_stack(
        particle_stack, (template.shape[-2], template.shape[-1])
    )

    return {
        "particle_stack_dft": particle_images_dft,
        "template_dft": template_dft,
        "euler_angles": euler_angles,
        "euler_angle_offsets": euler_angle_offsets,
        "defocus_u": defocus_u,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "defocus_offsets": defocus_offsets,
        "pixel_size_offsets": pixel_size_offsets,
        "corr_mean": corr_mean_stack,
        "corr_std": corr_std_stack,
        "ctf_kwargs": ctf_kwargs,
        "projective_filters": projective_filters,
        "device": device_list,
    }
