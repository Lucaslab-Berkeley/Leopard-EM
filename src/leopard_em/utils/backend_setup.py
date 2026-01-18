"""Backend setup and orchestration utility functions."""

from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from leopard_em.utils.ctf_utils import _setup_ctf_kwargs_from_particle_stack
from leopard_em.utils.image_processing import (
    apply_image_filtering,
    volume_to_rfft_fourier_slice,
)

# Using the TYPE_CHECKING statement to avoid circular imports
if TYPE_CHECKING:
    from leopard_em.pydantic_models.config.correlation_filters import (
        PreprocessingFilters,
    )
    from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack


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
    deformation_field: CubicCatmullRomGrid3d | None,
    particle_shifts: torch.Tensor | None,
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
    deformation_field: CubicCatmullRomGrid3d | None
        The deformation field tensor.
    particle_shifts: torch.Tensor | None
        The particle shifts tensor. If provided, takes precedence over
        deformation_field. Shape is (T, N, 2) where T is frames, N is particles.
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
    if movie is not None and (
        deformation_field is not None or particle_shifts is not None
    ):
        particle_images = particle_stack.construct_image_stack_from_movie(
            movie=movie,
            deformation_field=deformation_field,
            particle_shifts=particle_shifts,
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
    deformation_field: CubicCatmullRomGrid3d | None = None,
    particle_shifts: torch.Tensor | None = None,
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
    deformation_field: CubicCatmullRomGrid3d | None
        The deformation field tensor.
    particle_shifts: torch.Tensor | None
        The particle shifts tensor. If provided, takes precedence over
        deformation_field. Shape is (T, N, 2) where T is frames, N is particles.
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
        particle_shifts=particle_shifts,
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
    particle_shifts: torch.Tensor | None = None,
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
    deformation_field: CubicCatmullRomGrid3d | None
        The deformation field tensor.
    particle_shifts: torch.Tensor | None
        The particle shifts tensor. If provided, takes precedence over
        deformation_field. Shape is (T, N, 2) where T is frames, N is particles.
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
        particle_shifts=particle_shifts,
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

    # Extract mag_matrix from particle stack and convert to 2x2 tensor
    # All particles should have the same mag_matrix value
    mag_matrix_tensor = ctf_kwargs["mag_matrix"]

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
        "transform_matrix": mag_matrix_tensor,
    }
