"""Per-particle extraction from movies, with optional motion correction.

Private module: the particle stack classes are the public interface. Particle positions
arrive here already converted to crop origins and template centers (see
``ParticleBoxGeometry``); per-particle shifts are either given directly or sampled from
a deformation field at the template centers.
"""

import warnings
from typing import Literal

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import coordinate_grid
from torch_motion_correction.correct_motion import get_pixel_shifts
from torch_motion_correction.deformation_field import DeformationField

from leopard_em.utils.cropping import get_cropped_image_regions
from leopard_em.utils.image_processing import dose_weight_movie_to_micrograph


def check_motion_source(
    deformation_field: DeformationField | None,
    particle_shifts: torch.Tensor | None,
    require_motion_source: bool,
) -> None:
    """Validate that at most one (or, if required, exactly one) motion source is given.

    Parameters
    ----------
    deformation_field : DeformationField | None
        The deformation field grid.
    particle_shifts : torch.Tensor | None
        Per-particle shifts, shape ``(T, N, 2)``.
    require_motion_source : bool
        If True, raise when neither motion source is provided. If False, warn that the
        movie is assumed to be aligned already.
    """
    if deformation_field is not None and particle_shifts is not None:
        raise ValueError(
            "Only one of `deformation_field` or `particle_shifts` can be provided."
        )
    if deformation_field is None and particle_shifts is None:
        if require_motion_source:
            raise ValueError(
                "One of `deformation_field` or `particle_shifts` must be provided."
            )
        warnings.warn(
            "No deformation field or particle shifts were provided. Assuming the "
            "movie is already aligned and extracting each frame without shifts.",
            stacklevel=3,
        )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def shift_and_crop_frame(
    movie_frame: torch.Tensor,
    shifts: torch.Tensor,  # (N, 2) -> (dy, dx)
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    extracted_box_size: tuple[int, int],
    handle_bounds: Literal["pad", "error"],
    padding_mode: Literal["constant", "reflect", "replicate"],
    padding_value: float,
) -> torch.Tensor:
    """Process a single frame using precomputed particle shifts.

    Safe for gradient checkpointing; contains no deformation-field evaluation.

    Parameters
    ----------
    movie_frame : torch.Tensor
        Single movie frame (H, W).
    shifts : torch.Tensor
        Per-particle shifts with shape (N, 2) as (dy, dx).
    pos_y, pos_x : torch.Tensor
        Top-left extraction positions.
    extracted_box_size : tuple[int, int]
        ``(box_h, box_w)``.
    handle_bounds, padding_mode, padding_value
        Passed through to cropping.

    Returns
    -------
    torch.Tensor
        Shifted FFTs with shape ``(N, box_h, box_w//2 + 1)``.
    """
    box_h, box_w = extracted_box_size

    cropped_images = get_cropped_image_regions(
        movie_frame,
        pos_y,
        pos_x,
        extracted_box_size,
        pos_reference="top-left",
        handle_bounds=handle_bounds,
        padding_mode=padding_mode,
        padding_value=padding_value,
    )

    cropped_images_dft = torch.fft.rfftn(  # pylint: disable=not-callable
        cropped_images, dim=(-2, -1)
    )

    shifted_fft = fourier_shift_dft_2d(
        dft=cropped_images_dft,
        image_shape=(box_h, box_w),
        shifts=shifts,
        rfft=True,
        fftshifted=False,
    )

    return shifted_fft


def frame_particle_shifts_from_deformation(
    movie_frame: torch.Tensor,
    deformation_field: DeformationField,
    normalized_t_value: torch.Tensor,
    pixel_grid: torch.Tensor,
    pixel_spacing: float,
    pos_y_center: torch.Tensor,
    pos_x_center: torch.Tensor,
    gh: int,
    gw: int,
) -> torch.Tensor:
    """Compute per-particle shifts for a single frame from a deformation field.

    Parameters
    ----------
    movie_frame : torch.Tensor
        Single movie frame (H, W).
    deformation_field : CubicCatmullRomGrid3d
        The deformation field grid.
    normalized_t_value : torch.Tensor
        Normalized time value for the frame.
    pixel_grid : torch.Tensor
        The pixel grid tensor.
    pixel_spacing : float
        The pixel spacing.
    pos_y_center : torch.Tensor
        Center y positions.
    pos_x_center : torch.Tensor
        Center x positions.
    gh : int
        Height of the deformation field grid.
    gw : int
        Width of the deformation field grid.

    Returns
    -------
    torch.Tensor
        Shifts with shape ``(N, 2)`` as (dy, dx).
    """
    frame_deformation_field = deformation_field.evaluate_at_t(
        t=float(normalized_t_value.item()),
        grid_shape=(10 * gh, 10 * gw),
    )

    pixel_shifts = get_pixel_shifts(
        frame=movie_frame,
        pixel_spacing=pixel_spacing,
        frame_deformation_grid=frame_deformation_field,
        pixel_grid=pixel_grid,
    )

    y_shifts = -pixel_shifts[pos_y_center, pos_x_center, 0]
    x_shifts = -pixel_shifts[pos_y_center, pos_x_center, 1]

    return torch.stack((y_shifts, x_shifts), dim=-1)


# pylint: disable=too-many-locals
def particle_movie_rfft_stack(
    movie: torch.Tensor,
    origins: tuple[torch.Tensor, torch.Tensor],
    centers: tuple[torch.Tensor, torch.Tensor],
    extracted_box_size: tuple[int, int],
    pixel_sizes: torch.Tensor,
    *,
    deformation_field: DeformationField | None = None,
    particle_shifts: torch.Tensor | None = None,
    normalized_t_values: torch.Tensor | None = None,
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
    use_gradient_checkpointing: bool = True,
) -> torch.Tensor:
    """Crop each particle from every (mean-subtracted) frame and apply its shifts.

    Parameters
    ----------
    movie : torch.Tensor
        The ``(t, H, W)`` movie.
    origins : tuple[torch.Tensor, torch.Tensor]
        Top-left ``(y, x)`` crop origins of the N processed particles.
    centers : tuple[torch.Tensor, torch.Tensor]
        ``(y, x)`` template centers of the processed particles, where motion is
        sampled from ``deformation_field``.
    extracted_box_size : tuple[int, int]
        ``(box_h, box_w)``.
    pixel_sizes : torch.Tensor
        Per-row pixel sizes of the whole stack; the first is the deformation field's
        pixel spacing.
    deformation_field : DeformationField | None, optional
        Deformation field to sample per-particle shifts from.
    particle_shifts : torch.Tensor | None, optional
        Shifts of the processed particles, shape ``(t, N, 2)``. With neither motion
        source, frames are cropped without shifts.
    normalized_t_values : torch.Tensor | None, optional
        Normalized time of each frame, shape ``(t,)``. Defaults to a 0 to 1 ramp.
    handle_bounds, padding_mode, padding_value
        Passed through to cropping.
    use_gradient_checkpointing : bool, optional
        Trade compute for memory during frame processing, by default True.

    Returns
    -------
    torch.Tensor
        Per-particle movie frame DFTs of shape ``(N, t, box_h, box_w // 2 + 1)``.
    """
    pos_y, pos_x = origins
    pos_y_center, pos_x_center = centers
    box_h, box_w = extracted_box_size
    t, img_h, img_w = movie.shape
    num_particles_to_process = len(pos_y)
    if deformation_field is not None:
        _, _, gh, gw = deformation_field.data.shape
    else:
        gh = gw = 0
    if normalized_t_values is None:
        normalized_t = torch.linspace(0, 1, steps=t, device=movie.device)
    else:
        if normalized_t_values.numel() != t:
            raise ValueError("normalized_t_values must have one entry per movie frame.")
        normalized_t = normalized_t_values.to(device=movie.device).reshape(t)
    pixel_grid = coordinate_grid(
        image_shape=(img_h, img_w),
        device=movie.device,
    )

    aligned_particle_movies_rfft = torch.zeros(
        (num_particles_to_process, t, box_h, box_w // 2 + 1),
        dtype=torch.complex64,
        device=movie.device,
    )
    movie = movie - torch.mean(movie, dim=(-2, -1), keepdim=True)

    for frame_index, movie_frame in enumerate(movie):
        if particle_shifts is not None:
            frame_shifts = particle_shifts[frame_index]  # (N, 2)
        elif deformation_field is None:
            frame_shifts = torch.zeros(
                (num_particles_to_process, 2),
                dtype=movie.dtype,
                device=movie.device,
            )
        else:
            frame_shifts = frame_particle_shifts_from_deformation(
                movie_frame=movie_frame,
                deformation_field=deformation_field,
                normalized_t_value=normalized_t[frame_index],
                pixel_grid=pixel_grid,
                pixel_spacing=pixel_sizes[0].item(),
                pos_y_center=pos_y_center,
                pos_x_center=pos_x_center,
                gh=gh,
                gw=gw,
            )

        if use_gradient_checkpointing:
            shifted_fft = checkpoint(
                shift_and_crop_frame,
                movie_frame,
                frame_shifts,
                pos_y,
                pos_x,
                extracted_box_size,
                handle_bounds,
                padding_mode,
                padding_value,
                use_reentrant=False,
            )
        else:
            shifted_fft = shift_and_crop_frame(
                movie_frame=movie_frame,
                shifts=frame_shifts,
                pos_y=pos_y,
                pos_x=pos_x,
                extracted_box_size=extracted_box_size,
                handle_bounds=handle_bounds,
                padding_mode=padding_mode,
                padding_value=padding_value,
            )

        aligned_particle_movies_rfft[:, frame_index] = shifted_fft

        if frame_index % 10 == 0 and frame_index > 0:
            torch.cuda.empty_cache()
    return aligned_particle_movies_rfft


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def dose_weight_particle_movies(
    particle_movies_rfft: torch.Tensor,
    extracted_box_size: tuple[int, int],
    rows: list[int],
    pixel_sizes: torch.Tensor,
    voltages: np.ndarray,
    pre_exposure: float,
    fluence_per_frame: float,
) -> torch.Tensor:
    """Dose weight and sum each particle's movie frames into one image.

    Parameters
    ----------
    particle_movies_rfft : torch.Tensor
        Per-particle movie frame DFTs, shape ``(N, t, box_h, box_w // 2 + 1)``.
    extracted_box_size : tuple[int, int]
        ``(box_h, box_w)``.
    rows : list[int]
        Row position (in the whole stack) of each of the N particles.
    pixel_sizes : torch.Tensor
        Per-row pixel sizes of the whole stack.
    voltages : np.ndarray
        Per-row microscope voltages of the whole stack.
    pre_exposure : float
        Pre-exposure in electrons per pixel.
    fluence_per_frame : float
        Dose per frame in electrons per pixel.

    Returns
    -------
    torch.Tensor
        Dose-weighted particle images, shape ``(N, box_h, box_w)``.
    """
    num_particles = particle_movies_rfft.shape[0]
    box_h, box_w = extracted_box_size
    aligned_particle_images = torch.zeros(
        (num_particles, box_h, box_w),
        device=particle_movies_rfft.device,
    )
    for particle_index in range(num_particles):
        particle_dft = particle_movies_rfft[particle_index]

        # Row position of this particle in the full stack
        row = rows[particle_index]

        dw_sum = dose_weight_movie_to_micrograph(
            movie_fft=particle_dft,
            pixel_size=float(pixel_sizes[row].item()),
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            voltage=voltages[row],
        )
        aligned_particle_images[particle_index] = dw_sum

    return aligned_particle_images
