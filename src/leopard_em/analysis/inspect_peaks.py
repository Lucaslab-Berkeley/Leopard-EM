"""Functions for inspecting cross-correlations of peaks across local neighborhood."""

# TODO: Integrate Pydantic models into a manager-like class where a user can easily
# specify the offset ranges from the configuration files and get the correlation stack
# from there. This would be similar to the RefineTemplateManager... Calling big
# functions over and over again is kinda nasty.

import roma
import torch

from leopard_em.backend.cross_correlation import do_batched_orientation_cross_correlate
from leopard_em.pydantic_models.utils import calculate_ctf_filter_stack_full_args


def theta_phi_offsets_to_rotmats(
    theta_offsets: torch.Tensor,  # (num_thetas,)
    phi_offsets: torch.Tensor,  # (num_phis,)
    degrees: bool = True,
) -> torch.Tensor:  # (num_thetas * num_phis, 3, 3)
    """Convert two offset angles into a dense meshgrid of rotation matrices.

    Parameters
    ----------
    theta_offsets : torch.Tensor
        A tensor of shape (num_thetas,) containing the theta offsets in degrees or
        radians.
    phi_offsets : torch.Tensor
        A tensor of shape (num_phis,) containing the phi offsets in degrees or radians.
    degrees : bool, optional
        Whether the input angles are in degrees. If False, they are assumed to be in
        radians. Default is True.
    device : torch.device, optional
        The device on which the computation will be performed (e.g., 'cpu' or 'cuda').
        Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_thetas * num_phis, 3, 3) containing the rotation
        matrices corresponding to all combinations of the input theta and phi offsets.
    """
    tt, pp = torch.meshgrid(theta_offsets, phi_offsets, indexing="ij")
    tt = tt.flatten()
    pp = pp.flatten()

    euler_angle_offsets = torch.stack((pp, tt, torch.zeros_like(pp)), dim=1)

    return roma.euler_to_rotmat("ZYZ", euler_angle_offsets, degrees=degrees)


def inspect_correlation_peaks(
    particle_stack_dft: torch.Tensor,  # (N, H, W // 2 + 1)
    template_dft: torch.Tensor,  # (d, h, w // 2 + 1)
    projective_filters: torch.Tensor,  # (N, h, w // 2 + 1)
    pixel_size: torch.Tensor,  # (N,)
    defocus_u: torch.Tensor,  # (N,)
    defocus_v: torch.Tensor,  # (N,)
    defocus_angle: torch.Tensor,  # (N,)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    pixel_size_offsets: torch.Tensor,  # (num_Cs,)
    defocus_offsets: torch.Tensor,  # (num_defocus, )
    rotation_matrices_offset: torch.Tensor,  # (num_rotations, 3, 3)
    ctf_kwargs: dict,
    device: torch.device,
    pixel_size_batch_size: int = -1,
    defocus_batch_size: int = -1,
    orientation_batch_size: int = -1,
    apply_projection_normalization: bool = True,
    return_valid_region: bool = True,
) -> torch.Tensor:  # (N, num_Cs, num_defocus, num_rotations, h, w)
    """Helper func to inspect  cross-correlation of peaks across local neighborhood.

    This function, rather than integrating into a larger program, is intended as a user-
    friendly mechanism to compute the cross-correlation of particle images against a
    reference template. Note that this function can return a large tensor depending on
    the requested offsets and rotations.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        A stack of particle images in DFT (real-FFT) format with shape
        (N, H, W // 2 + 1), where N is the number of particles, H is the height, and W
        is the width of the images. These images can have pre-filtering applied
        (e.g., whitening).
    template_dft : torch.Tensor
        The DFT (real-FFT) of the 3D template volume ready to take Fourier slices from.
        Shape should be (d, h, w // 2 + 1), where d is the depth, h is the height, and w
        is the width of the template. Projections will be of shape (h, w).
    projective_filters : torch.Tensor
        Desired projective filter to apply per particle (except CTF) per-projection.
        Shape should be (N, h, w // 2 + 1).
    pixel_size : torch.Tensor
        A tensor of shape (N,) containing the pixel size for each particle image in
        Angstroms.
    defocus_u : torch.Tensor
        A tensor of shape (N,) containing the defocus values along the major axis for
        each particle image in Angstroms.
    defocus_v : torch.Tensor
        A tensor of shape (N,) containing the defocus values along the minor axis for
        each particle image in Angstroms.
    defocus_angle : torch.Tensor
        A tensor of shape (N,) containing the defocus angles for each particle image in
        degrees.
    rotation_matrices : torch.Tensor
        A tensor of shape (N, 3, 3) containing the rotation matrices for each particle
        image.
    pixel_size_offsets : torch.Tensor
        A tensor of shape (num_Cs,) containing the pixel size offsets to apply to each
        particle's pixel size. The actual pixel sizes used will be
        `pixel_size + pixel_size_offsets[i]` for each offset i.
    defocus_offsets : torch.Tensor
        A tensor of shape (num_defocus,) containing the defocus offsets to apply to
        each particle's defocus values. The actual defocus values used will be
        `defocus_u + defocus_offsets[i]` and `defocus_v + defocus_offsets[i]` for
        each offset i.
    rotation_matrices_offset : torch.Tensor
        A tensor of shape (num_rotations, 3, 3) containing the rotation matrices to
        apply as offsets to each particle's rotation matrix. The actual rotation
        matrices used will be `rotation_matrices @ rotation_matrices_offset[i]` for
        each offset i.
    ctf_kwargs : dict
        Additional keyword arguments for the CTF calculation, such as voltage,
        spherical aberration, amplitude contrast, etc. Constant across all particles.
    device : torch.device
        The device on which the computation will be performed (e.g., 'cpu' or 'cuda').
    pixel_size_batch_size : int, optional
        The batch size for processing different pixel size offsets. If -1, all pixel
        size offsets will be processed in a single batch. Default is -1.
    defocus_batch_size : int, optional
        The batch size for processing different defocus offsets. If -1, all defocus
        offsets will be processed in a single batch. Default is -1.
    orientation_batch_size : int, optional
        The batch size for processing different rotation matrix offsets. If -1, all
        rotation offsets will be processed in a single batch. Default is -1.
    apply_projection_normalization : bool, optional
        Whether to apply normalization to the projections. Default is True.
    return_valid_region : bool, optional
        Whether to return only the valid region of the cross-correlation. If True,
        the output will be cropped to the valid region. If False, the full cross-
        correlation will be returned. Default is True.

    Returns
    -------
    torch.Tensor
        A cross-correlation tensor of shape
        (N, num_Cs, num_defocus, num_rotations, H - h + 1, W - w + 1) if
        return_valid_region is True, otherwise
        (N, num_Cs, num_defocus, num_rotations, H, W), where N is the number of
        particles, num_Cs is the number of pixel size offsets, num_defocus is the
        number of defocus offsets, num_rotations is the number of rotation matrix
        offsets, and (H, W) is the shape of the input images.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # Send all tensors to the specified device
    particle_stack_dft = particle_stack_dft.to(device)
    template_dft = template_dft.to(device)
    projective_filters = projective_filters.to(device)
    rotation_matrices = rotation_matrices.to(device)
    rotation_matrices_offset = rotation_matrices_offset.to(device)

    # Allocate empty output tensor on device
    num_particles = particle_stack_dft.shape[0]
    num_Cs = pixel_size_offsets.shape[0]
    num_defocus = defocus_offsets.shape[0]
    num_rotations = rotation_matrices_offset.shape[0]
    last_dim_y = (
        particle_stack_dft.shape[-2] - template_dft.shape[-2] + 1
        if return_valid_region
        else particle_stack_dft.shape[-2]
    )
    last_dim_x = (
        (particle_stack_dft.shape[-1] * 2) - (template_dft.shape[-1] * 2) + 1
        if return_valid_region
        else particle_stack_dft.shape[-1] * 2 - 2
    )
    output = torch.empty(
        (num_particles, num_Cs, num_defocus, num_rotations, last_dim_y, last_dim_x),
        dtype=particle_stack_dft.real.dtype,
        device=device,
    )

    for i in range(0, particle_stack_dft.shape[0]):
        output[i] = cross_correlate_single_particle(
            image_dft=particle_stack_dft[i],
            template_dft=template_dft,
            projective_filter=projective_filters[i],
            pixel_size=pixel_size[i],
            defocus_u=defocus_u[i],
            defocus_v=defocus_v[i],
            defocus_angle=defocus_angle[i],
            rotation_matrix=rotation_matrices[i],
            pixel_size_offsets=pixel_size_offsets,
            defocus_offsets=defocus_offsets,
            rotation_matrices_offset=rotation_matrices_offset,
            ctf_kwargs=ctf_kwargs,
            pixel_size_batch_size=pixel_size_batch_size,
            defocus_batch_size=defocus_batch_size,
            orientation_batch_size=orientation_batch_size,
            apply_projection_normalization=apply_projection_normalization,
            return_valid_region=return_valid_region,
        )

    return output


def cross_correlate_single_particle(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    projective_filter: torch.Tensor,
    pixel_size: float,
    defocus_u: float,
    defocus_v: float,
    defocus_angle: float,
    rotation_matrix: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    defocus_offsets: torch.Tensor,
    rotation_matrices_offset: torch.Tensor,
    ctf_kwargs: dict,
    pixel_size_batch_size: int = -1,
    defocus_batch_size: int = -1,
    orientation_batch_size: int = -1,
    apply_projection_normalization: bool = True,
    return_valid_region: bool = True,
) -> torch.Tensor:
    """Function to compute the local cross-correlations for a single particle."""
    device = image_dft.device
    num_Cs = pixel_size_offsets.shape[0]
    num_defocus = defocus_offsets.shape[0]
    num_rotations = rotation_matrices_offset.shape[0]

    # Determine batch sizes
    pixel_size_batch_size = (
        num_Cs if pixel_size_batch_size == -1 else pixel_size_batch_size
    )
    defocus_batch_size = num_defocus if defocus_batch_size == -1 else defocus_batch_size
    orientation_batch_size = (
        num_rotations if orientation_batch_size == -1 else orientation_batch_size
    )

    # Valid slicing based on correlation shapes
    projection_shape_real = (template_dft.shape[1], template_dft.shape[2] * 2 - 2)
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)
    valid_slice_y = slice((image_shape_real[0] - projection_shape_real[0]) + 1)
    valid_slice_x = slice((image_shape_real[1] - projection_shape_real[1]) + 1)

    out = torch.empty(
        (
            num_Cs,
            num_defocus,
            num_rotations,
            image_shape_real[-2],
            image_shape_real[-1],
        ),
        dtype=image_dft.real.dtype,
        device=device,
    )

    for i in range(0, num_Cs, pixel_size_batch_size):
        pixel_size_batch = pixel_size_offsets[i : i + pixel_size_batch_size]
        for j in range(0, num_defocus, defocus_batch_size):
            defocus_u_batch = defocus_u + defocus_offsets[j : j + defocus_batch_size]
            defocus_v_batch = defocus_v + defocus_offsets[j : j + defocus_batch_size]

            # Re-calculate the CTF from the pixel size and defocus
            ctf_filters = calculate_ctf_filter_stack_full_args(
                defocus_u=defocus_u_batch,
                defocus_v=defocus_v_batch,
                astigmatism_angle=defocus_angle,
                defocus_offsets=defocus_offsets,
                pixel_size_offsets=pixel_size_batch,
                **ctf_kwargs,
            )
            ctf_filters = ctf_filters.to(device)

            # Combine into a single projective filter
            combined_projective_filter = (
                projective_filter[None, None, ...] * ctf_filters
            )

            for k in range(0, num_rotations, orientation_batch_size):
                rotation_offsets_batch = rotation_matrices_offset[
                    k : k + orientation_batch_size
                ]
                rotmat_composed = roma.rotmat_composition(
                    (rotation_offsets_batch, rotation_matrix)
                )

                tmp = do_batched_orientation_cross_correlate(
                    image_dft=image_dft,
                    template_dft=template_dft,
                    rotation_matrices=rotmat_composed,
                    projective_filters=combined_projective_filter,
                    apply_normalization=apply_projection_normalization,
                )

                out[
                    i : i + pixel_size_batch.shape[0],
                    j : j + defocus_u_batch.shape[0],
                    k : k + rotation_offsets_batch.shape[0],
                ] = tmp

    if return_valid_region:
        out = out[..., valid_slice_y, valid_slice_x]

    return out
