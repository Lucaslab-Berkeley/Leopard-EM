"""Backend functions related to correlating and refining particle stacks."""

from typing import Any, Literal

import roma
import torch
import tqdm
from torch_fourier_slice import extract_central_slices_rfft_3d

from leopard_em.backend.core_match_template import (
    _do_bached_orientation_cross_correlate,
    _do_bached_orientation_cross_correlate_cpu,
)
from leopard_em.backend.utils import normalize_template_projection
from leopard_em.utils.cross_correlation import handle_correlation_mode
from leopard_em.utils.pre_processing import calculate_ctf_filter_stack

# This is assuming the Euler angles are in the ZYZ intrinsic format
# AND that the angles are ordered in (phi, theta, psi)
EULER_ANGLE_FMT = "ZYZ"


def combine_euler_angles(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """Helper function for composing rotations defined by two sets of Euler angles."""
    rotmat_a = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, angle_a, degrees=True, device=angle_a.device
    )
    rotmat_b = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, angle_b, degrees=True, device=angle_b.device
    )
    rotmat_c = roma.rotmat_composition((rotmat_a, rotmat_b))
    euler_angles_c = roma.rotmat_to_euler(EULER_ANGLE_FMT, rotmat_c, degrees=True)

    return euler_angles_c


def core_refine_template(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    euler_angles: torch.Tensor,  # (3, N)
    euler_angle_offsets: torch.Tensor,  # (3, k)
    defocus_offsets: torch.Tensor,  # (l,)
    defocus_u: torch.Tensor,  # (N,)
    defocus_v: torch.Tensor,  # (N,)
    defocus_angle: torch.Tensor,  # (N,)
    pixel_size_offsets: torch.Tensor,  # (m,)
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,  # (N, h, w)
    batch_size: int = 64,
) -> Any:
    """Core function to refine orientations and defoci of a set of particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The Euler angles for each particle in the stack. Shape of (3, N).
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (3, k).
    defocus_u : torch.Tensor
        The defocus along the major axis for each particle in the stack. Shape of (N,).
    defocus_v : torch.Tensor
        The defocus along the minor for each particle in the stack. Shape of (N,).
    defocus_angle : torch.Tensor
        The defocus astigmatism angle for each particle in the stack. Shape of (N,).
        Is the same as the defocus for the micrograph the particle came from.
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    pixel_size_offsets : torch.Tensor
        The pixel size offsets to search over for each particle. Shape of (m,).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.
    """
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    # Send other tensors to the same device
    template_dft = template_dft.to(device)
    euler_angles = euler_angles.to(device)
    defocus_u = defocus_u.to(device)
    defocus_v = defocus_v.to(device)
    defocus_angle = defocus_angle.to(device)
    defocus_offsets = defocus_offsets.to(device)
    pixel_size_offsets = pixel_size_offsets.to(device)
    projective_filters = projective_filters.to(device)
    euler_angle_offsets = euler_angle_offsets.to(device)

    # tqdm progress bar
    pbar_iter = tqdm.tqdm(
        range(num_particles),
        total=num_particles,
        desc=f"Refining {num_particles} particles...",
        leave=True,
    )

    # Iterate over each particle in the stack to get the refined statistics
    refined_statistics = []
    for i in pbar_iter:
        particle_image_dft = particle_stack_dft[i]
        particle_index = i

        refined_stats = _core_refine_template_single_thread(
            particle_image_dft=particle_image_dft,
            particle_index=particle_index,
            template_dft=template_dft,
            euler_angles=euler_angles[i, :],
            euler_angle_offsets=euler_angle_offsets,
            defocus_u=defocus_u[i],
            defocus_v=defocus_v[i],
            defocus_angle=defocus_angle[i],
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            ctf_kwargs=ctf_kwargs,
            projective_filter=projective_filters[i],
            orientation_batch_size=batch_size,
        )
        refined_statistics.append(refined_stats)

    # For each particle, calculate the new best orientation, defocus, and position
    refined_cross_correlation = torch.tensor(
        [stats["max_cc"] for stats in refined_statistics], device=device
    )
    refined_defocus_offset = torch.tensor(
        [stats["refined_defocus_offset"] for stats in refined_statistics],
        device=device,
    )
    refined_pixel_size_offset = torch.tensor(
        [stats["refined_pixel_size_offset"] for stats in refined_statistics],
        device=device,
    )
    refined_pos_y = torch.tensor(
        [stats["refined_pos_y"] for stats in refined_statistics], device=device
    )
    refined_pos_x = torch.tensor(
        [stats["refined_pos_x"] for stats in refined_statistics], device=device
    )

    # Offset refined_pos_{x,y} by the extracted box size
    refined_pos_y -= (H - h + 1) // 2
    refined_pos_x -= (W - w + 1) // 2

    # Compose the previous Euler angles with the refined offsets
    refined_euler_angles = torch.empty((num_particles, 3), device=device)
    for i, stats in enumerate(refined_statistics):
        composed_refined_angle = combine_euler_angles(
            torch.tensor(
                [
                    stats["refined_phi_offset"],
                    stats["refined_theta_offset"],
                    stats["refined_psi_offset"],
                ],
                dtype=euler_angles.dtype,
                device=device,
            ),
            euler_angles[i, :],  # original angle
        )
        refined_euler_angles[i, :] = composed_refined_angle
        # wrap the euler angles back to original ranges,
        # If phi or psi less then 0 add 360

    refined_euler_angles[:, 0] = torch.where(
        refined_euler_angles[:, 0] < 0,
        refined_euler_angles[:, 0] + 360,
        refined_euler_angles[:, 0],
    )
    refined_euler_angles[:, 1] = torch.where(
        refined_euler_angles[:, 1] < 0,
        refined_euler_angles[:, 1] + 180,
        refined_euler_angles[:, 1],
    )
    refined_euler_angles[:, 2] = torch.where(
        refined_euler_angles[:, 2] < 0,
        refined_euler_angles[:, 2] + 360,
        refined_euler_angles[:, 2],
    )

    return {
        "refined_cross_correlation": refined_cross_correlation.cpu(),
        "refined_euler_angles": refined_euler_angles.cpu(),
        "refined_defocus_offset": refined_defocus_offset.cpu(),
        "refined_pixel_size_offset": refined_pixel_size_offset.cpu(),
        "refined_pos_y": refined_pos_y.cpu(),
        "refined_pos_x": refined_pos_x.cpu(),
    }


def _core_refine_template_single_thread(
    particle_image_dft: torch.Tensor,
    particle_index: int,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_u: float,
    defocus_v: float,
    defocus_angle: float,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    ctf_kwargs: dict,
    projective_filter: torch.Tensor,
    orientation_batch_size: int = 32,
) -> dict[str, float | int]:
    """Run the single-threaded core refine template function.

    Parameters
    ----------
    particle_image_dft : torch.Tensor
        The real-Fourier transformed particle image. Shape of (H, W).
    particle_index : int
        The index of the particle in the stack.
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The previous best euler angle for the particle. Shape of (3,).
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (k, 3).
    defocus_u : float
        The defocus along the major axis for the particle.
    defocus_v : float
        The defocus along the minor for the particle.
    defocus_angle : float
        The defocus astigmatism angle for the particle.
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    pixel_size_offsets : torch.Tensor
        The pixel size offsets to search over for each particle. Shape of (m,).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filter : torch.Tensor
        Projective filters to apply to the Fourier slice particle. Shape of (h, w).
    orientation_batch_size : int, optional
        The number of orientations to cross-correlate at once. Default is 32.

    Returns
    -------
    dict[str, float | int]
        The refined statistics for the particle.
    """
    H, W = particle_image_dft.shape
    _, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)
    # valid crop shape
    crop_H = H - h + 1
    crop_W = W - w + 1

    # Output best statistics
    max_cc = -1e9
    refined_phi_offset = 0.0
    refined_theta_offset = 0.0
    refined_psi_offset = 0.0
    refined_defocus_offset = 0.0
    refined_pixel_size_offset = 0.0
    refined_pos_y = 0
    refined_pos_x = 0

    # The "best" Euler angle from the match template program
    default_rot_matrix = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, euler_angles, degrees=True, device=particle_image_dft.device
    )

    default_rot_matrix = default_rot_matrix.to(torch.float32)
    # Calculate the CTF filters with the relative offsets
    ctf_filters = calculate_ctf_filter_stack(
        defocus_u=defocus_u * 1e-4,  # to µm
        defocus_v=defocus_v * 1e-4,  # to µm
        astigmatism_angle=defocus_angle,  # to µm
        defocus_offsets=defocus_offsets * 1e-4,  # to µm
        pixel_size_offsets=pixel_size_offsets,  # to µm
        **ctf_kwargs,
    )

    # Combine the single projective filter with the CTF filter
    combined_projective_filter = projective_filter[None, None, ...] * ctf_filters

    # Iterate over the Euler angle offsets in batches
    num_batches = euler_angle_offsets.shape[0] // orientation_batch_size
    tqdm_iter = tqdm.tqdm(
        range(num_batches),
        total=num_batches,
        desc=f"Refining particle {particle_index}",
        leave=False,
    )
    for i in tqdm_iter:
        start_idx = i * orientation_batch_size
        end_idx = min((i + 1) * orientation_batch_size, euler_angle_offsets.shape[0])
        euler_angle_offsets_batch = euler_angle_offsets[start_idx:end_idx]
        rot_matrix_batch = roma.euler_to_rotmat(
            EULER_ANGLE_FMT,
            euler_angle_offsets_batch,
            degrees=True,
            device=particle_image_dft.device,
        )
        rot_matrix_batch = rot_matrix_batch.to(torch.float32)

        # Rotate the default (best) orientation by the offsets
        rot_matrix_batch = roma.rotmat_composition(
            (rot_matrix_batch, default_rot_matrix)
        )

        # Calculate the cross-correlation
        if particle_image_dft.device.type == "cuda":
            cross_correlation = _do_bached_orientation_cross_correlate(
                image_dft=particle_image_dft,
                template_dft=template_dft,
                rotation_matrices=rot_matrix_batch,
                projective_filters=combined_projective_filter,
            )
        else:
            cross_correlation = _do_bached_orientation_cross_correlate_cpu(
                image_dft=particle_image_dft,
                template_dft=template_dft,
                rotation_matrices=rot_matrix_batch,
                projective_filters=combined_projective_filter,
            )

        cross_correlation = cross_correlation[..., :crop_H, :crop_W]  # valid crop
        # shape xc is (Npx, Ndefoc, Norientations, y, x)
        # Update the best refined statistics (only if max is greater than previous)
        if cross_correlation.max() > max_cc:
            max_cc = cross_correlation.max()

            # Find the maximum value and its indices
            max_values, max_indices = torch.max(
                cross_correlation.view(-1, crop_H, crop_W), dim=0
            )
            # Get the overall maximum value and its position
            max_value, max_pos = torch.max(max_values.view(-1), dim=0)
            y_idx, x_idx = max_pos // crop_W, max_pos % crop_W

            # Calculate the indices for each dimension
            flat_idx = max_indices[y_idx, x_idx]
            px_idx = flat_idx // (len(defocus_offsets) * len(euler_angle_offsets_batch))
            defocus_idx = (flat_idx // len(euler_angle_offsets_batch)) % len(
                defocus_offsets
            )
            angle_idx = flat_idx % len(euler_angle_offsets_batch)

            refined_phi_offset = euler_angle_offsets_batch[angle_idx, 0]
            refined_theta_offset = euler_angle_offsets_batch[angle_idx, 1]
            refined_psi_offset = euler_angle_offsets_batch[angle_idx, 2]
            refined_defocus_offset = defocus_offsets[defocus_idx]
            refined_pixel_size_offset = pixel_size_offsets[px_idx]
            refined_pos_y = y_idx
            refined_pos_x = x_idx

    # Return the refined statistics
    refined_stats = {
        "max_cc": max_cc,
        "refined_phi_offset": refined_phi_offset,
        "refined_theta_offset": refined_theta_offset,
        "refined_psi_offset": refined_psi_offset,
        "refined_defocus_offset": refined_defocus_offset,
        "refined_pixel_size_offset": refined_pixel_size_offset,
        "refined_pos_y": refined_pos_y,
        "refined_pos_x": refined_pos_x,
    }

    return refined_stats


def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
) -> torch.Tensor:
    """Cross-correlate a stack of particle images against a template.

    Here, the argument 'particle_stack_dft' is a set of RFFT-ed particle images with
    necessary filtering already applied. The zeroth dimension corresponds to unique
    particles.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        The stack of particle real-Fourier transformed and un-fftshifted images.
        Shape of (N, H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    rotation_matrices : torch.Tensor
        The orientations of the particles to take the Fourier slices of, as a long
        list of rotation matrices. Shape of (N, 3, 3).
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    mode : Literal["valid", "same"], optional
        Correlation mode to use, by default "valid". If "valid", the output will be
        the valid cross-correlation of the inputs. If "same", the output will be the
        same shape as the input particle stack.
    batch_size : int, optional
        The number of particle images to cross-correlate at once. Default is 1024.
        Larger sizes will consume more memory. If -1, then the entire stack will be
        cross-correlated at once.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used. If "valid", the output will be (N, H-h+1, W-w+1). If "same",
        the output will be (N, H, W).
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    _, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    if batch_size == -1:
        batch_size = num_particles

    if mode == "valid":
        output_shape = (num_particles, H - h + 1, W - w + 1)
    elif mode == "same":
        output_shape = (num_particles, H, W)

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Extract the Fourier slice and apply the projective filters
        fourier_slice = extract_central_slices_rfft_3d(
            volume_rfft=template_dft,
            image_shape=(h,) * 3,
            rotation_matrices=batch_rotation_matrices,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast
        fourier_slice *= batch_projective_filters

        # Inverse Fourier transform and normalize the projection
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection(projections, (h, w), (H, W))

        # Padded forward FFT and cross-correlate
        projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(H, W))
        projections_dft = batch_particles_dft * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Handle the output shape
        cross_correlation = handle_correlation_mode(
            cross_correlation, output_shape, mode
        )

        out_correlation[i : i + batch_size] = cross_correlation

    return out_correlation
