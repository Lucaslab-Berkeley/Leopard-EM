"""Backend functions related to correlating and refining particle stacks."""

from multiprocessing import Manager, Process
from typing import Literal

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
from leopard_em.utils.filter_preprocessing import calculate_ctf_filter_stack

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
    euler_angles: torch.Tensor,  # (N, 3)
    euler_angle_offsets: torch.Tensor,  # (k, 3)
    defocus_offsets: torch.Tensor,  # (l,)
    defocus_u: torch.Tensor,  # (N,)
    defocus_v: torch.Tensor,  # (N,)
    defocus_angle: torch.Tensor,  # (N,)
    pixel_size_offsets: torch.Tensor,  # (m,)
    corr_mean: torch.Tensor,  # (N, H - h + 1, W - w + 1)
    corr_std: torch.Tensor,  # (N, H - h + 1, W - w + 1)
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,  # (N, h, w)
    device: torch.device | list[torch.device] = None,
    batch_size: int = 64,
    # TODO: additional arguments for cc --> z-score scaling
) -> dict[str, torch.Tensor]:
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
        The Euler angles for each particle in the stack. Shape of (N, 3).
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (k, 3).
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
    corr_mean : torch.Tensor
        The mean of the cross-correlation values from the full orientation search
        for the pixels around the center of the particle.
        Shape of (H - h + 1, W - w + 1).
    corr_std : torch.Tensor
        The standard deviation of the cross-correlation values from the full
        orientation search for the pixels around the center of the particle.
        Shape of (H - h + 1, W - w + 1).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    device : torch.device | list[torch.device], optional
        Device or list of devices to use for processing.
    batch_size : int, optional
        The number of orientations to process at once. Default is 64.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the refined parameters for all particles.
    """
    # If no device specified, use the device  gpu 0
    if device is None:
        device = [torch.device("cuda:0")]

    # Convert single device to list for consistent handling
    if isinstance(device, torch.device):
        device = [device]

    ###########################################
    ### Split particle stack across devices ###
    ###########################################
    kwargs_per_device = construct_multi_gpu_refine_template_kwargs(
        particle_stack_dft=particle_stack_dft,
        template_dft=template_dft,
        euler_angles=euler_angles,
        euler_angle_offsets=euler_angle_offsets,
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        defocus_angle=defocus_angle,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
        corr_mean=corr_mean,
        corr_std=corr_std,
        ctf_kwargs=ctf_kwargs,
        projective_filters=projective_filters,
        batch_size=batch_size,
        devices=device,
    )

    ##################################################
    ### Initialize and start multiprocessing queue ###
    ##################################################
    manager = Manager()
    result_dict = manager.dict()

    # lists to track processes
    processes = []

    # Start processes
    for i, kwargs in enumerate(kwargs_per_device):
        p = Process(
            target=_core_refine_template_single_gpu,
            args=(result_dict, i),
            kwargs=kwargs,
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Get the results from all processes
    results = []
    for i in range(len(processes)):
        results.append(result_dict[i])

    # Shape information for offset calculations
    _, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    # Concatenate results from all devices
    refined_cross_correlation = torch.cat(
        [torch.from_numpy(r["refined_cross_correlation"]) for r in results]
    )
    refined_z_score = torch.cat(
        [torch.from_numpy(r["refined_z_score"]) for r in results]
    )
    refined_euler_angles = torch.cat(
        [torch.from_numpy(r["refined_euler_angles"]) for r in results]
    )
    refined_defocus_offset = torch.cat(
        [torch.from_numpy(r["refined_defocus_offset"]) for r in results]
    )
    refined_pixel_size_offset = torch.cat(
        [torch.from_numpy(r["refined_pixel_size_offset"]) for r in results]
    )
    refined_pos_y = torch.cat([torch.from_numpy(r["refined_pos_y"]) for r in results])
    refined_pos_x = torch.cat([torch.from_numpy(r["refined_pos_x"]) for r in results])

    # Ensure the results are sorted back to the original particle order
    # (If particles were split across devices, we need to reorder the results)
    particle_indices = torch.cat(
        [torch.from_numpy(r["particle_indices"]) for r in results]
    )
    angle_idx = torch.cat([torch.from_numpy(r["angle_idx"]) for r in results])
    sort_indices = torch.argsort(particle_indices)

    refined_cross_correlation = refined_cross_correlation[sort_indices]
    refined_z_score = refined_z_score[sort_indices]
    refined_euler_angles = refined_euler_angles[sort_indices]
    refined_defocus_offset = refined_defocus_offset[sort_indices]
    refined_pixel_size_offset = refined_pixel_size_offset[sort_indices]
    refined_pos_y = refined_pos_y[sort_indices]
    refined_pos_x = refined_pos_x[sort_indices]
    angle_idx = angle_idx[sort_indices]
    # Offset refined_pos_{x,y} by the extracted box size (same as original)
    refined_pos_y -= (H - h + 1) // 2
    refined_pos_x -= (W - w + 1) // 2

    return {
        "refined_cross_correlation": refined_cross_correlation,
        "refined_z_score": refined_z_score,
        "refined_euler_angles": refined_euler_angles,
        "refined_defocus_offset": refined_defocus_offset,
        "refined_pixel_size_offset": refined_pixel_size_offset,
        "refined_pos_y": refined_pos_y,
        "refined_pos_x": refined_pos_x,
        "angle_idx": angle_idx,
    }


def construct_multi_gpu_refine_template_kwargs(
    particle_stack_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_u: torch.Tensor,
    defocus_v: torch.Tensor,
    defocus_angle: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,
    batch_size: int,
    devices: list[torch.device],
) -> list[dict]:
    """Split particle stack between requested devices.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        Particle stack to split.
    template_dft : torch.Tensor
        Template volume.
    euler_angles : torch.Tensor
        Euler angles for each particle.
    euler_angle_offsets : torch.Tensor
        Euler angle offsets to search over.
    defocus_u : torch.Tensor
        Defocus U values for each particle.
    defocus_v : torch.Tensor
        Defocus V values for each particle.
    defocus_angle : torch.Tensor
        Defocus angle values for each particle.
    defocus_offsets : torch.Tensor
        Defocus offsets to search over.
    pixel_size_offsets : torch.Tensor
        Pixel size offsets to search over.
    corr_mean : torch.Tensor
        Mean of the cross-correlation
    corr_std : torch.Tensor
        Standard deviation of the cross-correlation
    ctf_kwargs : dict
        CTF calculation parameters.
    projective_filters : torch.Tensor
        Projective filters for each particle.
    batch_size : int
        Batch size for orientation processing.
    devices : list[torch.device]
        List of devices to split across.

    Returns
    -------
    list[dict]
        List of dictionaries containing the kwargs to call the single-GPU function.
    """
    num_devices = len(devices)
    kwargs_per_device = []
    num_particles = particle_stack_dft.shape[0]

    # Calculate how many particles to assign to each device
    particles_per_device = [num_particles // num_devices] * num_devices
    # Distribute remaining particles
    for i in range(num_particles % num_devices):
        particles_per_device[i] += 1

    # Split the particle stack across devices
    start_idx = 0
    for device_idx, num_device_particles in enumerate(particles_per_device):
        if num_device_particles == 0:
            continue

        end_idx = start_idx + num_device_particles
        device = devices[device_idx]

        # Get particle indices for this device
        particle_indices = torch.arange(start_idx, end_idx)

        # Split tensors for this device
        device_particle_stack_dft = particle_stack_dft[start_idx:end_idx].to(device)
        device_euler_angles = euler_angles[start_idx:end_idx].to(device)
        device_defocus_u = defocus_u[start_idx:end_idx].to(device)
        device_defocus_v = defocus_v[start_idx:end_idx].to(device)
        device_defocus_angle = defocus_angle[start_idx:end_idx].to(device)
        device_projective_filters = projective_filters[start_idx:end_idx].to(device)

        # These are shared across all particles
        device_template_dft = template_dft.to(device)
        device_euler_angle_offsets = euler_angle_offsets.to(device)
        device_defocus_offsets = defocus_offsets.to(device)
        device_pixel_size_offsets = pixel_size_offsets.to(device)
        device_corr_mean = corr_mean.to(device)
        device_corr_std = corr_std.to(device)

        kwargs = {
            "particle_stack_dft": device_particle_stack_dft,
            "particle_indices": particle_indices.cpu().numpy(),
            "template_dft": device_template_dft,
            "euler_angles": device_euler_angles,
            "euler_angle_offsets": device_euler_angle_offsets,
            "defocus_u": device_defocus_u,
            "defocus_v": device_defocus_v,
            "defocus_angle": device_defocus_angle,
            "defocus_offsets": device_defocus_offsets,
            "pixel_size_offsets": device_pixel_size_offsets,
            "corr_mean": device_corr_mean,
            "corr_std": device_corr_std,
            "ctf_kwargs": ctf_kwargs,
            "projective_filters": device_projective_filters,
            "batch_size": batch_size,
        }

        kwargs_per_device.append(kwargs)
        start_idx = end_idx

    return kwargs_per_device


def _core_refine_template_single_gpu(
    result_dict: dict,
    device_id: int,
    particle_stack_dft: torch.Tensor,
    particle_indices: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_u: torch.Tensor,
    defocus_v: torch.Tensor,
    defocus_angle: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    ctf_kwargs: dict,
    projective_filters: torch.Tensor,
    batch_size: int,
) -> None:
    """Run refine template on a subset of particles on a single GPU.

    Parameters
    ----------
    result_dict : dict
        Dictionary to store results, shared between processes.
    device_id : int
        ID of this device/process.
    particle_stack_dft : torch.Tensor
        Subset of particle stack for this device.
    particle_indices : torch.Tensor
        Original indices of particles in this subset.
    template_dft : torch.Tensor
        Template volume.
    euler_angles : torch.Tensor
        Euler angles for particles in this subset.
    euler_angle_offsets : torch.Tensor
        Euler angle offsets to search over.
    defocus_u : torch.Tensor
        Defocus U values for particles in this subset.
    defocus_v : torch.Tensor
        Defocus V values for particles in this subset.
    defocus_angle : torch.Tensor
        Defocus angle values for particles in this subset.
    defocus_offsets : torch.Tensor
        Defocus offsets to search over.
    pixel_size_offsets : torch.Tensor
        Pixel size offsets to search over.
    corr_mean : torch.Tensor
        Mean of the cross-correlation
    corr_std : torch.Tensor
        Standard deviation of the cross-correlation
    ctf_kwargs : dict
        CTF calculation parameters.
    projective_filters : torch.Tensor
        Projective filters for particles in this subset.
    batch_size : int
        Batch size for orientation processing.
    """
    device = particle_stack_dft.device
    num_particles, H, W = particle_stack_dft.shape
    d, h, w = template_dft.shape
    # account for RFFT
    W = 2 * (W - 1)
    w = 2 * (w - 1)

    # tqdm progress bar
    pbar_iter = tqdm.tqdm(
        range(num_particles),
        total=num_particles,
        desc=f"Refining particles on device {device.index}...",
        leave=True,
        position=device_id,
        dynamic_ncols=True,
    )

    # Iterate over each particle in the stack to get the refined statistics
    refined_statistics = []
    for i in pbar_iter:
        particle_image_dft = particle_stack_dft[i]
        particle_index = int(particle_indices[i])  # Original particle index

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
            corr_mean=corr_mean[i],
            corr_std=corr_std[i],
            projective_filter=projective_filters[i],
            orientation_batch_size=batch_size,
            device_id=device_id,
        )
        refined_statistics.append(refined_stats)

    # For each particle, calculate the new best orientation, defocus, and position
    refined_cross_correlation = torch.tensor(
        [stats["max_cc"] for stats in refined_statistics], device=device
    )
    refined_z_score = torch.tensor(
        [stats["max_z_score"] for stats in refined_statistics], device=device
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
    angle_idx = torch.tensor(
        [stats["angle_idx"] for stats in refined_statistics], device=device
    )

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
        # wrap the euler angles back to original ranges

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

    # Store the results in the shared dict
    result = {
        "refined_cross_correlation": refined_cross_correlation.cpu().numpy(),
        "refined_z_score": refined_z_score.cpu().numpy(),
        "refined_euler_angles": refined_euler_angles.cpu().numpy(),
        "refined_defocus_offset": refined_defocus_offset.cpu().numpy(),
        "refined_pixel_size_offset": refined_pixel_size_offset.cpu().numpy(),
        "refined_pos_y": refined_pos_y.cpu().numpy(),
        "refined_pos_x": refined_pos_x.cpu().numpy(),
        "particle_indices": particle_indices,  # Include original indices for sorting
        "angle_idx": angle_idx.cpu().numpy(),
    }

    result_dict[device_id] = result

    return None


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
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    ctf_kwargs: dict,
    projective_filter: torch.Tensor,
    orientation_batch_size: int = 32,
    device_id: int = 0,
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
    corr_mean : torch.Tensor
        The mean of the cross-correlation values from the full orientation search
        for the pixels around the center of the particle.
    corr_std : torch.Tensor
        The standard deviation of the cross-correlation values from the full
        orientation search for the pixels around the center of the particle.
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filter : torch.Tensor
        Projective filters to apply to the Fourier slice particle. Shape of (h, w).
    orientation_batch_size : int, optional
        The number of orientations to cross-correlate at once. Default is 32.
    device_id : int, optional
        The ID of the device/process. Default is 0.

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
    max_z_score = -1e9
    refined_phi_offset = 0.0
    refined_theta_offset = 0.0
    refined_psi_offset = 0.0
    full_angle_idx = 0
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
        desc=f"Refining particle {particle_index} on device {device_id}",
        leave=False,
        position=device_id + torch.cuda.device_count(),
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

        # Scale cross_correlation to be "z-score"-like
        z_score = (cross_correlation - corr_mean) / corr_std

        # shape xc is (Npx, Ndefoc, Norientations, y, x)
        # Update the best refined statistics (only if max is greater than previous)
        if z_score.max() > max_z_score:
            max_cc = cross_correlation.max()
            max_z_score = z_score.max()

            # Find the maximum value and its indices
            max_values, max_indices = torch.max(z_score.view(-1, crop_H, crop_W), dim=0)
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
            full_angle_idx = angle_idx + start_idx
    # Return the refined statistics
    refined_stats = {
        "max_cc": max_cc,
        "max_z_score": max_z_score,
        "refined_phi_offset": refined_phi_offset,
        "refined_theta_offset": refined_theta_offset,
        "refined_psi_offset": refined_psi_offset,
        "refined_defocus_offset": refined_defocus_offset,
        "refined_pixel_size_offset": refined_pixel_size_offset,
        "refined_pos_y": refined_pos_y,
        "refined_pos_x": refined_pos_x,
        "angle_idx": full_angle_idx,
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
    d, h, w = template_dft.shape
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
