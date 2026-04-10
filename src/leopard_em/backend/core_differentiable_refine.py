"""Backend functions related to correlating and refining particle stacks."""

# Following pylint error ignored because torc.fft.* is not recognized as callable
# pylint: disable=E1102
# pylint: disable=duplicate-code

import math

import roma
import torch
import tqdm

from leopard_em.backend.core_refine_template import (
    construct_multi_gpu_refine_template_kwargs,
)
from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate,
)
from leopard_em.backend.utils import EULER_ANGLE_FMT, combine_euler_angles
from leopard_em.utils.ctf_utils import calculate_ctf_filter_stack_full_args


# NOTE: Disabling pylint for too many arguments because we are taking a data-oriented
# approach where each argument is independent and explicit.
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
def core_differentiable_refine(
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
    device: torch.device | list[torch.device],
    batch_size: int = 32,
    num_cuda_streams: int = 1,
    mag_matrix: torch.Tensor | None = None,
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
    device : torch.device | list[torch.device]
        Device or list of devices to use for processing.
    batch_size : int, optional
        The number of cross-correlations to process in one batch, defaults to 32.
    num_cuda_streams : int, optional
        Number of CUDA streams to use for parallel processing. Defaults to 1.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.

    Returns
    -------
    dict[str, torch.Tensor]
        Tensor containing the refined parameters for all particles.
    """
    # Convert single device to list for consistent handling
    if isinstance(device, torch.device):
        device = [device]

    # Check that all devices are GPU devices (CUDA)
    # Differentiable refinement requires GPU for gradient computation
    for dev in device:
        if dev.type != "cuda":
            raise ValueError(
                f"Differentiable refinement can only be run on GPU devices. "
                f"Got device type: {dev.type}. Please use a CUDA device."
            )

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
        num_cuda_streams=num_cuda_streams,
        mag_matrix=mag_matrix,
    )

    results = {}
    for device_id, kwargs in enumerate(kwargs_per_device):
        result_dict: dict[int, dict[str, torch.Tensor]] = {}
        _core_refine_template_single_gpu(
            result_dict=result_dict, device_id=device_id, **kwargs
        )
        # Extract the actual result that was stored at result_dict[device_id]
        results[device_id] = result_dict[device_id]

    # Shape information for offset calculations
    _, img_h, img_w = particle_stack_dft.shape
    _, template_h, template_w = template_dft.shape
    # account for RFFT
    img_w = 2 * (img_w - 1)
    template_w = 2 * (template_w - 1)

    # Concatenate results from all devices
    refined_cross_correlation = torch.cat(
        [r["refined_cross_correlation"] for r in results.values()]
    )
    refined_z_score = torch.cat([r["refined_z_score"] for r in results.values()])
    refined_euler_angles = torch.cat(
        [r["refined_euler_angles"] for r in results.values()]
    )
    refined_defocus_offset = torch.cat(
        [r["refined_defocus_offset"] for r in results.values()]
    )
    refined_pixel_size_offset = torch.cat(
        [r["refined_pixel_size_offset"] for r in results.values()]
    )
    refined_pos_y = torch.cat([r["refined_pos_y"] for r in results.values()])
    refined_pos_x = torch.cat([r["refined_pos_x"] for r in results.values()])

    # Ensure the results are sorted back to the original particle order
    # (If particles were split across devices, we need to reorder the results)
    particle_indices = torch.cat([r["particle_indices"] for r in results.values()])
    angle_idx = torch.cat([r["angle_idx"] for r in results.values()])
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
    refined_pos_y -= (img_h - template_h + 1) // 2
    refined_pos_x -= (img_w - template_w + 1) // 2

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


# pylint: disable=too-many-locals, too-many-statements
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
    projective_filters: torch.Tensor,
    ctf_kwargs: dict,
    batch_size: int,
    device: torch.device,
    num_cuda_streams: int = 1,
    mag_matrix: torch.Tensor | None = None,
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
    projective_filters : torch.Tensor
        Projective filters for particles in this subset.
    ctf_kwargs : dict
        CTF calculation parameters.
    batch_size : int
        Batch size for orientation processing.
    device : torch.device
        Torch device to run this process on.
    num_cuda_streams : int, optional
        Number of CUDA streams to use for parallel processing. Defaults to 1.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.
    """
    streams = [torch.cuda.Stream(device=device) for _ in range(num_cuda_streams)]

    ######################################
    ### Send all tensors to the device ###
    ######################################

    particle_stack_dft = particle_stack_dft.to(device)
    particle_indices = particle_indices.to(device)
    template_dft = template_dft.to(device)
    euler_angles = euler_angles.to(device)
    euler_angle_offsets = euler_angle_offsets.to(device)
    defocus_u = defocus_u.to(device)
    defocus_v = defocus_v.to(device)
    defocus_angle = defocus_angle.to(device)
    defocus_offsets = defocus_offsets.to(device)
    pixel_size_offsets = pixel_size_offsets.to(device)
    corr_mean = corr_mean.to(device)
    corr_std = corr_std.to(device)
    projective_filters = projective_filters.to(device)
    if mag_matrix is not None:
        mag_matrix = mag_matrix.to(device)

    ########################################
    ### Setup constants and progress bar ###
    ########################################

    num_particles, _, img_w = particle_stack_dft.shape
    _, _, template_w = template_dft.shape
    # account for RFFT
    img_w = 2 * (img_w - 1)
    template_w = 2 * (template_w - 1)

    # tqdm progress bar
    pbar_iter = tqdm.tqdm(
        range(num_particles),
        total=num_particles,
        desc=f"Refining particles on device {device.index}...",
        leave=True,
        position=device_id,
        dynamic_ncols=True,
        unit="particle",
        smoothing=0.1,
    )

    #############################################################################
    ### Iterate over each particle in the stack to get the refined statistics ###
    #############################################################################

    refined_statistics = []
    for i in pbar_iter:
        particle_image_dft = particle_stack_dft[i]
        particle_index = int(particle_indices[i])  # Original particle index

        # Distribute different particles across streams
        stream = streams[i % num_cuda_streams]
        with torch.cuda.stream(stream):
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
                batch_size=batch_size,
                device_id=device_id,
                mag_matrix=mag_matrix,
            )
            refined_statistics.append(refined_stats)

    # Wait for all streams to finish
    for stream in streams:
        stream.synchronize()

    # For each particle, calculate the new best orientation, defocus, and position
    # Stack tensors directly to preserve gradients (they're already tensors from
    # _core_refine_template_single_thread)
    refined_cross_correlation = torch.stack(
        [stats["max_cc"] for stats in refined_statistics]
    )
    refined_z_score = torch.stack(
        [stats["max_z_score"] for stats in refined_statistics]
    )
    refined_defocus_offset = torch.stack(
        [stats["refined_defocus_offset"] for stats in refined_statistics]
    )
    refined_pixel_size_offset = torch.stack(
        [stats["refined_pixel_size_offset"] for stats in refined_statistics]
    )
    refined_pos_y = torch.stack(
        [stats["refined_pos_y"] for stats in refined_statistics]
    )
    refined_pos_x = torch.stack(
        [stats["refined_pos_x"] for stats in refined_statistics]
    )
    angle_idx = torch.tensor(
        [stats["angle_idx"] for stats in refined_statistics],
        device=device,
        dtype=torch.long,
    )

    # Compose the previous Euler angles with the refined offsets
    # Stack the offset tensors directly to preserve gradients
    refined_phi_offsets = torch.stack(
        [stats["refined_phi_offset"] for stats in refined_statistics]
    )
    refined_theta_offsets = torch.stack(
        [stats["refined_theta_offset"] for stats in refined_statistics]
    )
    refined_psi_offsets = torch.stack(
        [stats["refined_psi_offset"] for stats in refined_statistics]
    )
    refined_angle_offsets = torch.stack(
        [refined_phi_offsets, refined_theta_offsets, refined_psi_offsets], dim=1
    )

    # Compose angles - need to do this per particle since combine_euler_angles
    # works on single angle pairs
    refined_euler_angles = torch.empty(
        (num_particles, 3), device=device, dtype=euler_angles.dtype
    )
    for i in range(num_particles):
        composed_refined_angle = combine_euler_angles(
            refined_angle_offsets[i, :],
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
        "refined_cross_correlation": refined_cross_correlation,
        "refined_z_score": refined_z_score,
        "refined_euler_angles": refined_euler_angles,
        "refined_defocus_offset": refined_defocus_offset,
        "refined_pixel_size_offset": refined_pixel_size_offset,
        "refined_pos_y": refined_pos_y,
        "refined_pos_x": refined_pos_x,
        "particle_indices": particle_indices,  # Original idxs for sorting
        "angle_idx": angle_idx,
    }

    result_dict[device_id] = result


# pylint: disable=too-many-locals, too-many-statements
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
    batch_size: int = 32,
    device_id: int = 0,
    mag_matrix: torch.Tensor | None = None,
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
    batch_size : int, optional
        The number of orientations to cross-correlate at once. Default is 32.
    device_id : int, optional
        The ID of the device/process. Default is 0.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.

    Returns
    -------
    dict[str, torch.Tensor | int]
        The refined statistics for the particle. Most values are tensors to preserve
        gradients, except angle_idx which is an integer index.
    """
    img_h, img_w = particle_image_dft.shape
    _, template_h, template_w = template_dft.shape
    # account for RFFT
    img_w = 2 * (img_w - 1)
    template_w = 2 * (template_w - 1)
    # valid crop shape
    crop_h = img_h - template_h + 1
    crop_w = img_w - template_w + 1

    # Output best statistics - use tensors to preserve gradients
    device = particle_image_dft.device
    # Use float32 for max_cc and max_z_score since cross_correlation
    # and z_score are real
    max_cc = torch.tensor(-1e9, device=device, dtype=torch.float32)
    max_z_score = torch.tensor(-1e9, device=device, dtype=torch.float32)
    refined_phi_offset = torch.tensor(
        0.0, device=device, dtype=euler_angle_offsets.dtype
    )
    refined_theta_offset = torch.tensor(
        0.0, device=device, dtype=euler_angle_offsets.dtype
    )
    refined_psi_offset = torch.tensor(
        0.0, device=device, dtype=euler_angle_offsets.dtype
    )
    full_angle_idx = torch.tensor(0, device=device, dtype=torch.long)
    refined_defocus_offset = torch.tensor(
        0.0, device=device, dtype=defocus_offsets.dtype
    )
    refined_pixel_size_offset = torch.tensor(
        0.0, device=device, dtype=pixel_size_offsets.dtype
    )
    refined_pos_y = torch.tensor(0, device=device, dtype=torch.long)
    refined_pos_x = torch.tensor(0, device=device, dtype=torch.long)

    # The "best" Euler angle from the match template program
    default_rot_matrix = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, euler_angles, degrees=True, device=particle_image_dft.device
    )

    default_rot_matrix = default_rot_matrix.to(torch.float32)
    # Calculate the CTF filters with the relative offsets
    ctf_filters = calculate_ctf_filter_stack_full_args(
        defocus_u=defocus_u,  # in Angstrom
        defocus_v=defocus_v,  # in Angstrom
        astigmatism_angle=defocus_angle,  # in degrees
        defocus_offsets=defocus_offsets,  # in Angstrom
        pixel_size_offsets=pixel_size_offsets,  # in Angstrom
        **ctf_kwargs,
    )

    # Combine the single projective filter with the CTF filter
    combined_projective_filter = projective_filter[None, None, ...] * ctf_filters

    # Iterate over the Euler angle offsets in batches
    # The tqdm iterator is over batches, but we want to report cross-correlations/sec.
    # We therefore scale by the number of cross-correlations per batch.
    num_batches = math.ceil(euler_angle_offsets.shape[0] / batch_size)
    cross_corr_per_batch = len(defocus_offsets) * len(pixel_size_offsets) * batch_size

    tqdm_iter = tqdm.tqdm(
        range(num_batches),
        total=num_batches,
        desc=f"Refining particle {particle_index} on device {device_id}",
        leave=False,
        position=device_id + torch.cuda.device_count(),
        unit="corr",
        unit_scale=cross_corr_per_batch,
    )

    for i in tqdm_iter:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, euler_angle_offsets.shape[0])
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
        cross_correlation = do_batched_orientation_cross_correlate(
            image_dft=particle_image_dft,
            template_dft=template_dft,
            rotation_matrices=rot_matrix_batch,
            projective_filters=combined_projective_filter,
            requires_grad=True,
            mag_matrix=mag_matrix,
        )

        cross_correlation = cross_correlation[..., :crop_h, :crop_w]  # valid crop
        # Scale cross_correlation to be "z-score"-like
        z_score = (cross_correlation - corr_mean) / corr_std
        # shape xc is (num_Cs, num_defocus, num_orientations, y, x)
        # where num_Cs is the number of different pixel size offsets,
        # num_defocus is the number of defocus offsets,
        # and num_orientations is the number of Euler angle offsets.
        # Update the best refined statistics (only if max is greater than previous)
        # Keep as tensors to preserve gradients
        current_max_z_score = z_score.max()
        # Compare tensor values (use .item() only for comparison,
        # keep tensor for storage)

        # Find the maximum value and its indices
        max_values, max_indices = torch.max(z_score.view(-1, crop_h, crop_w), dim=0)

        # Get the overall maximum value and its position
        _, max_pos = torch.max(max_values.view(-1), dim=0)
        y_idx = max_pos // crop_w
        x_idx = max_pos % crop_w

        # Calculate the indices for each dimension
        flat_idx = max_indices[y_idx, x_idx]
        px_idx = flat_idx // (len(defocus_offsets) * len(euler_angle_offsets_batch))
        defocus_idx = (flat_idx // len(euler_angle_offsets_batch)) % len(
            defocus_offsets
        )
        angle_idx = flat_idx % len(euler_angle_offsets_batch)

        # Get the specific cross_correlation for the best configuration
        best_cross_correlation = cross_correlation[px_idx, defocus_idx, angle_idx]
        # Get the cross-correlation value at the max z-score pixel location
        current_max_cc = best_cross_correlation[y_idx, x_idx]

        # Use torch.where to preserve gradients through both branches
        is_better = current_max_z_score > max_z_score
        max_z_score = torch.where(is_better, current_max_z_score, max_z_score)
        max_cc = torch.where(is_better, current_max_cc, max_cc)

        # Keep as tensors to preserve gradients
        refined_phi_offset = torch.where(
            is_better, euler_angle_offsets_batch[angle_idx, 0], refined_phi_offset
        )
        refined_theta_offset = torch.where(
            is_better, euler_angle_offsets_batch[angle_idx, 1], refined_theta_offset
        )
        refined_psi_offset = torch.where(
            is_better, euler_angle_offsets_batch[angle_idx, 2], refined_psi_offset
        )
        refined_defocus_offset = torch.where(
            is_better, defocus_offsets[defocus_idx], refined_defocus_offset
        )
        refined_pixel_size_offset = torch.where(
            is_better, pixel_size_offsets[px_idx], refined_pixel_size_offset
        )
        refined_pos_y = torch.where(is_better, y_idx.to(torch.long), refined_pos_y)
        refined_pos_x = torch.where(is_better, x_idx.to(torch.long), refined_pos_x)
        full_angle_idx = torch.where(is_better, angle_idx + start_idx, full_angle_idx)

    refined_phi_offset = refined_phi_offset.detach()
    refined_theta_offset = refined_theta_offset.detach()
    refined_psi_offset = refined_psi_offset.detach()
    refined_defocus_offset = refined_defocus_offset.detach()
    refined_pixel_size_offset = refined_pixel_size_offset.detach()
    refined_pos_y = refined_pos_y.detach()
    refined_pos_x = refined_pos_x.detach()
    full_angle_idx = full_angle_idx.detach()

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
