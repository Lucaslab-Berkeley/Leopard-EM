"""Functions for inspecting local cross-correlations around identified peaks."""

# Kwargs/arity mirror the refine-template distributed API (many explicit tensors).
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=duplicate-code
import math
from collections.abc import Iterator
from typing import Literal

import roma
import torch

from leopard_em.backend.core_refine_template import (
    _device_stream_context,
    _iter_refine_particle_correlation_batches,
    _make_device_streams,
    _move_refine_template_stack_to_device,
    _synchronize_device_streams,
    _tqdm_for_refine_particle_loop,
    construct_multi_gpu_refine_template_kwargs,
)
from leopard_em.backend.cross_correlation import do_batched_orientation_frc
from leopard_em.backend.distributed import run_multiprocess_jobs
from leopard_em.backend.utils import EULER_ANGLE_FMT
from leopard_em.utils.ctf_utils import (
    calculate_ctf_filter_stack_full_args,
    move_ctf_kwargs_tensors_to_device,
)


def core_inspect_template(
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
    apply_projection_normalization: bool = True,
    output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Inspect all local hypotheses for each particle.

    Parameters
    ----------
    particle_stack_dft : torch.Tensor
        Particle image stack in RFFT form with shape ``(N, H, W_rfft)``.
    template_dft : torch.Tensor
        Template volume in RFFT form.
    euler_angles : torch.Tensor
        Per-particle base Euler angles with shape ``(N, 3)``.
    euler_angle_offsets : torch.Tensor
        Orientation offsets searched around each base orientation.
    defocus_offsets : torch.Tensor
        Relative defocus offsets searched per particle.
    defocus_u : torch.Tensor
        Per-particle defocus U values.
    defocus_v : torch.Tensor
        Per-particle defocus V values.
    defocus_angle : torch.Tensor
        Per-particle astigmatism angles (degrees).
    pixel_size_offsets : torch.Tensor
        Relative pixel-size offsets searched per particle.
    corr_mean : torch.Tensor
        Per-particle correlation means used for z-score normalization.
    corr_std : torch.Tensor
        Per-particle correlation standard deviations used for z-score normalization.
    ctf_kwargs : dict
        CTF keyword arguments passed to filter construction.
    projective_filters : torch.Tensor
        Per-particle projective filter stack.
    device : torch.device | list[torch.device]
        One or more devices used for distributed execution.
    batch_size : int, optional
        Number of orientation offsets processed per batch.
    num_cuda_streams : int, optional
        CUDA streams per device worker.
    mag_matrix : torch.Tensor | None, optional
        Optional anisotropic magnification matrix.
    apply_projection_normalization : bool, optional
        Whether to normalize each projection before scoring.
    output_mode : Literal["cross_correlation", "frc"], optional
        ``"cross_correlation"`` returns local CC maps.
        ``"frc"`` returns local FRC spectra.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        - ``"cross_correlation"``: ``(N, n_px, n_defocus, n_orient, H, W)``.
        - ``"frc"``: ``(frc_tensor, frequency_bins)`` where
          ``frc_tensor`` is ``(N, n_px, n_defocus, n_orient, n_freq)`` and
          ``frequency_bins`` is ``(n_freq,)``.
    """
    if isinstance(device, torch.device):
        device = [device]

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
    for kwargs in kwargs_per_device:
        kwargs["apply_projection_normalization"] = apply_projection_normalization
        kwargs["output_mode"] = output_mode

    results = run_multiprocess_jobs(
        target=_core_inspect_template_single_gpu,
        kwargs_list=kwargs_per_device,
    )

    for dev in device:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    ordered_results = [results[k] for k in sorted(results.keys(), key=int)]

    if output_mode == "cross_correlation":
        inspection_stack = torch.cat(
            [torch.from_numpy(r["inspection_stack"]) for r in ordered_results]
        )
        particle_indices = torch.cat(
            [torch.from_numpy(r["particle_indices"]) for r in ordered_results]
        )
        sort_indices = torch.argsort(particle_indices)
        return inspection_stack[sort_indices]

    frc_stack = torch.cat([torch.from_numpy(r["frc_stack"]) for r in ordered_results])
    particle_indices = torch.cat(
        [torch.from_numpy(r["particle_indices"]) for r in ordered_results]
    )
    sort_indices = torch.argsort(particle_indices)
    sorted_frc = frc_stack[sort_indices]
    frequency_bins = torch.from_numpy(ordered_results[0]["frequency_bins"])
    return sorted_frc, frequency_bins


def _core_inspect_template_single_gpu(
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
    apply_projection_normalization: bool = True,
    output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
) -> None:
    """Inspect all local hypotheses for one device subset.

    Parameters
    ----------
    result_dict : dict
        Shared multiprocessing dictionary to collect outputs.
    device_id : int
        Worker/device index used as the result key and tqdm position.
    particle_stack_dft : torch.Tensor
        Device-local particle stack chunk in RFFT form.
    particle_indices : torch.Tensor
        Original global particle indices for sorting merged outputs.
    template_dft : torch.Tensor
        Template volume in RFFT form.
    euler_angles : torch.Tensor
        Device-local base Euler angles.
    euler_angle_offsets : torch.Tensor
        Orientation offsets searched around each base orientation.
    defocus_u : torch.Tensor
        Device-local defocus U values.
    defocus_v : torch.Tensor
        Device-local defocus V values.
    defocus_angle : torch.Tensor
        Device-local astigmatism angles.
    defocus_offsets : torch.Tensor
        Relative defocus offsets searched per particle.
    pixel_size_offsets : torch.Tensor
        Relative pixel-size offsets searched per particle.
    corr_mean : torch.Tensor
        Device-local correlation means.
    corr_std : torch.Tensor
        Device-local correlation standard deviations.
    projective_filters : torch.Tensor
        Device-local projective filters.
    ctf_kwargs : dict
        CTF keyword arguments.
    batch_size : int
        Number of orientation offsets processed per batch.
    device : torch.device
        Device for this worker.
    num_cuda_streams : int, optional
        CUDA streams per worker.
    mag_matrix : torch.Tensor | None, optional
        Optional anisotropic magnification matrix.
    apply_projection_normalization : bool, optional
        Whether to normalize each projection before scoring.
    output_mode : Literal["cross_correlation", "frc"], optional
        Score mode for this worker.
    """
    if device.type == "cuda":
        torch.cuda.set_device(device)

    streams = _make_device_streams(device, num_cuda_streams)

    refine_stack = _move_refine_template_stack_to_device(
        device,
        particle_stack_dft,
        particle_indices,
        template_dft,
        euler_angles,
        euler_angle_offsets,
        defocus_u,
        defocus_v,
        defocus_angle,
        defocus_offsets,
        pixel_size_offsets,
        corr_mean,
        corr_std,
        projective_filters,
        mag_matrix,
    )

    num_particles = refine_stack.particle_stack_dft.shape[0]
    pbar_iter = _tqdm_for_refine_particle_loop(
        num_particles, device, device_id, "Inspecting"
    )

    inspection_results = []
    frc_frequency_bins: torch.Tensor | None = None
    for i in pbar_iter:
        stream = streams[i % len(streams)]
        with _device_stream_context(stream):
            inspection_stack = _core_inspect_template_single_thread(
                particle_image_dft=refine_stack.particle_stack_dft[i],
                template_dft=refine_stack.template_dft,
                euler_angles=refine_stack.euler_angles[i, :],
                euler_angle_offsets=refine_stack.euler_angle_offsets,
                defocus_u=refine_stack.defocus_u[i],
                defocus_v=refine_stack.defocus_v[i],
                defocus_angle=refine_stack.defocus_angle[i],
                defocus_offsets=refine_stack.defocus_offsets,
                pixel_size_offsets=refine_stack.pixel_size_offsets,
                corr_mean=refine_stack.corr_mean[i],
                corr_std=refine_stack.corr_std[i],
                ctf_kwargs=ctf_kwargs,
                projective_filter=refine_stack.projective_filters[i],
                batch_size=batch_size,
                mag_matrix=refine_stack.mag_matrix,
                apply_projection_normalization=apply_projection_normalization,
                output_mode=output_mode,
            )
            if output_mode == "cross_correlation":
                inspection_results.append(inspection_stack)
            else:
                frc_tensor, frequency_bins = inspection_stack
                inspection_results.append(frc_tensor)
                if frc_frequency_bins is None:
                    frc_frequency_bins = frequency_bins

    _synchronize_device_streams(streams)

    if output_mode == "cross_correlation":
        result_dict[device_id] = {
            "inspection_stack": torch.stack(inspection_results).cpu().numpy(),
            "particle_indices": refine_stack.particle_indices.cpu().numpy(),
        }
    else:
        if frc_frequency_bins is None:
            raise ValueError("No FRC frequencies were generated.")
        result_dict[device_id] = {
            "frc_stack": torch.stack(inspection_results).cpu().numpy(),
            "frequency_bins": frc_frequency_bins.cpu().numpy(),
            "particle_indices": refine_stack.particle_indices.cpu().numpy(),
        }


def _core_inspect_template_single_thread(
    particle_image_dft: torch.Tensor,
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
    mag_matrix: torch.Tensor | None = None,
    apply_projection_normalization: bool = True,
    output_mode: Literal["cross_correlation", "frc"] = "cross_correlation",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run inspect scoring for one particle without best-value reduction.

    Parameters
    ----------
    particle_image_dft : torch.Tensor
        Single particle image in RFFT form.
    template_dft : torch.Tensor
        Template volume in RFFT form.
    euler_angles : torch.Tensor
        Base Euler angle for this particle.
    euler_angle_offsets : torch.Tensor
        Orientation offsets searched around the base orientation.
    defocus_u : float
        Particle defocus U.
    defocus_v : float
        Particle defocus V.
    defocus_angle : float
        Particle astigmatism angle.
    defocus_offsets : torch.Tensor
        Relative defocus offsets searched per particle.
    pixel_size_offsets : torch.Tensor
        Relative pixel-size offsets searched per particle.
    corr_mean : torch.Tensor
        Per-particle correlation mean map used for z-score normalization.
    corr_std : torch.Tensor
        Per-particle correlation std map used for z-score normalization.
    ctf_kwargs : dict
        CTF keyword arguments.
    projective_filter : torch.Tensor
        Particle-specific projective filter.
    batch_size : int, optional
        Number of orientation offsets processed per batch.
    mag_matrix : torch.Tensor | None, optional
        Optional anisotropic magnification matrix.
    apply_projection_normalization : bool, optional
        Whether to normalize each projection before scoring.
    output_mode : Literal["cross_correlation", "frc"], optional
        Score mode (CC map or FRC spectrum).

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        - ``"cross_correlation"``: tensor ``(n_px, n_defocus, n_orient, H, W)``.
        - ``"frc"``: tuple ``(frc_tensor, frequency_bins)`` where
          ``frc_tensor`` is ``(n_px, n_defocus, n_orient, n_freq)``.
    """
    # TODO: corr_mean/corr_std are documented for z-score normalization but are
    # currently unused here; raw (non-normalized) scores are returned.
    _ = corr_mean
    _ = corr_std
    if output_mode == "frc":
        frc_batches = _iter_refine_particle_frc_batches(
            particle_image_dft=particle_image_dft,
            template_dft=template_dft,
            euler_angles=euler_angles,
            euler_angle_offsets=euler_angle_offsets,
            defocus_u=defocus_u,
            defocus_v=defocus_v,
            defocus_angle=defocus_angle,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_size_offsets,
            ctf_kwargs=ctf_kwargs,
            projective_filter=projective_filter,
            batch_size=batch_size,
            mag_matrix=mag_matrix,
            apply_projection_normalization=apply_projection_normalization,
        )
        return _reduce_refine_all_frc(
            frc_batches=frc_batches,
            num_orientations=euler_angle_offsets.shape[0],
        )

    correlation_batches = _iter_refine_particle_correlation_batches(
        particle_image_dft=particle_image_dft,
        template_dft=template_dft,
        euler_angles=euler_angles,
        euler_angle_offsets=euler_angle_offsets,
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        defocus_angle=defocus_angle,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
        ctf_kwargs=ctf_kwargs,
        projective_filter=projective_filter,
        batch_size=batch_size,
        mag_matrix=mag_matrix,
        apply_projection_normalization=apply_projection_normalization,
    )

    return _reduce_refine_all(
        correlation_batches=correlation_batches,
        num_orientations=euler_angle_offsets.shape[0],
    )


def _iter_refine_particle_frc_batches(
    particle_image_dft: torch.Tensor,
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
    batch_size: int = 32,
    mag_matrix: torch.Tensor | None = None,
    apply_projection_normalization: bool = True,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield FRC batches for a single particle over orientation offsets.

    Parameters
    ----------
    particle_image_dft : torch.Tensor
        Single particle image in RFFT form.
    particle_index : int
        Original global particle index (for progress labeling only).
    template_dft : torch.Tensor
        Template volume in RFFT form.
    euler_angles : torch.Tensor
        Base Euler angle for this particle.
    euler_angle_offsets : torch.Tensor
        Orientation offsets searched around the base orientation.
    defocus_u : float
        Particle defocus U.
    defocus_v : float
        Particle defocus V.
    defocus_angle : float
        Particle astigmatism angle.
    defocus_offsets : torch.Tensor
        Relative defocus offsets searched per particle.
    pixel_size_offsets : torch.Tensor
        Relative pixel-size offsets searched per particle.
    ctf_kwargs : dict
        CTF keyword arguments.
    projective_filter : torch.Tensor
        Particle-specific projective filter.
    batch_size : int, optional
        Number of orientation offsets processed per batch.
    device_id : int, optional
        Worker/device index for tqdm positioning.
    mag_matrix : torch.Tensor | None, optional
        Optional anisotropic magnification matrix.
    apply_projection_normalization : bool, optional
        Whether to normalize each projection before scoring.

    Yields
    ------
    tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(start_idx, angle_offsets_batch, frc_values, frequency_bins)`` where
        ``frc_values`` has shape ``(n_px, n_defocus, n_orient_batch, n_freq)``.
    """
    default_rot_matrix = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, euler_angles, degrees=True, device=particle_image_dft.device
    )
    default_rot_matrix = default_rot_matrix.to(torch.float32)

    ctf_dev_kwargs = move_ctf_kwargs_tensors_to_device(
        ctf_kwargs, particle_image_dft.device
    )
    ctf_filters = calculate_ctf_filter_stack_full_args(
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        astigmatism_angle=defocus_angle,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
        **ctf_dev_kwargs,
    )
    combined_projective_filter = projective_filter[None, None, ...] * ctf_filters

    num_batches = math.ceil(euler_angle_offsets.shape[0] / batch_size)
    for i in range(num_batches):
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
        rot_matrix_batch = roma.rotmat_composition(
            (rot_matrix_batch, default_rot_matrix)
        )

        frc_values, frequency_bins = do_batched_orientation_frc(
            image_dft=particle_image_dft,
            template_dft=template_dft,
            rotation_matrices=rot_matrix_batch,
            projective_filters=combined_projective_filter,
            apply_normalization=apply_projection_normalization,
            mag_matrix=mag_matrix,
        )
        yield (
            start_idx,
            euler_angle_offsets_batch,
            frc_values,
            frequency_bins,
        )


def _reduce_refine_all_frc(
    frc_batches: Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
    num_orientations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch batched FRC results into a full orientation tensor.

    Parameters
    ----------
    frc_batches : Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]
        Iterator from :func:`_iter_refine_particle_frc_batches`.
    num_orientations : int
        Total number of orientation offsets across all batches.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(frc_tensor, frequency_bins)`` where
        ``frc_tensor`` is ``(n_px, n_defocus, n_orient, n_freq)`` and
        ``frequency_bins`` is ``(n_freq,)``.
    """
    output = None
    frequency_bins = None
    for start_idx, angle_offsets_batch, frc_values, batch_frequency_bins in frc_batches:
        if output is None:
            output = torch.empty(
                (
                    frc_values.shape[0],
                    frc_values.shape[1],
                    num_orientations,
                    frc_values.shape[-1],
                ),
                dtype=frc_values.dtype,
                device=frc_values.device,
            )
            frequency_bins = batch_frequency_bins
        end_idx = start_idx + len(angle_offsets_batch)
        output[:, :, start_idx:end_idx] = frc_values

    if output is None or frequency_bins is None:
        raise ValueError("No FRC batches were generated.")

    return output, frequency_bins


def _reduce_refine_all(
    correlation_batches: Iterator[tuple[int, torch.Tensor, torch.Tensor, int, int]],
    num_orientations: int,
) -> torch.Tensor:
    """Stitch local CC batches into a full orientation tensor.

    Parameters
    ----------
    correlation_batches : Iterator[tuple[int, torch.Tensor, torch.Tensor, int, int]]
        Iterator from :func:`_iter_refine_particle_correlation_batches`.
    num_orientations : int
        Total number of orientation offsets across all batches.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(n_px, n_defocus, n_orient, H, W)``.
    """
    output = None
    for (
        start_idx,
        angle_offsets_batch,
        cross_correlation,
        crop_h,
        crop_w,
    ) in correlation_batches:
        if output is None:
            output = torch.empty(
                (
                    cross_correlation.shape[0],
                    cross_correlation.shape[1],
                    num_orientations,
                    crop_h,
                    crop_w,
                ),
                dtype=cross_correlation.dtype,
                device=cross_correlation.device,
            )
        end_idx = start_idx + len(angle_offsets_batch)
        output[:, :, start_idx:end_idx] = cross_correlation

    if output is None:
        raise ValueError("No orientation batches were generated.")

    return output
