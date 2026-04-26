"""Functions for inspecting local cross-correlations around identified peaks."""

# Kwargs/arity mirror the refine-template distributed API (many explicit tensors).
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=duplicate-code
from collections.abc import Iterator

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
from leopard_em.backend.distributed import run_multiprocess_jobs


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
) -> torch.Tensor:
    """Inspect all local correlations using the refine-template backend path."""
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

    results = run_multiprocess_jobs(
        target=_core_inspect_template_single_gpu,
        kwargs_list=kwargs_per_device,
    )

    for dev in device:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    inspection_stack = torch.cat(
        [torch.from_numpy(r["inspection_stack"]) for r in results.values()]
    )
    particle_indices = torch.cat(
        [torch.from_numpy(r["particle_indices"]) for r in results.values()]
    )
    sort_indices = torch.argsort(particle_indices)

    return inspection_stack[sort_indices]


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
) -> None:
    """Inspect all local correlations on a subset of particles on one device."""
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
    for i in pbar_iter:
        particle_index = int(refine_stack.particle_indices[i])
        stream = streams[i % len(streams)]
        with _device_stream_context(stream):
            inspection_stack = _core_inspect_template_single_thread(
                particle_image_dft=refine_stack.particle_stack_dft[i],
                particle_index=particle_index,
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
                device_id=device_id,
                mag_matrix=refine_stack.mag_matrix,
                apply_projection_normalization=apply_projection_normalization,
            )
            inspection_results.append(inspection_stack)

    _synchronize_device_streams(streams)

    result_dict[device_id] = {
        "inspection_stack": torch.stack(inspection_results).cpu().numpy(),
        "particle_indices": refine_stack.particle_indices.cpu().numpy(),
    }


def _core_inspect_template_single_thread(
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
    apply_projection_normalization: bool = True,
) -> torch.Tensor:
    """Run the non-reducing inspect-template path for one particle."""
    correlation_batches = _iter_refine_particle_correlation_batches(
        particle_image_dft=particle_image_dft,
        particle_index=particle_index,
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
        projective_filter=projective_filter,
        batch_size=batch_size,
        device_id=device_id,
        mag_matrix=mag_matrix,
        apply_projection_normalization=apply_projection_normalization,
        description="Inspecting",
    )

    return _reduce_refine_all(
        correlation_batches=correlation_batches,
        num_orientations=euler_angle_offsets.shape[0],
    )


def _reduce_refine_all(
    correlation_batches: Iterator[
        tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]
    ],
    num_orientations: int,
) -> torch.Tensor:
    """Stitch local correlation batches into one full inspection tensor."""
    output = None
    for start_idx, angle_offsets_batch, cross_correlation, _, crop_h, crop_w in (
        correlation_batches
    ):
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
