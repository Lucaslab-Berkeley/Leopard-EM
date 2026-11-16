"""Backend functions related to correlating and refining particle stacks."""

# Following pylint error ignored because torc.fft.* is not recognized as callable
# pylint: disable=E1102,too-many-lines,duplicate-code

import math
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, NamedTuple, cast

import roma
import torch
import tqdm

from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate,
    do_batched_orientation_cross_correlate_cpu,
)
from leopard_em.backend.distributed import run_multiprocess_jobs
from leopard_em.backend.refine_gemm import (
    SliceOperandFn,
    gemm_search_particle,
    prepare_slices_for_scoring,
    reference_slice_operands,
    score_hypotheses,
    slices_to_operands,
)
from leopard_em.backend.slice_kernels import (
    TritonSliceExtractor,
    can_use_triton_slices,
)
from leopard_em.backend.utils import (
    EULER_ANGLE_FMT,
    combine_euler_angles,
    normalize_template_projection,
)
from leopard_em.utils.cross_correlation import handle_correlation_mode
from leopard_em.utils.ctf_utils import (
    calculate_ctf_filter_stack_full_args,
    move_ctf_kwargs_tensors_to_device,
)
from leopard_em.utils.fourier_slice import (
    extract_template_fourier_slices,
    projection_shape_from_filters,
)


def _make_device_streams(
    device: torch.device, num_cuda_streams: int
) -> list[torch.cuda.Stream | None]:
    """Create CUDA streams when running on CUDA, otherwise a CPU placeholder."""
    if device.type == "cuda":
        return [torch.cuda.Stream(device=device) for _ in range(num_cuda_streams)]
    return [None]


def _synchronize_device_streams(streams: list[torch.cuda.Stream | None]) -> None:
    """Synchronize CUDA streams and no-op for CPU placeholders."""
    for stream in streams:
        if stream is not None:
            stream.synchronize()


class _RefineTemplateStackOnDevice(NamedTuple):
    """All per-device tensors for one refine- or inspect-template GPU chunk."""

    particle_stack_dft: torch.Tensor
    particle_indices: torch.Tensor
    template_dft: torch.Tensor
    euler_angles: torch.Tensor
    euler_angle_offsets: torch.Tensor
    defocus_u: torch.Tensor
    defocus_v: torch.Tensor
    defocus_angle: torch.Tensor
    defocus_offsets: torch.Tensor
    pixel_size_offsets: torch.Tensor
    corr_mean: torch.Tensor
    corr_std: torch.Tensor
    projective_filters: torch.Tensor
    mag_matrix: torch.Tensor | None


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _move_refine_template_stack_to_device(
    device: torch.device,
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
    mag_matrix: torch.Tensor | None,
) -> _RefineTemplateStackOnDevice:
    """Move stack, template, and search grids to ``device`` (refine and inspect)."""
    return _RefineTemplateStackOnDevice(
        particle_stack_dft=particle_stack_dft.to(device),
        particle_indices=particle_indices.to(device),
        template_dft=template_dft.to(device),
        euler_angles=euler_angles.to(device),
        euler_angle_offsets=euler_angle_offsets.to(device),
        defocus_u=defocus_u.to(device),
        defocus_v=defocus_v.to(device),
        defocus_angle=defocus_angle.to(device),
        defocus_offsets=defocus_offsets.to(device),
        pixel_size_offsets=pixel_size_offsets.to(device),
        corr_mean=corr_mean.to(device),
        corr_std=corr_std.to(device),
        projective_filters=projective_filters.to(device),
        mag_matrix=mag_matrix.to(device) if mag_matrix is not None else None,
    )


def _tqdm_for_refine_particle_loop(
    num_particles: int,
    device: torch.device,
    device_id: int,
    desc_verb: str,
) -> tqdm.tqdm:
    """Progress bar over particle index for a single device (refine or inspect)."""
    return tqdm.tqdm(
        range(num_particles),
        total=num_particles,
        desc=f"{desc_verb} particles on device {device.index}...",
        leave=True,
        position=device_id,
        dynamic_ncols=True,
        unit="particle",
        smoothing=0.1,
    )


RefineBackend = Literal["gemm", "fft"]

# Default half-width (in pixels) of the shift window searched by the GEMM backend
DEFAULT_SHIFT_SEARCH_RADIUS = 2

# Result keys returned by the per-device workers for internal use only
_INTERNAL_RESULT_KEYS = ("particle_indices", "defocus_idx", "pixel_size_idx")


# NOTE: Disabling pylint for too many arguments because we are taking a data-oriented
# approach where each argument is independent and explicit.
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
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
    device: torch.device | list[torch.device],
    batch_size: int = 32,
    num_cuda_streams: int = 1,
    mag_matrix: torch.Tensor | None = None,
    *,
    backend: RefineBackend = "gemm",
    shift_search_radius: int | None = DEFAULT_SHIFT_SEARCH_RADIUS,
    fft_centering: bool = True,
) -> dict[str, torch.Tensor]:
    """Refine the orientation, defocus, pixel size, and position of particles.

    For each particle, every combination of orientation offset, defocus offset, pixel
    size offset, and integer position in the valid correlation crop is scored, and the
    hypothesis with the largest z-score is returned.

    Two backends compute the same correlations:

    - ``"gemm"`` (default): scores the hypotheses with matrix multiplications over
      Fourier coefficients (see :mod:`leopard_em.backend.refine_gemm`), only at the
      positions inside a ``(2r + 1) x (2r + 1)`` window. The window starts at the peak
      of one FFT correlation at the prior orientation (``fft_centering``) and is
      re-centered if the best position lands on its edge. On CUDA, slices are taken
      with a Triton kernel when possible.
    - ``"fft"``: correlates every (orientation, defocus) projection with the whole
      particle image by FFT and searches the whole valid crop.

    Differentiability
    -----------------
    The search itself (an argmax) runs without gradients. When grad mode is enabled
    and any input tensor requires grad, the returned ``refined_cross_correlation`` and
    ``refined_z_score`` are re-computed at the winning hypothesis with differentiable
    operations, so gradients flow to the template, particle images, filters, CTF and
    angle inputs. Note that importing ``leopard_em.backend`` disables grad mode
    globally; use ``with torch.enable_grad():`` around the call.

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
    defocus_offsets : torch.Tensor
        The defocus offsets to search over for each particle. Shape of (l,).
    defocus_u : torch.Tensor
        The defocus along the major axis for each particle in the stack. Shape of (N,).
    defocus_v : torch.Tensor
        The defocus along the minor for each particle in the stack. Shape of (N,).
    defocus_angle : torch.Tensor
        The defocus astigmatism angle for each particle in the stack. Shape of (N,).
        Is the same as the defocus for the micrograph the particle came from.
    pixel_size_offsets : torch.Tensor
        The pixel size offsets to search over for each particle. Shape of (m,).
    corr_mean : torch.Tensor
        The mean of the cross-correlation values from the full orientation search
        for the pixels around the center of the particle.
        Shape of (N, H - h + 1, W - w + 1).
    corr_std : torch.Tensor
        The standard deviation of the cross-correlation values from the full
        orientation search for the pixels around the center of the particle.
        Shape of (N, H - h + 1, W - w + 1).
    ctf_kwargs : dict
        Keyword arguments to pass to the CTF calculation function.
    projective_filters : torch.Tensor
        Projective filters to apply to each Fourier slice particle. Shape of (N, h, w).
    device : torch.device | list[torch.device]
        Device or list of devices to use for processing.
    batch_size : int, optional
        The number of orientations to process in one batch, defaults to 32.
    num_cuda_streams : int, optional
        Number of CUDA streams to use for parallel processing (``"fft"`` backend
        only). Defaults to 1.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.
    backend : Literal["gemm", "fft"], optional
        Which backend scores the hypotheses, see above. Default is ``"gemm"``.
    shift_search_radius : int | None, optional
        Half-width ``r`` in pixels of the position window searched by the ``"gemm"``
        backend. None searches the whole valid crop (the ``"fft"`` behaviour).
        Ignored by the ``"fft"`` backend. Default is 2.
    fft_centering : bool, optional
        If True, the ``"gemm"`` window starts at the peak of one FFT correlation at
        the prior orientation, defocus, and pixel size. If False, it starts at the
        center of the valid crop. Ignored by the ``"fft"`` backend. Default is True.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the refined parameters for all particles (on CPU):
        ``refined_cross_correlation`` (the correlation at the best hypothesis),
        ``refined_z_score``, ``refined_euler_angles``, ``refined_defocus_offset``,
        ``refined_pixel_size_offset``, ``refined_pos_y``, ``refined_pos_x`` (relative
        to the crop center), and ``angle_idx`` (index into ``euler_angle_offsets``).

    Raises
    ------
    ValueError
        If ``backend`` is unknown or ``shift_search_radius`` is negative.
    """
    if backend not in ("gemm", "fft"):
        raise ValueError(f"Unknown refine backend '{backend}'. Use 'gemm' or 'fft'.")
    if shift_search_radius is not None and shift_search_radius < 0:
        raise ValueError(
            f"shift_search_radius must be >= 0 or None, got {shift_search_radius}."
        )

    # Convert single device to list for consistent handling
    if isinstance(device, torch.device):
        device = [device]

    inputs = {
        "particle_stack_dft": particle_stack_dft,
        "template_dft": template_dft,
        "euler_angles": euler_angles,
        "euler_angle_offsets": euler_angle_offsets,
        "defocus_u": defocus_u,
        "defocus_v": defocus_v,
        "defocus_angle": defocus_angle,
        "defocus_offsets": defocus_offsets,
        "pixel_size_offsets": pixel_size_offsets,
        "corr_mean": corr_mean,
        "corr_std": corr_std,
        "ctf_kwargs": ctf_kwargs,
        "projective_filters": projective_filters,
        "mag_matrix": mag_matrix,
    }
    differentiable = torch.is_grad_enabled() and _any_requires_grad(inputs)

    ###########################################
    ### Split particle stack across devices ###
    ###########################################

    # The search is an argmax and never needs gradients (and tensors that require
    # grad cannot be sent to the worker processes), so it runs on detached inputs
    kwargs_per_device = construct_multi_gpu_refine_template_kwargs(
        **_detach_tensors(inputs),
        batch_size=batch_size,
        devices=device,
        num_cuda_streams=num_cuda_streams,
    )
    for kwargs in kwargs_per_device:
        kwargs["backend"] = backend
        kwargs["shift_search_radius"] = shift_search_radius
        kwargs["fft_centering"] = fft_centering

    results = run_multiprocess_jobs(
        target=_core_refine_template_single_gpu,
        kwargs_list=kwargs_per_device,
    )

    # Synchronize all devices to ensure all computations are complete
    for dev in device:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    # Concatenate results from all devices, sorted back to the original particle order
    ordered_results = [results[k] for k in sorted(results.keys(), key=int)]
    merged = {
        key: torch.cat([torch.from_numpy(r[key]) for r in ordered_results])
        for key in ordered_results[0]
    }
    sort_indices = torch.argsort(merged["particle_indices"])
    merged = {key: value[sort_indices] for key, value in merged.items()}

    if differentiable:
        cross_correlation, z_score = _differentiable_rescore(
            inputs=inputs,
            angle_idx=merged["angle_idx"],
            defocus_idx=merged["defocus_idx"],
            pixel_size_idx=merged["pixel_size_idx"],
            pos_y=merged["refined_pos_y"],
            pos_x=merged["refined_pos_x"],
            device=device[0],
            batch_size=batch_size,
        )
        merged["refined_cross_correlation"] = cross_correlation.cpu()
        merged["refined_z_score"] = z_score.cpu()

    # Offset refined_pos_{x,y} from the valid crop to the crop center
    _, img_h, img_w = particle_stack_dft.shape
    img_w = 2 * (img_w - 1)  # account for RFFT
    # 'template_dft' may be padded, so use the filter shape
    template_h, template_w = projection_shape_from_filters(projective_filters)
    merged["refined_pos_y"] = merged["refined_pos_y"] - (img_h - template_h + 1) // 2
    merged["refined_pos_x"] = merged["refined_pos_x"] - (img_w - template_w + 1) // 2

    for key in _INTERNAL_RESULT_KEYS:
        merged.pop(key)
    return merged


def _iter_tensors(obj: object) -> Iterator[torch.Tensor]:
    """Yield the tensors in (possibly nested) dicts, lists, and tuples."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, list | tuple):
        for value in obj:
            yield from _iter_tensors(value)


def _any_requires_grad(obj: object) -> bool:
    """True if any tensor in ``obj`` requires grad."""
    return any(t.requires_grad for t in _iter_tensors(obj))


def _detach_tensors(obj: Any) -> Any:
    """Copy of ``obj`` with every (nested) tensor detached."""
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, dict):
        return {key: _detach_tensors(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_detach_tensors(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_tensors(value) for value in obj)
    return obj


def _differentiable_rescore(
    inputs: dict[str, Any],
    angle_idx: torch.Tensor,
    defocus_idx: torch.Tensor,
    pixel_size_idx: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable correlation and z-score of each particle's best hypothesis.

    Parameters
    ----------
    inputs : dict[str, Any]
        The (non-detached) inputs of :func:`core_refine_template`.
    angle_idx, defocus_idx, pixel_size_idx : torch.Tensor
        Indices of the best orientation offset, defocus offset, and pixel size offset
        for each particle, shape (N,).
    pos_y, pos_x : torch.Tensor
        Best positions in the valid crop, shape (N,).
    device : torch.device
        Device to compute on.
    batch_size : int
        Number of particles scored at once.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Cross-correlation and z-score, each with shape (N,), on ``device``.
    """
    index = {
        "angle": angle_idx.to(device),
        "defocus": defocus_idx.to(device),
        "pixel_size": pixel_size_idx.to(device),
        "y": pos_y.to(device),
        "x": pos_x.to(device),
    }
    ctf_kwargs = move_ctf_kwargs_tensors_to_device(inputs["ctf_kwargs"], device)
    mag_matrix = inputs["mag_matrix"]
    mag_matrix = mag_matrix.to(device) if mag_matrix is not None else None
    template_dft = inputs["template_dft"].to(device)
    particle_stack_dft = inputs["particle_stack_dft"].to(device)
    projective_filters = inputs["projective_filters"].to(device)
    defocus_u = inputs["defocus_u"].to(device)
    defocus_v = inputs["defocus_v"].to(device)
    defocus_angle = inputs["defocus_angle"].to(device)
    defocus_offsets = inputs["defocus_offsets"].to(device)
    pixel_size_offsets = inputs["pixel_size_offsets"].to(device)

    prior = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, inputs["euler_angles"].to(device), degrees=True
    )
    offsets = roma.euler_to_rotmat(
        EULER_ANGLE_FMT,
        inputs["euler_angle_offsets"].to(device)[index["angle"]],
        degrees=True,
    )
    rotations = roma.rotmat_composition(
        (offsets.to(torch.float32), prior.to(torch.float32))
    )
    positions = torch.stack([index["y"], index["x"]], dim=-1)

    num_particles = particle_stack_dft.shape[0]
    cross_correlation = []
    for start in range(0, num_particles, max(1, batch_size)):
        stop = min(start + max(1, batch_size), num_particles)
        filters = []
        for n in range(start, stop):
            d_i = int(index["defocus"][n])
            p_i = int(index["pixel_size"][n])
            ctf = calculate_ctf_filter_stack_full_args(
                defocus_u=defocus_u[n],
                defocus_v=defocus_v[n],
                astigmatism_angle=defocus_angle[n],
                defocus_offsets=defocus_offsets[d_i : d_i + 1],
                pixel_size_offsets=pixel_size_offsets[p_i : p_i + 1],
                **ctf_kwargs,
            )
            filters.append(projective_filters[n] * ctf[0, 0])
        cross_correlation.append(
            score_hypotheses(
                template_dft=template_dft,
                rotation_matrices=rotations[start:stop],
                filters=torch.stack(filters),
                particle_images_dft=particle_stack_dft[start:stop],
                positions=positions[start:stop],
                mag_matrix=mag_matrix,
            )
        )
    cc = torch.cat(cross_correlation)

    particles = torch.arange(num_particles, device=device)
    mean = inputs["corr_mean"].to(device)[particles, index["y"], index["x"]]
    std = inputs["corr_std"].to(device)[particles, index["y"], index["x"]]
    return cc, (cc - mean) / std


# pylint: disable=too-many-locals
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
    num_cuda_streams: int,
    mag_matrix: torch.Tensor | None = None,
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
    num_cuda_streams : int
        Number of CUDA streams to use per device.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.

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

        # Split tensors for this device. All these tensors are per-particle, that is
        # the i-th element in each tensor corresponds to the i-th particle in the stack.
        #
        # Move to CPU before passing to child processes: Python multiprocessing uses
        # fork on Linux, and CUDA tensors shared across forked processes via CUDA IPC
        # are unreliable (they silently read as zeros for non-primary GPUs). Sending CPU
        # tensors avoids CUDA IPC entirely; each worker does a clean CPU-->GPU transfer
        # via _move_refine_template_stack_to_device.
        device_particle_stack_dft = particle_stack_dft[start_idx:end_idx].cpu()
        device_euler_angles = euler_angles[start_idx:end_idx].cpu()
        device_defocus_u = defocus_u[start_idx:end_idx].cpu()
        device_defocus_v = defocus_v[start_idx:end_idx].cpu()
        device_defocus_angle = defocus_angle[start_idx:end_idx].cpu()
        device_projective_filters = projective_filters[start_idx:end_idx].cpu()
        device_corr_mean = corr_mean[start_idx:end_idx].cpu()
        device_corr_std = corr_std[start_idx:end_idx].cpu()

        kwargs = {
            "particle_stack_dft": device_particle_stack_dft,
            "particle_indices": particle_indices,
            "template_dft": template_dft.cpu(),
            "euler_angles": device_euler_angles,
            "euler_angle_offsets": euler_angle_offsets.cpu(),
            "defocus_u": device_defocus_u,
            "defocus_v": device_defocus_v,
            "defocus_angle": device_defocus_angle,
            "defocus_offsets": defocus_offsets.cpu(),
            "pixel_size_offsets": pixel_size_offsets.cpu(),
            "corr_mean": device_corr_mean,
            "corr_std": device_corr_std,
            "projective_filters": device_projective_filters,
            "ctf_kwargs": ctf_kwargs,
            "batch_size": batch_size,
            "num_cuda_streams": num_cuda_streams,
            "device": device,
            "mag_matrix": mag_matrix.cpu() if mag_matrix is not None else None,
        }

        kwargs_per_device.append(kwargs)
        start_idx = end_idx

    return kwargs_per_device


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
    backend: RefineBackend = "gemm",
    shift_search_radius: int | None = DEFAULT_SHIFT_SEARCH_RADIUS,
    fft_centering: bool = True,
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
        Number of CUDA streams to use for parallel processing (``"fft"`` backend
        only). Defaults to 1.
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.
    backend : Literal["gemm", "fft"], optional
        Scoring backend, see :func:`core_refine_template`. Default is ``"gemm"``.
    shift_search_radius : int | None, optional
        Position window half-width for the ``"gemm"`` backend. Default is 2.
    fft_centering : bool, optional
        Whether the ``"gemm"`` window starts at an FFT correlation peak. Default True.
    """
    if device.type == "cuda":
        torch.cuda.set_device(device)

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

    with torch.no_grad():
        if backend == "gemm":
            records = _refine_stack_gemm(
                refine_stack,
                ctf_kwargs=ctf_kwargs,
                batch_size=batch_size,
                shift_search_radius=shift_search_radius,
                fft_centering=fft_centering,
                device=device,
                device_id=device_id,
            )
        else:
            records = _refine_stack_fft(
                refine_stack,
                ctf_kwargs=ctf_kwargs,
                batch_size=batch_size,
                num_cuda_streams=num_cuda_streams,
                device=device,
                device_id=device_id,
            )
        result = _records_to_result(records, refine_stack)

    result_dict[device_id] = {k: v.cpu().numpy() for k, v in result.items()}


class _ParticleRecord(NamedTuple):
    """Best hypothesis of one particle, as indices into the search grids."""

    z_score: float
    cross_correlation: float
    angle_idx: int
    defocus_idx: int
    pixel_size_idx: int
    pos_y: int  # in the valid crop
    pos_x: int


def _records_to_result(
    records: list[_ParticleRecord], refine_stack: _RefineTemplateStackOnDevice
) -> dict[str, torch.Tensor]:
    """Turn per-particle best hypotheses into result tensors on the stack's device."""
    device = refine_stack.particle_stack_dft.device
    num_particles = len(records)

    def column(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [getattr(r, name) for r in records], dtype=dtype, device=device
        ).reshape(num_particles)

    angle_idx = column("angle_idx", torch.long)
    defocus_idx = column("defocus_idx", torch.long)
    pixel_size_idx = column("pixel_size_idx", torch.long)

    # Compose the previous Euler angles with the refined offsets
    euler_angles = refine_stack.euler_angles
    refined_offsets = refine_stack.euler_angle_offsets[angle_idx].to(euler_angles.dtype)
    if num_particles > 0:
        refined_euler_angles = combine_euler_angles(refined_offsets, euler_angles)
    else:
        refined_euler_angles = torch.empty((0, 3), device=device)

    # wrap the euler angles back to original ranges
    for dim, period in ((0, 360), (1, 180), (2, 360)):
        refined_euler_angles[:, dim] = torch.where(
            refined_euler_angles[:, dim] < 0,
            refined_euler_angles[:, dim] + period,
            refined_euler_angles[:, dim],
        )

    return {
        "refined_cross_correlation": column("cross_correlation", torch.float32),
        "refined_z_score": column("z_score", torch.float32),
        "refined_euler_angles": refined_euler_angles,
        "refined_defocus_offset": refine_stack.defocus_offsets[defocus_idx],
        "refined_pixel_size_offset": refine_stack.pixel_size_offsets[pixel_size_idx],
        "refined_pos_y": column("pos_y", torch.long),
        "refined_pos_x": column("pos_x", torch.long),
        "particle_indices": refine_stack.particle_indices,  # sort keys
        "angle_idx": angle_idx,
        "defocus_idx": defocus_idx,
        "pixel_size_idx": pixel_size_idx,
    }


def _refine_stack_fft(
    refine_stack: _RefineTemplateStackOnDevice,
    ctf_kwargs: dict,
    batch_size: int,
    num_cuda_streams: int,
    device: torch.device,
    device_id: int,
) -> list[_ParticleRecord]:
    """FFT backend: correlate every hypothesis with the whole particle image."""
    streams = _make_device_streams(device, num_cuda_streams)
    num_particles = refine_stack.particle_stack_dft.shape[0]
    pbar_iter = _tqdm_for_refine_particle_loop(
        num_particles, device, device_id, "Refining"
    )

    records = []
    for i in pbar_iter:
        # Distribute different particles across streams
        stream = streams[i % len(streams)]
        stream_context: AbstractContextManager[None] = (
            nullcontext()
            if stream is None
            else cast(AbstractContextManager[None], torch.cuda.stream(stream))
        )
        with stream_context:
            stats = _core_refine_template_single_thread(
                particle_image_dft=refine_stack.particle_stack_dft[i],
                template_dft=refine_stack.template_dft,
                euler_angles=refine_stack.euler_angles[i, :],
                euler_angle_offsets=refine_stack.euler_angle_offsets,
                defocus_u=refine_stack.defocus_u[i],
                defocus_v=refine_stack.defocus_v[i],
                defocus_angle=refine_stack.defocus_angle[i],
                defocus_offsets=refine_stack.defocus_offsets,
                pixel_size_offsets=refine_stack.pixel_size_offsets,
                ctf_kwargs=ctf_kwargs,
                corr_mean=refine_stack.corr_mean[i],
                corr_std=refine_stack.corr_std[i],
                projective_filter=refine_stack.projective_filters[i],
                batch_size=batch_size,
                mag_matrix=refine_stack.mag_matrix,
            )
        records.append(
            _ParticleRecord(
                z_score=float(stats["max_z_score"]),
                cross_correlation=float(stats["max_cc"]),
                angle_idx=int(stats["angle_idx"]),
                defocus_idx=int(stats["defocus_idx"]),
                pixel_size_idx=int(stats["pixel_size_idx"]),
                pos_y=int(stats["refined_pos_y"]),
                pos_x=int(stats["refined_pos_x"]),
            )
        )

    # Wait for all streams to finish
    _synchronize_device_streams(streams)
    return records


def _make_slice_operands(
    template_dft: torch.Tensor,
    template_shape: tuple[int, int],
    mag_matrix: torch.Tensor | None,
) -> SliceOperandFn:
    """Slice operand function for the GEMM backend (Triton kernel when possible)."""
    if not can_use_triton_slices(template_dft, mag_matrix):
        return reference_slice_operands(template_dft, template_shape, mag_matrix)

    extractor = TritonSliceExtractor(template_dft, template_shape)

    def operands(rotation_matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slices = prepare_slices_for_scoring(
            extractor(rotation_matrices), rows_centered=False
        )
        return slices_to_operands(slices)

    return operands


def _fft_centering_positions(
    refine_stack: _RefineTemplateStackOnDevice,
    ctf_kwargs: dict,
    prior_rotations: torch.Tensor,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Peak position of one FFT correlation per particle at its prior parameters.

    Uses the prior orientation and the defocus / pixel size offsets closest to zero.
    """
    d_0 = int(refine_stack.defocus_offsets.abs().argmin())
    p_0 = int(refine_stack.pixel_size_offsets.abs().argmin())
    filters = []
    for i in range(refine_stack.particle_stack_dft.shape[0]):
        ctf = calculate_ctf_filter_stack_full_args(
            defocus_u=refine_stack.defocus_u[i],
            defocus_v=refine_stack.defocus_v[i],
            astigmatism_angle=refine_stack.defocus_angle[i],
            defocus_offsets=refine_stack.defocus_offsets[d_0 : d_0 + 1],
            pixel_size_offsets=refine_stack.pixel_size_offsets[p_0 : p_0 + 1],
            **ctf_kwargs,
        )
        filters.append(refine_stack.projective_filters[i] * ctf[0, 0])

    cross_correlation = cross_correlate_particle_stack(
        particle_stack_dft=refine_stack.particle_stack_dft,
        template_dft=refine_stack.template_dft,
        rotation_matrices=prior_rotations,
        projective_filters=torch.stack(filters),
        mode="valid",
        batch_size=max(1, batch_size),
        mag_matrix=refine_stack.mag_matrix,
    )
    z_score = (cross_correlation - refine_stack.corr_mean) / refine_stack.corr_std
    flat = z_score.flatten(1).argmax(dim=1)
    crop_w = z_score.shape[-1]
    return list(zip((flat // crop_w).tolist(), (flat % crop_w).tolist(), strict=True))


def _refine_stack_gemm(
    refine_stack: _RefineTemplateStackOnDevice,
    ctf_kwargs: dict,
    batch_size: int,
    shift_search_radius: int | None,
    fft_centering: bool,
    device: torch.device,
    device_id: int,
) -> list[_ParticleRecord]:
    """GEMM backend: score hypotheses in a position window around each particle."""
    num_particles, img_h, img_w = refine_stack.particle_stack_dft.shape
    image_shape = (img_h, 2 * (img_w - 1))
    template_h, template_w = projection_shape_from_filters(
        refine_stack.projective_filters
    )
    crop_shape = (image_shape[0] - template_h + 1, image_shape[1] - template_w + 1)
    num_defocus = refine_stack.defocus_offsets.shape[0]
    ctf_dev_kwargs = move_ctf_kwargs_tensors_to_device(ctf_kwargs, device)

    slice_operands = _make_slice_operands(
        refine_stack.template_dft, (template_h, template_w), refine_stack.mag_matrix
    )
    prior_rotations = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, refine_stack.euler_angles, degrees=True, device=device
    ).to(torch.float32)
    offset_rotations = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, refine_stack.euler_angle_offsets, degrees=True, device=device
    ).to(torch.float32)

    window_smaller_than_crop = shift_search_radius is not None and any(
        2 * shift_search_radius + 1 < n for n in crop_shape
    )
    if fft_centering and window_smaller_than_crop and num_particles > 0:
        centers = _fft_centering_positions(
            refine_stack, ctf_dev_kwargs, prior_rotations, batch_size
        )
    else:
        centers = [(crop_shape[0] // 2, crop_shape[1] // 2)] * num_particles

    pbar_iter = _tqdm_for_refine_particle_loop(
        num_particles, device, device_id, "Refining"
    )
    records = []
    for i in pbar_iter:
        ctf_filters = calculate_ctf_filter_stack_full_args(
            defocus_u=refine_stack.defocus_u[i],
            defocus_v=refine_stack.defocus_v[i],
            astigmatism_angle=refine_stack.defocus_angle[i],
            defocus_offsets=refine_stack.defocus_offsets,
            pixel_size_offsets=refine_stack.pixel_size_offsets,
            **ctf_dev_kwargs,
        )  # (num_px, num_defocus, h, w // 2 + 1)
        filters = refine_stack.projective_filters[i] * ctf_filters
        filters = filters.reshape(-1, *filters.shape[-2:])  # filter = px * n_def + def

        particle_dft = refine_stack.particle_stack_dft[i]
        image = torch.fft.irfftn(particle_dft, s=image_shape, dim=(-2, -1))
        # Rotate the default (best) orientation by the offsets
        rotations = roma.rotmat_composition((offset_rotations, prior_rotations[i]))

        best = gemm_search_particle(
            slice_operands=slice_operands,
            rotation_matrices=rotations,
            image=image,
            image_sum=particle_dft[0, 0].real,
            filters=filters,
            corr_mean=refine_stack.corr_mean[i],
            corr_std=refine_stack.corr_std[i],
            center=centers[i],
            shift_search_radius=shift_search_radius,
            orientation_batch_size=max(1, batch_size),
        )
        records.append(
            _ParticleRecord(
                z_score=best.z_score,
                cross_correlation=best.cross_correlation,
                angle_idx=best.angle_idx,
                defocus_idx=best.filter_idx % num_defocus,
                pixel_size_idx=best.filter_idx // num_defocus,
                pos_y=best.pos_y,
                pos_x=best.pos_x,
            )
        )
    return records


def _iter_refine_particle_correlation_batches(
    particle_image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_u: float | torch.Tensor,
    defocus_v: float | torch.Tensor,
    defocus_angle: float | torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    ctf_kwargs: dict,
    projective_filter: torch.Tensor,
    batch_size: int = 32,
    mag_matrix: torch.Tensor | None = None,
    apply_projection_normalization: bool = True,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, int, int]]:
    """Yield batched local correlations for one particle using refine semantics.

    Note
    ----
    Only correlations are computed here, z-score normalization must happen externally to
    this function.
    """
    img_h, img_w = particle_image_dft.shape
    img_w = 2 * (img_w - 1)
    # 'template_dft' may be zero-padded, so take the template size from the filter
    template_h, template_w = projection_shape_from_filters(projective_filter)
    crop_h = img_h - template_h + 1
    crop_w = img_w - template_w + 1

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

        if particle_image_dft.device.type == "cuda":
            cross_correlation = do_batched_orientation_cross_correlate(
                image_dft=particle_image_dft,
                template_dft=template_dft,
                rotation_matrices=rot_matrix_batch,
                projective_filters=combined_projective_filter,
                apply_normalization=apply_projection_normalization,
                mag_matrix=mag_matrix,
            )
        else:
            cross_correlation = do_batched_orientation_cross_correlate_cpu(
                image_dft=particle_image_dft,
                template_dft=template_dft,
                rotation_matrices=rot_matrix_batch,
                projective_filters=combined_projective_filter,
                apply_normalization=apply_projection_normalization,
                mag_matrix=mag_matrix,
            )

        cross_correlation = cross_correlation[..., :crop_h, :crop_w]

        yield (
            start_idx,
            euler_angle_offsets_batch,
            cross_correlation,
            crop_h,
            crop_w,
        )


def _reduce_refine_best_zscore(
    correlation_batches: Iterator[tuple[int, torch.Tensor, torch.Tensor, int, int]],
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
) -> dict[str, Any]:
    """Reduce local correlation batches to the hypothesis with the best z-score.

    The returned ``max_cc`` is the cross-correlation at that hypothesis.
    """
    best: dict[str, Any] = {
        "max_cc": -1e9,
        "max_z_score": -1e9,
        "refined_phi_offset": 0.0,
        "refined_theta_offset": 0.0,
        "refined_psi_offset": 0.0,
        "refined_defocus_offset": 0.0,
        "refined_pixel_size_offset": 0.0,
        "refined_pos_y": 0,
        "refined_pos_x": 0,
        "angle_idx": 0,
        "defocus_idx": 0,
        "pixel_size_idx": 0,
    }

    for (
        start_idx,
        euler_angle_offsets_batch,
        cross_correlation,
        _crop_h,
        _crop_w,
    ) in correlation_batches:
        # shape is (num_px, num_defocus, num_orientations, crop_h, crop_w)
        z_score = (cross_correlation - corr_mean) / corr_std
        z_max, flat_idx = z_score.reshape(-1).max(dim=0)
        if z_max <= best["max_z_score"]:
            continue

        px_idx, defocus_idx, angle_idx, y_idx, x_idx = (
            int(i) for i in torch.unravel_index(flat_idx, z_score.shape)
        )
        best.update(
            max_cc=cross_correlation[px_idx, defocus_idx, angle_idx, y_idx, x_idx],
            max_z_score=z_max,
            refined_phi_offset=euler_angle_offsets_batch[angle_idx, 0],
            refined_theta_offset=euler_angle_offsets_batch[angle_idx, 1],
            refined_psi_offset=euler_angle_offsets_batch[angle_idx, 2],
            refined_defocus_offset=defocus_offsets[defocus_idx],
            refined_pixel_size_offset=pixel_size_offsets[px_idx],
            refined_pos_y=y_idx,
            refined_pos_x=x_idx,
            angle_idx=angle_idx + start_idx,
            defocus_idx=defocus_idx,
            pixel_size_idx=px_idx,
        )

    return best


# pylint: disable=too-many-locals, too-many-statements
def _core_refine_template_single_thread(
    particle_image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    euler_angle_offsets: torch.Tensor,
    defocus_u: float | torch.Tensor,
    defocus_v: float | torch.Tensor,
    defocus_angle: float | torch.Tensor,
    defocus_offsets: torch.Tensor,
    pixel_size_offsets: torch.Tensor,
    corr_mean: torch.Tensor,
    corr_std: torch.Tensor,
    ctf_kwargs: dict,
    projective_filter: torch.Tensor,
    batch_size: int = 32,
    mag_matrix: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run the single-threaded core refine template function.

    Parameters
    ----------
    particle_image_dft : torch.Tensor
        The real-Fourier transformed particle image. Shape of (H, W).
    template_dft : torch.Tensor
        The template volume to extract central slices from. Real-Fourier transformed
        and fftshifted.
    euler_angles : torch.Tensor
        The previous best euler angle for the particle. Shape of (3,).
    euler_angle_offsets : torch.Tensor
        The Euler angle offsets to apply to each particle. Shape of (k, 3).
    defocus_u : float | torch.Tensor
        The defocus along the major axis for the particle.
    defocus_v : float | torch.Tensor
        The defocus along the minor for the particle.
    defocus_angle : float | torch.Tensor
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
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.

    Returns
    -------
    dict[str, Any]
        The refined statistics for the particle, see :func:`_reduce_refine_best_zscore`.
    """
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
    )

    return _reduce_refine_best_zscore(
        correlation_batches=correlation_batches,
        corr_mean=corr_mean,
        corr_std=corr_std,
        defocus_offsets=defocus_offsets,
        pixel_size_offsets=pixel_size_offsets,
    )


# pylint: disable=too-many-locals
def cross_correlate_particle_stack(
    particle_stack_dft: torch.Tensor,  # (N, H, W)
    template_dft: torch.Tensor,  # (d, h, w)
    rotation_matrices: torch.Tensor,  # (N, 3, 3)
    projective_filters: torch.Tensor,  # (N, h, w)
    mode: Literal["valid", "same"] = "valid",
    batch_size: int = 1024,
    mag_matrix: torch.Tensor | None = None,
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
    mag_matrix : torch.Tensor | None, optional
        Anisotropic magnification matrix of shape (2, 2). If None,
        no magnification transform is applied. Default is None.

    Returns
    -------
    torch.Tensor
        The cross-correlation of the particle stack with the template. Shape will depend
        on the mode used. If "valid", the output will be (N, H-h+1, W-w+1). If "same",
        the output will be (N, H, W).

    Raises
    ------
    ValueError
        If the mode is not "valid" or "same".
    """
    # Helpful constants for later use
    device = particle_stack_dft.device
    num_particles, image_h, image_w = particle_stack_dft.shape
    # account for RFFT ('template_dft' may be padded, so use the filter shape)
    image_w = 2 * (image_w - 1)
    template_h, template_w = projection_shape_from_filters(projective_filters)

    if batch_size == -1:
        batch_size = num_particles

    if mode == "valid":
        output_shape = (
            num_particles,
            image_h - template_h + 1,
            image_w - template_w + 1,
        )
    elif mode == "same":
        output_shape = (num_particles, image_h, image_w)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'valid' or 'same'.")

    out_correlation = torch.zeros(output_shape, device=device)

    # Loop over the particle stack in batches
    for i in range(0, num_particles, batch_size):
        batch_particles_dft = particle_stack_dft[i : i + batch_size]
        batch_rotation_matrices = rotation_matrices[i : i + batch_size]
        batch_projective_filters = projective_filters[i : i + batch_size]

        # Extract template-size Fourier slices (cropping a padded volume and applying
        # the anisotropic magnification transform when provided)
        fourier_slice = extract_template_fourier_slices(
            template_dft=template_dft,
            rotation_matrices=batch_rotation_matrices,
            projection_shape=(template_h, template_w),
            mag_matrix=mag_matrix,
        )
        fourier_slice = torch.fft.ifftshift(fourier_slice, dim=(-2,))
        fourier_slice[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
        fourier_slice *= -1  # flip contrast
        fourier_slice *= batch_projective_filters

        # Inverse Fourier transform and normalize the projection
        projections = torch.fft.irfftn(fourier_slice, dim=(-2, -1))
        projections = torch.fft.ifftshift(projections, dim=(-2, -1))
        projections = normalize_template_projection(
            projections, (template_h, template_w), (image_h, image_w)
        )

        # Padded forward FFT and cross-correlate
        projections_dft = torch.fft.rfftn(
            projections, dim=(-2, -1), s=(image_h, image_w)
        )
        projections_dft = batch_particles_dft * projections_dft.conj()
        cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

        # Handle the output shape
        cross_correlation = handle_correlation_mode(
            cross_correlation, output_shape, mode
        )

        out_correlation[i : i + batch_size] = cross_correlation

    return out_correlation
