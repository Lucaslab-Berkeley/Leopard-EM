"""Pure PyTorch implementation of whole orientation search backend."""

# Following pylint error ignored because torc.fft.* is not recognized as callable
# pylint: disable=E1102

import math
import warnings
from multiprocessing import set_start_method

import roma
import torch
import tqdm

from leopard_em.backend.cross_correlation import (
    do_streamed_orientation_cross_correlate,
)
from leopard_em.backend.process_results import (
    aggregate_distributed_results,
    scale_mip,
)
from leopard_em.backend.utils import (
    do_iteration_statistics_updates_compiled,
    run_multiprocess_jobs,
)

DEFAULT_STATISTIC_DTYPE = torch.float32

# Turn off gradient calculations by default
torch.set_grad_enabled(False)

# Set multiprocessing start method to spawn
set_start_method("spawn", force=True)


###########################################################
###      Main function for whole orientation search     ###
### (inputs generalize beyond those in pydantic models) ###
###########################################################


# pylint: disable=too-many-locals
def core_match_template(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,  # already fftshifted
    ctf_filters: torch.Tensor,
    whitening_filter_template: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    euler_angles: torch.Tensor,
    device: torch.device | list[torch.device],
    orientation_batch_size: int = 1,
    num_cuda_streams: int = 1,
) -> dict[str, torch.Tensor]:
    """Core function for performing the whole-orientation search.

    With the RFFT, the last dimension (fastest dimension) is half the width
    of the input, hence the shape of W // 2 + 1 instead of W for some of the
    input parameters.

    Parameters
    ----------
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1) with the last dimension being the
        half-dimension for real-FFT transformation. NOTE: The original template volume
        should be a cubic volume, i.e. h == w == l.
    ctf_filters : torch.Tensor
        Stack of CTF filters at different pixel size (Cs) and  defocus values to use in
        the search. Has shape (num_Cs, num_defocus, h, w // 2 + 1) where num_Cs are the
        number of pixel sizes searched over, and num_defocus are the number of
        defocus values searched over.
    whitening_filter_template : torch.Tensor
        Whitening filter for the template volume. Has shape (h, w // 2 + 1).
        Gets multiplied with the ctf filters to create a filter stack applied to each
        orientation projection.
    euler_angles : torch.Tensor
        Euler angles (in 'ZYZ' convention) to search over. Has shape
        (num_orientations, 3).
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters, in units of Angstroms. Has
        shape (num_defocus,).
    pixel_values : torch.Tensor
        What pixel size values correspond with the CTF filters, in units of Angstroms.
        Has shape (num_Cs,).
    device : torch.device | list[torch.device]
        Device or devices to split computation across.
    orientation_batch_size : int, optional
        Number of projections, at different orientations, to calculate simultaneously.
        Larger values will use more memory, but can help amortize the cost of Fourier
        slice extraction. The default is 1, but generally values larger than 1 should
        be used for performance.
    num_cuda_streams : int, optional
        Number of CUDA streams to use for parallelizing cross-correlation computation.
        More streams can lead to better performance, especially for high-end GPUs, but
        the performance will degrade if too many streams are used. The default is 1
        which performs well in most cases, but high-end GPUs can benefit from
        increasing this value. NOTE: If the number of streams is greater than the
        number of cross-correlations to compute per batch, then the number of streams
        will be reduced to the number of cross-correlations per batch. This is done to
        avoid unnecessary overhead and performance degradation.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing the following key, value pairs:

            - "mip": Maximum intensity projection of the cross-correlation values across
              orientation and defocus search space.
            - "scaled_mip": Z-score scaled MIP of the cross-correlation values.
            - "best_phi": Best phi angle for each pixel.
            - "best_theta": Best theta angle for each pixel.
            - "best_psi": Best psi angle for each pixel.
            - "best_defocus": Best defocus value for each pixel.
            - "best_pixel_size": Best pixel size value for each pixel.
            - "correlation_sum": Sum of cross-correlation values for each pixel.
            - "correlation_squared_sum": Sum of squared cross-correlation values for
              each pixel.
            - "total_projections": Total number of projections calculated.
            - "total_orientations": Total number of orientations searched.
            - "total_defocus": Total number of defocus values searched.
    """
    ################################################################
    ### Initial checks for input parameters plus and adjustments ###
    ################################################################
    # If there are more streams than cross-correlations to compute per batch, then
    # reduce the number of streams to the number of cross-correlations per batch.
    total_cc_per_batch = (
        orientation_batch_size * defocus_values.shape[0] * pixel_values.shape[0]
    )
    if num_cuda_streams > total_cc_per_batch:
        warnings.warn(
            f"Number of CUDA streams ({num_cuda_streams}) is greater than the "
            f"number of cross-correlations per batch ({total_cc_per_batch}). "
            f"The total cross-correlations per batch is number of pixel sizes "
            f"({pixel_values.shape[0]}) * number of defocus values "
            f"({defocus_values.shape[0]}) * orientation batch size "
            f"({orientation_batch_size}). "
            f"Reducing number of streams to {total_cc_per_batch} for performance.",
            stacklevel=2,
        )
        num_cuda_streams = total_cc_per_batch

    ##############################################################
    ### Pre-multiply the whitening filter with the CTF filters ###
    ##############################################################

    projective_filters = ctf_filters * whitening_filter_template[None, None, ...]

    #########################################
    ### Split orientations across devices ###
    #########################################

    if isinstance(device, torch.device):
        device = [device]

    kwargs_per_device = construct_multi_gpu_match_template_kwargs(
        image_dft=image_dft,
        template_dft=template_dft,
        euler_angles=euler_angles,
        projective_filters=projective_filters,
        defocus_values=defocus_values,
        pixel_values=pixel_values,
        orientation_batch_size=orientation_batch_size,
        num_cuda_streams=num_cuda_streams,
        devices=device,
    )

    result_dict = run_multiprocess_jobs(
        target=_core_match_template_single_gpu,
        kwargs_list=kwargs_per_device,
    )

    # Get the aggregated results
    partial_results = [result_dict[i] for i in range(len(kwargs_per_device))]
    aggregated_results = aggregate_distributed_results(partial_results)
    mip = aggregated_results["mip"]
    best_phi = aggregated_results["best_phi"]
    best_theta = aggregated_results["best_theta"]
    best_psi = aggregated_results["best_psi"]
    best_defocus = aggregated_results["best_defocus"]
    correlation_sum = aggregated_results["correlation_sum"]
    correlation_squared_sum = aggregated_results["correlation_squared_sum"]
    total_projections = aggregated_results["total_projections"]

    mip_scaled = torch.empty_like(mip)
    mip, mip_scaled, correlation_mean, correlation_variance = scale_mip(
        mip=mip,
        mip_scaled=mip_scaled,
        correlation_sum=correlation_sum,
        correlation_squared_sum=correlation_squared_sum,
        total_correlation_positions=total_projections,
    )

    return {
        "mip": mip,
        "scaled_mip": mip_scaled,
        "best_phi": best_phi,
        "best_theta": best_theta,
        "best_psi": best_psi,
        "best_defocus": best_defocus,
        "correlation_mean": correlation_mean,
        "correlation_variance": correlation_variance,
        "total_projections": total_projections,
        "total_orientations": euler_angles.shape[0],
        "total_defocus": defocus_values.shape[0],
    }


def construct_multi_gpu_match_template_kwargs(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    orientation_batch_size: int,
    num_cuda_streams: int,
    devices: list[torch.device],
) -> list[dict[str, torch.Tensor | int]]:
    """Split orientations between requested devices.

    See the `core_match_template` function for further descriptions of the
    input parameters.

    Parameters
    ----------
    image_dft : torch.Tensor
        dft of image
    template_dft : torch.Tensor
        dft of template
    euler_angles : torch.Tensor
        euler angles to search
    projective_filters : torch.Tensor
        filters to apply to each projection
    defocus_values : torch.Tensor
        corresponding defocus values for each filter
    pixel_values : torch.Tensor
        corresponding pixel size values for each filter
    orientation_batch_size : int
        number of projections to calculate at once
    num_cuda_streams : int
        number of CUDA streams to use for parallelizing cross-correlation computation
    devices : list[torch.device]
        list of devices to split the orientations across

    Returns
    -------
    list[dict[str, torch.Tensor | int]]
        List of dictionaries containing the kwargs to call the single-GPU
        function. Each index in the list corresponds to a different device,
        and all tensors in the dictionary have been allocated to that device.
    """
    kwargs_per_device = []

    # Split the euler angles across devices
    euler_angles_split = euler_angles.chunk(len(devices))

    for device, euler_angles_device in zip(devices, euler_angles_split):
        # Allocate and construct the kwargs for this device
        kwargs = {
            "image_dft": image_dft.to(device),
            "template_dft": template_dft.to(device),
            "euler_angles": euler_angles_device.to(device),
            "projective_filters": projective_filters.to(device),
            "defocus_values": defocus_values.to(device),
            "pixel_values": pixel_values.to(device),
            "orientation_batch_size": orientation_batch_size,
            "num_cuda_streams": num_cuda_streams,
        }

        kwargs_per_device.append(kwargs)

    return kwargs_per_device


# pylint: disable=too-many-locals
def _core_match_template_single_gpu(
    result_dict: dict,
    device_id: int,
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    orientation_batch_size: int,
    num_cuda_streams: int,
) -> None:
    """Single-GPU call for template matching.

    NOTE: All tensors *must* be allocated on the same device. By calling the
    user-facing `core_match_template` function this is handled automatically.

    NOTE: The result_dict is a shared dictionary between processes and updated in-place
    with this processes's results under the 'device_id' key.

    Parameters
    ----------
    result_dict : dict
        Dictionary to store the results in.
    device_id : int
        ID of the device which computation is running on. Results will be stored
        in the dictionary with this key.
    image_dft : torch.Tensor
        Real-fourier transform (RFFT) of the image with large image filters
        already applied. Has shape (H, W // 2 + 1).
    template_dft : torch.Tensor
        Real-fourier transform (RFFT) of the template volume to take Fourier
        slices from. Has shape (l, h, w // 2 + 1).
    euler_angles : torch.Tensor
        Euler angles (in 'ZYZ' convention) to search over. Has shape
        (orientations // n_devices, 3). This has already been split (e.g.
        4 devices has shape (orientations // 4, 3).
    projective_filters : torch.Tensor
        Multiplied 'ctf_filters' with 'whitening_filter_template'. Has shape
        (defocus_batch, h, w // 2 + 1). Is RFFT and not fftshifted.
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (defocus_batch,).
    pixel_values : torch.Tensor
        What pixel size values correspond with the CTF filters. Has shape
        (pixel_size_batch,).
    orientation_batch_size : int
        The number of projections to calculate the correlation for at once.
    num_cuda_streams : int
        Number of CUDA streams to use for parallelizing cross-correlation computation.

    Returns
    -------
    None
    """
    device = image_dft.device
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)  # adj. for RFFT

    # Create CUDA streams for parallel computation
    streams = [torch.cuda.Stream(device=device) for _ in range(num_cuda_streams)]

    ################################################
    ### Initialize the tracked output statistics ###
    ################################################

    mip = torch.full(
        size=image_shape_real,
        fill_value=-float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_phi = torch.full(
        size=image_shape_real,
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_theta = torch.full(
        size=image_shape_real,
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_psi = torch.full(
        size=image_shape_real,
        fill_value=-1000.0,
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_defocus = torch.full(
        size=image_shape_real,
        fill_value=float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_pixel_size = torch.full(
        size=image_shape_real,
        fill_value=float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    correlation_sum = torch.zeros(
        size=image_shape_real, dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )
    correlation_squared_sum = torch.zeros(
        size=image_shape_real, dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )

    ########################################################
    ### Setup iterator object with tqdm for progress bar ###
    ########################################################

    total_projections = (
        euler_angles.shape[0] * defocus_values.shape[0] * pixel_values.shape[0]
    )

    num_batches = math.ceil(euler_angles.shape[0] / orientation_batch_size)
    orientation_batch_iterator = tqdm.tqdm(
        range(num_batches),
        desc=f"Progress on device: {device.index}",
        leave=True,
        total=num_batches,
        dynamic_ncols=True,
        position=device.index,
        mininterval=1,  # Slow down to reduce number of lines written
        smoothing=0.05,
        unit="corr",
        unit_scale=total_projections / num_batches,
    )

    ##################################
    ### Start the orientation loop ###
    ##################################

    for i in orientation_batch_iterator:
        euler_angles_batch = euler_angles[
            i * orientation_batch_size : (i + 1) * orientation_batch_size
        ]
        rot_matrix = roma.euler_to_rotmat(
            "ZYZ", euler_angles_batch, degrees=True, device=device
        )

        cross_correlation = do_streamed_orientation_cross_correlate(
            image_dft=image_dft,
            template_dft=template_dft,
            rotation_matrices=rot_matrix,
            projective_filters=projective_filters,
            streams=streams,
        )

        # Update the tracked statistics through compiled function
        do_iteration_statistics_updates_compiled(
            cross_correlation,
            euler_angles_batch,
            defocus_values,
            pixel_values,
            mip,
            best_phi,
            best_theta,
            best_psi,
            best_defocus,
            best_pixel_size,
            correlation_sum,
            correlation_squared_sum,
            image_shape_real[0],
            image_shape_real[1],
        )

    # NOTE: Need to send all tensors back to the CPU as numpy arrays for the shared
    # process dictionary. This is a workaround for now
    result = {
        "mip": mip.cpu().numpy(),
        "best_phi": best_phi.cpu().numpy(),
        "best_theta": best_theta.cpu().numpy(),
        "best_psi": best_psi.cpu().numpy(),
        "best_defocus": best_defocus.cpu().numpy(),
        "best_pixel_size": best_pixel_size.cpu().numpy(),
        "correlation_sum": correlation_sum.cpu().numpy(),
        "correlation_squared_sum": correlation_squared_sum.cpu().numpy(),
        "total_projections": total_projections,
    }

    # Place the results in the shared multi-process manager dictionary so accessible
    # by the main process.
    result_dict[device_id] = result
