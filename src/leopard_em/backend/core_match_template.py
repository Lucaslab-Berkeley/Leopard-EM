"""Pure PyTorch implementation of whole orientation search backend."""

# Following pylint error ignored because torc.fft.* is not recognized as callable
# pylint: disable=E1102

import time
import warnings
from functools import partial
from multiprocessing import set_start_method
from typing import Any, Union

import roma
import torch
import tqdm

from leopard_em.backend.cross_correlation import (
    do_batched_orientation_cross_correlate,
    do_streamed_orientation_cross_correlate,
)
from leopard_em.backend.distributed import (
    MultiprocessWorkIndexQueue,
    run_multiprocess_jobs,
)
from leopard_em.backend.process_results import (
    aggregate_distributed_results,
    decode_global_search_index,
    scale_mip,
)
from leopard_em.backend.utils import do_iteration_statistics_updates_compiled

DEFAULT_STATISTIC_DTYPE = torch.float32

# Turn off gradient calculations by default
torch.set_grad_enabled(False)

# Set multiprocessing start method to spawn
set_start_method("spawn", force=True)


def monitor_match_template_progress(
    queue: "MultiprocessWorkIndexQueue",
    pbar: tqdm.tqdm,
    device_pbars: dict[int, tqdm.tqdm],
    poll_interval: float = 1.0,  # in seconds
) -> None:
    """Helper function for periodic polling of shared queue by tqdm.

    This function monitors the progress of template matching and updates progress bars.
    """
    last_progress = 0
    last_per_device = [0] * len(device_pbars)

    try:
        while True:
            if queue.error_occurred():
                raise RuntimeError("Exiting due to error in another process.")
            progress = queue.get_current_index()
            delta = progress - last_progress

            # Update the global search progress bar
            if delta > 0:
                pbar.update(delta)
                last_progress = progress

            # Update each of the progress bars for each device
            device_counts = queue.get_process_counts()
            for i, dv_pbar in enumerate(device_pbars.values()):
                delta = device_counts[i] - last_per_device[i]
                if delta > 0:
                    dv_pbar.update(delta)
                    last_per_device[i] = device_counts[i]

            # Done with tracking when progress reaches the end of the queue
            if last_progress >= queue.total_indices:
                break

            time.sleep(poll_interval)
    except Exception as e:
        print(f"Error occurred: {e}")
        queue.set_error_flag()
        raise e
    finally:
        # Clean up progress bars
        for dv_pbar in device_pbars.values():
            dv_pbar.close()
        pbar.close()


def setup_progress_tracking(
    index_queue: "MultiprocessWorkIndexQueue",
    unit_scale: Union[float, int],
    devices: list[torch.device],
) -> tuple[tqdm.tqdm, dict[int, tqdm.tqdm]]:
    """Setup global and per-device tqdm progress bars for template matching.

    Parameters
    ----------
    index_queue : MultiprocessWorkIndexQueue
        The shared work queue tracking global indices.
    unit_scale : Union[float, int]
        Scaling factor to apply to units
    devices : list[torch.device]
        List of devices to create per-device progress bars for.

    Returns
    -------
    tuple[tqdm.tqdm, dict[int, tqdm.tqdm]]
        Global progress bar and dictionary of per-device progress bars.
    """
    # Global progress bar
    global_pbar = tqdm.tqdm(
        total=index_queue.total_indices,
        desc="2DTM progress",
        dynamic_ncols=True,
        smoothing=0.02,
        unit="corr",
        unit_scale=unit_scale,
    )

    # Per-device progress bars
    device_pbars = {
        i: tqdm.tqdm(
            desc=f"device - {d.type} {d.index}",
            dynamic_ncols=True,
            smoothing=0.02,
            unit="corr",
            unit_scale=unit_scale,
            position=i + 1,  # place below the global bar
            leave=True,
        )
        for i, d in enumerate(devices)
    }

    return global_pbar, device_pbars


###########################################################
###      Main function for whole orientation search     ###
### (inputs generalize beyond those in pydantic models) ###
###########################################################


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
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
    backend: str = "streamed",
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
        Euler angles (in 'ZYZ' convention & in units of degrees) to search over. Has
        shape (num_orientations, 3).
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
    backend : str, optional
        The backend to use for computation. Defaults to 'streamed'.
        Must be 'streamed' or 'batched'.

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

    # Ensure the tensors are all on the CPU. The _core_match_template_single_gpu
    # function will move them onto the correct device.
    image_dft = image_dft.cpu()
    template_dft = template_dft.cpu()
    ctf_filters = ctf_filters.cpu()
    whitening_filter_template = whitening_filter_template.cpu()
    defocus_values = defocus_values.cpu()
    pixel_values = pixel_values.cpu()
    euler_angles = euler_angles.cpu()

    ##############################################################
    ### Pre-multiply the whitening filter with the CTF filters ###
    ##############################################################

    projective_filters = ctf_filters * whitening_filter_template[None, None, ...]
    total_projections = (
        euler_angles.shape[0] * defocus_values.shape[0] * pixel_values.shape[0]
    )

    ############################################################
    ### Shared queue mechanism and multiprocessing arguments ###
    ############################################################

    if isinstance(device, torch.device):
        device = [device]

    index_queue = MultiprocessWorkIndexQueue(
        total_indices=euler_angles.shape[0],
        batch_size=orientation_batch_size,
        prefetch_size=10,
        num_processes=len(device),
    )
    global_pbar, device_pbars = setup_progress_tracking(
        index_queue=index_queue,
        unit_scale=defocus_values.shape[0] * pixel_values.shape[0],
        devices=device,
    )
    progress_callback = partial(
        monitor_match_template_progress,
        queue=index_queue,
        pbar=global_pbar,
        device_pbars=device_pbars,
    )

    kwargs_per_device = []
    for d in device:
        kwargs = {
            "index_queue": index_queue,
            "image_dft": image_dft,
            "template_dft": template_dft,
            "euler_angles": euler_angles,
            "projective_filters": projective_filters,
            "defocus_values": defocus_values,
            "pixel_values": pixel_values,
            "orientation_batch_size": orientation_batch_size,
            "num_cuda_streams": num_cuda_streams,
            "backend": backend,
            "device": d,
        }

        kwargs_per_device.append(kwargs)

    result_dict = run_multiprocess_jobs(
        target=_core_match_template_multiprocess_wrapper,
        kwargs_list=kwargs_per_device,
        post_start_callback=progress_callback,
    )

    # Get the aggregated results
    partial_results = [result_dict[i] for i in range(len(kwargs_per_device))]
    aggregated_results = aggregate_distributed_results(partial_results)
    mip = aggregated_results["mip"]
    best_global_index = aggregated_results["best_global_index"]
    correlation_sum = aggregated_results["correlation_sum"]
    correlation_squared_sum = aggregated_results["correlation_squared_sum"]

    # Map from global search index to the best defocus & angles
    best_phi, best_theta, best_psi, best_defocus = decode_global_search_index(
        best_global_index, pixel_values, defocus_values, euler_angles
    )

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


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def _core_match_template_single_gpu(
    rank: int,
    index_queue: MultiprocessWorkIndexQueue,
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    euler_angles: torch.Tensor,
    projective_filters: torch.Tensor,
    defocus_values: torch.Tensor,
    pixel_values: torch.Tensor,
    orientation_batch_size: int,
    num_cuda_streams: int,
    backend: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-GPU call for template matching.

    Parameters
    ----------
    rank : int
        Rank of the device which computation is running on. Used for tracking grabbed
        work from the shared queue.
    index_queue : MultiprocessWorkIndexQueue
        Torch multiprocessing object for retrieving the next batch of orientations to
        process during the 2DTM search.
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
        (num_Cs, num_defocus, h, w // 2 + 1). Is RFFT and not fftshifted.
    defocus_values : torch.Tensor
        What defoucs values correspond with the CTF filters. Has shape
        (num_defocus,).
    pixel_values : torch.Tensor
        What pixel size values correspond with the CTF filters. Has shape
        (pixel_size_batch,).
    orientation_batch_size : int
        The number of projections to calculate the correlation for at once.
    num_cuda_streams : int
        Number of CUDA streams to use for parallelizing cross-correlation computation.
    backend : str, optional
        The backend to use for computation.
        Defaults to 'streamed'. Must be 'streamed' or 'batched'.
    device : torch.device
        Device to run the computation on. All tensors must be allocated on this device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing the following tensors:
            - mip: Maximum intensity projection of the cross-correlation values across
              orientation and defocus search space.
            - best_global_index: Global index of the best match for each pixel.
            - correlation_sum: Sum of cross-correlation values for each pixel.
            - correlation_squared_sum: Sum of squared cross-correlation values for
              each pixel.
    """
    image_shape_real = (image_dft.shape[0], image_dft.shape[1] * 2 - 2)  # adj. for RFFT

    # Create CUDA streams for parallel computation
    streams = [torch.cuda.Stream(device=device) for _ in range(num_cuda_streams)]

    ########################################
    ### Pass all tensors onto the device ###
    ########################################

    image_dft = image_dft.to(device)
    template_dft = template_dft.to(device)
    euler_angles = euler_angles.to(device)
    projective_filters = projective_filters.to(device)

    num_orientations = euler_angles.shape[0]
    num_defocus = defocus_values.shape[0]
    num_cs = pixel_values.shape[0]

    local_to_global_idx_increment = torch.tensor(
        [
            df * num_orientations + cs * num_defocus * num_orientations
            for cs in range(num_cs)
            for df in range(num_defocus)
        ],
        dtype=torch.int32,
        device=device,
    )

    ################################################
    ### Initialize the tracked output statistics ###
    ################################################

    mip = torch.full(
        size=image_shape_real,
        fill_value=-float("inf"),
        dtype=DEFAULT_STATISTIC_DTYPE,
        device=device,
    )
    best_global_index = torch.full(
        image_shape_real, fill_value=-1, dtype=torch.int32, device=device
    )
    correlation_sum = torch.zeros(
        size=image_shape_real, dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )
    correlation_squared_sum = torch.zeros(
        size=image_shape_real, dtype=DEFAULT_STATISTIC_DTYPE, device=device
    )

    ##################################
    ### Start the orientation loop ###
    ##################################

    while True:
        if index_queue.error_occurred():
            raise RuntimeError("Exiting due to error in another process.")

        try:
            indices = index_queue.get_next_indices(process_id=rank)
            if indices is None:
                break

            # Fetching more than orientation_batch_size, so need inner loop
            start_idx, end_idx = indices

            for i in range(start_idx, end_idx, orientation_batch_size):
                euler_angles_batch = euler_angles[i : i + orientation_batch_size]
                rot_matrix = roma.euler_to_rotmat(
                    "ZYZ", euler_angles_batch, degrees=True, device=device
                )

                # Calculate the global search indices. These act as if the entire search
                # space of bach shape (num_cs, num_defocus, num_orientations) had been
                # flattened into one contiguous dimension.
                indices = torch.arange(
                    i,
                    i + orientation_batch_size,
                    dtype=torch.int32,
                    device=device,
                )
                batch_search_indices = indices + local_to_global_idx_increment[:, None]
                batch_search_indices = batch_search_indices.flatten()

                if backend == "batched":
                    cross_correlation = do_batched_orientation_cross_correlate(
                        image_dft=image_dft,
                        template_dft=template_dft,
                        rotation_matrices=rot_matrix,
                        projective_filters=projective_filters,
                    )
                else:
                    cross_correlation = do_streamed_orientation_cross_correlate(
                        image_dft=image_dft,
                        template_dft=template_dft,
                        rotation_matrices=rot_matrix,
                        projective_filters=projective_filters,
                        streams=streams,
                    )

                # Update the tracked statistics
                do_iteration_statistics_updates_compiled(
                    cross_correlation=cross_correlation,
                    current_indexes=batch_search_indices,
                    mip=mip,
                    best_global_index=best_global_index,
                    correlation_sum=correlation_sum,
                    correlation_squared_sum=correlation_squared_sum,
                    img_h=image_shape_real[0],
                    img_w=image_shape_real[1],
                )
        except Exception as e:
            index_queue.set_error_flag()
            print(f"Error occurred in process {rank}: {e}")
            raise e

    # Synchronization barrier post-computation
    for stream in streams:
        stream.synchronize()

    torch.cuda.synchronize(device)

    return mip, best_global_index, correlation_sum, correlation_squared_sum


def _core_match_template_multiprocess_wrapper(
    result_dict: dict, rank: int, **kwargs: dict[str, Any]
) -> None:
    """Wrapper around _core_match_template_single_gpu for use with multiprocessing.

    This function places results into a shared dictionary for retrieval by the main
    core_match_template function. These results are stored under the 'rank' key, and
    they need to exist on the CPU as numpy arrays for the shared dictionary.

    See the _core_match_template_single_gpu function for parameter descriptions.
    """
    mip, best_global_index, correlation_sum, correlation_squared_sum = (
        _core_match_template_single_gpu(rank, **kwargs)  # type: ignore[arg-type]
    )

    # NOTE: Need to send all tensors back to the CPU as numpy arrays for the shared
    # process dictionary. This is a workaround for now
    result = {
        "mip": mip.cpu().numpy(),
        "best_global_index": best_global_index.cpu().numpy(),
        "correlation_sum": correlation_sum.cpu().numpy(),
        "correlation_squared_sum": correlation_squared_sum.cpu().numpy(),
    }

    # Place the results in the shared multi-process manager dictionary so accessible
    # by the main process.
    result_dict[rank] = result

    # Final cleanup to release all tensors from this GPU
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
