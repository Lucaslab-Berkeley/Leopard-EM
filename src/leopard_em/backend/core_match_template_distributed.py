"""Distributed multi-node version of the core match_template implementation."""

import os
import random
import socket
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

from leopard_em.backend.core_match_template import (
    _core_match_template_single_gpu,
)
from leopard_em.backend.distributed import (
    DistributedTCPIndexQueue,
    TensorShapeDataclass,
)
from leopard_em.backend.process_results import (
    aggregate_distributed_results,
    decode_global_search_index,
    scale_mip,
)

# Turn off gradient calculations
torch.set_grad_enabled(False)


def _check_port_free(port: int) -> bool:
    """Check if a TCP port is free to use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def _find_free_port(
    start_port: int = 20000, end_port: int = 65000, tries: int = 64
) -> int:
    """Find a free TCP port to use for TCPStore."""
    for i in range(tries):
        p = random.randint(start_port, end_port)
        print(f"Retry {i} / {tries}: Trying port {p} for TCPStore")
        if _check_port_free(p):
            return p
    raise RuntimeError("Unable to find free port for TCPStore")


def _check_distributed_and_device(rank: int, device: torch.device) -> None:
    """Check that distributed is initialized and device is a single CUDA device."""
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed core_match_template_distributed called without "
            "initializing the torch distributed process group. Please call "
            "`dist.init_process_group` before calling this function."
        )

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise ValueError(
            "Distributed core_match_template_distributed must be called with a "
            "single CUDA device across all processes."
            f"Rank {rank} received device={device}."
        )


def _extract_and_broadcast_tensors(
    device: torch.device, rank: int, kwargs: dict[str, torch.Tensor]
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Helper fn to extract and broadcast tensor data from rank zero to all ranks.

    Parameters
    ----------
    device : torch.device
        The CUDA device to use for this process.
    rank : int
        Global rank of this process.
    kwargs : dict[str, torch.Tensor]
        Additional keyword arguments passed to the single-GPU core function. For the
        zeroth rank this should be a dictionary of Tensor objects (all other ranks can
        pass an empty dictionary):
    """
    # Only do extraction on rank zero
    if rank == 0:
        for k in [
            "image_dft",
            "template_dft",
            "ctf_filters",
            "whitening_filter_template",
            "defocus_values",
            "pixel_values",
            "euler_angles",
        ]:
            if k not in kwargs:
                raise ValueError(
                    f"Rank 0 missing tensor '{k}' to call."
                    "core_match_template_distributed."
                )
            if not isinstance(kwargs[k], torch.Tensor):
                raise ValueError(
                    f"Rank 0 received non-tensor '{k}' argument to "
                    "core_match_template_distributed."
                )
        whitening_filter_template = kwargs["whitening_filter_template"].to(device)
        defocus_values = kwargs["defocus_values"].to(device)
        template_dft = kwargs["template_dft"].to(device)
        pixel_values = kwargs["pixel_values"].to(device)
        euler_angles = kwargs["euler_angles"].to(device)
        ctf_filters = kwargs["ctf_filters"].to(device)
        image_dft = kwargs["image_dft"].to(device)

    #############################################################
    ### Logic for loading / broadcasting tensors to all ranks ###
    #############################################################

    # Rank zero has all the data. No other ranks "know" the size/shape of data, so
    # first must extract the shapes before a tensor broadcast can occur.
    broadcast_list: list[Optional[TensorShapeDataclass]] = [None]
    if rank == 0:
        # Create a dataclass with the expected tensor shapes
        expected_shapes = TensorShapeDataclass(
            image_dft_shape=tuple(image_dft.shape),
            template_dft_shape=tuple(template_dft.shape),
            ctf_filters_shape=tuple(ctf_filters.shape),
            whitening_filter_template_shape=tuple(whitening_filter_template.shape),
            euler_angles_shape=tuple(euler_angles.shape),
            defocus_values_shape=tuple(defocus_values.shape),
            pixel_values_shape=tuple(pixel_values.shape),
        )

        broadcast_list = [expected_shapes]
        dist.broadcast_object_list(broadcast_list, src=0)

    # For all other ranks, first receive the expected shapes
    else:
        dist.broadcast_object_list(broadcast_list, src=0)
        assert broadcast_list[0] is not None
        expected_shapes = broadcast_list[0]

    # Now all processes have the initialized 'expected_shapes' variable. Create
    # empty tensors of the correct shape on all non-zero ranks
    if rank != 0:
        # fmt: off
        # pylint: disable=line-too-long
        image_dft                   = torch.empty(expected_shapes.image_dft_shape,                  dtype=torch.complex64, device=device)  # noqa: E501
        template_dft                = torch.empty(expected_shapes.template_dft_shape,               dtype=torch.complex64, device=device)  # noqa: E501
        ctf_filters                 = torch.empty(expected_shapes.ctf_filters_shape,                dtype=torch.float32,   device=device)  # noqa: E501
        whitening_filter_template   = torch.empty(expected_shapes.whitening_filter_template_shape,  dtype=torch.float32,   device=device)  # noqa: E501
        euler_angles                = torch.empty(expected_shapes.euler_angles_shape,               dtype=torch.float32,   device=device)  # noqa: E501
        defocus_values              = torch.empty(expected_shapes.defocus_values_shape,             dtype=torch.float32,   device=device)  # noqa: E501
        pixel_values                = torch.empty(expected_shapes.pixel_values_shape,               dtype=torch.float32,   device=device)  # noqa: E501
        # pylint: enable=line-too-long
        # fmt: on

    # Now broadcast all the tensors from rank 0 to all other ranks.
    # Default is not to use async operations, so these are blocking calls.
    dist.broadcast(image_dft, src=0)
    dist.broadcast(template_dft, src=0)
    dist.broadcast(ctf_filters, src=0)
    dist.broadcast(whitening_filter_template, src=0)
    dist.broadcast(euler_angles, src=0)
    dist.broadcast(defocus_values, src=0)
    dist.broadcast(pixel_values, src=0)

    return (
        image_dft,
        template_dft,
        ctf_filters,
        whitening_filter_template,
        euler_angles,
        defocus_values,
        pixel_values,
    )


def _setup_distributed_queue(
    world_size: int, rank: int, orientation_batch_size: int, total_indices: int
) -> DistributedTCPIndexQueue:
    """Helper function to setup the distributed TCP index queue.

    Parameters
    ----------
    world_size : int
        Total number of processes in the distributed job.
    rank : int
        Global rank of this process.
    orientation_batch_size : int
        Number of orientations to process in a single batch.
    total_indices : int
        Total number of indices to process.

    Returns
    -------
    DistributedTCPIndexQueue
        The initialized distributed TCP index queue.
    """
    # NOTE: This following code was giving me trouble without setting
    # TORCHELASTIC_USE_AGENT_STORE=0 to avoid using the torchelastic default store from
    # interfering with the TCPStore initialization. Unsure the underlying mechanism
    # of causing the synchronization hanging -- Matthew Giammar
    os.environ.setdefault("TORCHELASTIC_USE_AGENT_STORE", "0")

    tcp_host_name = os.environ.get("MASTER_ADDR", None)
    tcp_host_port = os.environ.get("MASTER_PORT", None)

    assert tcp_host_name is not None, "MASTER_ADDR environment variable not set"
    assert tcp_host_port is not None, "MASTER_PORT environment variable not set"

    tcp_host_port = int(tcp_host_port) + 1  # type: ignore[assignment]

    # Only on rank 0 check if port is free. If not, find a free port.
    if rank == 0:
        if not _check_port_free(tcp_host_port):  # type: ignore[arg-type]
            tcp_host_port = _find_free_port()  # type: ignore[assignment]

    # Regardless of free or not, need to do a broadcast to synchronize all ranks
    port_list = [tcp_host_port]
    dist.broadcast_object_list(port_list, src=0)
    tcp_host_port = int(port_list[0])  # type: ignore[assignment]

    # Barrier for broadcast to complete
    dist.barrier()

    tcp_store = dist.TCPStore(
        host_name=tcp_host_name,
        port=tcp_host_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=30),  # reduce down from default of 5 minutes
        wait_for_workers=False,
    )

    # Ensure rank 0 initializes the shared keys on the store and synchronize.
    DistributedTCPIndexQueue.initialize_store(
        store=tcp_store,
        rank=rank,
        num_processes=world_size,
        counter_key="next_index",
        error_key="error_flag",
        process_counts_prefix="process_count_",
    )

    distributed_queue = DistributedTCPIndexQueue(
        store=tcp_store,
        total_indices=total_indices,
        batch_size=orientation_batch_size,
        num_processes=world_size,
        prefetch_size=10,  # NOTE: May need to adjust this up when many nodes
    )

    return distributed_queue


def _gather_tensors_to_rank_zero(
    world_size: int,
    rank: int,
    mip: torch.Tensor,
    best_global_index: torch.Tensor,
    correlation_sum: torch.Tensor,
    correlation_squared_sum: torch.Tensor,
) -> tuple[
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
]:
    """Helper function to gather distributed tensor results into rank zero.

    Parameters
    ----------
    world_size : int
        Total number of processes in the distributed job.
    rank : int
        Global rank of this process.
    mip : torch.Tensor
        Maximum intensity projection tensor from this rank.
    best_global_index : torch.Tensor
        Best global index tensor from this rank.
    correlation_sum : torch.Tensor
        Correlation sum tensor from this rank.
    correlation_squared_sum : torch.Tensor
        Correlation squared sum tensor from this rank.

    Returns
    -------
    tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]
        Four lists of tensors gathered on rank zero, or None on other ranks. Each item
        in the list (on rank zero) corresponds to a tensor from each rank. Ordering is
        mip, best_global_index, correlation_sum, correlation_squared_sum.
    """
    # Gather into a list of results on the main process
    # NOTE: This is assuming there is enough GPU memory on the zeroth rank to hold.
    # There are 4 tensors each ~64 MB per GPU (~256 MB total) so this is a fair
    # assumption for most systems. Would need >= 64 GPUs to exceed 16 GB of memory.
    # Can wrap this reduction into multiple groups, e.g. one per node, to reduce
    # memory pressure on the main process GPU
    # fmt: off
    # pylint: disable=line-too-long
    if rank == 0:
        gather_mip                     = [torch.zeros_like(mip) for                     _ in range(world_size)]  # noqa: E501
        gather_best_global_index       = [torch.zeros_like(best_global_index) for       _ in range(world_size)]  # noqa: E501
        gather_correlation_sum         = [torch.zeros_like(correlation_sum) for         _ in range(world_size)]  # noqa: E501
        gather_correlation_squared_sum = [torch.zeros_like(correlation_squared_sum) for _ in range(world_size)]  # noqa: E501
    else:
        gather_mip                     = None
        gather_best_global_index       = None
        gather_correlation_sum         = None
        gather_correlation_squared_sum = None
    # pylint: enable=line-too-long
    # fmt: on

    dist.barrier()
    dist.gather(tensor=mip, gather_list=gather_mip, dst=0)
    dist.gather(
        tensor=best_global_index,
        gather_list=gather_best_global_index,
        dst=0,
    )
    dist.gather(
        tensor=correlation_sum,
        gather_list=gather_correlation_sum,
        dst=0,
    )
    dist.gather(
        tensor=correlation_squared_sum,
        gather_list=gather_correlation_squared_sum,
        dst=0,
    )
    dist.barrier()

    return (
        gather_mip,
        gather_best_global_index,
        gather_correlation_sum,
        gather_correlation_squared_sum,
    )


# pylint: disable=too-many-locals
def core_match_template_distributed(
    world_size: int,
    rank: int,
    local_rank: int,
    device: torch.device,
    orientation_batch_size: int = 1,
    num_cuda_streams: int = 1,
    backend: str = "streamed",
    **kwargs: dict,
) -> dict[str, torch.Tensor]:
    """Distributed multi-node core function for the match template program.

    Parameters
    ----------
    world_size : int
        Total number of processes in the distributed job.
    rank : int
        Global rank of this process.
    local_rank : int
        Local rank of this process on the current node.
    device : torch.device
        The CUDA device to use for this process. This *must* be a single device.
    orientation_batch_size : int, optional
        Number of orientations to process in a single batch, by default 1.
    num_cuda_streams : int, optional
        Number of CUDA streams to use for overlapping data transfers and
        computation, by default 1.
    backend : str, optional
        The backend to use for computation. Defaults to 'streamed'.
        Must be 'streamed' or 'batched'.
    **kwargs : dict[str, torch.Tensor]
        Additional keyword arguments passed to the single-GPU core function. For the
        zeroth rank this should be a dictionary of Tensor objects with the following
        fields (all other ranks can pass an empty dictionary):
        - image_dft:
            Real-fourier transform (RFFT) of the image with large image filters
            already applied. Has shape (H, W // 2 + 1).
        - template_dft:
            Real-fourier transform (RFFT) of the template volume to take Fourier
            slices from. Has shape (l, h, w // 2 + 1) with the last dimension being the
            half-dimension for real-FFT transformation. NOTE: The original template
            volume should be a cubic volume, i.e. h == w == l.
        - ctf_filters:
            Stack of CTF filters at different pixel size (Cs) and  defocus values to use
            in the search. Has shape (num_Cs, num_defocus, h, w // 2 + 1) where num_Cs
            are the number of pixel sizes searched over, and num_defocus are the number
            of defocus values searched over.
        - whitening_filter_template: Precomputed whitening filter for the template.
            Whitening filter for the template volume. Has shape (h, w // 2 + 1).
            Gets multiplied with the ctf filters to create a filter stack applied to
            each orientation projection.
        - euler_angles:
            Euler angles (in 'ZYZ' convention & in units of degrees) to search over. Has
            shape (num_orientations, 3).
        - defocus_values: 1D tensor of defocus values to search.
            What defoucs values correspond with the CTF filters, in units of Angstroms.
            Has shape (num_defocus,).
        - pixel_values: 1D tensor of pixel values to search.
            What pixel size values correspond with the CTF filters, in units of
            Angstroms. Has shape (num_Cs,).
    """
    # Check proper distributed initialization and CUDA device
    _check_distributed_and_device(rank, device)
    _ = local_rank

    torch.cuda.set_device(device)

    # Extract (only on rank zero) and broadcast tensor data to all ranks
    (
        image_dft,
        template_dft,
        ctf_filters,
        whitening_filter_template,
        euler_angles,
        defocus_values,
        pixel_values,
    ) = _extract_and_broadcast_tensors(device, rank, kwargs)

    ##############################################################
    ### Pre-multiply the whitening filter with the CTF filters ###
    ##############################################################

    projective_filters = ctf_filters * whitening_filter_template[None, None, ...]
    total_projections = (
        euler_angles.shape[0] * defocus_values.shape[0] * pixel_values.shape[0]
    )

    ########################################################
    ### TCP Setup for distributed index queue management ###
    ########################################################

    distributed_queue = _setup_distributed_queue(
        world_size=world_size,
        rank=rank,
        orientation_batch_size=orientation_batch_size,
        total_indices=euler_angles.shape[0],
    )

    ###########################################################
    ### Calling the single GPU core match template function ###
    ###########################################################

    dist.barrier()
    (mip, best_global_index, correlation_sum, correlation_squared_sum) = (
        _core_match_template_single_gpu(
            rank=rank,
            index_queue=distributed_queue,  # type: ignore
            image_dft=image_dft,
            template_dft=template_dft,
            euler_angles=euler_angles,
            projective_filters=projective_filters,
            defocus_values=defocus_values,
            pixel_values=pixel_values,
            orientation_batch_size=orientation_batch_size,
            num_cuda_streams=num_cuda_streams,
            backend=backend,
            device=device,
        )
    )
    dist.barrier()

    # Gather all tensors to rank zero GPU
    (
        gather_mip,
        gather_best_global_index,
        gather_correlation_sum,
        gather_correlation_squared_sum,
    ) = _gather_tensors_to_rank_zero(
        world_size=world_size,
        rank=rank,
        mip=mip,
        best_global_index=best_global_index,
        correlation_sum=correlation_sum,
        correlation_squared_sum=correlation_squared_sum,
    )

    ##################################################
    ### Final aggregation step on the main process ###
    ##################################################

    if rank != 0:
        return {}

    # Continue on the main process only
    assert gather_mip is not None
    assert gather_best_global_index is not None
    assert gather_correlation_sum is not None
    assert gather_correlation_squared_sum is not None

    aggregated_results = aggregate_distributed_results(
        results=[
            {
                "mip": mip,
                "best_global_index": gidx,
                "correlation_sum": corr_sum,
                "correlation_squared_sum": corr_sq_sum,
            }
            for mip, gidx, corr_sum, corr_sq_sum in zip(
                gather_mip,
                gather_best_global_index,
                gather_correlation_sum,
                gather_correlation_squared_sum,
            )
        ]
    )
    mip = aggregated_results["mip"]
    best_global_index = aggregated_results["best_global_index"]
    correlation_sum = aggregated_results["correlation_sum"]
    correlation_squared_sum = aggregated_results["correlation_squared_sum"]

    # Ensuring all tensors are now on the CPU device:
    # fmt: off
    mip                     = mip.cpu()
    best_global_index       = best_global_index.cpu()
    correlation_sum         = correlation_sum.cpu()
    correlation_squared_sum = correlation_squared_sum.cpu()
    pixel_values            = pixel_values.cpu()
    defocus_values          = defocus_values.cpu()
    euler_angles            = euler_angles.cpu()
    # fmt: on

    # Map from global search index to the best defocus & angles
    # pylint: disable=duplicate-code
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
        "mip": mip.cpu(),
        "scaled_mip": mip_scaled.cpu(),
        "best_phi": best_phi.cpu(),
        "best_theta": best_theta.cpu(),
        "best_psi": best_psi.cpu(),
        "best_defocus": best_defocus.cpu(),
        "correlation_mean": correlation_mean.cpu(),
        "correlation_variance": correlation_variance.cpu(),
        "total_projections": total_projections,
        "total_orientations": euler_angles.shape[0],
        "total_defocus": defocus_values.shape[0],
    }
