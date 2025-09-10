"""Distributed multi-node version of the core match_template implementation."""

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

from leopard_em.backend.core_match_template import (
    _core_match_template_single_gpu,
)
from leopard_em.backend.process_results import (
    aggregate_distributed_results,
    decode_global_search_index,
    scale_mip,
)

# Turn off gradient calculations by default
torch.set_grad_enabled(False)


############################################################
### Coordinator class for shared async work across nodes ###
############################################################
class DistributedTCPIndexQueue:
    """Distributed work index queue backed by torch.distributed.TCPStore.

    Drop-in replacement for SharedWorkIndexQueue but for multi-node setups.

    Parameters
    ----------
    store : dist.TCPStore
        A torch.distributed.TCPStore object for managing shared state. Must be already
        initialized and reachable by all processes.
    total_indices : int
        The total number of indices (work items) to be processed. Each index is
        considered its own work item, and these items will generally batched together.
    batch_size : int
        The number of indices to be processed in each batch.
    num_processes : int
        The total number of processes grabbing work from this queue. Used as a way
        to track how fast each process is grabbing work from the queue
    prefetch_size : int
        The number of indices to prefetch for processing. Is a multiplicitive factor
        for batch_size. For example, if batch_size is 10 and prefetch_size is 3, then
        up to 30 indices will be prefetched for processing.
    counter_key : str
        The key in the TCPStore for the shared next index counter.
    error_key : str
        The key in the TCPStore for the shared error flag.
    process_counts_prefix : str
        The prefix for keys in the TCPStore for the per-process claimed counts.
    """

    store: dist.TCPStore
    total_indices: int
    batch_size: int
    num_processes: int
    prefetch_size: int
    counter_key: str
    error_key: str
    process_counts_prefix: str

    def __init__(
        self,
        store: dist.TCPStore,
        rank: int,  # process rank, only used for store initialization
        total_indices: int,
        batch_size: int,
        num_processes: int,
        prefetch_size: int = 10,
        counter_key: str = "next_index",
        error_key: str = "error_flag",
        process_counts_prefix: str = "process_count_",
    ):
        self.store = store
        self.total_indices = total_indices
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.num_processes = num_processes

        self.counter_key = counter_key
        self.error_key = error_key
        self.process_counts_prefix = process_counts_prefix

        # Initialize shared values only once (on rank 0)
        # Other ranks must call with store already containing keys
        if rank == 0:
            if not store.compare_set(counter_key, "0", "0"):  # no-op set if unset
                store.set(counter_key, "0")
            if not store.compare_set(error_key, "0", "0"):
                store.set(error_key, "0")
            for pid in range(num_processes):
                k = f"{process_counts_prefix}{pid}"
                if not store.compare_set(k, "0", "0"):
                    store.set(k, "0")

    def get_next_indices(
        self, process_id: Optional[int] = None
    ) -> Optional[tuple[int, int]]:
        """Atomically claim the next chunk of indices for a process."""
        delta = self.batch_size * self.prefetch_size

        # fetch-and-add returns the *new* value after increment
        new_val = self.store.add(self.counter_key, delta)
        end_idx = int(new_val)
        start_idx = end_idx - delta

        if start_idx >= self.total_indices:
            return None

        if end_idx > self.total_indices:
            end_idx = self.total_indices

        claimed = end_idx - start_idx
        if process_id is not None and claimed > 0:
            self.store.add(f"{self.process_counts_prefix}{process_id}", claimed)

        if claimed <= 0:
            return None
        return (start_idx, end_idx)

    def get_current_index(self) -> int:
        """Get the current progress of the queue."""
        return int(self.store.get(self.counter_key).decode("utf-8"))

    def get_process_counts(self) -> list[int]:
        """Get per-process claimed counts."""
        counts = []
        for pid in range(self.num_processes):
            v = int(
                self.store.get(f"{self.process_counts_prefix}{pid}").decode("utf-8")
            )
            counts.append(v)
        return counts

    def error_occurred(self) -> bool:
        """Check if an error has occurred."""
        return bool(self.store.get(self.error_key).decode("utf-8") == "1")

    def set_error_flag(self) -> None:
        """Set the error flag."""
        self.store.set(self.error_key, "1")


@dataclass
class TensorShapeDataclass:
    """Helper class for sending expected tensor shapes to distributed processes."""

    image_dft_shape: tuple[int, int]  # (H, W // 2 + 1)
    template_dft_shape: tuple[int, int, int]  # (l, h, w // 2 + 1)
    ctf_filters_shape: tuple[int, int, int, int]  # (num_Cs, num_defocus, h, w // 2 + 1)
    whitening_filter_template_shape: tuple[int, int]  # (h, w // 2 + 1)
    euler_angles_shape: tuple[int, int]  # (num_orientations, 3)
    defocus_values_shape: tuple[int]  # (num_defocus,)
    pixel_values_shape: tuple[int]  # (num_Cs,)


# pylint: disable=too-many-locals
def core_match_template_distributed(
    world_size: int,
    rank: int,
    local_rank: int,
    device: torch.device,
    orientation_batch_size: int = 1,
    num_cuda_streams: int = 1,
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
    ######################################################################
    ### Checks for proper distributed initialization and configuration ###
    ######################################################################

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

    torch.cuda.set_device(device)

    ### DEBUGGING: Print that this step has been reached
    print(f"Rank {rank} / {world_size} running on device {device} checks passed.")

    #############################################################
    ### Logic for loading / broadcasting tensors to all ranks ###
    #############################################################

    # Rank zero has all the data. No other ranks "know" the size/shape of data, so
    # first must extract the shapes before a tensor broadcast can occur.
    broadcast_list: list[Optional[TensorShapeDataclass]] = [None]
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
                f"Rank 0 missing tensor '{k}' to call core_match_template_distributed."

        # Extracting and moving all tensors to the device (zero rank only)
        assert isinstance(kwargs["image_dft"], torch.Tensor)
        assert isinstance(kwargs["template_dft"], torch.Tensor)
        assert isinstance(kwargs["ctf_filters"], torch.Tensor)
        assert isinstance(kwargs["whitening_filter_template"], torch.Tensor)
        assert isinstance(kwargs["defocus_values"], torch.Tensor)
        assert isinstance(kwargs["pixel_values"], torch.Tensor)
        assert isinstance(kwargs["euler_angles"], torch.Tensor)

        image_dft = kwargs["image_dft"].to(device)
        template_dft = kwargs["template_dft"].to(device)
        ctf_filters = kwargs["ctf_filters"].to(device)
        whitening_filter_template = kwargs["whitening_filter_template"].to(device)
        defocus_values = kwargs["defocus_values"].to(device)
        pixel_values = kwargs["pixel_values"].to(device)
        euler_angles = kwargs["euler_angles"].to(device)

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

        ### DEBUGGING: Print out the expected shapes
        print(f"Rank {rank} before broadcasting shaped: {expected_shapes}")

        broadcast_list = [expected_shapes]
        dist.broadcast_object_list(broadcast_list, src=0)

        ### DEBUGGING
        print(f"Rank {rank} finished broadcasting shapes.")

    # For all other ranks, first receive the expected shapes
    else:
        ### DEBUGGING
        print(f"non-main rank {rank} before broadcasting shapes.")

        dist.broadcast_object_list(broadcast_list, src=0)
        assert broadcast_list[0] is not None
        expected_shapes = broadcast_list[0]

        ### DEBUGGING: Print out the expected shapes
        print(f"Rank {rank} after broadcasting shapes: {expected_shapes}")

    # Now all processes have the initialized 'expected_shapes' variable. Create
    # empty tensors of the correct shape on all non-zero ranks
    if rank != 0:
        # fmt: off
        image_dft                   = torch.empty(expected_shapes.image_dft_shape,                  dtype=torch.complex64, device=device)  # noqa: E501
        template_dft                = torch.empty(expected_shapes.template_dft_shape,               dtype=torch.complex64, device=device)  # noqa: E501
        ctf_filters                 = torch.empty(expected_shapes.ctf_filters_shape,                dtype=torch.float32,   device=device)  # noqa: E501
        whitening_filter_template   = torch.empty(expected_shapes.whitening_filter_template_shape,  dtype=torch.float32,   device=device)  # noqa: E501
        euler_angles                = torch.empty(expected_shapes.euler_angles_shape,               dtype=torch.float32,   device=device)  # noqa: E501
        defocus_values              = torch.empty(expected_shapes.defocus_values_shape,             dtype=torch.float32,   device=device)  # noqa: E501
        pixel_values                = torch.empty(expected_shapes.pixel_values_shape,               dtype=torch.float32,   device=device)  # noqa: E501
        # fmt: on

    ### DEBUGGING: Print out rank and all tensor shapes before broadcasting
    print(f"Rank {rank} before broadcasting tensors:")
    print(f"  image_dft:                 {tuple(image_dft.shape)}")
    print(f"  template_dft:              {tuple(template_dft.shape)}")
    print(f"  ctf_filters:               {tuple(ctf_filters.shape)}")
    print(f"  whitening_filter_template: {tuple(whitening_filter_template.shape)}")
    print(f"  euler_angles:              {tuple(euler_angles.shape)}")
    print(f"  defocus_values:            {tuple(defocus_values.shape)}")
    print(f"  pixel_values:              {tuple(pixel_values.shape)}")

    # Now broadcast all the tensors from rank 0 to all other ranks.
    # Default is not to use async operations, so these are blocking calls.
    dist.broadcast(image_dft, src=0)
    dist.broadcast(template_dft, src=0)
    dist.broadcast(ctf_filters, src=0)
    dist.broadcast(whitening_filter_template, src=0)
    dist.broadcast(euler_angles, src=0)
    dist.broadcast(defocus_values, src=0)
    dist.broadcast(pixel_values, src=0)

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
    ### DEBUGGING: Print that the distributed queue is being created
    print(f"Rank {rank} creating distributed work index queue.")

    tcp_host_name = os.environ.get("MASTER_ADDR", None)
    tcp_host_port = os.environ.get("MASTER_PORT", None)

    assert tcp_host_name is not None, "MASTER_ADDR environment variable not set"
    assert tcp_host_port is not None, "MASTER_PORT environment variable not set"

    tcp_host_name = int(tcp_host_name)  # type: ignore[assignment]
    tcp_host_port = int(tcp_host_port)  # type: ignore[assignment]

    tcp_store = dist.TCPStore(
        host_name=tcp_host_name,
        port=tcp_host_port,
        world_size=world_size,
        is_master=(rank == 0),
        timeout=timedelta(seconds=30),  # reduce from default 300 seconds
    )

    distributed_queue = DistributedTCPIndexQueue(
        store=tcp_store,
        rank=rank,
        total_indices=euler_angles.shape[0],
        batch_size=orientation_batch_size,
        num_processes=world_size,
        prefetch_size=25,
    )

    print(f"Rank {rank} finished creating distributed work index queue.")

    ###########################################################
    ### Calling the single GPU core match template function ###
    ###########################################################

    dist.barrier()
    print(f"Rank {rank} after barrier, before calling single GPU core.")
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
            device=device,
        )
    )
    print(f"Rank {rank} after calling single GPU core.")
    dist.barrier()
    print(f"Rank {rank} after barrier, before gathering results.")

    # Gather into a list of results on the main process
    # NOTE: This is assuming there is enough GPU memory on the zeroth rank to hold.
    # There are 4 tensors each ~64 MB per GPU (~256 MB total) so this is a fair
    # assumption for most systems. Would need >= 64 GPUs to exceed 16 GB of memory.
    # TODO: Wrap this reduction into multiple groups, one per node, to reduce memory
    # pressure on the main process GPU
    # fmt: off
    if rank == 0:
        gather_mip                     = [torch.zeros_like(mip) for _ in range(world_size)]  # noqa: E501
        gather_best_global_index       = [torch.zeros_like(best_global_index) for _ in range(world_size)]  # noqa: E501
        gather_correlation_sum         = [torch.zeros_like(correlation_sum) for _ in range(world_size)]  # noqa: E501
        gather_correlation_squared_sum = [torch.zeros_like(correlation_squared_sum) for _ in range(world_size)]  # noqa: E501
    else:
        gather_mip                     = None
        gather_best_global_index       = None
        gather_correlation_sum         = None
        gather_correlation_squared_sum = None
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
