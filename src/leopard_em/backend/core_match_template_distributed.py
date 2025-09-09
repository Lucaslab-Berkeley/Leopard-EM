"""Distributed multi-node version of the core match_template implementation."""

import threading
from dataclasses import dataclass
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

_global_queue: "DistributedWorkIndexQueue | None" = None

###########################################################
### Wrapper functions for RPC calls to the main process ###
###########################################################


def _rpc_get_next_indices(process_id: int) -> Optional[tuple[int, int]]:
    """RPC function to get the next set of indices to process."""
    global _global_queue
    assert _global_queue is not None
    return _global_queue.get_next_indices(process_id)


def _rpc_get_current_index() -> int:
    """RPC function to get the current global index being processed."""
    global _global_queue
    assert _global_queue is not None
    return _global_queue.get_current_index()


def _rpc_get_process_counts() -> list[int]:
    """RPC function to get the current process counts from the main process."""
    global _global_queue
    assert _global_queue is not None
    return _global_queue.get_process_counts()


def _rpc_error_occurred() -> bool:
    """RPC function to check if an error has occurred in any process."""
    global _global_queue
    assert _global_queue is not None
    return _global_queue.error_occurred()


def _rpc_set_error_flag() -> None:
    """RPC function to set the error flag to indicate an error has occurred."""
    global _global_queue
    assert _global_queue is not None
    _global_queue.set_error_flag()


class DistributedWorkIndexQueue:
    """Distributed work manager for coordinating search indices across nodes.

    This class assumes that RPC has already been initialized and the main process
    (rank 0) is running on the node with name "worker0".

    Parameters
    ----------
    total_indices: int
        Total number of indices to process.
    batch_size: int
        Number of indices to process in a single batch.
    num_processes: int
        Total number of processes across all nodes.
    prefetch_size: int
        Multiplicative factor for batch_size to determine how many indices to
        prefetch at once.
    rank: int
        Global rank of this process.
    is_main: bool
        Whether this process is the main process (rank 0).
    lock: threading.Lock
        Lock for synchronizing access to shared state.
    """

    next_index: int
    process_counts: list[int]
    error_flag: bool  # True = error occurred
    num_processes: int
    total_indices: int
    batch_size: int
    prefetch_size: int  # multiplicative factor for batch_size
    lock: threading.Lock

    rank: int
    is_main: bool

    def __init__(
        self,
        total_indices: int,
        batch_size: int,
        num_processes: int,
        prefetch_size: int,
        rank: int,
    ):
        self.next_index = 0
        self.process_counts = [0] * num_processes
        self.error_flag = False
        self.num_processes = num_processes
        self.total_indices = total_indices
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.lock: threading.Lock

        # Specific to distributed version
        self.rank = rank
        self.is_main = rank == 0

    def get_next_indices(self, process_id: int) -> Optional[tuple[int, int]]:
        """Get the next set of indices to process returning None if all work is done.

        NOTE: This method ignores all incoming arguments and instead passes the process
        rank to the main process through RPC.
        """
        if not self.is_main:
            return dist.rpc.rpc_sync(
                to="worker0",
                func=_rpc_get_next_indices,
                args=(self.rank,),
            )

        with self.lock:
            process_id = self.rank  # For zeroth rank, use its own rank
            start_idx = self.next_index
            if start_idx >= self.total_indices:
                return None

            # Do not go past total_indices
            end_idx = min(
                start_idx + self.batch_size * self.prefetch_size, self.total_indices
            )
            self.next_index = end_idx

            # Update the per-process counter
            self.process_counts[process_id] += end_idx - start_idx

            return (start_idx, end_idx)

    def get_current_index(self) -> int:
        """Get the current global index being processed."""
        if not self.is_main:
            return dist.rpc.rpc_sync(
                to="worker0",
                func=_rpc_get_current_index,
                args=(),
            )

        with self.lock:
            return self.next_index

    def get_process_counts(self) -> list[int]:
        """Get the current process counts from the main process."""
        if not self.is_main:
            return dist.rpc.rpc_sync(
                to="worker0",
                func=_rpc_get_process_counts,
                args=(),
            )

        with self.lock:
            return self.process_counts.copy()

    def error_occurred(self) -> bool:
        """Check if an error has occurred in any process."""
        if not self.is_main:
            return dist.rpc.rpc_sync(
                to="worker0",
                func=_rpc_error_occurred,
                args=(),
            )

        with self.lock:
            return self.error_flag

    def set_error_flag(self) -> None:
        """Set the error flag to indicate an error has occurred."""
        if not self.is_main:
            dist.rpc.rpc_sync(
                to="worker0",
                func=_rpc_set_error_flag,
                args=(),
            )
            return

        with self.lock:
            self.error_flag = True


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

    #############################################################
    ### Logic for loading / broadcasting tensors to all ranks ###
    #############################################################

    # Rank zero has all the data. No other ranks "know" the size/shape of data, so
    # first must extract the shapes before a tensor broadcast can occur.
    broadcast_list: list[Optional[TensorShapeDataclass]] = [None]
    if rank == 0:
        try:
            image_dft = kwargs["image_dft"].to(device)
            template_dft = kwargs["template_dft"].to(device)
            ctf_filters = kwargs["ctf_filters"].to(device)
            whitening_filter_template = kwargs["whitening_filter_template"].to(device)
            defocus_values = kwargs["defocus_values"].to(device)
            pixel_values = kwargs["pixel_values"].to(device)
            euler_angles = kwargs["euler_angles"].to(device)
        except KeyError as e:
            raise KeyError(
                f"Rank 0 missing some tensors to call core_match_template_distributed; "
                f"missing key: {e.args[0]}"
            ) from e

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
        image_dft                   = torch.empty(expected_shapes.image_dft_shape,                  dtype=torch.complex64, device=device)  # noqa: E501
        template_dft                = torch.empty(expected_shapes.template_dft_shape,               dtype=torch.complex64, device=device)  # noqa: E501
        ctf_filters                 = torch.empty(expected_shapes.ctf_filters_shape,                dtype=torch.complex64, device=device)  # noqa: E501
        whitening_filter_template   = torch.empty(expected_shapes.whitening_filter_template_shape,  dtype=torch.float32,   device=device)  # noqa: E501
        euler_angles                = torch.empty(expected_shapes.euler_angles_shape,               dtype=torch.float32,   device=device)  # noqa: E501
        defocus_values              = torch.empty(expected_shapes.defocus_values_shape,             dtype=torch.float32,   device=device)  # noqa: E501
        pixel_values                = torch.empty(expected_shapes.pixel_values_shape,               dtype=torch.float32,   device=device)  # noqa: E501
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

    ##############################################################
    ### Pre-multiply the whitening filter with the CTF filters ###
    ##############################################################

    projective_filters = ctf_filters * whitening_filter_template[None, None, ...]
    total_projections = (
        euler_angles.shape[0] * defocus_values.shape[0] * pixel_values.shape[0]
    )

    ########################################################
    ### RPC Setup for distributed index queue management ###
    ########################################################

    dist.rpc.init_rpc(name=f"worker{rank}", rank=rank, world_size=world_size)

    distributed_queue = DistributedWorkIndexQueue(
        total_indices=euler_angles.shape[0],
        batch_size=orientation_batch_size,
        num_processes=world_size,
        prefetch_size=10,
        rank=rank,
    )
    if rank == 0:
        global _global_queue
        _global_queue = distributed_queue

    # TODO: Progress bar monitoring (and possibly another thread in main process)
    # for handling queue monitoring?

    ###########################################################
    ### Calling the single GPU core match template function ###
    ###########################################################

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

    dist.barrier()

    # ###############################################
    # ### Progress tracking setup on main process ###
    # ###############################################

    # if rank == 0:
    #     global _global_queue
    #     global_pbar, device_pbars = setup_progress_tracking(
    #         index_queue=_global_queue,
    #         unit_scale=defocus_values.shape[0] * pixel_values.shape[0],
    #         num_devices=len(device) * world_size,
    #     )
    #     progress_callback = partial(
    #         monitor_match_template_progress,
    #         queue=_global_queue,
    #         pbar=global_pbar,
    #         device_pbars=device_pbars,
    #     )
    # else:
    #     progress_callback = None

    # Gather into a list of results on the main process
    # NOTE: This is assuming there is enough GPU memory on the zeroth rank to hold.
    # There are 4 tensors each ~64 MB per GPU (~256 MB total) so this is a fair
    # assumption for most systems. Would need >= 64 GPUs to exceed 16 GB of memory.
    # TODO: Wrap this reduction into multiple groups, one per node, to reduce memory
    # pressure on the main process GPU
    # fmt: off
    # Pre-commit: ignore line-too-long (E501) for the following 10 lines.
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

    # Shutdown the RPC framework
    dist.rpc.shutdown()

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
