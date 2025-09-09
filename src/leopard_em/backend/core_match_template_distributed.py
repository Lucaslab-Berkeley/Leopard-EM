"""Distributed multi-node version of the core match_template implementation."""

from typing import Optional

import torch
import torch.distributed as dist

from leopard_em.backend.core_match_template import (
    _core_match_template_single_gpu,
)
from leopard_em.backend.distributed import (
    SharedWorkIndexQueue,
)
from leopard_em.backend.process_results import (
    aggregate_distributed_results,
    decode_global_search_index,
    scale_mip,
)

# Global queue variable for inter-node communication
_global_queue: Optional[SharedWorkIndexQueue] = None


# Wrapper functions for the global queue for RPC calls
def initialize_global_queue(
    total_indices: int, batch_size: int, prefetch_size: int, num_processes: int
) -> None:
    """Initialize the global SharedWorkIndexQueue for distributed processing."""
    global _global_queue
    if _global_queue is not None:
        raise RuntimeError("Global queue has already been initialized.")
    _global_queue = SharedWorkIndexQueue(
        total_indices=total_indices,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        num_processes=num_processes,
    )


def remote_get_next_indices(
    process_id: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    """Remote function to get next indices from the queue."""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError("Queue not initialized on main process")
    return _global_queue.get_next_indices(process_id)


def remote_get_current_index() -> int:
    """Remote function to get current index from the queue."""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError("Queue not initialized on main process")
    return _global_queue.get_current_index()


def remote_get_process_counts() -> list[int]:
    """Remote function to get process counts from the queue."""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError("Queue not initialized on main process")
    return _global_queue.get_process_counts()


def remote_error_occurred() -> bool:
    """Remote function to check if error occurred in the queue."""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError("Queue not initialized on main process")
    return _global_queue.error_occurred()


def remote_set_error_flag() -> None:
    """Remote function to set error flag in the queue."""
    global _global_queue
    if _global_queue is None:
        raise RuntimeError("Queue not initialized on main process")
    _global_queue.set_error_flag()


class DistributedWorkIndexQueue:
    """RPC wrapper around SharedWorkIndexQueue for multi-node runs."""

    main_name: str
    rank: int
    world_size: int

    def __init__(self, main_name: str, rank: int, world_size: int):
        self.main_name = main_name
        self.rank = rank
        self.world_size = world_size

    def get_next_indices(self, process_id: int) -> Optional[tuple[int, int]]:
        """Get the next set of indices to process returning None if all work is done."""
        result = dist.rpc.rpc_sync(
            to=self.main_name,
            func=remote_get_next_indices,
            args=(process_id,),
        )
        return result  # type: ignore[no-any-return]

    def get_current_index(self) -> int:
        """Get the current global index being processed."""
        result = dist.rpc.rpc_sync(
            to=self.main_name,
            func=remote_get_current_index,
            args=(),
        )
        return result  # type: ignore[no-any-return]

    def get_process_counts(self) -> list[int]:
        """Get the current process counts from the main process."""
        result = dist.rpc.rpc_sync(
            to=self.main_name,
            func=remote_get_process_counts,
            args=(),
        )
        return result  # type: ignore[no-any-return]

    def error_occurred(self) -> bool:
        """Check if an error has occurred in any process."""
        return bool(
            dist.rpc.rpc_sync(to=self.main_name, func=remote_error_occurred, args=())
        )

    def set_error_flag(self) -> None:
        """Set the error flag to indicate an error has occurred."""
        dist.rpc.rpc_sync(to=self.main_name, func=remote_set_error_flag, args=())


# pylint: disable=too-many-locals
def core_match_template_distributed(
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
    """Distributed multi-node core function for the match template program.

    See `core_match_template` for parameter descriptions and return values.
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

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size < 2:
        raise ValueError(
            "Distributed core_match_template_distributed called with world_size < 2. "
            "Did you mean to call the non-distributed core_match_template function?"
        )

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

    # Each GPU has its own task associated with it, so split the list of devices
    if isinstance(device, list):
        device = device[rank % len(device)]

    dist.rpc.init_rpc(name=f"worker{rank}", rank=rank, world_size=world_size)

    # Only initialize the global queue on the main process
    if rank == 0:
        initialize_global_queue(
            total_indices=euler_angles.shape[0],
            batch_size=orientation_batch_size,
            prefetch_size=10,
            num_processes=world_size,
        )

    distributed_queue = DistributedWorkIndexQueue(
        main_name="worker0", rank=rank, world_size=world_size
    )

    ##################################################
    ### Arguments for the per-GPU worker processes ###
    ##################################################

    dist.barrier()
    single_gpu_kwargs = {
        "rank": rank,
        "index_queue": distributed_queue,
        "image_dft": image_dft,
        "template_dft": template_dft,
        "euler_angles": euler_angles,
        "projective_filters": projective_filters,
        "defocus_values": defocus_values,
        "pixel_values": pixel_values,
        "orientation_batch_size": orientation_batch_size,
        "num_cuda_streams": num_cuda_streams,
        "device": device,
    }

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

    ##################################################################
    ### Call the single-GPU function on each process independently ###
    ##################################################################

    (mip, best_global_index, correlation_sum, correlation_squared_sum) = (
        _core_match_template_single_gpu(**single_gpu_kwargs)
    )

    dist.barrier()

    # Gather into a list of results on the main process
    # NOTE: This is assuming there is enough GPU memory on the zeroth rank to hold.
    # There are 4 tensors each ~64 MB per GPU (~256 MB total) so this is a fair
    # assumption for most systems.
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
