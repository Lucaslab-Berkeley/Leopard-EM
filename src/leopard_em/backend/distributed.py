"""Utilities related to distributed computing for the backend functions."""

from multiprocessing import Manager, Process
from typing import Any, Callable, Optional

import torch.distributed as dist
import torch.multiprocessing as mp


# pylint: disable=too-many-instance-attributes
class SharedWorkIndexQueue:
    """Simple queue class for managing a shared index counter tracking work.

    Parameters
    ----------
    next_index : mp.Value
        A shared integer value representing the next index to be processed.
    process_counts : mp.Array
        Shared counter array tracking how many pieces of work each individual process
        has grabbed.
    num_processes : int
        The total number of processes grabbing work from this queue. Used as a way
        to track how fast each process is grabbing work from the queue
    total_indices : int
        The total number of indices (work items) to be processed. Each index is
        considered its own work item, and these items will generally batched together.
    batch_size : int
        The number of indices to be processed in each batch.
    prefetch_size : int
        The number of indices to prefetch for processing. Is a multiplicitive factor
        for batch_size. For example, if batch_size is 10 and prefetch_size is 3, then
        up to 30 indices will be prefetched for processing.
    lock : mp.Lock
        A multiprocessing lock to ensure thread-safe access to the shared index.
    """

    next_index: mp.Value
    process_counts: mp.Array
    error_flag: mp.Value
    num_processes: int
    total_indices: int
    batch_size: int
    prefetch_size: int
    lock: mp.Lock

    def __init__(
        self,
        total_indices: int,
        batch_size: int,
        num_processes: int,
        prefetch_size: int = 10,
    ):
        self.next_index = mp.Value("i", 0)  # Shared counter
        self.process_counts = mp.Array("i", [0] * num_processes)
        self.error_flag = mp.Value("i", 0)  # 0 = no error, 1 = error occurred
        self.num_processes = num_processes
        self.total_indices = total_indices
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.lock = mp.Lock()

    def get_next_indices(
        self, process_id: Optional[int] = None
    ) -> Optional[tuple[int, int]]:
        """Get the next set of indices to process returning None if all work is done.

        Parameters
        ----------
        process_id: Optional[int]
            Optional process index to use for updating the 'process_counts' array.
            Default is None which corresponds to no update.
        """
        with self.lock:
            start_idx = self.next_index.value
            if start_idx >= self.total_indices:
                return None

            # Do not go past total_indices
            end_idx = min(
                start_idx + self.batch_size * self.prefetch_size, self.total_indices
            )
            self.next_index.value = end_idx

            # Update the per-process counter
            if process_id is not None:
                self.process_counts[process_id] += end_idx - start_idx

            return (start_idx, end_idx)

    def get_current_index(self) -> int:
        """Get the current progress of the work queue (as an integer)."""
        with self.lock:
            return int(self.next_index.value)

    def get_process_counts(self) -> list[int]:
        """Get the number of indexes of work processed by each process."""
        with self.lock:
            return list(self.process_counts)

    def error_occurred(self) -> bool:
        """Check if an error has occurred in any process."""
        with self.lock:
            return bool(self.error_flag.value == 1)

    def set_error_flag(self) -> None:
        """Set the error flag to indicate an error has occurred."""
        with self.lock:
            self.error_flag.value = 1


def _rpc_get_next_indices(
    rref_queue: Any, process_id: int
) -> Optional[tuple[int, int]]:
    """Helper function for RPC call to get next indices from SharedWorkIndexQueue."""
    return rref_queue.local_value().get_next_indices(process_id)  # type: ignore[no-any-return]


def _rpc_error_occurred(rref_queue: Any) -> bool:
    """Helper function for RPC call to check if an error has occurred."""
    return rref_queue.local_value().error_occurred()  # type: ignore[no-any-return]


def _rpc_set_error_flag(rref_queue: Any) -> None:
    """Helper function for RPC call to set the error flag."""
    rref_queue.local_value().set_error_flag()


class RemoteSharedWorkIndexQueue:
    """Torch RPC wrapper around SharedWorkIndexQueue class for multi-node runs."""

    master_name: str
    rank: int
    rref_queue: Any  # torch.distributed.rpc.RRef[SharedWorkIndexQueue]

    def __init__(self, master_name: str, rank: int, index_queue: SharedWorkIndexQueue):
        self.master_name = master_name
        self.rank = rank
        self.num_processes = index_queue.num_processes
        self.rref_queue = dist.rpc.RRef(index_queue)

    def get_next_indices(self, process_id: int) -> Optional[tuple[int, int]]:
        """Get the next set of indices to process returning None if all work is done."""
        global_process_id = self.rank * self.num_processes + process_id
        result = dist.rpc.rpc_sync(
            to=self.master_name,
            func=_rpc_get_next_indices,
            args=(self.rref_queue, global_process_id),
        )
        return tuple(result) if result is not None else None

    def error_occurred(self) -> bool:
        """Check if an error has occurred in any process."""
        return bool(
            dist.rpc.rpc_sync(
                to=self.master_name, func=_rpc_error_occurred, args=(self.rref_queue,)
            )
        )

    def set_error_flag(self) -> None:
        """Set the error flag to indicate an error has occurred."""
        dist.rpc.rpc_sync(
            to=self.master_name, func=_rpc_set_error_flag, args=(self.rref_queue,)
        )


def run_multiprocess_jobs(
    target: Callable,
    kwargs_list: list[dict[str, Any]],
    extra_args: tuple[Any, ...] = (),
    extra_kwargs: Optional[dict[str, Any]] = None,
    post_start_callback: Optional[Callable] = None,
    world_size: int = 1,
    rank: int = 0,
) -> dict[Any, Any]:
    """Helper function for running multiple processes on the same target function.

    Spawns multiple processes to run the same target function with different keyword
    arguments, aggregates results in a shared dictionary, and returns them.

    Parameters
    ----------
    target : Callable
        The function that each process will execute. It must accept at least two
        positional arguments: a shared dict and a unique index.
    kwargs_list : list[dict[str, Any]]
        A list of dictionaries containing keyword arguments for each process.
    extra_args : tuple[Any, ...], optional
        Additional positional arguments to pass to the target (prepending the shared
        parameters).
    extra_kwargs : Optional[dict[str, Any]], optional
        Additional common keyword arguments for all processes.
    post_start_callback : Optional[Callable], optional
        Callback function to call after all processes have been started.
    world_size : int, optional
        The total number of nodes in a distributed setting. Default is 1 to be
        compatible with non-distributed runs.
    rank : int, optional
        The rank of the current node in a distributed setting. Default is 0 to be
        compatible with non-distributed runs.

    Returns
    -------
    dict[Any, Any]
        Aggregated results stored in the shared dictionary.

    Raises
    ------
    RuntimeError
        If any child process encounters an error.

    Example
    -------
    ```
    def worker_fn(result_dict, idx, param1, param2):
        result_dict[idx] = param1 + param2


    kwargs_per_process = [
        {"param1": 1, "param2": 2},
        {"param1": 3, "param2": 4},
    ]
    results = run_multiprocess_jobs(worker_fn, kwargs_per_process)
    print(results)
    # {0: 3, 1: 7}
    ```
    """
    _ = world_size  # Unused but kept for signature compatibility
    if extra_kwargs is None:
        extra_kwargs = {}

    # Manager object for shared result data as a dictionary
    manager = Manager()
    result_dict = manager.dict()
    processes: list[Process] = []

    for i, kwargs in enumerate(kwargs_list):
        args = (*extra_args, result_dict, i + rank * len(kwargs_list))

        # Merge per-process kwargs with common kwargs.
        proc_kwargs = {**extra_kwargs, **kwargs}
        p = Process(target=target, args=args, kwargs=proc_kwargs)
        processes.append(p)
        p.start()

    if post_start_callback is not None:
        post_start_callback()

    for p in processes:
        p.join()

    return dict(result_dict)
