"""Utilities related to distributed computing for the backend functions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Manager, Process
from typing import Any, Callable, Optional

import torch.distributed as dist
import torch.multiprocessing as mp


class WorkIndexQueue(ABC):
    """Abstract base class for index queues that manage distributed work allocation.

    This class defines the common interface for both single-node multiprocessing
    and multi-node distributed computing scenarios.

    Parameters
    ----------
    total_indices : int
        The total number of indices (work items) to be processed.
    batch_size : int
        The number of indices to be processed in each batch.
    num_processes : int
        The total number of processes grabbing work from this queue.
    prefetch_size : int
        The number of indices to prefetch for processing (multiplicative factor
        for batch_size).
    """

    def __init__(
        self,
        total_indices: int,
        batch_size: int,
        num_processes: int,
        prefetch_size: int = 10,
    ):
        self.total_indices = total_indices
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.prefetch_size = prefetch_size

    @abstractmethod
    def get_next_indices(
        self, process_id: Optional[int] = None
    ) -> Optional[tuple[int, int]]:
        """Get the next set of indices to process, returning None if all work is done.

        Parameters
        ----------
        process_id : Optional[int]
            Optional process index for updating per-process counters.

        Returns
        -------
        Optional[tuple[int, int]]
            Tuple of (start_idx, end_idx) or None if no work remains.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_index(self) -> int:
        """Get the current progress of the work queue."""
        raise NotImplementedError

    @abstractmethod
    def get_process_counts(self) -> list[int]:
        """Get per-process work counts."""
        raise NotImplementedError

    @abstractmethod
    def error_occurred(self) -> bool:
        """Check if an error has occurred in any process."""
        raise NotImplementedError

    @abstractmethod
    def set_error_flag(self) -> None:
        """Set the error flag to indicate an error has occurred."""
        raise NotImplementedError


# pylint: disable=too-many-instance-attributes
class MultiprocessWorkIndexQueue(WorkIndexQueue):
    """Single-node (distributed memory) multiprocessing work index queue.

    Uses multiprocessing primitives for shared state management within a single machine
    by using shared memory.

    Parameters
    ----------
    total_indices : int
        The total number of indices (work items) to be processed.
    batch_size : int
        The number of indices to be processed in each batch.
    num_processes : int
        The total number of processes grabbing work from this queue.
    prefetch_size : int, optional
        The number of indices to prefetch for processing, by default 10.
    """

    def __init__(
        self,
        total_indices: int,
        batch_size: int,
        num_processes: int,
        prefetch_size: int = 10,
    ):
        super().__init__(total_indices, batch_size, num_processes, prefetch_size)
        self.next_index = mp.Value("i", 0)  # Shared counter
        self.process_counts = mp.Array("i", [0] * num_processes)
        self.error_flag = mp.Value("i", 0)  # 0 = no error, 1 = error occurred
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


class DistributedTCPIndexQueue(WorkIndexQueue):
    """Distributed work index queue backed by torch.distributed.TCPStore.

    Drop-in replacement for MultiprocessWorkIndexQueue but for multi-node setups.

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
        total_indices: int,
        batch_size: int,
        num_processes: int,
        prefetch_size: int = 10,
        counter_key: str = "next_index",
        error_key: str = "error_flag",
        process_counts_prefix: str = "process_count_",
    ):
        super().__init__(total_indices, batch_size, num_processes, prefetch_size)

        self.store = store
        self.counter_key = counter_key
        self.error_key = error_key
        self.process_counts_prefix = process_counts_prefix

    @staticmethod
    def initialize_store(
        store: dist.TCPStore,
        rank: int,
        num_processes: int,
        counter_key: str = "next_index",
        error_key: str = "error_flag",
        process_counts_prefix: str = "process_count_",
    ) -> None:
        """Have rank 0 initialize the shared keys in the store.

        NOTE: Includes a synchronization barrier so MUST be called by all processes.
        """
        if rank == 0:
            # set keys unconditionally on the server to avoid compare_set races
            store.set(counter_key, "0")
            store.set(error_key, "0")
            for pid in range(num_processes):
                store.set(f"{process_counts_prefix}{pid}", "0")
        # synchronize so other ranks can safely call store.get()/add()
        dist.barrier()

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

        end_idx = min(end_idx, self.total_indices)

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


def run_multiprocess_jobs(
    target: Callable,
    kwargs_list: list[dict[str, Any]],
    extra_args: tuple[Any, ...] = (),
    extra_kwargs: Optional[dict[str, Any]] = None,
    post_start_callback: Optional[Callable] = None,
    ranks: Optional[list[int]] = None,
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
    ranks : list[int], optional
        If not None, then pass these integers as the ranks to the processes. Otherwise,
        pass the the index of the kwargs_list (default). Must be the same length as
        kwargs_list.

    Returns
    -------
    dict[Any, Any]
        Aggregated results stored in the shared dictionary.

    Raises
    ------
    RuntimeError
        If any child process encounters an error.
    ValueError
        If ranks is not None and its length does not match kwargs_list.

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
    if ranks is not None:
        if len(kwargs_list) != len(ranks):
            raise ValueError("Length of ranks must match length of kwargs_list.")
    else:
        ranks = list(range(len(kwargs_list)))

    if extra_kwargs is None:
        extra_kwargs = {}

    # Manager object for shared result data as a dictionary
    manager = Manager()
    result_dict = manager.dict()
    processes: list[Process] = []

    for rank, kwargs in zip(ranks, kwargs_list):
        args = (*extra_args, result_dict, rank)

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
