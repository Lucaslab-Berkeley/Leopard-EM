"""Utilities related to distributed computing for the backend functions."""

from multiprocessing import Manager, Process
from typing import Any, Callable, Optional

import torch.multiprocessing as mp


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


def run_multiprocess_jobs(
    target: Callable,
    kwargs_list: list[dict[str, Any]],
    extra_args: tuple[Any, ...] = (),
    extra_kwargs: Optional[dict[str, Any]] = None,
    post_start_callback: Optional[Callable] = None,
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

    Returns
    -------
    dict[Any, Any]
        Aggregated results stored in the shared dictionary.

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
    if extra_kwargs is None:
        extra_kwargs = {}

    # Manager object for shared result data as a dictionary
    manager = Manager()
    result_dict = manager.dict()
    processes: list[Process] = []

    for i, kwargs in enumerate(kwargs_list):
        args = (*extra_args, result_dict, i)

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
