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
    total_indices: int
    batch_size: int
    prefetch_size: int
    lock: mp.Lock

    def __init__(self, total_indices: int, batch_size: int, prefetch_size: int = 10):
        self.next_index = mp.Value("i", 0)  # Shared counter
        self.total_indices = total_indices
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.lock = mp.Lock()

    def get_next_indices(self) -> Optional[tuple[int, int]]:
        """Get the next set of indices to process returning None if all work is done."""
        with self.lock:
            start_idx = self.next_index.value
            if start_idx >= self.total_indices:
                return None

            end_idx = min(
                start_idx + self.batch_size * self.prefetch_size, self.total_indices
            )
            self.next_index.value = end_idx

            return (start_idx, end_idx)

    def get_progress(self) -> float:
        """Get the current progress of the work queue as a fraction."""
        with self.lock:
            return float(self.next_index / self.total_indices)


def run_multiprocess_jobs(
    target: Callable,
    kwargs_list: list[dict[str, Any]],
    extra_args: tuple[Any, ...] = (),
    extra_kwargs: Optional[dict[str, Any]] = None,
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

    for p in processes:
        p.join()

    return dict(result_dict)
