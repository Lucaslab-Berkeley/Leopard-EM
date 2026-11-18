"""run_multiprocess_jobs: failed workers raise instead of silently dropping results.

Each call spawns processes that import torch, so the cases are grouped into few calls.
"""

import os

import pytest

from leopard_em.backend.distributed import run_multiprocess_jobs


# Module level so spawned child processes can import it
def _worker(result_dict, rank, behaviour="ok"):
    if behaviour == "raise":
        raise ValueError(f"simulated failure on rank {rank}")
    if behaviour == "exit":
        os._exit(3)  # dies without running exception handlers
    if behaviour == "no_result":
        return
    result_dict[rank] = rank * 10


def test_successful_workers_return_all_results_by_rank():
    results = run_multiprocess_jobs(_worker, [{}, {}, {}], ranks=[0, 4, 7])
    assert results == {0: 0, 4: 40, 7: 70}


def test_failed_workers_raise_with_every_reason():
    kwargs_list = [
        {},  # succeeds
        {"behaviour": "raise"},
        {"behaviour": "exit"},
        {"behaviour": "no_result"},
    ]
    with pytest.raises(RuntimeError) as excinfo:
        run_multiprocess_jobs(_worker, kwargs_list)

    message = str(excinfo.value)
    assert "3 of 4 worker processes failed" in message
    assert "process 0" not in message
    assert "process 1 raised an exception" in message
    assert "ValueError: simulated failure on rank 1" in message  # child traceback
    assert "process 2 exited with code 3" in message
    assert "process 3 exited without storing a result" in message
