"""Tests for the multiprocessing helpers in ``leopard_em.backend.distributed``."""

import pytest
import torch

from leopard_em.backend.distributed import run_device_jobs, run_multiprocess_jobs


def _worker(result_dict, rank, fail):
    """Module-level so spawned children can import it."""
    if fail:
        raise RuntimeError("simulated worker failure")
    result_dict[rank] = rank


def test_run_multiprocess_jobs_raises_on_failed_worker():
    """A crashed worker must raise instead of silently dropping its results."""
    with pytest.raises(RuntimeError, match=r"\(1, 1\)"):
        run_multiprocess_jobs(_worker, [{"fail": False}, {"fail": True}])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs 2+ CUDA devices to tell the caller's device from the worker's",
)
def test_run_device_jobs_restores_cuda_device_when_worker_raises():
    """The caller's current device is restored even if the in-process job fails."""

    def _sets_device_then_fails(result_dict, rank, device):
        torch.cuda.set_device(device)
        raise RuntimeError("simulated worker failure")

    torch.cuda.set_device(0)
    with pytest.raises(RuntimeError, match="simulated worker failure"):
        run_device_jobs(_sets_device_then_fails, [{"device": torch.device("cuda", 1)}])

    assert torch.cuda.current_device() == 0
