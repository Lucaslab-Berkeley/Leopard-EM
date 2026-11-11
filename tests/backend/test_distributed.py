"""Tests for the multiprocessing helpers in ``leopard_em.backend.distributed``."""

import pytest

from leopard_em.backend.distributed import run_multiprocess_jobs


def _worker(result_dict, rank, fail):
    """Module-level so spawned children can import it."""
    if fail:
        raise RuntimeError("simulated worker failure")
    result_dict[rank] = rank


def test_run_multiprocess_jobs_raises_on_failed_worker():
    """A crashed worker must raise instead of silently dropping its results."""
    with pytest.raises(RuntimeError, match=r"\(1, 1\)"):
        run_multiprocess_jobs(_worker, [{"fail": False}, {"fail": True}])
