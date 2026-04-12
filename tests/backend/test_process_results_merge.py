"""Tests for merge_runs_pooled_zscore and merge_runs_independent_zscore."""

import importlib.util
from pathlib import Path

import torch

# Load ``process_results`` without importing ``leopard_em.backend`` (heavy __init__).
_pr_path = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "leopard_em"
    / "backend"
    / "process_results.py"
)
_spec = importlib.util.spec_from_file_location(
    "leopard_em.backend.process_results", _pr_path
)
assert _spec and _spec.loader
_pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pr)
merge_runs_independent_zscore = _pr.merge_runs_independent_zscore
merge_runs_pooled_zscore = _pr.merge_runs_pooled_zscore
scale_mip = _pr.scale_mip


def test_merge_runs_pooled_matches_single_scale_mip() -> None:
    """One run: pooled merge should match direct scale_mip."""
    mip = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    csum = torch.ones_like(mip) * 8.0
    csq = torch.ones_like(mip) * 64.0
    bgi = torch.zeros_like(mip, dtype=torch.int32)
    runs = [
        {
            "mip": mip,
            "best_global_index": bgi,
            "correlation_sum": csum,
            "correlation_squared_sum": csq,
        }
    ]
    total = 100
    out = merge_runs_pooled_zscore(
        runs,
        pooled_total_correlation_positions=total,
        run_tag_values=[1.5],
    )
    buf = torch.empty_like(mip)
    _, exp_scaled, exp_mean, exp_var = scale_mip(
        mip.clone(), buf, csum.clone(), csq.clone(), total
    )
    assert torch.allclose(out["scaled_mip"], exp_scaled)
    assert torch.allclose(out["correlation_mean"], exp_mean)
    assert torch.allclose(out["correlation_variance"], exp_var)
    assert torch.allclose(out["best_run_tag"], torch.full_like(mip, 1.5))


def test_merge_runs_independent_picks_higher_z() -> None:
    """Two runs: winner per pixel follows larger per-run z-score."""
    h, w = 2, 2
    mip0 = torch.zeros(h, w, dtype=torch.float32)
    mip1 = torch.zeros(h, w, dtype=torch.float32)
    mip1[0, 0] = 10.0
    n = 100
    # Sums implying ~unit std per pixel so z-scores differ when mip differs.
    csum0 = torch.zeros(h, w, dtype=torch.float32)
    csq0 = torch.ones(h, w, dtype=torch.float32) * float(n)
    csum1 = csum0.clone()
    csq1 = csq0.clone()
    bgi0 = torch.zeros((h, w), dtype=torch.int32)
    bgi1 = torch.ones((h, w), dtype=torch.int32)
    runs = [
        {
            "mip": mip0,
            "best_global_index": bgi0,
            "correlation_sum": csum0,
            "correlation_squared_sum": csq0,
        },
        {
            "mip": mip1,
            "best_global_index": bgi1,
            "correlation_sum": csum1,
            "correlation_squared_sum": csq1,
        },
    ]
    out = merge_runs_independent_zscore(
        runs,
        total_correlation_positions_per_run=n,
    )
    assert out["winner_run_index"][0, 0].item() == 1
    assert (out["winner_run_index"][0, 1:] == 0).all()
    assert out["best_global_index"][0, 0].item() == 1
    # Full-search multiplicity: sum of per-run correlation counts (2 * 100 here).
    assert out["total_correlation_positions"] == 200


def test_merge_runs_pooled_two_planes_sums_correlation() -> None:
    """Pooled z uses summed correlation statistics across two identical mips."""
    mip = torch.ones(1, 1, dtype=torch.float32)
    csum = torch.ones_like(mip) * 2.0
    csq = torch.ones_like(mip) * 4.0
    bgi = torch.zeros_like(mip, dtype=torch.int32)
    run = {
        "mip": mip,
        "best_global_index": bgi,
        "correlation_sum": csum,
        "correlation_squared_sum": csq,
    }
    out = merge_runs_pooled_zscore(
        [run, run],
        pooled_total_correlation_positions=20,
    )
    buf = torch.empty_like(mip)
    mip2, z2, _, _ = scale_mip(
        mip.clone(),
        buf,
        csum * 2,
        csq * 2,
        total_correlation_positions=20,
    )
    assert torch.allclose(out["mip"], mip2)
    assert torch.allclose(out["scaled_mip"], z2)


def test_merge_runs_independent_total_sum_per_run_sequence() -> None:
    """total_correlation_positions sums heterogeneous per-run denominators."""
    mip = torch.zeros(1, 1, dtype=torch.float32)
    csum = torch.zeros_like(mip)
    csq = torch.ones_like(mip) * 10.0
    bgi = torch.zeros_like(mip, dtype=torch.int32)
    run = {
        "mip": mip,
        "best_global_index": bgi,
        "correlation_sum": csum,
        "correlation_squared_sum": csq,
    }
    out = merge_runs_independent_zscore(
        [run, run, run],
        total_correlation_positions_per_run=[10, 20, 30],
    )
    assert out["total_correlation_positions"] == 60
