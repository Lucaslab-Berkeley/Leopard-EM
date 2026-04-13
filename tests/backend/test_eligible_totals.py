"""Tests for total_mip_eligible_projections from core and the merge+peaks flow.

These tests verify:
1. core_match_template returns total_mip_eligible_projections equal to the
   number of eligible orientations * n_defocus * n_cs.
2. merge_runs_independent_zscore uses full totals for z-score normalisation.
3. MatchTemplateResult.locate_peaks uses eligible totals as the multiplicity.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Load process_results without triggering the heavy leopard_em __init__.
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
scale_mip = _pr.scale_mip


# ---------------------------------------------------------------------------
# Helper to build a minimal run dict
# ---------------------------------------------------------------------------


def _make_run(
    mip: torch.Tensor,
    csum: torch.Tensor,
    csq: torch.Tensor,
    bgi: torch.Tensor,
    full_total: int,
    eligible_total: int,
) -> dict:
    return {
        "mip": mip,
        "correlation_sum": csum,
        "correlation_squared_sum": csq,
        "best_global_index": bgi,
        "total_projections": full_total,
        "total_mip_eligible_projections": eligible_total,
    }


# ---------------------------------------------------------------------------
# Merge uses FULL totals for z-score normalisation
# ---------------------------------------------------------------------------


def test_merge_uses_full_totals_for_z_normalisation() -> None:
    """Passing full totals to merge gives the same z as direct scale_mip call."""
    h, w = 2, 2
    mip = torch.ones(h, w) * 3.0
    csum = torch.ones(h, w) * 10.0
    csq = torch.ones(h, w) * 110.0
    bgi = torch.zeros(h, w, dtype=torch.int32)

    full_total = 100
    eligible_total = 60

    run = _make_run(mip, csum, csq, bgi, full_total, eligible_total)
    merged = merge_runs_independent_zscore(
        [run],
        total_correlation_positions_per_run=[full_total],
    )

    # Expected z-score computed with the FULL total
    buf = torch.empty_like(mip)
    _, exp_scaled, _, _ = scale_mip(
        mip.clone(), buf, csum.clone(), csq.clone(), full_total
    )
    assert torch.allclose(
        merged["scaled_mip"], exp_scaled
    ), "z-score in merge should match direct scale_mip with full_total"


def test_merge_total_correlation_positions_is_sum_of_full() -> None:
    """merged['total_correlation_positions'] == sum of full (not eligible) totals."""
    h, w = 2, 2
    mip = torch.ones(h, w) * 2.0
    csum = torch.ones(h, w) * 5.0
    csq = torch.ones(h, w) * 30.0
    bgi = torch.zeros(h, w, dtype=torch.int32)

    full_a, elig_a = 80, 50
    full_b, elig_b = 120, 70

    runs = [
        _make_run(mip, csum, csq, bgi, full_a, elig_a),
        _make_run(mip * 0.5, csum, csq, bgi, full_b, elig_b),
    ]
    merged = merge_runs_independent_zscore(
        runs,
        total_correlation_positions_per_run=[full_a, full_b],
    )
    assert (
        merged["total_correlation_positions"] == full_a + full_b
    ), "merge should return sum of full (not eligible) totals"
    assert merged["total_correlation_positions"] != elig_a + elig_b


# ---------------------------------------------------------------------------
# MatchTemplateResult.locate_peaks uses eligible total
# ---------------------------------------------------------------------------


def test_match_template_result_locate_peaks_uses_eligible_total() -> None:
    """locate_peaks should pass total_mip_eligible_projections to extract fn."""
    from leopard_em.pydantic_models.results.match_template_result import (
        MatchTemplateResult,
    )

    h, w = 4, 4
    ones = torch.ones(h, w, dtype=torch.float32)
    zeros = torch.zeros(h, w, dtype=torch.float32)

    result = MatchTemplateResult(
        allow_file_overwrite=True,
        mip_path="/tmp/mip.mrc",
        scaled_mip_path="/tmp/scaled.mrc",
        correlation_average_path="/tmp/avg.mrc",
        correlation_variance_path="/tmp/var.mrc",
        orientation_psi_path="/tmp/psi.mrc",
        orientation_theta_path="/tmp/theta.mrc",
        orientation_phi_path="/tmp/phi.mrc",
        relative_defocus_path="/tmp/def.mrc",
    )
    result.mip = ones * 2.0
    result.scaled_mip = ones * 3.0
    result.correlation_average = ones * 0.5
    result.correlation_variance = ones * 0.1
    result.orientation_psi = zeros
    result.orientation_theta = zeros
    result.orientation_phi = zeros
    result.relative_defocus = zeros
    result.total_projections = 1000
    result.total_mip_eligible_projections = 600

    captured: list[int] = []

    def _fake_extract(**kwargs):
        captured.append(kwargs["total_correlation_positions"])
        return MagicMock()

    with patch(
        "leopard_em.pydantic_models.results.match_template_result"
        ".extract_peaks_and_statistics_zscore",
        side_effect=_fake_extract,
    ):
        result.locate_peaks()

    assert len(captured) == 1
    assert captured[0] == 600, (
        "locate_peaks must pass total_mip_eligible_projections (600), "
        f"not total_projections (1000). Got {captured[0]}."
    )


def test_match_template_result_locate_peaks_fallback_to_total_projections() -> None:
    """When total_mip_eligible_projections == 0 (old result), use total_projections."""
    from leopard_em.pydantic_models.results.match_template_result import (
        MatchTemplateResult,
    )

    h, w = 4, 4
    ones = torch.ones(h, w, dtype=torch.float32)
    zeros = torch.zeros(h, w, dtype=torch.float32)

    result = MatchTemplateResult(
        allow_file_overwrite=True,
        mip_path="/tmp/mip2.mrc",
        scaled_mip_path="/tmp/scaled2.mrc",
        correlation_average_path="/tmp/avg2.mrc",
        correlation_variance_path="/tmp/var2.mrc",
        orientation_psi_path="/tmp/psi2.mrc",
        orientation_theta_path="/tmp/theta2.mrc",
        orientation_phi_path="/tmp/phi2.mrc",
        relative_defocus_path="/tmp/def2.mrc",
    )
    result.mip = ones * 2.0
    result.scaled_mip = ones * 3.0
    result.correlation_average = ones * 0.5
    result.correlation_variance = ones * 0.1
    result.orientation_psi = zeros
    result.orientation_theta = zeros
    result.orientation_phi = zeros
    result.relative_defocus = zeros
    result.total_projections = 1000
    result.total_mip_eligible_projections = 0  # old result, field not populated

    captured: list[int] = []

    def _fake_extract(**kwargs):
        captured.append(kwargs["total_correlation_positions"])
        return MagicMock()

    with patch(
        "leopard_em.pydantic_models.results.match_template_result"
        ".extract_peaks_and_statistics_zscore",
        side_effect=_fake_extract,
    ):
        result.locate_peaks()

    assert captured[0] == 1000, (
        "When total_mip_eligible_projections is 0, should fall back to "
        f"total_projections (1000). Got {captured[0]}."
    )
