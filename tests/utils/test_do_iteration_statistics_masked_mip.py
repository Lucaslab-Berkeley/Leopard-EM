"""Tests for masked-MIP iteration statistics (utils)."""

import pytest
import torch

from leopard_em.backend.core_match_template import (
    resolve_match_template_backend,
    validate_orientation_eligible_for_match_template,
)
from leopard_em.backend.utils import do_iteration_statistics_updates_masked_mip


def test_resolve_match_template_backend() -> None:
    assert resolve_match_template_backend("streamed") == ("streamed", False)
    assert resolve_match_template_backend("batched") == ("batched", False)
    assert resolve_match_template_backend("streamed_masked_mip") == ("streamed", True)
    assert resolve_match_template_backend("batched_masked_mip") == ("batched", True)


def test_masked_mip_max_ignores_ineligible_high_value() -> None:
    """Ineligible row has higher CC but must not win MIP; sums include both rows."""
    cross = torch.zeros(1, 1, 2, 1, 1)
    cross[0, 0, 0, 0, 0] = 5.0
    cross[0, 0, 1, 0, 0] = 2.0

    mip = torch.full((1, 1), float("-inf"))
    best = torch.full((1, 1), -1, dtype=torch.int32)
    corr_sum = torch.zeros(1, 1)
    corr_sq = torch.zeros(1, 1)
    orient_eligible = torch.tensor([0.0, 1.0])
    current_indexes = torch.tensor([10, 11], dtype=torch.int32)

    do_iteration_statistics_updates_masked_mip(
        cross_correlation=cross,
        current_indexes=current_indexes,
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        orientation_batch_start=0,
        orientation_eligible=orient_eligible,
        img_h=1,
        img_w=1,
    )

    assert mip[0, 0].item() == 2.0
    assert best[0, 0].item() == 11
    assert corr_sum[0, 0].item() == 7.0
    assert corr_sq[0, 0].item() == 29.0


def test_validate_orientation_eligible_for_match_template_strict_none() -> None:
    validate_orientation_eligible_for_match_template(
        backend="streamed",
        orientation_eligible=None,
        num_orientations=3,
    )
    with pytest.raises(ValueError, match="orientation_eligible must be None"):
        validate_orientation_eligible_for_match_template(
            backend="streamed",
            orientation_eligible=torch.ones(3),
            num_orientations=3,
        )


def test_validate_orientation_eligible_masked_requires_tensor() -> None:
    with pytest.raises(ValueError, match="orientation_eligible is required"):
        validate_orientation_eligible_for_match_template(
            backend="streamed_masked_mip",
            orientation_eligible=None,
            num_orientations=2,
        )
