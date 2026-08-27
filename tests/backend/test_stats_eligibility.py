"""Tests for mask-before-max eligibility in the match-template stats kernel."""

import pytest
import tensordict
import torch

from leopard_em.backend.utils import do_iteration_and_correlation_table_updates


def _empty_table() -> tensordict.TensorDict:
    device = torch.device("cpu")
    return tensordict.TensorDict(
        {
            "threshold": 5.5,
            "global_idx": torch.tensor([], dtype=torch.int32, device=device),
            "pos_x": torch.tensor([], dtype=torch.int32, device=device),
            "pos_y": torch.tensor([], dtype=torch.int32, device=device),
            "corr_value": torch.tensor([], dtype=torch.float32, device=device),
        },
        device=device,
    )


def test_mask_before_max_skips_ineligible_pixels():
    height, width = 3, 4
    cc = torch.zeros((2, height, width), dtype=torch.float32)
    cc[0, 0, 0] = 9.0  # ineligible pixel, high CC
    cc[1, 1, 1] = 3.0  # eligible pixel, lower CC
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    pixel_ok = torch.zeros((height, width), dtype=torch.bool)
    pixel_ok[1, 1] = True

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        eligible_pixels=pixel_ok,
        stats_from_valid_orientations_defocus=False,
    )

    assert mip[0, 0] == -float("inf")
    assert mip[1, 1] == pytest.approx(3.0)
    assert corr_sum[0, 0] == pytest.approx(9.0)
    assert corr_sum[1, 1] == pytest.approx(3.0)


def test_stats_from_valid_orientations_defocus_masks_sums():
    height, width = 2, 2
    cc = torch.zeros((2, height, width), dtype=torch.float32)
    cc[0, 0, 0] = 4.0
    cc[1, 0, 1] = 5.0
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    corr_count = torch.zeros((height, width))
    pixel_ok = torch.zeros((height, width), dtype=torch.bool)
    pixel_ok[0, 1] = True

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        eligible_pixels=pixel_ok,
        stats_from_valid_orientations_defocus=True,
        correlation_count=corr_count,
    )

    assert corr_sum[0, 0] == pytest.approx(0.0)
    assert corr_sum[0, 1] == pytest.approx(5.0)
    assert corr_count[0, 1] == pytest.approx(2.0)
    assert corr_count[0, 0] == pytest.approx(0.0)
    assert mip[0, 0] == -float("inf")
    assert mip[0, 1] == pytest.approx(5.0)


def test_mask_before_max_skips_ineligible_defocus():
    height, width = 2, 2
    n_orient = 1
    n_defocus = 2
    cc = torch.zeros((n_defocus, height, width), dtype=torch.float32)
    cc[0, 0, 0] = 9.0  # defocus 0, illegal at this pixel
    cc[1, 0, 0] = 3.0  # defocus 1, allowed
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    defocus_ok = torch.zeros((n_defocus, height, width), dtype=torch.bool)
    defocus_ok[1, 0, 0] = True

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        defocus_eligible=defocus_ok,
        n_orientations=n_orient,
        stats_from_valid_orientations_defocus=False,
    )

    assert mip[0, 0] == pytest.approx(3.0)
    assert int(best[0, 0]) == 1
    assert corr_sum[0, 0] == pytest.approx(12.0)


def test_stats_from_valid_orientations_defocus_masks_illegal_defocus():
    height, width = 1, 1
    n_orient = 1
    n_defocus = 2
    cc = torch.tensor([[[4.0]], [[5.0]]], dtype=torch.float32)
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    corr_count = torch.zeros((height, width))
    defocus_ok = torch.zeros((n_defocus, height, width), dtype=torch.bool)
    defocus_ok[1, 0, 0] = True

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        defocus_eligible=defocus_ok,
        n_orientations=n_orient,
        stats_from_valid_orientations_defocus=True,
        correlation_count=corr_count,
    )

    assert mip[0, 0] == pytest.approx(5.0)
    assert corr_sum[0, 0] == pytest.approx(5.0)
    assert corr_count[0, 0] == pytest.approx(1.0)


def test_mask_before_max_skips_ineligible_orientation_per_pixel():
    height, width = 2, 2
    n_orient = 2
    cc = torch.zeros((n_orient, height, width), dtype=torch.float32)
    cc[0, 0, 0] = 9.0
    cc[1, 0, 0] = 3.0
    cc[0, 1, 1] = 8.0
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    orientation_eligible = torch.zeros((n_orient, height, width), dtype=torch.bool)
    orientation_eligible[1, 0, 0] = True
    orientation_eligible[0, 1, 1] = True

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        orientation_eligible=orientation_eligible,
        n_orientations=n_orient,
        stats_from_valid_orientations_defocus=False,
    )

    assert mip[0, 0] == pytest.approx(3.0)
    assert mip[1, 1] == pytest.approx(8.0)
    assert int(best[0, 0]) == 1
    assert int(best[1, 1]) == 0
    assert corr_sum[0, 0] == pytest.approx(12.0)


def test_stats_from_valid_orientations_masks_illegal_orientation():
    height, width = 1, 1
    n_orient = 2
    cc = torch.tensor([[[4.0]], [[5.0]]], dtype=torch.float32)
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    corr_count = torch.zeros((height, width))
    orientation_eligible = torch.tensor([False, True], dtype=torch.bool)

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        orientation_eligible=orientation_eligible,
        n_orientations=n_orient,
        stats_from_valid_orientations_defocus=True,
        correlation_count=corr_count,
    )

    assert mip[0, 0] == pytest.approx(5.0)
    assert corr_sum[0, 0] == pytest.approx(5.0)
    assert corr_count[0, 0] == pytest.approx(1.0)


def test_psi_center_masks_opposite_pole():
    height, width = 1, 1
    cc = torch.tensor([[[9.0]], [[3.0]]], dtype=torch.float32)
    current_indexes = torch.tensor([0, 1], dtype=torch.int32)
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_sum = torch.zeros((height, width))
    corr_sq = torch.zeros((height, width))
    corr_count = torch.zeros((height, width))
    euler_angles = torch.tensor(
        [[0.0, 90.0, 5.0], [0.0, 90.0, 180.0]], dtype=torch.float32
    )
    psi_center = torch.zeros((height, width), dtype=torch.float32)
    pole_mask = torch.full((height, width), 1, dtype=torch.uint8)

    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=current_indexes,
        correlation_table=_empty_table(),
        mip=mip,
        best_global_index=best,
        correlation_sum=corr_sum,
        correlation_squared_sum=corr_sq,
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=False,
        n_orientations=2,
        stats_from_valid_orientations_defocus=True,
        correlation_count=corr_count,
        psi_center=psi_center,
        pole_mask=pole_mask,
        euler_angles=euler_angles,
        psi_cone_half_angle_deg=10.0,
        psi_theta_center_deg=90.0,
        psi_phi_min=0.0,
        psi_phi_max=360.0,
    )

    assert mip[0, 0] == pytest.approx(9.0)
    assert int(best[0, 0]) == 0
    assert corr_sum[0, 0] == pytest.approx(9.0)
    assert corr_count[0, 0] == pytest.approx(1.0)
