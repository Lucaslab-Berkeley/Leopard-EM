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


# ------------------------------------------------------------------------------------
# Layered psi_center: accept an orientation if ANY covering region allows it
# ------------------------------------------------------------------------------------


def _run_psi_gate(cc, euler, psi_center, pole_mask, cone=10.0, table=True,
                  extra_layers=None):
    """One gated update on fresh state; returns (mip, best, table)."""
    n_orient, height, width = cc.shape
    mip = torch.full((height, width), -float("inf"))
    best = torch.full((height, width), -1, dtype=torch.int32)
    corr_table = _empty_table()
    do_iteration_and_correlation_table_updates(
        cross_correlation=cc,
        current_indexes=torch.arange(n_orient, dtype=torch.int32),
        correlation_table=corr_table,
        mip=mip,
        best_global_index=best,
        correlation_sum=torch.zeros((height, width)),
        correlation_squared_sum=torch.zeros((height, width)),
        threshold=5.5,
        valid_shape_h=height,
        valid_shape_w=width,
        needs_valid_cropping=False,
        compute_correlation_table=table,
        n_orientations=n_orient,
        psi_center=psi_center,
        pole_mask=pole_mask,
        euler_angles=euler,
        psi_cone_half_angle_deg=cone,
        psi_theta_center_deg=90.0,
        psi_phi_min=0.0,
        psi_phi_max=360.0,
        psi_extra_layers=extra_layers,
    )
    return mip, best, corr_table


def _crossing_case():
    """One pixel owned by a filament at psi 0, crossed by one at psi 30.

    Orientation 0 is the crossing filament's (psi 30, strongest), orientation 1 the
    owner's (psi 0), orientation 2 belongs to neither (psi 60).
    """
    cc = torch.tensor([[[9.0]], [[7.0]], [[8.0]]], dtype=torch.float32)
    euler = torch.tensor(
        [[0.0, 90.0, 30.0], [0.0, 90.0, 0.0], [0.0, 90.0, 60.0]], dtype=torch.float32
    )
    return cc, euler


def test_single_layer_psi_center_rejects_the_crossing_filament():
    cc, euler = _crossing_case()
    mip, best, table = _run_psi_gate(
        cc, euler, torch.zeros((1, 1)), torch.full((1, 1), 3, dtype=torch.uint8)
    )
    assert mip[0, 0] == pytest.approx(7.0)
    assert int(best[0, 0]) == 1
    assert sorted(table["global_idx"].tolist()) == [1]


def test_layered_psi_center_admits_the_crossing_filament():
    cc, euler = _crossing_case()
    psi = torch.tensor([[[0.0]], [[30.0]]])
    poles = torch.full((2, 1, 1), 3, dtype=torch.uint8)
    mip, best, table = _run_psi_gate(cc, euler, psi, poles)
    assert mip[0, 0] == pytest.approx(9.0)
    assert int(best[0, 0]) == 0
    # Both filaments' detections reach the table; the one neither allows does not.
    assert sorted(table["global_idx"].tolist()) == [0, 1]


def test_unused_layer_contributes_nothing():
    """Pole bits of 0 mark a layer unused at that pixel."""
    cc, euler = _crossing_case()
    psi = torch.tensor([[[0.0]], [[30.0]]])
    poles = torch.tensor([[[3]], [[0]]], dtype=torch.uint8)
    mip, best, _ = _run_psi_gate(cc, euler, psi, poles)
    assert int(best[0, 0]) == 1


def test_each_layer_keeps_its_own_polarity():
    """A positive-only crossing filament must not admit its opposite pole."""
    cc = torch.tensor([[[9.0]], [[7.0]]], dtype=torch.float32)
    euler = torch.tensor([[0.0, 90.0, 210.0], [0.0, 90.0, 30.0]], dtype=torch.float32)
    psi = torch.tensor([[[100.0]], [[30.0]]])
    poles = torch.tensor([[[3]], [[1]]], dtype=torch.uint8)  # layer 1 positive only
    mip, best, _ = _run_psi_gate(cc, euler, psi, poles)
    assert int(best[0, 0]) == 1
    assert mip[0, 0] == pytest.approx(7.0)


def test_one_layer_stack_is_bit_identical_to_a_plain_map():
    """K = 1 must reproduce today's behaviour exactly."""
    gen = torch.Generator().manual_seed(0)
    n_orient, height, width = 24, 6, 7
    cc = torch.randn((n_orient, height, width), generator=gen) * 3 + 4
    euler = torch.stack(
        [
            torch.rand(n_orient, generator=gen) * 360,
            80 + torch.rand(n_orient, generator=gen) * 20,
            torch.rand(n_orient, generator=gen) * 360,
        ],
        dim=1,
    )
    psi = torch.rand((height, width), generator=gen) * 360
    poles = torch.randint(1, 4, (height, width), generator=gen, dtype=torch.uint8)
    plain = _run_psi_gate(cc, euler, psi, poles, cone=40.0)
    stacked = _run_psi_gate(cc, euler, psi[None], poles[None], cone=40.0)
    assert torch.equal(plain[0], stacked[0])
    assert torch.equal(plain[1], stacked[1])
    for key in ("global_idx", "pos_x", "pos_y", "corr_value"):
        assert torch.equal(plain[2][key], stacked[2][key])


def test_precomputed_sparse_layers_match_building_them_per_call():
    from leopard_em.backend.utils import sparse_psi_layers

    gen = torch.Generator().manual_seed(1)
    n_orient, height, width = 30, 5, 9
    cc = torch.randn((n_orient, height, width), generator=gen) * 3 + 4
    euler = torch.stack(
        [
            torch.zeros(n_orient),
            torch.full((n_orient,), 90.0),
            torch.rand(n_orient, generator=gen) * 360,
        ],
        dim=1,
    )
    psi = torch.rand((3, height, width), generator=gen) * 360
    poles = torch.randint(0, 4, (3, height, width), generator=gen, dtype=torch.uint8)
    poles[0].clamp_(min=1)
    per_call = _run_psi_gate(cc, euler, psi, poles, cone=15.0)
    sparse = sparse_psi_layers(psi, poles)
    precomputed = _run_psi_gate(
        cc, euler, psi[0], poles[0], cone=15.0, extra_layers=sparse
    )
    assert torch.equal(per_call[0], precomputed[0])
    assert torch.equal(per_call[1], precomputed[1])
    assert torch.equal(per_call[2]["global_idx"], precomputed[2]["global_idx"])


def test_layered_gate_equals_the_union_of_single_layer_gates():
    """Brute force: a stack admits exactly what at least one layer alone admits."""
    from leopard_em.backend.utils import _orient_ok_from_psi_center

    gen = torch.Generator().manual_seed(2)
    n_orient, height, width = 64, 4, 6
    euler = torch.stack(
        [
            torch.rand(n_orient, generator=gen) * 360,
            70 + torch.rand(n_orient, generator=gen) * 40,
            torch.rand(n_orient, generator=gen) * 360,
        ],
        dim=1,
    )
    psi = torch.rand((3, height, width), generator=gen) * 360
    poles = torch.randint(0, 4, (3, height, width), generator=gen, dtype=torch.uint8)
    kwargs = dict(
        euler_angles=euler,
        orient_idx=torch.arange(n_orient),
        cone_half_angle_deg=20.0,
        theta_center_deg=90.0,
        phi_min=0.0,
        phi_max=360.0,
        height=height,
        width=width,
    )
    layered = _orient_ok_from_psi_center(psi_center=psi, pole_mask=poles, **kwargs)
    union = torch.zeros_like(layered)
    for layer in range(3):
        union |= _orient_ok_from_psi_center(
            psi_center=psi[layer], pole_mask=poles[layer], **kwargs
        )
    assert torch.equal(layered, union)
