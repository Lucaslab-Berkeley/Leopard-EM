"""Tests for OrientationSearchConfig.effective_euler_bounds and
orientation_eligible_mask."""

import torch

from leopard_em.pydantic_models.config import OrientationSearchConfig

# ---------------------------------------------------------------------------
# effective_euler_bounds
# ---------------------------------------------------------------------------


def test_effective_euler_bounds_defaults_c1() -> None:
    """C1 symmetry should give the full-sphere bounds."""
    cfg = OrientationSearchConfig(symmetry="C1")
    phi_min, phi_max, theta_min, theta_max, psi_min, psi_max = (
        cfg.effective_euler_bounds
    )
    assert phi_min == 0.0
    assert phi_max == 360.0
    assert theta_min == 0.0
    assert theta_max == 180.0
    assert psi_min == 0.0
    assert psi_max == 360.0


def test_effective_euler_bounds_manual() -> None:
    """Manual ranges should be returned unchanged."""
    cfg = OrientationSearchConfig(
        symmetry=None,
        phi_min=10.0,
        phi_max=200.0,
        theta_min=20.0,
        theta_max=90.0,
        psi_min=5.0,
        psi_max=300.0,
    )
    phi_min, phi_max, theta_min, theta_max, psi_min, psi_max = (
        cfg.effective_euler_bounds
    )
    assert phi_min == 10.0
    assert phi_max == 200.0
    assert theta_min == 20.0
    assert theta_max == 90.0
    assert psi_min == 5.0
    assert psi_max == 300.0


# ---------------------------------------------------------------------------
# orientation_eligible_mask — basic in/out
# ---------------------------------------------------------------------------


def test_orientation_eligible_mask_all_inside() -> None:
    """All rows within bounds → all True."""
    cfg = OrientationSearchConfig(
        symmetry=None,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=0.0,
        theta_max=180.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    euler = torch.tensor([[10.0, 45.0, 90.0], [180.0, 90.0, 180.0]])
    mask = cfg.orientation_eligible_mask(euler)
    assert mask.all()


def test_orientation_eligible_mask_outside_theta() -> None:
    """Rows with theta > theta_max → False."""
    cfg = OrientationSearchConfig(
        symmetry=None,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=0.0,
        theta_max=90.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    euler = torch.tensor(
        [
            [0.0, 45.0, 0.0],  # inside
            [0.0, 91.0, 0.0],  # outside (theta too large)
            [0.0, 0.0, 0.0],  # inside
        ]
    )
    mask = cfg.orientation_eligible_mask(euler)
    assert mask[0]
    assert not mask[1]
    assert mask[2]


def test_orientation_eligible_mask_outside_phi() -> None:
    """Rows with phi outside [phi_min, phi_max] → False."""
    cfg = OrientationSearchConfig(
        symmetry=None,
        phi_min=30.0,
        phi_max=150.0,
        theta_min=0.0,
        theta_max=180.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    euler = torch.tensor(
        [
            [90.0, 45.0, 0.0],  # inside
            [20.0, 45.0, 0.0],  # outside (phi too small)
            [200.0, 45.0, 0.0],  # outside (phi too large)
        ]
    )
    mask = cfg.orientation_eligible_mask(euler)
    assert mask[0]
    assert not mask[1]
    assert not mask[2]


# ---------------------------------------------------------------------------
# wrap-around bounds
# ---------------------------------------------------------------------------


def test_orientation_eligible_mask_phi_wrap_around() -> None:
    """Bounds that straddle 360° (phi_min > phi_max) are handled correctly."""
    cfg = OrientationSearchConfig(
        symmetry=None,
        phi_min=330.0,
        phi_max=30.0,
        theta_min=0.0,
        theta_max=180.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    euler = torch.tensor(
        [
            [350.0, 45.0, 0.0],  # inside (>= 330)
            [10.0, 45.0, 0.0],  # inside (<= 30)
            [180.0, 45.0, 0.0],  # outside
        ]
    )
    mask = cfg.orientation_eligible_mask(euler)
    assert mask[0], "350° should be inside [330, 30] wrap-around range"
    assert mask[1], "10° should be inside [330, 30] wrap-around range"
    assert not mask[2], "180° should be outside [330, 30] wrap-around range"


# ---------------------------------------------------------------------------
# C2 symmetry (non-trivial reduced bounds)
# ---------------------------------------------------------------------------


def test_orientation_eligible_mask_c2_symmetry() -> None:
    """C2 symmetry bounds restrict phi to [0, 180]; rows with phi > 180 are out."""
    cfg = OrientationSearchConfig(symmetry="C2")
    _phi_min, phi_max, _theta_min, _theta_max, _psi_min, _psi_max = (
        cfg.effective_euler_bounds
    )
    # C2 symmetry halves phi range to [0, 180]
    assert phi_max <= 180.0 + 1e-3

    euler = torch.tensor(
        [
            [90.0, 45.0, 0.0],  # inside (phi=90 within [0,180])
            [270.0, 45.0, 0.0],  # outside for C2 (phi=270 > 180)
        ]
    )
    mask = cfg.orientation_eligible_mask(euler)
    assert mask[0]
    assert not mask[1]
