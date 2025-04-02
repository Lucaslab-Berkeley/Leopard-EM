import pytest
import torch
from pydantic import ValidationError

from leopard_em.pydantic_models.orientation_search import (
    OrientationSearchConfig,
    RefineOrientationConfig,
)


def test_orientation_search_default_values():
    config = OrientationSearchConfig()
    assert config.psi_step == 1.5
    assert config.theta_step == 2.5
    assert config.psi_min is None
    assert config.psi_max is None
    assert config.theta_min is None
    assert config.theta_max is None
    assert config.phi_min is None
    assert config.phi_max is None
    assert config.base_grid_method == "uniform"


def test_orientation_search_invalid_in_plane_step():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(psi_step=-1.0)


def test_orientation_search_invalid_out_of_plane_step():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(theta_step=-1.0)


def test_orientation_search_euler_angles():
    config = OrientationSearchConfig(psi_step=90.0, theta_step=90.0)
    euler_angles = config.euler_angles

    assert isinstance(euler_angles, torch.Tensor)
    assert euler_angles.shape == (24, 3)


def test_refine_orientation_default_values():
    config = RefineOrientationConfig()
    assert config.psi_step_coarse == 1.5
    assert config.psi_step_fine == 0.1
    assert config.theta_step_coarse == 2.5
    assert config.theta_step_fine == 0.25


def test_refine_orientation_invalid_in_plane_angular_step_coarse():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(psi_step_coarse=-1.0)


def test_refine_orientation_invalid_in_plane_angular_step_fine():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(psi_step_fine=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_coarse():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(theta_step_coarse=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_fine():
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(theta_step_fine=-1.0)


def test_refine_orientation_euler_angles_offsets():
    config = RefineOrientationConfig(
        psi_step_coarse=1.5,
        psi_step_fine=0.5,
        theta_step_coarse=2.5,
        theta_step_fine=1.0,
        base_grid_method="uniform",
    )
    euler_angles_offsets = config.euler_angles_offsets

    assert isinstance(euler_angles_offsets, torch.Tensor)
    assert euler_angles_offsets.shape == (42, 3)
