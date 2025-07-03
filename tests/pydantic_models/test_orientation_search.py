"""Tests for the OrientationSearchConfig and RefineOrientationConfig models"""

import pytest
import torch
from pydantic import ValidationError

from leopard_em.pydantic_models.config import (
    OrientationSearchConfig,
    RefineOrientationConfig,
)


def test_orientation_search_default_values():
    """
    Test the default values of the OrientationSearchConfig.

    Verifies that all default properties, including angular ranges and steps,
    are correctly set to their expected values.
    """
    config = OrientationSearchConfig()
    assert config.psi_step == 1.5
    assert config.theta_step == 2.5
    assert config.psi_min is None
    assert config.psi_max is None
    assert config.theta_min is None
    assert config.theta_max is None
    assert config.phi_min is None
    assert config.phi_max is None
    assert config.symmetry == "C1"
    assert config.base_grid_method == "uniform"


def test_orientation_search_invalid_symmetry_and_angle_ranges():
    """
    Test that an error is raised when both symmetry and angle ranges are provided.
    """
    with pytest.raises(
        ValidationError,
        match="Symmetry group is provided, but angle ranges are also set. ",
    ):
        OrientationSearchConfig(
            symmetry="C1",
            psi_min=0.0,
            psi_max=360.0,
            theta_min=0.0,
            theta_max=180.0,
        )

    with pytest.raises(
        ValidationError,
        match="Either a symmetry group must be provided, or all angle ranges must",
    ):
        OrientationSearchConfig(
            symmetry=None,
            phi_min=None,
            phi_max=None,
            theta_min=None,
            theta_max=None,
            psi_min=None,
            psi_max=None,
        )


def test_orientation_search_invalid_symmetry():
    """
    Test that an error is raised for invalid symmetry group values.

    Verifies that a ValidationError is raised when an unsupported symmetry group
    is provided.
    """
    with pytest.raises(ValidationError, match="Invalid symmetry format: C2v"):
        OrientationSearchConfig(symmetry="C2v")


def test_orientation_search_invalid_in_plane_step():
    """
    Test that an error is raised for invalid in-plane step values.

    Verifies that a ValidationError is raised when in_plane_step is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(psi_step=-1.0)


def test_orientation_search_invalid_out_of_plane_step():
    """
    Test that an error is raised for invalid out-of-plane step values.

    Verifies that a ValidationError is raised when out_of_plane_step is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        OrientationSearchConfig(theta_step=-1.0)


def test_orientation_search_euler_angles():
    """
    Test that the euler_angles property generates the correct tensor.

    Verifies that the generated Euler angles tensor has the expected shape and type
    when using specific in-plane and out-of-plane step values.
    """
    config = OrientationSearchConfig(psi_step=90.0, theta_step=90.0)
    euler_angles = config.euler_angles

    assert isinstance(euler_angles, torch.Tensor)
    assert euler_angles.shape == (24, 3)


def test_refine_orientation_default_values():
    """
    Test the default values of the RefineOrientationConfig.

    Verifies that all default angular step parameters for coarse and fine
    refinement are correctly set to their expected values.
    """
    config = RefineOrientationConfig()
    assert config.psi_step_coarse == 1.5
    assert config.psi_step_fine == 0.1
    assert config.theta_step_coarse == 2.5
    assert config.theta_step_fine == 0.25


def test_refine_orientation_invalid_in_plane_angular_step_coarse():
    """
    Test that an error is raised for invalid in-plane coarse step values.

    Verifies that a ValidationError is raised when in_plane_angular_step_coarse
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(psi_step_coarse=-1.0)


def test_refine_orientation_invalid_in_plane_angular_step_fine():
    """
    Test that an error is raised for invalid in-plane fine step values.

    Verifies that a ValidationError is raised when in_plane_angular_step_fine
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(psi_step_fine=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_coarse():
    """
    Test that an error is raised for invalid out-of-plane coarse step values.

    Verifies that a ValidationError is raised when out_of_plane_angular_step_coarse
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(theta_step_coarse=-1.0)


def test_refine_orientation_invalid_out_of_plane_angular_step_fine():
    """
    Test that an error is raised for invalid out-of-plane fine step values.

    Verifies that a ValidationError is raised when out_of_plane_angular_step_fine
    is negative.
    """
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        RefineOrientationConfig(theta_step_fine=-1.0)


def test_refine_orientation_euler_angles_offsets():
    """
    Test that the euler_angles_offsets property generates the correct tensor.

    Verifies that the generated Euler angles offsets tensor has the expected shape and
    type when using specific angular step values for both coarse and fine refinement.
    """
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
