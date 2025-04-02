"""Serialization and validation of orientation search parameters for 2DTM."""

import re
from typing import Annotated, Literal

import roma
import torch
from pydantic import Field
from torch_so3 import (
    get_local_high_resolution_angles,
    get_symmetry_ranges,
    get_uniform_euler_angles,
)

from leopard_em.pydantic_models.types import BaseModel2DTM

EPS = 1e-6


class OrientationSearchConfig(BaseModel2DTM):
    """Serialization and validation of orientation search parameters for 2DTM.

    The angles -- phi, theta, and psi -- represent Euler angles in the 'ZYZ'
    convention.

    This model effectively acts as a connector into the
    `torch_so3.uniform_so3_sampling.get_uniform_euler_angles` function from the
    [torch-so3](https://github.com/teamtomo/torch-so3) package.

    TODO: Add parameters for template symmetry.

    TODO: Implement indexing to get the i-th or range of orientations in the
    search space (need to be ordered).

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
        Currently only 'C1' is supported.
    phi_min : float
        Minimum value for the phi angle in degrees.
    phi_max : float
        Maximum value for the phi angle in degrees.
    theta_min : float
        Minimum value for the theta angle in degrees.
    theta_max : float
        Maximum value for the theta angle in degrees.
    psi_min : float
        Minimum value for the psi angle in degrees.
    psi_max : float
        Maximum value for the psi angle in degrees.
    psi_step : float
        Angular step size for psi in degrees. Must be greater
        than 0.
    theta_step : float
        Angular step size for theta in degrees. Must be
        greater than 0.
    """

    # TODO: Particle symmetry options

    psi_step: Annotated[float, Field(ge=0.0)] = 1.5
    theta_step: Annotated[float, Field(ge=0.0)] = 2.5
    phi_min: float | None = None
    phi_max: float | None = None
    theta_min: float | None = None
    theta_max: float | None = None
    psi_min: float | None = None
    psi_max: float | None = None
    base_grid_method: Literal["uniform", "healpix", "cartesian"] = "uniform"
    symmetry: str = "C1"

    @property
    def euler_angles(self) -> torch.Tensor:
        """Returns the Euler angles ('ZYZ' convention) to search over.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the number of orientations to
            search over. The columns represent the psi, theta, and phi angles
            respectively.
        """
        # Get symmetry order and group
        match = re.match(r"([A-Za-z]+)(\d*)", self.symmetry)
        if not match:
            raise ValueError(f"Invalid symmetry format: {self.symmetry}")
        sym_group = match.group(1)
        sym_order = int(match.group(2)) if match.group(2) else 1

        # Check if all angle parameters are None
        all_none = all(
            x is None
            for x in [
                self.phi_min,
                self.phi_max,
                self.theta_min,
                self.theta_max,
                self.psi_min,
                self.psi_max,
            ]
        )

        # Check if all angle parameters are set (not None)
        all_set = all(
            x is not None
            for x in [
                self.phi_min,
                self.phi_max,
                self.theta_min,
                self.theta_max,
                self.psi_min,
                self.psi_max,
            ]
        )

        # Error if some are None and some are not
        if not (all_none or all_set):
            raise ValueError(
                "Either all angle parameters (phi_min, phi_max, theta_min, theta_max, "
                "psi_min, psi_max) must be set or all must be None"
            )

        # If all are None, use symmetry to set them
        if all_none:
            (
                self.phi_min,
                self.phi_max,
                self.theta_min,
                self.theta_max,
                self.psi_min,
                self.psi_max,
            ) = get_symmetry_ranges(sym_group, sym_order)

        return get_uniform_euler_angles(
            psi_step=self.psi_step,
            theta_step=self.theta_step,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            psi_min=self.psi_min,
            psi_max=self.psi_max,
            base_grid_method=self.base_grid_method,
        )


class RefineOrientationConfig(BaseModel2DTM):
    """Serialization and validation of orientation refinement parameters.

    Angles will be sampled from [-coarse_step, coarse_step] in increments of
    'fine_step' for the orientation refinement search.

    Attributes
    ----------
    orientation_sampling_method : str
        Method for sampling orientations. Default is 'Hopf Fibration'.
        Currently only 'Hopf Fibration' is supported.
    template_symmetry : str
        Symmetry group of the template. Default is 'C1'.
        Currently only 'C1' is supported.
    phi_step_coarse : float
        Angular step size for phi in degrees for previous, coarse search.
        This corresponds to the 'OrientationSearchConfig.phi_step' value
        for the match template program. Must be greater than or equal to 0.
    phi_step_fine : float
        Angular step size for phi in degrees for current, fine search.
        Must be greater than or equal to 0.
    theta_step_coarse : float
        Angular step size for theta in degrees for previous, coarse
        search. This corresponds to the
        'OrientationSearchConfig.theta_step' value for the match template
        program. Must be greater than or equal to 0.
    theta_step_fine : float
        Angular step size for theta in degrees for current, fine search.
        Must be greater than or equal to 0.
    psi_step_coarse : float
        Angular step size for psi in degrees for previous, coarse search.
        This corresponds to the 'OrientationSearchConfig.psi_step' value
        for the match template program. Must be greater than or equal to 0.
    psi_step_fine : float
        Angular step size for psi in degrees for current, fine search.
        Must be greater than or equal to 0.

    """

    enabled: bool = True
    phi_step_coarse: Annotated[float, Field(..., ge=0.0)] = 2.5
    phi_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.25
    theta_step_coarse: Annotated[float, Field(..., ge=0.0)] = 2.5
    theta_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.25
    psi_step_coarse: Annotated[float, Field(..., ge=0.0)] = 1.5
    psi_step_fine: Annotated[float, Field(..., ge=0.0)] = 0.1
    base_grid_method: Literal["uniform", "healpix", "basic"] = "uniform"

    @property
    def euler_angles_offsets(self) -> torch.Tensor:
        """Return the Euler angle offsets to search over.

        Note that this method uses a uniform grid search which approximates SO(3) space
        well when the angular ranges are small (e.g. ±2.5 degrees).

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the number of orientations to
            search over. The columns represent the phi, theta, and psi angles,
            respectively, in the 'ZYZ' convention.
        """
        if not self.enabled:
            return torch.zeros((1, 3))

        return get_local_high_resolution_angles(
            coarse_phi_step=self.phi_step_coarse,
            coarse_theta_step=self.theta_step_coarse,
            coarse_psi_step=self.psi_step_coarse,
            fine_phi_step=self.phi_step_fine,
            fine_theta_step=self.theta_step_fine,
            fine_psi_step=self.psi_step_fine,
            base_grid_method=self.base_grid_method,
        )


class ConstrainedOrientationConfig(BaseModel2DTM):
    """Serialization and validation of constrained orientation parameters.

    Attributes
    ----------
    enabled: bool
        Whether to enable constrained orientation search.
    phi_step: float
        Angular step size for phi in degrees.
        Must be greater than or equal to 0.
    theta_step: float
        Angular step size for theta in degrees.
        Must be greater than or equal to 0.
    psi_step: float
        Angular step size for psi in degrees.
        Must be greater than or equal to 0.
    rotation_axis_euler_angles: list[float]
        List of Euler angles (phi, theta, psi) for the rotation axis.
    phi_min: float
        Minimum value for the phi angle in degrees.
    phi_max: float
        Maximum value for the phi angle in degrees.
    theta_min: float
        Minimum value for the theta angle in degrees.
    theta_max: float
        Maximum value for the theta angle in degrees.
    psi_min: float
        Minimum value for the psi angle in degrees.
    psi_max: float
        Maximum value for the psi angle in degrees.
    """

    enabled: bool = True
    phi_step: float | None = None
    theta_step: float = 2.5
    psi_step: float = 1.5
    rotation_axis_euler_angles: tuple[float, float, float] = Field(
        default=[0.0, 0.0, 0.0]
    )
    phi_min: float = 0.0
    phi_max: float = 360.0
    theta_min: float = 0.0
    theta_max: float = 180.0
    psi_min: float = 0.0
    psi_max: float = 360.0
    base_grid_method: Literal["uniform", "healpix", "basic"] = "uniform"

    @property
    def euler_angles_offsets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Euler angle offsets to search over.

        Note that this method uses a uniform grid search which approximates SO(3) space
        well when the angular ranges are small.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple of two tensors of shape (N, 3) where N is the number of
            orientations to search over. The first tensor represents the Euler
            angles of the rotated template, and the second tensor represents
            the Euler angles of the rotation axis. The columns represent the
            phi, theta, and psi angles, respectively, in the 'ZYZ' convention.
        """
        if not self.enabled:
            return torch.zeros((1, 3)), torch.zeros((1, 3))

        euler_angles_offsets = get_uniform_euler_angles(
            phi_step=self.phi_step,
            theta_step=self.theta_step,
            psi_step=self.psi_step,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            psi_min=self.psi_min,
            psi_max=self.psi_max,
            base_grid_method=self.base_grid_method,
        )
        # Convert to rotation matrix
        rot_z_matrix = roma.euler_to_rotmat(
            "ZYZ",
            euler_angles_offsets,
            degrees=True,
            device=euler_angles_offsets.device,
        ).to(torch.float32)
        # Apply rotation to the template
        rot_axis_matrix = roma.euler_to_rotmat(
            "ZYZ",
            torch.tensor(self.rotation_axis_euler_angles),
            degrees=True,
            device=euler_angles_offsets.device,
        ).to(torch.float32)

        rot_matrix_batch = roma.rotmat_composition(
            sequence=(rot_axis_matrix, rot_z_matrix, rot_axis_matrix.transpose(-1, -2))
        )

        # Convert back to Euler angles
        euler_angles_offsets_rotated = roma.rotmat_to_euler(
            "ZYZ", rot_matrix_batch, degrees=True
        ).to(torch.float32)
        return euler_angles_offsets_rotated, euler_angles_offsets
