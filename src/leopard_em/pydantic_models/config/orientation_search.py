"""Serialization and validation of orientation search parameters for 2DTM."""

import re
from typing import Annotated, Literal, Optional

import roma
import torch
from pydantic import Field, model_validator
from torch_so3 import (
    get_local_high_resolution_angles,
    get_roll_angles,
    get_symmetry_ranges,
    get_uniform_euler_angles,
)
from typing_extensions import Self

from leopard_em.pydantic_models.custom_types import BaseModel2DTM

EPS = 1e-6


class OrientationSearchConfig(BaseModel2DTM):
    """Serialization and validation of orientation search parameters for 2DTM.

    The angles -- phi, theta, and psi -- represent Euler angles in the 'ZYZ'
    convention in units of degrees between 0 and 360 (for phi and psi) or
    between 0 and 180 (for theta).

    This model effectively acts as a connector into the
    `torch_so3.uniform_so3_sampling.get_uniform_euler_angles` function from the
    [torch-so3](https://github.com/teamtomo/torch-so3) package.

    TODO: Implement indexing to get the i-th or range of orientations in the
    search space (need to be ordered).

    Attributes
    ----------
    psi_step : float
        Angular step size for psi in degrees. Must be greater
        than 0.
    theta_step : float
        Angular step size for theta in degrees. Must be
        greater than 0.
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
    base_grid_method : str
        Method for sampling orientations. Default is 'uniform'.
        Currently only 'uniform' is supported.
    symmetry : str
        Symmetry group of the template. Default is 'C1'. Note that if symmetry is
        provided, then the angle min/max values must be all set to None (validation
        will set these automatically based on the symmetry group).
    """

    psi_step: Annotated[float, Field(ge=0.0)] = 1.5
    theta_step: Annotated[float, Field(ge=0.0)] = 2.5
    phi_min: Optional[float] = None
    phi_max: Optional[float] = None
    theta_min: Optional[float] = None
    theta_max: Optional[float] = None
    psi_min: Optional[float] = None
    psi_max: Optional[float] = None

    base_grid_method: Literal["uniform", "healpix", "cartesian"] = "uniform"
    symmetry: Optional[str] = "C1"

    @model_validator(mode="after")  # type: ignore
    def validate_angle_ranges_and_symmetry(self) -> Self:
        """Validate that angle ranges are consistent with symmetry.

        There should be only two valid cases for combinations of manually defined
        angle ranges and the symmetry group:
        1. Symmetry argument is *not* None, and all angle min/max values
           are set to None. In this case, the angle ranges will be set based on
           the symmetry group.
        2. Symmetry argument is None, and all angle min/max values are not None.

        If any other combination is provided, a ValueError will be raised.
        """
        _all_none = all(
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

        # Check that both symmetry and angle ranges are not None
        if not self.symmetry and _all_none:
            raise ValueError(
                "Either a symmetry group must be provided, or all angle ranges must "
                "not be None. Both symmetry and angle ranges were set to None."
            )

        # Case where both symmetry and angle ranges are provided
        if self.symmetry and not _all_none:
            raise ValueError(
                "Symmetry group is provided, but angle ranges are also set. "
                "Please set all angle ranges to None when using symmetry."
            )

        # Case where symmetry group is provided, validate the symmetry
        if self.symmetry:
            match = re.match(r"([A-Za-z]+)(\d*)$", self.symmetry)
            if not match:
                raise ValueError(f"Invalid symmetry format: {self.symmetry}")

        # If we reach here, it means that either symmetry is set or angle ranges are set
        # but not both, so we can proceed.
        return self

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
        # If the symmetry used for the angular ranges, calculate the angular ranges
        # based on the symmetry group.
        if self.symmetry is not None:
            match = re.match(r"([A-Za-z]+)(\d*)", self.symmetry)
            if match is None:
                raise ValueError(f"Invalid symmetry format: {self.symmetry}")

            sym_group = match.group(1)
            sym_order = int(match.group(2)) if match.group(2) else 1
            (phi_min, phi_max, theta_min, theta_max, psi_min, psi_max) = (
                get_symmetry_ranges(sym_group, sym_order)
            )
        # Otherwise, use the provided angular ranges replacing with default values if
        # any are set to None.
        else:
            phi_min = self.phi_min if self.phi_min is not None else 0.0
            phi_max = self.phi_max if self.phi_max is not None else 360.0
            theta_min = self.theta_min if self.theta_min is not None else 0.0
            theta_max = self.theta_max if self.theta_max is not None else 180.0
            psi_min = self.psi_min if self.psi_min is not None else 0.0
            psi_max = self.psi_max if self.psi_max is not None else 360.0

        # Generate angles
        return get_uniform_euler_angles(
            psi_step=self.psi_step,
            theta_step=self.theta_step,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
            psi_min=psi_min,
            psi_max=psi_max,
            base_grid_method=self.base_grid_method,
        )


class MultipleOrientationConfig(BaseModel2DTM):
    """Configuration for multiple orientation search ranges.

    This class allows specifying multiple complete orientation search
    configurations and concatenates their Euler angles.

    Attributes
    ----------
    orientation_configs : list[OrientationSearchConfig]
        List of orientation search configurations to combine.
    """

    orientation_configs: list[OrientationSearchConfig]

    @property
    def euler_angles(self) -> torch.Tensor:
        """Returns the concatenated Euler angles from all orientation configs.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 3) where N is the total number of orientations
            from all configurations. The columns represent the psi, theta, and phi
            angles respectively.
        """
        all_euler_angles = []
        for config in self.orientation_configs:
            all_euler_angles.append(config.euler_angles)

        return torch.cat(all_euler_angles, dim=0)


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
    phi_step: Optional[float] = None
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
    base_grid_method: Literal["uniform", "healpix", "basic", "roll"] = "uniform"

    search_roll_axis: bool = True
    roll_axis: Optional[tuple[float, float]] = Field(default=[0, 1])
    roll_step: float = 2.0

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

        if self.search_roll_axis:
            self.roll_axis = None
        roll_axis = None
        if self.roll_axis is not None:
            roll_axis = torch.tensor(self.roll_axis)

        if self.base_grid_method == "roll":
            euler_angles_offsets = get_roll_angles(
                psi_step=self.psi_step,
                psi_min=self.psi_min,
                psi_max=self.psi_max,
                theta_step=self.theta_step,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                roll_axis=roll_axis,
                roll_axis_step=self.roll_step,
            )
        else:
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
