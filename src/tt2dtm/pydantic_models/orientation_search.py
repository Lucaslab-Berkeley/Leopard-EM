"""Serialization and validation of orientation search parameters for 2DTM."""

from typing import Annotated, Literal

import torch
from pydantic import Field
from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

from tt2dtm.pydantic_models.types import BaseModel2DTM


class OrientationSearchConfig(BaseModel2DTM):
    """Serialization and validation of orientation search parameters for 2DTM.

    The angles -- psi, theta, and phi -- represent Euler angles in the 'ZYZ'
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
    psi_min : float
        Minimum value for the psi angle in degrees.
    psi_max : float
        Maximum value for the psi angle in degrees.
    theta_min : float
        Minimum value for the theta angle in degrees.
    theta_max : float
        Maximum value for the theta angle in degrees.
    phi_min : float
        Minimum value for the phi angle in degrees.
    phi_max : float
        Maximum value for the phi angle in degrees.
    in_plane_angular_step : float
        Angular step size for in-plane rotations in degrees. Must be greater
        than 0.
    out_of_plane_angular_step : float
        Angular step size for out-of-plane rotations in degrees. Must be
        greater than 0.
    """

    # TODO: Particle symmetry options

    in_plane_step: Annotated[float, Field(ge=0.0)] = 1.5
    out_of_plane_step: Annotated[float, Field(ge=0.0)] = 2.5
    psi_min: float = 0.0
    psi_max: float = 360.0
    theta_min: float = 0.0
    theta_max: float = 180.0
    phi_min: float = 0.0
    phi_max: float = 360.0
    base_grid_method: Literal["uniform", "healpix"] = "uniform"

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
        return get_uniform_euler_angles(
            in_plane_step=self.in_plane_step,
            out_of_plane_step=self.out_of_plane_step,
            psi_min=self.psi_min,
            psi_max=self.psi_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            phi_min=self.phi_min,
            phi_max=self.phi_max,
            base_grid_method=self.base_grid_method,
        )
