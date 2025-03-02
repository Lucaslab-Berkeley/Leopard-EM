"""Serialization and validation of defocus search parameters for 2DTM."""

from typing import Annotated

import torch
from pydantic import Field

from leopard_em.pydantic_models.types import BaseModel2DTM


class DefocusSearchConfig(BaseModel2DTM):
    """Serialization and validation of defocus search parameters for 2DTM.

    Attributes
    ----------
    enabled : bool
        Whether to enable defocus search. Default is True.
    defocus_min : float
        Minimum searched defocus relative to average defocus ('defocus_u' and
        'defocus_v' in OpticsGroup) of micrograph in units of Angstroms.
    defocus_max : float
        Maximum searched defocus relative to average defocus ('defocus_u' and
        'defocus_v' in OpticsGroup) of micrograph in units of Angstroms.
    defocus_step : float
        Step size for defocus search in units of Angstroms.

    Properties
    ----------
    defocus_values : list[float]
        List of relative defocus values to search over based on held params.
    """

    enabled: bool = True
    defocus_min: float
    defocus_max: float
    defocus_step: Annotated[float, Field(..., gt=0.0)]

    @property
    def defocus_values(self) -> torch.Tensor:
        """Relative defocus values to search over based on held params.

        Returns
        -------
        torch.Tensor
            Tensor of relative defocus values to search over, in units of Angstroms.

        Raises
        ------
        ValueError
            If defocus search parameters result in no defocus values to search over.
        """
        # Return a relative defocus of 0.0 if search is disabled.
        if not self.enabled:
            return torch.tensor([0.0])

        vals = torch.arange(
            self.defocus_min,
            self.defocus_max + self.defocus_step,
            self.defocus_step,
            dtype=torch.float32,
        )
        # If 0 not in defocuses add it, but keep it 1D
        if 0.0 not in vals:
            vals = torch.cat([vals, torch.tensor([0.0])])

        # re-sort defocuses
        vals = torch.sort(vals)[0]

        # Ensure that there is at least one defocus value to search over.
        if vals.numel() == 0:
            raise ValueError(
                "Defocus search parameters result in no values to search over!\n"
                f"  self.defocus_min: {self.defocus_min}\n"
                f"  self.defocus_max: {self.defocus_max}\n"
                f"  self.defocus_step: {self.defocus_step}\n"
            )

        return vals
