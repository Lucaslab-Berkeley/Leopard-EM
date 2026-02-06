"""Serialization and validation of defocus search parameters for 2DTM."""

from typing import Annotated

import torch
from pydantic import Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM
from leopard_em.utils.search_utils import get_search_tensors


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
    skip_enforce_zero : bool
        Whether to skip enforcing a zero value, by default False.

    Properties
    ----------
    defocus_values : torch.Tensor
        Tensor of relative defocus values to search over based on held params.
    """

    enabled: bool = True
    defocus_min: float = -1000.0
    defocus_max: float = 1000.0
    defocus_step: Annotated[float, Field(..., gt=0.0)] = 200.0
    skip_enforce_zero: bool = False

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

            # Check if parameters would result in valid range before calling arange
        if self.defocus_max < self.defocus_min:
            raise ValueError(
                "Defocus search parameters result in no values to search over!\n"
                f"  self.defocus_min: {self.defocus_min}\n"
                f"  self.defocus_max: {self.defocus_max}\n"
                f"  self.defocus_step: {self.defocus_step}\n"
            )

        return get_search_tensors(
            self.defocus_min,
            self.defocus_max,
            self.defocus_step,
            self.skip_enforce_zero,
        )
