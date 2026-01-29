"""Serialization and validation of pixel size search parameters for 2DTM."""

from typing import Annotated

import torch
from pydantic import Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM
from leopard_em.utils.search_utils import get_search_tensors


class PixelSizeSearchConfig(BaseModel2DTM):
    """Serialization and validation of pixel size search parameters for 2DTM.

    Attributes
    ----------
    enabled : bool
        Whether to enable pixel size search. Default is False.
    pixel_size_min : float
        Minimum searched pixel size in units of Angstroms.
    pixel_size_max : float
        Maximum searched pixel size in units of Angstroms.
    pixel_size_step : float
        Step size for pixel size search in units of Angstroms.
    skip_enforce_zero : bool
        Whether to skip enforcing a zero value, by default False.

    Properties
    ----------
    pixel_size_values : torch.Tensor
        Tensor of pixel sizes to search over based on held params.
    """

    enabled: bool = False
    pixel_size_min: float = 0.0
    pixel_size_max: float = 0.0
    pixel_size_step: Annotated[float, Field(..., gt=0.0)] = 0.0
    skip_enforce_zero: bool = False

    @property
    def pixel_size_values(self) -> torch.Tensor:
        """Pixel sizes to search over based on held params.

        Returns
        -------
        torch.Tensor
            Tensor of pixel sizes to search over, in units of Angstroms.

        Raises
        ------
        ValueError
            If pixel size search parameters result in no pixel sizes to search over.
        """
        # Return a relative pixel size of 0.0 if search is disabled.
        if not self.enabled:
            return torch.tensor([0.0])

        # Check if parameters would result in valid range before calling arange
        if self.pixel_size_max < self.pixel_size_min:
            raise ValueError(
                "Pixel size search parameters result in no values to search over!\n"
                f"  self.pixel_size_min: {self.pixel_size_min}\n"
                f"  self.pixel_size_max: {self.pixel_size_max}\n"
                f"  self.pixel_size_step: {self.pixel_size_step}\n"
            )

        return get_search_tensors(
            self.pixel_size_min,
            self.pixel_size_max,
            self.pixel_size_step,
            self.skip_enforce_zero,
        )
