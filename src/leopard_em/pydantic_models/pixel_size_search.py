"""Serialization and validation of pixel size search parameters for 2DTM."""

from typing import Annotated

import torch
from pydantic import Field

from leopard_em.pydantic_models.types import BaseModel2DTM


class PixelSizeSearchConfig(BaseModel2DTM):
    """Serialization and validation of pixel size search parameters for 2DTM.

    Attributes
    ----------
    enabled : bool
        Whether to enable defocus search. Default is True.
    pixel_size_min : float
        Minimum searched pixel size in units of Angstroms.
    pixel_size_max : float
        Maximum searched pixel size in units of Angstroms.
    pixel_size_step : float
        Step size for pixel size search in units of Angstroms.

    Properties
    ----------
    pixel_size_values : torch.Tensor
        Tensor of pixel sizes to search over based on held params.
    """

    enabled: bool = False
    pixel_size_min: float
    pixel_size_max: float
    pixel_size_step: Annotated[float, Field(..., gt=0.0)]

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

        vals = torch.arange(
            self.pixel_size_min,
            self.pixel_size_max + self.pixel_size_step,
            self.pixel_size_step,
            dtype=torch.float32,
        )
        # If 0 not in pixel sizes add it, but keep it 1D
        if 0.0 not in vals:
            vals = torch.cat([vals, torch.tensor([0.0])])

        # Re-sort pixel sizes
        vals = torch.sort(vals)[0]

        # Ensure that there is at least one pixel size value to search over.
        if vals.numel() == 0:
            raise ValueError(
                "Pixel size search parameters result in no values to search over!\n"
                f"  self.pixel_size_min: {self.pixel_size_min}\n"
                f"  self.pixel_size_max: {self.pixel_size_max}\n"
                f"  self.pixel_size_step: {self.pixel_size_step}\n"
            )

        return vals
