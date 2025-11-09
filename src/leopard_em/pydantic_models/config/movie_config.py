"""Serialization and validation of movie parameters for 2DTM."""

import torch
from torch_motion_correction.data_io import read_deformation_field_from_csv

from leopard_em.pydantic_models.custom_types import BaseModel2DTM
from leopard_em.utils.data_io import load_mrc_volume


class MovieConfig(BaseModel2DTM):
    """Serialization and validation of movie parameters for 2DTM.

    Attributes
    ----------
    enabled: bool
        Whether to enable movie configuration.
    movie_path: str
        Path to the movie file.
    deformation_field_path: str
        Path to the deformation field file.
    pre_exposure: float
        Pre-exposure time in seconds.
    fluence_per_frame: float
        Dose per frame in electrons per pixel.
    """

    enabled: bool = False
    movie_path: str = ""
    deformation_field_path: str = ""
    pre_exposure: float = 0.0
    fluence_per_frame: float = 1.0

    @property
    def movie(self) -> torch.Tensor:
        """Get the movie tensor."""
        if not self.enabled:
            return None
        return load_mrc_volume(self.movie_path)

    @property
    def deformation_field(self) -> torch.Tensor:
        """Get the deformation field tensor."""
        if not self.enabled:
            return None
        return read_deformation_field_from_csv(self.deformation_field_path)
