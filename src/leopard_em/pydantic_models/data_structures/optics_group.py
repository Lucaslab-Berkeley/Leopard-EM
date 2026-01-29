"""Microscope optics group model for micrograph parameters."""

from os import PathLike
from typing import Annotated, Optional, Union

import torch
from pydantic import Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


class OpticsGroup(BaseModel2DTM):
    """Stores optics group parameters for the imaging system on a microscope.

    Currently utilizes the minimal set of parameters for calculating a
    contrast transfer function (CTF) for a given optics group. Other parameters
    for future use are included but currently unused.

    Attributes
    ----------
    label : str
        Unique string (among other optics groups) for the optics group.
    pixel_size : float
        Pixel size in Angstrom.
    voltage : float
        Voltage in kV.
    spherical_aberration : float
        Spherical aberration in mm. Default is 2.7.
    amplitude_contrast_ratio : float
        Amplitude contrast ratio as a unitless percentage in [0, 1]. Default
        is 0.07.
    phase_shift : float
        Additional phase shift of the contrast transfer function in degrees.
        Default is 0.0 degrees.
    defocus_u : float
        Defocus (underfocus) along the major axis in Angstrom.
    defocus_v : float
        Defocus (underfocus) along the minor axis in Angstrom.
    astigmatism_angle : float
        Angle of defocus astigmatism relative to the X-axis in degrees.
    ctf_B_factor : float
        B-factor to apply in the contrast transfer function in A^2. Default
        is 0.0.

    Unused Attributes:
    ------------------
    chromatic_aberration : float
        Chromatic aberration in mm. Default is ???.
    mtf_reference : str | PathLike
        Path to MTF reference file.
    mtf_values : list[float]
        list of modulation transfer functions values on evenly spaced
        resolution grid [0.0, ..., 0.5].
    beam_tilt_x : float
        Beam tilt X in mrad.
    beam_tilt_y : float
        Beam tilt Y in mrad.
    odd_zernike : Optional[dict[str, float]]
        Optional dict of odd Zernike moments. Possible keys: "Z31c", "Z31s",
        "Z33c", "Z33s".
    even_zernike : Optional[dict[str, float]]
        Optional dict of even Zernike moments. Possible keys: "Z44c", "Z44s", "Z60".
    mag_matrix : Optional[list[float]]
        Optional list of floats of length 4 representing the magnification matrix.

    Methods
    -------
    model_dump()
        Returns a dictionary of the model parameters.
    """

    # Currently implemented parameters
    label: str
    pixel_size: Annotated[float, Field(ge=0.0)]
    voltage: Annotated[float, Field(ge=0.0)]
    spherical_aberration: Annotated[float, Field(ge=0.0, default=2.7)] = 2.7
    amplitude_contrast_ratio: Annotated[float, Field(ge=0.0, le=1.0, default=0.07)] = (
        0.07
    )
    phase_shift: Annotated[float, Field(default=0.0)] = 0.0
    defocus_u: float
    defocus_v: float
    astigmatism_angle: float
    ctf_B_factor: Annotated[float, Field(ge=0.0, default=0.0)] = 0.0

    chromatic_aberration: Optional[Annotated[float, Field(ge=0.0)]] = 0.0
    mtf_reference: Optional[Union[str, PathLike]] = None
    mtf_values: Optional[list[float]] = None
    beam_tilt_x: Optional[float] = None
    beam_tilt_y: Optional[float] = None
    odd_zernikes: Optional[dict[str, float]] = None
    even_zernikes: Optional[dict[str, float]] = None
    mag_matrix: Optional[Annotated[list[float], Field(min_length=4, max_length=4)]] = (
        None
    )

    @property
    def mag_matrix_tensor(self) -> Optional[torch.Tensor]:
        """Convert mag_matrix list to a 2x2 tensor.

        Returns
        -------
        Optional[torch.Tensor]
            A 2x2 tensor representation of the magnification matrix, or None if
            mag_matrix is None. The matrix is constructed from the list as:
            [[mag_matrix[0], mag_matrix[1]],
             [mag_matrix[2], mag_matrix[3]]]
        """
        if self.mag_matrix is None:
            return None
        # mag_matrix is guaranteed to be list[float] of length 4 by Field validation
        # Construct tensor from flat list and reshape
        return torch.tensor(self.mag_matrix, dtype=torch.float32).reshape(2, 2)
