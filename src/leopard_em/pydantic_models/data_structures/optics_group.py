"""Microscope optics group model for micrograph parameters."""

from os import PathLike
from typing import Annotated, Optional, Union

import torch
from pydantic import Field

from leopard_em.pydantic_models.custom_types import BaseModel2DTM


class LaserParams(BaseModel2DTM):
    """Laser phase plate parameters for optics groups using a laser phase plate.

    Default enabled is False (omit or set laser_params to null when not used).
    Include this block only when the optics group uses a laser phase plate.

    Attributes
    ----------
    NA : float
        Numerical aperture.
    laser_wavelength_angstrom : float
        Laser wavelength in Angstrom.
    focal_length_angstrom : float
        Focal length in Angstrom.
    laser_xy_angle_deg : float
        Laser angle in the XY plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the XZ plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset in Angstrom.
    laser_trans_offset_angstrom : float
        Transverse offset in Angstrom.
    laser_polarization_angle_deg : float
        Laser polarization angle in degrees.
    peak_phase_deg : float
        Peak phase in degrees.
    dual_laser : bool
        Whether a dual-laser setup is used. Default is False.
    """

    NA: float = 0.055
    laser_wavelength_angstrom: float = 10640
    focal_length_angstrom: float = 7.1e7
    laser_xy_angle_deg: float = 0.0
    laser_xz_angle_deg: float = 0.0
    laser_long_offset_angstrom: float = 0.0
    laser_trans_offset_angstrom: float = 0.0
    laser_polarization_angle_deg: float = 90.0
    peak_phase_deg: float = 45.0
    dual_laser: bool = True


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
    laser_params : Optional[LaserParams]
        Optional laser phase plate parameters. Omit or set to null when not using
        a laser phase plate. When present, the optics group uses a laser phase
        plate with the given parameters.

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
    laser_params: Optional[LaserParams] = None

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
