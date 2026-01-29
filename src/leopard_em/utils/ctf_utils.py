"""CTF (Contrast Transfer Function) utility functions."""

import ast
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch_ctf import calculate_ctf_2d
from torch_fourier_filter.envelopes import b_envelope

from leopard_em.utils.search_utils import get_cs_range

# Using the TYPE_CHECKING statement to avoid circular imports
if TYPE_CHECKING:
    from leopard_em.pydantic_models.data_structures.optics_group import OpticsGroup
    from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack


def calculate_ctf_filter_stack_full_args(
    template_shape: tuple[int, int],
    defocus_u: float,  # in Angstrom
    defocus_v: float,  # in Angstrom
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
    **kwargs: Any,
) -> torch.Tensor:
    """Calculate a CTF filter stack for a given set of parameters and search offsets.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    defocus_u : float
        Defocus along the major axis, in Angstroms.
    defocus_v : float
        Defocus along the minor axis, in Angstroms.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.
    **kwargs
        Additional keyword to pass to the calculate_ctf_2d function.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and parameters. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)
    """
    # Calculate the defocus values + offsets in terms of Angstrom
    defocus = defocus_offsets + ((defocus_u + defocus_v) / 2)
    astigmatism = abs(defocus_u - defocus_v) / 2

    # The different Cs values to search over as another dimension
    cs_values = get_cs_range(
        pixel_size=kwargs["pixel_size"],
        pixel_size_offsets=pixel_size_offsets,
        cs=kwargs["spherical_aberration"],
    )

    # Ensure defocus and astigmatism have a batch dimension so Cs and defocus can be
    # interleaved correctly
    if defocus.dim() == 0:
        defocus = defocus.unsqueeze(0)

    # Convert mag_matrix from list to 2x2 tensor if provided
    transform_matrix = kwargs.get("mag_matrix")
    if transform_matrix is not None:
        if isinstance(transform_matrix, list):
            transform_matrix = torch.tensor(
                [
                    [transform_matrix[0], transform_matrix[1]],
                    [transform_matrix[2], transform_matrix[3]],
                ],
                dtype=torch.float32,
            )
        elif not isinstance(transform_matrix, torch.Tensor):
            # If it's neither a list nor a tensor, try to convert it
            transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32)

    # Loop over spherical aberrations one at a time and collect results
    ctf_list = []
    for cs_val in cs_values:
        tmp = calculate_ctf_2d(
            defocus=defocus * 1e-4,  # Convert to um from Angstrom
            astigmatism=astigmatism * 1e-4,  # Convert to um from Angstrom
            astigmatism_angle=kwargs["astigmatism_angle"],
            voltage=kwargs["voltage"],
            spherical_aberration=cs_val,
            amplitude_contrast=kwargs["amplitude_contrast_ratio"],
            phase_shift=kwargs["phase_shift"],
            pixel_size=kwargs["pixel_size"],
            image_shape=template_shape,
            rfft=True,
            fftshift=False,
            even_zernike_coeffs=kwargs["even_zernikes"],
            odd_zernike_coeffs=kwargs["odd_zernikes"],
            transform_matrix=transform_matrix,
        )
        # calc B-envelope and apply
        b_envelope_tmp = b_envelope(
            B=kwargs["ctf_B_factor"],
            image_shape=template_shape,
            pixel_size=kwargs["pixel_size"],
            rfft=True,
            fftshift=False,
            device=tmp.device,
        )
        tmp *= b_envelope_tmp
        ctf_list.append(tmp)

    ctf = torch.stack(ctf_list, dim=0)

    return ctf


def calculate_ctf_filter_stack(
    template_shape: tuple[int, int],
    optics_group: "OpticsGroup",
    defocus_offsets: torch.Tensor,  # in Angstrom, relative
    pixel_size_offsets: torch.Tensor,  # in Angstrom, relative
) -> torch.Tensor:
    """Calculate searched CTF filter values for a given shape and optics group.

    Parameters
    ----------
    template_shape : tuple[int, int]
        Desired output shape for the filter, in real space.
    optics_group : OpticsGroup
        OpticsGroup object containing the optics defining the CTF parameters.
    defocus_offsets : torch.Tensor
        Tensor of defocus offsets to search over, in Angstroms.
    pixel_size_offsets : torch.Tensor
        Tensor of pixel size offsets to search over, in Angstroms.

    Returns
    -------
    torch.Tensor
        Tensor of CTF filter values for the specified shape and optics group. Will have
        shape (num_pixel_sizes, num_defocus_offsets, h, w // 2 + 1)
    """
    return calculate_ctf_filter_stack_full_args(
        template_shape,
        optics_group.defocus_u,
        optics_group.defocus_v,
        defocus_offsets,
        pixel_size_offsets,
        astigmatism_angle=optics_group.astigmatism_angle,
        voltage=optics_group.voltage,
        spherical_aberration=optics_group.spherical_aberration,
        amplitude_contrast_ratio=optics_group.amplitude_contrast_ratio,
        ctf_B_factor=optics_group.ctf_B_factor,
        phase_shift=optics_group.phase_shift,
        pixel_size=optics_group.pixel_size,
        even_zernikes=optics_group.even_zernikes,
        odd_zernikes=optics_group.odd_zernikes,
        mag_matrix=optics_group.mag_matrix_tensor,
    )


def _parse_json_string_from_series_value(value: Any) -> Any:
    """Parse a value that may be a JSON string, dict, None, or NaN.

    Parameters
    ----------
    value : Any
        The value to parse. Can be a JSON string, dict, None, or NaN
        (from empty CSV fields).

    Returns
    -------
    Any
        Parsed dict if value was a JSON string, original dict if already a dict,
        or None if value was None or NaN.
    """
    # Handle NaN values from empty CSV fields (pandas converts empty fields to NaN)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and value:
        return json.loads(value)
    # Already a dict (backward compatibility)
    return value


def _setup_ctf_kwargs_from_particle_stack(
    particle_stack: "ParticleStack", template_shape: tuple[int, int]
) -> dict[str, Any]:
    """Helper function for per-particle CTF kwargs.

    Parameters
    ----------
    particle_stack : ParticleStack
        The particle stack to extract the CTF parameters from.
    template_shape : tuple[int, int]
        The shape of the template to use for the CTF calculation.

    Returns
    -------
    dict[str, Any]
        A dictionary of CTF parameters to pass to the CTF calculation function.
    """
    # Keyword arguments for the CTF filter calculation call
    # NOTE: We currently enforce the parameters (other than the defocus values) are
    # all the same. This could be updated in the future...
    assert particle_stack["voltage"].nunique() == 1
    assert particle_stack["spherical_aberration"].nunique() == 1
    assert particle_stack["amplitude_contrast_ratio"].nunique() == 1
    assert particle_stack["phase_shift"].nunique() == 1
    assert particle_stack["ctf_B_factor"].nunique() == 1

    # Convert mag_matrix from list to 2x2 tensor if provided
    # Handle empty/NaN values from CSV (pandas converts empty fields to NaN)
    mag_matrix_value = particle_stack["mag_matrix"].iloc[0]
    mag_matrix_tensor = None
    if mag_matrix_value is not None and not (
        isinstance(mag_matrix_value, float) and np.isnan(mag_matrix_value)
    ):
        # mag_matrix_value might be a list or a string representation of a list
        if isinstance(mag_matrix_value, str):
            mag_matrix_list = ast.literal_eval(mag_matrix_value)
        else:
            mag_matrix_list = mag_matrix_value
        if isinstance(mag_matrix_list, list) and len(mag_matrix_list) == 4:
            # Check that all elements are valid numbers (not NaN)
            if all(
                isinstance(x, (int, float)) and not np.isnan(x) for x in mag_matrix_list
            ):
                mag_matrix_tensor = torch.tensor(
                    [
                        [mag_matrix_list[0], mag_matrix_list[1]],
                        [mag_matrix_list[2], mag_matrix_list[3]],
                    ],
                    dtype=torch.float32,
                )

    # Parse JSON strings for zernike coefficients if they're stored as strings
    even_zernikes_dict = _parse_json_string_from_series_value(
        particle_stack["even_zernikes"].iloc[0]
    )
    odd_zernikes_dict = _parse_json_string_from_series_value(
        particle_stack["odd_zernikes"].iloc[0]
    )

    # Convert dictionary values to tensors
    if even_zernikes_dict is not None:
        even_zernikes_dict = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in even_zernikes_dict.items()
        }
    if odd_zernikes_dict is not None:
        odd_zernikes_dict = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in odd_zernikes_dict.items()
        }

    return {
        "voltage": particle_stack["voltage"][0].item(),
        "spherical_aberration": particle_stack["spherical_aberration"][0].item(),
        "amplitude_contrast_ratio": particle_stack["amplitude_contrast_ratio"][
            0
        ].item(),
        "ctf_B_factor": particle_stack["ctf_B_factor"][0].item(),
        "phase_shift": particle_stack["phase_shift"][0].item(),
        "pixel_size": particle_stack["refined_pixel_size"].mean().item(),
        "template_shape": template_shape,
        "even_zernikes": even_zernikes_dict,
        "odd_zernikes": odd_zernikes_dict,
        "mag_matrix": mag_matrix_tensor,
    }
