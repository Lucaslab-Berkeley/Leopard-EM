"""Submodule for shared formats used in pydantic models."""

from collections.abc import Iterable

# Full-micrograph 2DTM result map paths, shared across the column-order
# lists below and by ``_DEFAULT_LOCAL_STAT_COLUMNS`` in particle_stack.py.
STATISTIC_MAP_PATH_COLUMNS = [
    "mip_path",
    "scaled_mip_path",
    "psi_path",
    "theta_path",
    "phi_path",
    "defocus_path",
]

# Unique, human-readable identifier column for each particle. Not indexed currently.
PARTICLE_ID_COLUMN = "particle_id"

# Name of the HDF5 group holding result tensors in a ``MatchTemplateResultHDF5`` file.
HDF5_TENSORS_GROUP = "tensors"

# Maps each dataframe "*_path" statistic-map column to the dataset name it corresponds
# to inside a ``MatchTemplateResultHDF5`` file's ``tensors`` group.
STATISTIC_MAP_PATH_TO_HDF5_DATASET = {
    "mip_path": "mip",
    "scaled_mip_path": "scaled_mip",
    "psi_path": "orientation_psi",
    "theta_path": "orientation_theta",
    "phi_path": "orientation_phi",
    "defocus_path": "relative_defocus",
    "correlation_average_path": "correlation_average",
    "correlation_variance_path": "correlation_variance",
}

# Scalar CTF parameter columns which must hold one value shared by all particles
SHARED_CTF_PARAMETER_COLUMNS = (
    "voltage",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
)

MATCH_TEMPLATE_DF_COLUMN_ORDER = [
    "particle_index",
    "mip",
    "scaled_mip",
    "correlation_mean",
    "correlation_variance",
    "total_correlations",
    "pos_x",
    "pos_y",
    "pos_x_img",
    "pos_y_img",
    "pos_x_img_angstrom",
    "pos_y_img_angstrom",
    "phi",
    "theta",
    "psi",
    "relative_defocus",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "pixel_size",
    "refined_pixel_size",
    "voltage",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
    "even_zernikes",
    "odd_zernikes",
    "mag_matrix",
    "micrograph_path",
    "template_path",
    "mip_path",
    "scaled_mip_path",
    "psi_path",
    "theta_path",
    "phi_path",
    "defocus_path",
    "correlation_average_path",
    "correlation_variance_path",
]

REFINED_DF_COLUMN_ORDER = [
    "particle_index",
    "mip",
    "scaled_mip",
    "refined_mip",
    "refined_scaled_mip",
    "correlation_mean",
    "correlation_variance",
    "total_correlations",
    "pos_x",
    "pos_y",
    "pos_x_img",
    "pos_y_img",
    "pos_x_img_angstrom",
    "pos_y_img_angstrom",
    "refined_pos_x",
    "refined_pos_y",
    "refined_pos_x_img",
    "refined_pos_y_img",
    "refined_pos_x_img_angstrom",
    "refined_pos_y_img_angstrom",
    "phi",
    "theta",
    "psi",
    "refined_phi",
    "refined_theta",
    "refined_psi",
    "relative_defocus",
    "refined_relative_defocus",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "pixel_size",
    "refined_pixel_size",
    "voltage",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
    "even_zernikes",
    "odd_zernikes",
    "mag_matrix",
    "micrograph_path",
    "template_path",
    "mip_path",
    "scaled_mip_path",
    "psi_path",
    "theta_path",
    "phi_path",
    "defocus_path",
    "correlation_average_path",
    "correlation_variance_path",
]

CONSTRAINED_DF_COLUMN_ORDER = [
    "particle_index",
    "mip",
    "scaled_mip",
    "refined_mip",
    "refined_scaled_mip",
    "correlation_mean",
    "correlation_variance",
    "total_correlations",
    "pos_x",
    "pos_y",
    "pos_x_img",
    "pos_y_img",
    "pos_x_img_angstrom",
    "pos_y_img_angstrom",
    "refined_pos_x",
    "refined_pos_y",
    "refined_pos_x_img",
    "refined_pos_y_img",
    "refined_pos_x_img_angstrom",
    "refined_pos_y_img_angstrom",
    "phi",
    "theta",
    "psi",
    "refined_phi",
    "refined_theta",
    "refined_psi",
    "original_offset_phi",
    "original_offset_theta",
    "original_offset_psi",
    "relative_defocus",
    "refined_relative_defocus",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "pixel_size",
    "refined_pixel_size",
    "voltage",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
    "even_zernikes",
    "odd_zernikes",
    "mag_matrix",
    "micrograph_path",
    "template_path",
    "mip_path",
    "scaled_mip_path",
    "psi_path",
    "theta_path",
    "phi_path",
    "defocus_path",
    "correlation_average_path",
    "correlation_variance_path",
]


def result_column_order(
    column_order: list[str], available_columns: Iterable[str]
) -> list[str]:
    """Return ``column_order``, prefixed with ``particle_id`` when it is available."""
    if PARTICLE_ID_COLUMN in available_columns:
        return [PARTICLE_ID_COLUMN, *column_order]
    return list(column_order)
