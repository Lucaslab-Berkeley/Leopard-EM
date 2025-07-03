import tempfile

import mrcfile
import numpy as np
import pandas as pd
import pytest

from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack

REQUIRED_COLUMNS = [
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
    "psi",
    "theta",
    "phi",
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


def make_minimal_df(num_rows=2):
    """Create a minimal DataFrame with required columns for testing."""
    data = {col: [0] * num_rows for col in REQUIRED_COLUMNS}
    return pd.DataFrame(data)


def make_reference_example_df(
    position_reference: str = "top-left", rng=None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float]:
    """Create example data for particle extraction using position reference.

    Parameters
    ----------
    position_reference : str
        The position reference to use for the MIP. Can be 'top-left' or 'center'.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, uses default_rng.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float]
        A tuple containing:
        - DataFrame with particle data
        - Image region 1 (ground truth) as a numpy array
        - Image region 2 (ground truth) as a numpy array
        - Value of MIP for region 1
        - Value of MIP for region 2
    """
    if rng is None:
        rng = np.random.default_rng()

    df = make_minimal_df()
    position_1 = (0, 0)
    position_2 = (100, 100)
    particle_shape = (32, 32)
    image_shape = (256, 256)
    correlation_shape = (
        image_shape[0] - particle_shape[0] + 1,
        image_shape[1] - particle_shape[1] + 1,
    )

    # Create test image and save to temporary file
    region_1 = rng.normal(size=particle_shape).astype(np.float32)
    region_2 = rng.normal(size=particle_shape).astype(np.float32)
    image = np.zeros(image_shape, dtype=np.float32)
    # Place the regions in the image
    image[
        position_1[1] : position_1[1] + region_1.shape[0],
        position_1[0] : position_1[0] + region_1.shape[1],
    ] = region_1
    image[
        position_2[1] : position_2[1] + region_2.shape[0],
        position_2[0] : position_2[0] + region_2.shape[1],
    ] = region_2
    # Save to temporary mrc file
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tmp_file:
        mrcfile.new(tmp_file.name, data=image, overwrite=True)
        mrc_path = tmp_file.name
    # Set the micrograph path in the DataFrame
    df["micrograph_path"] = mrc_path

    # Create an example MIP statistic map
    mip = np.zeros(correlation_shape, dtype=np.float32)
    mip[position_1] = 10
    mip[position_2] = 20

    # Simulate extra padding for MIP if using center as position reference
    if position_reference == "center":
        # Center the MIP around the particle positions
        mip = np.pad(
            mip,
            (
                (particle_shape[0] // 2, particle_shape[0] // 2 + 1),
                (particle_shape[1] // 2, particle_shape[1] // 2 + 1),
            ),
            mode="constant",
            constant_values=0,
        )
        # Adjust the positions to match the center reference
        position_1 = (
            position_1[0] + particle_shape[1] // 2,
            position_1[1] + particle_shape[0] // 2,
        )
        position_2 = (
            position_2[0] + particle_shape[1] // 2,
            position_2[1] + particle_shape[0] // 2,
        )

    # Save to temporary mrc file
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tmp_file:
        mrcfile.new(tmp_file.name, data=mip, overwrite=True)
        mip_path = tmp_file.name
    # Set the mip path in the DataFrame
    df["mip_path"] = mip_path

    # Only need to set the pos_x and pos_y columns as particle data
    df["pos_x"] = [position_1[0], position_2[0]]
    df["pos_y"] = [position_1[1], position_2[1]]

    return df, region_1, region_2, 10, 20


def test_particle_stack_init_and_load_df(tmp_path):
    """Test initializing ParticleStack and loading DataFrame from CSV."""
    df = make_minimal_df()
    df_path = tmp_path / "particles.csv"
    df.to_csv(df_path, index=False)

    # Should initialize and load without error
    ps = ParticleStack(
        df_path=str(df_path),
        extracted_box_size=(32, 32),
        original_template_size=(16, 16),
    )
    assert isinstance(ps._df, pd.DataFrame)
    assert set(REQUIRED_COLUMNS).issubset(ps._df.columns)
    assert ps.extracted_box_size == (32, 32)
    assert ps.original_template_size == (16, 16)


def test_particle_stack_missing_columns(tmp_path):
    """Test initializing ParticleStack with missing required columns."""
    df = make_minimal_df()
    df = df.drop(columns=["mip"])
    df_path = tmp_path / "particles_missing.csv"
    df.to_csv(df_path, index=False)

    # Should raise ValueError for missing columns
    with pytest.raises(ValueError) as excinfo:
        ParticleStack(
            df_path=str(df_path),
            extracted_box_size=(32, 32),
            original_template_size=(16, 16),
        )
    assert "Missing the following columns" in str(excinfo.value)


def test_particle_stack_skip_df_load():
    """Test initializing ParticleStack with skip_df_load=True."""
    ps = ParticleStack(
        df_path="not_a_real_file.csv",
        extracted_box_size=(32, 32),
        original_template_size=(16, 16),
        skip_df_load=True,
    )
    # _df should not be set
    assert not hasattr(ps, "_df")


def test_particle_stack_image_extraction():
    """Test that the image extraction works correctly for the ParticleStack."""
    df, r1_ground_truth, r2_ground_truth, mip1, mip2 = make_reference_example_df()

    # Add padding to the ground truth images to match the extraction box size
    r1_ground_truth = np.pad(
        r1_ground_truth, ((1, 1), (1, 1)), mode="constant", constant_values=0
    )
    r2_ground_truth = np.pad(
        r2_ground_truth, ((1, 1), (1, 1)), mode="constant", constant_values=0
    )

    # Create ground truth MIP "images"
    mip1_ground_truth = np.array([[0, 0, 0], [0, mip1, 0], [0, 0, 0]], dtype=np.float32)
    mip2_ground_truth = np.array([[0, 0, 0], [0, mip2, 0], [0, 0, 0]], dtype=np.float32)

    ps = ParticleStack(
        df_path="",  # Setting data frame directly, so give empty path
        extracted_box_size=(34, 34),  # This will give a correlation box size of 3x3
        original_template_size=(32, 32),
        skip_df_load=True,
    )
    ps._df = df  # Set the DataFrame directly

    # Default is top-left reference, make sure it works itself
    extracted_images = ps.construct_image_stack()
    extracted_mips = ps.construct_cropped_statistic_stack(stat="mip")

    assert extracted_images.shape == (2, 34, 34)
    assert extracted_mips.shape == (2, 3, 3)

    # Check the first image region
    assert np.allclose(extracted_images[0], r1_ground_truth)
    assert np.allclose(extracted_images[1], r2_ground_truth)

    assert np.allclose(extracted_mips[0], mip1_ground_truth)
    assert np.allclose(extracted_mips[1], mip2_ground_truth)


def test_particle_stack_top_left_and_center_self_consistency():
    """Ensure top-left and center position references are self-consistent."""
    rng = np.random.default_rng(seed=42)
    df_tl, r1_tl, r2_tl, mip1_tl, mip2_tl = make_reference_example_df(
        position_reference="top-left",
        rng=rng,
    )
    rng = np.random.default_rng(seed=42)
    df_center, r1_center, r2_center, mip1_center, mip2_center = (
        make_reference_example_df(
            position_reference="center",
            rng=rng,
        )
    )

    # Add padding to the ground truth images to match the extraction box size
    r1_tl = np.pad(r1_tl, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    r2_tl = np.pad(r2_tl, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    r1_center = np.pad(r1_center, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    r2_center = np.pad(r2_center, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    mip1_tl = np.array([[0, 0, 0], [0, mip1_tl, 0], [0, 0, 0]], dtype=np.float32)
    mip2_tl = np.array([[0, 0, 0], [0, mip2_tl, 0], [0, 0, 0]], dtype=np.float32)
    mip1_center = np.array(
        [[0, 0, 0], [0, mip1_center, 0], [0, 0, 0]], dtype=np.float32
    )
    mip2_center = np.array(
        [[0, 0, 0], [0, mip2_center, 0], [0, 0, 0]], dtype=np.float32
    )

    # Create ParticleStack instances for both references
    ps_tl = ParticleStack(
        df_path="",  # Setting data frame directly, so give empty path
        extracted_box_size=(34, 34),  # This will give a correlation box size of 3x3
        original_template_size=(32, 32),
        skip_df_load=True,
    )
    ps_tl._df = df_tl  # Set the DataFrame directly

    ps_center = ParticleStack(
        df_path="",  # Setting data frame directly, so give empty path
        extracted_box_size=(34, 34),  # This will give a correlation box size of 3x3
        original_template_size=(32, 32),
        skip_df_load=True,
    )
    ps_center._df = df_center  # Set the DataFrame directly

    # Extract images and MIPs for both references
    extracted_images_tl = ps_tl.construct_image_stack(pos_reference="top-left")
    extracted_images_center = ps_center.construct_image_stack(pos_reference="center")

    extracted_mips_tl = ps_tl.construct_cropped_statistic_stack(stat="mip")
    extracted_mips_center = ps_center.construct_cropped_statistic_stack(stat="mip")

    # Check shapes
    assert extracted_images_tl.shape == (2, 34, 34)
    assert extracted_images_center.shape == (2, 34, 34)

    assert extracted_mips_tl.shape == (2, 3, 3)
    assert extracted_mips_center.shape == (2, 3, 3)

    # Check for self-consistency
    assert np.allclose(extracted_images_tl, extracted_images_center)
    assert np.allclose(extracted_mips_tl, extracted_mips_center)

    # Check against ground truth image regions
    assert np.allclose(extracted_images_tl[0], r1_tl)
    assert np.allclose(extracted_images_tl[1], r2_tl)
    assert np.allclose(extracted_images_center[0], r1_center)
    assert np.allclose(extracted_images_center[1], r2_center)

    # Check against ground truth MIPs
    assert np.allclose(extracted_mips_tl[0], mip1_tl)
    assert np.allclose(extracted_mips_tl[1], mip2_tl)
    assert np.allclose(extracted_mips_center[0], mip1_center)
    assert np.allclose(extracted_mips_center[1], mip2_center)
