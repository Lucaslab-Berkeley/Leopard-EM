"""Tests for the data_io module"""

import os
import pathlib
import tempfile

import h5py
import mrcfile
import numpy as np
import pytest
import torch

from leopard_em.pydantic_models.formats import HDF5_TENSORS_GROUP
from leopard_em.utils.data_io import (
    atomic_write_path,
    load_mrc_image,
    load_mrc_volume,
    load_result_map_image,
    read_mrc_to_numpy,
    read_mrc_to_tensor,
    write_mrc_from_numpy,
    write_mrc_from_tensor,
)

EXAMPLE_IMAGE_PATH = "data/test_image.mrc"
EXAMPLE_VOLUME_PATH = "data/test_volume.mrc"

# Evaluate the relative paths to this file
EXAMPLE_IMAGE_PATH = pathlib.Path(__file__).parent.parent / EXAMPLE_IMAGE_PATH
EXAMPLE_VOLUME_PATH = pathlib.Path(__file__).parent.parent / EXAMPLE_VOLUME_PATH


def create_test_mrc_file(data: np.ndarray, file_path: str) -> None:
    """Helper fn to create a test MRC file."""
    with mrcfile.new(file_path, overwrite=True) as mrc:
        mrc.set_data(data)


def test_read_mrc_to_numpy():
    """Test the read_mrc_to_numpy function."""
    result = read_mrc_to_numpy(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


def test_read_mrc_to_tensor():
    """Test the read_mrc_to_tensor function."""
    result = read_mrc_to_tensor(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 2


def test_write_mrc_from_numpy():
    """Test writing an MRC file from a numpy array."""
    data = np.random.rand(10, 10).astype(np.float32)

    # Create a temporary file to write to
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as temp_file:
        write_mrc_from_numpy(data, temp_file.name, overwrite=True)
        with mrcfile.open(temp_file.name) as mrc:
            np.testing.assert_array_equal(mrc.data, data)

    # Finally, remove the temporary file
    os.remove(temp_file.name)


def test_write_mrc_from_tensor():
    """Test wriniting an MRC file from a torch tensor."""
    data = torch.rand(10, 10, dtype=torch.float32)

    # Create a temporary file to write to
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as temp_file:
        write_mrc_from_tensor(data, temp_file.name, overwrite=True)
        with mrcfile.open(temp_file.name) as mrc:
            np.testing.assert_array_equal(mrc.data, data.numpy())

    # Finally, remove the temporary file
    os.remove(temp_file.name)


def test_load_mrc_image():
    """Test loading an MRC image into a tensor."""
    result = load_mrc_image(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 2

    # Ensure the method raises a ValueError if the MRC file is not two-dimensional
    with pytest.raises(ValueError, match="MRC file is not two-dimensional"):
        load_mrc_image(EXAMPLE_VOLUME_PATH)


def test_load_result_map_image_mrc_matches_load_mrc_image():
    """MRC paths should be dispatched to identical behavior as load_mrc_image."""
    expected = load_mrc_image(EXAMPLE_IMAGE_PATH)
    result = load_result_map_image(EXAMPLE_IMAGE_PATH)

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, expected)


def test_load_result_map_image_hdf5_reads_named_dataset():
    """HDF5 paths should read the dataset named by 'dataset_name'."""
    psi_data = np.full((4, 5), 7.0, dtype=np.float32)
    defocus_data = np.full((4, 5), 9.0, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        with h5py.File(temp_file.name, "w") as f:
            grp = f.create_group(HDF5_TENSORS_GROUP)
            grp.create_dataset("orientation_psi", data=psi_data)
            grp.create_dataset("relative_defocus", data=defocus_data)

        psi = load_result_map_image(temp_file.name, dataset_name="orientation_psi")
        defocus = load_result_map_image(temp_file.name, dataset_name="relative_defocus")

        assert isinstance(psi, torch.Tensor)
        np.testing.assert_array_equal(psi.numpy(), psi_data)
        np.testing.assert_array_equal(defocus.numpy(), defocus_data)

        # dataset_name is required for HDF5 files
        with pytest.raises(ValueError, match="dataset_name"):
            load_result_map_image(temp_file.name, dataset_name=None)

    os.remove(temp_file.name)


def test_load_mrc_volume():
    """Test loading an MRC volume into a tensor."""
    result = load_mrc_volume(EXAMPLE_VOLUME_PATH)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 3

    # Ensure the method raises a ValueError if the MRC file is not three-dimensional
    with pytest.raises(ValueError, match="MRC file is not three-dimensional"):
        load_mrc_volume(EXAMPLE_IMAGE_PATH)


def _write_tensors_file(path, datasets):
    with h5py.File(path, "w") as f:
        grp = f.create_group(HDF5_TENSORS_GROUP)
        for name, data in datasets.items():
            grp.create_dataset(name, data=data)


@pytest.mark.parametrize("file_name", ["result.hdf", "result.he5", "result"])
def test_load_result_map_image_detects_hdf5_by_content(tmp_path, file_name):
    """HDF5 files are detected by signature, not by file extension."""
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = tmp_path / file_name
    _write_tensors_file(path, {"mip": data})

    result = load_result_map_image(path, dataset_name="mip")
    np.testing.assert_array_equal(result.numpy(), data)


def test_load_result_map_image_hdf5_squeezes_singleton_dims(tmp_path):
    """A (1, H, W) dataset is returned as (H, W), matching the MRC behavior."""
    data = np.ones((1, 4, 5), dtype=np.float64)
    path = tmp_path / "result.h5"
    _write_tensors_file(path, {"mip": data})

    result = load_result_map_image(path, dataset_name="mip")
    assert result.shape == (4, 5)
    assert result.dtype == torch.float32

    _write_tensors_file(path, {"mip": np.ones((2, 4, 5), dtype=np.float32)})
    with pytest.raises(ValueError, match="not two-dimensional"):
        load_result_map_image(path, dataset_name="mip")


def test_load_result_map_image_missing_dataset_names_available(tmp_path):
    """A missing dataset raises a ValueError naming the file and available datasets."""
    path = tmp_path / "result.h5"
    _write_tensors_file(path, {"mip": np.ones((4, 5), dtype=np.float32)})

    with pytest.raises(ValueError, match=r"correlation_variance.*\['mip'\]"):
        load_result_map_image(path, dataset_name="correlation_variance")


def test_load_result_map_image_missing_file(tmp_path):
    """A missing file raises FileNotFoundError, not an MRC parsing error."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_result_map_image(tmp_path / "missing.h5", dataset_name="mip")


def test_atomic_write_path_replaces_and_cleans_up(tmp_path):
    """Successful writes replace the target; failed writes leave it untouched."""
    target = tmp_path / "nested" / "out.txt"

    with atomic_write_path(target) as tmp:
        tmp.write_text("first")
    assert target.read_text() == "first"
    os.chmod(target, 0o640)

    with atomic_write_path(target) as tmp:
        tmp.write_text("second")
    assert target.read_text() == "second"
    assert (target.stat().st_mode & 0o777) == 0o640  # existing mode is preserved

    with pytest.raises(RuntimeError), atomic_write_path(target) as tmp:
        tmp.write_text("partial")
        raise RuntimeError("simulated failure mid-write")
    assert target.read_text() == "second"
    assert sorted(p.name for p in target.parent.iterdir()) == ["out.txt"]
