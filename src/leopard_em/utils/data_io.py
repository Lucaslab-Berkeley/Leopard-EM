"""Utility functions dealing with basic data I/O operations."""

import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import mrcfile
import numpy as np
import pandas as pd
import torch

from leopard_em.pydantic_models.formats import HDF5_TENSORS_GROUP


@contextmanager
def atomic_write_path(path: str | os.PathLike | Path) -> Iterator[Path]:
    """Context manager yielding a temporary path that atomically replaces ``path``.

    Notes
    -----
    Write the output to the yielded path. If the ``with`` block succeeds, the temporary
    file replaces ``path`` in a single ``os.replace`` call, so readers never see a
    partially written file and a failed write leaves any existing file intact. If the
    block raises, the temporary file is removed. An existing target's file mode is
    preserved; otherwise the file gets the default (umask) permissions.

    Parameters
    ----------
    path : str | os.PathLike | Path
        The final output path. Its parent directory is created if needed.

    Yields
    ------
    Path
        A temporary path in the same directory as ``path``.

    Raises
    ------
    PermissionError
        If the parent directory is not writable.
    """
    target = Path(path)
    if target.is_symlink():
        target = target.resolve()
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(
            f"Directory '{directory}' does not permit writing to '{target}'."
        )

    tmp_path = directory / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        yield tmp_path
        if target.exists():
            shutil.copymode(target, tmp_path)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_mrc_to_numpy(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    """Reads an MRC file and returns the data as a numpy array.

    Attributes
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        The MRC data as a numpy array, copied.
    """
    with mrcfile.open(mrc_path) as mrc:
        return mrc.data.copy()


def read_mrc_to_tensor(mrc_path: str | os.PathLike | Path) -> torch.Tensor:
    """Reads an MRC file and returns the data as a torch tensor.

    Attributes
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC data as a tensor, copied and converted to float32 if needed.
    """
    tensor = torch.tensor(read_mrc_to_numpy(mrc_path))
    # Convert float16 to float32 for FFT compatibility
    if tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)
    return tensor


def write_mrc_from_numpy(
    data: np.ndarray,
    mrc_path: str | os.PathLike | Path,
    mrc_header: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Writes a numpy array to an MRC file.

    NOTE: Writing header information is not currently implemented.

    Attributes
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    overwrite : bool
        Overwrite argument passed to mrcfile.new. Default is False.
    """
    if mrc_header is not None:
        raise NotImplementedError("Setting header info is not yet implemented.")

    with mrcfile.new(mrc_path, overwrite=overwrite) as mrc:
        mrc.set_data(data)


def write_mrc_from_tensor(
    data: torch.Tensor,
    mrc_path: str | os.PathLike | Path,
    mrc_header: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Writes a tensor array to an MRC file.

    NOTE: Not currently implemented.

    Attributes
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    overwrite : bool
        Overwrite argument passed to mrcfile.new. Default is False.
    """
    write_mrc_from_numpy(data.numpy(), mrc_path, mrc_header, overwrite)


def load_mrc_image(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an two-dimensional MRC image into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC image as a tensor, converted to float32 for FFT compatibility.

    Raises
    ------
    ValueError
        If the MRC file is not two-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    return _squeeze_to_2d(tensor, source="MRC file", location=str(file_path))


def _squeeze_to_2d(tensor: torch.Tensor, source: str, location: str) -> torch.Tensor:
    """Squeeze singleton dimensions and check the result is a 2D image."""
    tensor = tensor.squeeze()
    if tensor.ndim != 2:
        raise ValueError(
            f"{source} is not two-dimensional. Got shape: {tuple(tensor.shape)} "
            f"(from {location})."
        )
    return tensor


def load_result_map_image(
    file_path: str | os.PathLike | Path, dataset_name: str | None = None
) -> torch.Tensor:
    """Load a single 2DTM result map, dispatching on the file's storage back-end.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC or HDF5 file.
    dataset_name : str | None
        Name of the dataset to read from the HDF5 file's ``tensors`` group.
        Ignored for MRC files. Required when ``file_path`` is an HDF5 file.

    Returns
    -------
    torch.Tensor
        The result map as a 2D tensor (float32 for HDF5 files).

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If ``file_path`` is an HDF5 file and ``dataset_name`` is ``None`` or the
        dataset is absent, or if the map is not two-dimensional.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Result map file '{file_path}' does not exist.")

    if not h5py.is_hdf5(file_path):
        return load_mrc_image(file_path)

    if dataset_name is None:
        raise ValueError(
            "'dataset_name' is required to load a result map from an HDF5 "
            f"file, but got None for file '{file_path}'."
        )
    with h5py.File(file_path, "r") as f:
        group = f.get(HDF5_TENSORS_GROUP)
        if not isinstance(group, h5py.Group) or dataset_name not in group:
            available = sorted(group.keys()) if isinstance(group, h5py.Group) else []
            raise ValueError(
                f"HDF5 result file '{file_path}' has no dataset "
                f"'{HDF5_TENSORS_GROUP}/{dataset_name}'. Available datasets in "
                f"'{HDF5_TENSORS_GROUP}': {available}."
            )
        data = group[dataset_name][()]

    tensor = torch.from_numpy(data).to(torch.float32)
    return _squeeze_to_2d(
        tensor,
        source="HDF5 result dataset",
        location=f"'{HDF5_TENSORS_GROUP}/{dataset_name}' in '{file_path}'",
    )


def load_mrc_volume(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an three-dimensional MRC volume into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC volume as a tensor, converted to float32 for FFT compatibility.

    Raises
    ------
    ValueError
        If the MRC file is not three-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 3D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 3:
        raise ValueError(
            f"MRC file is not three-dimensional. Got shape: {tensor.shape}"
        )

    return tensor


def load_template_tensor(
    template_volume: torch.Tensor | Any | None = None,
    template_volume_path: str | os.PathLike | Path | None = None,
) -> torch.Tensor:
    """Load and convert template volume to a torch.Tensor.

    This function ensures that the template volume is a torch.Tensor.
    If template_volume is None, it loads the volume from template_volume_path.
    If template_volume is not a torch.Tensor, it converts it to one.

    Parameters
    ----------
    template_volume : Optional[Union[torch.Tensor, Any]], optional
        The template volume object, by default None
    template_volume_path : Optional[Union[str, os.PathLike, Path]], optional
        Path to the template volume file, by default None

    Returns
    -------
    torch.Tensor
        The template volume as a torch.Tensor

    Raises
    ------
    ValueError
        If both template_volume and template_volume_path are None
    """
    if template_volume is None:
        if template_volume_path is None:
            raise ValueError("template_volume or template_volume_path must be provided")
        template_volume = load_mrc_volume(template_volume_path)

    if not isinstance(template_volume, torch.Tensor):
        template = torch.from_numpy(template_volume)
    else:
        template = template_volume

    # Convert float16 to float32 for FFT compatibility
    if template.dtype == torch.float16:
        template = template.to(torch.float32)

    return template


def read_particle_shifts_from_csv(
    csv_path: str | os.PathLike | Path,
    num_frames: int,
    num_particles: int,
) -> torch.Tensor:
    """Read particle shifts from a CSV file and convert to tensor format.

    The CSV file should have columns: particle_index, frame, y_shift, x_shift.
    The output tensor has shape (T, N, 2) where T is the number of frames,
    N is the number of particles, and 2 represents (y_shift, x_shift).

    Parameters
    ----------
    csv_path : str | os.PathLike | Path
        Path to the CSV file containing particle shifts.
    num_frames : int
        Number of frames in the movie.
    num_particles : int
        Number of particles.

    Returns
    -------
    torch.Tensor
        Particle shifts tensor with shape (T, N, 2) where T is frames,
        N is particles, and 2 is (y_shift, x_shift).
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = ["particle_index", "frame", "y_shift", "x_shift"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV file must have columns: {required_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Initialize output tensor with zeros
    shifts = torch.zeros((num_frames, num_particles, 2), dtype=torch.float32)

    # Fill in the shifts from the CSV
    for _, row in df.iterrows():
        particle_idx = int(row["particle_index"])
        frame_idx = int(row["frame"])
        y_shift = float(row["y_shift"])
        x_shift = float(row["x_shift"])

        # Validate indices
        if particle_idx < 0 or particle_idx >= num_particles:
            raise ValueError(
                f"Particle index {particle_idx} out of range [0, {num_particles})"
            )
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {num_frames})")

        shifts[frame_idx, particle_idx, 0] = y_shift
        shifts[frame_idx, particle_idx, 1] = x_shift

    return shifts
