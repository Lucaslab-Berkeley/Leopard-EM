"""Utility functions dealing with basic data I/O operations."""

import os
from pathlib import Path
from typing import Any, Optional, Union

import mrcfile
import numpy as np
import torch


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
    mrc_header: Optional[dict] = None,
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
    mrc_header: Optional[dict] = None,
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

    # Check that tensor is 2D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 2:
        raise ValueError(f"MRC file is not two-dimensional. Got shape: {tensor.shape}")

    return tensor


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
    template_volume: Optional[Union[torch.Tensor, Any]] = None,
    template_volume_path: Optional[Union[str, os.PathLike, Path]] = None,
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
