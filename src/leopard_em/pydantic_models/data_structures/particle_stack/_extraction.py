"""Per-particle loading, cropping and scattering, grouped by source image.

Private module: the particle stack classes are the public interface. Particles are
handled in groups, one group per source image (micrograph or statistic map), where a
group is the list of particle row positions referencing that image. The functions here
take images, groups and crop origins explicitly (origins come from
``ParticleBoxGeometry.box_origins``) and never touch a particle stack.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from leopard_em.pydantic_models.data_structures.particle_stack._table import (
    as_row_positions,
)
from leopard_em.pydantic_models.formats import STATISTIC_MAP_PATH_TO_HDF5_DATASET
from leopard_em.utils.cropping import get_cropped_image_regions
from leopard_em.utils.data_io import load_result_map_image


def load_grouped_images(
    groups: dict[Any, np.ndarray], column_name: str
) -> tuple[torch.Tensor, list[pd.Index]]:
    """Load the image referenced by each group.

    Parameters
    ----------
    groups : dict[Any, np.ndarray]
        Maps each image path to the row positions of its particles (see
        ``_table.group_rows``).
    column_name : str
        The column the paths come from. Statistic-map columns may reference a
        ``MatchTemplateResultHDF5`` file, read from that column's dataset.

    Returns
    -------
    tuple[torch.Tensor, list[pd.Index]]
        The ``(num_images, H, W)`` image stack and, per image, the row positions of
        its particles.
    """
    dataset_name = STATISTIC_MAP_PATH_TO_HDF5_DATASET.get(column_name)

    images_list = []
    indices = []
    for img_path, positions in groups.items():
        try:
            img = load_result_map_image(img_path, dataset_name=dataset_name)
        except (FileNotFoundError, ValueError) as err:
            raise type(err)(
                f"Could not load '{column_name}' for {len(positions)} particle(s): "
                f"{err}"
            ) from err
        images_list.append(img)
        indices.append(pd.Index(positions))

    images_tensor = torch.stack(images_list, dim=0)
    return images_tensor, indices


def _check_group_count(num_images: int, indices: list[pd.Index]) -> None:
    if num_images != len(indices):
        raise ValueError(
            f"Number of images ({num_images}) does not match the number of "
            f"indices ({len(indices)})."
        )


# pylint: disable=too-many-arguments
def crop_by_group(
    images: torch.Tensor,
    indices: list[pd.Index],
    origin_y: np.ndarray,
    origin_x: np.ndarray,
    extraction_size: tuple[int, int],
    *,
    handle_bounds: Literal["pad", "error"] = "pad",
    padding_mode: Literal["constant", "reflect", "replicate"] = "constant",
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Crop one region per particle from the image its group belongs to.

    Parameters
    ----------
    images : torch.Tensor
        Source images with shape ``(num_images, H, W)``.
    indices : list[pd.Index]
        Row positions of the particles belonging to each image.
    origin_y : np.ndarray
        Top-left crop y coordinate of *every* particle (one entry per row).
    origin_x : np.ndarray
        Top-left crop x coordinate of every particle.
    extraction_size : tuple[int, int]
        Size ``(h, w)`` of each crop.
    handle_bounds : Literal["pad", "error"], optional
        How to handle out-of-bounds regions, by default "pad".
    padding_mode : Literal["constant", "reflect", "replicate"], optional
        Padding mode when ``handle_bounds="pad"``, by default "constant".
    padding_value : float, optional
        Constant padding value, by default 0.0.

    Returns
    -------
    torch.Tensor
        ``(num_particles, h, w)`` crops on the images' device. Rows not in any group
        are zero.
    """
    device = images.device
    num_particles = len(origin_y)
    region_stack = torch.zeros((num_particles, *extraction_size), device=device)
    _check_group_count(images.shape[0], indices)

    for i, indexes in enumerate(indices):
        img = images[i]
        positions = as_row_positions(indexes, num_particles)
        cropped_images = get_cropped_image_regions(
            img,
            torch.tensor(origin_y[positions], device=img.device),
            torch.tensor(origin_x[positions], device=img.device),
            extraction_size,
            pos_reference="top-left",
            handle_bounds=handle_bounds,
            padding_mode=padding_mode,
            padding_value=padding_value,
        )
        region_stack[torch.as_tensor(positions, device=device)] = cropped_images

    return region_stack


def scatter_by_group(
    values: torch.Tensor, indices: list[pd.Index], num_particles: int
) -> torch.Tensor:
    """Give every particle of a group a copy of that group's value.

    Parameters
    ----------
    values : torch.Tensor
        One value per group (image), shape ``(num_images, ...)``.
    indices : list[pd.Index]
        Row positions of the particles belonging to each image.
    num_particles : int
        Number of rows in the particle table.

    Returns
    -------
    torch.Tensor
        ``(num_particles, ...)`` tensor on ``values``' device. Rows not in any group
        are zero.
    """
    _check_group_count(values.shape[0], indices)
    device = values.device
    out = torch.zeros((num_particles, *values.shape[1:]), device=device)
    for i, indexes in enumerate(indices):
        positions = torch.as_tensor(
            as_row_positions(indexes, num_particles), device=device
        )
        out[positions] = values[i]
    return out
