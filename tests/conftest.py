"""Shared fixtures: a tiny synthetic match_template workspace on disk."""

from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER
from leopard_em.pydantic_models.results.match_template_result import (
    MatchTemplateResultHDF5,
)

TEMPLATE_SIZE = 16
BOX_SIZE = 20
MICROGRAPH_SIZE = 64

STATISTIC_COLUMN_TO_TENSOR_NAME = {
    "mip_path": "mip",
    "scaled_mip_path": "scaled_mip",
    "psi_path": "orientation_psi",
    "theta_path": "orientation_theta",
    "phi_path": "orientation_phi",
    "defocus_path": "relative_defocus",
    "correlation_average_path": "correlation_average",
    "correlation_variance_path": "correlation_variance",
}


@dataclass
class Workspace:
    """Paths and in-memory data of a synthetic match_template output."""

    root: Path
    template_path: Path
    micrograph_paths: list[Path]
    result_paths: list[Path]
    df: pd.DataFrame

    @property
    def extracted_box_size(self) -> tuple[int, int]:
        return (BOX_SIZE, BOX_SIZE)

    @property
    def original_template_size(self) -> tuple[int, int]:
        return (TEMPLATE_SIZE, TEMPLATE_SIZE)


def _write_mrc(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    """Two micrographs with bundled HDF5 results and a 6-particle table.

    Particles from the two micrographs are interleaved so position/label handling is
    exercised, and the statistic maps vary spatially.
    """
    rng = np.random.default_rng(0)
    template_path = tmp_path / "template.mrc"
    _write_mrc(template_path, rng.normal(size=(TEMPLATE_SIZE,) * 3))

    valid = MICROGRAPH_SIZE - TEMPLATE_SIZE + 1
    micrograph_paths, result_paths = [], []
    for i in range(2):
        mic_path = tmp_path / f"mic_{i}" / "micrograph.mrc"  # same stem, different dir
        mic_path.parent.mkdir()
        _write_mrc(mic_path, rng.normal(size=(MICROGRAPH_SIZE, MICROGRAPH_SIZE)))
        micrograph_paths.append(mic_path)

        result_path = tmp_path / f"mic_{i}" / "result.h5"
        tensors = {
            name: torch.from_numpy(rng.uniform(0.5, 1.5, size=(valid, valid)))
            .to(torch.float32)
            .contiguous()
            for name in STATISTIC_COLUMN_TO_TENSOR_NAME.values()
        }
        MatchTemplateResultHDF5(
            hdf5_path=str(result_path),
            allow_file_overwrite=True,
            total_projections=1,
            total_orientations=1,
            total_defocus=1,
            **tensors,
        ).to_hdf5()
        result_paths.append(result_path)

    source = [0, 1, 0, 1, 1, 0]
    num = len(source)
    df = pd.DataFrame({col: [0.0] * num for col in MATCH_TEMPLATE_DF_COLUMN_ORDER})
    df["particle_index"] = list(range(num))
    df["total_correlations"] = [1] * num
    df["pos_x"] = [3, 10, 25, 40, 7, 30]
    df["pos_y"] = [5, 30, 12, 20, 44, 36]
    df["phi"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    df["theta"] = [15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
    df["psi"] = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0]
    df["defocus_u"] = [12000.0] * num
    df["defocus_v"] = [11800.0] * num
    df["pixel_size"] = [1.0] * num
    df["refined_pixel_size"] = [1.0] * num
    df["voltage"] = [300.0] * num
    df["spherical_aberration"] = [2.7] * num
    df["amplitude_contrast_ratio"] = [0.07] * num
    df["mag_matrix"] = [[1.0, 0.0, 0.0, 1.0]] * num
    df["even_zernikes"] = [None] * num
    df["odd_zernikes"] = [None] * num
    df["micrograph_path"] = [str(micrograph_paths[i]) for i in source]
    df["template_path"] = str(template_path)
    for column in STATISTIC_COLUMN_TO_TENSOR_NAME:
        df[column] = [str(result_paths[i]) for i in source]

    return Workspace(
        root=tmp_path,
        template_path=template_path,
        micrograph_paths=micrograph_paths,
        result_paths=result_paths,
        df=df,
    )
