"""Tests for per-pixel spatial constraint maps and HDF5 I/O."""

from pathlib import Path

import numpy as np
import pytest
import torch

from leopard_em.analysis.zscore_metric import extract_peaks_and_statistics_zscore
from leopard_em.pydantic_models.config import FilamentConstraint, SpatialBox
from leopard_em.pydantic_models.config.spatial_constraint import (
    rasterize_rectangle,
    read_spatial_constraint_hdf5,
    write_spatial_constraint_hdf5,
)


def test_rasterize_rectangle_inclusive_bounds():
    maps = rasterize_rectangle(
        image_shape=(8, 10),
        box=(2.0, 3.0, 4.0, 6.0),
        n_orientations=12,
    )
    assert maps.eligible.shape == (8, 10)
    assert maps.eligible[2:5, 3:7].all()
    assert maps.eligible[1, 3] == 0
    assert maps.eligible[2, 2] == 0
    assert int(maps.n_orientations[3, 4]) == 12
    assert int(maps.n_orientations[0, 0]) == 0
    assert int(maps.region_id[3, 4]) == 1
    expected = int(maps.eligible.sum()) * 12 * 2
    assert maps.allowed_num_ccg(n_defocus=2, n_cs=1) == expected


def test_hdf5_roundtrip(tmp_path: Path):
    maps = rasterize_rectangle(
        (6, 7),
        box=SpatialBox(y0=1, x0=1, y1=3, x1=4),
        n_orientations=5,
    )
    maps.pixel_size_angstrom = 0.9194
    maps.regions = [
        {
            "cone_half_angle_deg": 10.0,
            "theta_center_deg": 90.0,
            "psi_min": 0.0,
            "psi_max": 360.0,
            "psi_step": 1.5,
            "theta_step": 2.5,
            "base_grid_method": "uniform",
            "region_id": 1,
            "box": (1.0, 1.0, 3.0, 4.0),
            "line": (0.0, 0.0, 0.0, 5.0),
            "orientation_configs": [
                {"phi_min": 350.0, "phi_max": 360.0, "symmetry": None}
            ],
        }
    ]
    path = tmp_path / "constraint.h5"
    write_spatial_constraint_hdf5(str(path), maps, leopard_em_version="test")
    loaded = read_spatial_constraint_hdf5(str(path))
    np.testing.assert_array_equal(loaded.eligible, maps.eligible)
    np.testing.assert_array_equal(loaded.n_orientations, maps.n_orientations)
    assert loaded.pixel_size_angstrom == pytest.approx(0.9194)
    assert loaded.regions[0]["cone_half_angle_deg"] == 10.0
    assert loaded.regions[0]["box"] == (1.0, 1.0, 3.0, 4.0)
    assert loaded.regions[0]["orientation_configs"][0]["phi_min"] == 350.0


def test_image_to_stats_map_coords():
    maps = rasterize_rectangle((10, 10), box=(4.0, 4.0, 6.0, 6.0), n_orientations=3)
    stats = maps.to_stats_map_coords(half_template_width=2, stats_shape=(7, 7))
    assert stats.coordinate_frame == "pos_xy"
    assert stats.eligible.shape == (7, 7)
    # stats (2, 2) -> image (4, 4), inside the box
    assert stats.eligible[2, 2] == 1
    # stats (0, 0) -> image (2, 2), outside the box
    assert stats.eligible[0, 0] == 0


def test_filament_constraint_writes_and_loads_spatial_hdf5(tmp_path: Path):
    constraint = FilamentConstraint.from_line(
        y0=0.0,
        x0=0.0,
        y1=0.0,
        x1=10.0,
        cone_half_angle_deg=10.0,
        spatial_box=SpatialBox(y0=2, x0=2, y1=5, x1=6),
    )
    h5_path = tmp_path / "filament_constraint.h5"
    constraint.write_spatial_hdf5(str(h5_path), image_shape=(12, 12))
    constraint.spatial_constraint_path = str(h5_path)

    loaded = constraint.load_spatial_maps()
    assert loaded is not None
    n_orient = int(constraint.to_orientation_config().euler_angles.shape[0])
    assert int(loaded.n_orientations.max()) == n_orient
    assert loaded.eligible[3, 4] == 1
    assert loaded.eligible[0, 0] == 0

    stats = constraint.stats_maps_for_template(image_shape=(12, 12), template_width=4)
    assert stats is not None
    assert stats.coordinate_frame == "pos_xy"
    assert stats.eligible.shape == (9, 9)


def test_num_ccg_and_peaks_outside_box():
    height, width = 8, 8
    scaled = torch.zeros((height, width))
    scaled[1, 1] = 20.0  # outside box
    scaled[4, 4] = 19.0  # inside box
    mip = scaled.clone()
    dummy = torch.zeros((height, width))
    n_orients = torch.zeros((height, width), dtype=torch.int32)
    n_orients[3:6, 3:6] = 10

    peaks = extract_peaks_and_statistics_zscore(
        mip=mip,
        scaled_mip=scaled,
        best_psi=dummy,
        best_theta=dummy,
        best_phi=dummy,
        best_defocus=dummy,
        correlation_average=dummy,
        correlation_variance=torch.ones((height, width)),
        total_correlation_positions=100,
        n_orientations_map=n_orients,
        n_defocus=2,
        n_cs=1,
        z_score_cutoff=1.0,
        mask_radius=1.0,
    )
    assert len(peaks.pos_y) == 1
    assert int(peaks.pos_y[0]) == 4
    assert int(peaks.pos_x[0]) == 4
