"""Tests for per-pixel spatial constraint maps and HDF5 I/O."""

from pathlib import Path

import numpy as np
import pytest
import torch

from leopard_em.analysis.zscore_metric import extract_peaks_and_statistics_zscore
from leopard_em.pydantic_models.config import FilamentConstraint, SpatialBox
from leopard_em.pydantic_models.config.spatial_constraint import (
    rasterize_polygon,
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


def test_legacy_aabb_yaml_becomes_four_corners():
    box = SpatialBox(y0=2, x0=3, y1=5, x1=6)
    assert len(box.corners) == 4
    assert box.as_ymin_xmin_ymax_xmax() == (2.0, 3.0, 5.0, 6.0)


def test_rasterize_sheared_quad_excludes_aabb_outside():
    corners = np.array(
        [[1.0, 1.0], [1.0, 5.0], [5.0, 7.0], [5.0, 3.0]],
        dtype=np.float64,
    )
    maps = rasterize_polygon((8, 10), corners=corners, n_orientations=2)
    assert maps.eligible[2, 4] == 1
    assert maps.eligible[2, 6] == 0
    assert maps.eligible[0, 0] == 0
    assert len(maps.regions[0]["box"]) == 4


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


def test_expand_orientation_from_region_boxes():
    maps = rasterize_rectangle((6, 6), box=(2.0, 2.0, 4.0, 4.0), n_orientations=99)
    maps.regions[0]["orientation_configs"] = [
        {
            "phi_min": 40.0,
            "phi_max": 50.0,
            "theta_min": 80.0,
            "theta_max": 100.0,
            "psi_min": 0.0,
            "psi_max": 360.0,
        }
    ]
    euler = np.array(
        [
            [45.0, 90.0, 0.0],
            [90.0, 90.0, 0.0],
            [45.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    maps.expand_orientation_against_grid(euler)
    assert maps.orientation_eligible is not None
    assert maps.orientation_eligible.shape == (3,)
    assert int(maps.orientation_eligible[0]) == 1
    assert int(maps.orientation_eligible[1]) == 0
    assert int(maps.orientation_eligible[2]) == 0
    assert int(maps.n_orientations[3, 3]) == 1
    assert int(maps.n_orientations[0, 0]) == 0
    assert maps.allowed_num_ccg(n_defocus=2) == int(maps.eligible.sum()) * 1 * 2


def test_missing_orientation_maps_keep_full_yaml_grid():
    maps = rasterize_rectangle((4, 4), box=(1.0, 1.0, 2.0, 2.0), n_orientations=3)
    euler = np.array(
        [[0.0, 90.0, 0.0], [10.0, 90.0, 0.0], [20.0, 90.0, 0.0]],
        dtype=np.float64,
    )
    maps.expand_orientation_against_grid(euler)
    assert maps.orientation_eligible is None
    assert int(maps.n_orientations[1, 1]) == 3
    assert int(maps.n_orientations[0, 0]) == 0


def test_expand_orientation_dense_mask_and_shape_mismatch():
    maps = rasterize_rectangle((4, 4), box=(1.0, 1.0, 2.0, 2.0), n_orientations=2)
    euler = np.array(
        [[0.0, 90.0, 0.0], [90.0, 90.0, 0.0]],
        dtype=np.float64,
    )
    dense = np.zeros((2, 4, 4), dtype=np.uint8)
    dense[0, 1, 1] = 1
    maps.orientation_eligible = dense
    maps.expand_orientation_against_grid(euler)
    assert maps.orientation_eligible.shape == (2, 4, 4)
    assert int(maps.orientation_eligible[0, 1, 1]) == 1
    assert int(maps.orientation_eligible[1, 1, 1]) == 0
    assert int(maps.n_orientations[1, 1]) == 1
    assert int(maps.n_orientations[0, 0]) == 0

    maps.orientation_eligible = np.zeros((3, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="orientation_eligible shape"):
        maps.expand_orientation_against_grid(euler)


def test_hdf5_roundtrip_orientation_eligible(tmp_path: Path):
    maps = rasterize_rectangle((5, 5), box=(1.0, 1.0, 3.0, 3.0), n_orientations=2)
    dense = np.zeros((2, 5, 5), dtype=np.uint8)
    dense[0] = maps.eligible
    maps.orientation_eligible = dense
    path = tmp_path / "orient_constraint.h5"
    write_spatial_constraint_hdf5(str(path), maps)
    loaded = read_spatial_constraint_hdf5(str(path))
    np.testing.assert_array_equal(loaded.orientation_eligible, dense)
    euler = np.array([[0.0, 90.0, 0.0], [10.0, 90.0, 0.0]], dtype=np.float64)
    loaded.expand_orientation_against_grid(euler)
    assert int(loaded.n_orientations[2, 2]) == 1
    assert int(loaded.n_orientations[0, 0]) == 0


def test_orientation_eligible_to_stats_map_coords():
    maps = rasterize_rectangle((10, 10), box=(4.0, 4.0, 6.0, 6.0), n_orientations=3)
    dense = np.zeros((2, 10, 10), dtype=np.uint8)
    dense[1] = maps.eligible
    maps.orientation_eligible = dense
    stats = maps.to_stats_map_coords(half_template_width=2, stats_shape=(7, 7))
    assert stats.orientation_eligible is not None
    assert stats.orientation_eligible.shape == (2, 7, 7)
    assert int(stats.orientation_eligible[1, 2, 2]) == 1
    assert int(stats.orientation_eligible[1, 0, 0]) == 0


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
    assert loaded.eligible[3, 4] == 1
    assert loaded.eligible[0, 0] == 0

    euler = np.array(
        [
            [0.0, 90.0, 0.0],
            [90.0, 90.0, 90.0],
        ],
        dtype=np.float64,
    )
    stats = constraint.stats_maps_for_template(
        image_shape=(12, 12),
        template_width=4,
        euler_angles=euler,
    )
    assert stats is not None
    assert stats.coordinate_frame == "pos_xy"
    assert stats.eligible.shape == (9, 9)
    assert stats.orientation_eligible is not None
    assert int(stats.n_orientations.max()) == 1
    assert int(stats.orientation_eligible[0]) == 1
    assert int(stats.orientation_eligible[1]) == 0


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


def test_expand_defocus_min_max_against_grid():
    maps = rasterize_rectangle((6, 6), box=(2.0, 2.0, 4.0, 4.0), n_orientations=4)
    maps.defocus_min = np.full((6, 6), -50.0, dtype=np.float32)
    maps.defocus_max = np.full((6, 6), 50.0, dtype=np.float32)
    maps.expand_defocus_against_grid(np.array([-200.0, 0.0, 200.0]))

    assert maps.defocus_eligible is not None
    assert maps.defocus_eligible.shape == (3, 6, 6)
    assert int(maps.n_defocus[3, 3]) == 1
    assert int(maps.n_defocus[0, 0]) == 0
    assert maps.defocus_eligible[1, 3, 3] == 1
    assert maps.defocus_eligible[0, 3, 3] == 0
    assert maps.allowed_num_ccg() == int(maps.eligible.sum()) * 4 * 1


def test_missing_defocus_maps_keep_full_grid():
    maps = rasterize_rectangle((4, 4), box=(1.0, 1.0, 2.0, 2.0), n_orientations=3)
    maps.expand_defocus_against_grid(np.array([-100.0, 0.0, 100.0]))
    assert maps.defocus_eligible is None
    assert int(maps.n_defocus[1, 1]) == 3
    assert int(maps.n_defocus[0, 0]) == 0
    assert maps.allowed_num_ccg(n_defocus=3) == int(maps.eligible.sum()) * 3 * 3


def test_region_table_defocus_bounds():
    maps = rasterize_rectangle((5, 5), box=(1.0, 1.0, 3.0, 3.0), n_orientations=2)
    maps.regions[0]["defocus_min"] = -10.0
    maps.regions[0]["defocus_max"] = 10.0
    maps.expand_defocus_against_grid(np.array([-100.0, 0.0, 100.0]))
    assert maps.defocus_eligible[1, 2, 2] == 1
    assert maps.defocus_eligible[0, 2, 2] == 0
    assert int(maps.n_defocus[2, 2]) == 1


def test_hdf5_roundtrip_defocus_maps(tmp_path: Path):
    maps = rasterize_rectangle((5, 5), box=(1.0, 1.0, 3.0, 3.0), n_orientations=2)
    maps.defocus_min = np.full((5, 5), -80.0, dtype=np.float32)
    maps.defocus_max = np.full((5, 5), 80.0, dtype=np.float32)
    maps.regions[0]["defocus_min"] = -80.0
    maps.regions[0]["defocus_max"] = 80.0
    path = tmp_path / "defocus_constraint.h5"
    write_spatial_constraint_hdf5(str(path), maps)
    loaded = read_spatial_constraint_hdf5(str(path))
    np.testing.assert_allclose(loaded.defocus_min, maps.defocus_min)
    np.testing.assert_allclose(loaded.defocus_max, maps.defocus_max)
    assert loaded.regions[0]["defocus_min"] == -80.0
    loaded.expand_defocus_against_grid(np.array([-100.0, 0.0, 100.0]))
    assert int(loaded.n_defocus[2, 2]) == 1


def test_defocus_maps_to_stats_map_coords():
    maps = rasterize_rectangle((10, 10), box=(4.0, 4.0, 6.0, 6.0), n_orientations=3)
    maps.defocus_min = np.full((10, 10), -20.0, dtype=np.float32)
    maps.defocus_max = np.full((10, 10), 20.0, dtype=np.float32)
    stats = maps.to_stats_map_coords(half_template_width=2, stats_shape=(7, 7))
    assert stats.defocus_min.shape == (7, 7)
    stats.expand_defocus_against_grid(np.array([-100.0, 0.0, 100.0]))
    assert stats.eligible[2, 2] == 1
    assert int(stats.n_defocus[2, 2]) == 1
    assert int(stats.n_defocus[0, 0]) == 0


def test_num_ccg_uses_per_pixel_defocus():
    height, width = 8, 8
    scaled = torch.zeros((height, width))
    scaled[1, 1] = 20.0
    scaled[4, 4] = 19.0
    mip = scaled.clone()
    dummy = torch.zeros((height, width))
    n_orients = torch.zeros((height, width), dtype=torch.int32)
    n_orients[3:6, 3:6] = 10
    n_def = torch.zeros((height, width), dtype=torch.int32)
    n_def[3:6, 3:6] = 2

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
        n_defocus_map=n_def,
        n_cs=1,
        z_score_cutoff=1.0,
        mask_radius=1.0,
    )
    assert len(peaks.pos_y) == 1
    assert int(peaks.pos_y[0]) == 4
    assert int(peaks.pos_x[0]) == 4
