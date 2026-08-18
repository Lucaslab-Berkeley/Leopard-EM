"""Tests for filament Euler-box orientation constraints."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from leopard_em.pydantic_models.config import (
    FilamentConstraint,
    MultipleOrientationConfig,
    OrientationSearchConfig,
)
from leopard_em.pydantic_models.config.filament_constraint import (
    filament_psi_from_image_line,
    periodic_euler_box_intervals,
)
from leopard_em.pydantic_models.managers.match_template_manager import (
    MatchTemplateManager,
)


def test_horizontal_line_psi_is_zero():
    psi = filament_psi_from_image_line(y0=10.0, x0=0.0, y1=10.0, x1=50.0)
    assert psi == pytest.approx(0.0)


def test_vertical_line_toward_image_top_psi_is_90():
    # Smaller row index is toward the top of the image.
    psi = filament_psi_from_image_line(y0=50.0, x0=10.0, y1=0.0, x1=10.0)
    assert psi == pytest.approx(90.0)


def test_vertical_line_toward_image_bottom_psi_is_270():
    psi = filament_psi_from_image_line(y0=0.0, x0=10.0, y1=50.0, x1=10.0)
    assert psi == pytest.approx(270.0)


def test_zero_length_line_raises():
    with pytest.raises(ValueError, match="zero length"):
        filament_psi_from_image_line(1.0, 1.0, 1.0, 1.0)


def test_periodic_interval_no_wrap():
    assert periodic_euler_box_intervals(180.0, 10.0) == [(170.0, 190.0)]


def test_periodic_interval_wraps_across_zero():
    intervals = periodic_euler_box_intervals(0.0, 10.0)
    assert intervals == [(350.0, 360.0), (0.0, 10.0)]


def test_periodic_interval_hi_exactly_zero():
    # center=350, ±10 → [340, 360]
    intervals = periodic_euler_box_intervals(350.0, 10.0)
    assert intervals[0][0] == pytest.approx(340.0)
    assert intervals[0][1] == pytest.approx(360.0)
    assert len(intervals) == 1


def test_two_poles_are_180_apart():
    constraint = FilamentConstraint.from_line(
        y0=0.0, x0=0.0, y1=0.0, x1=10.0, cone_half_angle_deg=10.0
    )
    pole_1, pole_2 = constraint.pole_psi_angles_deg()
    assert pole_1 == pytest.approx(0.0)
    assert pole_2 == pytest.approx(180.0)


def test_euler_box_theta_and_phi():
    constraint = FilamentConstraint(filament_angle_deg=40.0, cone_half_angle_deg=10.0)
    theta_min, theta_max = constraint.theta_range_deg()
    assert (theta_min, theta_max) == (80.0, 100.0)

    config = constraint.to_orientation_config()
    assert isinstance(config, MultipleOrientationConfig)
    assert len(config.orientation_configs) == 2

    psi_ranges = {(b.psi_min, b.psi_max) for b in config.orientation_configs}
    assert psi_ranges == {(30.0, 50.0), (210.0, 230.0)}

    for block in config.orientation_configs:
        assert isinstance(block, OrientationSearchConfig)
        assert block.symmetry is None
        assert block.phi_min == 0.0
        assert block.phi_max == 360.0
        assert block.theta_min == 80.0
        assert block.theta_max == 100.0
        assert block.psi_step == 1.5
        assert block.theta_step == 2.5


def test_wrap_splits_only_the_wrapped_pole():
    constraint = FilamentConstraint(filament_angle_deg=0.0, cone_half_angle_deg=10.0)
    blocks = constraint.to_orientation_config().orientation_configs
    # Pole 1 wraps into two psi intervals; pole 2 (180°) does not.
    assert len(blocks) == 3
    psi_ranges = {(b.psi_min, b.psi_max) for b in blocks}
    assert (350.0, 360.0) in psi_ranges
    assert (0.0, 10.0) in psi_ranges
    assert (170.0, 190.0) in psi_ranges
    for block in blocks:
        assert block.phi_min == 0.0
        assert block.phi_max == 360.0


def test_from_line_stores_endpoints():
    constraint = FilamentConstraint.from_line(
        y0=1.0, x0=2.0, y1=11.0, x1=22.0, cone_half_angle_deg=5.0
    )
    assert constraint.line is not None
    assert constraint.line.y0 == 1.0
    assert constraint.line.x1 == 22.0
    expected = filament_psi_from_image_line(1.0, 2.0, 11.0, 22.0)
    assert constraint.filament_angle_deg == pytest.approx(expected)


def test_sidecar_roundtrip(tmp_path: Path):
    constraint = FilamentConstraint.from_line(
        y0=10.0,
        x0=0.0,
        y1=10.0,
        x1=40.0,
        cone_half_angle_deg=8.0,
        micrograph_path="/tmp/micrograph.mrc",
    )
    path = tmp_path / "filament_constraint.yaml"
    constraint.save_sidecar(str(path))

    loaded = FilamentConstraint.from_yaml(path)
    assert loaded.filament_angle_deg == pytest.approx(0.0)
    assert loaded.cone_half_angle_deg == 8.0
    assert loaded.micrograph_path == "/tmp/micrograph.mrc"
    assert loaded.line is not None
    assert loaded.to_orientation_config().orientation_configs[
        0
    ].psi_min == pytest.approx(352.0)

    sidecar = yaml.safe_load(path.read_text())
    assert "orientation_search_config" in sidecar
    first = sidecar["orientation_search_config"]["orientation_configs"][0]
    assert first["symmetry"] is None
    assert first["phi_min"] == 0.0
    assert first["phi_max"] == 360.0


def test_loads_standalone_gui_sidecar(tmp_path: Path):
    """Sidecar written without leopard_em must still load in FilamentConstraint."""
    text = """
filament_angle_deg: 0.0
cone_half_angle_deg: 10.0
theta_center_deg: 90.0
phi_min: 0.0
phi_max: 360.0
psi_step: 1.5
theta_step: 2.5
base_grid_method: uniform
micrograph_path: /tmp/micrograph.mrc
line:
  y0: 10.0
  x0: 0.0
  y1: 10.0
  x1: 40.0
orientation_search_config:
  orientation_configs:
    - psi_step: 1.5
      theta_step: 2.5
      symmetry: null
      phi_min: 0.0
      phi_max: 360.0
      theta_min: 80.0
      theta_max: 100.0
      psi_min: 350.0
      psi_max: 360.0
      base_grid_method: uniform
"""
    path = tmp_path / "filament_constraint.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = FilamentConstraint.from_yaml(path)
    assert loaded.filament_angle_deg == pytest.approx(0.0)
    assert loaded.line is not None
    psi_ranges = {
        (b.psi_min, b.psi_max)
        for b in loaded.to_orientation_config().orientation_configs
    }
    assert (350.0, 360.0) in psi_ranges
    assert (0.0, 10.0) in psi_ranges
    assert (170.0, 190.0) in psi_ranges


def test_generated_config_is_valid_for_euler_angles():
    constraint = FilamentConstraint(
        filament_angle_deg=37.0,
        cone_half_angle_deg=10.0,
        phi_min=0.0,
        phi_max=10.0,
    )
    angles = constraint.to_orientation_config().euler_angles
    assert angles.ndim == 2
    assert angles.shape[1] == 3
    assert angles.shape[0] > 1

    psi = angles[:, 2]
    theta = angles[:, 1]
    pole_1, pole_2 = constraint.pole_psi_angles_deg()
    half = constraint.cone_half_angle_deg

    in_either = ((((psi - pole_1 + 180.0) % 360.0) - 180.0).abs() <= half + 1e-3) | (
        (((psi - pole_2 + 180.0) % 360.0) - 180.0).abs() <= half + 1e-3
    )
    assert bool(in_either.all())

    assert float(theta.min()) >= 80.0 - 1e-3
    assert float(theta.max()) <= 100.0 + 1e-3


def test_orientation_allowed_mask_subsets_yaml_angles():
    constraint = FilamentConstraint(filament_angle_deg=0.0, cone_half_angle_deg=10.0)
    euler = np.array(
        [
            [0.0, 90.0, 0.0],
            [90.0, 90.0, 180.0],
            [0.0, 90.0, 90.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    mask = constraint.orientation_allowed_mask(euler)
    assert mask.tolist() == [True, True, False, False]


def test_apply_filament_constraint_keeps_yaml_orientation_search():
    source = inspect.getsource(MatchTemplateManager.apply_filament_constraint)
    assert "constraint.to_orientation_config()" not in source
    assert "euler_angles=euler_angles" in source


def test_stats_flag_accepts_legacy_yaml_key():
    legacy = FilamentConstraint(
        filament_angle_deg=0.0,
        stats_from_valid_orientations=True,
    )
    assert legacy.stats_from_valid_orientations_defocus is True
    current = FilamentConstraint(
        filament_angle_deg=0.0,
        stats_from_valid_orientations_defocus=True,
    )
    assert current.stats_from_valid_orientations_defocus is True
