"""Tests for filament Euler-box orientation constraints."""

from pathlib import Path

import pytest
import yaml

from leopard_em.pydantic_models.config import (
    FilamentConstraint,
    MultipleOrientationConfig,
    OrientationSearchConfig,
)
from leopard_em.pydantic_models.config.filament_constraint import (
    filament_phi_from_image_line,
    phi_euler_box_intervals,
)


def test_horizontal_line_phi_is_zero():
    phi = filament_phi_from_image_line(y0=10.0, x0=0.0, y1=10.0, x1=50.0)
    assert phi == pytest.approx(0.0)


def test_vertical_line_toward_image_top_phi_is_90():
    # Smaller row index is toward the top of the image.
    phi = filament_phi_from_image_line(y0=50.0, x0=10.0, y1=0.0, x1=10.0)
    assert phi == pytest.approx(90.0)


def test_vertical_line_toward_image_bottom_phi_is_270():
    phi = filament_phi_from_image_line(y0=0.0, x0=10.0, y1=50.0, x1=10.0)
    assert phi == pytest.approx(270.0)


def test_zero_length_line_raises():
    with pytest.raises(ValueError, match="zero length"):
        filament_phi_from_image_line(1.0, 1.0, 1.0, 1.0)


def test_phi_interval_no_wrap():
    assert phi_euler_box_intervals(180.0, 10.0) == [(170.0, 190.0)]


def test_phi_interval_wraps_across_zero():
    intervals = phi_euler_box_intervals(0.0, 10.0)
    assert intervals == [(350.0, 360.0), (0.0, 10.0)]


def test_phi_interval_hi_exactly_zero():
    # center=350, ±10 → [340, 360]
    intervals = phi_euler_box_intervals(350.0, 10.0)
    assert intervals[0][0] == pytest.approx(340.0)
    assert intervals[0][1] == pytest.approx(360.0)
    assert len(intervals) == 1


def test_two_poles_are_180_apart():
    constraint = FilamentConstraint.from_line(
        y0=0.0, x0=0.0, y1=0.0, x1=10.0, cone_half_angle_deg=10.0
    )
    pole_1, pole_2 = constraint.pole_phi_angles_deg()
    assert pole_1 == pytest.approx(0.0)
    assert pole_2 == pytest.approx(180.0)


def test_euler_box_theta_and_psi():
    constraint = FilamentConstraint(filament_angle_deg=40.0, cone_half_angle_deg=10.0)
    theta_min, theta_max = constraint.theta_range_deg()
    assert (theta_min, theta_max) == (80.0, 100.0)

    config = constraint.to_orientation_config()
    assert isinstance(config, MultipleOrientationConfig)
    assert len(config.orientation_configs) == 2

    phi_ranges = {(b.phi_min, b.phi_max) for b in config.orientation_configs}
    assert phi_ranges == {(30.0, 50.0), (210.0, 230.0)}

    for block in config.orientation_configs:
        assert isinstance(block, OrientationSearchConfig)
        assert block.symmetry is None
        assert block.psi_min == 0.0
        assert block.psi_max == 360.0
        assert block.theta_min == 80.0
        assert block.theta_max == 100.0
        assert block.psi_step == 1.5
        assert block.theta_step == 2.5


def test_wrap_splits_only_the_wrapped_pole():
    constraint = FilamentConstraint(filament_angle_deg=0.0, cone_half_angle_deg=10.0)
    blocks = constraint.to_orientation_config().orientation_configs
    # Pole 1 wraps into two phi intervals; pole 2 (180°) does not.
    assert len(blocks) == 3
    phi_ranges = {(b.phi_min, b.phi_max) for b in blocks}
    assert (350.0, 360.0) in phi_ranges
    assert (0.0, 10.0) in phi_ranges
    assert (170.0, 190.0) in phi_ranges
    for block in blocks:
        assert block.psi_min == 0.0
        assert block.psi_max == 360.0


def test_from_line_stores_endpoints():
    constraint = FilamentConstraint.from_line(
        y0=1.0, x0=2.0, y1=11.0, x1=22.0, cone_half_angle_deg=5.0
    )
    assert constraint.line is not None
    assert constraint.line.y0 == 1.0
    assert constraint.line.x1 == 22.0
    expected = filament_phi_from_image_line(1.0, 2.0, 11.0, 22.0)
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
    ].phi_min == pytest.approx(352.0)

    sidecar = yaml.safe_load(path.read_text())
    assert "orientation_search_config" in sidecar
    first = sidecar["orientation_search_config"]["orientation_configs"][0]
    assert first["symmetry"] is None
    assert first["psi_min"] == 0.0
    assert first["psi_max"] == 360.0


def test_loads_standalone_gui_sidecar(tmp_path: Path):
    """Sidecar written without leopard_em must still load in FilamentConstraint."""
    text = """
filament_angle_deg: 0.0
cone_half_angle_deg: 10.0
theta_center_deg: 90.0
psi_min: 0.0
psi_max: 360.0
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
      phi_min: 350.0
      phi_max: 360.0
      theta_min: 80.0
      theta_max: 100.0
      psi_min: 0.0
      psi_max: 360.0
      base_grid_method: uniform
"""
    path = tmp_path / "filament_constraint.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = FilamentConstraint.from_yaml(path)
    assert loaded.filament_angle_deg == pytest.approx(0.0)
    assert loaded.line is not None
    phi_ranges = {
        (b.phi_min, b.phi_max)
        for b in loaded.to_orientation_config().orientation_configs
    }
    assert (350.0, 360.0) in phi_ranges
    assert (0.0, 10.0) in phi_ranges
    assert (170.0, 190.0) in phi_ranges


def test_generated_config_is_valid_for_euler_angles():
    constraint = FilamentConstraint(filament_angle_deg=37.0, cone_half_angle_deg=10.0)
    angles = constraint.to_orientation_config().euler_angles
    assert angles.ndim == 2
    assert angles.shape[1] == 3
    assert angles.shape[0] > 1

    phi = angles[:, 0]
    theta = angles[:, 1]
    pole_1, pole_2 = constraint.pole_phi_angles_deg()
    half = constraint.cone_half_angle_deg

    in_either = ((((phi - pole_1 + 180.0) % 360.0) - 180.0).abs() <= half + 1e-3) | (
        (((phi - pole_2 + 180.0) % 360.0) - 180.0).abs() <= half + 1e-3
    )
    assert bool(in_either.all())

    assert float(theta.min()) >= 80.0 - 1e-3
    assert float(theta.max()) <= 100.0 + 1e-3
