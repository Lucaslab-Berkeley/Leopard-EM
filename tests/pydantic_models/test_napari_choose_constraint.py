"""Tests for standalone napari constraint-helper dump and rasterize logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from leopard_em.pydantic_models.config import FilamentConstraint


def _load_napari_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "programs"
        / "constrained_search"
        / "napari_choose_constraint.py"
    )
    spec = importlib.util.spec_from_file_location("napari_choose_constraint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_combine_search_boxes_later_region_wins():
    napari_mod = _load_napari_module()
    left = napari_mod.rasterize_search_box(
        (8, 10),
        np.array([[2.0, 2.0], [2.0, 6.0], [5.0, 6.0], [5.0, 2.0]]),
        n_orientations=1,
        region_id=1,
    )
    right = napari_mod.rasterize_search_box(
        (8, 10),
        np.array([[2.0, 5.0], [2.0, 8.0], [5.0, 8.0], [5.0, 5.0]]),
        n_orientations=2,
        region_id=2,
    )
    eligible, region_id, n_orients = napari_mod.combine_search_boxes([left, right])
    assert int(region_id[3, 3]) == 1
    assert int(region_id[3, 6]) == 2
    assert int(n_orients[3, 6]) == 2
    assert int(eligible[0, 0]) == 0


def test_single_region_yaml_dump_still_loads():
    napari_mod = _load_napari_module()
    payload = napari_mod.build_constraint_payload(
        y0=10.0,
        x0=0.0,
        y1=10.0,
        x1=40.0,
        cone_half_angle_deg=10.0,
        psi_step=1.5,
        theta_step=2.5,
        theta_center_deg=90.0,
        micrograph_path="/tmp/micrograph.mrc",
        spatial_box=np.array([[1.0, 1.0], [1.0, 4.0], [5.0, 4.0], [5.0, 1.0]]),
    )
    text = napari_mod.dump_constraint_yaml(payload)
    parsed = yaml.safe_load(text)
    assert "filament_angle_deg" in parsed
    assert "regions" not in parsed
    loaded = FilamentConstraint.model_validate(parsed)
    assert loaded.filament_angle_deg == pytest.approx(0.0)
    assert loaded.spatial_box is not None


def test_multi_region_yaml_dump_loads():
    napari_mod = _load_napari_module()
    first = napari_mod.build_constraint_payload(
        y0=2.0,
        x0=0.0,
        y1=2.0,
        x1=10.0,
        cone_half_angle_deg=10.0,
        psi_step=1.5,
        theta_step=2.5,
        theta_center_deg=90.0,
        micrograph_path="/tmp/micrograph.mrc",
        spatial_box=np.array([[1.0, 1.0], [1.0, 3.0], [4.0, 3.0], [4.0, 1.0]]),
    )
    second = napari_mod.build_constraint_payload(
        y0=8.0,
        x0=5.0,
        y1=0.0,
        x1=5.0,
        cone_half_angle_deg=10.0,
        psi_step=1.5,
        theta_step=2.5,
        theta_center_deg=90.0,
        micrograph_path="/tmp/micrograph.mrc",
        spatial_box=np.array([[1.0, 5.0], [1.0, 7.0], [4.0, 7.0], [4.0, 5.0]]),
    )
    sidecar = napari_mod.sidecar_payload_from_regions(
        [first, second], spatial_constraint_path="/tmp/constraint.h5"
    )
    text = napari_mod.dump_constraint_yaml(sidecar)
    parsed = yaml.safe_load(text)
    assert "regions" in parsed
    assert "filament_angle_deg" not in parsed
    loaded = FilamentConstraint.model_validate(parsed)
    assert len(loaded.regions) == 2
    assert loaded.regions[0].filament_angle_deg == pytest.approx(0.0)
    assert loaded.regions[1].filament_angle_deg == pytest.approx(90.0)
    assert loaded.spatial_constraint_path == "/tmp/constraint.h5"
