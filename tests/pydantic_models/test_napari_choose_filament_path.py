"""Tests for the curved-filament exporter: every covering path is kept, nearest first."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from leopard_em.pydantic_models.config.spatial_constraint import (
    read_spatial_constraint_hdf5,
)


def _load():
    path = (
        Path(__file__).resolve().parents[2]
        / "programs"
        / "constrained_search"
        / "napari_choose_filament_path.py"
    )
    spec = importlib.util.spec_from_file_location("napari_choose_filament_path", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line(y0, x0, y1, x1, width, n=6):
    return {
        "points": np.column_stack([np.linspace(y0, y1, n), np.linspace(x0, x1, n)]).tolist(),
        "width_px": float(width),
        "polarity": "both",
    }


SHAPE = (120, 160)


def test_separate_paths_have_no_extra_layers():
    mod = _load()
    layers = mod.paint_path_layers(
        SHAPE, [_line(20, 10, 20, 150, 16), _line(90, 10, 90, 150, 16)], smooth_px=0
    )
    assert layers["psi_center_extra"] is None
    assert layers["region_id_extra"] is None
    assert set(np.unique(layers["region_id"])) == {0, 1, 2}


def test_parallel_overlap_keeps_both_paths_nearest_first():
    mod = _load()
    # Centrelines 20 px apart, bands 40 px wide: a 20 px strip is covered twice.
    layers = mod.paint_path_layers(
        SHAPE, [_line(50, 10, 50, 150, 40), _line(70, 10, 70, 150, 40)], smooth_px=0
    )
    extra = layers["region_id_extra"]
    assert extra is not None and extra.shape == (1, *SHAPE)
    # Row 55 is 5 px from path 1 and 15 px from path 2: both kept, path 1 first.
    assert int(layers["region_id"][55, 80]) == 1
    assert int(extra[0, 55, 80]) == 2
    # Row 65 is the mirror image.
    assert int(layers["region_id"][65, 80]) == 2
    assert int(extra[0, 65, 80]) == 1
    # Row 35 is inside path 1's band only.
    assert int(layers["region_id"][35, 80]) == 1
    assert int(extra[0, 35, 80]) == 0


def test_crossing_keeps_each_paths_own_tangent():
    mod = _load()
    layers = mod.paint_path_layers(
        SHAPE, [_line(60, 10, 60, 150, 30), _line(10, 40, 110, 120, 30)], smooth_px=0
    )
    y, x = 60, 80  # on both centrelines
    psis = {int(layers["region_id"][y, x]): float(layers["psi_center"][y, x]),
            int(layers["region_id_extra"][0, y, x]):
                float(layers["psi_center_extra"][0, y, x])}
    assert set(psis) == {1, 2}
    assert psis[1] == pytest.approx(0.0, abs=1e-3) or psis[1] == pytest.approx(360.0)
    expected = np.degrees(np.arctan2(-100.0, 80.0)) % 360.0
    assert psis[2] == pytest.approx(expected, abs=0.5)


def test_eligible_is_any_coverage_even_with_unequal_widths():
    """A pixel in a WIDE band but nearer a NARROW path is still in play."""
    mod = _load()
    layers = mod.paint_path_layers(
        SHAPE, [_line(40, 10, 40, 150, 80), _line(56, 10, 56, 150, 6)], smooth_px=0
    )
    # Row 62: 22 px from path 1 (half-width 40, covered) and 6 px from path 2
    # (half-width 3, not covered). The old nearest-wins rule dropped this pixel.
    assert int(layers["eligible"][62, 80]) == 1
    assert int(layers["region_id"][62, 80]) == 1


def test_layer_zero_matches_the_old_nearest_wins_rule_with_equal_widths():
    from scipy.ndimage import distance_transform_edt

    mod = _load()
    paths = [_line(50, 10, 50, 150, 40), _line(70, 10, 70, 150, 40),
             _line(10, 40, 110, 120, 40)]
    layers = mod.paint_path_layers(SHAPE, paths, smooth_px=0)

    # The old exporter: one distance transform, nearest centreline pixel wins.
    centre = np.zeros(SHAPE, dtype=np.int16)
    distances = []
    for index, path in enumerate(paths, start=1):
        curve, _ = mod.spline_through_points(path["points"], smooth_px=0)
        rows = np.clip(np.rint(curve[:, 0]).astype(int), 0, SHAPE[0] - 1)
        cols = np.clip(np.rint(curve[:, 1]).astype(int), 0, SHAPE[1] - 1)
        centre[rows, cols] = index
        mask = np.zeros(SHAPE, dtype=bool)
        mask[rows, cols] = True
        distances.append(distance_transform_edt(~mask))
    distance, (ny, nx) = distance_transform_edt(centre == 0, return_indices=True)
    old_region = np.where(distance <= 20.0, centre[ny, nx], 0)

    ranked = np.sort(np.stack(distances), axis=0)
    unambiguous = (ranked[1] - ranked[0]) > 1e-6  # exact ties may pick either
    assert np.array_equal(layers["eligible"] > 0, old_region > 0)
    assert np.array_equal(layers["region_id"][unambiguous], old_region[unambiguous])


def test_standalone_writer_is_read_back_by_leopard_em(tmp_path: Path):
    pytest.importorskip("h5py")
    mod = _load()
    paths = [_line(50, 10, 50, 150, 40), _line(70, 10, 70, 150, 40)]
    layers = mod.paint_path_layers(SHAPE, paths, smooth_px=0)
    out = tmp_path / "constraint.h5"
    mod.write_constraint_hdf5(
        str(out), layers["eligible"], layers["region_id"], layers["psi_center"],
        mod.estimate_n_orientations(layers["eligible"], 20.0, 1.5, 2.5),
        signed_distance=layers["distance"], paths=paths,
        cone_half_angle_deg=20.0, theta_center_deg=90.0, psi_step=1.5, theta_step=2.5,
        psi_center_extra=layers["psi_center_extra"],
        region_id_extra=layers["region_id_extra"],
    )
    maps = read_spatial_constraint_hdf5(str(out))
    assert maps.n_psi_layers == 2
    np.testing.assert_array_equal(maps.region_id_extra, layers["region_id_extra"])
    np.testing.assert_allclose(maps.psi_center_extra, layers["psi_center_extra"])


def test_refitted_curve_and_psi_are_painted_with_layers():
    """A caller's own curve and psi (a refit from data) get the same layered output."""
    mod = _load()
    drawn = [_line(50, 10, 50, 150, 40), _line(70, 10, 70, 150, 40)]
    refit = []
    for path in drawn:
        curve, _ = mod.spline_through_points(path["points"], smooth_px=0)
        refit.append({**path, "curve_yx": curve, "psi_deg": np.full(len(curve), 7.5)})
    layers = mod.paint_path_layers(SHAPE, refit, smooth_px=0)
    covered = layers["eligible"] > 0
    assert np.allclose(layers["psi_center"][covered], 7.5)
    assert layers["region_id_extra"] is not None
    assert int(layers["region_id_extra"][0, 60, 80]) > 0


def test_refitted_psi_must_match_its_curve():
    mod = _load()
    path = _line(50, 10, 50, 150, 40)
    curve, _ = mod.spline_through_points(path["points"], smooth_px=0)
    with pytest.raises(ValueError, match="one value per"):
        mod.paint_path_layers(
            SHAPE, [{**path, "curve_yx": curve, "psi_deg": np.zeros(3)}], smooth_px=0)
