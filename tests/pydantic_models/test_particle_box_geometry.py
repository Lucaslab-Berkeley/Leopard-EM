"""Tests for ParticleBoxGeometry: crop origins, valid sizes and template centers."""

import numpy as np
import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.data_structures.particle_box_geometry import (
    ParticleBoxGeometry,
)
from leopard_em.pydantic_models.data_structures.particle_stack import ParticleStack

# Non-square sizes with odd differences so swapped axes or rounding errors show up
BOX = (24, 22)
TEMPLATE = (16, 15)
GEOMETRY = ParticleBoxGeometry(extracted_box_size=BOX, original_template_size=TEMPLATE)


def test_valid_size_and_margin():
    assert GEOMETRY.valid_size == (9, 8)
    assert GEOMETRY.margin == (4, 3)


@pytest.mark.parametrize(
    ("pos_reference", "expected_top_left", "expected_origin", "expected_center"),
    [
        ("top-left", (30, 40), (26, 37), (38, 47)),
        ("center", (22, 33), (18, 30), (30, 40)),
    ],
)
def test_positions_numpy(
    pos_reference, expected_top_left, expected_origin, expected_center
):
    pos_y, pos_x = np.array([30]), np.array([40])

    top = GEOMETRY.template_top_left(pos_y, pos_x, pos_reference)
    origin = GEOMETRY.box_origins(pos_y, pos_x, pos_reference)
    center = GEOMETRY.template_centers(pos_y, pos_x, pos_reference)

    assert (int(top[0][0]), int(top[1][0])) == expected_top_left
    assert (int(origin[0][0]), int(origin[1][0])) == expected_origin
    assert (int(center[0][0]), int(center[1][0])) == expected_center


def test_positions_keep_torch_type():
    pos_y, pos_x = torch.tensor([30, 5]), torch.tensor([40, 0])
    origin_y, origin_x = GEOMETRY.box_origins(pos_y, pos_x)
    assert isinstance(origin_y, torch.Tensor)
    assert origin_y.tolist() == [26, 1]
    assert origin_x.tolist() == [37, -3]


def test_stat_map_crop_matches_particle_box_correlation():
    """Image box and valid stat region share an origin; peak sits at the margin."""
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.normal(size=(80, 70)).astype(np.float32))
    template = torch.from_numpy(rng.normal(size=TEMPLATE).astype(np.float32))

    def valid_xcorr(img):
        return torch.nn.functional.conv2d(img[None, None], template[None, None])[0, 0]

    full_map = valid_xcorr(image)
    valid_h, valid_w = GEOMETRY.valid_size
    margin_y, margin_x = GEOMETRY.margin

    for pos_y, pos_x in [(20, 25), (33, 41), (50, 12)]:
        image[pos_y : pos_y + TEMPLATE[0], pos_x : pos_x + TEMPLATE[1]] += 5 * template
        full_map = valid_xcorr(image)

        (oy,), (ox,) = GEOMETRY.box_origins(np.array([pos_y]), np.array([pos_x]))
        box = image[oy : oy + BOX[0], ox : ox + BOX[1]]
        stat_crop = full_map[oy : oy + valid_h, ox : ox + valid_w]

        torch.testing.assert_close(valid_xcorr(box), stat_crop)
        assert stat_crop[margin_y, margin_x] == full_map[pos_y, pos_x]


def test_particle_stack_geometry_property():
    stack = ParticleStack(
        df_path="",
        extracted_box_size=BOX,
        original_template_size=TEMPLATE,
        skip_df_load=True,
    )
    assert stack.geometry == GEOMETRY

    stack.extracted_box_size = (30, 30)
    assert stack.geometry.valid_size == (15, 16)


def test_particle_stack_crops_at_geometry_origins():
    """Particle boxes and valid-size regions are cropped from the same origin."""
    image = torch.arange(80 * 70, dtype=torch.float32).reshape(80, 70)
    positions = [(20, 25), (50, 12)]
    df = pd.DataFrame(
        {"pos_y": [p[0] for p in positions], "pos_x": [p[1] for p in positions]}
    )
    stack = ParticleStack(
        df_path="",
        extracted_box_size=BOX,
        original_template_size=TEMPLATE,
        skip_df_load=True,
    )
    stack._df = df

    groups = [pd.Index([0, 1])]
    particles = stack.construct_image_stack(image[None], groups, BOX)
    stat_regions = stack.construct_image_stack(image[None], groups, GEOMETRY.valid_size)

    for i, (pos_y, pos_x) in enumerate(positions):
        oy, ox = pos_y - GEOMETRY.margin[0], pos_x - GEOMETRY.margin[1]
        torch.testing.assert_close(
            particles[i], image[oy : oy + BOX[0], ox : ox + BOX[1]]
        )
        torch.testing.assert_close(
            stat_regions[i],
            image[oy : oy + GEOMETRY.valid_size[0], ox : ox + GEOMETRY.valid_size[1]],
        )
