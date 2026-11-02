"""Tests for the stateless particle-table helpers behind the particle stacks."""

import numpy as np
import pandas as pd
import pytest
import torch

from leopard_em.pydantic_models.data_structures._particle_stack_table import (
    as_row_positions,
    choose_refined_columns,
    column_tensor,
    group_rows,
    pixel_positions,
    position_columns,
)


def _choose(df, prefer=True, **kwargs):
    return choose_refined_columns(
        df,
        ["phi", "psi"],
        prefer,
        refined_name="angles",
        original_name="angles",
        **kwargs,
    )


def test_choose_refined_columns_prefers_complete_refined_set():
    df = pd.DataFrame({"phi": [1.0], "psi": [2.0], "refined_phi": [3.0]})
    with pytest.warns(UserWarning, match="Refined angles not found"):
        assert _choose(df) == ["phi", "psi"]

    df["refined_psi"] = [4.0]
    assert _choose(df) == ["refined_phi", "refined_psi"]
    assert _choose(df, prefer=False) == ["phi", "psi"]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_choose_refined_columns_non_finite(bad_value):
    df = pd.DataFrame(
        {"phi": [1.0], "psi": [2.0], "refined_phi": [bad_value], "refined_psi": [4.0]}
    )
    with pytest.warns(UserWarning, match="contain NaN or inf values"):
        assert _choose(df) == ["phi", "psi"]
    # Without the finiteness check the refined columns are used as-is
    assert _choose(df, check_finite=False) == ["refined_phi", "refined_psi"]


def test_position_columns_fall_back_per_axis():
    df = pd.DataFrame({"pos_y": [1], "pos_x": [2], "refined_pos_x": [3.0]})
    assert position_columns(df) == ("pos_y", "refined_pos_x")


def test_pixel_positions_round_float_coordinates():
    df = pd.DataFrame(
        {"pos_y": [1.4, 2.6], "pos_x": [0.0, 7.5], "refined_pos_y": [3.5, -0.6]}
    )
    pos_y, pos_x = pixel_positions(df)
    assert pos_y.dtype == np.int64
    assert pos_y.tolist() == [4, -1]  # round half to even, refined preferred
    assert pos_x.tolist() == [0, 8]


def test_as_row_positions():
    assert as_row_positions(pd.Index([2, 0]), 3).tolist() == [2, 0]
    assert as_row_positions([], 3).dtype == np.int64
    with pytest.raises(TypeError, match="integer row positions"):
        as_row_positions(["mic_00000"], 3)
    with pytest.raises(IndexError, match="out of range"):
        as_row_positions([0, 3], 3)
    with pytest.raises(IndexError, match="out of range"):
        as_row_positions([-1], 3)


def test_group_rows_returns_positions_for_any_index():
    df = pd.DataFrame(
        {"micrograph_path": ["b.mrc", "a.mrc", "b.mrc"]},
        index=pd.Index(["x", "y", "z"], name="particle_id"),
    )
    groups = group_rows(df, "micrograph_path")
    assert list(groups) == ["a.mrc", "b.mrc"]
    assert groups["a.mrc"].tolist() == [1]
    assert groups["b.mrc"].tolist() == [0, 2]

    with pytest.raises(ValueError, match="not found"):
        group_rows(df, "mip_path")
    df.loc["y", "micrograph_path"] = None
    with pytest.raises(ValueError, match="empty for 1 particle"):
        group_rows(df, "micrograph_path")


def test_column_tensor_does_not_share_memory():
    df = pd.DataFrame({"defocus_u": [1.0, 2.0]})
    tensor = column_tensor(df, "defocus_u")
    tensor += 1
    assert df["defocus_u"].tolist() == [1.0, 2.0]
    assert torch.equal(tensor, torch.tensor([2.0, 3.0], dtype=torch.float64))
